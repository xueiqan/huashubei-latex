"""问题4：决策区精确优化与压缩前瞻的算—储—电联合滚动MILP。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


QUESTION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = QUESTION_DIR.parents[1]
SHARED_DIR = PROJECT_DIR / "question" / "question_01" / "data" / "processed" / "shared"
Q4_DIR = QUESTION_DIR / "data" / "processed" / "q4"
TABLES_DIR = QUESTION_DIR / "outputs" / "tables"

MAIN_END = 2400
OPERATION_END = 2406
METRICS = ("Cost", "Carbon", "Latency", "Delay", "RenewableUnusedRate", "Peak")
EPS = 1e-8
SOLVER_PROGRESS_INTERVAL_SECONDS = 30.0
CHECKPOINT_VERSION = 1
CHECKPOINT_PATH = TABLES_DIR / ".q4_progress_checkpoint.pkl"
PROGRESS_ASSIGNMENTS_PATH = TABLES_DIR / "q4_task_assignments_progress.csv"
PROGRESS_DISPATCH_PATH = TABLES_DIR / "q4_region_hour_dispatch_progress.csv"
PROGRESS_SOLVER_PATH = TABLES_DIR / "q4_window_solver_progress.csv"
PROGRESS_FORECAST_PATH = TABLES_DIR / "q4_forecast_profile_progress.csv"
PROGRESS_SCALING_PATH = TABLES_DIR / "q4_scaling_progress.csv"
PROGRESS_CALIBRATION_PATH = TABLES_DIR / "q4_calibration_records_progress.csv"
FAILED_WINDOW_PATH = TABLES_DIR / "q4_failed_window.json"


def _progress(message: str) -> None:
    print(f"[Q4进度] {message}", flush=True)


def _checkpoint_header(path: Path) -> dict[str, object] | None:
    try:
        with path.open("rb") as handle:
            state = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError):
        return None
    return state if isinstance(state, dict) else None


def _select_auto_resume_checkpoint(signature: str) -> tuple[Path, int] | None:
    """选择签名匹配且推进最远的正式/修复分支断点。"""

    candidates: list[tuple[int, float, Path]] = []
    for path in TABLES_DIR.glob(".q4_progress_checkpoint*.pkl"):
        state = _checkpoint_header(path)
        if (
            state is None
            or state.get("checkpoint_version") != CHECKPOINT_VERSION
            or state.get("signature") != signature
        ):
            continue
        try:
            next_tau = int(state["next_tau"])
        except (KeyError, TypeError, ValueError):
            continue
        candidates.append((next_tau, path.stat().st_mtime, path.resolve()))
    if not candidates:
        return None
    next_tau, _, path = max(candidates, key=lambda item: (item[0], item[1]))
    return path, next_tau


def _activate_progress_namespace(checkpoint: Path) -> None:
    """恢复分支继续写回自己的进度文件，避免覆盖原始断点。"""

    global CHECKPOINT_PATH
    global PROGRESS_ASSIGNMENTS_PATH, PROGRESS_DISPATCH_PATH, PROGRESS_SOLVER_PATH
    global PROGRESS_FORECAST_PATH, PROGRESS_SCALING_PATH, PROGRESS_CALIBRATION_PATH
    global FAILED_WINDOW_PATH

    checkpoint = checkpoint.resolve()
    base = ".q4_progress_checkpoint"
    if checkpoint.name == f"{base}.pkl":
        suffix = ""
    elif checkpoint.name.startswith(base) and checkpoint.name.endswith(".pkl"):
        suffix = checkpoint.name[len(base):-4]
    else:
        suffix = "_" + checkpoint.stem.removeprefix(".q4_").removesuffix("_checkpoint")
    CHECKPOINT_PATH = checkpoint
    PROGRESS_ASSIGNMENTS_PATH = TABLES_DIR / f"q4_task_assignments_progress{suffix}.csv"
    PROGRESS_DISPATCH_PATH = TABLES_DIR / f"q4_region_hour_dispatch_progress{suffix}.csv"
    PROGRESS_SOLVER_PATH = TABLES_DIR / f"q4_window_solver_progress{suffix}.csv"
    PROGRESS_FORECAST_PATH = TABLES_DIR / f"q4_forecast_profile_progress{suffix}.csv"
    PROGRESS_SCALING_PATH = TABLES_DIR / f"q4_scaling_progress{suffix}.csv"
    PROGRESS_CALIBRATION_PATH = TABLES_DIR / f"q4_calibration_records_progress{suffix}.csv"
    FAILED_WINDOW_PATH = TABLES_DIR / f"q4_failed_window{suffix}.json"


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    """先完整写入同目录临时文件，再原子替换目标文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_replace_bytes(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
    )


def _atomic_write_pickle(path: Path, payload: object) -> None:
    _atomic_replace_bytes(path, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            frame.to_csv(handle, index=False, float_format="%.15g")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _result_float(result: object, field: str) -> float:
    """将求解器可选的数值字段安全转换为float；连续松弛阶段可能返回None。"""

    value = getattr(result, field, np.nan)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


@dataclass(frozen=True)
class ModelConfig:
    decision_hours: int = 24
    lookahead_hours: int = 48
    block_hours: int = 4
    tie_break_weight: float = 1e-3
    mip_relative_gap: float = 0.01
    normal_time_limit_seconds: float = 120.0
    difficult_time_limit_seconds: float = 240.0
    difficult_due_task_threshold: int = 350
    feasibility_tolerance: float = 1e-6


@dataclass(frozen=True)
class CandidateRegion:
    region: str
    region_index: int
    latency_ms: float
    max_latency_ms: float


@dataclass(frozen=True)
class TaskOption:
    task_id: str
    task_row: int
    region: str
    region_index: int
    start_hour: float
    expected_start_hour: float
    duration_h: float
    gpu_demand: float
    power_mw: float
    latency_ratio: float
    overlaps: tuple[tuple[int, float], ...]
    is_block: bool
    block_start: int | None = None
    block_end: int | None = None


@dataclass
class InputData:
    region_hour: pd.DataFrame
    tasks: pd.DataFrame
    candidates: pd.DataFrame
    storage: pd.DataFrame
    regions: tuple[str, ...]
    region_index: dict[str, int]
    candidate_map: dict[str, tuple[CandidateRegion, ...]]
    task_lookup: pd.DataFrame


@dataclass
class WindowProblem:
    tau: int
    decision_end: int
    plan_end: int
    regions: tuple[str, ...]
    x_options: tuple[TaskOption, ...]
    u_options: tuple[TaskOption, ...]
    task_rules: dict[str, tuple[bool, bool]]
    indices: dict[str, np.ndarray]
    lower: np.ndarray
    upper: np.ndarray
    integrality: np.ndarray
    matrix: object
    constraint_lower: np.ndarray
    constraint_upper: np.ndarray
    metric_vectors: dict[str, np.ndarray]
    metric_constants: dict[str, float]
    metadata: dict[str, float | int]


@dataclass
class WindowSolution:
    vector: np.ndarray
    status: int
    message: str
    objective: float
    mip_gap: float
    mip_node_count: float
    elapsed_seconds: float


def _read_csv(path: Path, required: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"缺少模型输入：{path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}缺少字段：{missing}")
    return frame


def _numeric(frame: pd.DataFrame, columns: Iterable[str], source: str) -> pd.DataFrame:
    result = frame.copy()
    columns = tuple(columns)
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[list(columns)].isna().any().any():
        raise ValueError(f"{source}存在无法解析的数值")
    return result


def load_data() -> InputData:
    region_hour = _read_csv(
        Q4_DIR / "q4_region_hour_input.csv",
        (
            "Hour", "Region", "ElectricityPrice_CNY_per_MWh", "SellPrice_CNY_per_MWh",
            "CarbonIntensity_tCO2_per_MWh", "AvailableRenewable_MW", "NonAI_IT_Load_MW",
            "Available_GPU", "Max_IT_Power_MW", "PUE", "Max_Facility_Power_MW",
        ),
    )
    tasks = _read_csv(
        SHARED_DIR / "tasks_clean.csv",
        (
            "TaskID", "TaskType", "ArrivalHour", "SourceRegion", "GPU_Demand",
            "MaxLatency_ms", "EarliestStartHour", "LatestFinishHour", "Duration_h",
            "Task_Full_IT_Power_MW",
        ),
    )
    candidates = _read_csv(
        SHARED_DIR / "task_candidate_regions.csv",
        ("TaskID", "TargetRegion", "NetworkLatency_ms", "MaxLatency_ms"),
    )
    storage = _read_csv(
        SHARED_DIR / "storage_params.csv",
        (
            "Region", "StorageCapacity_MWh", "MinSOC_MWh", "InitialSOC_MWh",
            "MaxChargePower_MW", "MaxDischargePower_MW", "ChargeEfficiency",
            "DischargeEfficiency", "SellLimit_MW", "MaxGridImport_MW", "MaxGridExport_MW",
        ),
    )
    region_hour = _numeric(
        region_hour,
        [
            "Hour", "ElectricityPrice_CNY_per_MWh", "SellPrice_CNY_per_MWh",
            "CarbonIntensity_tCO2_per_MWh", "AvailableRenewable_MW", "NonAI_IT_Load_MW",
            "Available_GPU", "Max_IT_Power_MW", "PUE", "Max_Facility_Power_MW",
        ],
        "q4_region_hour_input.csv",
    )
    tasks = _numeric(
        tasks,
        [
            "ArrivalHour", "GPU_Demand", "MaxLatency_ms", "EarliestStartHour",
            "LatestFinishHour", "Duration_h", "Task_Full_IT_Power_MW",
        ],
        "tasks_clean.csv",
    )
    candidates = _numeric(candidates, ["NetworkLatency_ms", "MaxLatency_ms"], "task_candidate_regions.csv")
    storage = _numeric(
        storage,
        [column for column in storage.columns if column != "Region"],
        "storage_params.csv",
    )
    region_hour["Hour"] = region_hour["Hour"].astype(int)
    region_hour["Region"] = region_hour["Region"].astype(str)
    tasks["TaskID"] = tasks["TaskID"].astype(str)
    tasks["TaskType"] = tasks["TaskType"].astype(str)
    tasks["SourceRegion"] = tasks["SourceRegion"].astype(str)
    candidates["TaskID"] = candidates["TaskID"].astype(str)
    candidates["TargetRegion"] = candidates["TargetRegion"].astype(str)
    storage["Region"] = storage["Region"].astype(str)
    regions = tuple(storage["Region"])
    region_index = {region: index for index, region in enumerate(regions)}
    if len(region_index) != len(regions):
        raise ValueError("storage_params.csv的Region必须唯一")
    expected_hours = set(range(OPERATION_END + 1))
    for region in regions:
        actual = set(region_hour.loc[region_hour["Region"].eq(region), "Hour"])
        if actual != expected_hours:
            raise ValueError(f"{region}没有完整覆盖0--2406小时")
    region_hour = region_hour.sort_values(["Hour", "Region"], kind="stable").reset_index(drop=True)
    candidate_map: dict[str, tuple[CandidateRegion, ...]] = {}
    for task_id, group in candidates.groupby("TaskID", sort=False):
        rows: list[CandidateRegion] = []
        for row in group.itertuples(index=False):
            region = str(row.TargetRegion)
            if region not in region_index:
                raise ValueError(f"任务{task_id}的候选区域{region}不在储能区域集合中")
            if float(row.NetworkLatency_ms) > float(row.MaxLatency_ms) + EPS:
                raise ValueError(f"任务{task_id}存在超时延候选区域")
            rows.append(CandidateRegion(region, region_index[region], float(row.NetworkLatency_ms), float(row.MaxLatency_ms)))
        candidate_map[str(task_id)] = tuple(rows)
    missing = set(tasks["TaskID"]) - set(candidate_map)
    if missing:
        raise ValueError(f"任务缺少候选区域，示例：{sorted(missing)[:10]}")
    task_lookup = tasks.set_index("TaskID", drop=False)
    return InputData(region_hour, tasks, candidates, storage, regions, region_index, candidate_map, task_lookup)


def _latest_integer_start(task: pd.Series | object) -> int:
    return int(math.floor(float(task.LatestFinishHour) - float(task.Duration_h) + EPS))


def _hour_overlaps(start: float, duration: float, lower: int, upper: int) -> tuple[tuple[int, float], ...]:
    first = max(lower, int(math.floor(start)))
    last = min(upper - 1, int(math.ceil(start + duration)) - 1)
    rows: list[tuple[int, float]] = []
    for hour in range(first, last + 1):
        overlap = max(0.0, min(start + duration, hour + 1.0) - max(start, float(hour)))
        if overlap > EPS:
            rows.append((hour, overlap))
    return tuple(rows)


def _feasible_starts(task: pd.Series | object, lower: int, upper: int) -> tuple[int, ...]:
    arrival = int(task.ArrivalHour)
    earliest = int(math.ceil(float(task.EarliestStartHour) - EPS))
    latest = _latest_integer_start(task)
    if str(task.TaskType) == "RealTimeInference":
        return (arrival,) if lower <= arrival < upper and earliest <= arrival <= latest else tuple()
    start = max(lower, arrival, earliest)
    end = min(upper - 1, latest)
    return tuple(range(start, end + 1)) if start <= end else tuple()


def _fixed_task_loads(
    assignments: pd.DataFrame,
    lower: int,
    upper: int,
    region_index: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    shape = (upper - lower, len(region_index))
    gpu = np.zeros(shape, dtype=float)
    ai = np.zeros(shape, dtype=float)
    if assignments.empty:
        return gpu, ai
    for row in assignments.itertuples(index=False):
        region = str(row.TargetRegion)
        if region not in region_index:
            raise ValueError(f"既有任务{row.TaskID}的区域{region}无效")
        r = region_index[region]
        for hour, overlap in _hour_overlaps(float(row.StartHour), float(row.Duration_h), lower, upper):
            gpu[hour - lower, r] += float(row.GPU_Demand) * overlap
            ai[hour - lower, r] += float(row.Task_Full_IT_Power_MW) * overlap
    return gpu, ai


def _make_task_options(
    data: InputData,
    tau: int,
    decision_end: int,
    plan_end: int,
    assignments: pd.DataFrame,
    block_hours: int,
    calibration: bool,
) -> tuple[list[TaskOption], list[TaskOption], dict[str, tuple[bool, bool]]]:
    assigned = set(assignments["TaskID"].astype(str)) if not assignments.empty else set()
    considered = data.tasks.loc[
        (~data.tasks["TaskID"].isin(assigned))
        & (data.tasks["ArrivalHour"] < plan_end)
        & (data.tasks["EarliestStartHour"] < plan_end)
    ].copy()
    if calibration:
        considered = considered.loc[considered["ArrivalHour"] >= tau]
    considered = considered.sort_values(["LatestFinishHour", "ArrivalHour", "TaskID"], kind="stable")
    x_options: list[TaskOption] = []
    u_options: list[TaskOption] = []
    task_rule: dict[str, tuple[bool, bool]] = {}
    for row_index, task in considered.iterrows():
        task_id = str(task.TaskID)
        candidates = data.candidate_map[task_id]
        h_starts = _feasible_starts(task, tau, decision_end)
        k_starts = _feasible_starts(task, decision_end, plan_end)
        must_commit = (
            str(task.TaskType) == "RealTimeInference" and int(task.ArrivalHour) < decision_end
        ) or _latest_integer_start(task) < decision_end
        if str(task.TaskType) == "RealTimeInference" and int(task.ArrivalHour) < tau and not calibration:
            raise RuntimeError(f"实时任务{task_id}在到达窗口未被执行")
        if must_commit and not h_starts:
            raise RuntimeError(f"任务{task_id}已到最晚开工窗口但没有H区合法开工时刻")
        for candidate in candidates:
            for start in h_starts:
                x_options.append(TaskOption(
                    task_id=task_id,
                    task_row=int(row_index),
                    region=candidate.region,
                    region_index=candidate.region_index,
                    start_hour=float(start),
                    expected_start_hour=float(start),
                    duration_h=float(task.Duration_h),
                    gpu_demand=float(task.GPU_Demand),
                    power_mw=float(task.Task_Full_IT_Power_MW),
                    latency_ratio=float(candidate.latency_ms) / max(float(task.MaxLatency_ms), EPS),
                    overlaps=_hour_overlaps(float(start), float(task.Duration_h), tau, plan_end),
                    is_block=False,
                ))
        if k_starts:
            for block_start in range(decision_end, plan_end, block_hours):
                block_end = min(block_start + block_hours, plan_end)
                starts = tuple(start for start in k_starts if block_start <= start < block_end)
                if not starts:
                    continue
                overlap_sum: dict[int, float] = {}
                for start in starts:
                    for hour, overlap in _hour_overlaps(float(start), float(task.Duration_h), tau, plan_end):
                        overlap_sum[hour] = overlap_sum.get(hour, 0.0) + overlap / len(starts)
                average = tuple(sorted((hour, value) for hour, value in overlap_sum.items() if value > EPS))
                expected_start = float(np.mean(starts))
                for candidate in candidates:
                    u_options.append(TaskOption(
                        task_id=task_id,
                        task_row=int(row_index),
                        region=candidate.region,
                        region_index=candidate.region_index,
                        start_hour=float("nan"),
                        expected_start_hour=expected_start,
                        duration_h=float(task.Duration_h),
                        gpu_demand=float(task.GPU_Demand),
                        power_mw=float(task.Task_Full_IT_Power_MW),
                        latency_ratio=float(candidate.latency_ms) / max(float(task.MaxLatency_ms), EPS),
                        overlaps=average,
                        is_block=True,
                        block_start=block_start,
                        block_end=block_end,
                    ))
        if h_starts or k_starts or must_commit:
            task_rule[task_id] = (must_commit, bool(k_starts))
    return x_options, u_options, task_rule


def _rows_to_sparse(rows: Sequence[Mapping[int, float]], variable_count: int):
    from scipy.sparse import coo_matrix

    row_ids: list[int] = []
    col_ids: list[int] = []
    values: list[float] = []
    for row_index, row in enumerate(rows):
        for column, value in row.items():
            if abs(value) > 0.0:
                row_ids.append(row_index)
                col_ids.append(int(column))
                values.append(float(value))
    return coo_matrix((values, (row_ids, col_ids)), shape=(len(rows), variable_count)).tocsr()


def _linear_constraint_components(
    matrix: object,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[object | None, np.ndarray | None, object | None, np.ndarray | None]:
    """将SciPy LinearConstraint边界拆成linprog可用的等式/不等式矩阵。"""

    from scipy.sparse import csr_matrix, vstack

    constraint_matrix = matrix.tocsr() if hasattr(matrix, "tocsr") else csr_matrix(matrix)
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    equality_mask = (
        finite_lower
        & finite_upper
        & np.isclose(lower, upper, rtol=0.0, atol=EPS)
    )
    equality_index = np.flatnonzero(equality_mask)
    a_eq = constraint_matrix[equality_index] if len(equality_index) else None
    b_eq = lower[equality_index] if len(equality_index) else None

    upper_index = np.flatnonzero(finite_upper & ~equality_mask)
    lower_index = np.flatnonzero(finite_lower & ~equality_mask)
    blocks = []
    bounds = []
    if len(upper_index):
        blocks.append(constraint_matrix[upper_index])
        bounds.append(upper[upper_index])
    if len(lower_index):
        blocks.append(-constraint_matrix[lower_index])
        bounds.append(-lower[lower_index])
    if blocks:
        a_ub = vstack(blocks, format="csr")
        b_ub = np.concatenate(bounds)
    else:
        a_ub = None
        b_ub = None
    return a_ub, b_ub, a_eq, b_eq


def _allocate(offset: int, shape: tuple[int, ...]) -> tuple[np.ndarray, int]:
    count = int(np.prod(shape))
    return np.arange(offset, offset + count, dtype=int).reshape(shape), offset + count


def build_window_problem(
    data: InputData,
    config: ModelConfig,
    tau: int,
    decision_hours: int,
    lookahead_hours: int,
    assignments: pd.DataFrame,
    current_soc: np.ndarray,
    historical_peak: np.ndarray,
    scaling: Mapping[str, tuple[float, float]] | None,
    *,
    block_hours: int | None = None,
    k_integer: bool = False,
    calibration: bool = False,
    protect_h_tail_capacity: bool = False,
) -> WindowProblem:
    decision_end = min(tau + decision_hours, OPERATION_END)
    plan_end = min(decision_end + lookahead_hours, OPERATION_END)
    h_count = decision_end - tau
    t_count = plan_end - tau
    r_count = len(data.regions)
    if h_count <= 0 or t_count <= 0:
        raise ValueError("滚动窗口长度必须为正")
    block_hours = int(block_hours or config.block_hours)
    x_options, u_options, task_rules = _make_task_options(
        data, tau, decision_end, plan_end, assignments, block_hours, calibration
    )
    offset = 0
    indices: dict[str, np.ndarray] = {}
    indices["x"], offset = _allocate(offset, (len(x_options),))
    indices["u"], offset = _allocate(offset, (len(u_options),))
    for name in ("renewable_direct", "renewable_charge", "export", "curtailment", "grid_purchase", "grid_charge", "discharge", "mode"):
        indices[name], offset = _allocate(offset, (t_count, r_count))
    indices["soc"], offset = _allocate(offset, (t_count + 1, r_count))
    indices["peak"], offset = _allocate(offset, (r_count,))
    indices["peak_increment"], offset = _allocate(offset, (r_count,))
    if scaling is not None:
        indices["deviation"], offset = _allocate(offset, (len(METRICS),))
        indices["max_deviation"], offset = _allocate(offset, (1,))
    variable_count = offset
    lower = np.zeros(variable_count, dtype=float)
    upper = np.full(variable_count, np.inf, dtype=float)
    integrality = np.zeros(variable_count, dtype=np.uint8)
    upper[indices["x"]] = 1.0
    upper[indices["u"]] = 1.0
    integrality[indices["x"]] = 1
    if k_integer:
        integrality[indices["u"]] = 1
    upper[indices["mode"]] = 1.0
    integrality[indices["mode"][:h_count].ravel()] = 1

    storage = data.storage.set_index("Region").loc[list(data.regions)]
    min_soc = storage["MinSOC_MWh"].to_numpy(dtype=float)
    max_soc = storage["StorageCapacity_MWh"].to_numpy(dtype=float)
    max_charge = storage["MaxChargePower_MW"].to_numpy(dtype=float)
    max_discharge = storage["MaxDischargePower_MW"].to_numpy(dtype=float)
    eta_c = storage["ChargeEfficiency"].to_numpy(dtype=float)
    eta_d = storage["DischargeEfficiency"].to_numpy(dtype=float)
    initial_soc = storage["InitialSOC_MWh"].to_numpy(dtype=float)
    max_grid = storage["MaxGridImport_MW"].to_numpy(dtype=float)
    max_export = np.minimum(
        storage["SellLimit_MW"].to_numpy(dtype=float),
        storage["MaxGridExport_MW"].to_numpy(dtype=float),
    )
    lower[indices["soc"].ravel()] = np.tile(min_soc, t_count + 1)
    upper[indices["soc"].ravel()] = np.tile(max_soc, t_count + 1)
    lower[indices["soc"][0]] = current_soc
    upper[indices["soc"][0]] = current_soc
    recoverable = np.maximum(min_soc, initial_soc - eta_c * max_charge * max(OPERATION_END - decision_end, 0))
    lower[indices["soc"][h_count]] = np.maximum(lower[indices["soc"][h_count]], recoverable)
    upper[indices["grid_purchase"].ravel()] = np.tile(max_grid, t_count)
    upper[indices["export"].ravel()] = np.tile(max_export, t_count)
    upper[indices["discharge"].ravel()] = np.tile(max_discharge, t_count)

    frame = data.region_hour.loc[
        data.region_hour["Hour"].between(tau, plan_end - 1)
    ].copy()
    frame["Region"] = pd.Categorical(frame["Region"], categories=data.regions, ordered=True)
    frame = frame.sort_values(["Hour", "Region"], kind="stable")
    if len(frame) != t_count * r_count:
        raise ValueError(f"窗口{tau}的逐时区域数据不完整")
    arrays = {
        column: frame[column].to_numpy(dtype=float).reshape(t_count, r_count)
        for column in (
            "ElectricityPrice_CNY_per_MWh", "SellPrice_CNY_per_MWh",
            "CarbonIntensity_tCO2_per_MWh", "AvailableRenewable_MW", "NonAI_IT_Load_MW",
            "Available_GPU", "Max_IT_Power_MW", "PUE", "Max_Facility_Power_MW",
        )
    }
    upper[indices["renewable_direct"].ravel()] = arrays["AvailableRenewable_MW"].ravel()
    upper[indices["renewable_charge"].ravel()] = arrays["AvailableRenewable_MW"].ravel()
    upper[indices["curtailment"].ravel()] = arrays["AvailableRenewable_MW"].ravel()
    fixed_gpu, fixed_ai = _fixed_task_loads(assignments, tau, plan_end, data.region_index)

    rows: list[dict[int, float]] = []
    row_lower: list[float] = []
    row_upper: list[float] = []

    def add_row(values: Mapping[int, float], lb: float, ub: float) -> None:
        rows.append(dict(values))
        row_lower.append(float(lb))
        row_upper.append(float(ub))

    x_by_task: dict[str, list[int]] = {}
    u_by_task: dict[str, list[int]] = {}
    for index, option in enumerate(x_options):
        x_by_task.setdefault(option.task_id, []).append(int(indices["x"][index]))
    for index, option in enumerate(u_options):
        u_by_task.setdefault(option.task_id, []).append(int(indices["u"][index]))
    for task_id, (must_commit, has_k) in task_rules.items():
        x_cols = x_by_task.get(task_id, [])
        u_cols = u_by_task.get(task_id, [])
        if u_cols:
            add_row({column: 1.0 for column in x_cols + u_cols}, 1.0, 1.0)
        elif must_commit:
            add_row({column: 1.0 for column in x_cols}, 1.0, 1.0)
        elif x_cols:
            add_row({column: 1.0 for column in x_cols}, -np.inf, 1.0)
        elif has_k:
            raise RuntimeError(f"任务{task_id}的K区标记与变量不一致")

    gpu_rows = [[{} for _ in range(r_count)] for _ in range(t_count)]
    ai_rows = [[{} for _ in range(r_count)] for _ in range(t_count)]
    all_options: tuple[tuple[TaskOption, int], ...] = tuple(
        [(option, int(indices["x"][index])) for index, option in enumerate(x_options)]
        + [(option, int(indices["u"][index])) for index, option in enumerate(u_options)]
    )
    for option, column in all_options:
        for hour, overlap in option.overlaps:
            local = hour - tau
            if 0 <= local < t_count:
                gpu_rows[local][option.region_index][column] = option.gpu_demand * overlap
                ai_rows[local][option.region_index][column] = option.power_mw * overlap
    for t in range(t_count):
        for r in range(r_count):
            ai_capacity = min(
                arrays["Max_IT_Power_MW"][t, r] - arrays["NonAI_IT_Load_MW"][t, r],
                arrays["Max_Facility_Power_MW"][t, r] / arrays["PUE"][t, r] - arrays["NonAI_IT_Load_MW"][t, r],
            )
            add_row(gpu_rows[t][r], -np.inf, arrays["Available_GPU"][t, r] - fixed_gpu[t, r])
            add_row(ai_rows[t][r], -np.inf, ai_capacity - fixed_ai[t, r])

    tail_capacity_constraint_count = 0
    tail_protection_end = decision_end
    if protect_h_tail_capacity:
        possible_finishes = [
            option.start_hour + option.duration_h
            for option in x_options
            if option.start_hour < decision_end
        ]
        if not assignments.empty:
            possible_finishes.extend(
                pd.to_numeric(assignments.loc[
                    pd.to_numeric(assignments["FinishHour"]) > decision_end + EPS,
                    "FinishHour",
                ]).tolist()
            )
        if possible_finishes:
            tail_protection_end = min(
                OPERATION_END,
                max(decision_end, int(math.ceil(max(possible_finishes) - EPS))),
            )
        tail_count = tail_protection_end - decision_end
        if tail_count > 0:
            tail_frame = data.region_hour.loc[
                data.region_hour["Hour"].between(decision_end, tail_protection_end - 1)
            ].copy()
            tail_frame["Region"] = pd.Categorical(
                tail_frame["Region"], categories=data.regions, ordered=True
            )
            tail_frame = tail_frame.sort_values(["Hour", "Region"], kind="stable")
            if len(tail_frame) != tail_count * r_count:
                raise ValueError(
                    f"H区尾部容量保护缺少{decision_end}--{tail_protection_end - 1}小时数据"
                )
            tail_available_gpu = tail_frame["Available_GPU"].to_numpy(dtype=float).reshape(
                tail_count, r_count
            )
            tail_non_ai = tail_frame["NonAI_IT_Load_MW"].to_numpy(dtype=float).reshape(
                tail_count, r_count
            )
            tail_max_it = tail_frame["Max_IT_Power_MW"].to_numpy(dtype=float).reshape(
                tail_count, r_count
            )
            tail_pue = tail_frame["PUE"].to_numpy(dtype=float).reshape(tail_count, r_count)
            tail_max_facility = tail_frame["Max_Facility_Power_MW"].to_numpy(
                dtype=float
            ).reshape(tail_count, r_count)
            tail_fixed_gpu, tail_fixed_ai = _fixed_task_loads(
                assignments, decision_end, tail_protection_end, data.region_index
            )
            tail_gpu_rows = [[{} for _ in range(r_count)] for _ in range(tail_count)]
            tail_ai_rows = [[{} for _ in range(r_count)] for _ in range(tail_count)]
            for index, option in enumerate(x_options):
                column = int(indices["x"][index])
                for hour, overlap in _hour_overlaps(
                    option.start_hour,
                    option.duration_h,
                    decision_end,
                    tail_protection_end,
                ):
                    local = hour - decision_end
                    tail_gpu_rows[local][option.region_index][column] = (
                        option.gpu_demand * overlap
                    )
                    tail_ai_rows[local][option.region_index][column] = (
                        option.power_mw * overlap
                    )
            for t in range(tail_count):
                for r in range(r_count):
                    tail_ai_capacity = min(
                        tail_max_it[t, r] - tail_non_ai[t, r],
                        tail_max_facility[t, r] / tail_pue[t, r] - tail_non_ai[t, r],
                    )
                    add_row(
                        tail_gpu_rows[t][r],
                        -np.inf,
                        tail_available_gpu[t, r] - tail_fixed_gpu[t, r],
                    )
                    add_row(
                        tail_ai_rows[t][r],
                        -np.inf,
                        tail_ai_capacity - tail_fixed_ai[t, r],
                    )
                    tail_capacity_constraint_count += 2

    for t in range(t_count):
        for r in range(r_count):
            renewable = float(arrays["AvailableRenewable_MW"][t, r])
            pue = float(arrays["PUE"][t, r])
            base_facility = pue * (float(arrays["NonAI_IT_Load_MW"][t, r]) + fixed_ai[t, r])
            task_facility = {column: pue * value for column, value in ai_rows[t][r].items()}
            add_row({
                int(indices["renewable_direct"][t, r]): 1.0,
                int(indices["renewable_charge"][t, r]): 1.0,
                int(indices["export"][t, r]): 1.0,
                int(indices["curtailment"][t, r]): 1.0,
            }, renewable, renewable)
            balance = {
                int(indices["grid_purchase"][t, r]): 1.0,
                int(indices["renewable_direct"][t, r]): 1.0,
                int(indices["discharge"][t, r]): 1.0,
                int(indices["grid_charge"][t, r]): -1.0,
            }
            for column, value in task_facility.items():
                balance[column] = balance.get(column, 0.0) - value
            add_row(balance, base_facility, base_facility)
            direct_limit = {int(indices["renewable_direct"][t, r]): 1.0}
            for column, value in task_facility.items():
                direct_limit[column] = direct_limit.get(column, 0.0) - value
            add_row(direct_limit, -np.inf, base_facility)
            add_row({
                int(indices["grid_charge"][t, r]): 1.0,
                int(indices["grid_purchase"][t, r]): -1.0,
            }, -np.inf, 0.0)
            add_row({
                int(indices["renewable_charge"][t, r]): 1.0,
                int(indices["grid_charge"][t, r]): 1.0,
                int(indices["mode"][t, r]): -max_charge[r],
            }, -np.inf, 0.0)
            add_row({
                int(indices["discharge"][t, r]): 1.0,
                int(indices["mode"][t, r]): max_discharge[r],
            }, -np.inf, max_discharge[r])
            add_row({
                int(indices["soc"][t + 1, r]): 1.0,
                int(indices["soc"][t, r]): -1.0,
                int(indices["renewable_charge"][t, r]): -eta_c[r],
                int(indices["grid_charge"][t, r]): -eta_c[r],
                int(indices["discharge"][t, r]): 1.0 / eta_d[r],
            }, 0.0, 0.0)
    for r in range(r_count):
        for t in range(h_count):
            add_row({
                int(indices["grid_purchase"][t, r]): 1.0,
                int(indices["export"][t, r]): -1.0,
                int(indices["peak"][r]): -1.0,
            }, -np.inf, 0.0)
        add_row({
            int(indices["peak"][r]): 1.0,
            int(indices["peak_increment"][r]): -1.0,
        }, -np.inf, float(historical_peak[r]))

    metric_vectors = {metric: np.zeros(variable_count, dtype=float) for metric in METRICS}
    metric_constants = {metric: 0.0 for metric in METRICS}
    k_count = plan_end - decision_end
    beta = h_count / k_count if k_count > 0 else 0.0
    time_weight = np.ones(t_count, dtype=float)
    if k_count:
        time_weight[h_count:] = beta
    price = arrays["ElectricityPrice_CNY_per_MWh"]
    sell = arrays["SellPrice_CNY_per_MWh"]
    carbon = arrays["CarbonIntensity_tCO2_per_MWh"]
    metric_vectors["Cost"][indices["grid_purchase"].ravel()] = (price * time_weight[:, None]).ravel()
    metric_vectors["Cost"][indices["export"].ravel()] = (-sell * time_weight[:, None]).ravel()
    metric_vectors["Carbon"][indices["grid_purchase"].ravel()] = (carbon * time_weight[:, None]).ravel()
    renewable_denominator = float(np.sum(arrays["AvailableRenewable_MW"] * time_weight[:, None]))
    if renewable_denominator > EPS:
        metric_vectors["RenewableUnusedRate"][indices["curtailment"].ravel()] = (
            np.repeat(time_weight, r_count) / renewable_denominator
        )
    task_ids = set(task_rules)
    latency_denominator = max(len(task_ids), 1)
    flexible_ids = {
        task_id for task_id in task_ids
        if float(data.task_lookup.loc[task_id, "LatestFinishHour"])
        - float(data.task_lookup.loc[task_id, "Duration_h"])
        - float(data.task_lookup.loc[task_id, "EarliestStartHour"]) > EPS
    }
    delay_denominator = max(len(flexible_ids), 1)
    for index, option in enumerate(x_options):
        column = int(indices["x"][index])
        task = data.task_lookup.loc[option.task_id]
        metric_vectors["Latency"][column] = option.latency_ratio / latency_denominator
        slack = float(task.LatestFinishHour) - float(task.Duration_h) - float(task.EarliestStartHour)
        if slack > EPS:
            metric_vectors["Delay"][column] = max(option.expected_start_hour - float(task.EarliestStartHour), 0.0) / slack / delay_denominator
    for index, option in enumerate(u_options):
        column = int(indices["u"][index])
        task = data.task_lookup.loc[option.task_id]
        metric_vectors["Latency"][column] = beta * option.latency_ratio / latency_denominator
        slack = float(task.LatestFinishHour) - float(task.Duration_h) - float(task.EarliestStartHour)
        if slack > EPS:
            metric_vectors["Delay"][column] = beta * max(option.expected_start_hour - float(task.EarliestStartHour), 0.0) / slack / delay_denominator
    metric_vectors["Peak"][indices["peak_increment"]] = 1.0

    if scaling is not None:
        for metric_index, metric in enumerate(METRICS):
            anchor, scale = scaling[metric]
            row = {
                column: float(value)
                for column, value in enumerate(metric_vectors[metric]) if abs(value) > 0.0
            }
            row[int(indices["deviation"][metric_index])] = -float(scale)
            add_row(row, -np.inf, float(anchor) - metric_constants[metric])
            add_row({
                int(indices["deviation"][metric_index]): 1.0,
                int(indices["max_deviation"][0]): -1.0,
            }, -np.inf, 0.0)

    matrix = _rows_to_sparse(rows, variable_count)
    return WindowProblem(
        tau=tau,
        decision_end=decision_end,
        plan_end=plan_end,
        regions=data.regions,
        x_options=tuple(x_options),
        u_options=tuple(u_options),
        task_rules=task_rules,
        indices=indices,
        lower=lower,
        upper=upper,
        integrality=integrality,
        matrix=matrix,
        constraint_lower=np.asarray(row_lower, dtype=float),
        constraint_upper=np.asarray(row_upper, dtype=float),
        metric_vectors=metric_vectors,
        metric_constants=metric_constants,
        metadata={
            "TaskCount": len(task_rules), "HOptionCount": len(x_options),
            "KOptionCount": len(u_options), "VariableCount": variable_count,
            "ConstraintCount": len(rows), "Beta": beta,
            "BlockHours": block_hours,
            "HTailCapacityProtection": bool(protect_h_tail_capacity),
            "HTailProtectionEnd": int(tail_protection_end),
            "HTailCapacityConstraintCount": int(tail_capacity_constraint_count),
        },
    )


def _heuristic_incumbent(
    data: InputData,
    problem: WindowProblem,
    assignments: pd.DataFrame,
    current_soc: np.ndarray,
    historical_peak: np.ndarray,
    scaling: Mapping[str, tuple[float, float]] | None,
    warm_hints: Mapping[str, tuple[str, float]],
    tolerance: float,
) -> np.ndarray | None:
    """构造任务EDF可行解和逐时能源可行解，作为主MILP的可行上界。"""

    tau = problem.tau
    h_count = problem.decision_end - tau
    t_count = problem.plan_end - tau
    r_count = len(data.regions)
    frame = data.region_hour.loc[
        data.region_hour["Hour"].between(tau, problem.plan_end - 1)
    ].copy()
    frame["Region"] = pd.Categorical(frame["Region"], categories=data.regions, ordered=True)
    frame = frame.sort_values(["Hour", "Region"], kind="stable")
    arrays = {
        column: frame[column].to_numpy(dtype=float).reshape(t_count, r_count)
        for column in (
            "ElectricityPrice_CNY_per_MWh", "CarbonIntensity_tCO2_per_MWh",
            "AvailableRenewable_MW", "NonAI_IT_Load_MW", "Available_GPU",
            "Max_IT_Power_MW", "PUE", "Max_Facility_Power_MW",
        )
    }
    fixed_gpu, fixed_ai = _fixed_task_loads(assignments, tau, problem.plan_end, data.region_index)
    used_gpu = fixed_gpu.copy()
    used_ai = fixed_ai.copy()
    gpu_limit = arrays["Available_GPU"]
    ai_limit = np.minimum(
        arrays["Max_IT_Power_MW"] - arrays["NonAI_IT_Load_MW"],
        arrays["Max_Facility_Power_MW"] / arrays["PUE"] - arrays["NonAI_IT_Load_MW"],
    )
    vector = np.zeros(problem.lower.size, dtype=float)
    x_by_task: dict[str, list[tuple[TaskOption, int]]] = {}
    u_by_task: dict[str, list[tuple[TaskOption, int]]] = {}
    for index, option in enumerate(problem.x_options):
        x_by_task.setdefault(option.task_id, []).append((option, int(problem.indices["x"][index])))
    for index, option in enumerate(problem.u_options):
        u_by_task.setdefault(option.task_id, []).append((option, int(problem.indices["u"][index])))

    def option_key(item: tuple[TaskOption, int]) -> tuple[float | int | str, ...]:
        option = item[0]
        hint = warm_hints.get(option.task_id)
        if hint is None:
            hint_region_penalty = 0
            hint_time_distance = 0.0
        else:
            hint_region_penalty = int(option.region != hint[0])
            hint_time_distance = abs(option.expected_start_hour - hint[1])
        energy_rows = [
            (hour - tau, overlap) for hour, overlap in option.overlaps
            if tau <= hour < problem.plan_end
        ]
        weight = sum(overlap for _, overlap in energy_rows)
        if weight > EPS:
            shortage = sum(
                overlap * max(
                    arrays["PUE"][local, option.region_index]
                    * arrays["NonAI_IT_Load_MW"][local, option.region_index]
                    - arrays["AvailableRenewable_MW"][local, option.region_index],
                    0.0,
                )
                for local, overlap in energy_rows
            ) / weight
            price = sum(
                overlap * arrays["ElectricityPrice_CNY_per_MWh"][local, option.region_index]
                for local, overlap in energy_rows
            ) / weight
            carbon = sum(
                overlap * arrays["CarbonIntensity_tCO2_per_MWh"][local, option.region_index]
                for local, overlap in energy_rows
            ) / weight
        else:
            shortage = price = carbon = 0.0
        return (
            hint_region_penalty, hint_time_distance, int(option.is_block),
            shortage, price, carbon, option.latency_ratio,
            option.expected_start_hour, option.region,
        )

    def fits(option: TaskOption) -> bool:
        for hour, overlap in option.overlaps:
            local = hour - tau
            r = option.region_index
            if used_gpu[local, r] + option.gpu_demand * overlap > gpu_limit[local, r] + tolerance:
                return False
            if used_ai[local, r] + option.power_mw * overlap > ai_limit[local, r] + tolerance:
                return False
        return True

    def select(option: TaskOption, column: int) -> None:
        vector[column] = 1.0
        for hour, overlap in option.overlaps:
            local = hour - tau
            r = option.region_index
            used_gpu[local, r] += option.gpu_demand * overlap
            used_ai[local, r] += option.power_mw * overlap

    ordered_tasks = sorted(
        problem.task_rules,
        key=lambda task_id: (
            not problem.task_rules[task_id][0],
            _latest_integer_start(data.task_lookup.loc[task_id]),
            int(data.task_lookup.loc[task_id, "ArrivalHour"]),
            task_id,
        ),
    )
    for task_id in ordered_tasks:
        must_commit, has_k = problem.task_rules[task_id]
        if must_commit:
            choices = x_by_task.get(task_id, [])
        elif has_k:
            choices = x_by_task.get(task_id, []) + u_by_task.get(task_id, [])
        else:
            continue
        chosen = next((item for item in sorted(choices, key=option_key) if fits(item[0])), None)
        if chosen is None:
            return None
        select(*chosen)

    storage = data.storage.set_index("Region").loc[list(data.regions)]
    min_soc = storage["MinSOC_MWh"].to_numpy(dtype=float)
    max_soc = storage["StorageCapacity_MWh"].to_numpy(dtype=float)
    initial_soc = storage["InitialSOC_MWh"].to_numpy(dtype=float)
    max_charge = storage["MaxChargePower_MW"].to_numpy(dtype=float)
    max_discharge = storage["MaxDischargePower_MW"].to_numpy(dtype=float)
    eta_c = storage["ChargeEfficiency"].to_numpy(dtype=float)
    eta_d = storage["DischargeEfficiency"].to_numpy(dtype=float)
    max_grid = storage["MaxGridImport_MW"].to_numpy(dtype=float)
    max_export = np.minimum(
        storage["SellLimit_MW"].to_numpy(dtype=float),
        storage["MaxGridExport_MW"].to_numpy(dtype=float),
    )
    recoverable = np.maximum(
        min_soc,
        initial_soc - eta_c * max_charge * max(OPERATION_END - problem.decision_end, 0),
    )
    vector[problem.indices["soc"][0]] = current_soc
    for t in range(t_count):
        for r in range(r_count):
            soc_now = float(vector[problem.indices["soc"][t, r]])
            if t < h_count:
                remaining_h = h_count - t - 1
                next_floor = max(min_soc[r], recoverable[r] - eta_c[r] * max_charge[r] * remaining_h)
            else:
                next_floor = min_soc[r]
            facility = arrays["PUE"][t, r] * (
                arrays["NonAI_IT_Load_MW"][t, r] + used_ai[t, r]
            )
            renewable = arrays["AvailableRenewable_MW"][t, r]
            direct = min(renewable, facility)
            remaining_load = facility - direct
            renewable_surplus = renewable - direct
            room_input = max((max_soc[r] - soc_now) / eta_c[r], 0.0)
            renewable_charge = min(renewable_surplus, max_charge[r], room_input)
            soc_after_renewable = soc_now + eta_c[r] * renewable_charge
            remaining_charge_power = max(max_charge[r] - renewable_charge, 0.0)
            remaining_room = max(room_input - renewable_charge, 0.0)
            required_grid_charge = max((next_floor - soc_after_renewable) / eta_c[r], 0.0)
            grid_charge = min(required_grid_charge, remaining_charge_power, remaining_room)
            if required_grid_charge > grid_charge + tolerance:
                return None
            if grid_charge > EPS:
                discharge = 0.0
            else:
                discharge = min(
                    remaining_load,
                    max_discharge[r],
                    max((soc_after_renewable - next_floor) * eta_d[r], 0.0),
                )
            grid_purchase = remaining_load - discharge + grid_charge
            if grid_purchase > max_grid[r] + tolerance:
                return None
            soc_next = (
                soc_now + eta_c[r] * (renewable_charge + grid_charge)
                - discharge / eta_d[r]
            )
            if soc_next < min_soc[r] - tolerance or soc_next > max_soc[r] + tolerance:
                return None
            renewable_left = renewable_surplus - renewable_charge
            export = min(renewable_left, max_export[r])
            curtailment = renewable_left - export
            vector[problem.indices["renewable_direct"][t, r]] = direct
            vector[problem.indices["renewable_charge"][t, r]] = renewable_charge
            vector[problem.indices["export"][t, r]] = export
            vector[problem.indices["curtailment"][t, r]] = curtailment
            vector[problem.indices["grid_purchase"][t, r]] = grid_purchase
            vector[problem.indices["grid_charge"][t, r]] = grid_charge
            vector[problem.indices["discharge"][t, r]] = discharge
            vector[problem.indices["mode"][t, r]] = float(renewable_charge + grid_charge > EPS)
            vector[problem.indices["soc"][t + 1, r]] = soc_next
    for r in range(r_count):
        peak = max(
            (
                vector[problem.indices["grid_purchase"][t, r]]
                - vector[problem.indices["export"][t, r]]
                for t in range(h_count)
            ),
            default=0.0,
        )
        vector[problem.indices["peak"][r]] = max(peak, 0.0)
        vector[problem.indices["peak_increment"][r]] = max(peak - historical_peak[r], 0.0)
    if scaling is not None:
        for metric_index, metric in enumerate(METRICS):
            value = float(problem.metric_constants[metric] + np.dot(problem.metric_vectors[metric], vector))
            anchor, scale = scaling[metric]
            vector[problem.indices["deviation"][metric_index]] = max((value - anchor) / scale, 0.0)
        vector[problem.indices["max_deviation"][0]] = float(np.max(vector[problem.indices["deviation"]]))
    activity = problem.matrix @ vector
    violation = max(
        float(np.max(np.maximum(problem.constraint_lower - activity, 0.0), initial=0.0)),
        float(np.max(np.maximum(activity - problem.constraint_upper, 0.0), initial=0.0)),
    )
    bound_violation = max(
        float(np.max(np.maximum(problem.lower - vector, 0.0), initial=0.0)),
        float(np.max(np.maximum(vector - problem.upper, 0.0), initial=0.0)),
    )
    return vector if max(violation, bound_violation) <= tolerance else None


def _k_capacity_violation(
    data: InputData,
    problem: WindowProblem,
    vector: np.ndarray,
    assignments: pd.DataFrame,
) -> float:
    """复核K区块平均GPU、IT和设施容量，用于触发4h到2h局部细化。"""

    h_count = problem.decision_end - problem.tau
    t_count = problem.plan_end - problem.tau
    if h_count >= t_count:
        return 0.0
    frame = data.region_hour.loc[
        data.region_hour["Hour"].between(problem.tau, problem.plan_end - 1)
    ].copy()
    frame["Region"] = pd.Categorical(frame["Region"], categories=data.regions, ordered=True)
    frame = frame.sort_values(["Hour", "Region"], kind="stable")
    r_count = len(data.regions)
    available_gpu = frame["Available_GPU"].to_numpy(dtype=float).reshape(t_count, r_count)
    non_ai = frame["NonAI_IT_Load_MW"].to_numpy(dtype=float).reshape(t_count, r_count)
    max_it = frame["Max_IT_Power_MW"].to_numpy(dtype=float).reshape(t_count, r_count)
    pue = frame["PUE"].to_numpy(dtype=float).reshape(t_count, r_count)
    max_facility = frame["Max_Facility_Power_MW"].to_numpy(dtype=float).reshape(t_count, r_count)
    gpu, ai = _fixed_task_loads(assignments, problem.tau, problem.plan_end, data.region_index)
    for index, option in enumerate(problem.x_options):
        value = float(vector[problem.indices["x"][index]])
        for hour, overlap in option.overlaps:
            local = hour - problem.tau
            gpu[local, option.region_index] += value * option.gpu_demand * overlap
            ai[local, option.region_index] += value * option.power_mw * overlap
    for index, option in enumerate(problem.u_options):
        value = float(vector[problem.indices["u"][index]])
        for hour, overlap in option.overlaps:
            local = hour - problem.tau
            gpu[local, option.region_index] += value * option.gpu_demand * overlap
            ai[local, option.region_index] += value * option.power_mw * overlap
    gpu_violation = np.maximum(gpu[h_count:] - available_gpu[h_count:], 0.0)
    it_violation = np.maximum(non_ai[h_count:] + ai[h_count:] - max_it[h_count:], 0.0)
    facility_violation = np.maximum(
        pue[h_count:] * (non_ai[h_count:] + ai[h_count:]) - max_facility[h_count:], 0.0
    )
    return float(max(
        np.max(gpu_violation, initial=0.0),
        np.max(it_violation, initial=0.0),
        np.max(facility_violation, initial=0.0),
    ))


def _next_warm_hints(problem: WindowProblem, vector: np.ndarray) -> dict[str, tuple[str, float]]:
    """保留上一窗口K区权重最大的区域—时间块，供下一窗口启发式使用。"""

    best: dict[str, tuple[float, str, float]] = {}
    for index, option in enumerate(problem.u_options):
        value = float(vector[problem.indices["u"][index]])
        current = best.get(option.task_id)
        if current is None or value > current[0]:
            best[option.task_id] = (value, option.region, option.expected_start_hour)
    return {
        task_id: (region, expected_start)
        for task_id, (value, region, expected_start) in best.items()
        if value > EPS
    }


def _solve(
    problem: WindowProblem,
    objective: np.ndarray,
    *,
    time_limit: float,
    mip_gap: float,
    relax: bool = False,
    incumbent: np.ndarray | None = None,
) -> WindowSolution:
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp
    from scipy.sparse import csr_matrix

    stage = f"窗口{problem.tau} {'连续松弛' if relax else 'MILP'}"
    _progress(
        f"{stage}开始：变量={objective.size}，约束={problem.matrix.shape[0]}，"
        f"时间上限={time_limit:.0f}s，mip_rel_gap={mip_gap:g}。"
    )
    integrality = np.zeros_like(problem.integrality) if relax else problem.integrality
    constraints: list[LinearConstraint] = [
        LinearConstraint(problem.matrix, problem.constraint_lower, problem.constraint_upper)
    ]
    if incumbent is not None:
        incumbent = np.asarray(incumbent, dtype=float)
        if incumbent.shape != objective.shape:
            raise ValueError("启发式初始解与窗口变量维度不一致")
        cutoff = float(np.dot(objective, incumbent))
        constraints.append(LinearConstraint(
            csr_matrix(np.asarray(objective, dtype=float).reshape(1, -1)),
            -np.inf,
            cutoff + 1e-6 * max(abs(cutoff), 1.0),
        ))
    started = time.perf_counter()
    heartbeat_stop = threading.Event()

    def _heartbeat() -> None:
        while not heartbeat_stop.wait(SOLVER_PROGRESS_INTERVAL_SECONDS):
            _progress(
                f"{stage}仍在求解：已耗时{time.perf_counter() - started:.1f}s，"
                f"变量={objective.size}，约束={problem.matrix.shape[0]}。"
            )

    heartbeat = threading.Thread(
        target=_heartbeat,
        name=f"q4-window-{problem.tau}-heartbeat",
        daemon=True,
    )
    heartbeat.start()
    try:
        if relax:
            a_ub, b_ub, a_eq, b_eq = _linear_constraint_components(
                problem.matrix, problem.constraint_lower, problem.constraint_upper
            )
            result = linprog(
                c=np.asarray(objective, dtype=float),
                A_ub=a_ub,
                b_ub=b_ub,
                A_eq=a_eq,
                b_eq=b_eq,
                bounds=list(zip(problem.lower, problem.upper)),
                method="highs",
                options={
                    "presolve": True,
                    "time_limit": float(time_limit),
                },
            )
        else:
            result = milp(
                c=np.asarray(objective, dtype=float),
                integrality=integrality,
                bounds=Bounds(problem.lower, problem.upper),
                constraints=constraints,
                options={
                    "presolve": True,
                    "time_limit": float(time_limit),
                    "mip_rel_gap": float(mip_gap),
                },
            )
    finally:
        heartbeat_stop.set()
        heartbeat.join(timeout=1.0)
    elapsed = time.perf_counter() - started
    mip_gap = _result_float(result, "mip_gap")
    _progress(
        f"{stage}完成：status={getattr(result, 'status', 'NA')}，"
        f"耗时{elapsed:.2f}s，mip_gap={mip_gap}。"
    )
    vector = getattr(result, "x", None)
    if vector is None or not np.all(np.isfinite(vector)):
        raise RuntimeError(f"窗口{problem.tau}没有得到可行解：{getattr(result, 'message', '')}")
    activity = problem.matrix @ vector
    lower_violation = np.maximum(problem.constraint_lower - activity, 0.0)
    upper_violation = np.maximum(activity - problem.constraint_upper, 0.0)
    max_violation = float(max(lower_violation.max(initial=0.0), upper_violation.max(initial=0.0)))
    if max_violation > 1e-5:
        raise RuntimeError(f"窗口{problem.tau}求解向量最大约束违反为{max_violation:.3g}")
    return WindowSolution(
        vector=np.asarray(vector, dtype=float),
        status=int(getattr(result, "status", -1)),
        message=str(getattr(result, "message", "")),
        objective=float(getattr(result, "fun", np.dot(objective, vector))),
        mip_gap=mip_gap,
        mip_node_count=_result_float(result, "mip_node_count"),
        elapsed_seconds=elapsed,
    )


def _metric_values(problem: WindowProblem, vector: np.ndarray) -> dict[str, float]:
    return {
        metric: float(problem.metric_constants[metric] + np.dot(problem.metric_vectors[metric], vector))
        for metric in METRICS
    }


def _representative_windows(data: InputData, config: ModelConfig) -> tuple[int, ...]:
    candidates = np.arange(0, MAIN_END, config.decision_hours, dtype=int)
    latest_starts = np.floor(data.tasks["LatestFinishHour"] - data.tasks["Duration_h"] + EPS).astype(int)
    urgency = latest_starts.value_counts().reindex(range(OPERATION_END), fill_value=0).to_numpy(dtype=float)
    hourly = data.region_hour.loc[data.region_hour["Hour"] < MAIN_END].copy()
    hourly["BaseFacility"] = hourly["PUE"] * hourly["NonAI_IT_Load_MW"]
    base_load = hourly.groupby("Hour")["BaseFacility"].sum().reindex(range(MAIN_END), fill_value=0.0).to_numpy(dtype=float)
    arrival_ai_load = (
        data.tasks.groupby("ArrivalHour")["Task_Full_IT_Power_MW"].sum()
        .reindex(range(MAIN_END), fill_value=0.0).to_numpy(dtype=float)
    )
    high_load = base_load + arrival_ai_load
    renewable = hourly.groupby("Hour")["AvailableRenewable_MW"].sum().reindex(range(MAIN_END), fill_value=0.0).to_numpy(dtype=float)
    renewable_surplus = renewable - base_load

    def rolling_score(values: np.ndarray) -> np.ndarray:
        return np.asarray([
            values[tau:min(tau + config.decision_hours + config.lookahead_hours, len(values))].sum()
            for tau in candidates
        ])

    scores = (
        rolling_score(high_load),
        rolling_score(renewable_surplus),
        rolling_score(urgency),
    )
    selected: list[int] = []
    for score in scores:
        for index in np.argsort(score)[::-1]:
            tau = int(candidates[index])
            if tau not in selected:
                selected.append(tau)
                break
    return tuple(sorted(selected))


def _certified_box_lower_bound(
    problem: WindowProblem,
    metric: str,
) -> tuple[float, bool, str]:
    """由当前指标自身的变量边界给出可证明安全的线性目标下界。"""

    coefficients = problem.metric_vectors[metric]
    value = float(problem.metric_constants[metric])
    for column in np.flatnonzero(np.abs(coefficients) > 0.0):
        coefficient = float(coefficients[column])
        bound = float(problem.lower[column] if coefficient >= 0.0 else problem.upper[column])
        if not np.isfinite(bound):
            return float("nan"), False, "UnboundedVariableBox"
        value += coefficient * bound
    return value, True, "CertifiedVariableBox"


def _calibrate(data: InputData, config: ModelConfig) -> tuple[dict[str, tuple[float, float]], pd.DataFrame]:
    storage = data.storage.set_index("Region").loc[list(data.regions)]
    initial_soc = storage["InitialSOC_MWh"].to_numpy(dtype=float)
    empty_assignments = pd.DataFrame(columns=[
        "TaskID", "TargetRegion", "StartHour", "Duration_h", "GPU_Demand", "Task_Full_IT_Power_MW"
    ])
    records: list[dict[str, float | int | str]] = []
    lower_by_metric: dict[str, list[float]] = {metric: [] for metric in METRICS}
    reference_by_metric: dict[str, list[float]] = {metric: [] for metric in METRICS}
    representative_windows = _representative_windows(data, config)
    _progress(
        f"参数校准开始：共{len(representative_windows)}个代表窗口，"
        f"每个窗口{len(METRICS)}个连续下界和1个整数参考解。"
    )
    for window_index, tau in enumerate(representative_windows, start=1):
        _progress(
            f"参数校准进度：{window_index}/{len(representative_windows)}，"
            f"窗口起点={tau}。"
        )
        lookahead = min(config.lookahead_hours, OPERATION_END - (tau + config.decision_hours))
        problem = build_window_problem(
            data, config, tau, config.decision_hours, lookahead, empty_assignments,
            initial_soc.copy(), np.zeros(len(data.regions)), None, calibration=True,
        )
        lower_values: dict[str, float] = {}
        for metric_index, metric in enumerate(METRICS, start=1):
            _progress(
                f"参数校准窗口{tau}：连续下界{metric_index}/{len(METRICS)}，"
                f"指标={metric}。"
            )
            stage_name = "ContinuousLowerBound"
            status: int | str
            elapsed_seconds: float
            solve_message: str
            exact_ideal = False
            certified_safe = False
            bound_source = ""
            try:
                solution = _solve(
                    problem, problem.metric_vectors[metric], time_limit=60.0,
                    mip_gap=config.mip_relative_gap, relax=True,
                )
                status = solution.status
                elapsed_seconds = solution.elapsed_seconds
                solve_message = solution.message
                if solution.status == 0:
                    value = _metric_values(problem, solution.vector)[metric]
                    exact_ideal = True
                    certified_safe = True
                    bound_source = "OptimalContinuousRelaxation"
                else:
                    value, certified_safe, bound_source = _certified_box_lower_bound(problem, metric)
                    stage_name = "CertifiedSafeBound"
            except RuntimeError as exc:
                value, certified_safe, bound_source = _certified_box_lower_bound(problem, metric)
                stage_name = "CertifiedSafeBound" if certified_safe else "UncertifiedBound"
                status = "LP_NO_CERTIFIED_OPTIMUM"
                elapsed_seconds = float("nan")
                solve_message = str(exc)
            if not certified_safe or not np.isfinite(value):
                raise RuntimeError(
                    f"参数校准窗口{tau}指标{metric}没有得到可证明安全下界；"
                    f"BoundSource={bound_source}"
                )
            if not exact_ideal:
                _progress(
                    f"参数校准窗口{tau}指标={metric}未得到精确连续理想点，"
                    f"改用该指标自身的{bound_source}={value:.6g}。"
                )
            lower_values[metric] = value
            lower_by_metric[metric].append(value)
            records.append({
                "WindowStart": tau, "Stage": stage_name, "Metric": metric,
                "Value": value, "Status": status, "ElapsedSeconds": elapsed_seconds,
                "SolveMessage": solve_message,
                "BoundSource": bound_source,
                "CertifiedSafeBound": certified_safe,
                "ExactIdealPoint": exact_ideal,
                "FallbackSourceMetric": "",
            })
        _progress(f"参数校准窗口{tau}：开始求解整数参考解。")
        reference_objective = np.zeros(problem.lower.size, dtype=float)
        for metric in METRICS:
            denominator = max(abs(lower_values[metric]), 1.0)
            reference_objective += problem.metric_vectors[metric] / denominator
        reference_stage = "FeasibleReference"
        reference_status: int | str
        reference_elapsed_seconds: float
        reference_message: str
        try:
            reference = _solve(
                problem, reference_objective, time_limit=config.difficult_time_limit_seconds,
                mip_gap=config.mip_relative_gap, relax=False,
            )
            if reference.status != 0:
                reference_stage = "TimeLimitedFeasibleReference"
            reference_status = reference.status
            reference_elapsed_seconds = reference.elapsed_seconds
            reference_message = reference.message
        except RuntimeError as exc:
            heuristic = _heuristic_incumbent(
                data, problem, empty_assignments, initial_soc.copy(),
                np.zeros(len(data.regions)), None, {}, config.feasibility_tolerance,
            )
            if heuristic is None:
                raise RuntimeError(
                    f"参数校准窗口{tau}既没有求解器可行参考解，也没有启发式可行参考解：{exc}"
                ) from exc
            reference = WindowSolution(
                vector=heuristic,
                status=-2,
                message=f"Heuristic feasible reference after solver failure: {exc}",
                objective=float(np.dot(reference_objective, heuristic)),
                mip_gap=float("nan"),
                mip_node_count=float("nan"),
                elapsed_seconds=0.0,
            )
            reference_stage = "HeuristicFeasibleReference"
            reference_status = "HEURISTIC_FEASIBLE"
            reference_elapsed_seconds = 0.0
            reference_message = reference.message
        values = _metric_values(problem, reference.vector)
        for metric in METRICS:
            reference_by_metric[metric].append(values[metric])
            records.append({
                "WindowStart": tau, "Stage": reference_stage, "Metric": metric,
                "Value": values[metric], "Status": reference_status,
                "ElapsedSeconds": reference_elapsed_seconds,
                "SolveMessage": reference_message,
                "BoundSource": "FeasibleReferenceNotLowerBound",
                "CertifiedSafeBound": False,
                "ExactIdealPoint": False,
                "FallbackSourceMetric": "",
            })
    scaling: dict[str, tuple[float, float]] = {}
    for metric in METRICS:
        anchor = float(np.mean(lower_by_metric[metric]))
        reference = float(np.mean(reference_by_metric[metric]))
        epsilon = max(abs(anchor), abs(reference), 1.0) * 1e-6
        scale = max(reference - anchor, epsilon)
        scaling[metric] = (anchor, scale)
    _progress("参数校准完成。")
    return scaling, pd.DataFrame(records)


def _empty_assignments() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "TaskID", "TaskType", "ArrivalHour", "SourceRegion", "TargetRegion",
        "NetworkLatency_ms", "MaxLatency_ms", "StartHour", "FinishHour", "Duration_h",
        "GPU_Demand", "Task_Full_IT_Power_MW", "WaitHours", "IsMigrated", "DecisionWindowStart",
    ])


def _selected_assignments(data: InputData, problem: WindowProblem, vector: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, option in enumerate(problem.x_options):
        if vector[problem.indices["x"][index]] <= 0.5:
            continue
        task = data.task_lookup.loc[option.task_id]
        candidate = next(item for item in data.candidate_map[option.task_id] if item.region == option.region)
        start = float(option.start_hour)
        rows.append({
            "TaskID": option.task_id,
            "TaskType": str(task.TaskType),
            "ArrivalHour": int(task.ArrivalHour),
            "SourceRegion": str(task.SourceRegion),
            "TargetRegion": option.region,
            "NetworkLatency_ms": candidate.latency_ms,
            "MaxLatency_ms": float(task.MaxLatency_ms),
            "StartHour": start,
            "FinishHour": start + float(task.Duration_h),
            "Duration_h": float(task.Duration_h),
            "GPU_Demand": float(task.GPU_Demand),
            "Task_Full_IT_Power_MW": float(task.Task_Full_IT_Power_MW),
            "WaitHours": start - float(task.EarliestStartHour),
            "IsMigrated": int(option.region != str(task.SourceRegion)),
            "DecisionWindowStart": problem.tau,
        })
    return pd.DataFrame(rows, columns=_empty_assignments().columns)


def _forecast_rows(problem: WindowProblem, vector: np.ndarray, fixed_ai: np.ndarray) -> list[dict[str, object]]:
    h_count = problem.decision_end - problem.tau
    t_count = problem.plan_end - problem.tau
    forecast = fixed_ai.copy()
    for index, option in enumerate(problem.x_options):
        value = vector[problem.indices["x"][index]]
        if value <= EPS:
            continue
        for hour, overlap in option.overlaps:
            forecast[hour - problem.tau, option.region_index] += option.power_mw * overlap * value
    for index, option in enumerate(problem.u_options):
        value = vector[problem.indices["u"][index]]
        if value <= EPS:
            continue
        for hour, overlap in option.overlaps:
            forecast[hour - problem.tau, option.region_index] += option.power_mw * overlap * value
    rows: list[dict[str, object]] = []
    for local in range(h_count, t_count):
        for r, region in enumerate(problem.regions):
            rows.append({
                "WindowStart": problem.tau,
                "ForecastHour": problem.tau + local,
                "Region": region,
                "Predicted_AI_IT_Load_MW": forecast[local, r],
            })
    return rows


def _dispatch_rows(
    data: InputData,
    problem: WindowProblem,
    solution: WindowSolution,
    assignments_after: pd.DataFrame,
) -> list[dict[str, object]]:
    vector = solution.vector
    h_count = problem.decision_end - problem.tau
    _, actual_ai = _fixed_task_loads(assignments_after, problem.tau, problem.decision_end, data.region_index)
    frame = data.region_hour.loc[
        data.region_hour["Hour"].between(problem.tau, problem.decision_end - 1)
    ].copy()
    frame["Region"] = pd.Categorical(frame["Region"], categories=data.regions, ordered=True)
    frame = frame.sort_values(["Hour", "Region"], kind="stable").reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for local in range(h_count):
        for r, region in enumerate(data.regions):
            source = frame.iloc[local * len(data.regions) + r]
            ai = actual_ai[local, r]
            total_it = float(source.NonAI_IT_Load_MW) + ai
            facility = float(source.PUE) * total_it
            purchase = float(vector[problem.indices["grid_purchase"][local, r]])
            export = float(vector[problem.indices["export"][local, r]])
            rows.append({
                "Hour": problem.tau + local,
                "Region": region,
                "AI_IT_Load_MW": ai,
                "NonAI_IT_Load_MW": float(source.NonAI_IT_Load_MW),
                "Total_IT_Load_MW": total_it,
                "Facility_Load_MW": facility,
                "AvailableRenewable_MW": float(source.AvailableRenewable_MW),
                "RenewableDirectUse_MW": float(vector[problem.indices["renewable_direct"][local, r]]),
                "RenewableCharge_MW": float(vector[problem.indices["renewable_charge"][local, r]]),
                "GridCharge_MW": float(vector[problem.indices["grid_charge"][local, r]]),
                "ChargePower_MW": float(vector[problem.indices["renewable_charge"][local, r]] + vector[problem.indices["grid_charge"][local, r]]),
                "DischargePower_MW": float(vector[problem.indices["discharge"][local, r]]),
                "GridPurchase_MW": purchase,
                "GridSell_MW": export,
                "RenewableCurtailment_MW": float(vector[problem.indices["curtailment"][local, r]]),
                "NetGridImport_MW": purchase - export,
                "SOCStart_MWh": float(vector[problem.indices["soc"][local, r]]),
                "SOCEnd_MWh": float(vector[problem.indices["soc"][local + 1, r]]),
                "ElectricityPrice_CNY_per_MWh": float(source.ElectricityPrice_CNY_per_MWh),
                "SellPrice_CNY_per_MWh": float(source.SellPrice_CNY_per_MWh),
                "CarbonIntensity_tCO2_per_MWh": float(source.CarbonIntensity_tCO2_per_MWh),
                "CarbonEmission_tCO2": float(source.CarbonIntensity_tCO2_per_MWh) * purchase,
                "Available_GPU": float(source.Available_GPU),
                "Max_IT_Power_MW": float(source.Max_IT_Power_MW),
                "Max_Facility_Power_MW": float(source.Max_Facility_Power_MW),
                "PUE": float(source.PUE),
                "DecisionWindowStart": problem.tau,
            })
    return rows


def _final_metrics(data: InputData, assignments: pd.DataFrame, dispatch: pd.DataFrame) -> dict[str, float]:
    renewable_total = float(dispatch["AvailableRenewable_MW"].sum())
    values = {
        "Cost": float(np.sum(
            dispatch["ElectricityPrice_CNY_per_MWh"] * dispatch["GridPurchase_MW"]
            - dispatch["SellPrice_CNY_per_MWh"] * dispatch["GridSell_MW"]
        )),
        "Carbon": float(np.sum(
            dispatch["CarbonIntensity_tCO2_per_MWh"] * dispatch["GridPurchase_MW"]
        )),
        "Latency": float(np.mean(assignments["NetworkLatency_ms"] / assignments["MaxLatency_ms"])),
        "RenewableUnusedRate": (
            float(dispatch["RenewableCurtailment_MW"].sum() / renewable_total)
            if renewable_total > EPS else 0.0
        ),
        "Peak": float(dispatch.groupby("Region")["NetGridImport_MW"].max().clip(lower=0.0).sum()),
    }
    joined = assignments[["TaskID", "StartHour"]].merge(
        data.tasks[["TaskID", "EarliestStartHour", "LatestFinishHour", "Duration_h", "TaskType"]],
        on="TaskID", how="left", validate="one_to_one",
    )
    joined["Slack"] = joined["LatestFinishHour"] - joined["Duration_h"] - joined["EarliestStartHour"]
    flexible = joined.loc[joined["Slack"] > EPS]
    values["Delay"] = float(np.mean(
        (flexible["StartHour"] - flexible["EarliestStartHour"]) / flexible["Slack"]
    )) if not flexible.empty else 0.0
    return values


def _write_csv(frame: pd.DataFrame, filename: str) -> None:
    _atomic_write_csv(frame, TABLES_DIR / filename)


def _simple_validation(
    data: InputData,
    assignments: pd.DataFrame,
    dispatch: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:
    """对最终实际轨迹做最小硬约束自检，不执行额外模型或敏感性分析。"""

    records: list[dict[str, object]] = []

    def add(check: str, violation: float, limit: float = tolerance) -> None:
        value = float(max(violation, 0.0))
        records.append({
            "Check": check,
            "MaxViolation": value,
            "Tolerance": float(limit),
            "Passed": bool(value <= limit),
        })

    counts = assignments.groupby("TaskID").size().reindex(data.tasks["TaskID"], fill_value=0)
    add("TaskUniqueExecution", float(np.max(np.abs(counts.to_numpy(dtype=float) - 1.0))), 0.0)
    joined = assignments.merge(
        data.tasks[[
            "TaskID", "TaskType", "ArrivalHour", "EarliestStartHour",
            "LatestFinishHour", "Duration_h",
        ]],
        on="TaskID", how="left", validate="one_to_one", suffixes=("", "_Input"),
    )
    earliest = np.maximum(joined["ArrivalHour_Input"], joined["EarliestStartHour"])
    add("TaskEarliestStart", float(np.maximum(earliest - joined["StartHour"], 0.0).max()))
    recomputed_finish = joined["StartHour"] + joined["Duration_h_Input"]
    add("TaskLatestFinish", float(np.maximum(recomputed_finish - joined["LatestFinishHour"], 0.0).max()))
    real_time = joined["TaskType_Input"].eq("RealTimeInference")
    add(
        "RealTimeImmediateStart",
        float(np.abs(joined.loc[real_time, "StartHour"] - joined.loc[real_time, "ArrivalHour_Input"]).max())
        if real_time.any() else 0.0,
    )
    add("NetworkLatency", float(np.maximum(
        assignments["NetworkLatency_ms"] - assignments["MaxLatency_ms"], 0.0
    ).max()))
    valid_pairs = set(zip(
        data.candidates["TaskID"].astype(str),
        data.candidates["TargetRegion"].astype(str),
    ))
    invalid_pair_count = sum(
        (str(row.TaskID), str(row.TargetRegion)) not in valid_pairs
        for row in assignments.itertuples(index=False)
    )
    add("TaskCandidateRegion", float(invalid_pair_count > 0), 0.0)

    gpu, ai = _fixed_task_loads(assignments, 0, OPERATION_END, data.region_index)
    expected = pd.DataFrame({
        "Hour": np.repeat(np.arange(OPERATION_END), len(data.regions)),
        "Region": np.tile(np.asarray(data.regions), OPERATION_END),
        "Recomputed_GPU": gpu.ravel(),
        "Recomputed_AI_IT_Load_MW": ai.ravel(),
    })
    checked = dispatch.merge(expected, on=["Hour", "Region"], how="left", validate="one_to_one")
    add("AIITLoadRecompute", float(np.abs(
        checked["AI_IT_Load_MW"] - checked["Recomputed_AI_IT_Load_MW"]
    ).max()))
    add("GPUCapacity", float(np.maximum(
        checked["Recomputed_GPU"] - checked["Available_GPU"], 0.0
    ).max()))
    add("ITCapacity", float(np.maximum(
        dispatch["Total_IT_Load_MW"] - dispatch["Max_IT_Power_MW"], 0.0
    ).max()))
    add("FacilityCapacity", float(np.maximum(
        dispatch["Facility_Load_MW"] - dispatch["Max_Facility_Power_MW"], 0.0
    ).max()))

    renewable_residual = (
        dispatch["AvailableRenewable_MW"] - dispatch["RenewableDirectUse_MW"]
        - dispatch["RenewableCharge_MW"] - dispatch["GridSell_MW"]
        - dispatch["RenewableCurtailment_MW"]
    )
    load_residual = (
        dispatch["GridPurchase_MW"] + dispatch["RenewableDirectUse_MW"]
        + dispatch["DischargePower_MW"] - dispatch["Facility_Load_MW"]
        - dispatch["GridCharge_MW"]
    )
    add("RenewableBalance", float(np.abs(renewable_residual).max()))
    add("FacilityLoadBalance", float(np.abs(load_residual).max()))
    storage = data.storage.set_index("Region").loc[list(data.regions)]
    eta_c = dispatch["Region"].map(storage["ChargeEfficiency"]).to_numpy(dtype=float)
    eta_d = dispatch["Region"].map(storage["DischargeEfficiency"]).to_numpy(dtype=float)
    soc_residual = (
        dispatch["SOCEnd_MWh"] - dispatch["SOCStart_MWh"]
        - eta_c * (dispatch["RenewableCharge_MW"] + dispatch["GridCharge_MW"])
        + dispatch["DischargePower_MW"] / eta_d
    )
    add("SOCRecurrence", float(np.abs(soc_residual).max()))
    ordered_soc = dispatch.sort_values(["Region", "Hour"], kind="stable")
    continuity_parts = []
    for _, region_rows in ordered_soc.groupby("Region", sort=False, observed=True):
        continuity_parts.append(
            region_rows["SOCStart_MWh"].to_numpy(dtype=float)[1:]
            - region_rows["SOCEnd_MWh"].to_numpy(dtype=float)[:-1]
        )
    continuity = np.concatenate(continuity_parts) if continuity_parts else np.array([0.0])
    add("SOCContinuity", float(np.abs(continuity).max()))
    min_soc = dispatch["Region"].map(storage["MinSOC_MWh"]).to_numpy(dtype=float)
    max_soc = dispatch["Region"].map(storage["StorageCapacity_MWh"]).to_numpy(dtype=float)
    soc_values = np.concatenate([
        dispatch["SOCStart_MWh"].to_numpy(dtype=float),
        dispatch["SOCEnd_MWh"].to_numpy(dtype=float),
    ])
    soc_min_values = np.concatenate([min_soc, min_soc])
    soc_max_values = np.concatenate([max_soc, max_soc])
    add("SOCBounds", float(max(
        np.maximum(soc_min_values - soc_values, 0.0).max(),
        np.maximum(soc_values - soc_max_values, 0.0).max(),
    )))
    initial_rows = dispatch.loc[dispatch["Hour"].eq(0)].set_index("Region").loc[list(data.regions)]
    add("InitialSOC", float(np.abs(
        initial_rows["SOCStart_MWh"].to_numpy(dtype=float)
        - storage["InitialSOC_MWh"].to_numpy(dtype=float)
    ).max()))
    max_charge = dispatch["Region"].map(storage["MaxChargePower_MW"]).to_numpy(dtype=float)
    max_discharge = dispatch["Region"].map(storage["MaxDischargePower_MW"]).to_numpy(dtype=float)
    max_import = dispatch["Region"].map(storage["MaxGridImport_MW"]).to_numpy(dtype=float)
    max_export = np.minimum(
        dispatch["Region"].map(storage["SellLimit_MW"]).to_numpy(dtype=float),
        dispatch["Region"].map(storage["MaxGridExport_MW"]).to_numpy(dtype=float),
    )
    add("StorageChargePower", float(np.maximum(
        dispatch["ChargePower_MW"].to_numpy(dtype=float) - max_charge, 0.0
    ).max()))
    add("StorageDischargePower", float(np.maximum(
        dispatch["DischargePower_MW"].to_numpy(dtype=float) - max_discharge, 0.0
    ).max()))
    add("GridImportBoundary", float(np.maximum(
        dispatch["GridPurchase_MW"].to_numpy(dtype=float) - max_import, 0.0
    ).max()))
    add("GridExportBoundary", float(np.maximum(
        dispatch["GridSell_MW"].to_numpy(dtype=float) - max_export, 0.0
    ).max()))
    add("ChargeDischargeExclusion", float(np.minimum(
        dispatch["ChargePower_MW"], dispatch["DischargePower_MW"]
    ).max()))
    terminal = dispatch.loc[dispatch["Hour"].eq(OPERATION_END - 1)].set_index("Region").loc[list(data.regions)]
    add("TerminalSOC", float(np.maximum(
        storage["InitialSOC_MWh"].to_numpy(dtype=float) - terminal["SOCEnd_MWh"].to_numpy(dtype=float), 0.0
    ).max()))
    return pd.DataFrame(records)


def _cache_signature(config: ModelConfig) -> str:
    digest = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode("utf-8"))
    for path in (
        Path(__file__), QUESTION_DIR / "preprocess.py", Q4_DIR / "q4_region_hour_input.csv",
        SHARED_DIR / "tasks_clean.csv", SHARED_DIR / "task_candidate_regions.csv",
        SHARED_DIR / "storage_params.csv",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _scaling_frame(scaling: Mapping[str, tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame([
        {"Metric": metric, "Anchor": scaling[metric][0], "Scale": scaling[metric][1]}
        for metric in METRICS
    ])


def _next_tau_after(tau: int) -> int:
    return tau + 24 if tau < MAIN_END else OPERATION_END


def _validate_resume_state(
    data: InputData,
    config: ModelConfig,
    assignments: pd.DataFrame,
    dispatch: pd.DataFrame,
    solver: pd.DataFrame,
    resume_from: int,
    stored_soc: np.ndarray | None,
    stored_peak: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """严格复核断点确实对应0至resume_from-1的完整实际轨迹。"""

    if resume_from < 0 or resume_from > OPERATION_END:
        raise ValueError(f"恢复时刻{resume_from}不在0--{OPERATION_END}范围内")
    valid_starts = set(range(0, MAIN_END, config.decision_hours)) | {MAIN_END, OPERATION_END}
    if resume_from not in valid_starts:
        raise ValueError(f"恢复时刻{resume_from}不是合法滚动窗口边界")

    assignments = assignments.copy()
    dispatch = dispatch.copy()
    solver = solver.copy()
    if assignments["TaskID"].astype(str).duplicated().any():
        duplicates = assignments.loc[
            assignments["TaskID"].astype(str).duplicated(keep=False), "TaskID"
        ].astype(str).unique()[:10]
        raise ValueError(f"恢复记录存在重复执行任务：{duplicates.tolist()}")
    known_ids = set(data.tasks["TaskID"].astype(str))
    assigned_ids = set(assignments["TaskID"].astype(str))
    unknown = sorted(assigned_ids - known_ids)
    if unknown:
        raise ValueError(f"恢复记录存在原始任务表之外的TaskID，示例：{unknown[:10]}")
    if len(assigned_ids) + len(known_ids - assigned_ids) != len(known_ids):
        raise ValueError("恢复记录中的已执行与未执行任务数量不能覆盖原始任务全集")
    if not assignments.empty:
        if (pd.to_numeric(assignments["DecisionWindowStart"]) >= resume_from).any():
            raise ValueError("恢复记录包含恢复时刻之后窗口写入的任务")
        if (pd.to_numeric(assignments["StartHour"]) >= resume_from - EPS).any():
            raise ValueError("恢复记录包含尚未进入实际H区的任务开工结果")
        joined = assignments.merge(
            data.tasks[[
                "TaskID", "TaskType", "ArrivalHour", "EarliestStartHour",
                "LatestFinishHour", "Duration_h",
            ]],
            on="TaskID", how="left", validate="one_to_one", suffixes=("", "_Input"),
        )
        earliest = np.maximum(joined["ArrivalHour_Input"], joined["EarliestStartHour"])
        if float(np.maximum(earliest - joined["StartHour"], 0.0).max()) > config.feasibility_tolerance:
            raise ValueError("恢复任务记录存在早于到达/最早开工时刻的任务")
        finish = joined["StartHour"] + joined["Duration_h_Input"]
        if float(np.maximum(finish - joined["LatestFinishHour"], 0.0).max()) > config.feasibility_tolerance:
            raise ValueError("恢复任务记录存在超过完成期限的任务")
        real_time = joined["TaskType_Input"].eq("RealTimeInference")
        if real_time.any() and float(np.abs(
            joined.loc[real_time, "StartHour"] - joined.loc[real_time, "ArrivalHour_Input"]
        ).max()) > config.feasibility_tolerance:
            raise ValueError("恢复任务记录中的实时任务未在到达时刻启动")
        valid_pairs = set(zip(
            data.candidates["TaskID"].astype(str), data.candidates["TargetRegion"].astype(str)
        ))
        bad_pairs = [
            (str(row.TaskID), str(row.TargetRegion))
            for row in assignments.itertuples(index=False)
            if (str(row.TaskID), str(row.TargetRegion)) not in valid_pairs
        ]
        if bad_pairs:
            raise ValueError(f"恢复任务记录存在非法任务—区域组合，示例：{bad_pairs[:10]}")

    expected_windows = list(range(0, min(resume_from, MAIN_END), config.decision_hours))
    if resume_from > MAIN_END:
        expected_windows.append(MAIN_END)
    actual_windows = sorted(pd.to_numeric(solver.get("WindowStart", pd.Series(dtype=float))).astype(int).tolist())
    if actual_windows != expected_windows:
        raise ValueError(
            f"窗口求解记录不连续：应为{len(expected_windows)}个窗口，实际为{len(actual_windows)}个"
        )

    if resume_from == 0:
        if not dispatch.empty:
            raise ValueError("0小时恢复状态不应包含逐时实际调度")
        storage = data.storage.set_index("Region").loc[list(data.regions)]
        return (
            storage["InitialSOC_MWh"].to_numpy(dtype=float),
            np.zeros(len(data.regions), dtype=float),
        )

    required_dispatch = {
        "Hour", "Region", "SOCStart_MWh", "SOCEnd_MWh", "RenewableCharge_MW",
        "GridCharge_MW", "DischargePower_MW", "NetGridImport_MW",
        "AvailableRenewable_MW", "RenewableDirectUse_MW", "GridSell_MW",
        "RenewableCurtailment_MW", "GridPurchase_MW", "Facility_Load_MW",
    }
    missing_columns = sorted(required_dispatch - set(dispatch.columns))
    if missing_columns:
        raise ValueError(f"逐时调度恢复表缺少字段：{missing_columns}")
    if dispatch.duplicated(["Hour", "Region"]).any():
        raise ValueError("逐时调度恢复表存在重复的Hour—Region记录")
    expected_pairs = pd.MultiIndex.from_product(
        [range(resume_from), data.regions], names=["Hour", "Region"]
    )
    actual_pairs = pd.MultiIndex.from_frame(dispatch[["Hour", "Region"]])
    missing_pairs = expected_pairs.difference(actual_pairs)
    extra_pairs = actual_pairs.difference(expected_pairs)
    if len(missing_pairs) or len(extra_pairs):
        raise ValueError(
            f"逐时调度不能完整覆盖0--{resume_from - 1}小时："
            f"缺少{len(missing_pairs)}行，多出{len(extra_pairs)}行"
        )
    ordered = dispatch.copy()
    ordered["Region"] = pd.Categorical(ordered["Region"], categories=data.regions, ordered=True)
    ordered = ordered.sort_values(["Hour", "Region"], kind="stable").reset_index(drop=True)
    storage = data.storage.set_index("Region").loc[list(data.regions)]
    eta_c = ordered["Region"].astype(str).map(storage["ChargeEfficiency"]).to_numpy(dtype=float)
    eta_d = ordered["Region"].astype(str).map(storage["DischargeEfficiency"]).to_numpy(dtype=float)
    recurrence = (
        ordered["SOCEnd_MWh"] - ordered["SOCStart_MWh"]
        - eta_c * (ordered["RenewableCharge_MW"] + ordered["GridCharge_MW"])
        + ordered["DischargePower_MW"] / eta_d
    )
    if float(np.abs(recurrence).max()) > config.feasibility_tolerance:
        raise ValueError("逐时调度的SOC递推不连续")
    for region in data.regions:
        region_rows = ordered.loc[ordered["Region"].astype(str).eq(region)]
        initial_soc = float(storage.loc[region, "InitialSOC_MWh"])
        if abs(float(region_rows.iloc[0]["SOCStart_MWh"]) - initial_soc) > config.feasibility_tolerance:
            raise ValueError(f"区域{region}的0小时SOC与初始SOC不一致")
        cross = (
            region_rows["SOCStart_MWh"].to_numpy(dtype=float)[1:]
            - region_rows["SOCEnd_MWh"].to_numpy(dtype=float)[:-1]
        )
        if len(cross) and float(np.abs(cross).max()) > config.feasibility_tolerance:
            raise ValueError(f"区域{region}相邻小时SOC不连续")
    renewable_balance = (
        ordered["AvailableRenewable_MW"] - ordered["RenewableDirectUse_MW"]
        - ordered["RenewableCharge_MW"] - ordered["GridSell_MW"]
        - ordered["RenewableCurtailment_MW"]
    )
    load_balance = (
        ordered["GridPurchase_MW"] + ordered["RenewableDirectUse_MW"]
        + ordered["DischargePower_MW"] - ordered["Facility_Load_MW"]
        - ordered["GridCharge_MW"]
    )
    if max(float(np.abs(renewable_balance).max()), float(np.abs(load_balance).max())) > config.feasibility_tolerance:
        raise ValueError("逐时调度恢复表的能量平衡不成立")
    final_hour = ordered.loc[ordered["Hour"].eq(resume_from - 1)].set_index("Region").loc[list(data.regions)]
    recomputed_soc = final_hour["SOCEnd_MWh"].to_numpy(dtype=float)
    recomputed_peak = (
        ordered.groupby(ordered["Region"].astype(str), observed=True)["NetGridImport_MW"]
        .max().reindex(data.regions).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    )
    if stored_soc is not None and not np.allclose(
        np.asarray(stored_soc, dtype=float), recomputed_soc,
        rtol=0.0, atol=config.feasibility_tolerance,
    ):
        raise ValueError("断点保存的1608小时SOC与逐时调度重新计算值不一致")
    if stored_peak is not None and not np.allclose(
        np.asarray(stored_peak, dtype=float), recomputed_peak,
        rtol=0.0, atol=config.feasibility_tolerance,
    ):
        raise ValueError("断点保存的历史峰值与逐时净购电重新计算值不一致")
    return recomputed_soc, recomputed_peak


def _load_progress_state(
    data: InputData,
    config: ModelConfig,
    signature: str,
    resume_from: int,
    state_file: Path | None,
) -> dict[str, object]:
    checkpoint = state_file.resolve() if state_file is not None else CHECKPOINT_PATH
    if checkpoint.is_file():
        try:
            with checkpoint.open("rb") as handle:
                state = pickle.load(handle)
        except (OSError, pickle.PickleError, EOFError) as exc:
            raise RuntimeError(f"恢复状态文件损坏或不可读取：{checkpoint}") from exc
        if not isinstance(state, dict) or state.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError(f"恢复状态文件版本不受支持：{checkpoint}")
        if state.get("signature") != signature:
            raise ValueError("恢复状态的数据、参数或model.py签名与当前工程不一致")
        if int(state.get("next_tau", -1)) != resume_from:
            raise ValueError(
                f"恢复状态的next_tau={state.get('next_tau')}，与请求的{resume_from}不一致"
            )
    else:
        required = (
            PROGRESS_ASSIGNMENTS_PATH, PROGRESS_DISPATCH_PATH,
            PROGRESS_SOLVER_PATH, PROGRESS_SCALING_PATH,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "不能从日志或目标值伪造1608小时状态；缺少可恢复文件：\n- "
                + "\n- ".join(missing)
            )
        assignments = pd.read_csv(PROGRESS_ASSIGNMENTS_PATH, encoding="utf-8-sig")
        dispatch = pd.read_csv(PROGRESS_DISPATCH_PATH, encoding="utf-8-sig")
        solver = pd.read_csv(PROGRESS_SOLVER_PATH, encoding="utf-8-sig")
        scaling_frame = pd.read_csv(PROGRESS_SCALING_PATH, encoding="utf-8-sig")
        scaling = {
            str(row.Metric): (float(row.Anchor), float(row.Scale))
            for row in scaling_frame.itertuples(index=False)
        }
        if set(scaling) != set(METRICS):
            raise ValueError("进度定标表没有完整包含六个目标")
        forecast = (
            pd.read_csv(PROGRESS_FORECAST_PATH, encoding="utf-8-sig")
            if PROGRESS_FORECAST_PATH.is_file() else pd.DataFrame()
        )
        calibration = (
            pd.read_csv(PROGRESS_CALIBRATION_PATH, encoding="utf-8-sig")
            if PROGRESS_CALIBRATION_PATH.is_file() else pd.DataFrame()
        )
        state = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "signature": signature,
            "next_tau": resume_from,
            "assignments": assignments,
            "dispatch": dispatch,
            "solver": solver,
            "forecast": forecast,
            "scaling": scaling,
            "calibration_records": calibration,
            "current_soc": None,
            "historical_peak": None,
            "warm_hints": {},
        }
        _progress("未找到pkl断点，已从完整进度CSV尝试重建状态；将重新计算SOC和历史峰值。")
    assignments = pd.DataFrame(state.get("assignments", _empty_assignments()))
    dispatch = pd.DataFrame(state.get("dispatch", pd.DataFrame()))
    solver = pd.DataFrame(state.get("solver", pd.DataFrame()))
    current_soc, historical_peak = _validate_resume_state(
        data, config, assignments, dispatch, solver, resume_from,
        state.get("current_soc"), state.get("historical_peak"),
    )
    state["assignments"] = assignments
    state["dispatch"] = dispatch
    state["solver"] = solver
    state["current_soc"] = np.array(
        current_soc, dtype=np.float64, copy=True, order="C"
    )
    state["historical_peak"] = np.array(
        historical_peak, dtype=np.float64, copy=True, order="C"
    )
    state["next_tau"] = resume_from
    return state


def _save_progress_checkpoint(
    *,
    data: InputData,
    config: ModelConfig,
    signature: str,
    last_completed_tau: int | None,
    next_tau: int,
    assignments: pd.DataFrame,
    dispatch: pd.DataFrame,
    solver: pd.DataFrame,
    forecast: pd.DataFrame,
    scaling: Mapping[str, tuple[float, float]],
    calibration_records: pd.DataFrame,
    current_soc: np.ndarray,
    historical_peak: np.ndarray,
    warm_hints: Mapping[str, tuple[str, float]],
    mode: Mapping[str, object],
) -> None:
    assigned_ids = set(assignments["TaskID"].astype(str)) if not assignments.empty else set()
    active = assignments.loc[
        (pd.to_numeric(assignments["StartHour"]) < next_tau)
        & (pd.to_numeric(assignments["FinishHour"]) > next_tau)
    ].copy() if not assignments.empty else assignments.copy()
    if not active.empty:
        active["RemainingRunTime_h"] = np.maximum(
            pd.to_numeric(active["FinishHour"]).to_numpy(dtype=float) - next_tau,
            0.0,
        )
    unstarted_ids = tuple(
        task_id for task_id in data.tasks["TaskID"].astype(str) if task_id not in assigned_ids
    )
    unstarted = data.tasks.loc[data.tasks["TaskID"].astype(str).isin(unstarted_ids)].copy()
    if not unstarted.empty:
        unstarted["RemainingEarliestStartHour"] = np.maximum.reduce([
            np.full(len(unstarted), next_tau, dtype=float),
            unstarted["ArrivalHour"].to_numpy(dtype=float),
            unstarted["EarliestStartHour"].to_numpy(dtype=float),
        ])
        unstarted["RemainingLatestStartHour"] = np.floor(
            unstarted["LatestFinishHour"].to_numpy(dtype=float)
            - unstarted["Duration_h"].to_numpy(dtype=float) + EPS
        ).astype(int)
        candidate_counts = data.candidates.groupby("TaskID").size()
        unstarted["RemainingCandidateRegionCount"] = (
            unstarted["TaskID"].map(candidate_counts).fillna(0).astype(int)
        )
    solver_to_write = solver.copy()
    if not solver_to_write.empty:
        solver_to_write["ModelSignature"] = signature
    _atomic_write_csv(assignments, PROGRESS_ASSIGNMENTS_PATH)
    _atomic_write_csv(dispatch, PROGRESS_DISPATCH_PATH)
    _atomic_write_csv(solver_to_write, PROGRESS_SOLVER_PATH)
    _atomic_write_csv(forecast, PROGRESS_FORECAST_PATH)
    _atomic_write_csv(_scaling_frame(scaling), PROGRESS_SCALING_PATH)
    _atomic_write_csv(calibration_records, PROGRESS_CALIBRATION_PATH)
    state = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "signature": signature,
        "last_completed_tau": last_completed_tau,
        "next_tau": next_tau,
        "active_tasks": active,
        "unstarted_task_ids": unstarted_ids,
        "unstarted_tasks": unstarted,
        "assignments": assignments,
        "dispatch": dispatch,
        "solver": solver_to_write,
        "forecast": forecast,
        "scaling": dict(scaling),
        "calibration_records": calibration_records,
        "current_soc": np.array(
            current_soc, dtype=np.float64, copy=True, order="C"
        ),
        "historical_peak": np.array(
            historical_peak, dtype=np.float64, copy=True, order="C"
        ),
        "warm_hints": dict(warm_hints),
        "mode": dict(mode),
        "config": asdict(config),
    }
    _atomic_write_pickle(CHECKPOINT_PATH, state)


def _record_failed_window(
    *, tau: int, signature: str, reason: str, mode: Mapping[str, object]
) -> None:
    _atomic_write_json(FAILED_WINDOW_PATH, {
        "WindowStart": tau,
        "Reason": reason,
        "Mode": dict(mode),
        "ModelSignature": signature,
        "CheckpointPreserved": str(CHECKPOINT_PATH),
    })


def _can_reuse(signature: str) -> bool:
    state_path = TABLES_DIR / ".q4_cache.json"
    required = (
        "q4_task_assignments.csv", "q4_region_hour_dispatch.csv", "q4_objective_summary.csv",
        "q4_window_solver.csv", "q4_scaling.csv", "q4_model_configuration.csv",
        "q4_simple_validation.csv",
    )
    if not state_path.is_file() or not all((TABLES_DIR / filename).is_file() for filename in required):
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return state.get("signature") == signature and state.get("complete") is True


def _balanced_objective(problem: WindowProblem, config: ModelConfig) -> np.ndarray:
    objective = np.zeros(problem.lower.size, dtype=float)
    objective[problem.indices["max_deviation"]] = 1.0
    objective[problem.indices["deviation"]] = config.tie_break_weight
    return objective


def _solve_with_time_extension(
    problem: WindowProblem,
    objective: np.ndarray,
    config: ModelConfig,
    incumbent: np.ndarray | None,
    initial_time_limit: float,
) -> tuple[WindowSolution, int, float]:
    """正常窗口先求120秒；无可行解时保持同一模型延长到240秒。"""

    limits = [float(initial_time_limit)]
    if initial_time_limit < config.difficult_time_limit_seconds:
        limits.append(float(config.difficult_time_limit_seconds))
    unique_limits = list(dict.fromkeys(limits))
    errors: list[str] = []
    for attempt, limit in enumerate(unique_limits, start=1):
        try:
            solution = _solve(
                problem,
                objective,
                time_limit=limit,
                mip_gap=config.mip_relative_gap,
                relax=False,
                incumbent=incumbent,
            )
            return solution, attempt, limit
        except RuntimeError as exc:
            errors.append(f"{limit:.0f}s: {exc}")
            _progress(
                f"窗口{problem.tau}第{attempt}/{len(unique_limits)}次求解失败，"
                f"时间上限={limit:.0f}s：{exc}。"
            )
    raise RuntimeError("；".join(errors))


def _future_feasibility_check(
    data: InputData,
    assignments_after: pd.DataFrame,
    next_tau: int,
    tolerance: float,
) -> dict[str, int]:
    """确认H区决策后，每个剩余任务仍至少保留一个物理合法的未来方案。"""

    assigned_ids = set(assignments_after["TaskID"].astype(str)) if not assignments_after.empty else set()
    remaining = data.tasks.loc[~data.tasks["TaskID"].isin(assigned_ids)]
    if next_tau >= OPERATION_END:
        if not remaining.empty:
            raise RuntimeError(f"终端时刻仍有{len(remaining)}个任务未安排")
        return {"RemainingTaskCount": 0, "IndividuallyFeasibleTaskCount": 0}

    fixed_gpu, fixed_ai = _fixed_task_loads(
        assignments_after, next_tau, OPERATION_END, data.region_index
    )
    frame = data.region_hour.loc[
        data.region_hour["Hour"].between(next_tau, OPERATION_END - 1)
    ].copy()
    frame["Region"] = pd.Categorical(frame["Region"], categories=data.regions, ordered=True)
    frame = frame.sort_values(["Hour", "Region"], kind="stable")
    t_count = OPERATION_END - next_tau
    r_count = len(data.regions)
    available_gpu = frame["Available_GPU"].to_numpy(dtype=float).reshape(t_count, r_count)
    non_ai = frame["NonAI_IT_Load_MW"].to_numpy(dtype=float).reshape(t_count, r_count)
    max_it = frame["Max_IT_Power_MW"].to_numpy(dtype=float).reshape(t_count, r_count)
    pue = frame["PUE"].to_numpy(dtype=float).reshape(t_count, r_count)
    max_facility = frame["Max_Facility_Power_MW"].to_numpy(dtype=float).reshape(t_count, r_count)
    ai_capacity = np.minimum(max_it - non_ai, max_facility / pue - non_ai)
    fixed_gpu_violation = fixed_gpu - available_gpu
    fixed_ai_violation = fixed_ai - ai_capacity
    if np.any(fixed_gpu_violation > tolerance):
        local, r = np.argwhere(fixed_gpu_violation > tolerance)[0]
        raise RuntimeError(
            "H区方案产生的跨窗口固定GPU负荷超过未来容量："
            f"Hour={next_tau + int(local)}，Region={data.regions[int(r)]}，"
            f"Load={fixed_gpu[local, r]:.12g}，Capacity={available_gpu[local, r]:.12g}，"
            f"Violation={fixed_gpu_violation[local, r]:.12g}"
        )
    if np.any(fixed_ai_violation > tolerance):
        local, r = np.argwhere(fixed_ai_violation > tolerance)[0]
        raise RuntimeError(
            "H区方案产生的跨窗口固定IT/设施负荷超过未来容量："
            f"Hour={next_tau + int(local)}，Region={data.regions[int(r)]}，"
            f"AILoad={fixed_ai[local, r]:.12g}，AICapacity={ai_capacity[local, r]:.12g}，"
            f"Violation={fixed_ai_violation[local, r]:.12g}"
        )
    storage = data.storage.set_index("Region").loc[list(data.regions)]
    max_grid = storage["MaxGridImport_MW"].to_numpy(dtype=float)
    max_discharge = storage["MaxDischargePower_MW"].to_numpy(dtype=float)
    renewable = frame["AvailableRenewable_MW"].to_numpy(dtype=float).reshape(t_count, r_count)
    fixed_facility = pue * (non_ai + fixed_ai)
    necessary_supply_shortfall = (
        fixed_facility - renewable - max_grid[None, :] - max_discharge[None, :]
    )
    if np.any(necessary_supply_shortfall > tolerance):
        local, r = np.argwhere(necessary_supply_shortfall > tolerance)[0]
        raise RuntimeError(
            "H区方案产生的跨窗口固定设施负荷超过新能源、最大购电和最大放电之和："
            f"Hour={next_tau + int(local)}，Region={data.regions[int(r)]}，"
            f"Violation={necessary_supply_shortfall[local, r]:.12g}MW"
        )
    infeasible: list[str] = []
    for task in remaining.itertuples(index=False):
        starts = _feasible_starts(task, next_tau, OPERATION_END)
        if not starts:
            infeasible.append(str(task.TaskID))
            if len(infeasible) >= 10:
                break
            continue
        task_feasible = False
        for candidate in data.candidate_map[str(task.TaskID)]:
            r = candidate.region_index
            for start in starts:
                fits = True
                for hour, overlap in _hour_overlaps(
                    float(start), float(task.Duration_h), next_tau, OPERATION_END
                ):
                    local = hour - next_tau
                    if (
                        fixed_gpu[local, r] + float(task.GPU_Demand) * overlap
                        > available_gpu[local, r] + tolerance
                        or fixed_ai[local, r] + float(task.Task_Full_IT_Power_MW) * overlap
                        > ai_capacity[local, r] + tolerance
                    ):
                        fits = False
                        break
                if fits:
                    task_feasible = True
                    break
            if task_feasible:
                break
        if not task_feasible:
            infeasible.append(str(task.TaskID))
            if len(infeasible) >= 10:
                break
    if infeasible:
        raise RuntimeError(
            "当前H区方案会使剩余任务失去合法区域、开工时刻或完成期限，示例："
            f"{infeasible}"
        )
    return {
        "RemainingTaskCount": int(len(remaining)),
        "IndividuallyFeasibleTaskCount": int(len(remaining)),
    }


def _solve_h_only_recovery(
    data: InputData,
    config: ModelConfig,
    tau: int,
    decision_hours: int,
    assignments: pd.DataFrame,
    current_soc: np.ndarray,
    historical_peak: np.ndarray,
    scaling: Mapping[str, tuple[float, float]],
    warm_hints: Mapping[str, tuple[str, float]],
    *,
    feasibility_only: bool,
) -> tuple[WindowProblem, WindowSolution, np.ndarray | None, str]:
    problem = build_window_problem(
        data, config, tau, decision_hours, 0, assignments, current_soc,
        historical_peak, None if feasibility_only else scaling,
        block_hours=config.block_hours,
        protect_h_tail_capacity=True,
    )
    incumbent = _heuristic_incumbent(
        data, problem, assignments, current_soc, historical_peak,
        None if feasibility_only else scaling, warm_hints, config.feasibility_tolerance,
    )
    objective = (
        np.zeros(problem.lower.size, dtype=float)
        if feasibility_only else _balanced_objective(problem, config)
    )
    mode = "HOnlyFeasibility" if feasibility_only else "HOnlyBalanced"
    _progress(
        f"窗口{tau}进入{mode}：H={decision_hours}h，K=0h，"
        f"变量={problem.lower.size}，约束={problem.matrix.shape[0]}。"
    )
    solution = _solve(
        problem, objective,
        time_limit=config.difficult_time_limit_seconds,
        mip_gap=config.mip_relative_gap,
        relax=False,
        incumbent=incumbent,
    )
    return problem, solution, incumbent, mode


def _legacy_run_with_2h_fallback_disabled(
    *, force: bool = False, config: ModelConfig | None = None
) -> None:
    raise RuntimeError("旧版4小时失败后自动改2小时的运行入口已永久禁用")
    config = config or ModelConfig()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    _progress(f"Q4模型开始运行：force={force}。")
    signature = _cache_signature(config)
    if not force and _can_reuse(signature):
        _progress("检测到完整且匹配的缓存结果，跳过求解。")
        return
    _progress("开始读取模型输入。")
    data = load_data()
    _progress(
        f"输入读取完成：任务={len(data.tasks)}，区域={len(data.regions)}，"
        f"逐时区域记录={len(data.region_hour)}。"
    )
    scaling, calibration_records = _calibrate(data, config)
    _progress("开始执行滚动窗口优化。")
    scaling_rows = pd.DataFrame([
        {"Metric": metric, "Anchor": scaling[metric][0], "Scale": scaling[metric][1]}
        for metric in METRICS
    ])
    assignments = _empty_assignments()
    storage = data.storage.set_index("Region").loc[list(data.regions)]
    current_soc = storage["InitialSOC_MWh"].to_numpy(dtype=float)
    historical_peak = np.zeros(len(data.regions), dtype=float)
    dispatch_rows: list[dict[str, object]] = []
    forecast_rows: list[dict[str, object]] = []
    solver_rows: list[dict[str, object]] = []
    warm_hints: dict[str, tuple[str, float]] = {}
    window_starts = list(range(0, MAIN_END, config.decision_hours)) + [MAIN_END]
    total_windows = len(window_starts)
    for window_index, tau in enumerate(window_starts, start=1):
        decision_hours = min(config.decision_hours, MAIN_END - tau) if tau < MAIN_END else OPERATION_END - MAIN_END
        lookahead = min(config.lookahead_hours, max(OPERATION_END - (tau + decision_hours), 0)) if tau < MAIN_END else 0
        _progress(
            f"滚动窗口进度：{window_index}/{total_windows}（{window_index / total_windows:.1%}），"
            f"窗口={tau}，决策区={decision_hours}h，前瞻区={lookahead}h。"
        )
        fixed_gpu, fixed_ai = _fixed_task_loads(assignments, tau, tau + decision_hours + lookahead, data.region_index)
        del fixed_gpu
        problem = build_window_problem(
            data, config, tau, decision_hours, lookahead, assignments, current_soc,
            historical_peak, scaling, block_hours=config.block_hours,
        )
        due_count = sum(1 for task_id in set(option.task_id for option in problem.x_options)
                        if _latest_integer_start(data.task_lookup.loc[task_id]) < problem.decision_end)
        near_min_soc = np.any(current_soc <= storage["MinSOC_MWh"].to_numpy(dtype=float) + 0.1 * (
            storage["StorageCapacity_MWh"].to_numpy(dtype=float) - storage["MinSOC_MWh"].to_numpy(dtype=float)
        ))
        initial_time_limit = config.difficult_time_limit_seconds if (
            due_count >= config.difficult_due_task_threshold or near_min_soc
        ) else config.normal_time_limit_seconds
        _progress(
            f"窗口{tau}问题构建完成：变量={problem.lower.size}，约束={problem.matrix.shape[0]}，"
            f"待处理任务={due_count}，时间上限={initial_time_limit:.0f}s。"
        )
        objective = _balanced_objective(problem, config)
        incumbent = _heuristic_incumbent(
            data, problem, assignments, current_soc, historical_peak,
            scaling, warm_hints, config.feasibility_tolerance,
        )
        solve_attempts = 0
        refinement_reason = "none"
        try:
            solution, attempts, time_limit = _solve_with_time_extension(
                problem, objective, config, incumbent, initial_time_limit
            )
            solve_attempts += attempts
        except RuntimeError as block4_error:
            solve_attempts += 1 if initial_time_limit >= config.difficult_time_limit_seconds else 2
            if lookahead <= 0 or config.block_hours <= 2:
                raise
            refinement_reason = "block4_no_feasible_solution"
            _progress(f"窗口{tau}的4小时块未获得可行解，改用2小时块重建并重试。")
            problem = build_window_problem(
                data, config, tau, decision_hours, lookahead, assignments, current_soc,
                historical_peak, scaling, block_hours=2,
            )
            objective = _balanced_objective(problem, config)
            incumbent = _heuristic_incumbent(
                data, problem, assignments, current_soc, historical_peak,
                scaling, warm_hints, config.feasibility_tolerance,
            )
            try:
                solution, attempts, time_limit = _solve_with_time_extension(
                    problem, objective, config, incumbent,
                    config.difficult_time_limit_seconds,
                )
                solve_attempts += attempts
            except RuntimeError as block2_error:
                raise RuntimeError(
                    f"窗口{tau}的4小时块和2小时块均未获得可行解；"
                    f"4小时块：{block4_error}；2小时块：{block2_error}"
                ) from block2_error
        k_capacity_violation = _k_capacity_violation(data, problem, solution.vector, assignments)
        if (
            lookahead > 0
            and int(problem.metadata["BlockHours"]) > 2
            and k_capacity_violation > config.feasibility_tolerance
        ):
            refinement_reason = "block4_capacity_audit"
            _progress(
                f"窗口{tau}通过4小时块求解但K区容量复核超限（{k_capacity_violation:.6g}），"
                "改用2小时块精修。"
            )
            problem = build_window_problem(
                data, config, tau, decision_hours, lookahead, assignments, current_soc,
                historical_peak, scaling, block_hours=2,
            )
            objective = _balanced_objective(problem, config)
            incumbent = _heuristic_incumbent(
                data, problem, assignments, current_soc, historical_peak,
                scaling, warm_hints, config.feasibility_tolerance,
            )
            solution, attempts, time_limit = _solve_with_time_extension(
                problem, objective, config, incumbent,
                config.difficult_time_limit_seconds,
            )
            solve_attempts += attempts
            k_capacity_violation = _k_capacity_violation(data, problem, solution.vector, assignments)
            if k_capacity_violation > config.feasibility_tolerance:
                raise RuntimeError(
                    f"窗口{tau}改用2小时块后K区容量复核仍超限：{k_capacity_violation:.6g}"
                )
        new_assignments = _selected_assignments(data, problem, solution.vector)
        if not new_assignments.empty:
            duplicates = set(new_assignments["TaskID"]) & set(assignments["TaskID"])
            if duplicates:
                raise RuntimeError(f"任务被重复执行，示例：{sorted(duplicates)[:5]}")
            assignments = pd.concat([assignments, new_assignments], ignore_index=True)
        dispatch_rows.extend(_dispatch_rows(data, problem, solution, assignments))
        forecast_rows.extend(_forecast_rows(problem, solution.vector, fixed_ai))
        h_index = problem.decision_end - problem.tau
        current_soc = solution.vector[problem.indices["soc"][h_index]].copy()
        warm_hints = _next_warm_hints(problem, solution.vector)
        for r in range(len(data.regions)):
            local_net = [
                solution.vector[problem.indices["grid_purchase"][t, r]]
                - solution.vector[problem.indices["export"][t, r]]
                for t in range(h_index)
            ]
            historical_peak[r] = max(historical_peak[r], max(local_net, default=0.0), 0.0)
        metrics = _metric_values(problem, solution.vector)
        solver_rows.append({
            "WindowStart": tau, "DecisionEnd": problem.decision_end, "PlanEnd": problem.plan_end,
            **problem.metadata, "DueTaskCount": due_count,
            "InitialTimeLimitSeconds": initial_time_limit,
            "TimeLimitSeconds": time_limit,
            "SolveAttemptCount": solve_attempts,
            "HeuristicIncumbentAvailable": incumbent is not None,
            "WarmHintTaskCount": len(warm_hints),
            "RefinementReason": refinement_reason,
            "KCapacityViolation": k_capacity_violation,
            "SolverStatus": solution.status, "SolverMessage": solution.message,
            "ObjectiveValue": solution.objective, "MIPGap": solution.mip_gap,
            "MIPNodeCount": solution.mip_node_count, "ElapsedSeconds": solution.elapsed_seconds,
            **{f"Window{metric}": value for metric, value in metrics.items()},
        })
        _progress(
            f"滚动窗口{window_index}/{total_windows}完成：窗口={tau}，"
            f"求解耗时={solution.elapsed_seconds:.2f}s，新增任务={len(new_assignments)}，"
            f"累计任务={len(assignments)}，尝试次数={solve_attempts}。"
        )
    _progress("滚动窗口优化完成，开始整理和写出结果。")
    assignments = assignments.sort_values(["StartHour", "TaskID"], kind="stable").reset_index(drop=True)
    if len(assignments) != len(data.tasks) or assignments["TaskID"].nunique() != len(data.tasks):
        missing = sorted(set(data.tasks["TaskID"]) - set(assignments["TaskID"]))[:10]
        raise RuntimeError(f"滚动结束后仍有任务未执行，示例：{missing}")
    dispatch = pd.DataFrame(dispatch_rows).sort_values(["Hour", "Region"], kind="stable").reset_index(drop=True)
    expected_dispatch_rows = OPERATION_END * len(data.regions)
    if len(dispatch) != expected_dispatch_rows:
        raise RuntimeError(f"实际能源轨迹应有{expected_dispatch_rows}行，当前为{len(dispatch)}行")
    metrics = _final_metrics(data, assignments, dispatch)
    summary = pd.DataFrame([
        {"Scheme": "Q4JointRolling", "Metric": metric, "Value": metrics[metric]}
        for metric in METRICS
    ])
    configuration = pd.DataFrame([
        {"Parameter": key, "Value": value} for key, value in asdict(config).items()
    ] + [
        {"Parameter": "PythonRequirement", "Value": "3.12.13"},
        {"Parameter": "Solver", "Value": "scipy.optimize.milp (HiGHS)"},
        {"Parameter": "NativeMIPStartSupported", "Value": False},
        {"Parameter": "WarmStartMethod", "Value": "feasible incumbent cutoff plus previous-window K hints"},
        {"Parameter": "ActualOperationHours", "Value": "0--2405"},
        {"Parameter": "TerminalSOCHour", "Value": 2406},
    ])
    _write_csv(assignments, "q4_task_assignments.csv")
    _write_csv(dispatch, "q4_region_hour_dispatch.csv")
    _write_csv(summary, "q4_objective_summary.csv")
    _write_csv(pd.DataFrame(solver_rows), "q4_window_solver.csv")
    _write_csv(scaling_rows, "q4_scaling.csv")
    _write_csv(calibration_records, "q4_calibration_records.csv")
    _write_csv(pd.DataFrame(forecast_rows), "q4_forecast_profile.csv")
    _write_csv(configuration, "q4_model_configuration.csv")
    _progress("结果表写出完成，开始执行Q4最小硬约束自检。")
    simple_validation = _simple_validation(
        data, assignments, dispatch, config.feasibility_tolerance
    )
    _write_csv(simple_validation, "q4_simple_validation.csv")
    if not bool(simple_validation["Passed"].all()):
        failed = simple_validation.loc[
            ~simple_validation["Passed"], ["Check", "MaxViolation", "Tolerance"]
        ]
        raise RuntimeError(f"Q4最小硬约束自检未通过：\n{failed.to_string(index=False)}")
    _progress(f"Q4最小硬约束自检通过：{len(simple_validation)}项。")
    (TABLES_DIR / ".q4_cache.json").write_text(
        json.dumps({"signature": signature, "complete": True}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _progress("Q4模型运行完成。")


def run(
    *,
    force: bool = False,
    config: ModelConfig | None = None,
    resume_from: int | None = None,
    state_file: Path | None = None,
) -> None:
    """执行固定定标、可恢复滚动优化和完整轨迹独立复核。"""

    config = config or ModelConfig()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    _progress(
        f"Q4模型开始运行：force={force}，resume_from={resume_from}，"
        f"state_file={state_file or '默认断点'}。"
    )
    signature = _cache_signature(config)
    if resume_from is None and state_file is None and not force and _can_reuse(signature):
        _progress("检测到完整且匹配的缓存结果，跳过求解。")
        return
    if resume_from is None and state_file is None and not force:
        auto_resume = _select_auto_resume_checkpoint(signature)
        if auto_resume is not None:
            state_file, resume_from = auto_resume
            _activate_progress_namespace(state_file)
            _progress(
                f"无参数启动自动选择有效断点：{state_file.name}，"
                f"next_tau={resume_from}；后续仍写入该分支，不覆盖原始断点。"
            )
    elif state_file is not None:
        state_file = state_file.resolve()
        _activate_progress_namespace(state_file)
        if resume_from is None:
            state = _checkpoint_header(state_file)
            try:
                resume_from = int(state["next_tau"]) if state is not None else None
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"无法从state-file读取next_tau：{state_file}") from exc
            if resume_from is None:
                raise RuntimeError(f"无法从state-file读取next_tau：{state_file}")

    data = load_data()
    _progress(
        f"输入读取完成：任务={len(data.tasks)}，区域={len(data.regions)}，"
        f"逐时区域记录={len(data.region_hour)}。"
    )
    storage = data.storage.set_index("Region").loc[list(data.regions)]
    if resume_from is not None:
        state = _load_progress_state(
            data, config, signature, int(resume_from), state_file
        )
        assignments = pd.DataFrame(state["assignments"])
        dispatch = pd.DataFrame(state["dispatch"])
        solver_frame = pd.DataFrame(state["solver"])
        forecast = pd.DataFrame(state.get("forecast", pd.DataFrame()))
        scaling = dict(state["scaling"])
        calibration_records = pd.DataFrame(state.get("calibration_records", pd.DataFrame()))
        current_soc = np.array(
            state["current_soc"], dtype=np.float64, copy=True, order="C"
        )
        historical_peak = np.array(
            state["historical_peak"], dtype=np.float64, copy=True, order="C"
        )
        warm_hints = dict(state.get("warm_hints", {}))
        start_tau = int(resume_from)
        _progress(
            f"恢复校验通过：前置窗口={len(solver_frame)}，已分配任务={len(assignments)}，"
            f"逐时调度={len(dispatch)}行，下一窗口={start_tau}。"
        )
    else:
        existing_progress = [
            path for path in (
                CHECKPOINT_PATH, PROGRESS_ASSIGNMENTS_PATH,
                PROGRESS_DISPATCH_PATH, PROGRESS_SOLVER_PATH,
            ) if path.exists()
        ]
        if existing_progress:
            raise RuntimeError(
                "检测到已有Q4进度，拒绝从0覆盖；请用--resume-from恢复："
                f"{[str(path) for path in existing_progress]}"
            )
        scaling, calibration_records = _calibrate(data, config)
        assignments = _empty_assignments()
        dispatch = pd.DataFrame()
        solver_frame = pd.DataFrame()
        forecast = pd.DataFrame()
        current_soc = storage["InitialSOC_MWh"].to_numpy(dtype=float)
        historical_peak = np.zeros(len(data.regions), dtype=float)
        warm_hints: dict[str, tuple[str, float]] = {}
        start_tau = 0
        _save_progress_checkpoint(
            data=data, config=config, signature=signature,
            last_completed_tau=None, next_tau=0,
            assignments=assignments, dispatch=dispatch, solver=solver_frame,
            forecast=forecast, scaling=scaling,
            calibration_records=calibration_records,
            current_soc=current_soc, historical_peak=historical_peak,
            warm_hints=warm_hints,
            mode={"H": 24, "K": 48, "BlockHours": 4, "RecoveryMode": "Initial"},
        )
        _progress("固定定标已完成并写入初始断点；正式滚动中不会重复定标。")

    historical_peak = np.array(
        historical_peak,
        dtype=np.float64,
        copy=True,
        order="C",
    )
    if not historical_peak.flags.writeable:
        historical_peak = historical_peak.copy()
    if set(scaling) != set(METRICS):
        raise ValueError("断点定标参数没有完整包含六个目标")
    all_windows = list(range(0, MAIN_END, config.decision_hours)) + [MAIN_END]
    windows_to_run = [tau for tau in all_windows if tau >= start_tau]
    for tau in windows_to_run:
        window_index = all_windows.index(tau) + 1
        decision_hours = (
            min(config.decision_hours, MAIN_END - tau)
            if tau < MAIN_END else OPERATION_END - MAIN_END
        )
        normal_lookahead = (
            min(config.lookahead_hours, max(OPERATION_END - tau - decision_hours, 0))
            if tau < MAIN_END else 0
        )
        _progress(
            f"滚动窗口{window_index}/{len(all_windows)}：tau={tau}，"
            f"H={decision_hours}h，正常K={normal_lookahead}h。"
        )
        assignments_before = assignments.copy()
        incoming_warm_hint_count = len(warm_hints)
        solve_attempts = 0
        recovery_errors: list[str] = []
        future_check = {
            "RemainingTaskCount": -1,
            "IndividuallyFeasibleTaskCount": -1,
        }
        problem: WindowProblem | None = None
        solution: WindowSolution | None = None
        incumbent: np.ndarray | None = None
        model_mode = ""
        k_capacity_violation = 0.0
        time_limit = config.difficult_time_limit_seconds

        if tau != 1608:
            try:
                problem = build_window_problem(
                    data, config, tau, decision_hours, normal_lookahead,
                    assignments_before, current_soc, historical_peak, scaling,
                    block_hours=config.block_hours,
                )
                due_count = sum(
                    1 for task_id in {option.task_id for option in problem.x_options}
                    if _latest_integer_start(data.task_lookup.loc[task_id]) < problem.decision_end
                )
                near_min_soc = np.any(
                    current_soc <= storage["MinSOC_MWh"].to_numpy(dtype=float) + 0.1 * (
                        storage["StorageCapacity_MWh"].to_numpy(dtype=float)
                        - storage["MinSOC_MWh"].to_numpy(dtype=float)
                    )
                )
                time_limit = (
                    config.difficult_time_limit_seconds
                    if due_count >= config.difficult_due_task_threshold or near_min_soc
                    else config.normal_time_limit_seconds
                )
                incumbent = _heuristic_incumbent(
                    data, problem, assignments_before, current_soc, historical_peak,
                    scaling, warm_hints, config.feasibility_tolerance,
                )
                solve_attempts += 1
                solution = _solve(
                    problem, _balanced_objective(problem, config),
                    time_limit=time_limit, mip_gap=config.mip_relative_gap,
                    relax=False, incumbent=incumbent,
                )
                model_mode = "NormalHK"
                k_capacity_violation = _k_capacity_violation(
                    data, problem, solution.vector, assignments_before
                )
                if k_capacity_violation > config.feasibility_tolerance:
                    raise RuntimeError(
                        f"K区4小时块容量审计超限{k_capacity_violation:.6g}；"
                        "自动2小时细化已禁用"
                    )
            except RuntimeError as exc:
                recovery_errors.append(f"NormalHK: {exc}")
                problem = None
                solution = None
                _progress(f"窗口{tau}正常H+K失败，转入H-only恢复：{exc}")
        else:
            _progress("窗口1608直接使用H=24、K=0，不创建K区任务预测变量。")

        if solution is None:
            for feasibility_only in (False, True):
                try:
                    solve_attempts += 1
                    candidate_problem, candidate_solution, candidate_incumbent, candidate_mode = (
                        _solve_h_only_recovery(
                            data, config, tau, decision_hours, assignments_before,
                            current_soc, historical_peak, scaling, warm_hints,
                            feasibility_only=feasibility_only,
                        )
                    )
                    candidate_new = _selected_assignments(
                        data, candidate_problem, candidate_solution.vector
                    )
                    candidate_assignments = pd.concat(
                        [assignments_before, candidate_new], ignore_index=True
                    )
                    future_check = _future_feasibility_check(
                        data, candidate_assignments, candidate_problem.decision_end,
                        config.feasibility_tolerance,
                    )
                    problem = candidate_problem
                    solution = candidate_solution
                    incumbent = candidate_incumbent
                    model_mode = candidate_mode
                    time_limit = config.difficult_time_limit_seconds
                    k_capacity_violation = 0.0
                    break
                except RuntimeError as exc:
                    stage = "HOnlyFeasibility" if feasibility_only else "HOnlyBalanced"
                    recovery_errors.append(f"{stage}: {exc}")
                    _progress(f"窗口{tau}的{stage}未通过：{exc}")
        if problem is None or solution is None:
            mode = {
                "H": decision_hours, "K": 0, "BlockHours": config.block_hours,
                "RecoveryMode": "FailedAfterHOnlyFeasibility",
            }
            reason = "；".join(recovery_errors)
            _record_failed_window(
                tau=tau, signature=signature, reason=reason, mode=mode
            )
            raise RuntimeError(
                f"窗口{tau}在正常模型、H-only联合目标和同硬约束可行性模型中"
                f"均未获得合法整数方案；最后成功断点保持不变。{reason}"
            )

        new_assignments = _selected_assignments(data, problem, solution.vector)
        duplicates = set(new_assignments["TaskID"]) & set(assignments_before["TaskID"])
        if duplicates:
            raise RuntimeError(f"任务被重复执行，示例：{sorted(duplicates)[:5]}")
        assignments = pd.concat(
            [assignments_before, new_assignments], ignore_index=True
        )
        if model_mode == "NormalHK":
            future_check = {
                "RemainingTaskCount": int(len(data.tasks) - len(assignments)),
                "IndividuallyFeasibleTaskCount": -1,
            }
        _, fixed_ai = _fixed_task_loads(
            assignments_before, problem.tau, problem.plan_end, data.region_index
        )
        dispatch = pd.concat([
            dispatch,
            pd.DataFrame(_dispatch_rows(data, problem, solution, assignments)),
        ], ignore_index=True)
        new_forecast = pd.DataFrame(_forecast_rows(problem, solution.vector, fixed_ai))
        if not new_forecast.empty:
            forecast = pd.concat([forecast, new_forecast], ignore_index=True)
        h_count = problem.decision_end - problem.tau
        current_soc = solution.vector[problem.indices["soc"][h_count]].copy()
        warm_hints = _next_warm_hints(problem, solution.vector)
        for r in range(len(data.regions)):
            local_net = [
                solution.vector[problem.indices["grid_purchase"][t, r]]
                - solution.vector[problem.indices["export"][t, r]]
                for t in range(h_count)
            ]
            historical_peak[r] = max(
                historical_peak[r], max(local_net, default=0.0), 0.0
            )
        due_count = sum(
            1 for task_id in {option.task_id for option in problem.x_options}
            if _latest_integer_start(data.task_lookup.loc[task_id]) < problem.decision_end
        )
        metrics = _metric_values(problem, solution.vector)
        solver_frame = pd.concat([solver_frame, pd.DataFrame([{
            "WindowStart": tau,
            "DecisionEnd": problem.decision_end,
            "PlanEnd": problem.plan_end,
            **problem.metadata,
            "ModelMode": model_mode,
            "DueTaskCount": due_count,
            "TimeLimitSeconds": time_limit,
            "SolveAttemptCount": solve_attempts,
            "HeuristicIncumbentAvailable": incumbent is not None,
            "IncomingWarmHintTaskCount": incoming_warm_hint_count,
            "OutgoingWarmHintTaskCount": len(warm_hints),
            "RefinementReason": "Automatic2HourRefinementDisabled",
            "KCapacityViolation": k_capacity_violation,
            **future_check,
            "SolverStatus": solution.status,
            "SolverMessage": solution.message,
            "ObjectiveValue": solution.objective,
            "MIPGap": solution.mip_gap,
            "MIPNodeCount": solution.mip_node_count,
            "ElapsedSeconds": solution.elapsed_seconds,
            **{f"Window{metric}": value for metric, value in metrics.items()},
        }])], ignore_index=True)
        next_tau = _next_tau_after(tau)
        mode = {
            "H": decision_hours,
            "K": problem.plan_end - problem.decision_end,
            "BlockHours": int(problem.metadata["BlockHours"]),
            "RecoveryMode": model_mode,
        }
        _save_progress_checkpoint(
            data=data, config=config, signature=signature,
            last_completed_tau=tau, next_tau=next_tau,
            assignments=assignments, dispatch=dispatch, solver=solver_frame,
            forecast=forecast, scaling=scaling,
            calibration_records=calibration_records,
            current_soc=current_soc, historical_peak=historical_peak,
            warm_hints=warm_hints, mode=mode,
        )
        _progress(
            f"窗口{tau}完成并保存断点：模式={model_mode}，变量={problem.lower.size}，"
            f"约束={problem.matrix.shape[0]}，耗时={solution.elapsed_seconds:.2f}s，"
            f"新增任务={len(new_assignments)}，累计任务={len(assignments)}，next_tau={next_tau}。"
        )

    _progress("全部窗口完成，开始从0--2405小时完整实际轨迹独立复算。")
    assignments = assignments.sort_values(
        ["StartHour", "TaskID"], kind="stable"
    ).reset_index(drop=True)
    if len(assignments) != len(data.tasks) or assignments["TaskID"].nunique() != len(data.tasks):
        missing = sorted(set(data.tasks["TaskID"]) - set(assignments["TaskID"]))[:10]
        raise RuntimeError(f"滚动结束后仍有任务未执行，示例：{missing}")
    dispatch = dispatch.sort_values(
        ["Hour", "Region"], kind="stable"
    ).reset_index(drop=True)
    expected_dispatch_rows = OPERATION_END * len(data.regions)
    if len(dispatch) != expected_dispatch_rows:
        raise RuntimeError(
            f"实际能源轨迹应有{expected_dispatch_rows}行，当前为{len(dispatch)}行"
        )
    final_metrics = _final_metrics(data, assignments, dispatch)
    summary = pd.DataFrame([
        {"Scheme": "Q4JointRolling", "Metric": metric, "Value": final_metrics[metric]}
        for metric in METRICS
    ])
    configuration = pd.DataFrame([
        {"Parameter": key, "Value": value} for key, value in asdict(config).items()
    ] + [
        {"Parameter": "PythonRequirement", "Value": "3.12.13"},
        {"Parameter": "Solver", "Value": "scipy.optimize.milp (HiGHS)"},
        {"Parameter": "NormalWindowSolveCount", "Value": 1},
        {"Parameter": "Automatic2HourRefinement", "Value": False},
        {"Parameter": "ActualOperationHours", "Value": "0--2405"},
        {"Parameter": "TerminalSOCHour", "Value": 2406},
    ])
    _write_csv(assignments, "q4_task_assignments.csv")
    _write_csv(dispatch, "q4_region_hour_dispatch.csv")
    _write_csv(summary, "q4_objective_summary.csv")
    _write_csv(solver_frame, "q4_window_solver.csv")
    _write_csv(_scaling_frame(scaling), "q4_scaling.csv")
    _write_csv(calibration_records, "q4_calibration_records.csv")
    _write_csv(forecast, "q4_forecast_profile.csv")
    _write_csv(configuration, "q4_model_configuration.csv")
    hard_validation = _simple_validation(
        data, assignments, dispatch, config.feasibility_tolerance
    )
    _write_csv(hard_validation, "q4_simple_validation.csv")
    _write_csv(hard_validation, "q4_hard_constraint_validation.csv")
    if not bool(hard_validation["Passed"].all()):
        failed = hard_validation.loc[
            ~hard_validation["Passed"], ["Check", "MaxViolation", "Tolerance"]
        ]
        raise RuntimeError(f"Q4最终硬约束验证未通过：\n{failed.to_string(index=False)}")
    _atomic_write_json(
        TABLES_DIR / ".q4_cache.json",
        {"signature": signature, "complete": True},
    )
    _progress(f"Q4最终硬约束验证通过：{len(hard_validation)}项。")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Q4联合滚动MILP")
    parser.add_argument(
        "--resume-from", type=int, default=None,
        help="从指定滚动窗口边界恢复，例如1608",
    )
    parser.add_argument(
        "--state-file", type=Path, default=None,
        help="显式指定pkl恢复状态文件",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="忽略完整结果缓存；不会覆盖已有进度断点",
    )
    args = parser.parse_args(argv)
    run(
        force=args.force,
        resume_from=args.resume_from,
        state_file=args.state_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
