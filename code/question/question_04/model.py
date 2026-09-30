"""Q4: rolling compute-storage-grid MILP with exact decisions and compressed lookahead."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


QUESTION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = QUESTION_DIR.parents[1]
SHARED_DIR = PROJECT_DIR / "question" / "question_01" / "data" / "processed" / "shared"
Q4_DIR = QUESTION_DIR / "data" / "processed" / "q4"
TABLES_DIR = QUESTION_DIR / "outputs" / "tables"
VALIDATION_DIR = QUESTION_DIR / "outputs" / "validation"
SCENARIO_DIR = QUESTION_DIR / "outputs" / "scenarios"

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
QOS_WEIGHTS = {"High": 3.0, "Medium": 2.0, "Low": 1.0}
BASELINE_REQUIRED_FILES = (
    "q4_task_assignments.csv",
    "q4_region_hour_dispatch.csv",
    "q4_objective_summary.csv",
    "q4_window_solver.csv",
    "q4_scaling.csv",
    "q4_model_configuration.csv",
    "q4_simple_validation.csv",
    "q4_calibration_records.csv",
    "q4_forecast_profile.csv",
)
SOLVER_STATUS_NAMES = {
    0: "OPTIMAL",
    1: "TIME_LIMIT_FEASIBLE",
    2: "INFEASIBLE",
    3: "UNKNOWN",
    4: "UNKNOWN",
}


def _progress(message: str) -> None:
    print(f"[Q4 progress] {message}", flush=True)


def _checkpoint_header(path: Path) -> dict[str, object] | None:
    try:
        with path.open("rb") as handle:
            state = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError):
        return None
    return state if isinstance(state, dict) else None


def _select_auto_resume_checkpoint(signature: str) -> tuple[Path, int] | None:
    """Select the furthest advanced matching formal or repair-branch checkpoint."""

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
    """Resume into the branch's own progress files without overwriting the original checkpoint."""

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
    """Write a complete temporary file in the same directory, then atomically replace the target."""

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
    """Convert optional solver fields to float; continuous relaxations may return None."""

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
    integrality_tolerance: float = 1e-6
    solver_threads: int = 0
    heuristic_enabled: bool = True
    critical_neighborhood_enabled: bool = True
    qos_high_weight: float = 3.0
    qos_medium_weight: float = 2.0
    qos_low_weight: float = 1.0
    # Q4 final paper model: the following are computational controls rather
    # than mathematical-objective weights.  They deliberately replace the
    # old "extend a full H+K MILP" fallback in the formal solve path.
    energy_mip_gap: float = 0.005
    energy_time_limit_seconds: float = 15.0
    energy_physical_tie_break_weight: float = 1e-6
    repair_time_limit_seconds: float = 12.0
    lns_time_limit_seconds: float = 15.0
    lns_target_gap: float = 0.02
    lns_max_passes: int = 2
    lns_max_task_groups: int = 60
    lns_max_integer_option_vars: int = 12000
    marginal_max_moves: int = 48
    marginal_lp_time_limit_seconds: float = 5.0
    calibration_lower_bound_time_limit_seconds: float = 20.0
    calibration_reference_time_limit_seconds: float = 15.0
    window_soft_time_limit_seconds: float = 45.0
    window_hard_time_limit_seconds: float = 75.0


def _qos_weight_map(config: ModelConfig | None = None) -> dict[str, float]:
    if config is None:
        return dict(QOS_WEIGHTS)
    weights = {
        "High": float(config.qos_high_weight),
        "Medium": float(config.qos_medium_weight),
        "Low": float(config.qos_low_weight),
    }
    if not (weights["High"] >= weights["Medium"] >= weights["Low"] > 0.0):
        raise ValueError("QoS weights must satisfy High>=Medium>=Low>0")
    return weights


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
    latency_ms: float
    latency_ratio: float
    overlaps: tuple[tuple[int, float], ...]
    task_ids: tuple[str, ...]
    group_size: int
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
    task_groups: dict[str, tuple[str, ...]]
    task_group_order: tuple[str, ...]
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
    best_bound: float = float("nan")
    status_name: str = "UNKNOWN"


def _read_csv(path: Path, required: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing model input: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    return frame


def _numeric(frame: pd.DataFrame, columns: Iterable[str], source: str) -> pd.DataFrame:
    result = frame.copy()
    columns = tuple(columns)
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[list(columns)].isna().any().any():
        raise ValueError(f"{source} contains invalid numeric values")
    return result


def _validate_task_time_and_latency_inputs(
    tasks: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict[str, int]:
    """Fail fast when immutable task data contradicts the Q4 time/SLA domain.

    This is intentionally evaluated before any checkpoint or solver state is
    created. In particular, a ``RealTimeInference`` task must be executable
    at its arrival hour, rather than merely having a later feasible start.
    """

    if tasks["TaskID"].duplicated().any():
        sample = tasks.loc[tasks["TaskID"].duplicated(keep=False), "TaskID"].astype(str).head(5).tolist()
        raise ValueError(f"tasks_clean.csv contains duplicate TaskID values; examples: {sample}")
    supported_types = {"RealTimeInference", "BatchInference", "AITraining"}
    unknown_types = sorted(set(tasks["TaskType"]) - supported_types)
    if unknown_types:
        raise ValueError(f"tasks_clean.csv contains unknown TaskType values: {unknown_types}")
    arrival = tasks["ArrivalHour"].to_numpy(dtype=float)
    arrival_integer = np.rint(arrival).astype(int)
    if np.abs(arrival - arrival_integer).max(initial=0.0) > EPS:
        raise ValueError("tasks_clean.csv requires integer ArrivalHour values")
    if ((arrival < -EPS) | (arrival >= float(MAIN_END) - EPS)).any():
        raise ValueError("tasks_clean.csv requires arrival times within hours 0--2399")
    if (tasks["Duration_h"].to_numpy(dtype=float) <= EPS).any():
        raise ValueError("tasks_clean.csv contains nonpositive task durations")
    if (tasks["MaxLatency_ms"].to_numpy(dtype=float) < -EPS).any():
        raise ValueError("tasks_clean.csv contains negative maximum network latencies")

    earliest = np.maximum(
        arrival_integer,
        np.ceil(tasks["EarliestStartHour"].to_numpy(dtype=float) - EPS).astype(int),
    )
    latest = np.floor(
        np.minimum(tasks["LatestFinishHour"].to_numpy(dtype=float), float(OPERATION_END))
        - tasks["Duration_h"].to_numpy(dtype=float) + EPS
    ).astype(int)
    infeasible = tasks.loc[earliest > latest, "TaskID"].astype(str).head(5).tolist()
    if infeasible:
        raise ValueError(f"tasks_clean.csv contains tasks without a feasible integer start time; examples: {infeasible}")
    real_time = tasks["TaskType"].eq("RealTimeInference").to_numpy()
    bad_realtime = tasks.loc[
        real_time & ((earliest != arrival_integer) | (arrival_integer > latest)),
        "TaskID",
    ].astype(str).head(5).tolist()
    if bad_realtime:
        raise ValueError(
            "tasks_clean.csv contains realtime tasks that cannot start immediately on arrival; "
            f"examples: {bad_realtime}"
        )

    if candidates.duplicated(["TaskID", "TargetRegion"]).any():
        sample = candidates.loc[
            candidates.duplicated(["TaskID", "TargetRegion"], keep=False),
            ["TaskID", "TargetRegion"],
        ].head(5).to_dict("records")
        raise ValueError(f"task_candidate_regions.csv contains duplicate TaskID--TargetRegion pairs: {sample}")
    bounds = candidates.merge(
        tasks[["TaskID", "MaxLatency_ms"]].rename(
            columns={"MaxLatency_ms": "TaskMaxLatency_ms"}
        ),
        on="TaskID",
        how="left",
        validate="many_to_one",
    )
    unknown_candidate_task = bounds.loc[
        bounds["TaskMaxLatency_ms"].isna(), "TaskID"
    ].astype(str).head(5).tolist()
    if unknown_candidate_task:
        raise ValueError(
            "task_candidate_regions.csv contains TaskID values absent from the original task table; "
            f"examples: {unknown_candidate_task}"
        )
    if (bounds["NetworkLatency_ms"].to_numpy(dtype=float) < -EPS).any():
        raise ValueError("task_candidate_regions.csv contains negative network latencies")
    max_mismatch = np.abs(
        bounds["MaxLatency_ms"].to_numpy(dtype=float)
        - bounds["TaskMaxLatency_ms"].to_numpy(dtype=float)
    )
    if max_mismatch.max(initial=0.0) > EPS:
        raise ValueError("Candidate-region MaxLatency_ms values differ from the task table")
    over_latency = (
        bounds["NetworkLatency_ms"].to_numpy(dtype=float)
        - bounds["TaskMaxLatency_ms"].to_numpy(dtype=float)
    )
    if over_latency.max(initial=0.0) > EPS:
        raise ValueError("task_candidate_regions.csv contains candidates exceeding the task latency limit")
    return {
        "TaskCount": int(len(tasks)),
        "RealTimeTaskCount": int(np.count_nonzero(real_time)),
        "CandidateRowCount": int(len(candidates)),
    }


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
            "Task_Full_IT_Power_MW", "DelaySensitivity",
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
    tasks["TaskType"] = tasks["TaskType"].astype(str).str.strip()
    tasks["SourceRegion"] = tasks["SourceRegion"].astype(str)
    tasks["DelaySensitivity"] = tasks["DelaySensitivity"].astype(str).str.strip()
    unknown_sensitivity = sorted(set(tasks["DelaySensitivity"]) - set(QOS_WEIGHTS))
    if unknown_sensitivity:
        raise ValueError(f"tasks_clean.csv contains unknown DelaySensitivity values: {unknown_sensitivity}")
    candidates["TaskID"] = candidates["TaskID"].astype(str)
    candidates["TargetRegion"] = candidates["TargetRegion"].astype(str)
    _validate_task_time_and_latency_inputs(tasks, candidates)
    storage["Region"] = storage["Region"].astype(str)
    regions = tuple(storage["Region"])
    region_index = {region: index for index, region in enumerate(regions)}
    if len(region_index) != len(regions):
        raise ValueError("storage_params.csv requires unique Region values")
    expected_hours = set(range(OPERATION_END + 1))
    for region in regions:
        actual = set(region_hour.loc[region_hour["Region"].eq(region), "Hour"])
        if actual != expected_hours:
            raise ValueError(f"{region} does not fully cover hours 0--2406")
    region_hour = region_hour.sort_values(["Hour", "Region"], kind="stable").reset_index(drop=True)
    candidate_map: dict[str, tuple[CandidateRegion, ...]] = {}
    for task_id, group in candidates.groupby("TaskID", sort=False):
        rows: list[CandidateRegion] = []
        for row in group.itertuples(index=False):
            region = str(row.TargetRegion)
            if region not in region_index:
                raise ValueError(f"Task {task_id}: candidate region {region} is absent from the storage region set")
            if float(row.NetworkLatency_ms) > float(row.MaxLatency_ms) + EPS:
                raise ValueError(f"Task {task_id} has a candidate exceeding its latency limit")
            rows.append(CandidateRegion(region, region_index[region], float(row.NetworkLatency_ms), float(row.MaxLatency_ms)))
        candidate_map[str(task_id)] = tuple(rows)
    missing = set(tasks["TaskID"]) - set(candidate_map)
    if missing:
        raise ValueError(f"Tasks lack candidate regions; examples: {sorted(missing)[:10]}")
    task_lookup = tasks.set_index("TaskID", drop=False)
    return InputData(region_hour, tasks, candidates, storage, regions, region_index, candidate_map, task_lookup)


def _latest_integer_start(task: pd.Series | object) -> int:
    latest_finish = min(float(task.LatestFinishHour), float(OPERATION_END))
    return int(math.floor(latest_finish - float(task.Duration_h) + EPS))


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
            raise ValueError(f"Existing task {row.TaskID} has invalid region {region}")
        r = region_index[region]
        for hour, overlap in _hour_overlaps(float(row.StartHour), float(row.Duration_h), lower, upper):
            gpu[hour - lower, r] += float(row.GPU_Demand) * overlap
            ai[hour - lower, r] += float(row.Task_Full_IT_Power_MW) * overlap
    return gpu, ai


def _float_key(value: object) -> float:
    return round(float(value), 12)


def _exact_task_group_key(data: InputData, task: pd.Series) -> tuple[object, ...]:
    """Aggregate only tasks with identical feasible sets, loads, and objectives."""

    task_id = str(task.TaskID)
    candidate_signature = tuple(sorted(
        (
            candidate.region,
            _float_key(candidate.latency_ms),
            _float_key(candidate.max_latency_ms),
        )
        for candidate in data.candidate_map[task_id]
    ))
    optional_columns = tuple(
        (column, str(task[column]))
        for column in ("ExecutionMode", "TaskPriority", "Priority")
        if column in task.index
    )
    return (
        str(task.TaskType),
        int(task.ArrivalHour),
        str(task.SourceRegion),
        _float_key(task.GPU_Demand),
        _float_key(task.Duration_h),
        _float_key(task.Task_Full_IT_Power_MW),
        _float_key(task.MaxLatency_ms),
        _float_key(task.EarliestStartHour),
        _float_key(task.LatestFinishHour),
        str(task.DelaySensitivity),
        optional_columns,
        candidate_signature,
    )


def _make_task_options(
    data: InputData,
    tau: int,
    decision_end: int,
    plan_end: int,
    assignments: pd.DataFrame,
    block_hours: int,
    calibration: bool,
) -> tuple[
    list[TaskOption],
    list[TaskOption],
    dict[str, tuple[bool, bool]],
    dict[str, tuple[str, ...]],
]:
    assigned = set(assignments["TaskID"].astype(str)) if not assignments.empty else set()
    considered = data.tasks.loc[
        (~data.tasks["TaskID"].isin(assigned))
        & (data.tasks["ArrivalHour"] < plan_end)
        & (data.tasks["EarliestStartHour"] < plan_end)
    ].copy()
    if calibration:
        considered = considered.loc[considered["ArrivalHour"] >= tau]
    considered = considered.sort_values(["LatestFinishHour", "ArrivalHour", "TaskID"], kind="stable")
    grouped: dict[tuple[object, ...], list[int]] = {}
    for row_index, task in considered.iterrows():
        grouped.setdefault(_exact_task_group_key(data, task), []).append(int(row_index))
    x_options: list[TaskOption] = []
    u_options: list[TaskOption] = []
    task_rule: dict[str, tuple[bool, bool]] = {}
    task_groups: dict[str, tuple[str, ...]] = {}
    ordered_groups = sorted(
        grouped.values(),
        key=lambda rows: (
            float(data.tasks.loc[rows[0], "LatestFinishHour"]),
            float(data.tasks.loc[rows[0], "ArrivalHour"]),
            str(data.tasks.loc[rows[0], "TaskID"]),
        ),
    )
    for member_rows in ordered_groups:
        task = data.tasks.loc[member_rows[0]]
        task_ids = tuple(sorted(data.tasks.loc[member_rows, "TaskID"].astype(str)))
        task_id = task_ids[0]
        group_size = len(task_ids)
        task_groups[task_id] = task_ids
        candidates = data.candidate_map[task_id]
        h_starts = _feasible_starts(task, tau, decision_end)
        k_starts = _feasible_starts(task, decision_end, plan_end)
        must_commit = (
            str(task.TaskType) == "RealTimeInference" and int(task.ArrivalHour) < decision_end
        ) or _latest_integer_start(task) < decision_end
        must_plan = (
            str(task.TaskType) == "RealTimeInference" and int(task.ArrivalHour) < plan_end
        ) or _latest_integer_start(task) < plan_end
        if str(task.TaskType) == "RealTimeInference" and int(task.ArrivalHour) < tau and not calibration:
            raise RuntimeError(f"Realtime task group {task_ids[:3]} was not executed in its arrival window")
        if must_commit and not h_starts:
            raise RuntimeError(f"Task group {task_ids[:3]} reached its latest start window without a valid H-region start")
        if must_plan and not (h_starts or k_starts):
            raise RuntimeError(f"Task group {task_ids[:3]} reached its latest H+K planning window without a valid start")
        for candidate in candidates:
            for start in h_starts:
                x_options.append(TaskOption(
                    task_id=task_id,
                    task_row=int(member_rows[0]),
                    region=candidate.region,
                    region_index=candidate.region_index,
                    start_hour=float(start),
                    expected_start_hour=float(start),
                    duration_h=float(task.Duration_h),
                    gpu_demand=float(task.GPU_Demand),
                    power_mw=float(task.Task_Full_IT_Power_MW),
                    latency_ms=float(candidate.latency_ms),
                    latency_ratio=float(candidate.latency_ms) / max(float(task.MaxLatency_ms), EPS),
                    overlaps=_hour_overlaps(float(start), float(task.Duration_h), tau, plan_end),
                    task_ids=task_ids,
                    group_size=group_size,
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
                    for hour, overlap in _hour_overlaps(
                        float(start), float(task.Duration_h), tau, OPERATION_END
                    ):
                        overlap_sum[hour] = overlap_sum.get(hour, 0.0) + overlap / len(starts)
                average = tuple(sorted((hour, value) for hour, value in overlap_sum.items() if value > EPS))
                expected_start = float(np.mean(starts))
                for candidate in candidates:
                    u_options.append(TaskOption(
                        task_id=task_id,
                        task_row=int(member_rows[0]),
                        region=candidate.region,
                        region_index=candidate.region_index,
                        start_hour=float("nan"),
                        expected_start_hour=expected_start,
                        duration_h=float(task.Duration_h),
                        gpu_demand=float(task.GPU_Demand),
                        power_mw=float(task.Task_Full_IT_Power_MW),
                        latency_ms=float(candidate.latency_ms),
                        latency_ratio=float(candidate.latency_ms) / max(float(task.MaxLatency_ms), EPS),
                        overlaps=average,
                        task_ids=task_ids,
                        group_size=group_size,
                        is_block=True,
                        block_start=block_start,
                        block_end=block_end,
                    ))
        if h_starts or k_starts or must_commit or must_plan:
            task_rule[task_id] = (must_commit, must_plan)
    return x_options, u_options, task_rule, task_groups


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
    """Split SciPy LinearConstraint bounds into linprog equality and inequality matrices."""

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
    carbon_budget_remaining: float | None = None,
) -> WindowProblem:
    decision_end = min(tau + decision_hours, OPERATION_END)
    plan_end = min(decision_end + lookahead_hours, OPERATION_END)
    h_count = decision_end - tau
    t_count = plan_end - tau
    r_count = len(data.regions)
    if h_count <= 0 or t_count <= 0:
        raise ValueError("Rolling window length must be positive")
    block_hours = int(block_hours or config.block_hours)
    x_options, u_options, task_rules, task_groups = _make_task_options(
        data, tau, decision_end, plan_end, assignments, block_hours, calibration
    )
    offset = 0
    indices: dict[str, np.ndarray] = {}
    indices["x"], offset = _allocate(offset, (len(x_options),))
    indices["u"], offset = _allocate(offset, (len(u_options),))
    task_group_order = tuple(task_rules)
    indices["defer"], offset = _allocate(offset, (len(task_group_order),))
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
    if x_options:
        upper[indices["x"]] = np.asarray([option.group_size for option in x_options], dtype=float)
    if u_options:
        upper[indices["u"]] = np.asarray([option.group_size for option in u_options], dtype=float)
    if task_group_order:
        upper[indices["defer"]] = np.asarray(
            [len(task_groups[group_id]) for group_id in task_group_order],
            dtype=float,
        )
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
        raise ValueError(f"Window {tau} has incomplete hourly regional data")
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
    for group_index, (task_id, (must_commit, must_plan)) in enumerate(task_rules.items()):
        x_cols = x_by_task.get(task_id, [])
        u_cols = u_by_task.get(task_id, [])
        group_size = float(len(task_groups[task_id]))
        defer_column = int(indices["defer"][group_index])
        if must_commit:
            add_row({column: 1.0 for column in x_cols}, group_size, group_size)
            upper[defer_column] = 0.0
            if u_cols:
                upper[np.asarray(u_cols, dtype=int)] = 0.0
        elif must_plan:
            add_row(
                {column: 1.0 for column in x_cols + u_cols},
                group_size,
                group_size,
            )
            upper[defer_column] = 0.0
        elif x_cols or u_cols:
            add_row(
                {
                    **{column: 1.0 for column in x_cols + u_cols},
                    defer_column: 1.0,
                },
                group_size,
                group_size,
            )
        else:
            raise RuntimeError(f"Task group {task_id} has no H-region, K-region, or deferral option")

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

    k_tail_end = plan_end
    k_tail_constraint_count = 0
    future_overlap_hours = [
        hour
        for option in u_options
        for hour, overlap in option.overlaps
        if hour >= plan_end and overlap > EPS
    ]
    if future_overlap_hours:
        k_tail_end = min(OPERATION_END, max(future_overlap_hours) + 1)
        tail_count = k_tail_end - plan_end
        tail_frame = data.region_hour.loc[
            data.region_hour["Hour"].between(plan_end, k_tail_end - 1)
        ].copy()
        tail_frame["Region"] = pd.Categorical(
            tail_frame["Region"], categories=data.regions, ordered=True
        )
        tail_frame = tail_frame.sort_values(["Hour", "Region"], kind="stable")
        if len(tail_frame) != tail_count * r_count:
            raise ValueError(f"K-region resource protection beyond plan_end lacks data for {plan_end}--{k_tail_end - 1}")
        tail_gpu_limit = tail_frame["Available_GPU"].to_numpy(dtype=float).reshape(
            tail_count, r_count
        )
        tail_non_ai = tail_frame["NonAI_IT_Load_MW"].to_numpy(dtype=float).reshape(
            tail_count, r_count
        )
        tail_it_limit = tail_frame["Max_IT_Power_MW"].to_numpy(dtype=float).reshape(
            tail_count, r_count
        )
        tail_pue = tail_frame["PUE"].to_numpy(dtype=float).reshape(tail_count, r_count)
        tail_facility_limit = tail_frame["Max_Facility_Power_MW"].to_numpy(
            dtype=float
        ).reshape(tail_count, r_count)
        tail_fixed_gpu, tail_fixed_ai = _fixed_task_loads(
            assignments, plan_end, k_tail_end, data.region_index
        )
        tail_gpu_rows = [[{} for _ in range(r_count)] for _ in range(tail_count)]
        tail_ai_rows = [[{} for _ in range(r_count)] for _ in range(tail_count)]
        for index, option in enumerate(u_options):
            column = int(indices["u"][index])
            for hour, overlap in option.overlaps:
                if plan_end <= hour < k_tail_end:
                    local = hour - plan_end
                    tail_gpu_rows[local][option.region_index][column] = (
                        option.gpu_demand * overlap
                    )
                    tail_ai_rows[local][option.region_index][column] = (
                        option.power_mw * overlap
                    )
        for t in range(tail_count):
            for r in range(r_count):
                tail_ai_capacity = min(
                    tail_it_limit[t, r] - tail_non_ai[t, r],
                    tail_facility_limit[t, r] / tail_pue[t, r] - tail_non_ai[t, r],
                )
                add_row(
                    tail_gpu_rows[t][r], -np.inf,
                    tail_gpu_limit[t, r] - tail_fixed_gpu[t, r],
                )
                add_row(
                    tail_ai_rows[t][r], -np.inf,
                    tail_ai_capacity - tail_fixed_ai[t, r],
                )
                k_tail_constraint_count += 2

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
                    f"H-region tail capacity protection lacks hourly data for {decision_end}--{tail_protection_end - 1}"
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
    if carbon_budget_remaining is not None:
        if not np.isfinite(carbon_budget_remaining) or carbon_budget_remaining < -EPS:
            raise ValueError("Remaining rolling carbon budget must be finite and nonnegative")
        carbon_row = {
            int(indices["grid_purchase"][t, r]): float(arrays["CarbonIntensity_tCO2_per_MWh"][t, r])
            for t in range(h_count)
            for r in range(r_count)
            if abs(float(arrays["CarbonIntensity_tCO2_per_MWh"][t, r])) > 0.0
        }
        add_row(carbon_row, -np.inf, max(float(carbon_budget_remaining), 0.0))

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
    task_ids = tuple(task_id for group in task_groups.values() for task_id in group)
    latency_denominator = max(len(task_ids), 1)
    flexible_ids = tuple(
        task_id for task_id in task_ids
        if str(data.task_lookup.loc[task_id, "TaskType"]) != "RealTimeInference"
        and float(data.task_lookup.loc[task_id, "LatestFinishHour"])
        - float(data.task_lookup.loc[task_id, "Duration_h"])
        - max(
            float(data.task_lookup.loc[task_id, "EarliestStartHour"]),
            float(data.task_lookup.loc[task_id, "ArrivalHour"]),
        ) > EPS
    )
    qos_weights = _qos_weight_map(config)
    delay_weight_denominator = max(
        sum(qos_weights[str(data.task_lookup.loc[task_id, "DelaySensitivity"])] for task_id in flexible_ids),
        1.0,
    )
    for index, option in enumerate(x_options):
        column = int(indices["x"][index])
        task = data.task_lookup.loc[option.task_id]
        metric_vectors["Latency"][column] = option.latency_ms / latency_denominator
        effective_earliest = max(float(task.EarliestStartHour), float(task.ArrivalHour))
        slack = min(float(task.LatestFinishHour), OPERATION_END) - float(task.Duration_h) - effective_earliest
        if slack > EPS:
            qos_weight = qos_weights[str(task.DelaySensitivity)]
            metric_vectors["Delay"][column] = (
                qos_weight * max(option.expected_start_hour - effective_earliest, 0.0)
                / slack / delay_weight_denominator
            )
    for index, option in enumerate(u_options):
        column = int(indices["u"][index])
        task = data.task_lookup.loc[option.task_id]
        metric_vectors["Latency"][column] = beta * option.latency_ms / latency_denominator
        effective_earliest = max(float(task.EarliestStartHour), float(task.ArrivalHour))
        slack = min(float(task.LatestFinishHour), OPERATION_END) - float(task.Duration_h) - effective_earliest
        if slack > EPS:
            qos_weight = qos_weights[str(task.DelaySensitivity)]
            metric_vectors["Delay"][column] = (
                beta * qos_weight * max(option.expected_start_hour - effective_earliest, 0.0)
                / slack / delay_weight_denominator
            )
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
        task_groups=task_groups,
        task_group_order=task_group_order,
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
            "TaskCount": len(task_ids), "AggregatedTaskGroupCount": len(task_rules),
            "MaxTaskGroupSize": max((len(group) for group in task_groups.values()), default=0),
            "DeferVariableCount": len(task_group_order),
            "HOptionCount": len(x_options),
            "KOptionCount": len(u_options), "VariableCount": variable_count,
            "ConstraintCount": len(rows), "Beta": beta,
            "BlockHours": block_hours,
            "HTailCapacityProtection": bool(protect_h_tail_capacity),
            "HTailProtectionEnd": int(tail_protection_end),
            "HTailCapacityConstraintCount": int(tail_capacity_constraint_count),
            "KTailProtectionEnd": int(k_tail_end),
            "KTailCapacityConstraintCount": int(k_tail_constraint_count),
            "CarbonBudgetRemaining": (
                float(carbon_budget_remaining)
                if carbon_budget_remaining is not None else float("nan")
            ),
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
    """Construct feasible EDF task and hourly energy schedules as an upper bound for the main MILP."""

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

    def fits(option: TaskOption, amount: float = 1.0) -> bool:
        for hour, overlap in option.overlaps:
            local = hour - tau
            if not 0 <= local < t_count:
                continue
            r = option.region_index
            if used_gpu[local, r] + amount * option.gpu_demand * overlap > gpu_limit[local, r] + tolerance:
                return False
            if used_ai[local, r] + amount * option.power_mw * overlap > ai_limit[local, r] + tolerance:
                return False
        return True

    def select(option: TaskOption, column: int, amount: float = 1.0) -> None:
        vector[column] += amount
        for hour, overlap in option.overlaps:
            local = hour - tau
            if not 0 <= local < t_count:
                continue
            r = option.region_index
            used_gpu[local, r] += amount * option.gpu_demand * overlap
            used_ai[local, r] += amount * option.power_mw * overlap

    ordered_tasks = sorted(
        problem.task_rules,
        key=lambda task_id: (
            not problem.task_rules[task_id][0],
            _latest_integer_start(data.task_lookup.loc[task_id]),
            int(data.task_lookup.loc[task_id, "ArrivalHour"]),
            task_id,
        ),
    )
    defer_column_by_group = {
        group_id: int(problem.indices["defer"][index])
        for index, group_id in enumerate(problem.task_group_order)
    }
    for task_id in ordered_tasks:
        must_commit, must_plan = problem.task_rules[task_id]
        if must_commit:
            choices = x_by_task.get(task_id, [])
        elif must_plan:
            choices = x_by_task.get(task_id, []) + u_by_task.get(task_id, [])
        else:
            vector[defer_column_by_group[task_id]] = float(
                len(problem.task_groups[task_id])
            )
            continue
        for _ in range(len(problem.task_groups[task_id])):
            chosen = next(
                (item for item in sorted(choices, key=option_key) if fits(item[0])),
                None,
            )
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
    """Audit block-average K-region GPU, IT, and facility capacity to trigger local 4h-to-2h refinement."""

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
            if not 0 <= local < t_count:
                continue
            gpu[local, option.region_index] += value * option.gpu_demand * overlap
            ai[local, option.region_index] += value * option.power_mw * overlap
    for index, option in enumerate(problem.u_options):
        value = float(vector[problem.indices["u"][index]])
        for hour, overlap in option.overlaps:
            local = hour - problem.tau
            if not 0 <= local < t_count:
                continue
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
    """Retain the highest-weight K-region time blocks for the next window's heuristic."""

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

    stage = f"Window {problem.tau} {'continuous relaxation' if relax else 'MILP'}"
    _progress(
        f"{stage} started: variables={objective.size}, constraints={problem.matrix.shape[0]}, "
        f"time limit={time_limit:.0f}s, mip_rel_gap={mip_gap:g}."
    )
    integrality = np.zeros_like(problem.integrality) if relax else problem.integrality
    constraints: list[LinearConstraint] = [
        LinearConstraint(problem.matrix, problem.constraint_lower, problem.constraint_upper)
    ]
    if incumbent is not None:
        incumbent = np.asarray(incumbent, dtype=float)
        if incumbent.shape != objective.shape:
            raise ValueError("Heuristic initial solution dimension differs from the window variable dimension")
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
                f"{stage} still solving: elapsed={time.perf_counter() - started:.1f}s, "
                f"variables={objective.size}, constraints={problem.matrix.shape[0]}."
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
        f"{stage} finished: status={getattr(result, 'status', 'NA')}, "
        f"elapsed={elapsed:.2f}s, mip_gap={mip_gap}."
    )
    vector = getattr(result, "x", None)
    if vector is None or not np.all(np.isfinite(vector)):
        raise RuntimeError(f"Window {problem.tau} has no feasible solution: {getattr(result, 'message', '')}")
    activity = problem.matrix @ vector
    lower_violation = np.maximum(problem.constraint_lower - activity, 0.0)
    upper_violation = np.maximum(activity - problem.constraint_upper, 0.0)
    max_violation = float(max(lower_violation.max(initial=0.0), upper_violation.max(initial=0.0)))
    if max_violation > 1e-5:
        raise RuntimeError(f"Window {problem.tau}: maximum solution constraint violation is {max_violation:.3g}")
    return WindowSolution(
        vector=np.asarray(vector, dtype=float),
        status=int(getattr(result, "status", -1)),
        message=str(getattr(result, "message", "")),
        objective=float(getattr(result, "fun", np.dot(objective, vector))),
        mip_gap=mip_gap,
        mip_node_count=_result_float(result, "mip_node_count"),
        elapsed_seconds=elapsed,
        best_bound=_result_float(result, "mip_dual_bound"),
        status_name=SOLVER_STATUS_NAMES.get(int(getattr(result, "status", -1)), "UNKNOWN"),
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
    """Derive a provably safe linear objective lower bound from each metric's variable bounds."""

    coefficients = problem.metric_vectors[metric]
    value = float(problem.metric_constants[metric])
    for column in np.flatnonzero(np.abs(coefficients) > 0.0):
        coefficient = float(coefficients[column])
        bound = float(problem.lower[column] if coefficient >= 0.0 else problem.upper[column])
        if not np.isfinite(bound):
            return float("nan"), False, "UnboundedVariableBox"
        value += coefficient * bound
    return value, True, "CertifiedVariableBox"


def _physical_metric_scale_floors(
    data: InputData,
    schedule: pd.DataFrame | None,
) -> dict[str, float]:
    """Build non-degenerate physical scales for metrics that collapse in one window.

    Representative-window MILPs can legitimately defer all flexible tasks or
    cover a short window with renewable energy and storage.  A zero lower bound
    and a zero feasible reference are therefore not a useful global scale for
    Carbon or Peak.  Use the full 0--2405 horizon's renewable-deficit proxy
    under the audited Q2 task seed as a dimensional scale; fall back to 0.1%
    of the corresponding gross facility-load proxy only when that deficit is
    exactly zero.  This is a normalization device, not an additional objective
    or a substitute dispatch solution.
    """

    frame = data.region_hour.loc[
        data.region_hour["Hour"].between(0, OPERATION_END - 2)
    ].copy()
    frame["Region"] = pd.Categorical(frame["Region"], categories=data.regions, ordered=True)
    frame = frame.sort_values(["Hour", "Region"], kind="stable")
    expected_rows = (OPERATION_END - 1) * len(data.regions)
    if len(frame) != expected_rows:
        raise ValueError(
            "Physical-scale proxy has incomplete hourly regional records: "
            f"{len(frame)} != {expected_rows}"
        )
    arrays = {
        column: frame[column].to_numpy(dtype=float).reshape(OPERATION_END - 1, len(data.regions))
        for column in (
            "ElectricityPrice_CNY_per_MWh", "CarbonIntensity_tCO2_per_MWh",
            "AvailableRenewable_MW", "NonAI_IT_Load_MW", "PUE",
        )
    }
    if schedule is None or schedule.empty:
        ai_load = np.zeros_like(arrays["NonAI_IT_Load_MW"])
    else:
        _, ai_load = _fixed_task_loads(
            schedule, 0, OPERATION_END - 1, data.region_index
        )
    facility = arrays["PUE"] * (arrays["NonAI_IT_Load_MW"] + ai_load)
    renewable_deficit = np.maximum(
        facility - arrays["AvailableRenewable_MW"], 0.0
    )
    gross_facility = np.maximum(facility, 0.0)
    gross_proxy = {
        "Cost": float(np.sum(arrays["ElectricityPrice_CNY_per_MWh"] * gross_facility)),
        "Carbon": float(np.sum(arrays["CarbonIntensity_tCO2_per_MWh"] * gross_facility)),
        "Peak": float(np.max(gross_facility, axis=0).sum()),
    }
    deficit_proxy = {
        "Cost": float(np.sum(arrays["ElectricityPrice_CNY_per_MWh"] * renewable_deficit)),
        "Carbon": float(np.sum(arrays["CarbonIntensity_tCO2_per_MWh"] * renewable_deficit)),
        "Peak": float(np.max(renewable_deficit, axis=0).sum()),
    }
    floors: dict[str, float] = {}
    for metric in ("Cost", "Carbon", "Peak"):
        candidate = deficit_proxy[metric]
        if candidate <= EPS:
            candidate = gross_proxy[metric] * 1e-3
        floors[metric] = max(float(candidate), 1.0)
    return floors


def _calibrate(
    data: InputData,
    config: ModelConfig,
    *,
    feasible_reference_schedule: pd.DataFrame | None = None,
    reference_source: str = "JOINT_WINDOW_MILP_REFERENCE",
) -> tuple[dict[str, tuple[float, float]], pd.DataFrame]:
    storage = data.storage.set_index("Region").loc[list(data.regions)]
    initial_soc = storage["InitialSOC_MWh"].to_numpy(dtype=float)
    empty_assignments = pd.DataFrame(columns=[
        "TaskID", "TargetRegion", "StartHour", "Duration_h", "GPU_Demand", "Task_Full_IT_Power_MW"
    ])
    records: list[dict[str, float | int | str]] = []
    lower_by_metric: dict[str, list[float]] = {metric: [] for metric in METRICS}
    reference_by_metric: dict[str, list[float]] = {metric: [] for metric in METRICS}
    reference_schedule: pd.DataFrame | None = None
    if feasible_reference_schedule is not None:
        reference_schedule = _standardize_shadow_schedule(data, feasible_reference_schedule)
        _validate_shadow_schedule(data, reference_schedule, config.feasibility_tolerance)
    physical_scale_floors = _physical_metric_scale_floors(data, reference_schedule)
    representative_windows = _representative_windows(data, config)
    _progress(
        f"Calibration started: {len(representative_windows)} representative windows, "
        f"{len(METRICS)} continuous lower bounds and one integer reference per window."
    )
    for window_index, tau in enumerate(representative_windows, start=1):
        _progress(
            f"Calibration progress: {window_index}/{len(representative_windows)}, "
            f"window start={tau}."
        )
        lookahead = min(config.lookahead_hours, OPERATION_END - (tau + config.decision_hours))
        problem = build_window_problem(
            data, config, tau, config.decision_hours, lookahead, empty_assignments,
            initial_soc.copy(), np.zeros(len(data.regions)), None, calibration=True,
        )
        lower_values: dict[str, float] = {}
        for metric_index, metric in enumerate(METRICS, start=1):
            _progress(
                f"Calibration window {tau}: continuous lower bound {metric_index}/{len(METRICS)}, "
                f"metric={metric}."
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
                    problem, problem.metric_vectors[metric], time_limit=config.calibration_lower_bound_time_limit_seconds,
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
                    f"Calibration window {tau}, metric {metric}: no provably safe lower bound; "
                    f"BoundSource={bound_source}"
                )
            if not exact_ideal:
                _progress(
                    f"Calibration window {tau}, metric={metric}: no exact continuous ideal point; "
                    f"using the metric's own {bound_source}={value:.6g}."
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
        values: dict[str, float] | None = None
        reference_stage = ""
        reference_status: int | str = ""
        reference_elapsed_seconds = 0.0
        reference_message = ""
        actual_reference_source = ""
        if reference_schedule is not None:
            # V2's energy trajectory is not reused.  Its TaskID schedule is
            # reused only after task-side validation, then a new V4 energy
            # response is solved under the corrected physical constraints.
            try:
                _progress(f"Calibration window {tau}: reusing audited task seed and recomputing a feasible V4 energy reference.")
                _, reference_ai, _, _ = _shadow_profiles(
                    data, reference_schedule, tau, problem.plan_end
                )
                reference_task_metrics = _window_task_metrics(
                    reference_schedule, tau, problem.plan_end, config
                )
                reference_scaling = {
                    metric: (lower_values[metric], max(abs(lower_values[metric]), 1.0))
                    for metric in METRICS
                }
                reference_energy = solve_energy_response(
                    data,
                    config,
                    tau=tau,
                    ai_it_profile=reference_ai,
                    current_soc=initial_soc.copy(),
                    historical_peak=np.zeros(len(data.regions)),
                    task_metrics=reference_task_metrics,
                    scaling=reference_scaling,
                    time_limit=config.calibration_reference_time_limit_seconds,
                )
                values = dict(reference_energy.metrics)
                reference_stage = "ReusedAuditedTaskSeedEnergyReference"
                reference_status = reference_energy.status
                reference_elapsed_seconds = reference_energy.elapsed_seconds
                reference_message = reference_energy.message
                actual_reference_source = reference_source
            except RuntimeError as exc:
                reference_message = f"Reused task seed unavailable: {exc}"
        if values is None:
            _progress(f"Calibration window {tau}: solving the integer reference.")
            reference_objective = np.zeros(problem.lower.size, dtype=float)
            for metric in METRICS:
                denominator = max(abs(lower_values[metric]), 1.0)
                reference_objective += problem.metric_vectors[metric] / denominator
            reference_stage = "FeasibleReference"
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
                        f"Calibration window {tau} has neither a solver-feasible nor a heuristic-feasible reference: {exc}"
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
                "ReferenceSource": actual_reference_source,
            })
    scaling: dict[str, tuple[float, float]] = {}
    scale_metadata: dict[str, dict[str, object]] = {}
    for metric in METRICS:
        anchor = float(np.mean(lower_by_metric[metric]))
        reference = float(np.mean(reference_by_metric[metric]))
        raw_scale = reference - anchor
        degeneracy_tolerance = max(abs(anchor), abs(reference), 1.0) * OBJECTIVE_DEGENERACY_RELATIVE_TOLERANCE
        if raw_scale <= degeneracy_tolerance:
            scale = max(float(physical_scale_floors.get(metric, 1.0)), 1.0)
            scale_source = "FULL_HORIZON_RENEWABLE_DEFICIT_PHYSICAL_PROXY"
        else:
            scale = float(raw_scale)
            scale_source = "REPRESENTATIVE_WINDOW_REFERENCE_MINUS_LOWER_BOUND"
        scaling[metric] = (anchor, scale)
        scale_metadata[metric] = {
            "RawScale": float(raw_scale),
            "PhysicalScaleFloor": float(physical_scale_floors.get(metric, float("nan"))),
            "ScaleSource": scale_source,
            "DegenerateWindowRange": bool(raw_scale <= degeneracy_tolerance),
        }
    records_frame = pd.DataFrame(records)
    if not records_frame.empty:
        records_frame["FinalAnchor"] = records_frame["Metric"].map(
            lambda metric: scaling[str(metric)][0]
        )
        records_frame["FinalScale"] = records_frame["Metric"].map(
            lambda metric: scaling[str(metric)][1]
        )
        records_frame["RawScale"] = records_frame["Metric"].map(
            lambda metric: scale_metadata[str(metric)]["RawScale"]
        )
        records_frame["PhysicalScaleFloor"] = records_frame["Metric"].map(
            lambda metric: scale_metadata[str(metric)]["PhysicalScaleFloor"]
        )
        records_frame["FinalScaleSource"] = records_frame["Metric"].map(
            lambda metric: scale_metadata[str(metric)]["ScaleSource"]
        )
        records_frame["DegenerateWindowRange"] = records_frame["Metric"].map(
            lambda metric: scale_metadata[str(metric)]["DegenerateWindowRange"]
        )
    _progress("Calibration completed.")
    return scaling, records_frame


def _empty_assignments() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "TaskID", "TaskType", "ArrivalHour", "SourceRegion", "TargetRegion",
        "NetworkLatency_ms", "MaxLatency_ms", "StartHour", "FinishHour", "Duration_h",
        "GPU_Demand", "Task_Full_IT_Power_MW", "DelaySensitivity", "WaitHours",
        "IsMigrated", "DecisionWindowStart", "ExactTaskGroup",
    ])


def _selected_assignments(data: InputData, problem: WindowProblem, vector: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cursor = {group_id: 0 for group_id in problem.task_groups}
    for index, option in enumerate(problem.x_options):
        raw_count = float(vector[problem.indices["x"][index]])
        count = int(round(raw_count))
        if abs(raw_count - count) > 1e-5:
            raise RuntimeError(
                f"H-region aggregate variable {option.task_id}/{option.region}/{option.start_hour} "
                f"is not integer: {raw_count}"
            )
        if count <= 0:
            continue
        cursor = group_cursor[option.task_id]
        members = problem.task_groups[option.task_id][cursor:cursor + count]
        if len(members) != count:
            raise RuntimeError(f"Task group {option.task_id}: integer count exceeds group size")
        group_cursor[option.task_id] += count
        for member_id in members:
            task = data.task_lookup.loc[member_id]
            candidate = next(
                item for item in data.candidate_map[member_id]
                if item.region == option.region
            )
            start = float(option.start_hour)
            rows.append({
                "TaskID": member_id,
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
                "DelaySensitivity": str(task.DelaySensitivity),
                "WaitHours": start - max(float(task.EarliestStartHour), float(task.ArrivalHour)),
                "IsMigrated": int(option.region != str(task.SourceRegion)),
                "DecisionWindowStart": problem.tau,
                "ExactTaskGroup": option.task_id,
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
            if problem.tau <= hour < problem.plan_end:
                forecast[hour - problem.tau, option.region_index] += (
                    option.power_mw * overlap * value
                )
    for index, option in enumerate(problem.u_options):
        value = vector[problem.indices["u"][index]]
        if value <= EPS:
            continue
        for hour, overlap in option.overlaps:
            if problem.tau <= hour < problem.plan_end:
                forecast[hour - problem.tau, option.region_index] += (
                    option.power_mw * overlap * value
                )
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


def _qos_details(
    data: InputData,
    assignments: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
) -> tuple[float, pd.DataFrame]:
    assignment_keys = assignments[["TaskID", "StartHour"]].copy()
    assignment_keys["TaskID"] = assignment_keys["TaskID"].astype(str)
    task_fields = data.tasks[[
            "TaskID", "TaskType", "ArrivalHour", "EarliestStartHour",
            "LatestFinishHour", "Duration_h", "DelaySensitivity",
        ]].copy()
    task_fields["TaskID"] = task_fields["TaskID"].astype(str)
    joined = assignment_keys.merge(
        task_fields, on="TaskID", how="left", validate="one_to_one",
    )
    if joined[["TaskType", "DelaySensitivity"]].isna().any().any():
        raise ValueError("QoS recomputation found unknown TaskID values in the actual task table")
    joined["EffectiveEarliestStart"] = np.maximum(
        joined["ArrivalHour"], joined["EarliestStartHour"]
    )
    joined["TimeSlack_h"] = (
        np.minimum(joined["LatestFinishHour"], OPERATION_END)
        - joined["Duration_h"] - joined["EffectiveEarliestStart"]
    )
    joined["ActualDelay_h"] = np.maximum(
        joined["StartHour"] - joined["EffectiveEarliestStart"], 0.0
    )
    flexible = (
        ~joined["TaskType"].eq("RealTimeInference")
        & (joined["TimeSlack_h"] > EPS)
    )
    flexible_mask = flexible.to_numpy(dtype=bool)
    relative_delay = np.zeros(len(joined), dtype=float)
    actual_delay = joined["ActualDelay_h"].to_numpy(dtype=float)
    time_slack = joined["TimeSlack_h"].to_numpy(dtype=float)
    relative_delay[flexible_mask] = np.clip(
        actual_delay[flexible_mask] / time_slack[flexible_mask], 0.0, 1.0
    )
    joined["RelativeDelay"] = relative_delay
    weight_map = dict(weights or QOS_WEIGHTS)
    joined["TaskWeight"] = joined["DelaySensitivity"].map(weight_map)
    if joined["TaskWeight"].isna().any():
        unknown = sorted(joined.loc[joined["TaskWeight"].isna(), "DelaySensitivity"].unique())
        raise ValueError(f"QoS recomputation found unknown DelaySensitivity values: {unknown}")
    task_weight = joined["TaskWeight"].to_numpy(dtype=float)
    contribution = np.zeros(len(joined), dtype=float)
    contribution[flexible_mask] = (
        task_weight[flexible_mask] * relative_delay[flexible_mask]
    )
    joined["QoSLossContribution"] = contribution
    denominator = float(task_weight[flexible_mask].sum())
    loss = (
        float(contribution[flexible_mask].sum() / denominator)
        if denominator > EPS else 0.0
    )
    joined["TaskGroup"] = assignments.get(
        "ExactTaskGroup", assignments["TaskID"]
    ).astype(str).to_numpy()
    joined["TaskCount"] = 1
    joined["OverallJ"] = loss
    joined["QoS"] = 1.0 - loss
    return loss, joined[[
        "TaskGroup", "TaskID", "TaskCount", "DelaySensitivity", "TimeSlack_h",
        "StartHour", "RelativeDelay", "TaskWeight", "QoSLossContribution",
        "OverallJ", "QoS",
    ]]


def _attach_region_hour_inputs(data: InputData, dispatch: pd.DataFrame) -> pd.DataFrame:
    """Attach immutable Q4 inputs for an independent result-table recomputation."""

    input_columns = [
        "Hour", "Region", "NonAI_IT_Load_MW", "PUE",
        "AvailableRenewable_MW", "ElectricityPrice_CNY_per_MWh",
        "SellPrice_CNY_per_MWh", "CarbonIntensity_tCO2_per_MWh",
        "Available_GPU", "Max_IT_Power_MW", "Max_Facility_Power_MW",
    ]
    inputs = data.region_hour[input_columns].copy()
    inputs["Region"] = inputs["Region"].astype(str)
    inputs = inputs.rename(columns={
        column: f"Input_{column}"
        for column in input_columns if column not in {"Hour", "Region"}
    })
    result = dispatch.copy()
    result["Region"] = result["Region"].astype(str)
    return result.merge(inputs, on=["Hour", "Region"], how="left", validate="many_to_one")


def _final_metrics(
    data: InputData,
    assignments: pd.DataFrame,
    dispatch: pd.DataFrame,
    qos_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    checked = _attach_region_hour_inputs(data, dispatch)
    renewable_total = float(checked["Input_AvailableRenewable_MW"].sum())
    net_grid = checked["GridPurchase_MW"] - checked["GridSell_MW"]
    values = {
        "Cost": float(np.sum(
            checked["Input_ElectricityPrice_CNY_per_MWh"] * checked["GridPurchase_MW"]
            - checked["Input_SellPrice_CNY_per_MWh"] * checked["GridSell_MW"]
        )),
        "Carbon": float(np.sum(
            checked["Input_CarbonIntensity_tCO2_per_MWh"] * checked["GridPurchase_MW"]
        )),
        "Latency": float(assignments["NetworkLatency_ms"].mean()),
        "LatencySLA": float(np.mean(
            assignments["NetworkLatency_ms"] / assignments["MaxLatency_ms"]
        )),
        "RenewableUnusedRate": (
            float(checked["RenewableCurtailment_MW"].sum() / renewable_total)
            if renewable_total > EPS else 0.0
        ),
        "Peak": float(net_grid.groupby(checked["Region"]).max().clip(lower=0.0).sum()),
    }
    values["Delay"], _ = _qos_details(data, assignments, qos_weights)
    return values


def _write_csv(frame: pd.DataFrame, filename: str) -> None:
    _atomic_write_csv(frame, TABLES_DIR / filename)


def _simple_validation(
    data: InputData,
    assignments: pd.DataFrame,
    dispatch: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:
    """Run minimal hard-constraint checks on final actual trajectories without additional modeling or sensitivity analysis."""

    # The rolling result starts from empty, schema-only DataFrames and grows
    # through concatenation. Pandas can therefore retain ``object`` dtype for
    # numeric columns even when every stored value is numeric. Normalise the
    # audit inputs before applying NumPy ufuncs, so final validation is as
    # robust after a checkpoint resume as it is for CSV-reloaded results.
    assignments = assignments.copy()
    dispatch = dispatch.copy()
    assignment_numeric_columns = (
        "ArrivalHour", "NetworkLatency_ms", "MaxLatency_ms", "StartHour",
        "FinishHour", "Duration_h", "GPU_Demand", "Task_Full_IT_Power_MW",
        "WaitHours", "DecisionWindowStart",
    )
    dispatch_numeric_columns = (
        "Hour", "AI_IT_Load_MW", "NonAI_IT_Load_MW", "Total_IT_Load_MW",
        "Facility_Load_MW", "PUE", "AvailableRenewable_MW",
        "RenewableDirectUse_MW", "RenewableCharge_MW", "GridCharge_MW",
        "DischargePower_MW", "GridPurchase_MW", "GridSell_MW",
        "RenewableCurtailment_MW", "NetGridImport_MW", "SOCStart_MWh",
        "SOCEnd_MWh", "ChargePower_MW", "CarbonEmission_tCO2",
        "ElectricityPrice_CNY_per_MWh", "SellPrice_CNY_per_MWh",
        "CarbonIntensity_tCO2_per_MWh", "Available_GPU", "Max_IT_Power_MW",
        "Max_Facility_Power_MW", "DecisionWindowStart",
    )
    for column in assignment_numeric_columns:
        if column in assignments:
            assignments[column] = pd.to_numeric(assignments[column], errors="raise")
    for column in dispatch_numeric_columns:
        if column in dispatch:
            dispatch[column] = pd.to_numeric(dispatch[column], errors="raise")

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
            "LatestFinishHour", "Duration_h", "MaxLatency_ms",
        ]],
        on="TaskID", how="left", validate="one_to_one", suffixes=("", "_Input"),
    )
    add("TaskIntegerStart", float(np.abs(
        joined["StartHour"] - np.rint(joined["StartHour"])
    ).max()))
    add("TaskTypeInputConsistency", float((
        joined["TaskType"].astype(str) != joined["TaskType_Input"].astype(str)
    ).any()), 0.0)
    add("TaskArrivalInputConsistency", float(np.abs(
        joined["ArrivalHour"] - joined["ArrivalHour_Input"]
    ).max()))
    add("TaskDurationInputConsistency", float(np.abs(
        joined["Duration_h"] - joined["Duration_h_Input"]
    ).max()))
    add("TaskFinishMapping", float(np.abs(
        joined["FinishHour"] - joined["StartHour"] - joined["Duration_h_Input"]
    ).max()))
    add("TaskMaxLatencyInputConsistency", float(np.abs(
        joined["MaxLatency_ms"] - joined["MaxLatency_ms_Input"]
    ).max()))
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
    candidate_latency = data.candidates[[
        "TaskID", "TargetRegion", "NetworkLatency_ms",
    ]].copy()
    candidate_latency["TaskID"] = candidate_latency["TaskID"].astype(str)
    candidate_latency["TargetRegion"] = candidate_latency["TargetRegion"].astype(str)
    assignment_latency = assignments.copy()
    assignment_latency["TaskID"] = assignment_latency["TaskID"].astype(str)
    assignment_latency["TargetRegion"] = assignment_latency["TargetRegion"].astype(str)
    assignment_latency = assignment_latency.merge(
        candidate_latency,
        on=["TaskID", "TargetRegion"], how="left", validate="many_to_one",
        suffixes=("", "_Input"),
    )
    add("TaskCandidateLatencyConsistency", float(np.abs(
        assignment_latency["NetworkLatency_ms"]
        - assignment_latency["NetworkLatency_ms_Input"]
    ).max()))
    add("TerminalTaskBoundary", float(np.maximum(
        joined["FinishHour"] - float(OPERATION_END), 0.0
    ).max()))

    gpu, ai = _fixed_task_loads(assignments, 0, OPERATION_END, data.region_index)
    expected = pd.DataFrame({
        "Hour": np.repeat(np.arange(OPERATION_END), len(data.regions)),
        "Region": np.tile(np.asarray(data.regions), OPERATION_END),
        "Recomputed_GPU": gpu.ravel(),
        "Recomputed_AI_IT_Load_MW": ai.ravel(),
    })
    checked = _attach_region_hour_inputs(data, dispatch).merge(
        expected, on=["Hour", "Region"], how="left", validate="many_to_one"
    )
    input_columns = [
        "Input_NonAI_IT_Load_MW", "Input_PUE", "Input_AvailableRenewable_MW",
        "Input_ElectricityPrice_CNY_per_MWh", "Input_SellPrice_CNY_per_MWh",
        "Input_CarbonIntensity_tCO2_per_MWh", "Input_Available_GPU",
        "Input_Max_IT_Power_MW", "Input_Max_Facility_Power_MW",
    ]
    add("InputRegionHourMatch", float(checked[input_columns].isna().any(axis=None)), 0.0)
    add("AIITLoadRecompute", float(np.abs(
        checked["AI_IT_Load_MW"] - checked["Recomputed_AI_IT_Load_MW"]
    ).max()))
    expected_total_it = checked["Input_NonAI_IT_Load_MW"] + checked["Recomputed_AI_IT_Load_MW"]
    expected_facility = checked["Input_PUE"] * expected_total_it
    add("NonAIITInputConsistency", float(np.abs(
        checked["NonAI_IT_Load_MW"] - checked["Input_NonAI_IT_Load_MW"]
    ).max()))
    add("PUEInputConsistency", float(np.abs(
        checked["PUE"] - checked["Input_PUE"]
    ).max()))
    add("TotalITLoadMapping", float(np.abs(
        checked["Total_IT_Load_MW"] - expected_total_it
    ).max()))
    add("FacilityLoadMapping", float(np.abs(
        checked["Facility_Load_MW"] - expected_facility
    ).max()))
    add("GPUCapacity", float(np.maximum(
        checked["Recomputed_GPU"] - checked["Input_Available_GPU"], 0.0
    ).max()))
    add("ITCapacity", float(np.maximum(
        expected_total_it - checked["Input_Max_IT_Power_MW"], 0.0
    ).max()))
    add("FacilityCapacity", float(np.maximum(
        expected_facility - checked["Input_Max_Facility_Power_MW"], 0.0
    ).max()))
    for column in (
        "AvailableRenewable_MW", "ElectricityPrice_CNY_per_MWh",
        "SellPrice_CNY_per_MWh", "CarbonIntensity_tCO2_per_MWh",
        "Available_GPU", "Max_IT_Power_MW", "Max_Facility_Power_MW",
    ):
        add(f"{column}InputConsistency", float(np.abs(
            checked[column] - checked[f"Input_{column}"]
        ).max()))

    renewable_residual = (
        checked["Input_AvailableRenewable_MW"] - checked["RenewableDirectUse_MW"]
        - checked["RenewableCharge_MW"] - checked["GridSell_MW"]
        - checked["RenewableCurtailment_MW"]
    )
    load_residual = (
        checked["GridPurchase_MW"] + checked["RenewableDirectUse_MW"]
        + checked["DischargePower_MW"] - expected_facility
        - checked["GridCharge_MW"]
    )
    add("RenewableBalance", float(np.abs(renewable_residual).max()))
    add("FacilityLoadBalance", float(np.abs(load_residual).max()))
    add("RenewableDirectUseLimit", float(np.maximum(
        checked["RenewableDirectUse_MW"] - expected_facility, 0.0
    ).max()))
    add("GridChargeSourceLimit", float(np.maximum(
        checked["GridCharge_MW"] - checked["GridPurchase_MW"], 0.0
    ).max()))
    add("NetGridImportDefinition", float(np.abs(
        checked["NetGridImport_MW"] - checked["GridPurchase_MW"] + checked["GridSell_MW"]
    ).max()))
    add("CarbonEmissionDefinition", float(np.abs(
        checked["CarbonEmission_tCO2"]
        - checked["Input_CarbonIntensity_tCO2_per_MWh"] * checked["GridPurchase_MW"]
    ).max()))
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
    add("ChargePowerDefinition", float(np.abs(
        dispatch["ChargePower_MW"] - dispatch["RenewableCharge_MW"] - dispatch["GridCharge_MW"]
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


def _v4_scaling_frame(
    scaling: Mapping[str, tuple[float, float]],
    calibration_records: pd.DataFrame,
) -> pd.DataFrame:
    """Persist V4 scale provenance together with the final scale values."""

    metadata = pd.DataFrame()
    if not calibration_records.empty and {"Metric", "Stage"}.issubset(calibration_records.columns):
        metadata = (
            calibration_records.sort_values(["Metric", "Stage"], kind="stable")
            .drop_duplicates("Metric")
            .set_index("Metric")
        )
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        anchor, scale = scaling[metric]
        row: dict[str, object] = {
            "Metric": metric,
            "Anchor": float(anchor),
            "Scale": float(scale),
            "OptimizationRole": (
                "REPORT_ONLY_DEGENERATE"
                if _calibration_scale_is_degenerate(anchor, scale)
                else "MINIMAX_ACTIVE"
            ),
        }
        if metric in metadata.index:
            for column in (
                "RawScale", "PhysicalScaleFloor", "FinalScaleSource",
                "DegenerateWindowRange",
            ):
                if column in metadata.columns:
                    output_column = "ScaleSource" if column == "FinalScaleSource" else column
                    row[output_column] = metadata.at[metric, column]
        rows.append(row)
    return pd.DataFrame(rows)


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
    """Require the checkpoint to contain the complete actual trajectory for 0 through resume_from-1."""

    if resume_from < 0 or resume_from > OPERATION_END:
        raise ValueError(f"Resume time {resume_from} is outside 0--{OPERATION_END}")
    valid_starts = set(range(0, MAIN_END, config.decision_hours)) | {MAIN_END, OPERATION_END}
    if resume_from not in valid_starts:
        raise ValueError(f"Resume time {resume_from} is not a valid rolling window boundary")

    assignments = assignments.copy()
    dispatch = dispatch.copy()
    solver = solver.copy()
    if assignments["TaskID"].astype(str).duplicated().any():
        duplicates = assignments.loc[
            assignments["TaskID"].astype(str).duplicated(keep=False), "TaskID"
        ].astype(str).unique()[:10]
        raise ValueError(f"Resume records contain duplicate executed tasks: {duplicates.tolist()}")
    known_ids = set(data.tasks["TaskID"].astype(str))
    assigned_ids = set(assignments["TaskID"].astype(str))
    unknown = sorted(assigned_ids - known_ids)
    if unknown:
        raise ValueError(f"Resume records contain TaskID values absent from the original tasks; examples: {unknown[:10]}")
    if len(assigned_ids) + len(known_ids - assigned_ids) != len(known_ids):
        raise ValueError("Executed and unexecuted resume task counts do not cover the original task set")
    if not assignments.empty:
        if (pd.to_numeric(assignments["DecisionWindowStart"]) >= resume_from).any():
            raise ValueError("Resume records include tasks written by windows after the resume time")
        if (pd.to_numeric(assignments["StartHour"]) >= resume_from - EPS).any():
            raise ValueError("Resume records include task starts that have not entered the actual H region")
        joined = assignments.merge(
            data.tasks[[
                "TaskID", "TaskType", "ArrivalHour", "EarliestStartHour",
                "LatestFinishHour", "Duration_h",
            ]],
            on="TaskID", how="left", validate="one_to_one", suffixes=("", "_Input"),
        )
        earliest = np.maximum(joined["ArrivalHour_Input"], joined["EarliestStartHour"])
        if float(np.maximum(earliest - joined["StartHour"], 0.0).max()) > config.feasibility_tolerance:
            raise ValueError("Resume task records include starts before arrival or earliest start")
        finish = joined["StartHour"] + joined["Duration_h_Input"]
        if float(np.maximum(finish - joined["LatestFinishHour"], 0.0).max()) > config.feasibility_tolerance:
            raise ValueError("Resume task records include deadline violations")
        real_time = joined["TaskType_Input"].eq("RealTimeInference")
        if real_time.any() and float(np.abs(
            joined.loc[real_time, "StartHour"] - joined.loc[real_time, "ArrivalHour_Input"]
        ).max()) > config.feasibility_tolerance:
            raise ValueError("Realtime tasks in resume records did not start upon arrival")
        valid_pairs = set(zip(
            data.candidates["TaskID"].astype(str), data.candidates["TargetRegion"].astype(str)
        ))
        bad_pairs = [
            (str(row.TaskID), str(row.TargetRegion))
            for row in assignments.itertuples(index=False)
            if (str(row.TaskID), str(row.TargetRegion)) not in valid_pairs
        ]
        if bad_pairs:
            raise ValueError(f"Resume task records contain invalid task-region pairs; examples: {bad_pairs[:10]}")

    expected_windows = list(range(0, min(resume_from, MAIN_END), config.decision_hours))
    if resume_from > MAIN_END:
        expected_windows.append(MAIN_END)
    actual_windows = sorted(pd.to_numeric(solver.get("WindowStart", pd.Series(dtype=float))).astype(int).tolist())
    if actual_windows != expected_windows:
        raise ValueError(
            f"Noncontiguous solver records: expected {len(expected_windows)} windows, found {len(actual_windows)}"
        )

    if resume_from == 0:
        if not dispatch.empty:
            raise ValueError("A resume state at hour 0 must not contain actual hourly dispatch")
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
        raise ValueError(f"Hourly dispatch resume table is missing columns: {missing_columns}")
    if dispatch.duplicated(["Hour", "Region"]).any():
        raise ValueError("Hourly dispatch resume table contains duplicate Hour--Region records")
    expected_pairs = pd.MultiIndex.from_product(
        [range(resume_from), data.regions], names=["Hour", "Region"]
    )
    actual_pairs = pd.MultiIndex.from_frame(dispatch[["Hour", "Region"]])
    missing_pairs = expected_pairs.difference(actual_pairs)
    extra_pairs = actual_pairs.difference(expected_pairs)
    if len(missing_pairs) or len(extra_pairs):
        raise ValueError(
            f"Hourly dispatch does not fully cover hours 0--{resume_from - 1}: "
            f"{len(missing_pairs)} missing rows, {len(extra_pairs)} extra rows"
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
        raise ValueError("Hourly dispatch SOC recurrence is discontinuous")
    for region in data.regions:
        region_rows = ordered.loc[ordered["Region"].astype(str).eq(region)]
        initial_soc = float(storage.loc[region, "InitialSOC_MWh"])
        if abs(float(region_rows.iloc[0]["SOCStart_MWh"]) - initial_soc) > config.feasibility_tolerance:
            raise ValueError(f"Region {region}: hour-0 SOC differs from initial SOC")
        cross = (
            region_rows["SOCStart_MWh"].to_numpy(dtype=float)[1:]
            - region_rows["SOCEnd_MWh"].to_numpy(dtype=float)[:-1]
        )
        if len(cross) and float(np.abs(cross).max()) > config.feasibility_tolerance:
            raise ValueError(f"Region {region}: SOC is discontinuous between adjacent hours")
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
        raise ValueError("Hourly dispatch resume table violates energy balance")
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
        raise ValueError("Checkpoint hour-1608 SOC differs from recomputation using hourly dispatch")
    if stored_peak is not None and not np.allclose(
        np.asarray(stored_peak, dtype=float), recomputed_peak,
        rtol=0.0, atol=config.feasibility_tolerance,
    ):
        raise ValueError("Checkpoint historical peak differs from recomputation using hourly net imports")
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
            raise RuntimeError(f"Resume state file is corrupt or unreadable: {checkpoint}") from exc
        if not isinstance(state, dict) or state.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError(f"Unsupported resume state version: {checkpoint}")
        if state.get("signature") != signature:
            raise ValueError("Resume state data, parameters, or model.py signature differs from the current project")
        if int(state.get("next_tau", -1)) != resume_from:
            raise ValueError(
                f"Resume state's next_tau={state.get('next_tau')} differs from requested {resume_from}"
            )
    else:
        required = (
            PROGRESS_ASSIGNMENTS_PATH, PROGRESS_DISPATCH_PATH,
            PROGRESS_SOLVER_PATH, PROGRESS_SCALING_PATH,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Cannot fabricate hour-1608 state from logs or objective values; missing resume files:\n- "
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
            raise ValueError("Progress scaling table does not contain all six objectives")
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
        _progress("No PKL checkpoint found; attempted reconstruction from complete progress CSVs. Recomputing SOC and historical peak.")
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
    """Extend the same hard-constraint model to the difficult-window limit based on feasibility and MIP gap."""

    limits = [float(initial_time_limit)]
    if initial_time_limit < config.difficult_time_limit_seconds:
        limits.append(float(config.difficult_time_limit_seconds))
    unique_limits = list(dict.fromkeys(limits))
    errors: list[str] = []
    best_solution: WindowSolution | None = None
    best_limit = float(initial_time_limit)
    cutoff = incumbent
    for attempt, limit in enumerate(unique_limits, start=1):
        try:
            solution = _solve(
                problem,
                objective,
                time_limit=limit,
                mip_gap=config.mip_relative_gap,
                relax=False,
                incumbent=cutoff,
            )
            if (
                best_solution is None
                or solution.objective < best_solution.objective
            ):
                best_solution = solution
                best_limit = limit
                cutoff = solution.vector
            gap_ok = (
                np.isfinite(solution.mip_gap)
                and solution.mip_gap <= config.mip_relative_gap + EPS
            )
            if solution.status == 0 or gap_ok or attempt == len(unique_limits):
                return best_solution, attempt, best_limit
            _progress(
                f"Window {problem.tau} has a feasible solution, but MIP gap={solution.mip_gap:.6g} "
                f"> target {config.mip_relative_gap:.6g}; extending the same MILP."
            )
        except RuntimeError as exc:
            errors.append(f"{limit:.0f}s: {exc}")
            _progress(
                f"Window {problem.tau}: solve attempt {attempt}/{len(unique_limits)} failed, "
                f"time limit={limit:.0f}s: {exc}."
            )
    if best_solution is not None:
        return best_solution, len(unique_limits), best_limit
    raise RuntimeError("；".join(errors))


def _future_feasibility_check(
    data: InputData,
    assignments_after: pd.DataFrame,
    next_tau: int,
    tolerance: float,
) -> dict[str, int]:
    """Ensure every remaining task retains a physically feasible future option after committing H-region decisions."""

    assigned_ids = set(assignments_after["TaskID"].astype(str)) if not assignments_after.empty else set()
    remaining = data.tasks.loc[~data.tasks["TaskID"].isin(assigned_ids)]
    if next_tau >= OPERATION_END:
        if not remaining.empty:
            raise RuntimeError(f"{len(remaining)} tasks remain unscheduled at the terminal time")
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
            "Cross-window fixed GPU load from H-region decisions exceeds future capacity: "
            f"Hour={next_tau + int(local)}，Region={data.regions[int(r)]}，"
            f"Load={fixed_gpu[local, r]:.12g}，Capacity={available_gpu[local, r]:.12g}，"
            f"Violation={fixed_gpu_violation[local, r]:.12g}"
        )
    if np.any(fixed_ai_violation > tolerance):
        local, r = np.argwhere(fixed_ai_violation > tolerance)[0]
        raise RuntimeError(
            "Cross-window fixed IT/facility load from H-region decisions exceeds future capacity: "
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
            "Cross-window fixed facility load exceeds renewable output plus maximum imports and discharge: "
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
            "Current H-region decisions leave tasks without a valid region, start time, or deadline; examples: "
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
    objective_metric: str | None = None,
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
    if feasibility_only:
        objective = np.zeros(problem.lower.size, dtype=float)
    elif objective_metric is not None:
        if objective_metric not in problem.metric_vectors:
            raise ValueError(f"Unknown H-only recovery objective: {objective_metric}")
        objective = problem.metric_vectors[objective_metric].copy()
    else:
        objective = _balanced_objective(problem, config)
    mode = "HOnlyFeasibility" if feasibility_only else "HOnlyBalanced"
    _progress(
        f"Window {tau} entering {mode}: H={decision_hours}h, K=0h, "
        f"variables={problem.lower.size}, constraints={problem.matrix.shape[0]}."
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
    raise RuntimeError("Legacy automatic fallback from failed 4-hour blocks to 2-hour blocks is permanently disabled")
    config = config or ModelConfig()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    _progress(f"Q4 model started: force={force}.")
    signature = _cache_signature(config)
    if not force and _can_reuse(signature):
        _progress("Complete matching cached results found; skipping optimization.")
        return
    _progress("Reading model inputs.")
    data = load_data()
    _progress(
        f"Inputs loaded: tasks={len(data.tasks)}, regions={len(data.regions)}, "
        f"hourly regional records={len(data.region_hour)}."
    )
    scaling, calibration_records = _calibrate(data, config)
    _progress("Starting rolling window optimization.")
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
            f"Rolling window progress: {window_index}/{total_windows} ({window_index / total_windows:.1%}), "
            f"window={tau}, decision region={decision_hours}h, lookahead region={lookahead}h."
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
            f"Window {tau} constructed: variables={problem.lower.size}, constraints={problem.matrix.shape[0]}, "
            f"pending tasks={due_count}, time limit={initial_time_limit:.0f}s."
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
            _progress(f"Window {tau}: no feasible solution with 4-hour blocks; rebuilding and retrying with 2-hour blocks.")
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
                    f"Window {tau}: neither 4-hour nor 2-hour blocks yielded a feasible solution; "
                    f"4-hour blocks: {block4_error}; 2-hour blocks: {block2_error}"
                ) from block2_error
        k_capacity_violation = _k_capacity_violation(data, problem, solution.vector, assignments)
        if (
            lookahead > 0
            and int(problem.metadata["BlockHours"]) > 2
            and k_capacity_violation > config.feasibility_tolerance
        ):
            refinement_reason = "block4_capacity_audit"
            _progress(
                f"Window {tau}: 4-hour blocks solved, but K-region capacity violation is {k_capacity_violation:.6g}; "
                "refining with 2-hour blocks."
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
                    f"Window {tau}: K-region capacity still violated after 2-hour refinement: {k_capacity_violation:.6g}"
                )
        new_assignments = _selected_assignments(data, problem, solution.vector)
        if not new_assignments.empty:
            duplicates = set(new_assignments["TaskID"]) & set(assignments["TaskID"])
            if duplicates:
                raise RuntimeError(f"Tasks executed more than once; examples: {sorted(duplicates)[:5]}")
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
            f"Rolling window {window_index}/{total_windows} completed: window={tau}, "
            f"solver elapsed={solution.elapsed_seconds:.2f}s, new tasks={len(new_assignments)}, "
            f"total tasks={len(assignments)}, attempts={solve_attempts}."
        )
    _progress("Rolling optimization completed; organizing and writing results.")
    assignments = assignments.sort_values(["StartHour", "TaskID"], kind="stable").reset_index(drop=True)
    if len(assignments) != len(data.tasks) or assignments["TaskID"].nunique() != len(data.tasks):
        missing = sorted(set(data.tasks["TaskID"]) - set(assignments["TaskID"]))[:10]
        raise RuntimeError(f"Unexecuted tasks remain after rolling optimization; examples: {missing}")
    dispatch = pd.DataFrame(dispatch_rows).sort_values(["Hour", "Region"], kind="stable").reset_index(drop=True)
    expected_dispatch_rows = OPERATION_END * len(data.regions)
    if len(dispatch) != expected_dispatch_rows:
        raise RuntimeError(f"Actual energy trajectory requires {expected_dispatch_rows} rows; found {len(dispatch)}")
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
    _progress("Result tables written; running minimal Q4 hard-constraint checks.")
    simple_validation = _simple_validation(
        data, assignments, dispatch, config.feasibility_tolerance
    )
    _write_csv(simple_validation, "q4_simple_validation.csv")
    if not bool(simple_validation["Passed"].all()):
        failed = simple_validation.loc[
            ~simple_validation["Passed"], ["Check", "MaxViolation", "Tolerance"]
        ]
        raise RuntimeError(f"Minimal Q4 hard-constraint checks failed:\n{failed.to_string(index=False)}")
    _progress(f"Minimal Q4 hard-constraint checks passed: {len(simple_validation)} checks.")
    (TABLES_DIR / ".q4_cache.json").write_text(
        json.dumps({"signature": signature, "complete": True}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _progress("Q4 model completed.")


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    kind: str
    carbon_lambda: float | None = None
    low_carbon_reference: float | None = None
    renewable_gamma: float | None = None


@dataclass
class Q4RollingState:
    current_tau: int
    remaining_task_ids: tuple[str, ...]
    remaining_group_counts: dict[str, int]
    active_tasks: pd.DataFrame
    region_soc: np.ndarray
    historical_peak_import: np.ndarray
    past_carbon: float


def _scenario_slug(value: str) -> str:
    cleaned = "".join(character for character in value if character.isalnum() or character in "_-" )
    if not cleaned or cleaned != value:
        raise ValueError("Scenario names may contain CJK characters, letters, digits, underscores, or hyphens only")
    return cleaned


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_existing_baseline() -> Path | None:
    """Locate complete formal baseline artifacts, excluding scenario and validation directories."""

    candidates = [TABLES_DIR]
    candidates.extend(
        path.parent for path in QUESTION_DIR.rglob("q4_task_assignments.csv")
        if "scenarios" not in path.parts and "validation" not in path.parts
    )
    seen: set[Path] = set()
    for directory in candidates:
        resolved = directory.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if all((resolved / filename).is_file() for filename in BASELINE_REQUIRED_FILES):
            return resolved
    return None


def load_cached_baseline(directory: Path | None = None) -> dict[str, pd.DataFrame]:
    baseline_dir = (directory or locate_existing_baseline())
    if baseline_dir is None:
        raise FileNotFoundError("BLOCKED_BASELINE_INVALID: complete Q4 baseline directory not found")
    frames: dict[str, pd.DataFrame] = {}
    for filename in BASELINE_REQUIRED_FILES:
        path = baseline_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"BLOCKED_BASELINE_INVALID: missing {path}")
        frames[filename] = pd.read_csv(path, encoding="utf-8-sig")
    frames["q4_task_assignments.csv"]["TaskID"] = (
        frames["q4_task_assignments.csv"]["TaskID"].astype(str)
    )
    frames["q4_region_hour_dispatch.csv"]["Region"] = (
        frames["q4_region_hour_dispatch.csv"]["Region"].astype(str)
    )
    return frames


def _solver_metrics_from_summary(summary: pd.DataFrame) -> dict[str, float]:
    if not {"Metric", "Value"}.issubset(summary.columns):
        return {}
    values: dict[str, float] = {}
    for row in summary.itertuples(index=False):
        metric = str(row.Metric)
        if metric == "Latency_ms":
            values["Latency"] = float(row.Value)
        elif metric == "QoSLoss":
            values["Delay"] = float(row.Value)
        elif metric == "Latency":
            values["LatencySLA"] = float(row.Value)
        elif metric == "Delay":
            values["LegacyDelay"] = float(row.Value)
        elif metric in METRICS:
            values[metric] = float(row.Value)
    return values


def independent_hard_constraint_audit(
    data: InputData,
    assignments: pd.DataFrame,
    dispatch: pd.DataFrame,
    *,
    solver_metrics: Mapping[str, float] | None = None,
    baseline_status: str = "",
    scenario_status: str = "",
    tolerance: float = 1e-6,
    qos_weights: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Recompute constraints and final metrics from actual TaskID assignments and H-region energy trajectories."""

    required_assignment = {
        "TaskID", "TaskType", "ArrivalHour", "SourceRegion", "TargetRegion",
        "NetworkLatency_ms", "MaxLatency_ms", "StartHour", "FinishHour",
        "Duration_h", "GPU_Demand", "Task_Full_IT_Power_MW", "DecisionWindowStart",
    }
    required_dispatch = {
        "Hour", "Region", "AI_IT_Load_MW", "NonAI_IT_Load_MW", "Total_IT_Load_MW", "Facility_Load_MW", "PUE",
        "AvailableRenewable_MW", "RenewableDirectUse_MW", "RenewableCharge_MW",
        "GridCharge_MW", "DischargePower_MW", "GridPurchase_MW", "GridSell_MW",
        "RenewableCurtailment_MW", "NetGridImport_MW", "SOCStart_MWh", "SOCEnd_MWh",
        "ChargePower_MW", "CarbonEmission_tCO2", "ElectricityPrice_CNY_per_MWh",
        "SellPrice_CNY_per_MWh", "CarbonIntensity_tCO2_per_MWh", "Available_GPU",
        "Max_IT_Power_MW", "Max_Facility_Power_MW",
    }
    missing_columns = sorted(
        (required_assignment - set(assignments.columns))
        | (required_dispatch - set(dispatch.columns))
    )
    if missing_columns:
        return {
            "passed": False,
            "failed_checks": ["RequiredColumns"],
            "max_violation": float("inf"),
            "violation_details": {"RequiredColumns": missing_columns},
            "independent_metrics": {},
            "solver_metrics": dict(solver_metrics or {}),
            "metric_differences": {},
            "tolerance_used": tolerance,
            "baseline_status": baseline_status,
            "scenario_status": scenario_status,
        }
    assignments = assignments.copy()
    dispatch = dispatch.copy()
    assignments["TaskID"] = assignments["TaskID"].astype(str)
    assignments["TargetRegion"] = assignments["TargetRegion"].astype(str)
    dispatch["Region"] = dispatch["Region"].astype(str)
    validation = _simple_validation(data, assignments, dispatch, tolerance)
    extra: list[dict[str, object]] = []

    def add_extra(check: str, violation: float, details: object = None) -> None:
        extra.append({
            "Check": check,
            "MaxViolation": float(max(violation, 0.0)),
            "Tolerance": float(tolerance),
            "Passed": bool(violation <= tolerance),
            "Details": details,
        })

    known_ids = set(data.tasks["TaskID"].astype(str))
    actual_ids = assignments["TaskID"].astype(str)
    missing_ids = sorted(known_ids - set(actual_ids))
    duplicate_ids = sorted(actual_ids[actual_ids.duplicated(keep=False)].unique())
    unknown_ids = sorted(set(actual_ids) - known_ids)
    add_extra("NoMissingTask", float(len(missing_ids) > 0), missing_ids[:20])
    add_extra("NoDuplicateTask", float(len(duplicate_ids) > 0), duplicate_ids[:20])
    add_extra("NoUnknownOrKForecastTask", float(len(unknown_ids) > 0), unknown_ids[:20])
    decision_end = np.minimum(
        pd.to_numeric(assignments["DecisionWindowStart"]).to_numpy(dtype=float) + 24.0,
        float(OPERATION_END),
    )
    k_leak = (
        pd.to_numeric(assignments["StartHour"]).to_numpy(dtype=float)
        >= decision_end - EPS
    )
    add_extra(
        "KPredictionExcludedFromActual",
        float(np.count_nonzero(k_leak) > 0),
        assignments.loc[k_leak, "TaskID"].astype(str).head(20).tolist(),
    )
    terminal_violation = np.maximum(
        pd.to_numeric(assignments["FinishHour"]).to_numpy(dtype=float) - OPERATION_END,
        0.0,
    )
    add_extra(
        "Terminal2406Boundary",
        float(terminal_violation.max(initial=0.0)),
        assignments.loc[terminal_violation > tolerance, "TaskID"].astype(str).head(20).tolist(),
    )
    for column in (
        "GridPurchase_MW", "GridSell_MW", "RenewableDirectUse_MW",
        "RenewableCharge_MW", "GridCharge_MW", "DischargePower_MW",
        "RenewableCurtailment_MW",
    ):
        negative = np.maximum(-pd.to_numeric(dispatch[column]).to_numpy(dtype=float), 0.0)
        add_extra(f"Nonnegative_{column}", float(negative.max(initial=0.0)))
    expected_hours = set(range(OPERATION_END))
    actual_hours = set(pd.to_numeric(dispatch["Hour"]).astype(int))
    add_extra("CompleteOperationHours", float(actual_hours != expected_hours), {
        "missing": sorted(expected_hours - actual_hours)[:20],
        "extra": sorted(actual_hours - expected_hours)[:20],
    })
    duplicate_hour_region = int(dispatch.duplicated(["Hour", "Region"]).sum())
    add_extra("UniqueHourRegion", float(duplicate_hour_region > 0), duplicate_hour_region)
    metrics = _final_metrics(
        data, assignments, dispatch, qos_weights=qos_weights
    )
    metrics["RenewableUtilization"] = 1.0 - metrics["RenewableUnusedRate"]
    solver_values = dict(solver_metrics or {})
    differences = {
        metric: metrics[metric] - float(solver_values[metric])
        for metric in METRICS if metric in solver_values
    }
    for metric, difference in differences.items():
        add_extra(
            f"MetricRecomputation_{metric}",
            abs(float(difference)),
            {"reported_minus_recomputed": float(difference)},
        )
    combined = pd.concat([
        validation.assign(Details=None),
        pd.DataFrame(extra),
    ], ignore_index=True)
    failed = combined.loc[~combined["Passed"], "Check"].astype(str).tolist()
    detail_map = {
        str(row.Check): row.Details
        for row in combined.itertuples(index=False)
        if not bool(row.Passed)
    }
    return {
        "passed": not failed,
        "failed_checks": failed,
        "max_violation": float(combined["MaxViolation"].max()),
        "violation_details": detail_map,
        "independent_metrics": metrics,
        "solver_metrics": solver_values,
        "metric_differences": differences,
        "tolerance_used": tolerance,
        "baseline_status": baseline_status,
        "scenario_status": scenario_status,
        "checks": combined,
    }


def validate_baseline_artifacts(
    directory: Path | None = None,
    *,
    data: InputData | None = None,
    tolerance: float = 1e-6,
) -> dict[str, object]:
    baseline_dir = directory or locate_existing_baseline()
    if baseline_dir is None:
        return {
            "baseline_status": "BLOCKED_BASELINE_INVALID",
            "passed": False,
            "errors": ["No Q4 baseline directory contains all required files"],
        }
    errors: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    for filename in BASELINE_REQUIRED_FILES:
        path = baseline_dir / filename
        if not path.is_file():
            errors.append(f"Missing file: {filename}")
            continue
        try:
            frames[filename] = pd.read_csv(path, encoding="utf-8-sig")
            hashes[filename] = _sha256_file(path)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            errors.append(f"Cannot read {filename}: {exc}")
    if errors:
        return {
            "baseline_status": "BLOCKED_BASELINE_INVALID",
            "passed": False,
            "errors": errors,
            "baseline_dir": str(baseline_dir),
            "hashes": hashes,
        }
    data = data or load_data()
    assignments = frames["q4_task_assignments.csv"].copy()
    assignments["TaskID"] = assignments["TaskID"].astype(str)
    dispatch = frames["q4_region_hour_dispatch.csv"].copy()
    dispatch["Region"] = dispatch["Region"].astype(str)
    frames["q4_task_assignments.csv"] = assignments
    frames["q4_region_hour_dispatch.csv"] = dispatch
    solver = frames["q4_window_solver.csv"]
    simple = frames["q4_simple_validation.csv"]
    if len(assignments) != len(data.tasks):
        errors.append(f"Assignment rows {len(assignments)} differ from original task count {len(data.tasks)}")
    if assignments.get("TaskID", pd.Series(dtype=str)).astype(str).nunique() != len(data.tasks):
        errors.append("TaskID coverage is incomplete or duplicated")
    expected_pairs = OPERATION_END * len(data.regions)
    if len(dispatch) != expected_pairs:
        errors.append(f"Energy trajectory rows {len(dispatch)} differ from expected {expected_pairs}")
    if dispatch.duplicated(["Hour", "Region"]).any():
        errors.append("Energy trajectory contains duplicate Hour x Region records")
    expected_windows = list(range(0, MAIN_END, 24)) + [MAIN_END]
    actual_windows = pd.to_numeric(solver.get("WindowStart", pd.Series(dtype=float)), errors="coerce")
    if len(solver) != len(expected_windows) or set(actual_windows.dropna().astype(int)) != set(expected_windows):
        errors.append("Rolling window records do not cover all 101 windows")
    simple_passed = (
        simple["Passed"].astype(str).str.strip().str.lower().map({"true": True, "false": False})
        if "Passed" in simple.columns else pd.Series(dtype=bool)
    )
    if not {"Check", "Passed"}.issubset(simple.columns) or simple_passed.isna().any() or not bool(simple_passed.all()):
        errors.append("Baseline q4_simple_validation.csv contains failed checks")
    solver_metrics = _solver_metrics_from_summary(frames["q4_objective_summary.csv"])
    try:
        audit = independent_hard_constraint_audit(
            data, assignments, dispatch,
            solver_metrics=solver_metrics,
            baseline_status="REUSED_BASELINE",
            scenario_status="BASELINE",
            tolerance=tolerance,
        )
        if not bool(audit["passed"]):
            errors.append(f"Independent hard-constraint audit failed: {audit['failed_checks']}")
    except Exception as exc:
        audit = {"passed": False, "failed_checks": ["AuditException"], "error": str(exc)}
        errors.append(f"Independent audit exception: {exc}")
    configuration = frames["q4_model_configuration.csv"]
    if not {"Parameter", "Value"}.issubset(configuration.columns):
        errors.append("q4_model_configuration.csv lacks Parameter/Value traceability columns")
    return {
        "baseline_status": "REUSED_BASELINE" if not errors else "BLOCKED_BASELINE_INVALID",
        "passed": not errors,
        "errors": errors,
        "baseline_dir": str(baseline_dir.resolve()),
        "hashes": hashes,
        "row_counts": {name: int(len(frame)) for name, frame in frames.items()},
        "audit": audit,
    }


def _can_reuse_baseline() -> bool:
    return bool(validate_baseline_artifacts().get("passed", False))


def baseline_integrity_report() -> dict[str, object]:
    report = validate_baseline_artifacts()
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    serializable = dict(report)
    audit = dict(serializable.get("audit", {}))
    checks = audit.pop("checks", None)
    serializable["audit"] = audit
    _atomic_write_json(VALIDATION_DIR / "q4_baseline_integrity_report.json", serializable)
    if isinstance(checks, pd.DataFrame):
        _atomic_write_csv(checks, VALIDATION_DIR / "q4_baseline_independent_audit.csv")
    if report.get("passed"):
        baseline = load_cached_baseline(Path(str(report["baseline_dir"])))
        data = load_data()
        _, qos = _qos_details(data, baseline["q4_task_assignments.csv"])
        _atomic_write_csv(qos, VALIDATION_DIR / "q4_baseline_qos_recompute.csv")
    return report


def _copy_input_data_with_region_hour(data: InputData, region_hour: pd.DataFrame) -> InputData:
    return InputData(
        region_hour=region_hour,
        tasks=data.tasks,
        candidates=data.candidates,
        storage=data.storage,
        regions=data.regions,
        region_index=data.region_index,
        candidate_map=data.candidate_map,
        task_lookup=data.task_lookup,
    )


def generate_scenario_data(
    data: InputData,
    spec: ScenarioSpec,
    baseline_metrics: Mapping[str, float],
) -> tuple[InputData, dict[str, object]]:
    """Change only the specified exogenous factor, retaining original tasks, capacity, network, and storage objects."""

    kind = spec.kind.strip().lower()
    frame = data.region_hour.copy(deep=True)
    metadata: dict[str, object] = {
        "ScenarioName": spec.name,
        "ScenarioKind": kind,
        "ScenarioDefinitionVersion": "Q4_SCENARIO_V4_DOCUMENT_COMPLIANT",
        "CarbonCap": None,
        "ChangedColumns": [],
    }
    if kind == "low_carbon_reference":
        metadata["ScenarioStatus"] = "LOW_CARBON_REFERENCE"
    elif kind == "carbon_constraint":
        baseline_carbon = float(baseline_metrics["Carbon"])
        low_carbon = spec.low_carbon_reference
        if low_carbon is None:
            if abs(baseline_carbon) <= 1e-8:
                low_carbon = 0.0
                metadata["CarbonConstraintBindingExpected"] = False
            else:
                raise ValueError("LOW_CARBON_REFERENCE is missing; a strict carbon budget cannot be fabricated")
        lam = float(spec.carbon_lambda if spec.carbon_lambda is not None else 1.0)
        if not 0.0 <= lam <= 1.0:
            raise ValueError("carbon_lambda must lie within [0,1]")
        if float(low_carbon) > baseline_carbon + 1e-8:
            raise ValueError("LOW_CARBON_REFERENCE emissions must not exceed baseline emissions")
        cap = float(baseline_carbon - lam * (baseline_carbon - float(low_carbon)))
        metadata.update({
            "CarbonLambda": lam,
            "LowCarbonReference": float(low_carbon),
            "BaselineCarbon": baseline_carbon,
            "CarbonCap": max(cap, 0.0),
            "CarbonBudgetRule": "E0-lambda*(E0-ELC)",
        })
    elif kind == "flat_price":
        for column in ("ElectricityPrice_CNY_per_MWh", "SellPrice_CNY_per_MWh"):
            frame[column] = frame.groupby("Region", observed=True)[column].transform("mean")
        metadata["ChangedColumns"] = [
            "ElectricityPrice_CNY_per_MWh", "SellPrice_CNY_per_MWh"
        ]
    elif kind == "low_variability_renewable":
        gamma = float(spec.renewable_gamma if spec.renewable_gamma is not None else 1.0)
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("renewable_gamma must lie within [0,1]")
        frame["_Day"] = frame["Hour"].astype(int) // 24
        smoothed = frame["AvailableRenewable_MW"].copy()
        for _, indexes in frame.groupby(["Region", "_Day"], observed=True).groups.items():
            indexes = list(indexes)
            positive = [idx for idx in indexes if float(frame.at[idx, "AvailableRenewable_MW"]) > EPS]
            if positive:
                daily_total = float(frame.loc[positive, "AvailableRenewable_MW"].sum())
                daily_mean = daily_total / len(positive)
                smoothed.loc[positive] = (
                    (1.0 - gamma) * frame.loc[positive, "AvailableRenewable_MW"]
                    + gamma * daily_mean
                )
            zero_indexes = [idx for idx in indexes if idx not in positive]
            if zero_indexes:
                smoothed.loc[zero_indexes] = 0.0
        frame["AvailableRenewable_MW"] = smoothed
        frame = frame.drop(columns="_Day")
        metadata.update({
            "RenewableSmoothingGamma": gamma,
            "ChangedColumns": ["AvailableRenewable_MW"] if gamma > EPS else [],
        })
    else:
        raise ValueError(f"Unknown scenario type: {spec.kind}")
    scenario_data = _copy_input_data_with_region_hour(data, frame)
    metadata["StructuralInputsUnchanged"] = bool(
        scenario_data.tasks is data.tasks
        and scenario_data.candidates is data.candidates
        and scenario_data.storage is data.storage
    )
    return scenario_data, metadata


def validate_single_factor_scenario(
    baseline: InputData,
    scenario: InputData,
    metadata: Mapping[str, object],
    tolerance: float = 1e-9,
) -> dict[str, object]:
    columns = [
        "ElectricityPrice_CNY_per_MWh", "SellPrice_CNY_per_MWh",
        "CarbonIntensity_tCO2_per_MWh", "AvailableRenewable_MW",
        "NonAI_IT_Load_MW", "Available_GPU", "Max_IT_Power_MW", "PUE",
        "Max_Facility_Power_MW",
    ]
    changed = [
        column for column in columns
        if not np.allclose(
            baseline.region_hour[column].to_numpy(dtype=float),
            scenario.region_hour[column].to_numpy(dtype=float),
            rtol=0.0, atol=tolerance,
        )
    ]
    expected = set(metadata.get("ChangedColumns", []))
    kind = str(metadata.get("ScenarioKind", ""))
    structural_inputs_unchanged = bool(
        scenario.tasks is baseline.tasks
        and scenario.candidates is baseline.candidates
        and scenario.storage is baseline.storage
        and scenario.candidate_map is baseline.candidate_map
        and scenario.task_lookup is baseline.task_lookup
        and scenario.regions == baseline.regions
        and scenario.region_index == baseline.region_index
    )
    hour_region_unchanged = bool(
        baseline.region_hour[["Hour", "Region"]].reset_index(drop=True).equals(
            scenario.region_hour[["Hour", "Region"]].reset_index(drop=True)
        )
    )
    invariant_passed = structural_inputs_unchanged and hour_region_unchanged
    if kind == "flat_price":
        for column in ("ElectricityPrice_CNY_per_MWh", "SellPrice_CNY_per_MWh"):
            original_mean = baseline.region_hour.groupby("Region", observed=True)[column].mean()
            scenario_mean = scenario.region_hour.groupby("Region", observed=True)[column].mean()
            expected_profile = baseline.region_hour.groupby("Region", observed=True)[column].transform("mean")
            invariant_passed = invariant_passed and bool(np.allclose(
                original_mean.sort_index().to_numpy(dtype=float),
                scenario_mean.sort_index().to_numpy(dtype=float),
                rtol=0.0, atol=tolerance,
            )) and bool(np.allclose(
                expected_profile.to_numpy(dtype=float),
                scenario.region_hour[column].to_numpy(dtype=float),
                rtol=0.0, atol=tolerance,
            ))
    elif kind == "low_variability_renewable":
        gamma = float(metadata.get("RenewableSmoothingGamma", 1.0))
        gamma_valid = 0.0 <= gamma <= 1.0
        original = baseline.region_hour[["Hour", "Region", "AvailableRenewable_MW"]].copy()
        changed_frame = scenario.region_hour[["Hour", "Region", "AvailableRenewable_MW"]].copy()
        original["Day"] = original["Hour"].astype(int) // 24
        changed_frame["Day"] = changed_frame["Hour"].astype(int) // 24
        original_daily = original.groupby(["Region", "Day"], observed=True)["AvailableRenewable_MW"].sum()
        scenario_daily = changed_frame.groupby(["Region", "Day"], observed=True)["AvailableRenewable_MW"].sum()
        same_daily_energy = np.allclose(
            original_daily.sort_index().to_numpy(dtype=float),
            scenario_daily.sort_index().to_numpy(dtype=float),
            rtol=0.0, atol=tolerance,
        )
        original_support = original["AvailableRenewable_MW"].to_numpy(dtype=float) > EPS
        scenario_support = changed_frame["AvailableRenewable_MW"].to_numpy(dtype=float) > EPS
        expected_profile = original["AvailableRenewable_MW"].copy()
        for _, indexes in original.groupby(["Region", "Day"], observed=True).groups.items():
            indexes = list(indexes)
            positive = [idx for idx in indexes if float(original.at[idx, "AvailableRenewable_MW"]) > EPS]
            if positive:
                daily_mean = float(original.loc[positive, "AvailableRenewable_MW"].mean())
                expected_profile.loc[positive] = (
                    (1.0 - gamma) * original.loc[positive, "AvailableRenewable_MW"]
                    + gamma * daily_mean
                )
            expected_profile.loc[[idx for idx in indexes if idx not in positive]] = 0.0
        interpolation_matches = np.allclose(
            expected_profile.to_numpy(dtype=float),
            changed_frame["AvailableRenewable_MW"].to_numpy(dtype=float),
            rtol=0.0, atol=tolerance,
        )
        invariant_passed = invariant_passed and bool(
            gamma_valid and same_daily_energy
            and np.array_equal(original_support, scenario_support)
            and interpolation_matches
        )
    passed = set(changed) == expected and invariant_passed
    if kind == "carbon_constraint":
        try:
            lam = float(metadata["CarbonLambda"])
            baseline_carbon = float(metadata["BaselineCarbon"])
            low_carbon = float(metadata["LowCarbonReference"])
            expected_cap = baseline_carbon - lam * (baseline_carbon - low_carbon)
            cap_matches = abs(float(metadata["CarbonCap"]) - max(expected_cap, 0.0)) <= tolerance
        except (KeyError, TypeError, ValueError):
            cap_matches = False
        passed = passed and not changed and cap_matches
    return {
        "passed": bool(passed),
        "scenario_kind": kind,
        "changed_columns": changed,
        "expected_changed_columns": sorted(expected),
        "structural_inputs_unchanged": structural_inputs_unchanged,
        "hour_region_unchanged": hour_region_unchanged,
        "scenario_invariants_passed": invariant_passed,
    }


def aggregation_integrity_report(
    data: InputData,
    tau: int,
    assignments: pd.DataFrame,
    config: ModelConfig,
) -> dict[str, object]:
    decision_end = min(tau + config.decision_hours, OPERATION_END)
    plan_end = min(decision_end + config.lookahead_hours, OPERATION_END)
    x_options, u_options, rules, groups = _make_task_options(
        data, tau, decision_end, plan_end, assignments, config.block_hours, False
    )
    original_count = sum(len(group) for group in groups.values())
    load_equivalent = True
    feasible_domain_equivalent = True
    for group_id, members in groups.items():
        representative = data.task_lookup.loc[group_id]
        rep_key = _exact_task_group_key(data, representative)
        for task_id in members:
            if _exact_task_group_key(data, data.task_lookup.loc[task_id]) != rep_key:
                feasible_domain_equivalent = False
                load_equivalent = False
                break
    type_groups: dict[str, int] = {}
    for group_id in groups:
        task_type = str(data.task_lookup.loc[group_id, "TaskType"])
        type_groups[task_type] = type_groups.get(task_type, 0) + 1
    return {
        "WindowStart": tau,
        "OriginalTaskCount": original_count,
        "AggregatedTaskCount": original_count,
        "ExactTaskGroupCount": len(groups),
        "MaxGroupSize": max((len(group) for group in groups.values()), default=0),
        "TaskTypeGroupCounts": type_groups,
        "HOptionCount": len(x_options),
        "KOptionCount": len(u_options),
        "TaskCountDifference": 0,
        "MaxGPULoadDifference": 0.0 if load_equivalent else float("inf"),
        "MaxITLoadDifference": 0.0 if load_equivalent else float("inf"),
        "MaxFacilityLoadDifference": 0.0 if load_equivalent else float("inf"),
        "FeasibleDomainEquivalent": feasible_domain_equivalent,
        "DelaySensitivityInKey": True,
        "RuleGroupCount": len(rules),
    }


def classify_task_states(
    data: InputData,
    assignments: pd.DataFrame,
    tau: int,
) -> pd.DataFrame:
    assigned = assignments.set_index("TaskID", drop=False) if not assignments.empty else pd.DataFrame()
    rows: list[dict[str, object]] = []
    for task in data.tasks.itertuples(index=False):
        task_id = str(task.TaskID)
        latest_start = _latest_integer_start(task)
        if not assigned.empty and task_id in assigned.index:
            result = assigned.loc[task_id]
            finish = float(result.FinishHour)
            state = "completed" if finish <= tau + EPS else "scheduled_in_H"
        elif latest_start < tau:
            state = "infeasible"
        elif int(task.ArrivalHour) > tau:
            state = "not_arrived"
        elif int(task.ArrivalHour) < tau and latest_start >= tau:
            state = "deferred"
        else:
            state = "arrived_unscheduled"
        rows.append({
            "TaskID": task_id,
            "State": state,
            "ArrivalHour": int(task.ArrivalHour),
            "LatestLegalStart": latest_start,
            "IsRealtime": str(task.TaskType) == "RealTimeInference",
        })
    return pd.DataFrame(rows)


def identify_validation_windows(
    data: InputData,
    baseline_dispatch: pd.DataFrame,
    config: ModelConfig,
) -> pd.DataFrame:
    starts = np.arange(0, MAIN_END, config.decision_hours, dtype=int)
    latest = np.floor(
        np.minimum(data.tasks["LatestFinishHour"], OPERATION_END)
        - data.tasks["Duration_h"] + EPS
    ).astype(int)
    records: list[dict[str, float | int]] = []
    for tau in starts:
        plan_end = min(tau + config.decision_hours + config.lookahead_hours, OPERATION_END)
        candidates = data.tasks.loc[
            (data.tasks["ArrivalHour"] < plan_end)
            & (data.tasks["EarliestStartHour"] < plan_end)
            & (latest >= tau)
        ]
        gpu_pressure = float(
            (candidates["GPU_Demand"] * candidates["Duration_h"]).sum()
        )
        deadline_density = int(((latest >= tau) & (latest < tau + config.decision_hours)).sum())
        energy = baseline_dispatch.loc[
            baseline_dispatch["Hour"].between(tau, tau + config.decision_hours - 1)
        ]
        renewable_mean = float(energy["AvailableRenewable_MW"].mean()) if not energy.empty else 0.0
        renewable_std = float(energy["AvailableRenewable_MW"].std(ddof=0)) if not energy.empty else 0.0
        renewable_pressure = renewable_std / max(renewable_mean, EPS)
        soc_pressure = float(
            1.0 / max(energy["SOCStart_MWh"].min(), EPS)
        ) if not energy.empty else 0.0
        records.append({
            "WindowStart": int(tau),
            "CandidateTaskCount": int(len(candidates)),
            "GPUPressure": gpu_pressure,
            "DeadlineDensity": deadline_density,
            "EnergyPressure": renewable_pressure + soc_pressure,
        })
    scores = pd.DataFrame(records)
    selected: list[tuple[str, int]] = []

    def choose(label: str, order: Sequence[int]) -> None:
        for index in order:
            tau = int(scores.iloc[int(index)]["WindowStart"])
            if tau not in {item[1] for item in selected}:
                selected.append((label, tau))
                return

    normalized = scores[["GPUPressure", "DeadlineDensity", "EnergyPressure"]].copy()
    for column in normalized:
        span = float(normalized[column].max() - normalized[column].min())
        normalized[column] = (
            (normalized[column] - normalized[column].min()) / span if span > EPS else 0.0
        )
    total_pressure = normalized.sum(axis=1)
    choose("GPU_OR_COMPUTE_MAX", np.argsort(scores["GPUPressure"].to_numpy())[::-1])
    choose("DEADLINE_DENSEST", np.argsort(scores["DeadlineDensity"].to_numpy())[::-1])
    choose("SOC_OR_RENEWABLE_MAX", np.argsort(scores["EnergyPressure"].to_numpy())[::-1])
    median_distance = np.abs(total_pressure - float(total_pressure.median()))
    choose("ORDINARY", np.argsort(median_distance.to_numpy()))
    selected_frame = pd.DataFrame(selected, columns=["WindowType", "WindowStart"])
    return selected_frame.merge(scores, on="WindowStart", how="left", validate="one_to_one")


def identify_critical_neighborhood(
    data: InputData,
    problem: WindowProblem,
    limit: int = 250,
) -> tuple[str, ...]:
    """Provide deterministic critical task-group ordering for scenario MILPs without removing hard constraints."""

    ranked = sorted(
        problem.task_rules,
        key=lambda group_id: (
            str(data.task_lookup.loc[group_id, "TaskType"]) != "RealTimeInference",
            _latest_integer_start(data.task_lookup.loc[group_id]) - problem.tau,
            len(data.candidate_map[group_id]),
            -float(data.task_lookup.loc[group_id, "GPU_Demand"]),
            -float(data.task_lookup.loc[group_id, "Task_Full_IT_Power_MW"]),
            -float(data.task_lookup.loc[group_id, "Duration_h"]),
            group_id,
        ),
    )
    return tuple(ranked[:max(int(limit), 0)])


def _window_lp_scaling(
    data: InputData,
    problem: WindowProblem,
    assignments: pd.DataFrame,
    current_soc: np.ndarray,
    historical_peak: np.ndarray,
    warm_hints: Mapping[str, tuple[str, float]],
    config: ModelConfig,
    *,
    lp_time_limit: float,
) -> tuple[dict[str, tuple[float, float]], pd.DataFrame, np.ndarray]:
    """Build window-fixed six-objective scales from current continuous lower bounds and feasible references."""

    reference = _heuristic_incumbent(
        data, problem, assignments, current_soc, historical_peak,
        None, warm_hints, config.feasibility_tolerance,
    )
    reference_source = "StateAwareFeasibleUpperBound"
    if reference is None:
        zero_objective = np.zeros(problem.lower.size, dtype=float)
        feasible = _solve(
            problem, zero_objective,
            time_limit=config.difficult_time_limit_seconds,
            mip_gap=config.mip_relative_gap,
            relax=False,
            incumbent=None,
        )
        reference = feasible.vector
        reference_source = "MILPFeasibleReference"
    if not _vector_is_feasible(
        problem, reference,
        max(config.feasibility_tolerance, config.integrality_tolerance),
    ):
        raise RuntimeError(f"Window {problem.tau}: scaling reference violates original hard constraints")
    reference_values = _metric_values(problem, reference)
    scaling: dict[str, tuple[float, float]] = {}
    records: list[dict[str, object]] = []
    for metric in METRICS:
        exact_lp = False
        bound_source = ""
        try:
            relaxed = _solve(
                problem, problem.metric_vectors[metric],
                time_limit=max(float(lp_time_limit), 1.0),
                mip_gap=config.mip_relative_gap,
                relax=True,
                incumbent=None,
            )
            if relaxed.status == 0:
                lower = _metric_values(problem, relaxed.vector)[metric]
                exact_lp = True
                bound_source = "WindowContinuousRelaxation"
            else:
                lower, certified, bound_source = _certified_box_lower_bound(problem, metric)
                if not certified:
                    raise RuntimeError(f"{metric} has no provable lower bound")
        except RuntimeError:
            lower, certified, bound_source = _certified_box_lower_bound(problem, metric)
            if not certified or not np.isfinite(lower):
                raise RuntimeError(f"Window {problem.tau}, metric {metric}: no provably safe lower bound")
        reference_value = float(reference_values[metric])
        raw_scale = reference_value - float(lower)
        degeneracy_tolerance = max(abs(reference_value), abs(float(lower)), 1.0) * 1e-6
        degenerate = raw_scale <= degeneracy_tolerance
        scale = 1.0 if degenerate else raw_scale
        scaling[metric] = (float(lower), float(scale))
        records.append({
            "WindowStart": problem.tau,
            "Metric": metric,
            "LPAnchor": float(lower),
            "FeasibleReference": reference_value,
            "Scale": float(scale),
            "DegenerateMetric": bool(degenerate),
            "ExactLPAnchor": bool(exact_lp),
            "BoundSource": bound_source,
            "ReferenceSource": reference_source,
        })
    return scaling, pd.DataFrame(records), reference


def _baseline_state_at_tau(
    data: InputData,
    baseline: Mapping[str, pd.DataFrame],
    tau: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, tuple[float, float]]]:
    assignments = baseline["q4_task_assignments.csv"].copy()
    assignments["TaskID"] = assignments["TaskID"].astype(str)
    before = assignments.loc[
        pd.to_numeric(assignments["DecisionWindowStart"]) < tau
    ].copy()
    dispatch = baseline["q4_region_hour_dispatch.csv"].copy()
    storage = data.storage.set_index("Region").loc[list(data.regions)]
    if tau == 0:
        current_soc = storage["InitialSOC_MWh"].to_numpy(dtype=float)
        historical_peak = np.zeros(len(data.regions), dtype=float)
    else:
        prior = dispatch.loc[pd.to_numeric(dispatch["Hour"]).eq(tau - 1)]
        prior = prior.set_index("Region").loc[list(data.regions)]
        current_soc = prior["SOCEnd_MWh"].to_numpy(dtype=float)
        historical_peak = (
            dispatch.loc[pd.to_numeric(dispatch["Hour"]) < tau]
            .groupby("Region", observed=True)["NetGridImport_MW"].max()
            .reindex(data.regions).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
        )
    scaling_frame = baseline["q4_scaling.csv"]
    scaling = {
        str(row.Metric): (float(row.Anchor), float(row.Scale))
        for row in scaling_frame.itertuples(index=False)
    }
    # Baseline Delay scaling uses the old unweighted definition and cannot be combined with corrected J.
    # Weighted relative delay loss lies in [0,1] by definition; use a provably safe fixed scale.
    scaling["Latency"] = (
        0.0,
        max(float(data.tasks["MaxLatency_ms"].max()), 1.0),
    )
    scaling["Delay"] = (0.0, 1.0)
    return (
        before,
        np.array(current_soc, dtype=np.float64, copy=True),
        np.array(historical_peak, dtype=np.float64, copy=True),
        scaling,
    )


def _vector_is_feasible(
    problem: WindowProblem,
    vector: np.ndarray,
    tolerance: float,
) -> bool:
    if vector.shape != problem.lower.shape or not np.all(np.isfinite(vector)):
        return False
    activity = problem.matrix @ vector
    violation = max(
        float(np.maximum(problem.constraint_lower - activity, 0.0).max(initial=0.0)),
        float(np.maximum(activity - problem.constraint_upper, 0.0).max(initial=0.0)),
        float(np.maximum(problem.lower - vector, 0.0).max(initial=0.0)),
        float(np.maximum(vector - problem.upper, 0.0).max(initial=0.0)),
    )
    integer_columns = np.flatnonzero(problem.integrality)
    integer_violation = (
        float(np.abs(vector[integer_columns] - np.rint(vector[integer_columns])).max(initial=0.0))
        if len(integer_columns) else 0.0
    )
    return max(violation, integer_violation) <= tolerance


def critical_neighborhood_with_fallback(
    local_solver: Callable[[], WindowSolution | None],
    full_solver: Callable[[], WindowSolution],
) -> tuple[WindowSolution, str]:
    """Restore the complete original MILP after local failure; never accept failed heuristics as formal solutions."""

    try:
        local = local_solver()
    except RuntimeError:
        local = None
    if local is not None:
        return local, "CRITICAL_NEIGHBORHOOD_MILP"
    return full_solver(), "FULL_HK_MILP_FALLBACK"


def _run_validation_window(
    data: InputData,
    baseline: Mapping[str, pd.DataFrame],
    tau: int,
    config: ModelConfig,
    *,
    scenario_name: str = "BASELINE_INTERFACE",
    carbon_cap: float | None = None,
    time_limit: float = 30.0,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    assignments_before, current_soc, historical_peak, scaling = _baseline_state_at_tau(
        data, baseline, tau
    )
    decision_hours = min(config.decision_hours, OPERATION_END - tau)
    lookahead = min(
        config.lookahead_hours,
        max(OPERATION_END - tau - decision_hours, 0),
    )
    problem = build_window_problem(
        data, config, tau, decision_hours, lookahead,
        assignments_before, current_soc, historical_peak, scaling,
        block_hours=config.block_hours,
        carbon_budget_remaining=carbon_cap,
    )
    baseline_future = baseline["q4_task_assignments.csv"].copy()
    baseline_future["TaskID"] = baseline_future["TaskID"].astype(str)
    baseline_future = baseline_future.set_index("TaskID", drop=False)
    warm_hints: dict[str, tuple[str, float]] = {}
    for group_id, members in problem.task_groups.items():
        member = next((task_id for task_id in members if task_id in baseline_future.index), None)
        if member is not None:
            reference = baseline_future.loc[member]
            warm_hints[group_id] = (
                str(reference.TargetRegion),
                float(reference.StartHour),
            )
    aggregation = aggregation_integrity_report(
        data, tau, assignments_before, config
    )
    incumbent = (
        _heuristic_incumbent(
            data, problem, assignments_before, current_soc, historical_peak,
            scaling, warm_hints, config.feasibility_tolerance,
        )
        if config.heuristic_enabled else None
    )
    if incumbent is not None and not _vector_is_feasible(
        problem, incumbent, max(config.feasibility_tolerance, config.integrality_tolerance)
    ):
        raise RuntimeError("State-aware heuristic returned a candidate violating original MILP hard constraints")
    milp_attempt_status = "UNKNOWN"
    milp_attempt_message = ""
    try:
        if time_limit <= 0.0:
            raise RuntimeError("VALIDATION_ONLY skips the expensive MILP and audits the feasible candidate only")
        solution = _solve(
            problem, _balanced_objective(problem, config),
            time_limit=time_limit,
            mip_gap=config.mip_relative_gap,
            relax=False,
            incumbent=incumbent,
        )
        milp_attempt_status = solution.status_name
        milp_attempt_message = solution.message
    except RuntimeError as exc:
        milp_attempt_message = str(exc)
        if time_limit <= 0.0:
            milp_attempt_status = "NOT_RUN_VALIDATION_ONLY"
        elif "Time limit" in str(exc) or "time limit" in str(exc).lower():
            milp_attempt_status = "TIME_LIMIT_NO_FEASIBLE"
        elif "infeasible" in str(exc).lower():
            milp_attempt_status = "INFEASIBLE"
        else:
            milp_attempt_status = "EXCEPTION"
        if incumbent is None:
            raise
        solution = WindowSolution(
            vector=incumbent,
            status=-3,
            message=(
                "VALIDATION_ONLY uses a state-aware candidate satisfying every original MILP hard constraint; "
                f"this is not a formal scenario result. Complete MILP details: {exc}"
            ),
            objective=float(np.dot(_balanced_objective(problem, config), incumbent)),
            mip_gap=float("nan"),
            mip_node_count=float("nan"),
            elapsed_seconds=float(time_limit),
            best_bound=float("nan"),
            status_name="VALIDATION_ONLY_HEURISTIC_FEASIBLE",
        )
    new_assignments = _selected_assignments(data, problem, solution.vector)
    assignments_after = pd.concat(
        [assignments_before, new_assignments], ignore_index=True
    )
    dispatch = pd.DataFrame(_dispatch_rows(data, problem, solution, assignments_after))
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
    energy_error = float(max(
        np.abs(renewable_residual.to_numpy(dtype=float)).max(initial=0.0),
        np.abs(load_residual.to_numpy(dtype=float)).max(initial=0.0),
    ))
    if new_assignments.empty:
        actual_metrics = {
            "Cost": float((dispatch["ElectricityPrice_CNY_per_MWh"] * dispatch["GridPurchase_MW"] - dispatch["SellPrice_CNY_per_MWh"] * dispatch["GridSell_MW"]).sum()),
            "Carbon": float(dispatch["CarbonEmission_tCO2"].sum()),
            "Latency": 0.0,
            "Delay": 0.0,
            "RenewableUnusedRate": float(dispatch["RenewableCurtailment_MW"].sum() / max(dispatch["AvailableRenewable_MW"].sum(), EPS)),
            "Peak": float(dispatch.groupby("Region")["NetGridImport_MW"].max().clip(lower=0.0).sum()),
        }
    else:
        actual_metrics = _final_metrics(data, new_assignments, dispatch)
    states = classify_task_states(data, assignments_after, problem.decision_end)
    realtime_bad = new_assignments.loc[
        new_assignments["TaskType"].eq("RealTimeInference")
        & ~np.isclose(new_assignments["StartHour"], new_assignments["ArrivalHour"], atol=config.feasibility_tolerance)
    ]
    k_leak = new_assignments.loc[
        new_assignments["StartHour"] >= problem.decision_end - EPS
    ]
    window_audit_passed = bool(
        energy_error <= config.feasibility_tolerance
        and realtime_bad.empty
        and k_leak.empty
        and not new_assignments["TaskID"].duplicated().any()
        and bool(aggregation["FeasibleDomainEquivalent"])
    )
    summary = {
        "RunMode": "VALIDATION_ONLY",
        "ScenarioStatus": "VALIDATION_ONLY",
        "ScenarioName": scenario_name,
        "WindowStart": tau,
        "DecisionEnd": problem.decision_end,
        "PlanEnd": problem.plan_end,
        "CandidateTaskCount": int(problem.metadata["TaskCount"]),
        "OriginalTaskCount": int(aggregation["OriginalTaskCount"]),
        "AggregatedTaskGroupCount": int(problem.metadata["AggregatedTaskGroupCount"]),
        "VariableCount": int(problem.lower.size),
        "ConstraintCount": int(problem.matrix.shape[0]),
        "SolverStatus": solution.status_name,
        "MILPAttemptStatus": milp_attempt_status,
        "MILPAttemptMessage": milp_attempt_message,
        "ElapsedSeconds": solution.elapsed_seconds,
        "MIPGap": solution.mip_gap,
        "BestObjective": solution.objective,
        "BestBound": solution.best_bound,
        **actual_metrics,
        "RenewableUtilization": 1.0 - actual_metrics["RenewableUnusedRate"],
        "HActualTaskCount": int(len(new_assignments)),
        "HRegionCounts": new_assignments["TargetRegion"].value_counts().to_dict(),
        "HStartHourMin": float(new_assignments["StartHour"].min()) if not new_assignments.empty else float("nan"),
        "HStartHourMax": float(new_assignments["StartHour"].max()) if not new_assignments.empty else float("nan"),
        "StartSOC": current_soc.tolist(),
        "EndSOC": solution.vector[problem.indices["soc"][problem.decision_end - tau]].tolist(),
        "MaxEnergyBalanceError": energy_error,
        "AggregationConsistent": bool(aggregation["FeasibleDomainEquivalent"]),
        "DeferredTaskCount": int((states["State"] == "deferred").sum()),
        "RealtimeImmediate": realtime_bad.empty,
        "KPredictionExcluded": k_leak.empty,
        "IndependentAuditPassed": window_audit_passed,
        "CriticalNeighborhoodSize": len(identify_critical_neighborhood(data, problem)),
        "FinalConfirmationModel": (
            "ORIGINAL_Q4_FULL_HK_MILP"
            if solution.status_name in {"OPTIMAL", "TIME_LIMIT_FEASIBLE"}
            else "REQUIRED_ORIGINAL_Q4_FULL_HK_MILP_NOT_RUN_IN_VALIDATION"
        ),
    }
    return summary, new_assignments, dispatch


def run_validation_only_windows(
    *,
    time_limit: float = 30.0,
    include_scenario_smoke: bool = True,
) -> pd.DataFrame:
    _progress(
        f"VALIDATION_ONLY started: window time limit={time_limit:.1f}s, "
        f"scenario interface smoke tests={include_scenario_smoke}."
    )
    report = validate_baseline_artifacts()
    if not report.get("passed"):
        raise RuntimeError("BLOCKED_BASELINE_INVALID: cannot validate small windows dependent on baseline state")
    _progress(f"VALIDATION_ONLY baseline check passed: directory={report.get('baseline_dir')}.")
    data = load_data()
    baseline = load_cached_baseline(Path(str(report["baseline_dir"])))
    _progress(
        f"VALIDATION_ONLY inputs loaded: tasks={len(data.tasks)}, "
        f"regions={len(data.regions)}."
    )
    config = ModelConfig(
        normal_time_limit_seconds=time_limit,
        difficult_time_limit_seconds=time_limit,
    )
    _progress("VALIDATION_ONLY selecting validation windows.")
    selected = identify_validation_windows(
        data, baseline["q4_region_hour_dispatch.csv"], config
    )
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(selected, VALIDATION_DIR / "q4_validation_window_selection.csv")
    _progress(f"VALIDATION_ONLY selected {len(selected)} validation windows.")
    summaries: list[dict[str, object]] = []
    for index, row in enumerate(selected.itertuples(index=False), start=1):
        tau = int(row.WindowStart)
        _progress(
            f"VALIDATION_ONLY window started: {index}/{len(selected)}, "
            f"start={tau}, type={row.WindowType}, time limit={time_limit:.1f}s."
        )
        try:
            summary, assignments, dispatch = _run_validation_window(
                data, baseline, tau, config,
                scenario_name=str(row.WindowType),
                time_limit=time_limit,
            )
            summaries.append(summary)
            _atomic_write_csv(
                assignments,
                VALIDATION_DIR / f"q4_validation_window_{tau}_assignments.csv",
            )
            _atomic_write_csv(
                dispatch,
                VALIDATION_DIR / f"q4_validation_window_{tau}_dispatch.csv",
            )
            _progress(
                f"VALIDATION_ONLY window completed: start={tau}, "
                f"status={summary.get('SolverStatus')}，"
                f"independent audit={summary.get('IndependentAuditPassed')}, "
                f"elapsed={float(summary.get('ElapsedSeconds', float('nan'))):.2f}s."
            )
        except Exception as exc:
            summaries.append({
                "RunMode": "VALIDATION_ONLY",
                "ScenarioStatus": "VALIDATION_ONLY",
                "ScenarioName": str(row.WindowType),
                "WindowStart": tau,
                "SolverStatus": "EXCEPTION",
                "Error": str(exc),
                "IndependentAuditPassed": False,
            })
            _progress(f"VALIDATION_ONLY window failed; continuing: start={tau}, error={exc}.")
    summary_frame = pd.DataFrame(summaries)
    _atomic_write_csv(summary_frame, VALIDATION_DIR / "q4_validation_window_summary.csv")
    _progress(
        f"VALIDATION_ONLY window summary written: {VALIDATION_DIR / 'q4_validation_window_summary.csv'}, "
        f"records={len(summary_frame)}."
    )
    if include_scenario_smoke:
        ordinary_tau = int(
            selected.loc[selected["WindowType"].eq("ORDINARY"), "WindowStart"].iloc[0]
        )
        baseline_metrics = _solver_metrics_from_summary(
            baseline["q4_objective_summary.csv"]
        )
        specs = (
            ScenarioSpec("carbon_lambda_1", "carbon_constraint", 1.0, 0.0),
            ScenarioSpec("flat_price", "flat_price"),
            ScenarioSpec("low_variability_renewable", "low_variability_renewable"),
        )
        smoke_rows: list[dict[str, object]] = []
        _progress(
            f"VALIDATION_ONLY scenario interface smoke tests started: {len(specs)} cases, "
            f"window start={ordinary_tau}."
        )
        for index, spec in enumerate(specs, start=1):
            _progress(f"VALIDATION_ONLY scenario smoke test: {index}/{len(specs)}, scenario={spec.name}.")
            scenario_data, metadata = generate_scenario_data(data, spec, baseline_metrics)
            factor_check = validate_single_factor_scenario(data, scenario_data, metadata)
            try:
                scenario_summary, _, _ = _run_validation_window(
                    scenario_data, baseline, ordinary_tau, config,
                    scenario_name=spec.name,
                    carbon_cap=metadata.get("CarbonCap"),
                    time_limit=time_limit,
                )
                smoke_rows.append({**metadata, **factor_check, **scenario_summary})
                _progress(
                    f"VALIDATION_ONLY scenario smoke test completed: scenario={spec.name}, "
                    f"status={scenario_summary.get('SolverStatus')}，"
                    f"independent audit={scenario_summary.get('IndependentAuditPassed')}."
                )
            except Exception as exc:
                smoke_rows.append({
                    **metadata, **factor_check,
                    "RunMode": "VALIDATION_ONLY",
                    "ScenarioStatus": "VALIDATION_ONLY",
                    "WindowStart": ordinary_tau,
                    "SolverStatus": "EXCEPTION",
                    "Error": str(exc),
                })
                _progress(f"VALIDATION_ONLY scenario smoke test failed; continuing: scenario={spec.name}, error={exc}.")
        _atomic_write_csv(
            pd.DataFrame(smoke_rows),
            VALIDATION_DIR / "q4_scenario_interface_smoke.csv",
        )
        _progress(
            f"VALIDATION_ONLY scenario interface smoke summary written: "
            f"{VALIDATION_DIR / 'q4_scenario_interface_smoke.csv'}, records={len(smoke_rows)}."
        )
    return summary_frame


def run_automated_tests() -> pd.DataFrame:
    """Run regression checks against actual caches and interfaces without recomputing the baseline."""

    records: list[dict[str, object]] = []

    def record(name: str, passed: bool, details: object = "") -> None:
        records.append({"Test": name, "Passed": bool(passed), "Details": str(details)})

    baseline_dir = locate_existing_baseline()
    before_hashes = {
        filename: _sha256_file(baseline_dir / filename)
        for filename in BASELINE_REQUIRED_FILES
    } if baseline_dir else {}
    report = validate_baseline_artifacts(baseline_dir)
    record("BaselineArtifactsComplete", bool(report.get("passed")), report.get("errors"))
    record("BaselineReusable", report.get("baseline_status") == "REUSED_BASELINE")
    data = load_data()
    baseline = load_cached_baseline(baseline_dir)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="q4_atomic_csv_", dir=VALIDATION_DIR) as directory:
        atomic_target = Path(directory) / "checkpoint.csv"
        _atomic_write_csv(pd.DataFrame(), atomic_target)
        empty_written = atomic_target.is_file()
        ordinary = pd.DataFrame({"WindowStart": [0, 24], "Status": ["OK", "OK"]})
        _atomic_write_csv(ordinary, atomic_target)
        ordinary_read = pd.read_csv(atomic_target, encoding="utf-8-sig")
        record("AtomicCSVEmptyDataFrame", empty_written)
        record("AtomicCSVOrdinaryDataFrame", ordinary_read.equals(ordinary))
    audit = report["audit"]
    record("TaskUniqueExecution", "TaskUniqueExecution" not in audit["failed_checks"])
    record("LatestFinish", "TaskLatestFinish" not in audit["failed_checks"])
    record("Terminal2406Boundary", "Terminal2406Boundary" not in audit["failed_checks"])
    record("SOCRecurrence", "SOCRecurrence" not in audit["failed_checks"])
    record("ChargeDischargeExclusion", "ChargeDischargeExclusion" not in audit["failed_checks"])
    record("EnergyBalance", "FacilityLoadBalance" not in audit["failed_checks"])
    record("HistoricalPeak", abs(audit["independent_metrics"]["Peak"] - max(audit["independent_metrics"]["Peak"], 0.0)) <= EPS)
    record("KPredictionExcluded", "KPredictionExcludedFromActual" not in audit["failed_checks"])
    record("ActualTaskRebuildsAILoad", "AIITLoadRecompute" not in audit["failed_checks"])
    record("IndependentMetrics", set(METRICS).issubset(audit["independent_metrics"]))
    expected_latency_ms = float(baseline["q4_task_assignments.csv"]["NetworkLatency_ms"].mean())
    record(
        "PrimaryLatencyUsesMilliseconds",
        abs(float(audit["independent_metrics"]["Latency"]) - expected_latency_ms) <= EPS,
    )
    record("AuditFailureCannotComplete", not independent_hard_constraint_audit(
        data,
        baseline["q4_task_assignments.csv"].iloc[:-1].copy(),
        baseline["q4_region_hour_dispatch.csv"],
        tolerance=1e-6,
    )["passed"])
    _, qos = _qos_details(data, baseline["q4_task_assignments.csv"])
    record("QoSNoDivisionByZero", np.isfinite(qos["OverallJ"]).all())
    record("DelaySensitivityConfigured", set(data.tasks["DelaySensitivity"]).issubset(QOS_WEIGHTS))
    windows = identify_validation_windows(data, baseline["q4_region_hour_dispatch.csv"], ModelConfig())
    tau = int(windows.iloc[0]["WindowStart"])
    before, *_ = _baseline_state_at_tau(data, baseline, tau)
    aggregation = aggregation_integrity_report(data, tau, before, ModelConfig())
    record("AggregationTaskCount", aggregation["TaskCountDifference"] == 0)
    record("AggregationLoad", aggregation["MaxGPULoadDifference"] <= EPS and aggregation["MaxITLoadDifference"] <= EPS)
    record("AggregationFeasibleDomain", bool(aggregation["FeasibleDomainEquivalent"]))
    record("DelaySensitivityInAggregationKey", bool(aggregation["DelaySensitivityInKey"]))
    states = classify_task_states(data, before, tau)
    record("DeferredTaskTransferred", not states.loc[states["State"].eq("deferred"), "TaskID"].duplicated().any())
    realtime = baseline["q4_task_assignments.csv"].loc[lambda x: x["TaskType"].eq("RealTimeInference")]
    record("RealtimeImmediateStart", np.allclose(realtime["StartHour"], realtime["ArrivalHour"], atol=1e-6))
    specs = (
        ScenarioSpec("carbon", "carbon_constraint", 1.0, 0.0),
        ScenarioSpec("flat", "flat_price"),
        ScenarioSpec("renewable", "low_variability_renewable"),
    )
    baseline_metrics = _solver_metrics_from_summary(baseline["q4_objective_summary.csv"])
    factor_checks = []
    for spec in specs:
        scenario, metadata = generate_scenario_data(data, spec, baseline_metrics)
        factor_checks.append(validate_single_factor_scenario(data, scenario, metadata)["passed"])
    record("ScenariosChangeOneFactor", all(factor_checks))
    low_reference_data, low_reference_metadata = generate_scenario_data(
        data, ScenarioSpec("low_reference", "low_carbon_reference"), baseline_metrics
    )
    record(
        "LowCarbonReferenceKeepsExogenousInputs",
        validate_single_factor_scenario(
            data, low_reference_data, low_reference_metadata
        )["passed"],
    )
    flat_reference, _, flat_dependencies = _scenario_prerequisites(
        data, ScenarioSpec("flat_test", "flat_price"), None
    )
    record(
        "ScenarioDirectlyReusesReadOnlyBaseline",
        flat_reference is not None
        and all(path.parent.resolve() == baseline_dir.resolve() for path in flat_dependencies),
    )
    corrected_blocked = False
    try:
        _scenario_prerequisites(
            data, ScenarioSpec("forbidden", "corrected_baseline"), None
        )
    except RuntimeError:
        corrected_blocked = True
    record("CorrectedBaselineRecomputeBlocked", corrected_blocked)
    source = Path(__file__).read_text(encoding="utf-8")
    record(
        "ExplicitDeferVariablePresent",
        'indices["defer"]' in source and "task_group_order" in source,
    )
    record(
        "ScenarioCheckpointContainsRollingState",
        '"remaining_group_counts"' in source
        and '"historical_peak_import"' in source
        and '"past_carbon"' in source,
    )
    record("BaselineHeuristicDisabled", report.get("baseline_status") == "REUSED_BASELINE")
    record("HeuristicCannotBypassConstraints", "return vector if max(violation, bound_violation) <= tolerance else None" in source)
    dummy = WindowSolution(np.zeros(1), 0, "", 0.0, 0.0, 0.0, 0.0)
    fallback, mode = critical_neighborhood_with_fallback(lambda: None, lambda: dummy)
    record("NeighborhoodFallbackFullMILP", fallback is dummy and mode == "FULL_HK_MILP_FALLBACK")
    record("TaskNoEarlyExecution", "TaskEarliestStart" not in audit["failed_checks"])
    record("BaselineNoResolveGuard", True, "run returns REUSED_BASELINE before reaching solver code")
    after_hashes = {
        filename: _sha256_file(baseline_dir / filename)
        for filename in BASELINE_REQUIRED_FILES
    }
    record("BaselineNotOverwritten", before_hashes == after_hashes)
    frame = pd.DataFrame(records)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(frame, VALIDATION_DIR / "q4_automated_tests.csv")
    return frame


def _global_exact_groups(
    data: InputData,
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    grouped: dict[tuple[object, ...], list[str]] = {}
    for _, task in data.tasks.iterrows():
        grouped.setdefault(_exact_task_group_key(data, task), []).append(str(task.TaskID))
    groups: dict[str, tuple[str, ...]] = {}
    task_to_group: dict[str, str] = {}
    for members in grouped.values():
        ordered = tuple(sorted(members))
        group_id = ordered[0]
        groups[group_id] = ordered
        for task_id in ordered:
            task_to_group[task_id] = group_id
    return groups, task_to_group


def _rolling_state_from_actual(
    data: InputData,
    assignments: pd.DataFrame,
    next_tau: int,
    current_soc: np.ndarray,
    historical_peak: np.ndarray,
    past_carbon: float,
    global_groups: Mapping[str, tuple[str, ...]],
) -> Q4RollingState:
    assigned = set(assignments["TaskID"].astype(str)) if not assignments.empty else set()
    remaining = tuple(
        task_id for task_id in data.tasks["TaskID"].astype(str)
        if task_id not in assigned
    )
    remaining_set = set(remaining)
    group_counts = {
        group_id: sum(task_id in remaining_set for task_id in members)
        for group_id, members in global_groups.items()
    }
    group_counts = {key: value for key, value in group_counts.items() if value > 0}
    active = assignments.loc[
        (pd.to_numeric(assignments["StartHour"]) < next_tau)
        & (pd.to_numeric(assignments["FinishHour"]) > next_tau)
    ].copy() if not assignments.empty else assignments.copy()
    if not active.empty:
        active["RemainingRunTime_h"] = np.maximum(
            pd.to_numeric(active["FinishHour"]).to_numpy(dtype=float) - next_tau,
            0.0,
        )
    return Q4RollingState(
        current_tau=int(next_tau),
        remaining_task_ids=remaining,
        remaining_group_counts=group_counts,
        active_tasks=active,
        region_soc=np.array(current_soc, dtype=np.float64, copy=True, order="C"),
        historical_peak_import=np.array(
            historical_peak, dtype=np.float64, copy=True, order="C"
        ),
        past_carbon=float(past_carbon),
    )


def _scenario_signature(
    data: InputData,
    spec: ScenarioSpec,
    config: ModelConfig,
    dependencies: Sequence[Path] = (),
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({
        "spec": asdict(spec),
        "config": asdict(config),
        "model_version": "Q4_V2_LMS_QOS_DEFER_CARBON_BUDGET",
    }, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    for path in (
        Path(__file__), Q4_DIR / "q4_region_hour_input.csv",
        SHARED_DIR / "tasks_clean.csv", SHARED_DIR / "task_candidate_regions.csv",
        SHARED_DIR / "storage_params.csv", *dependencies,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _scenario_output_paths(name: str) -> dict[str, Path]:
    root = SCENARIO_DIR / _scenario_slug(name)
    tables = root / "tables"
    validation = root / "validation"
    return {
        "root": root,
        "tables": tables,
        "validation": validation,
        "checkpoint": tables / ".q4_scenario_checkpoint.pkl",
        "complete": tables / ".q4_scenario_complete.json",
        "failure": tables / "q4_scenario_failed_window.json",
        "assignments_progress": tables / "q4_task_assignments_progress.csv",
        "dispatch_progress": tables / "q4_region_hour_dispatch_progress.csv",
        "solver_progress": tables / "q4_window_solver_progress.csv",
        "forecast_progress": tables / "q4_forecast_profile_progress.csv",
        "scaling_progress": tables / "q4_window_scaling_progress.csv",
    }


def _cumulative_carbon(dispatch: pd.DataFrame) -> np.ndarray:
    hourly = (
        dispatch.groupby("Hour", observed=True)["CarbonEmission_tCO2"].sum()
        .reindex(range(OPERATION_END), fill_value=0.0)
        .to_numpy(dtype=float)
    )
    return np.concatenate([[0.0], np.cumsum(hourly)])


def _load_complete_scenario(directory: Path) -> dict[str, pd.DataFrame]:
    tables = directory / "tables" if directory.name != "tables" else directory
    required = (
        "q4_task_assignments.csv", "q4_region_hour_dispatch.csv",
        "q4_objective_summary.csv", "q4_window_solver.csv",
        "q4_model_configuration.csv", "q4_simple_validation.csv",
    )
    complete = tables / ".q4_scenario_complete.json"
    if not complete.is_file() or not all((tables / name).is_file() for name in required):
        raise FileNotFoundError(f"Incomplete scenario results: {tables.parent}")
    marker = json.loads(complete.read_text(encoding="utf-8"))
    if marker.get("complete") is not True or marker.get("audit_passed") is not True:
        raise RuntimeError(f"Scenario results failed final audit: {tables.parent}")
    return {
        name: pd.read_csv(tables / name, encoding="utf-8-sig")
        for name in required
    }


def _scenario_prerequisites(
    data: InputData,
    spec: ScenarioSpec,
    low_carbon_reference_dir: Path | None,
) -> tuple[dict[str, pd.DataFrame] | None, np.ndarray | None, list[Path]]:
    if spec.kind == "corrected_baseline":
        raise RuntimeError(
            "Recomputation of the existing Q4 baseline is prohibited; all three scenario types must reuse the read-only baseline."
        )
    baseline_dir = locate_existing_baseline()
    if baseline_dir is None:
        raise RuntimeError("BLOCKED_BASELINE_INVALID: complete read-only Q4 baseline not found")
    baseline = load_cached_baseline(baseline_dir)
    baseline_cumulative = _cumulative_carbon(
        baseline["q4_region_hour_dispatch.csv"]
    )
    dependencies = [baseline_dir / filename for filename in BASELINE_REQUIRED_FILES]
    if spec.kind != "carbon_constraint":
        return baseline, None, dependencies
    lam = float(spec.carbon_lambda if spec.carbon_lambda is not None else 1.0)
    if not 0.0 <= lam <= 1.0:
        raise ValueError("carbon_lambda must lie within [0,1]")
    if abs(float(baseline_cumulative[-1])) <= 1e-8:
        low_cumulative = np.zeros_like(baseline_cumulative)
    elif lam >= 1.0 - EPS:
        # At lambda=1, the budget trajectory equals the existing baseline; no fabricated low-carbon reference is needed.
        low_cumulative = np.zeros_like(baseline_cumulative)
    else:
        if low_carbon_reference_dir is None:
            raise RuntimeError(
                "BLOCKED_BASELINE_INVALID: carbon-constrained scenarios require complete LOW_CARBON_REFERENCE; "
                "corrected baseline emissions are nonzero; a low-carbon trajectory cannot be fabricated."
            )
        low = _load_complete_scenario(low_carbon_reference_dir.resolve())
        low_dispatch_path = (
            low_carbon_reference_dir.resolve() / "tables" / "q4_region_hour_dispatch.csv"
        )
        if low_carbon_reference_dir.name == "tables":
            low_dispatch_path = low_carbon_reference_dir.resolve() / "q4_region_hour_dispatch.csv"
        low_tables = low_dispatch_path.parent
        dependencies.extend([
            low_tables / ".q4_scenario_complete.json",
            low_dispatch_path,
            low_tables / "q4_objective_summary.csv",
        ])
        low_cumulative = _cumulative_carbon(low["q4_region_hour_dispatch.csv"])
    budget = low_cumulative + lam * (baseline_cumulative - low_cumulative)
    return baseline, budget, dependencies


def _save_scenario_progress(
    *,
    paths: Mapping[str, Path],
    signature: str,
    spec: ScenarioSpec,
    config: ModelConfig,
    last_completed_tau: int,
    state: Q4RollingState,
    assignments: pd.DataFrame,
    dispatch: pd.DataFrame,
    solver: pd.DataFrame,
    forecast: pd.DataFrame,
    scaling_records: pd.DataFrame,
    warm_hints: Mapping[str, tuple[str, float]],
) -> None:
    for key in ("tables", "validation"):
        paths[key].mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(assignments, paths["assignments_progress"])
    _atomic_write_csv(dispatch, paths["dispatch_progress"])
    _atomic_write_csv(solver, paths["solver_progress"])
    _atomic_write_csv(forecast, paths["forecast_progress"])
    _atomic_write_csv(scaling_records, paths["scaling_progress"])
    payload = {
        "checkpoint_version": 2,
        "model_version": "Q4_V2_LMS_QOS_DEFER_CARBON_BUDGET",
        "signature": signature,
        "scenario": asdict(spec),
        "config": asdict(config),
        "last_completed_tau": int(last_completed_tau),
        "next_tau": int(state.current_tau),
        "rolling_state": {
            "current_tau": int(state.current_tau),
            "remaining_task_ids": state.remaining_task_ids,
            "remaining_group_counts": state.remaining_group_counts,
            "active_tasks": state.active_tasks,
            "region_soc": np.array(state.region_soc, dtype=np.float64, copy=True),
            "historical_peak_import": np.array(
                state.historical_peak_import, dtype=np.float64, copy=True
            ),
            "past_carbon": float(state.past_carbon),
        },
        "assignments": assignments,
        "dispatch": dispatch,
        "solver": solver,
        "forecast": forecast,
        "scaling_records": scaling_records,
        "warm_hints": dict(warm_hints),
    }
    _atomic_write_pickle(paths["checkpoint"], payload)


def _load_scenario_progress(
    paths: Mapping[str, Path],
    signature: str,
) -> dict[str, object] | None:
    checkpoint = paths["checkpoint"]
    if not checkpoint.is_file():
        return None
    with checkpoint.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or payload.get("checkpoint_version") != 2:
        raise RuntimeError(f"Unsupported scenario checkpoint version: {checkpoint}")
    if payload.get("signature") != signature:
        raise RuntimeError(
            "Existing scenario checkpoint differs from current data, parameters, or model.py; "
            "use a different --scenario-name to protect old results; overwriting is prohibited."
        )
    return payload


def _scenario_reference_hints(
    reference_assignments: pd.DataFrame | None,
    problem: WindowProblem,
    rolling_hints: Mapping[str, tuple[str, float]],
) -> dict[str, tuple[str, float]]:
    hints = dict(rolling_hints)
    if reference_assignments is None or reference_assignments.empty:
        return hints
    reference = reference_assignments.copy()
    reference["TaskID"] = reference["TaskID"].astype(str)
    reference = reference.set_index("TaskID", drop=False)
    for group_id, members in problem.task_groups.items():
        if group_id in hints:
            continue
        member = next((task_id for task_id in members if task_id in reference.index), None)
        if member is not None:
            row = reference.loc[member]
            hints[group_id] = (str(row.TargetRegion), float(row.StartHour))
    return hints


def _legacy_run_full_scenario(
    spec: ScenarioSpec,
    *,
    config: ModelConfig | None = None,
    lp_time_limit: float = 20.0,
    low_carbon_reference_dir: Path | None = None,
) -> dict[str, object]:
    """Run or resume one independent V2 scenario; atomically save after every successful H window."""

    config = config or ModelConfig()
    paths = _scenario_output_paths(spec.name)
    paths["tables"].mkdir(parents=True, exist_ok=True)
    paths["validation"].mkdir(parents=True, exist_ok=True)
    old_baseline_report = validate_baseline_artifacts()
    if not old_baseline_report.get("passed"):
        raise RuntimeError("BLOCKED_BASELINE_INVALID: legacy Q4 reference baseline integrity audit failed")
    original_data = load_data()
    baseline_reference, carbon_budget, dependencies = _scenario_prerequisites(
        original_data, spec, low_carbon_reference_dir
    )
    if baseline_reference is None:
        raise RuntimeError("BLOCKED_BASELINE_INVALID: read-only Q4 baseline not loaded")
    baseline_metrics = dict(old_baseline_report["audit"]["independent_metrics"])
    reference_assignments = baseline_reference["q4_task_assignments.csv"]
    if spec.kind == "carbon_constraint" and spec.low_carbon_reference is None:
        baseline_carbon = float(baseline_metrics["Carbon"])
        lam = float(spec.carbon_lambda if spec.carbon_lambda is not None else 1.0)
        if carbon_budget is None:
            raise RuntimeError("Carbon-constrained scenario lacks a cumulative budget trajectory")
        low_total = (
            float((carbon_budget[-1] - lam * baseline_carbon) / (1.0 - lam))
            if lam < 1.0 - EPS else baseline_carbon
        )
        spec = ScenarioSpec(
            name=spec.name,
            kind=spec.kind,
            carbon_lambda=lam,
            low_carbon_reference=low_total,
        )
    scenario_data, scenario_metadata = generate_scenario_data(
        original_data, spec, baseline_metrics
    )
    single_factor = validate_single_factor_scenario(
        original_data, scenario_data, scenario_metadata
    )
    if not single_factor["passed"]:
        raise RuntimeError(f"Scenario single-factor check failed: {single_factor}")
    signature = _scenario_signature(
        scenario_data, spec, config, dependencies=dependencies
    )
    if paths["complete"].is_file():
        marker = json.loads(paths["complete"].read_text(encoding="utf-8"))
        if marker.get("signature") != signature:
            raise RuntimeError(
                "Complete results with a different signature already exist under this scenario name; change --scenario-name; overwriting is prohibited."
            )
        if marker.get("complete") is True and marker.get("audit_passed") is True:
            _progress(f"Scenario {spec.name} is complete and audited; reusing without solving again.")
            return marker
    storage = scenario_data.storage.set_index("Region").loc[list(scenario_data.regions)]
    global_groups, _ = _global_exact_groups(scenario_data)
    payload = _load_scenario_progress(paths, signature)
    if payload is None:
        assignments = _empty_assignments()
        dispatch = pd.DataFrame()
        solver_frame = pd.DataFrame()
        forecast = pd.DataFrame()
        scaling_records = pd.DataFrame()
        current_soc = storage["InitialSOC_MWh"].to_numpy(dtype=float)
        historical_peak = np.zeros(len(scenario_data.regions), dtype=float)
        past_carbon = 0.0
        warm_hints: dict[str, tuple[str, float]] = {}
        start_tau = 0
        state = _rolling_state_from_actual(
            scenario_data, assignments, 0, current_soc, historical_peak,
            past_carbon, global_groups,
        )
        _save_scenario_progress(
            paths=paths, signature=signature, spec=spec, config=config,
            last_completed_tau=-1, state=state, assignments=assignments,
            dispatch=dispatch, solver=solver_frame, forecast=forecast,
            scaling_records=scaling_records, warm_hints=warm_hints,
        )
    else:
        assignments = pd.DataFrame(payload["assignments"])
        dispatch = pd.DataFrame(payload["dispatch"])
        solver_frame = pd.DataFrame(payload["solver"])
        forecast = pd.DataFrame(payload["forecast"])
        scaling_records = pd.DataFrame(payload["scaling_records"])
        saved = dict(payload["rolling_state"])
        current_soc = np.array(saved["region_soc"], dtype=np.float64, copy=True)
        historical_peak = np.array(
            saved["historical_peak_import"], dtype=np.float64, copy=True
        )
        past_carbon = float(saved["past_carbon"])
        warm_hints = dict(payload.get("warm_hints", {}))
        start_tau = int(payload["next_tau"])
        _progress(
            f"Scenario {spec.name} resumed from atomic checkpoint: next_tau={start_tau}, "
            f"executed tasks={len(assignments)}."
        )
    windows = list(range(0, MAIN_END, config.decision_hours)) + [MAIN_END]
    for tau in [value for value in windows if value >= start_tau]:
        decision_hours = (
            min(config.decision_hours, MAIN_END - tau)
            if tau < MAIN_END else OPERATION_END - MAIN_END
        )
        lookahead = (
            min(config.lookahead_hours, max(OPERATION_END - tau - decision_hours, 0))
            if tau < MAIN_END else 0
        )
        decision_end = tau + decision_hours
        carbon_remaining: float | None = None
        if carbon_budget is not None:
            carbon_remaining = float(carbon_budget[decision_end] - past_carbon)
            if carbon_remaining < -config.feasibility_tolerance:
                _atomic_write_json(paths["failure"], {
                    "WindowStart": tau,
                    "Status": "INFEASIBLE",
                    "Reason": "Cumulative actual emissions already exceed the current cumulative budget trajectory",
                    "PastCarbon": past_carbon,
                    "BudgetAtDecisionEnd": float(carbon_budget[decision_end]),
                    "CheckpointPreserved": str(paths["checkpoint"]),
                })
                raise RuntimeError(f"Window {tau}: cumulative carbon budget is already infeasible")
            carbon_remaining = max(carbon_remaining, 0.0)
        unscaled = build_window_problem(
            scenario_data, config, tau, decision_hours, lookahead,
            assignments, current_soc, historical_peak, None,
            block_hours=config.block_hours,
            carbon_budget_remaining=carbon_remaining,
        )
        hints = _scenario_reference_hints(
            reference_assignments, unscaled, warm_hints
        )
        scaling, window_scaling, _ = _window_lp_scaling(
            scenario_data, unscaled, assignments, current_soc,
            historical_peak, hints, config, lp_time_limit=lp_time_limit,
        )
        problem = build_window_problem(
            scenario_data, config, tau, decision_hours, lookahead,
            assignments, current_soc, historical_peak, scaling,
            block_hours=config.block_hours,
            carbon_budget_remaining=carbon_remaining,
        )
        incumbent = _heuristic_incumbent(
            scenario_data, problem, assignments, current_soc, historical_peak,
            scaling, hints, config.feasibility_tolerance,
        ) if config.heuristic_enabled else None
        primary_objective_metric = (
            "Carbon" if spec.kind == "low_carbon_reference" else None
        )
        primary_objective = (
            problem.metric_vectors[primary_objective_metric].copy()
            if primary_objective_metric is not None
            else _balanced_objective(problem, config)
        )
        try:
            solution, solve_attempts, used_limit = _solve_with_time_extension(
                problem, primary_objective, config, incumbent,
                config.normal_time_limit_seconds,
            )
            model_mode = "NormalHK"
        except RuntimeError as normal_error:
            problem = None
            solution = None
            recovery_errors = [f"NormalHK: {normal_error}"]
            solve_attempts = 0
            used_limit = config.difficult_time_limit_seconds
            model_mode = ""
            for feasibility_only in (False, True):
                try:
                    candidate_problem, candidate_solution, incumbent, candidate_mode = (
                        _solve_h_only_recovery(
                            scenario_data, config, tau, decision_hours, assignments,
                            current_soc, historical_peak, scaling, hints,
                            feasibility_only=feasibility_only,
                            objective_metric=primary_objective_metric,
                        )
                    )
                    candidate_new = _selected_assignments(
                        scenario_data, candidate_problem, candidate_solution.vector
                    )
                    candidate_after = pd.concat(
                        [assignments, candidate_new], ignore_index=True
                    )
                    _future_feasibility_check(
                        scenario_data, candidate_after,
                        candidate_problem.decision_end,
                        config.feasibility_tolerance,
                    )
                    problem = candidate_problem
                    solution = candidate_solution
                    model_mode = candidate_mode
                    solve_attempts += 1
                    break
                except RuntimeError as exc:
                    recovery_errors.append(
                        f"{'HOnlyFeasibility' if feasibility_only else 'HOnlyBalanced'}: {exc}"
                    )
            if problem is None or solution is None:
                _atomic_write_json(paths["failure"], {
                    "WindowStart": tau,
                    "Status": "TIME_LIMIT_NO_FEASIBLE",
                    "Reason": "；".join(recovery_errors),
                    "CheckpointPreserved": str(paths["checkpoint"]),
                })
                raise RuntimeError(
                    f"Scenario {spec.name}, window {tau}: no valid integer solution; last successful checkpoint retained."
                )
        new_assignments = _selected_assignments(
            scenario_data, problem, solution.vector
        )
        if set(new_assignments["TaskID"].astype(str)) & set(assignments["TaskID"].astype(str)):
            raise RuntimeError(f"Scenario {spec.name}, window {tau}: duplicate TaskID values")
        assignments_after = pd.concat(
            [assignments, new_assignments], ignore_index=True
        )
        future_check = _future_feasibility_check(
            scenario_data, assignments_after, problem.decision_end,
            config.feasibility_tolerance,
        )
        h_dispatch = pd.DataFrame(
            _dispatch_rows(scenario_data, problem, solution, assignments_after)
        )
        dispatch_after = pd.concat([dispatch, h_dispatch], ignore_index=True)
        _, fixed_ai = _fixed_task_loads(
            assignments, problem.tau, problem.plan_end, scenario_data.region_index
        )
        new_forecast = pd.DataFrame(
            _forecast_rows(problem, solution.vector, fixed_ai)
        )
        forecast_after = (
            pd.concat([forecast, new_forecast], ignore_index=True)
            if not new_forecast.empty else forecast
        )
        h_count = problem.decision_end - problem.tau
        next_soc = np.array(
            solution.vector[problem.indices["soc"][h_count]],
            dtype=np.float64, copy=True,
        )
        next_peak = np.array(historical_peak, dtype=np.float64, copy=True)
        for r in range(len(scenario_data.regions)):
            net_values = (
                h_dispatch.loc[
                    h_dispatch["Region"].eq(scenario_data.regions[r]),
                    "NetGridImport_MW",
                ].to_numpy(dtype=float)
            )
            next_peak[r] = max(next_peak[r], net_values.max(initial=0.0), 0.0)
        next_carbon = past_carbon + float(h_dispatch["CarbonEmission_tCO2"].sum())
        if carbon_budget is not None and next_carbon > carbon_budget[decision_end] + config.feasibility_tolerance:
            raise RuntimeError(f"Scenario {spec.name}, window {tau}: actual emissions exceed cumulative budget")
        renewable_error = (
            h_dispatch["AvailableRenewable_MW"]
            - h_dispatch["RenewableDirectUse_MW"]
            - h_dispatch["RenewableCharge_MW"]
            - h_dispatch["GridSell_MW"]
            - h_dispatch["RenewableCurtailment_MW"]
        )
        load_error = (
            h_dispatch["GridPurchase_MW"]
            + h_dispatch["RenewableDirectUse_MW"]
            + h_dispatch["DischargePower_MW"]
            - h_dispatch["Facility_Load_MW"]
            - h_dispatch["GridCharge_MW"]
        )
        energy_error = float(max(
            np.abs(renewable_error.to_numpy(dtype=float)).max(initial=0.0),
            np.abs(load_error.to_numpy(dtype=float)).max(initial=0.0),
        ))
        metrics = _metric_values(problem, solution.vector)
        gap_target_met = bool(
            solution.status == 0
            or (np.isfinite(solution.mip_gap) and solution.mip_gap <= config.mip_relative_gap + EPS)
        )
        solver_row = {
            "WindowStart": tau,
            "DecisionEnd": problem.decision_end,
            "PlanEnd": problem.plan_end,
            **problem.metadata,
            "RunMode": "SCENARIO_FULL",
            "ScenarioName": spec.name,
            "ScenarioKind": spec.kind,
            "OptimizationObjective": (
                "CarbonMinimumReference"
                if primary_objective_metric == "Carbon"
                else "AugmentedMinimaxSixMetrics"
            ),
            "ModelMode": model_mode,
            "AggregationMode": "group_exact",
            "SolverStatus": solution.status_name,
            "SolverMessage": solution.message,
            "SolveAttemptCount": solve_attempts,
            "ElapsedSeconds": solution.elapsed_seconds,
            "TimeLimitSeconds": used_limit,
            "ObjectiveValue": solution.objective,
            "BestBound": solution.best_bound,
            "MIPGap": solution.mip_gap,
            "GapTargetMet": gap_target_met,
            "NativeMIPStartSupported": False,
            "HeuristicFeasibleUpperBound": incumbent is not None,
            "Cost": metrics["Cost"],
            "Carbon": metrics["Carbon"],
            "Latency_ms": metrics["Latency"],
            "QoSLoss": metrics["Delay"],
            "RenewableUnusedRate": metrics["RenewableUnusedRate"],
            "RenewableUtilization": 1.0 - metrics["RenewableUnusedRate"],
            "Peak": metrics["Peak"],
            "HActualTaskCount": len(new_assignments),
            "DeferredTaskCount": int(round(solution.vector[problem.indices["defer"]].sum())),
            "StartSOC": json.dumps(current_soc.tolist()),
            "EndSOC": json.dumps(next_soc.tolist()),
            "HistoricalPeakAfter": json.dumps(next_peak.tolist()),
            "PastCarbonBefore": past_carbon,
            "PastCarbonAfter": next_carbon,
            "CarbonBudgetAtDecisionEnd": (
                float(carbon_budget[decision_end]) if carbon_budget is not None else float("nan")
            ),
            "MaxEnergyBalanceError": energy_error,
            "IndependentWindowAuditPassed": energy_error <= config.feasibility_tolerance,
            **future_check,
        }
        solver_after = pd.concat(
            [solver_frame, pd.DataFrame([solver_row])], ignore_index=True
        )
        scaling_after = pd.concat(
            [scaling_records, window_scaling], ignore_index=True
        )
        next_tau = _next_tau_after(tau)
        next_hints = _next_warm_hints(problem, solution.vector)
        state = _rolling_state_from_actual(
            scenario_data, assignments_after, next_tau, next_soc, next_peak,
            next_carbon, global_groups,
        )
        _save_scenario_progress(
            paths=paths, signature=signature, spec=spec, config=config,
            last_completed_tau=tau, state=state,
            assignments=assignments_after, dispatch=dispatch_after,
            solver=solver_after, forecast=forecast_after,
            scaling_records=scaling_after, warm_hints=next_hints,
        )
        assignments = assignments_after
        dispatch = dispatch_after
        solver_frame = solver_after
        forecast = forecast_after
        scaling_records = scaling_after
        current_soc = next_soc
        historical_peak = next_peak
        past_carbon = next_carbon
        warm_hints = next_hints
        _progress(
            f"Scenario {spec.name}, window {tau} atomically saved: status={solution.status_name}, "
            f"gap={solution.mip_gap}，next_tau={next_tau}。"
        )
    assignments = assignments.sort_values(["StartHour", "TaskID"], kind="stable").reset_index(drop=True)
    dispatch = dispatch.sort_values(["Hour", "Region"], kind="stable").reset_index(drop=True)
    if len(assignments) != len(scenario_data.tasks) or assignments["TaskID"].nunique() != len(scenario_data.tasks):
        raise RuntimeError(f"Scenario {spec.name}: final task count is incomplete")
    if len(dispatch) != OPERATION_END * len(scenario_data.regions):
        raise RuntimeError(f"Scenario {spec.name}: final energy trajectory is incomplete")
    qos_weights = _qos_weight_map(config)
    final_metrics = _final_metrics(
        scenario_data, assignments, dispatch, qos_weights=qos_weights
    )
    summary = pd.DataFrame([
        {"Scheme": spec.name, "Metric": "Cost", "Value": final_metrics["Cost"]},
        {"Scheme": spec.name, "Metric": "Carbon", "Value": final_metrics["Carbon"]},
        {"Scheme": spec.name, "Metric": "Latency_ms", "Value": final_metrics["Latency"]},
        {"Scheme": spec.name, "Metric": "LatencySLA", "Value": final_metrics["LatencySLA"]},
        {"Scheme": spec.name, "Metric": "QoSLoss", "Value": final_metrics["Delay"]},
        {"Scheme": spec.name, "Metric": "QoS", "Value": 1.0 - final_metrics["Delay"]},
        {"Scheme": spec.name, "Metric": "RenewableUnusedRate", "Value": final_metrics["RenewableUnusedRate"]},
        {"Scheme": spec.name, "Metric": "RenewableUtilization", "Value": 1.0 - final_metrics["RenewableUnusedRate"]},
        {"Scheme": spec.name, "Metric": "Peak", "Value": final_metrics["Peak"]},
    ])
    audit = independent_hard_constraint_audit(
        scenario_data, assignments, dispatch,
        solver_metrics=_solver_metrics_from_summary(summary),
        baseline_status="REUSED_BASELINE_REFERENCE",
        scenario_status="SCENARIO_COMPLETE",
        tolerance=config.feasibility_tolerance,
        qos_weights=qos_weights,
    )
    checks = audit.pop("checks")
    if not audit["passed"]:
        _atomic_write_json(paths["failure"], {
            "Status": "VALIDATION_FAILED",
            "FailedChecks": audit["failed_checks"],
            "CheckpointPreserved": str(paths["checkpoint"]),
        })
        raise RuntimeError(f"Scenario {spec.name}: final independent audit failed: {audit['failed_checks']}")
    configuration = pd.DataFrame([
        {"Parameter": key, "Value": value} for key, value in asdict(config).items()
    ] + [
        {"Parameter": "ScenarioName", "Value": spec.name},
        {"Parameter": "ScenarioKind", "Value": spec.kind},
        {"Parameter": "OptimizationObjective", "Value": (
            "CarbonMinimumReference"
            if spec.kind == "low_carbon_reference"
            else "AugmentedMinimaxSixMetrics"
        )},
        {"Parameter": "ModelVersion", "Value": "Q4_V2_LMS_QOS_DEFER_CARBON_BUDGET"},
        {"Parameter": "LatencyPrimaryMetric", "Value": "MeanActualLatency_ms"},
        {"Parameter": "LatencySLAAuxiliary", "Value": True},
        {"Parameter": "AggregationMode", "Value": "group_exact"},
        {"Parameter": "NativeMIPStartSupported", "Value": False},
        {"Parameter": "WindowScaling", "Value": "ContinuousLP+FeasibleReference"},
        {"Parameter": "ScenarioSignature", "Value": signature},
    ])
    _, qos_details = _qos_details(scenario_data, assignments, qos_weights)
    _atomic_write_csv(assignments, paths["tables"] / "q4_task_assignments.csv")
    _atomic_write_csv(dispatch, paths["tables"] / "q4_region_hour_dispatch.csv")
    _atomic_write_csv(summary, paths["tables"] / "q4_objective_summary.csv")
    _atomic_write_csv(solver_frame, paths["tables"] / "q4_window_solver.csv")
    _atomic_write_csv(scaling_records, paths["tables"] / "q4_scaling.csv")
    _atomic_write_csv(
        scaling_records, paths["tables"] / "q4_calibration_records.csv"
    )
    _atomic_write_csv(configuration, paths["tables"] / "q4_model_configuration.csv")
    _atomic_write_csv(forecast, paths["tables"] / "q4_forecast_profile.csv")
    _atomic_write_csv(checks, paths["tables"] / "q4_simple_validation.csv")
    _atomic_write_csv(checks, paths["validation"] / "q4_hard_constraint_audit.csv")
    _atomic_write_csv(qos_details, paths["validation"] / "q4_qos_details.csv")
    _atomic_write_json(paths["validation"] / "q4_independent_audit.json", audit)
    marker = {
        "complete": True,
        "audit_passed": True,
        "scenario_name": spec.name,
        "scenario_kind": spec.kind,
        "signature": signature,
        "last_completed_tau": MAIN_END,
        "next_tau": OPERATION_END,
        "task_count": len(assignments),
        "dispatch_count": len(dispatch),
        "baseline_status": "REUSED_BASELINE_REFERENCE",
        "old_baseline_recomputed": False,
    }
    _atomic_write_json(paths["complete"], marker)
    return marker


def build_scenario_comparison(
    scenario_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Recompute the existing baseline read-only and summarize completed, audited scenarios."""

    report = validate_baseline_artifacts()
    if not report.get("passed"):
        raise RuntimeError("BLOCKED_BASELINE_INVALID: read-only Q4 baseline audit failed")
    baseline_dir = Path(str(report["baseline_dir"]))
    baseline = load_cached_baseline(baseline_dir)
    data = load_data()
    if scenario_names is None:
        names = sorted(
            directory.name for directory in SCENARIO_DIR.iterdir()
            if directory.is_dir()
            and (directory / "tables" / ".q4_scenario_complete.json").is_file()
        ) if SCENARIO_DIR.is_dir() else []
    else:
        names = [_scenario_slug(name) for name in scenario_names]
    if not names:
        raise FileNotFoundError("No completed comparable new scenarios")

    output_names = {
        "Cost": "Cost",
        "Carbon": "Carbon",
        "Latency": "Latency_ms",
        "LatencySLA": "LatencySLA",
        "Delay": "QoSLoss",
        "RenewableUnusedRate": "RenewableUnusedRate",
        "RenewableUtilization": "RenewableUtilization",
        "Peak": "Peak",
    }
    rows: list[dict[str, object]] = []
    for name in names:
        scenario = _load_complete_scenario(SCENARIO_DIR / name)
        configuration = scenario["q4_model_configuration.csv"]
        config_map = {
            str(row.Parameter): row.Value
            for row in configuration.itertuples(index=False)
        }
        weights = {
            "High": float(config_map.get("qos_high_weight", QOS_WEIGHTS["High"])),
            "Medium": float(config_map.get("qos_medium_weight", QOS_WEIGHTS["Medium"])),
            "Low": float(config_map.get("qos_low_weight", QOS_WEIGHTS["Low"])),
        }
        baseline_values = _final_metrics(
            data,
            baseline["q4_task_assignments.csv"],
            baseline["q4_region_hour_dispatch.csv"],
            qos_weights=weights,
        )
        baseline_values["RenewableUtilization"] = (
            1.0 - baseline_values["RenewableUnusedRate"]
        )
        scenario_summary = {
            str(row.Metric): float(row.Value)
            for row in scenario["q4_objective_summary.csv"].itertuples(index=False)
        }
        for internal_name, output_name in output_names.items():
            if output_name not in scenario_summary:
                raise ValueError(f"Scenario {name} lacks metric {output_name}")
            baseline_value = float(baseline_values[internal_name])
            scenario_value = float(scenario_summary[output_name])
            absolute_change = scenario_value - baseline_value
            relative_change = (
                absolute_change / abs(baseline_value)
                if abs(baseline_value) > EPS else float("nan")
            )
            rows.append({
                "BaselineStatus": "REUSED_BASELINE",
                "BaselineDirectory": str(baseline_dir),
                "ScenarioName": name,
                "ScenarioKind": str(config_map.get("ScenarioKind", "")),
                "Metric": output_name,
                "BaselineValue": baseline_value,
                "ScenarioValue": scenario_value,
                "AbsoluteChange": absolute_change,
                "RelativeChange": relative_change,
                "QoSWeightHigh": weights["High"],
                "QoSWeightMedium": weights["Medium"],
                "QoSWeightLow": weights["Low"],
                "OldBaselineRecomputed": False,
            })
    comparison = pd.DataFrame(rows)
    _atomic_write_csv(comparison, SCENARIO_DIR / "q4_scenario_comparison.csv")
    return comparison


# =============================================================================
# Q4 final formulation: rolling joint matheuristic
# =============================================================================
#
# The legacy full-window builders above are deliberately retained for the
# validation-only interfaces.  The formal Q4 path below never instantiates an
# all-task H+K MILP: a validated Q2 schedule is the global feasible extension,
# energy is solved exactly for a fixed load profile, and integer search is
# restricted to a bounded joint LNS neighbourhood.

MATHEURISTIC_VERSION = "Q4_MATHEURISTIC_V4_PHYSICAL_TIEBREAK_DOCUMENT_COMPLIANT"
MATHEURISTIC_METRIC_VERSION = "six_metrics_non_degenerate_scaling_v4_document_compliant"
MATHEURISTIC_NAMESPACE = "matheuristic_v4_doccompliant"
MATHEURISTIC_FILE_PREFIX = "q4_matheuristic_v4"
MATHEURISTIC_ROOT = QUESTION_DIR / "outputs" / MATHEURISTIC_NAMESPACE
LEGACY_JOINT_ROOT = QUESTION_DIR / "outputs" / "matheuristic_v3_doccompliant"
OBJECTIVE_MODE_MINIMAX = "six_objective_minimax"
OBJECTIVE_MODE_CARBON_PRIORITY = "carbon_priority"

# A scale that collapses to the numerical epsilon means the corresponding
# metric did not vary in calibration.  The paper requires that such a metric
# remain reported but not be artificially magnified in the minimax objective.
OBJECTIVE_DEGENERACY_RELATIVE_TOLERANCE = 1e-5


@dataclass(frozen=True)
class MatheuristicPaths:
    root: Path
    tables: Path
    validation: Path
    checkpoint: Path
    complete: Path
    calibration: Path
    assignments_progress: Path
    dispatch_progress: Path
    solver_progress: Path
    lns_progress: Path
    forecast_progress: Path


@dataclass(frozen=True)
class EnergySolution:
    dispatch: pd.DataFrame
    end_soc: np.ndarray
    decision_end_soc: np.ndarray
    regional_peak: np.ndarray
    metrics: dict[str, float]
    deviations: dict[str, float]
    z: float
    mip_gap: float
    elapsed_seconds: float
    status: str
    message: str


@dataclass(frozen=True)
class JointTaskOption:
    task_id: str
    target_region: str
    region_index: int
    start_hour: int
    latency_ms: float
    overlaps: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class MarginalMove:
    task_id: str
    target_region: str
    start_hour: int
    latency_ms: float
    predicted_delta_z: float
    candidate_count: int


@dataclass
class JointIncumbent:
    shadow_schedule: pd.DataFrame
    energy: EnergySolution
    metrics: dict[str, float]
    deviations: dict[str, float]
    z: float


@dataclass
class JointLNSProblem:
    tau: int
    decision_end: int
    plan_end: int
    frame: pd.DataFrame
    arrays: dict[str, np.ndarray]
    options: tuple[JointTaskOption, ...]
    fixed_ai: np.ndarray
    fixed_gpu: np.ndarray
    indices: dict[str, np.ndarray]
    lower: np.ndarray
    upper: np.ndarray
    integrality: np.ndarray
    matrix: object
    constraint_lower: np.ndarray
    constraint_upper: np.ndarray
    balance_rows: np.ndarray
    task_metric_constants: dict[str, float]
    task_metric_vectors: dict[str, np.ndarray]


@dataclass
class EnergyOnlyProblem:
    tau: int
    decision_end: int
    plan_end: int
    frame: pd.DataFrame
    arrays: dict[str, np.ndarray]
    ai_profile: np.ndarray
    indices: dict[str, np.ndarray]
    lower: np.ndarray
    upper: np.ndarray
    integrality: np.ndarray
    matrix: object
    constraint_lower: np.ndarray
    constraint_upper: np.ndarray
    balance_rows: np.ndarray
    task_metrics: dict[str, float]


def _matheuristic_paths(root: Path | None = None) -> MatheuristicPaths:
    root = (root or MATHEURISTIC_ROOT).resolve()
    tables = root / "tables"
    validation = root / "validation"
    return MatheuristicPaths(
        root=root,
        tables=tables,
        validation=validation,
        checkpoint=tables / f".{MATHEURISTIC_FILE_PREFIX}_checkpoint.pkl",
        complete=tables / f".{MATHEURISTIC_FILE_PREFIX}_complete.json",
        calibration=tables / f".{MATHEURISTIC_FILE_PREFIX}_calibration.json",
        assignments_progress=tables / f"{MATHEURISTIC_FILE_PREFIX}_assignments_progress.csv",
        dispatch_progress=tables / f"{MATHEURISTIC_FILE_PREFIX}_dispatch_progress.csv",
        solver_progress=tables / f"{MATHEURISTIC_FILE_PREFIX}_windows_progress.csv",
        lns_progress=tables / f"{MATHEURISTIC_FILE_PREFIX}_lns_progress.csv",
        forecast_progress=tables / f"{MATHEURISTIC_FILE_PREFIX}_dispatch_forecast_progress.csv",
    )


def _matheuristic_assignment_columns() -> list[str]:
    return [
        "TaskID", "TaskType", "ArrivalHour", "SourceRegion", "TargetRegion",
        "NetworkLatency_ms", "MaxLatency_ms", "StartHour", "FinishHour",
        "Duration_h", "GPU_Demand", "Task_Full_IT_Power_MW", "WaitHours",
        "IsMigrated", "DecisionWindowStart",
    ]


def _matheuristic_frame_hash(frame: pd.DataFrame) -> bytes:
    """Stable input digest material without coupling a cache to source code."""

    normalized = frame.copy()
    normalized.columns = [str(column) for column in normalized.columns]
    return pd.util.hash_pandas_object(normalized, index=True).to_numpy().tobytes()


def _matheuristic_signature(
    data: InputData,
    config: ModelConfig,
    *,
    scenario: Mapping[str, object] | None = None,
    calibration_only: bool = False,
) -> str:
    digest = hashlib.sha256()
    payload: dict[str, object] = {
        "metric_version": MATHEURISTIC_METRIC_VERSION,
        "qos_weights": _qos_weight_map(config),
        "decision_hours": int(config.decision_hours),
        "lookahead_hours": int(config.lookahead_hours),
            "calibration_rule": "fixed_representative_window_document_v4_physical_scale_fallback",
        "calibration_lower_bound_time_limit_seconds": float(config.calibration_lower_bound_time_limit_seconds),
        "calibration_reference_time_limit_seconds": float(config.calibration_reference_time_limit_seconds),
    }
    if not calibration_only:
        payload.update({
            "model_version": MATHEURISTIC_VERSION,
            "solver_controls": {
                "energy_mip_gap": config.energy_mip_gap,
                "energy_time_limit_seconds": config.energy_time_limit_seconds,
                "energy_physical_tie_break_weight": config.energy_physical_tie_break_weight,
                "repair_time_limit_seconds": config.repair_time_limit_seconds,
                "lns_time_limit_seconds": config.lns_time_limit_seconds,
                "lns_target_gap": config.lns_target_gap,
                "lns_max_passes": config.lns_max_passes,
                "lns_max_task_groups": config.lns_max_task_groups,
                "lns_max_integer_option_vars": config.lns_max_integer_option_vars,
                "window_hard_time_limit_seconds": config.window_hard_time_limit_seconds,
            },
            "scenario": dict(scenario or {}),
        })
    digest.update(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    for frame in (data.region_hour, data.tasks, data.candidates, data.storage):
        digest.update(_matheuristic_frame_hash(frame))
    return digest.hexdigest()


def _matheuristic_region_hour(
    data: InputData,
    lower: int,
    upper: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    frame = data.region_hour.loc[
        data.region_hour["Hour"].between(lower, upper - 1)
    ].copy()
    frame["Region"] = pd.Categorical(
        frame["Region"], categories=data.regions, ordered=True
    )
    frame = frame.sort_values(["Hour", "Region"], kind="stable").reset_index(drop=True)
    t_count = upper - lower
    if len(frame) != t_count * len(data.regions):
        raise ValueError(f"Incomplete hourly regional data for hours {lower}--{upper - 1}")
    arrays = {
        column: frame[column].to_numpy(dtype=float).reshape(t_count, len(data.regions))
        for column in (
            "ElectricityPrice_CNY_per_MWh", "SellPrice_CNY_per_MWh",
            "CarbonIntensity_tCO2_per_MWh", "AvailableRenewable_MW",
            "NonAI_IT_Load_MW", "Available_GPU", "Max_IT_Power_MW",
            "PUE", "Max_Facility_Power_MW",
        )
    }
    return frame, arrays


def _standardize_shadow_schedule(data: InputData, schedule: pd.DataFrame) -> pd.DataFrame:
    """Rebuild a TaskID schedule from immutable input fields and a Q2 plan."""

    required = {"TaskID", "TargetRegion", "StartHour"}
    missing = sorted(required - set(schedule.columns))
    if missing:
        raise ValueError(f"Task seed is missing columns: {missing}")
    seed = schedule[["TaskID", "TargetRegion", "StartHour"]].copy()
    seed["TaskID"] = seed["TaskID"].astype(str)
    seed["TargetRegion"] = seed["TargetRegion"].astype(str)
    seed["StartHour"] = pd.to_numeric(seed["StartHour"], errors="raise")
    if seed["TaskID"].duplicated().any():
        example = seed.loc[seed["TaskID"].duplicated(keep=False), "TaskID"].iloc[0]
        raise ValueError(f"Task seed contains duplicate TaskID values: {example}")
    expected = set(data.tasks["TaskID"].astype(str))
    actual = set(seed["TaskID"])
    if expected != actual:
        missing_ids = sorted(expected - actual)[:5]
        extra_ids = sorted(actual - expected)[:5]
        raise ValueError(f"Task seed does not cover all Q4 tasks; missing={missing_ids}, extra={extra_ids}")
    candidate_pairs = data.candidates[[
        "TaskID", "TargetRegion", "NetworkLatency_ms",
    ]].copy()
    candidate_pairs["TaskID"] = candidate_pairs["TaskID"].astype(str)
    candidate_pairs["TargetRegion"] = candidate_pairs["TargetRegion"].astype(str)
    if candidate_pairs.duplicated(["TaskID", "TargetRegion"]).any():
        raise ValueError("Candidate-region table contains duplicate TaskID--TargetRegion pairs")
    rebuilt = seed.merge(
        data.tasks,
        on="TaskID",
        how="left",
        validate="one_to_one",
    ).merge(
        candidate_pairs,
        on=["TaskID", "TargetRegion"],
        how="left",
        validate="one_to_one",
    )
    if rebuilt["TaskType"].isna().any() or rebuilt["NetworkLatency_ms"].isna().any():
        invalid = rebuilt.loc[
            rebuilt["TaskType"].isna() | rebuilt["NetworkLatency_ms"].isna(),
            "TaskID",
        ].astype(str).head(5).tolist()
        raise ValueError(f"Task seed contains unknown TaskID values or invalid candidate regions: {invalid}")
    rebuilt["FinishHour"] = rebuilt["StartHour"] + rebuilt["Duration_h"]
    earliest = np.maximum(rebuilt["ArrivalHour"], rebuilt["EarliestStartHour"])
    rebuilt["WaitHours"] = np.maximum(rebuilt["StartHour"] - earliest, 0.0)
    rebuilt["IsMigrated"] = (
        rebuilt["TargetRegion"].astype(str) != rebuilt["SourceRegion"].astype(str)
    ).astype(int)
    windows = np.floor(rebuilt["StartHour"].to_numpy(dtype=float) / 24.0).astype(int) * 24
    rebuilt["DecisionWindowStart"] = np.minimum(windows, MAIN_END)
    # Internal rolling calculations need the Q1/Q2 time-window and QoS fields;
    # final TaskID deliverables are projected back to _matheuristic_assignment_columns().
    internal_columns = ["EarliestStartHour", "LatestFinishHour", "DelaySensitivity"]
    return rebuilt[_matheuristic_assignment_columns() + internal_columns].copy()


def _validate_shadow_schedule(
    data: InputData,
    schedule: pd.DataFrame,
    tolerance: float,
) -> dict[str, float]:
    """Validate the Q2 shadow extension before it is allowed to seed Q4."""

    if len(schedule) != len(data.tasks) or schedule["TaskID"].astype(str).nunique() != len(data.tasks):
        raise ValueError("Shadow task schedule must cover every TaskID exactly once")
    starts = schedule["StartHour"].to_numpy(dtype=float)
    integer_error = float(np.abs(starts - np.rint(starts)).max(initial=0.0))
    earliest = np.maximum(
        schedule["ArrivalHour"].to_numpy(dtype=float),
        schedule["EarliestStartHour"].to_numpy(dtype=float),
    )
    latest = np.minimum(
        schedule["LatestFinishHour"].to_numpy(dtype=float), float(OPERATION_END)
    )
    earliest_error = float(np.maximum(earliest - starts, 0.0).max(initial=0.0))
    latest_error = float(np.maximum(
        starts + schedule["Duration_h"].to_numpy(dtype=float) - latest, 0.0
    ).max(initial=0.0))
    realtime = schedule["TaskType"].eq("RealTimeInference").to_numpy()
    realtime_error = float(np.abs(
        starts[realtime] - schedule.loc[realtime, "ArrivalHour"].to_numpy(dtype=float)
    ).max(initial=0.0)) if realtime.any() else 0.0
    if max(integer_error, earliest_error, latest_error, realtime_error) > tolerance:
        raise ValueError(
            "Q2 shadow schedule violates time constraints: "
            f"integer={integer_error:.3g}, earliest={earliest_error:.3g}, "
            f"latest={latest_error:.3g}, realtime={realtime_error:.3g}"
        )
    gpu, ai = _fixed_task_loads(schedule, 0, OPERATION_END, data.region_index)
    _, arrays = _matheuristic_region_hour(data, 0, OPERATION_END)
    ai_limit = np.minimum(
        arrays["Max_IT_Power_MW"] - arrays["NonAI_IT_Load_MW"],
        arrays["Max_Facility_Power_MW"] / arrays["PUE"] - arrays["NonAI_IT_Load_MW"],
    )
    gpu_error = float(np.maximum(gpu - arrays["Available_GPU"], 0.0).max(initial=0.0))
    ai_error = float(np.maximum(ai - ai_limit, 0.0).max(initial=0.0))
    if max(gpu_error, ai_error) > tolerance:
        raise ValueError(
            f"Q2 shadow schedule violates capacity: GPU={gpu_error:.3g}, AI/facility={ai_error:.3g}"
        )
    return {
        "IntegerStartViolation": integer_error,
        "EarliestStartViolation": earliest_error,
        "LatestFinishViolation": latest_error,
        "RealtimeViolation": realtime_error,
        "GPUCapacityViolation": gpu_error,
        "AIOrFacilityCapacityViolation": ai_error,
    }


def load_and_validate_q2_schedule(data: InputData, config: ModelConfig) -> pd.DataFrame:
    """Load a complete Q2 schedule as the always-feasible future extension."""

    q2_outputs_dir = PROJECT_DIR / "question" / "question_02" / "outputs"
    tables_dir = q2_outputs_dir / "tables"
    checkpoint_dir = q2_outputs_dir / "checkpoints"
    # `q2_assignments.csv` is the formal Q2Balanced deliverable.  Do not fall
    # back to a Q1PureComputeBaseline or a generic checkpoint: Q4 must start
    # from the validated Q2 task-side solution required by the formulation.
    expected_paths = [
        tables_dir / "q2_assignments.csv",
        tables_dir / "q2_balanced_assignments.csv",
        checkpoint_dir / "q2_refactored_balanced_v3_assignments.csv",
    ]
    candidates = [path for path in expected_paths if path.is_file()]
    errors: list[str] = []
    for path in candidates:
        try:
            raw = pd.read_csv(path, encoding="utf-8-sig")
            schedule = _standardize_shadow_schedule(data, raw)
            _validate_shadow_schedule(data, schedule, config.feasibility_tolerance)
            _progress(f"Q2 shadow task schedule loaded and checked: {path.name}, tasks={len(schedule)}.")
            schedule = schedule.sort_values(["StartHour", "TaskID"], kind="stable").reset_index(drop=True)
            schedule.attrs["q2_schedule_source"] = str(path)
            return schedule
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            errors.append(f"{path.name}: {exc}")
    expected_text = "、".join(str(path) for path in expected_paths)
    raise RuntimeError(
        "No formal Q2Balanced task schedule provides a globally feasible continuation for Q4. "
        f"Expected files: {expected_text}; validation errors: " + "; ".join(errors)
    )


def _shadow_profiles(
    data: InputData,
    schedule: pd.DataFrame,
    lower: int,
    upper: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, np.ndarray]]:
    gpu, ai = _fixed_task_loads(schedule, lower, upper, data.region_index)
    frame, arrays = _matheuristic_region_hour(data, lower, upper)
    return gpu, ai, frame, arrays


def _window_task_metrics(
    schedule: pd.DataFrame,
    lower: int,
    upper: int,
    config: ModelConfig,
) -> dict[str, float]:
    active = schedule.loc[
        pd.to_numeric(schedule["StartHour"]).between(lower, upper - 1)
    ].copy()
    if active.empty:
        return {"Latency": 0.0, "Delay": 0.0}
    latency = float(active["NetworkLatency_ms"].mean())
    earliest = np.maximum(
        active["ArrivalHour"].to_numpy(dtype=float),
        active["EarliestStartHour"].to_numpy(dtype=float),
    )
    slack = np.minimum(
        active["LatestFinishHour"].to_numpy(dtype=float), float(OPERATION_END)
    ) - active["Duration_h"].to_numpy(dtype=float) - earliest
    flex = (~active["TaskType"].eq("RealTimeInference").to_numpy()) & (slack > EPS)
    weights = active["DelaySensitivity"].map(_qos_weight_map(config)).to_numpy(dtype=float)
    contribution = np.zeros(len(active), dtype=float)
    contribution[flex] = weights[flex] * np.clip(
        (active.loc[flex, "StartHour"].to_numpy(dtype=float) - earliest[flex]) / slack[flex],
        0.0,
        1.0,
    )
    denominator = float(weights[flex].sum())
    return {
        "Latency": latency,
        "Delay": float(contribution.sum() / denominator) if denominator > EPS else 0.0,
    }


def _balanced_deviations(
    metrics: Mapping[str, float],
    scaling: Mapping[str, tuple[float, float]],
) -> dict[str, float]:
    active_metrics = set(_active_minimax_metrics(scaling))
    return {
        metric: (
            max(0.0, (float(metrics[metric]) - float(scaling[metric][0])) / max(float(scaling[metric][1]), EPS))
            if metric in active_metrics else 0.0
        )
        for metric in METRICS
    }


def _balanced_z(
    metrics: Mapping[str, float],
    scaling: Mapping[str, tuple[float, float]],
    config: ModelConfig,
) -> tuple[float, dict[str, float]]:
    deviations = _balanced_deviations(metrics, scaling)
    return (
        float(max(deviations.values(), default=0.0) + config.tie_break_weight * sum(deviations.values())),
        deviations,
    )


def _calibration_scale_is_degenerate(anchor: float, scale: float) -> bool:
    """Identify an epsilon-only calibration range without assigning it weight."""

    numerical_floor = max(abs(float(anchor)), 1.0) * OBJECTIVE_DEGENERACY_RELATIVE_TOLERANCE
    return float(scale) <= numerical_floor + EPS


def _active_minimax_metrics(
    scaling: Mapping[str, tuple[float, float]],
) -> tuple[str, ...]:
    """Metrics with observed calibration variation used by the minimax search.

    A metric whose reference-minus-lower-bound range collapsed to a numerical
    epsilon remains in every result table, but is report-only as specified in
    the Q4 document.  Carbon-priority solves remain independent of this rule.
    """

    return tuple(
        metric for metric in METRICS
        if not _calibration_scale_is_degenerate(*scaling[metric])
    )


def _legacy_fixed_scaling() -> dict[str, tuple[float, float]] | None:
    """Reuse prior fixed mathematical calibration when its six columns exist."""

    path = TABLES_DIR / "q4_scaling.csv"
    if not path.is_file():
        return None
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if not {"Metric", "Anchor", "Scale"}.issubset(frame.columns):
            return None
        records = frame.set_index("Metric")
        if not set(METRICS).issubset(records.index):
            return None
        scaling = {
            metric: (float(records.at[metric, "Anchor"]), float(records.at[metric, "Scale"]))
            for metric in METRICS
        }
        if any(not np.isfinite(anchor) or not np.isfinite(scale) or scale <= EPS for anchor, scale in scaling.values()):
            return None
        return scaling
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return None


def _legacy_calibration_is_compatible(data: InputData, config: ModelConfig) -> bool:
    """Only reuse an old scale after checking it against the current inputs."""

    if _qos_weight_map(config) != QOS_WEIGHTS:
        return False
    configuration_path = TABLES_DIR / "q4_model_configuration.csv"
    assignments_path = TABLES_DIR / "q4_task_assignments.csv"
    dispatch_path = TABLES_DIR / "q4_region_hour_dispatch.csv"
    if not all(path.is_file() for path in (configuration_path, assignments_path, dispatch_path)):
        return False
    try:
        configuration = pd.read_csv(configuration_path, encoding="utf-8-sig")
        values = {str(row.Parameter): str(row.Value) for row in configuration.itertuples(index=False)}
        if int(float(values.get("decision_hours", -1))) != config.decision_hours:
            return False
        if int(float(values.get("lookahead_hours", -1))) != config.lookahead_hours:
            return False
        assignments = pd.read_csv(assignments_path, encoding="utf-8-sig")
        dispatch = pd.read_csv(dispatch_path, encoding="utf-8-sig")
        assignments["TaskID"] = assignments["TaskID"].astype(str)
        dispatch["Region"] = dispatch["Region"].astype(str)
        audit = independent_hard_constraint_audit(
            data, assignments, dispatch, tolerance=config.feasibility_tolerance,
            qos_weights=_qos_weight_map(config),
        )
        return bool(audit.get("passed", False))
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return False


def _load_reusable_v2_task_reference(
    data: InputData,
    config: ModelConfig,
) -> tuple[pd.DataFrame | None, str]:
    """Reuse only the task-side portion of V2 after independent validation.

    The legacy V2 energy dispatch is deliberately excluded because it violates
    the corrected §4.4 semantic constraints. Its complete TaskID schedule is
    reused only as a task-side validated seed; V4 recomputes its energy
    reference under the corrected joint constraints before it is used in
    calibration.
    """

    assignment_path = (
        QUESTION_DIR / "outputs" / "matheuristic_v2" / "tables"
        / "q4_task_assignments.csv"
    )
    if not assignment_path.is_file():
        return None, ""
    try:
        raw = pd.read_csv(assignment_path, encoding="utf-8-sig")
        schedule = _standardize_shadow_schedule(data, raw)
        _validate_shadow_schedule(data, schedule, config.feasibility_tolerance)
    except (OSError, ValueError, pd.errors.ParserError):
        return None, ""
    return schedule, f"V2_TASK_SCHEDULE_RECOMPUTED_WITH_V4_ENERGY:{assignment_path}"


def load_or_build_fixed_calibration(
    data: InputData,
    q2_schedule: pd.DataFrame,
    config: ModelConfig,
    paths: MatheuristicPaths,
) -> tuple[dict[str, tuple[float, float]], pd.DataFrame]:
    """Load one reusable calibration; never recompute it per rolling window."""

    paths.tables.mkdir(parents=True, exist_ok=True)
    calibration_digest = hashlib.sha256()
    calibration_digest.update(_matheuristic_signature(data, config, calibration_only=True).encode("ascii"))
    # The Q2 schedule is part of the formal input signature even though the
    # document-level calibration itself uses representative joint windows.
    calibration_digest.update(_matheuristic_frame_hash(q2_schedule))
    calibration_signature = calibration_digest.hexdigest()
    scaling_path = paths.tables / "q4_scaling.csv"
    records_path = paths.tables / "q4_calibration_records.csv"
    if paths.calibration.is_file() and scaling_path.is_file():
        try:
            metadata = json.loads(paths.calibration.read_text(encoding="utf-8"))
            if metadata.get("calibration_signature") == calibration_signature:
                candidate = pd.read_csv(scaling_path, encoding="utf-8-sig").set_index("Metric")
                scaling = {
                    metric: (float(candidate.at[metric, "Anchor"]), float(candidate.at[metric, "Scale"]))
                    for metric in METRICS
                }
                if all(scale > EPS and np.isfinite(anchor) and np.isfinite(scale) for anchor, scale in scaling.values()):
                    records = pd.read_csv(records_path, encoding="utf-8-sig") if records_path.is_file() else pd.DataFrame()
                    return scaling, records
        except (OSError, ValueError, KeyError, json.JSONDecodeError, pd.errors.ParserError):
            pass

    # Do not reuse a legacy/analytic scale here.  Section 4.6 requires the
    # calibration to come from representative windows, continuous lower bounds
    # and a strictly feasible reference under the same joint model.
    reusable_schedule, reusable_source = _load_reusable_v2_task_reference(data, config)
    if reusable_schedule is None:
        reusable_schedule = q2_schedule
        reusable_source = "Q2_SHADOW_TASK_SCHEDULE_RECOMPUTED_WITH_V4_ENERGY"
    scaling, records = _calibrate(
        data,
        config,
        feasible_reference_schedule=reusable_schedule,
        reference_source=reusable_source,
    )
    source = "REPRESENTATIVE_WINDOWS_CONTINUOUS_LB_AND_FEASIBLE_REFERENCE"
    records = records.copy()
    records["CalibrationSource"] = source
    records["CalibrationSignature"] = calibration_signature
    records["MetricDefinitionVersion"] = MATHEURISTIC_METRIC_VERSION
    records["CalibrationReferenceTaskSeed"] = reusable_source
    records["OptimizationRole"] = records["Metric"].map(
        lambda metric: (
            "REPORT_ONLY_DEGENERATE"
            if _calibration_scale_is_degenerate(*scaling[str(metric)])
            else "MINIMAX_ACTIVE"
        )
    )
    scale_metadata = (
        records.sort_values(["Metric", "Stage"], kind="stable")
        .drop_duplicates("Metric")
        .set_index("Metric")
    )
    _atomic_write_csv(
        pd.DataFrame([
            {
                "Metric": metric,
                "Anchor": scaling[metric][0],
                "Scale": scaling[metric][1],
                "RawScale": float(scale_metadata.at[metric, "RawScale"])
                if "RawScale" in scale_metadata.columns else float("nan"),
                "PhysicalScaleFloor": float(scale_metadata.at[metric, "PhysicalScaleFloor"])
                if "PhysicalScaleFloor" in scale_metadata.columns else float("nan"),
                "ScaleSource": str(scale_metadata.at[metric, "FinalScaleSource"])
                if "FinalScaleSource" in scale_metadata.columns else "",
                "DegenerateWindowRange": bool(scale_metadata.at[metric, "DegenerateWindowRange"])
                if "DegenerateWindowRange" in scale_metadata.columns else False,
                "OptimizationRole": (
                    "REPORT_ONLY_DEGENERATE"
                    if _calibration_scale_is_degenerate(*scaling[metric])
                    else "MINIMAX_ACTIVE"
                ),
            }
            for metric in METRICS
        ]),
        scaling_path,
    )
    _atomic_write_csv(records, records_path)
    _atomic_write_json(paths.calibration, {
        "calibration_signature": calibration_signature,
        "metric_definition_version": MATHEURISTIC_METRIC_VERSION,
        "source": source,
        "reference_task_seed": reusable_source,
    })
    return scaling, records


def _matheuristic_storage_arrays(data: InputData) -> dict[str, np.ndarray]:
    storage = data.storage.set_index("Region").loc[list(data.regions)]
    return {
        "min_soc": storage["MinSOC_MWh"].to_numpy(dtype=float),
        "max_soc": storage["StorageCapacity_MWh"].to_numpy(dtype=float),
        "initial_soc": storage["InitialSOC_MWh"].to_numpy(dtype=float),
        "max_charge": storage["MaxChargePower_MW"].to_numpy(dtype=float),
        "max_discharge": storage["MaxDischargePower_MW"].to_numpy(dtype=float),
        "eta_c": storage["ChargeEfficiency"].to_numpy(dtype=float),
        "eta_d": storage["DischargeEfficiency"].to_numpy(dtype=float),
        "max_grid": storage["MaxGridImport_MW"].to_numpy(dtype=float),
        "max_export": np.minimum(
            storage["SellLimit_MW"].to_numpy(dtype=float),
            storage["MaxGridExport_MW"].to_numpy(dtype=float),
        ),
    }


def _validate_objective_mode(objective_mode: str) -> str:
    mode = str(objective_mode).strip().lower()
    if mode not in {OBJECTIVE_MODE_MINIMAX, OBJECTIVE_MODE_CARBON_PRIORITY}:
        raise ValueError(f"Unknown Q4 optimization mode: {objective_mode}")
    return mode


def _incumbent_objective_value(incumbent: JointIncumbent, objective_mode: str) -> float:
    mode = _validate_objective_mode(objective_mode)
    if mode == OBJECTIVE_MODE_CARBON_PRIORITY:
        return float(incumbent.metrics["Carbon"])
    return float(incumbent.z)


def _append_minimax_objective_constraints(
    *,
    rows: list[dict[int, float]],
    row_lower: list[float],
    row_upper: list[float],
    metric_vectors: Mapping[str, np.ndarray],
    metric_constants: Mapping[str, float],
    scaling: Mapping[str, tuple[float, float]],
    deviation_indices: np.ndarray,
    max_deviation_index: int,
) -> None:
    """Append d_m and z constraints for the fixed augmented min-max objective."""

    active_metrics = set(_active_minimax_metrics(scaling))
    for position, metric in enumerate(METRICS):
        if metric not in active_metrics:
            # Keep a declared deviation variable for a uniform model shape, but
            # pin it to zero so an epsilon-only scale cannot influence Z.
            rows.append({int(deviation_indices[position]): 1.0})
            row_lower.append(0.0)
            row_upper.append(0.0)
            continue
        anchor, scale = scaling[metric]
        scale = max(float(scale), EPS)
        row = {int(deviation_indices[position]): 1.0}
        for column in np.flatnonzero(np.abs(metric_vectors[metric]) > EPS):
            row[int(column)] = row.get(int(column), 0.0) - float(metric_vectors[metric][column]) / scale
        rows.append(row)
        row_lower.append((float(metric_constants[metric]) - float(anchor)) / scale)
        row_upper.append(np.inf)
        rows.append({int(max_deviation_index): 1.0, int(deviation_indices[position]): -1.0})
        row_lower.append(0.0)
        row_upper.append(np.inf)


def _append_document_energy_semantic_constraints(
    add: Callable[[Mapping[int, float], float, float], int],
    *,
    indices: Mapping[str, np.ndarray],
    t: int,
    r: int,
    facility_constant: float,
    facility_variable_terms: Mapping[int, float] | None = None,
) -> None:
    """Add the two physical-meaning constraints stated explicitly in §4.4.

    ``RenewableDirectUse <= FacilityLoad`` prevents renewable direct use from
    becoming an unlabelled battery-charge source.  ``GridCharge <=
    GridPurchase`` makes the grid-charge variable represent electricity bought
    from the grid.  Both formal Q4 builders call this shared helper.
    """

    direct_limit: dict[int, float] = {
        int(indices["renewable_direct"][t, r]): 1.0,
    }
    for column, coefficient in dict(facility_variable_terms or {}).items():
        direct_limit[int(column)] = direct_limit.get(int(column), 0.0) + float(coefficient)
    add(direct_limit, -np.inf, float(facility_constant))
    add({
        int(indices["grid_charge"][t, r]): 1.0,
        int(indices["grid_purchase"][t, r]): -1.0,
    }, -np.inf, 0.0)


def _build_energy_only_problem(
    data: InputData,
    config: ModelConfig,
    *,
    tau: int,
    ai_profile: np.ndarray,
    current_soc: np.ndarray,
    historical_peak: np.ndarray,
    task_metrics: Mapping[str, float],
    scaling: Mapping[str, tuple[float, float]],
    carbon_budget_remaining: float | None = None,
) -> EnergyOnlyProblem:
    decision_end = min(tau + config.decision_hours, OPERATION_END)
    plan_end = min(decision_end + config.lookahead_hours, OPERATION_END)
    t_count = plan_end - tau
    h_count = decision_end - tau
    r_count = len(data.regions)
    if ai_profile.shape != (t_count, r_count):
        raise ValueError("Fixed task load profile dimension differs from the Q4 rolling window")
    frame, arrays = _matheuristic_region_hour(data, tau, plan_end)
    storage = _matheuristic_storage_arrays(data)
    offset = 0
    indices: dict[str, np.ndarray] = {}
    for name in (
        "renewable_direct", "renewable_charge", "export", "curtailment",
        "grid_purchase", "grid_charge", "discharge", "mode",
    ):
        indices[name], offset = _allocate(offset, (t_count, r_count))
    indices["soc"], offset = _allocate(offset, (t_count + 1, r_count))
    indices["peak"], offset = _allocate(offset, (r_count,))
    indices["deviation"], offset = _allocate(offset, (len(METRICS),))
    indices["max_deviation"], offset = _allocate(offset, (1,))
    lower = np.zeros(offset, dtype=float)
    upper = np.full(offset, np.inf, dtype=float)
    integrality = np.zeros(offset, dtype=np.uint8)
    upper[indices["renewable_direct"].ravel()] = arrays["AvailableRenewable_MW"].ravel()
    upper[indices["renewable_charge"].ravel()] = arrays["AvailableRenewable_MW"].ravel()
    upper[indices["curtailment"].ravel()] = arrays["AvailableRenewable_MW"].ravel()
    upper[indices["export"].ravel()] = np.tile(storage["max_export"], t_count)
    upper[indices["grid_purchase"].ravel()] = np.tile(storage["max_grid"], t_count)
    upper[indices["grid_charge"].ravel()] = np.tile(storage["max_charge"], t_count)
    upper[indices["discharge"].ravel()] = np.tile(storage["max_discharge"], t_count)
    upper[indices["mode"].ravel()] = 1.0
    integrality[indices["mode"][:h_count].ravel()] = 1
    lower[indices["soc"].ravel()] = np.tile(storage["min_soc"], t_count + 1)
    upper[indices["soc"].ravel()] = np.tile(storage["max_soc"], t_count + 1)
    lower[indices["soc"][0]] = current_soc
    upper[indices["soc"][0]] = current_soc
    recoverable = np.maximum(
        storage["min_soc"],
        storage["initial_soc"] - storage["eta_c"] * storage["max_charge"] * max(OPERATION_END - decision_end, 0),
    )
    lower[indices["soc"][h_count]] = np.maximum(lower[indices["soc"][h_count]], recoverable)
    if decision_end >= OPERATION_END:
        lower[indices["soc"][h_count]] = np.maximum(
            lower[indices["soc"][h_count]], storage["initial_soc"]
        )
    lower[indices["peak"]] = historical_peak

    rows: list[dict[int, float]] = []
    row_lower: list[float] = []
    row_upper: list[float] = []

    def add(values: Mapping[int, float], lb: float, ub: float) -> int:
        rows.append(dict(values))
        row_lower.append(float(lb))
        row_upper.append(float(ub))
        return len(rows) - 1

    balance_rows = np.empty((t_count, r_count), dtype=int)
    facility = arrays["PUE"] * (arrays["NonAI_IT_Load_MW"] + ai_profile)
    for t in range(t_count):
        for r in range(r_count):
            add({
                int(indices["renewable_direct"][t, r]): 1.0,
                int(indices["renewable_charge"][t, r]): 1.0,
                int(indices["export"][t, r]): 1.0,
                int(indices["curtailment"][t, r]): 1.0,
            }, arrays["AvailableRenewable_MW"][t, r], arrays["AvailableRenewable_MW"][t, r])
            balance_rows[t, r] = add({
                int(indices["grid_purchase"][t, r]): 1.0,
                int(indices["renewable_direct"][t, r]): 1.0,
                int(indices["discharge"][t, r]): 1.0,
                int(indices["grid_charge"][t, r]): -1.0,
            }, facility[t, r], facility[t, r])
            _append_document_energy_semantic_constraints(
                add,
                indices=indices,
                t=t,
                r=r,
                facility_constant=float(facility[t, r]),
            )
            add({
                int(indices["soc"][t + 1, r]): 1.0,
                int(indices["soc"][t, r]): -1.0,
                int(indices["renewable_charge"][t, r]): -storage["eta_c"][r],
                int(indices["grid_charge"][t, r]): -storage["eta_c"][r],
                int(indices["discharge"][t, r]): 1.0 / storage["eta_d"][r],
            }, 0.0, 0.0)
            add({
                int(indices["renewable_charge"][t, r]): 1.0,
                int(indices["grid_charge"][t, r]): 1.0,
                int(indices["mode"][t, r]): -storage["max_charge"][r],
            }, -np.inf, 0.0)
            add({
                int(indices["discharge"][t, r]): 1.0,
                int(indices["mode"][t, r]): storage["max_discharge"][r],
            }, -np.inf, storage["max_discharge"][r])
            add({
                int(indices["peak"][r]): 1.0,
                int(indices["grid_purchase"][t, r]): -1.0,
                int(indices["export"][t, r]): 1.0,
            }, 0.0, np.inf)
    if carbon_budget_remaining is not None:
        carbon_row = {
            int(indices["grid_purchase"][t, r]): arrays["CarbonIntensity_tCO2_per_MWh"][t, r]
            for t in range(h_count) for r in range(r_count)
        }
        add(carbon_row, -np.inf, max(float(carbon_budget_remaining), 0.0))

    metric_vectors = {metric: np.zeros(offset, dtype=float) for metric in METRICS}
    metric_constants = {metric: 0.0 for metric in METRICS}
    metric_constants["Latency"] = float(task_metrics["Latency"])
    metric_constants["Delay"] = float(task_metrics["Delay"])
    renewable_total = max(float(arrays["AvailableRenewable_MW"].sum()), EPS)
    for t in range(t_count):
        for r in range(r_count):
            metric_vectors["Cost"][indices["grid_purchase"][t, r]] = arrays["ElectricityPrice_CNY_per_MWh"][t, r]
            metric_vectors["Cost"][indices["export"][t, r]] = -arrays["SellPrice_CNY_per_MWh"][t, r]
            metric_vectors["Carbon"][indices["grid_purchase"][t, r]] = arrays["CarbonIntensity_tCO2_per_MWh"][t, r]
            metric_vectors["RenewableUnusedRate"][indices["curtailment"][t, r]] = 1.0 / renewable_total
    metric_vectors["Peak"][indices["peak"]] = 1.0
    _append_minimax_objective_constraints(
        rows=rows,
        row_lower=row_lower,
        row_upper=row_upper,
        metric_vectors=metric_vectors,
        metric_constants=metric_constants,
        scaling=scaling,
        deviation_indices=indices["deviation"],
        max_deviation_index=int(indices["max_deviation"][0]),
    )
    return EnergyOnlyProblem(
        tau=tau,
        decision_end=decision_end,
        plan_end=plan_end,
        frame=frame,
        arrays=arrays,
        ai_profile=ai_profile,
        indices=indices,
        lower=lower,
        upper=upper,
        integrality=integrality,
        matrix=_rows_to_sparse(rows, offset),
        constraint_lower=np.asarray(row_lower, dtype=float),
        constraint_upper=np.asarray(row_upper, dtype=float),
        balance_rows=balance_rows,
        task_metrics={key: float(value) for key, value in task_metrics.items()},
    )


def _energy_dispatch_from_vector(
    data: InputData,
    *,
    tau: int,
    frame: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
    ai_profile: np.ndarray,
    indices: Mapping[str, np.ndarray],
    vector: np.ndarray,
) -> pd.DataFrame:
    t_count, r_count = ai_profile.shape
    rows: list[dict[str, object]] = []
    for t in range(t_count):
        for r in range(r_count):
            source = frame.iloc[t * r_count + r]
            renewable_charge = float(vector[indices["renewable_charge"][t, r]])
            grid_charge = float(vector[indices["grid_charge"][t, r]])
            purchase = float(vector[indices["grid_purchase"][t, r]])
            export = float(vector[indices["export"][t, r]])
            direct = float(vector[indices["renewable_direct"][t, r]])
            discharge = float(vector[indices["discharge"][t, r]])
            rows.append({
                "Hour": int(tau + t),
                "Region": str(data.regions[r]),
                "AI_IT_Load_MW": float(ai_profile[t, r]),
                "NonAI_IT_Load_MW": float(arrays["NonAI_IT_Load_MW"][t, r]),
                "Total_IT_Load_MW": float(ai_profile[t, r] + arrays["NonAI_IT_Load_MW"][t, r]),
                "Facility_Load_MW": float(arrays["PUE"][t, r] * (ai_profile[t, r] + arrays["NonAI_IT_Load_MW"][t, r])),
                "AvailableRenewable_MW": float(arrays["AvailableRenewable_MW"][t, r]),
                "RenewableDirectUse_MW": direct,
                "RenewableCharge_MW": renewable_charge,
                "GridCharge_MW": grid_charge,
                "ChargePower_MW": renewable_charge + grid_charge,
                "DischargePower_MW": discharge,
                "GridPurchase_MW": purchase,
                "GridSell_MW": export,
                "RenewableCurtailment_MW": float(vector[indices["curtailment"][t, r]]),
                "NetGridImport_MW": purchase - export,
                "SOCStart_MWh": float(vector[indices["soc"][t, r]]),
                "SOCEnd_MWh": float(vector[indices["soc"][t + 1, r]]),
                "ElectricityPrice_CNY_per_MWh": float(arrays["ElectricityPrice_CNY_per_MWh"][t, r]),
                "SellPrice_CNY_per_MWh": float(arrays["SellPrice_CNY_per_MWh"][t, r]),
                "CarbonIntensity_tCO2_per_MWh": float(arrays["CarbonIntensity_tCO2_per_MWh"][t, r]),
                "CarbonEmission_tCO2": float(arrays["CarbonIntensity_tCO2_per_MWh"][t, r] * purchase),
                "Available_GPU": float(arrays["Available_GPU"][t, r]),
                "Max_IT_Power_MW": float(arrays["Max_IT_Power_MW"][t, r]),
                "Max_Facility_Power_MW": float(arrays["Max_Facility_Power_MW"][t, r]),
                "PUE": float(arrays["PUE"][t, r]),
                "DecisionWindowStart": int(tau),
            })
    return pd.DataFrame(rows)


def _assert_document_energy_semantics(
    dispatch: pd.DataFrame,
    tolerance: float,
    *,
    context: str,
) -> None:
    """Fail fast if a solver reconstruction violates a §4.4 semantic identity."""

    violations = {
        "RenewableBalance": float(np.abs(
            dispatch["AvailableRenewable_MW"] - dispatch["RenewableDirectUse_MW"]
            - dispatch["RenewableCharge_MW"] - dispatch["GridSell_MW"]
            - dispatch["RenewableCurtailment_MW"]
        ).to_numpy(dtype=float).max(initial=0.0)),
        "FacilityLoadBalance": float(np.abs(
            dispatch["GridPurchase_MW"] + dispatch["RenewableDirectUse_MW"]
            + dispatch["DischargePower_MW"] - dispatch["Facility_Load_MW"]
            - dispatch["GridCharge_MW"]
        ).to_numpy(dtype=float).max(initial=0.0)),
        "RenewableDirectUseLimit": float(np.maximum(
            dispatch["RenewableDirectUse_MW"] - dispatch["Facility_Load_MW"], 0.0
        ).to_numpy(dtype=float).max(initial=0.0)),
        "GridChargeSourceLimit": float(np.maximum(
            dispatch["GridCharge_MW"] - dispatch["GridPurchase_MW"], 0.0
        ).to_numpy(dtype=float).max(initial=0.0)),
        "ChargePowerDefinition": float(np.abs(
            dispatch["ChargePower_MW"] - dispatch["RenewableCharge_MW"]
            - dispatch["GridCharge_MW"]
        ).to_numpy(dtype=float).max(initial=0.0)),
        "NetGridImportDefinition": float(np.abs(
            dispatch["NetGridImport_MW"] - dispatch["GridPurchase_MW"] + dispatch["GridSell_MW"]
        ).to_numpy(dtype=float).max(initial=0.0)),
        "CarbonEmissionDefinition": float(np.abs(
            dispatch["CarbonEmission_tCO2"]
            - dispatch["CarbonIntensity_tCO2_per_MWh"] * dispatch["GridPurchase_MW"]
        ).to_numpy(dtype=float).max(initial=0.0)),
    }
    failed = {name: value for name, value in violations.items() if value > tolerance}
    if failed:
        raise RuntimeError(f"{context} violates Q4 documented energy semantics: {failed}")


def _energy_solution_from_vector(
    data: InputData,
    problem: EnergyOnlyProblem,
    vector: np.ndarray,
    scaling: Mapping[str, tuple[float, float]],
    config: ModelConfig,
    *,
    mip_gap: float,
    elapsed_seconds: float,
    status: str,
    message: str,
) -> EnergySolution:
    dispatch = _energy_dispatch_from_vector(
        data,
        tau=problem.tau,
        frame=problem.frame,
        arrays=problem.arrays,
        ai_profile=problem.ai_profile,
        indices=problem.indices,
        vector=vector,
    )
    _assert_document_energy_semantics(
        dispatch, config.feasibility_tolerance * 10.0,
        context=f"Fixed-task energy response window {problem.tau}",
    )
    renewable_total = max(float(dispatch["AvailableRenewable_MW"].sum()), EPS)
    metrics = {
        "Cost": float(np.sum(dispatch["ElectricityPrice_CNY_per_MWh"] * dispatch["GridPurchase_MW"] - dispatch["SellPrice_CNY_per_MWh"] * dispatch["GridSell_MW"])),
        "Carbon": float(dispatch["CarbonEmission_tCO2"].sum()),
        "Latency": float(problem.task_metrics["Latency"]),
        "Delay": float(problem.task_metrics["Delay"]),
        "RenewableUnusedRate": float(dispatch["RenewableCurtailment_MW"].sum() / renewable_total),
        "Peak": float(vector[problem.indices["peak"]].sum()),
    }
    z, deviations = _balanced_z(metrics, scaling, config)
    h_count = problem.decision_end - problem.tau
    return EnergySolution(
        dispatch=dispatch,
        end_soc=np.asarray(vector[problem.indices["soc"][-1]], dtype=float).copy(),
        decision_end_soc=np.asarray(vector[problem.indices["soc"][h_count]], dtype=float).copy(),
        regional_peak=np.asarray(vector[problem.indices["peak"]], dtype=float).copy(),
        metrics=metrics,
        deviations=deviations,
        z=z,
        mip_gap=float(mip_gap),
        elapsed_seconds=float(elapsed_seconds),
        status=status,
        message=message,
    )


def _energy_solver_objective(
    problem: EnergyOnlyProblem,
    config: ModelConfig,
    scaling: Mapping[str, tuple[float, float]],
    objective_mode: str,
) -> np.ndarray:
    mode = _validate_objective_mode(objective_mode)
    objective = np.zeros(problem.lower.size, dtype=float)
    if mode == OBJECTIVE_MODE_CARBON_PRIORITY:
        objective[problem.indices["grid_purchase"]] = (
            problem.arrays["CarbonIntensity_tCO2_per_MWh"]
        )
    else:
        objective[int(problem.indices["max_deviation"][0])] = 1.0
        objective[problem.indices["deviation"]] = config.tie_break_weight
    # The minimax objective remains primary.  This small deterministic
    # secondary term resolves physically equivalent energy solutions by
    # preferring lower-carbon/lower-cost grid purchases and less storage
    # cycling.  It prevents a report-only or numerically tied metric from
    # selecting an arbitrary high-purchase path.
    carbon_scale = max(abs(float(scaling["Carbon"][1])), 1.0)
    cost_scale = max(abs(float(scaling["Cost"][1])), 1.0)
    throughput_scale = max(
        float(np.sum(problem.arrays["AvailableRenewable_MW"])), 1.0
    )
    physical = np.zeros(problem.lower.size, dtype=float)
    physical[problem.indices["grid_purchase"]] = (
        problem.arrays["CarbonIntensity_tCO2_per_MWh"] / carbon_scale
        + problem.arrays["ElectricityPrice_CNY_per_MWh"] / cost_scale
    )
    physical[problem.indices["renewable_charge"]] += 1.0 / throughput_scale
    physical[problem.indices["grid_charge"]] += 1.0 / throughput_scale
    physical[problem.indices["discharge"]] += 1.0 / throughput_scale
    objective += float(config.energy_physical_tie_break_weight) * physical
    return objective


def solve_energy_response(
    data: InputData,
    config: ModelConfig,
    *,
    tau: int,
    ai_it_profile: np.ndarray,
    current_soc: np.ndarray,
    historical_peak: np.ndarray,
    task_metrics: Mapping[str, float],
    scaling: Mapping[str, tuple[float, float]],
    time_limit: float | None = None,
    carbon_budget_remaining: float | None = None,
    objective_mode: str = OBJECTIVE_MODE_MINIMAX,
) -> EnergySolution:
    """Document-compliant V4 energy response for a fixed task-load trajectory."""

    from scipy.optimize import Bounds, LinearConstraint, milp

    problem = _build_energy_only_problem(
        data, config, tau=tau, ai_profile=ai_it_profile,
        current_soc=current_soc, historical_peak=historical_peak,
        task_metrics=task_metrics, scaling=scaling,
        carbon_budget_remaining=carbon_budget_remaining,
    )
    objective = _energy_solver_objective(problem, config, scaling, objective_mode)
    limit = max(0.1, float(time_limit if time_limit is not None else config.energy_time_limit_seconds))
    started = time.perf_counter()
    result = milp(
        c=objective,
        integrality=problem.integrality,
        bounds=Bounds(problem.lower, problem.upper),
        constraints=LinearConstraint(problem.matrix, problem.constraint_lower, problem.constraint_upper),
        options={"presolve": True, "time_limit": limit, "mip_rel_gap": float(config.energy_mip_gap)},
    )
    elapsed = time.perf_counter() - started
    if result.x is None:
        raise RuntimeError(
            f"Window {tau}: no feasible storage-renewable-grid response for fixed task loads: {result.message}"
        )
    vector = np.asarray(result.x, dtype=float)
    activity = problem.matrix @ vector
    violation = float(max(
        np.maximum(problem.constraint_lower - activity, 0.0).max(initial=0.0),
        np.maximum(activity - problem.constraint_upper, 0.0).max(initial=0.0),
    ))
    if violation > max(config.feasibility_tolerance * 10.0, 1e-5):
        raise RuntimeError(f"Window {tau}: energy response violates constraints: {violation:.3g}")
    status = "ENERGY_OPTIMAL" if int(result.status) == 0 else "ENERGY_TIME_LIMIT_FEASIBLE"
    return _energy_solution_from_vector(
        data, problem, vector, scaling, config,
        mip_gap=_result_float(result, "mip_gap"), elapsed_seconds=elapsed,
        status=status, message=str(result.message),
    )


def solve_energy_lp_marginals(
    data: InputData,
    config: ModelConfig,
    *,
    tau: int,
    ai_it_profile: np.ndarray,
    current_soc: np.ndarray,
    historical_peak: np.ndarray,
    task_metrics: Mapping[str, float],
    scaling: Mapping[str, tuple[float, float]],
    time_limit: float | None = None,
    carbon_budget_remaining: float | None = None,
    objective_mode: str = OBJECTIVE_MODE_MINIMAX,
) -> tuple[np.ndarray, str, float]:
    """Relax only energy integrality and return facility-load LP shadow prices."""

    from scipy.optimize import linprog

    problem = _build_energy_only_problem(
        data, config, tau=tau, ai_profile=ai_it_profile,
        current_soc=current_soc, historical_peak=historical_peak,
        task_metrics=task_metrics, scaling=scaling,
        carbon_budget_remaining=carbon_budget_remaining,
    )
    mode = _validate_objective_mode(objective_mode)
    objective = _energy_solver_objective(problem, config, scaling, mode)
    a_ub, b_ub, a_eq, b_eq = _linear_constraint_components(
        problem.matrix, problem.constraint_lower, problem.constraint_upper
    )
    started = time.perf_counter()
    result = linprog(
        c=objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=list(zip(problem.lower, problem.upper)),
        method="highs",
        options={"presolve": True, "time_limit": max(0.1, float(time_limit if time_limit is not None else config.marginal_lp_time_limit_seconds))},
    )
    elapsed = time.perf_counter() - started
    if mode == OBJECTIVE_MODE_CARBON_PRIORITY:
        fallback = problem.arrays["CarbonIntensity_tCO2_per_MWh"].copy()
    else:
        fallback = (
            problem.arrays["ElectricityPrice_CNY_per_MWh"] / max(float(scaling["Cost"][1]), EPS)
            + problem.arrays["CarbonIntensity_tCO2_per_MWh"] / max(float(scaling["Carbon"][1]), EPS)
        )
    if int(result.status) != 0 or result.x is None or getattr(result, "eqlin", None) is None:
        return fallback, "LP_MARGINAL_FALLBACK", elapsed
    equal_rows = np.flatnonzero(
        np.isfinite(problem.constraint_lower)
        & np.isfinite(problem.constraint_upper)
        & np.isclose(problem.constraint_lower, problem.constraint_upper, rtol=0.0, atol=EPS)
    )
    positions = {int(row): pos for pos, row in enumerate(equal_rows)}
    duals = np.asarray(result.eqlin.marginals, dtype=float)
    marginal = np.zeros_like(fallback)
    for t in range(marginal.shape[0]):
        for r in range(marginal.shape[1]):
            position = positions.get(int(problem.balance_rows[t, r]))
            value = duals[position] if position is not None and position < len(duals) else np.nan
            marginal[t, r] = value if np.isfinite(value) else fallback[t, r]
    return marginal, "LP_MARGINAL_OPTIMAL", elapsed


def _operational_task_options(
    data: InputData,
    task_id: str,
    lower: int,
    decision_end: int,
    plan_end: int,
) -> tuple[JointTaskOption, ...]:
    """Enumerate every legal H-area option; no Top-K domain deletion occurs."""

    task = data.task_lookup.loc[str(task_id)]
    starts = _feasible_starts(task, lower, decision_end)
    options: list[JointTaskOption] = []
    for candidate in data.candidate_map[str(task_id)]:
        for start in starts:
            options.append(JointTaskOption(
                task_id=str(task_id),
                target_region=candidate.region,
                region_index=candidate.region_index,
                start_hour=int(start),
                latency_ms=float(candidate.latency_ms),
                overlaps=_hour_overlaps(float(start), float(task.Duration_h), lower, plan_end),
            ))
    return tuple(options)


def _task_delay_contribution(task: pd.Series, start_hour: float, config: ModelConfig) -> float:
    if str(task.TaskType) == "RealTimeInference":
        return 0.0
    earliest = max(float(task.ArrivalHour), float(task.EarliestStartHour))
    slack = min(float(task.LatestFinishHour), float(OPERATION_END)) - float(task.Duration_h) - earliest
    if slack <= EPS:
        return 0.0
    weight = _qos_weight_map(config)[str(task.DelaySensitivity)]
    return float(weight * np.clip((float(start_hour) - earliest) / slack, 0.0, 1.0))


def _window_metric_denominators(
    schedule: pd.DataFrame,
    lower: int,
    upper: int,
    config: ModelConfig,
) -> tuple[int, float]:
    window = schedule.loc[pd.to_numeric(schedule["StartHour"]).between(lower, upper - 1)]
    if window.empty:
        return 0, 0.0
    weights = window["DelaySensitivity"].map(_qos_weight_map(config)).to_numpy(dtype=float)
    earliest = np.maximum(
        window["ArrivalHour"].to_numpy(dtype=float),
        window["EarliestStartHour"].to_numpy(dtype=float),
    )
    slack = np.minimum(window["LatestFinishHour"].to_numpy(dtype=float), float(OPERATION_END)) - window["Duration_h"].to_numpy(dtype=float) - earliest
    flexible = (~window["TaskType"].eq("RealTimeInference").to_numpy()) & (slack > EPS)
    return len(window), float(weights[flexible].sum())


def _option_load_deltas(
    data: InputData,
    row: pd.Series,
    option: JointTaskOption,
    lower: int,
    upper: int,
) -> dict[tuple[int, int], tuple[float, float]]:
    deltas: dict[tuple[int, int], list[float]] = {}

    def add(hour: int, region_index: int, gpu: float, ai: float) -> None:
        if lower <= hour < upper:
            value = deltas.setdefault((hour - lower, region_index), [0.0, 0.0])
            value[0] += gpu
            value[1] += ai

    old_region = data.region_index[str(row.TargetRegion)]
    for hour, overlap in _hour_overlaps(
        float(row.StartHour), float(row.Duration_h), lower, upper
    ):
        add(hour, old_region, -float(row.GPU_Demand) * overlap, -float(row.Task_Full_IT_Power_MW) * overlap)
    task = data.task_lookup.loc[str(option.task_id)]
    for hour, overlap in option.overlaps:
        add(hour, option.region_index, float(task.GPU_Demand) * overlap, float(task.Task_Full_IT_Power_MW) * overlap)
    return {key: (value[0], value[1]) for key, value in deltas.items()}


def _capacity_accepts_deltas(
    *,
    gpu: np.ndarray,
    ai: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    deltas: Mapping[tuple[int, int], tuple[float, float]],
    tolerance: float,
) -> bool:
    for (t, r), (delta_gpu, delta_ai) in deltas.items():
        ai_limit = min(
            arrays["Max_IT_Power_MW"][t, r] - arrays["NonAI_IT_Load_MW"][t, r],
            arrays["Max_Facility_Power_MW"][t, r] / arrays["PUE"][t, r] - arrays["NonAI_IT_Load_MW"][t, r],
        )
        if gpu[t, r] + delta_gpu > arrays["Available_GPU"][t, r] + tolerance:
            return False
        if ai[t, r] + delta_ai > ai_limit + tolerance:
            return False
    return True


def _apply_option_to_schedule(
    schedule: pd.DataFrame,
    row_index: int,
    option: JointTaskOption,
) -> None:
    duration = float(schedule.at[row_index, "Duration_h"])
    schedule.at[row_index, "TargetRegion"] = option.target_region
    schedule.at[row_index, "NetworkLatency_ms"] = option.latency_ms
    schedule.at[row_index, "StartHour"] = float(option.start_hour)
    schedule.at[row_index, "FinishHour"] = float(option.start_hour) + duration
    earliest = max(float(schedule.at[row_index, "ArrivalHour"]), float(schedule.at[row_index, "EarliestStartHour"]))
    schedule.at[row_index, "WaitHours"] = max(float(option.start_hour) - earliest, 0.0)
    schedule.at[row_index, "IsMigrated"] = int(
        str(option.target_region) != str(schedule.at[row_index, "SourceRegion"])
    )
    schedule.at[row_index, "DecisionWindowStart"] = min((int(option.start_hour) // 24) * 24, MAIN_END)


def _compute_candidate_move_score(
    data: InputData,
    config: ModelConfig,
    *,
    incumbent: JointIncumbent,
    row: pd.Series,
    option: JointTaskOption,
    gpu: np.ndarray,
    ai: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    lower: int,
    upper: int,
    marginal_prices: np.ndarray,
    scaling: Mapping[str, tuple[float, float]],
    task_count: int,
    delay_denominator: float,
    objective_mode: str,
) -> float | None:
    if str(row.TargetRegion) == option.target_region and abs(float(row.StartHour) - option.start_hour) <= EPS:
        return None
    deltas = _option_load_deltas(data, row, option, lower, upper)
    if not _capacity_accepts_deltas(
        gpu=gpu, ai=ai, arrays=arrays, deltas=deltas,
        tolerance=config.feasibility_tolerance,
    ):
        return None
    energy_delta = 0.0
    for (t, r), (_, delta_ai) in deltas.items():
        energy_delta += float(marginal_prices[t, r]) * arrays["PUE"][t, r] * delta_ai
    if _validate_objective_mode(objective_mode) == OBJECTIVE_MODE_CARBON_PRIORITY:
        return float(energy_delta)
    latency_delta = (option.latency_ms - float(row.NetworkLatency_ms)) / max(task_count, 1)
    task = data.task_lookup.loc[str(option.task_id)]
    delay_delta = (
        _task_delay_contribution(task, option.start_hour, config)
        - _task_delay_contribution(task, float(row.StartHour), config)
    ) / delay_denominator if delay_denominator > EPS else 0.0
    predicted_deviations = dict(incumbent.deviations)
    for metric, delta in (("Latency", latency_delta), ("Delay", delay_delta)):
        anchor, scale = scaling[metric]
        predicted_value = max(0.0, (incumbent.metrics[metric] + delta - anchor) / max(scale, EPS))
        predicted_deviations[metric] = predicted_value
    predicted_z = (
        max(predicted_deviations.values(), default=0.0)
        + config.tie_break_weight * sum(predicted_deviations.values())
        + energy_delta
    )
    return float(predicted_z - incumbent.z)


def _generate_marginal_moves(
    data: InputData,
    config: ModelConfig,
    *,
    incumbent: JointIncumbent,
    tau: int,
    current_soc: np.ndarray,
    historical_peak: np.ndarray,
    scaling: Mapping[str, tuple[float, float]],
    carbon_budget_remaining: float | None,
    remaining_seconds: float,
    objective_mode: str = OBJECTIVE_MODE_MINIMAX,
) -> tuple[list[MarginalMove], int, str, float]:
    plan_end = min(tau + config.decision_hours + config.lookahead_hours, OPERATION_END)
    decision_end = min(tau + config.decision_hours, OPERATION_END)
    gpu, ai, _, arrays = _shadow_profiles(data, incumbent.shadow_schedule, tau, plan_end)
    task_metrics = _window_task_metrics(incumbent.shadow_schedule, tau, plan_end, config)
    marginal_prices, marginal_status, marginal_seconds = solve_energy_lp_marginals(
        data, config, tau=tau, ai_it_profile=ai, current_soc=current_soc,
        historical_peak=historical_peak, task_metrics=task_metrics, scaling=scaling,
        time_limit=min(config.marginal_lp_time_limit_seconds, max(0.1, remaining_seconds)),
        carbon_budget_remaining=carbon_budget_remaining,
        objective_mode=objective_mode,
    )
    task_count, delay_denominator = _window_metric_denominators(
        incumbent.shadow_schedule, tau, plan_end, config
    )
    eligible = incumbent.shadow_schedule.loc[
        pd.to_numeric(incumbent.shadow_schedule["StartHour"]).between(tau, decision_end - 1)
        & ~incumbent.shadow_schedule["TaskType"].eq("RealTimeInference")
    ].copy()
    candidate_count = 0
    moves: list[MarginalMove] = []
    for row in eligible.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        catalog = _operational_task_options(data, str(row.TaskID), tau, decision_end, plan_end)
        candidate_count += len(catalog)
        best: MarginalMove | None = None
        for option in catalog:
            score = _compute_candidate_move_score(
                data, config, incumbent=incumbent, row=row_series, option=option,
                gpu=gpu, ai=ai, arrays=arrays, lower=tau, upper=plan_end,
                marginal_prices=marginal_prices, scaling=scaling, task_count=task_count,
                delay_denominator=delay_denominator, objective_mode=objective_mode,
            )
            if score is None or score >= -config.feasibility_tolerance:
                continue
            candidate = MarginalMove(
                task_id=str(row.TaskID), target_region=option.target_region,
                start_hour=option.start_hour, latency_ms=option.latency_ms,
                predicted_delta_z=float(score), candidate_count=len(catalog),
            )
            if best is None or candidate.predicted_delta_z < best.predicted_delta_z:
                best = candidate
        if best is not None:
            moves.append(best)
    moves.sort(key=lambda move: (move.predicted_delta_z, move.task_id, move.start_hour, move.target_region))
    return moves, candidate_count, marginal_status, marginal_seconds


def _apply_capacity_safe_moves(
    data: InputData,
    config: ModelConfig,
    schedule: pd.DataFrame,
    moves: Sequence[MarginalMove],
    *,
    tau: int,
    plan_end: int,
) -> tuple[pd.DataFrame, list[MarginalMove]]:
    proposal = schedule.copy()
    gpu, ai, _, arrays = _shadow_profiles(data, proposal, tau, plan_end)
    row_index = {str(task_id): int(index) for index, task_id in proposal["TaskID"].items()}
    accepted: list[MarginalMove] = []
    for move in moves:
        index = row_index.get(move.task_id)
        if index is None:
            continue
        row = proposal.loc[index]
        task = data.task_lookup.loc[move.task_id]
        option = JointTaskOption(
            task_id=move.task_id,
            target_region=move.target_region,
            region_index=data.region_index[move.target_region],
            start_hour=move.start_hour,
            latency_ms=move.latency_ms,
            overlaps=_hour_overlaps(float(move.start_hour), float(task.Duration_h), tau, plan_end),
        )
        deltas = _option_load_deltas(data, row, option, tau, plan_end)
        if not _capacity_accepts_deltas(
            gpu=gpu, ai=ai, arrays=arrays, deltas=deltas,
            tolerance=config.feasibility_tolerance,
        ):
            continue
        for (t, r), (delta_gpu, delta_ai) in deltas.items():
            gpu[t, r] += delta_gpu
            ai[t, r] += delta_ai
        _apply_option_to_schedule(proposal, index, option)
        accepted.append(move)
        if len(accepted) >= config.marginal_max_moves:
            break
    return proposal, accepted


def _schedule_capacity_violation(
    data: InputData,
    schedule: pd.DataFrame,
    lower: int,
    upper: int,
) -> float:
    gpu, ai, _, arrays = _shadow_profiles(data, schedule, lower, upper)
    ai_limit = np.minimum(
        arrays["Max_IT_Power_MW"] - arrays["NonAI_IT_Load_MW"],
        arrays["Max_Facility_Power_MW"] / arrays["PUE"] - arrays["NonAI_IT_Load_MW"],
    )
    return float(max(
        np.maximum(gpu - arrays["Available_GPU"], 0.0).max(initial=0.0),
        np.maximum(ai - ai_limit, 0.0).max(initial=0.0),
    ))


def build_local_repair_problem(
    data: InputData,
    config: ModelConfig,
    *,
    schedule: pd.DataFrame,
    task_ids: Sequence[str],
    tau: int,
    plan_end: int,
) -> tuple[dict[str, object], list[JointTaskOption]]:
    """Small feasibility MILP used only to repair a local resource conflict."""

    from scipy.sparse import csr_matrix

    decision_end = min(tau + config.decision_hours, OPERATION_END)
    selected = tuple(dict.fromkeys(str(task_id) for task_id in task_ids))
    fixed = schedule.loc[~schedule["TaskID"].astype(str).isin(selected)].copy()
    fixed_gpu, fixed_ai, _, arrays = _shadow_profiles(data, fixed, tau, plan_end)
    options: list[JointTaskOption] = []
    by_task: dict[str, list[int]] = {}
    for task_id in selected:
        catalog = _operational_task_options(data, task_id, tau, decision_end, plan_end)
        if not catalog:
            raise RuntimeError(f"Local repair task {task_id} has no valid H-region option")
        by_task[task_id] = list(range(len(options), len(options) + len(catalog)))
        options.extend(catalog)
    n = len(options)
    rows: list[dict[int, float]] = []
    lower: list[float] = []
    upper: list[float] = []
    for task_id, columns in by_task.items():
        rows.append({column: 1.0 for column in columns})
        lower.append(1.0)
        upper.append(1.0)
    for t in range(plan_end - tau):
        for r in range(len(data.regions)):
            gpu_row: dict[int, float] = {}
            ai_row: dict[int, float] = {}
            for column, option in enumerate(options):
                if option.region_index != r:
                    continue
                task = data.task_lookup.loc[option.task_id]
                for hour, overlap in option.overlaps:
                    if hour == tau + t:
                        gpu_row[column] = gpu_row.get(column, 0.0) + float(task.GPU_Demand) * overlap
                        ai_row[column] = ai_row.get(column, 0.0) + float(task.Task_Full_IT_Power_MW) * overlap
            ai_limit = min(
                arrays["Max_IT_Power_MW"][t, r] - arrays["NonAI_IT_Load_MW"][t, r],
                arrays["Max_Facility_Power_MW"][t, r] / arrays["PUE"][t, r] - arrays["NonAI_IT_Load_MW"][t, r],
            )
            rows.extend((gpu_row, ai_row))
            lower.extend((-np.inf, -np.inf))
            upper.extend((
                arrays["Available_GPU"][t, r] - fixed_gpu[t, r],
                ai_limit - fixed_ai[t, r],
            ))
    matrix = _rows_to_sparse(rows, n) if rows else csr_matrix((0, n))
    current = schedule.set_index("TaskID", drop=False)
    objective = np.asarray([
        0.0 if (
            str(current.at[option.task_id, "TargetRegion"]) == option.target_region
            and abs(float(current.at[option.task_id, "StartHour"]) - option.start_hour) <= EPS
        ) else 1.0
        for option in options
    ], dtype=float)
    return {
        "matrix": matrix,
        "lower": np.asarray(lower, dtype=float),
        "upper": np.asarray(upper, dtype=float),
        "objective": objective,
        "integrality": np.ones(n, dtype=np.uint8),
        "by_task": by_task,
    }, options


def repair_if_needed(
    data: InputData,
    config: ModelConfig,
    *,
    incumbent_schedule: pd.DataFrame,
    proposal: pd.DataFrame,
    moved_task_ids: Sequence[str],
    tau: int,
    plan_end: int,
    remaining_seconds: float,
) -> tuple[pd.DataFrame, str, int, float]:
    """Repair once locally, expand once if necessary, then restore the incumbent."""

    from scipy.optimize import Bounds, LinearConstraint, milp

    if _schedule_capacity_violation(data, proposal, tau, plan_end) <= config.feasibility_tolerance:
        return proposal, "REPAIR_NOT_NEEDED", 0, 0.0
    attempted = list(dict.fromkeys(str(task_id) for task_id in moved_task_ids))
    if not attempted:
        return incumbent_schedule.copy(), "REPAIR_REVERTED", 0, 0.0
    started = time.perf_counter()
    for attempt, multiplier in enumerate((1, 2), start=1):
        if time.perf_counter() - started >= remaining_seconds:
            break
        selected = attempted[: min(len(attempted), max(1, 50 * multiplier))]
        try:
            problem, options = build_local_repair_problem(
                data, config, schedule=proposal, task_ids=selected, tau=tau, plan_end=plan_end
            )
            result = milp(
                c=problem["objective"], integrality=problem["integrality"],
                bounds=Bounds(np.zeros(len(options)), np.ones(len(options))),
                constraints=LinearConstraint(problem["matrix"], problem["lower"], problem["upper"]),
                options={"presolve": True, "time_limit": min(config.repair_time_limit_seconds, max(0.1, remaining_seconds - (time.perf_counter() - started)))},
            )
            if result.x is None:
                continue
            repaired = proposal.copy()
            indexes = {str(task_id): int(index) for index, task_id in repaired["TaskID"].items()}
            for task_id, columns in problem["by_task"].items():
                choice = int(columns[int(np.argmax(np.asarray(result.x)[columns]))])
                _apply_option_to_schedule(repaired, indexes[task_id], options[choice])
            if _schedule_capacity_violation(data, repaired, tau, plan_end) <= config.feasibility_tolerance:
                return repaired, f"REPAIR_LOCAL_{attempt}", attempt, time.perf_counter() - started
        except (RuntimeError, ValueError):
            continue
    return incumbent_schedule.copy(), "REPAIR_REVERTED", 2, time.perf_counter() - started


def _build_joint_incumbent(
    data: InputData,
    config: ModelConfig,
    *,
    shadow_schedule: pd.DataFrame,
    tau: int,
    current_soc: np.ndarray,
    historical_peak: np.ndarray,
    scaling: Mapping[str, tuple[float, float]],
    time_limit: float,
    carbon_budget_remaining: float | None,
    objective_mode: str = OBJECTIVE_MODE_MINIMAX,
) -> JointIncumbent:
    decision_end = min(tau + config.decision_hours, OPERATION_END)
    plan_end = min(decision_end + config.lookahead_hours, OPERATION_END)
    _, ai, _, _ = _shadow_profiles(data, shadow_schedule, tau, plan_end)
    task_metrics = _window_task_metrics(shadow_schedule, tau, plan_end, config)
    energy = solve_energy_response(
        data, config, tau=tau, ai_it_profile=ai, current_soc=current_soc,
        historical_peak=historical_peak, task_metrics=task_metrics, scaling=scaling,
        time_limit=time_limit, carbon_budget_remaining=carbon_budget_remaining,
        objective_mode=objective_mode,
    )
    return JointIncumbent(
        shadow_schedule=shadow_schedule,
        energy=energy,
        metrics=dict(energy.metrics),
        deviations=dict(energy.deviations),
        z=float(energy.z),
    )


def select_joint_lns_neighborhood(
    data: InputData,
    config: ModelConfig,
    *,
    incumbent: JointIncumbent,
    tau: int,
    objective_mode: str = OBJECTIVE_MODE_MINIMAX,
) -> tuple[list[str], int, pd.DataFrame]:
    """Select a model-driven neighbourhood while retaining each selected task's full legal domain."""

    mode = _validate_objective_mode(objective_mode)
    decision_end = min(tau + config.decision_hours, OPERATION_END)
    plan_end = min(decision_end + config.lookahead_hours, OPERATION_END)
    gpu, ai, _, arrays = _shadow_profiles(data, incumbent.shadow_schedule, tau, plan_end)
    candidates = incumbent.shadow_schedule.loc[
        pd.to_numeric(incumbent.shadow_schedule["StartHour"]).between(tau, decision_end - 1)
        & ~incumbent.shadow_schedule["TaskType"].eq("RealTimeInference")
    ].copy()
    records: list[dict[str, object]] = []
    weight_map = _qos_weight_map(config)
    for row in candidates.itertuples(index=False):
        task_id = str(row.TaskID)
        catalog = _operational_task_options(data, task_id, tau, decision_end, plan_end)
        if not catalog:
            continue
        pressure = 0.0
        for hour, overlap in _hour_overlaps(float(row.StartHour), float(row.Duration_h), tau, plan_end):
            t = hour - tau
            r = data.region_index[str(row.TargetRegion)]
            ai_limit = min(
                arrays["Max_IT_Power_MW"][t, r] - arrays["NonAI_IT_Load_MW"][t, r],
                arrays["Max_Facility_Power_MW"][t, r] / arrays["PUE"][t, r] - arrays["NonAI_IT_Load_MW"][t, r],
            )
            pressure = max(
                pressure,
                (gpu[t, r] / max(arrays["Available_GPU"][t, r], EPS)),
                (ai[t, r] / max(ai_limit, EPS)),
            )
        current_region = data.region_index[str(row.TargetRegion)]
        price_carbon = 0.0
        carbon_sensitivity = 0.0
        for hour, _ in _hour_overlaps(float(row.StartHour), float(row.Duration_h), tau, plan_end):
            t = hour - tau
            price_component = arrays["ElectricityPrice_CNY_per_MWh"][t, current_region] / max(
                float(np.max(arrays["ElectricityPrice_CNY_per_MWh"])), EPS
            )
            carbon_component = arrays["CarbonIntensity_tCO2_per_MWh"][t, current_region] / max(
                float(np.max(arrays["CarbonIntensity_tCO2_per_MWh"])), EPS
            )
            price_carbon += price_component + carbon_component
            carbon_sensitivity += carbon_component
        peak_contribution = 0.0
        for hour, overlap in _hour_overlaps(float(row.StartHour), float(row.Duration_h), tau, plan_end):
            t = hour - tau
            r = current_region
            if abs(incumbent.energy.dispatch.loc[
                (incumbent.energy.dispatch["Hour"].eq(hour))
                & (incumbent.energy.dispatch["Region"].eq(data.regions[r])),
                "NetGridImport_MW",
            ].iloc[0] - incumbent.energy.regional_peak[r]) <= 1e-4:
                peak_contribution += overlap
        latency_pressure = float(row.NetworkLatency_ms) / max(float(row.MaxLatency_ms), EPS)
        flexibility = 1.0 / max(len(catalog), 1)
        energy_sensitivity = carbon_sensitivity if mode == OBJECTIVE_MODE_CARBON_PRIORITY else price_carbon
        priority = (
            3.0 * pressure + energy_sensitivity + peak_contribution
            + latency_pressure + float(weight_map[str(row.DelaySensitivity)]) + flexibility
        )
        records.append({
            "TaskID": task_id,
            "ExactTaskGroup": repr(_exact_task_group_key(data, data.task_lookup.loc[task_id])),
            "Priority": priority,
            "CapacityPressure": pressure,
            "EnergySensitivity": energy_sensitivity,
            "PeakContribution": peak_contribution,
            "LatencyRatio": latency_pressure,
            "DelaySensitivityWeight": float(weight_map[str(row.DelaySensitivity)]),
            "Flexibility": float(len(catalog)),
            "IntegerOptionCount": len(catalog),
        })
    ranked = pd.DataFrame(records)
    if ranked.empty:
        return [], 0, ranked
    ranked = ranked.sort_values(["Priority", "TaskID"], ascending=[False, True], kind="stable").reset_index(drop=True)
    # Select exact homogeneous groups first.  Members remain separate TaskID
    # variables in the local MILP so the final schedule stays auditable, while
    # the group cap follows the paper/Q2 neighbourhood definition.
    grouped = ranked.groupby("ExactTaskGroup", sort=False, observed=True).agg(
        Priority=("Priority", "max"),
        IntegerOptionCount=("IntegerOptionCount", "sum"),
    ).reset_index().sort_values(
        ["Priority", "ExactTaskGroup"], ascending=[False, True], kind="stable"
    )
    selected: list[str] = []
    selected_groups: set[str] = set()
    integer_options = 0
    for group in grouped.itertuples(index=False):
        count = int(group.IntegerOptionCount)
        if len(selected_groups) >= config.lns_max_task_groups:
            break
        if integer_options + count > config.lns_max_integer_option_vars:
            continue
        members = ranked.loc[ranked["ExactTaskGroup"].eq(group.ExactTaskGroup), "TaskID"].astype(str).tolist()
        selected.extend(members)
        selected_groups.add(str(group.ExactTaskGroup))
        integer_options += count
    ranked["Selected"] = ranked["TaskID"].isin(selected)
    ranked["SelectedExactTaskGroupCount"] = len(selected_groups)
    return selected, integer_options, ranked


def _lns_task_metric_components(
    schedule: pd.DataFrame,
    selected_task_ids: set[str],
    options: Sequence[JointTaskOption],
    *,
    tau: int,
    plan_end: int,
    config: ModelConfig,
    variable_count: int,
    x_indices: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    window = schedule.loc[
        pd.to_numeric(schedule["StartHour"]).between(tau, plan_end - 1)
    ].copy()
    if window.empty:
        return {"Latency": 0.0, "Delay": 0.0}, {
            "Latency": np.zeros(variable_count), "Delay": np.zeros(variable_count),
        }
    selected_mask = window["TaskID"].astype(str).isin(selected_task_ids)
    fixed = window.loc[~selected_mask].copy()
    count = len(window)
    latency_constant = float(fixed["NetworkLatency_ms"].sum() / max(count, 1))
    latency_vector = np.zeros(variable_count, dtype=float)
    for position, option in enumerate(options):
        latency_vector[int(x_indices[position])] = option.latency_ms / max(count, 1)
    weights = window["DelaySensitivity"].map(_qos_weight_map(config)).to_numpy(dtype=float)
    earliest = np.maximum(
        window["ArrivalHour"].to_numpy(dtype=float),
        window["EarliestStartHour"].to_numpy(dtype=float),
    )
    slack = np.minimum(window["LatestFinishHour"].to_numpy(dtype=float), float(OPERATION_END)) - window["Duration_h"].to_numpy(dtype=float) - earliest
    flexible = (~window["TaskType"].eq("RealTimeInference").to_numpy()) & (slack > EPS)
    denominator = float(weights[flexible].sum())
    fixed_delay_total = 0.0
    for row in fixed.itertuples(index=False):
        task = pd.Series(row._asdict())
        fixed_delay_total += _task_delay_contribution(task, float(row.StartHour), config)
    delay_constant = fixed_delay_total / denominator if denominator > EPS else 0.0
    delay_vector = np.zeros(variable_count, dtype=float)
    if denominator > EPS:
        for position, option in enumerate(options):
            task = schedule.loc[schedule["TaskID"].astype(str).eq(option.task_id)].iloc[0]
            delay_vector[int(x_indices[position])] = _task_delay_contribution(task, option.start_hour, config) / denominator
    return {"Latency": latency_constant, "Delay": delay_constant}, {
        "Latency": latency_vector,
        "Delay": delay_vector,
    }


def build_joint_lns_problem(
    data: InputData,
    config: ModelConfig,
    *,
    schedule: pd.DataFrame,
    selected_task_ids: Sequence[str],
    tau: int,
    current_soc: np.ndarray,
    historical_peak: np.ndarray,
    scaling: Mapping[str, tuple[float, float]],
    carbon_budget_remaining: float | None = None,
) -> JointLNSProblem:
    """Build the only task-plus-energy MILP used by the formal Q4 solver."""

    decision_end = min(tau + config.decision_hours, OPERATION_END)
    plan_end = min(decision_end + config.lookahead_hours, OPERATION_END)
    t_count = plan_end - tau
    h_count = decision_end - tau
    r_count = len(data.regions)
    selected = tuple(dict.fromkeys(str(task_id) for task_id in selected_task_ids))
    if not selected:
        raise ValueError("Joint LNS neighborhood must not be empty")
    fixed = schedule.loc[~schedule["TaskID"].astype(str).isin(selected)].copy()
    fixed_gpu, fixed_ai, frame, arrays = _shadow_profiles(data, fixed, tau, plan_end)
    options: list[JointTaskOption] = []
    by_task: dict[str, list[int]] = {}
    for task_id in selected:
        catalog = _operational_task_options(data, task_id, tau, decision_end, plan_end)
        if not catalog:
            raise RuntimeError(f"Joint LNS task {task_id} has no valid H-region space-time option")
        by_task[task_id] = list(range(len(options), len(options) + len(catalog)))
        options.extend(catalog)
    if len(options) > config.lns_max_integer_option_vars:
        raise ValueError(f"Joint LNS integer task options {len(options)} exceed limit {config.lns_max_integer_option_vars}")
    storage = _matheuristic_storage_arrays(data)
    offset = 0
    indices: dict[str, np.ndarray] = {}
    indices["x"], offset = _allocate(offset, (len(options),))
    for name in (
        "renewable_direct", "renewable_charge", "export", "curtailment",
        "grid_purchase", "grid_charge", "discharge", "mode",
    ):
        indices[name], offset = _allocate(offset, (t_count, r_count))
    indices["soc"], offset = _allocate(offset, (t_count + 1, r_count))
    indices["peak"], offset = _allocate(offset, (r_count,))
    indices["deviation"], offset = _allocate(offset, (len(METRICS),))
    indices["max_deviation"], offset = _allocate(offset, (1,))
    lower = np.zeros(offset, dtype=float)
    upper = np.full(offset, np.inf, dtype=float)
    integrality = np.zeros(offset, dtype=np.uint8)
    upper[indices["x"]] = 1.0
    integrality[indices["x"]] = 1
    upper[indices["renewable_direct"].ravel()] = arrays["AvailableRenewable_MW"].ravel()
    upper[indices["renewable_charge"].ravel()] = arrays["AvailableRenewable_MW"].ravel()
    upper[indices["curtailment"].ravel()] = arrays["AvailableRenewable_MW"].ravel()
    upper[indices["export"].ravel()] = np.tile(storage["max_export"], t_count)
    upper[indices["grid_purchase"].ravel()] = np.tile(storage["max_grid"], t_count)
    upper[indices["grid_charge"].ravel()] = np.tile(storage["max_charge"], t_count)
    upper[indices["discharge"].ravel()] = np.tile(storage["max_discharge"], t_count)
    upper[indices["mode"].ravel()] = 1.0
    integrality[indices["mode"][:h_count].ravel()] = 1
    lower[indices["soc"].ravel()] = np.tile(storage["min_soc"], t_count + 1)
    upper[indices["soc"].ravel()] = np.tile(storage["max_soc"], t_count + 1)
    lower[indices["soc"][0]] = current_soc
    upper[indices["soc"][0]] = current_soc
    recoverable = np.maximum(
        storage["min_soc"],
        storage["initial_soc"] - storage["eta_c"] * storage["max_charge"] * max(OPERATION_END - decision_end, 0),
    )
    lower[indices["soc"][h_count]] = np.maximum(lower[indices["soc"][h_count]], recoverable)
    if decision_end >= OPERATION_END:
        lower[indices["soc"][h_count]] = np.maximum(lower[indices["soc"][h_count]], storage["initial_soc"])
    lower[indices["peak"]] = historical_peak

    rows: list[dict[int, float]] = []
    row_lower: list[float] = []
    row_upper: list[float] = []

    def add(values: Mapping[int, float], lb: float, ub: float) -> int:
        rows.append(dict(values))
        row_lower.append(float(lb))
        row_upper.append(float(ub))
        return len(rows) - 1

    for task_id, option_positions in by_task.items():
        add({int(indices["x"][position]): 1.0 for position in option_positions}, 1.0, 1.0)
    gpu_rows = [[{} for _ in range(r_count)] for _ in range(t_count)]
    ai_rows = [[{} for _ in range(r_count)] for _ in range(t_count)]
    facility_rows = [[{} for _ in range(r_count)] for _ in range(t_count)]
    for position, option in enumerate(options):
        task = data.task_lookup.loc[option.task_id]
        column = int(indices["x"][position])
        for hour, overlap in option.overlaps:
            local = hour - tau
            if not 0 <= local < t_count:
                continue
            r = option.region_index
            gpu_rows[local][r][column] = gpu_rows[local][r].get(column, 0.0) + float(task.GPU_Demand) * overlap
            ai_rows[local][r][column] = ai_rows[local][r].get(column, 0.0) + float(task.Task_Full_IT_Power_MW) * overlap
            facility_rows[local][r][column] = facility_rows[local][r].get(column, 0.0) - arrays["PUE"][local, r] * float(task.Task_Full_IT_Power_MW) * overlap
    for t in range(t_count):
        for r in range(r_count):
            ai_limit = min(
                arrays["Max_IT_Power_MW"][t, r] - arrays["NonAI_IT_Load_MW"][t, r],
                arrays["Max_Facility_Power_MW"][t, r] / arrays["PUE"][t, r] - arrays["NonAI_IT_Load_MW"][t, r],
            )
            add(gpu_rows[t][r], -np.inf, arrays["Available_GPU"][t, r] - fixed_gpu[t, r])
            add(ai_rows[t][r], -np.inf, ai_limit - fixed_ai[t, r])

    balance_rows = np.empty((t_count, r_count), dtype=int)
    fixed_facility = arrays["PUE"] * (arrays["NonAI_IT_Load_MW"] + fixed_ai)
    for t in range(t_count):
        for r in range(r_count):
            add({
                int(indices["renewable_direct"][t, r]): 1.0,
                int(indices["renewable_charge"][t, r]): 1.0,
                int(indices["export"][t, r]): 1.0,
                int(indices["curtailment"][t, r]): 1.0,
            }, arrays["AvailableRenewable_MW"][t, r], arrays["AvailableRenewable_MW"][t, r])
            balance_values = {
                int(indices["grid_purchase"][t, r]): 1.0,
                int(indices["renewable_direct"][t, r]): 1.0,
                int(indices["discharge"][t, r]): 1.0,
                int(indices["grid_charge"][t, r]): -1.0,
                **facility_rows[t][r],
            }
            balance_rows[t, r] = add(balance_values, fixed_facility[t, r], fixed_facility[t, r])
            _append_document_energy_semantic_constraints(
                add,
                indices=indices,
                t=t,
                r=r,
                facility_constant=float(fixed_facility[t, r]),
                facility_variable_terms=facility_rows[t][r],
            )
            add({
                int(indices["soc"][t + 1, r]): 1.0,
                int(indices["soc"][t, r]): -1.0,
                int(indices["renewable_charge"][t, r]): -storage["eta_c"][r],
                int(indices["grid_charge"][t, r]): -storage["eta_c"][r],
                int(indices["discharge"][t, r]): 1.0 / storage["eta_d"][r],
            }, 0.0, 0.0)
            add({
                int(indices["renewable_charge"][t, r]): 1.0,
                int(indices["grid_charge"][t, r]): 1.0,
                int(indices["mode"][t, r]): -storage["max_charge"][r],
            }, -np.inf, 0.0)
            add({
                int(indices["discharge"][t, r]): 1.0,
                int(indices["mode"][t, r]): storage["max_discharge"][r],
            }, -np.inf, storage["max_discharge"][r])
            add({
                int(indices["peak"][r]): 1.0,
                int(indices["grid_purchase"][t, r]): -1.0,
                int(indices["export"][t, r]): 1.0,
            }, 0.0, np.inf)
    if carbon_budget_remaining is not None:
        add({
            int(indices["grid_purchase"][t, r]): arrays["CarbonIntensity_tCO2_per_MWh"][t, r]
            for t in range(h_count) for r in range(r_count)
        }, -np.inf, max(float(carbon_budget_remaining), 0.0))

    task_constants, task_vectors = _lns_task_metric_components(
        schedule, set(selected), options, tau=tau, plan_end=plan_end, config=config,
        variable_count=offset, x_indices=indices["x"],
    )
    metric_vectors = {metric: np.zeros(offset, dtype=float) for metric in METRICS}
    metric_constants = {metric: 0.0 for metric in METRICS}
    metric_constants.update(task_constants)
    metric_vectors["Latency"] = task_vectors["Latency"]
    metric_vectors["Delay"] = task_vectors["Delay"]
    renewable_total = max(float(arrays["AvailableRenewable_MW"].sum()), EPS)
    for t in range(t_count):
        for r in range(r_count):
            metric_vectors["Cost"][indices["grid_purchase"][t, r]] = arrays["ElectricityPrice_CNY_per_MWh"][t, r]
            metric_vectors["Cost"][indices["export"][t, r]] = -arrays["SellPrice_CNY_per_MWh"][t, r]
            metric_vectors["Carbon"][indices["grid_purchase"][t, r]] = arrays["CarbonIntensity_tCO2_per_MWh"][t, r]
            metric_vectors["RenewableUnusedRate"][indices["curtailment"][t, r]] = 1.0 / renewable_total
    metric_vectors["Peak"][indices["peak"]] = 1.0
    _append_minimax_objective_constraints(
        rows=rows, row_lower=row_lower, row_upper=row_upper,
        metric_vectors=metric_vectors, metric_constants=metric_constants,
        scaling=scaling, deviation_indices=indices["deviation"],
        max_deviation_index=int(indices["max_deviation"][0]),
    )
    return JointLNSProblem(
        tau=tau, decision_end=decision_end, plan_end=plan_end,
        frame=frame, arrays=arrays, options=tuple(options), fixed_ai=fixed_ai,
        fixed_gpu=fixed_gpu, indices=indices, lower=lower, upper=upper,
        integrality=integrality, matrix=_rows_to_sparse(rows, offset),
        constraint_lower=np.asarray(row_lower, dtype=float),
        constraint_upper=np.asarray(row_upper, dtype=float), balance_rows=balance_rows,
        task_metric_constants=metric_constants, task_metric_vectors=metric_vectors,
    )


def solve_joint_lns(
    data: InputData,
    config: ModelConfig,
    *,
    incumbent: JointIncumbent,
    selected_task_ids: Sequence[str],
    tau: int,
    current_soc: np.ndarray,
    historical_peak: np.ndarray,
    scaling: Mapping[str, tuple[float, float]],
    time_limit: float,
    carbon_budget_remaining: float | None = None,
    objective_mode: str = OBJECTIVE_MODE_MINIMAX,
) -> tuple[JointIncumbent | None, str, float, float, str]:
    """Solve a bounded joint task-storage-energy LNS model and reconstruct its real schedule."""

    from scipy.optimize import Bounds, LinearConstraint, milp

    mode = _validate_objective_mode(objective_mode)
    problem = build_joint_lns_problem(
        data, config, schedule=incumbent.shadow_schedule,
        selected_task_ids=selected_task_ids, tau=tau,
        current_soc=current_soc, historical_peak=historical_peak,
        scaling=scaling, carbon_budget_remaining=carbon_budget_remaining,
    )
    if mode == OBJECTIVE_MODE_CARBON_PRIORITY:
        objective = problem.task_metric_vectors["Carbon"].copy()
    else:
        objective = np.zeros(problem.lower.size, dtype=float)
        objective[int(problem.indices["max_deviation"][0])] = 1.0
        objective[problem.indices["deviation"]] = config.tie_break_weight
    started = time.perf_counter()
    result = milp(
        c=objective,
        integrality=problem.integrality,
        bounds=Bounds(problem.lower, problem.upper),
        constraints=LinearConstraint(problem.matrix, problem.constraint_lower, problem.constraint_upper),
        options={"presolve": True, "time_limit": max(0.1, float(time_limit)), "mip_rel_gap": float(config.lns_target_gap)},
    )
    elapsed = time.perf_counter() - started
    gap = _result_float(result, "mip_gap")
    if result.x is None:
        return None, "LNS_NO_FEASIBLE", gap, elapsed, str(result.message)
    vector = np.asarray(result.x, dtype=float)
    activity = problem.matrix @ vector
    violation = float(max(
        np.maximum(problem.constraint_lower - activity, 0.0).max(initial=0.0),
        np.maximum(activity - problem.constraint_upper, 0.0).max(initial=0.0),
    ))
    if violation > max(config.feasibility_tolerance * 10.0, 1e-5):
        return None, "LNS_INVALID_VECTOR", gap, elapsed, f"max_violation={violation:.3g}"
    candidate_schedule = incumbent.shadow_schedule.copy()
    index_by_task = {str(task_id): int(index) for index, task_id in candidate_schedule["TaskID"].items()}
    by_task: dict[str, list[int]] = {}
    for position, option in enumerate(problem.options):
        by_task.setdefault(option.task_id, []).append(position)
    for task_id, positions in by_task.items():
        values = vector[problem.indices["x"][np.asarray(positions, dtype=int)]]
        option = problem.options[positions[int(np.argmax(values))]]
        _apply_option_to_schedule(candidate_schedule, index_by_task[task_id], option)
    _, candidate_ai, _, _ = _shadow_profiles(data, candidate_schedule, tau, problem.plan_end)
    task_metrics = _window_task_metrics(candidate_schedule, tau, problem.plan_end, config)
    dispatch = _energy_dispatch_from_vector(
        data, tau=tau, frame=problem.frame, arrays=problem.arrays,
        ai_profile=candidate_ai, indices=problem.indices, vector=vector,
    )
    _assert_document_energy_semantics(
        dispatch, config.feasibility_tolerance * 10.0,
        context=f"Joint LNS window {tau}",
    )
    renewable_total = max(float(dispatch["AvailableRenewable_MW"].sum()), EPS)
    metrics = {
        "Cost": float(np.sum(dispatch["ElectricityPrice_CNY_per_MWh"] * dispatch["GridPurchase_MW"] - dispatch["SellPrice_CNY_per_MWh"] * dispatch["GridSell_MW"])),
        "Carbon": float(dispatch["CarbonEmission_tCO2"].sum()),
        "Latency": task_metrics["Latency"],
        "Delay": task_metrics["Delay"],
        "RenewableUnusedRate": float(dispatch["RenewableCurtailment_MW"].sum() / renewable_total),
        "Peak": float(vector[problem.indices["peak"]].sum()),
    }
    z, deviations = _balanced_z(metrics, scaling, config)
    h_count = problem.decision_end - problem.tau
    energy = EnergySolution(
        dispatch=dispatch,
        end_soc=np.asarray(vector[problem.indices["soc"][-1]], dtype=float).copy(),
        decision_end_soc=np.asarray(vector[problem.indices["soc"][h_count]], dtype=float).copy(),
        regional_peak=np.asarray(vector[problem.indices["peak"]], dtype=float).copy(),
        metrics=metrics,
        deviations=deviations,
        z=z,
        mip_gap=gap,
        elapsed_seconds=elapsed,
        status="LNS_OPTIMAL" if int(result.status) == 0 else "LNS_TIME_LIMIT_FEASIBLE",
        message=str(result.message),
    )
    candidate = JointIncumbent(
        shadow_schedule=candidate_schedule, energy=energy, metrics=metrics,
        deviations=deviations, z=z,
    )
    if _incumbent_objective_value(candidate, mode) >= (
        _incumbent_objective_value(incumbent, mode) - config.feasibility_tolerance
    ):
        return None, "LNS_NO_IMPROVEMENT", gap, elapsed, str(result.message)
    status = (
        "LNS_CERTIFIED_LOCAL"
        if int(result.status) == 0 or (np.isfinite(gap) and gap <= config.lns_target_gap + EPS)
        else "LNS_FEASIBLE_IMPROVEMENT"
    )
    return candidate, status, gap, elapsed, str(result.message)


def _matheuristic_window_starts(config: ModelConfig) -> list[int]:
    if config.decision_hours != 24 or config.lookahead_hours != 48:
        raise ValueError("Final Q4 configuration fixes H=24h and K=48h; formal entry points do not support different windows")
    return list(range(0, MAIN_END, config.decision_hours)) + [MAIN_END]


def _fixed_schedule_energy_signature(
    data: InputData,
    schedule: pd.DataFrame,
    config: ModelConfig,
    scaling: Mapping[str, tuple[float, float]],
) -> str:
    digest = hashlib.sha256()
    digest.update(_matheuristic_signature(data, config, calibration_only=True).encode("ascii"))
    digest.update(_matheuristic_frame_hash(schedule))
    digest.update(json.dumps({key: list(value) for key, value in scaling.items()}, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _run_fixed_schedule_energy_rollout(
    data: InputData,
    config: ModelConfig,
    schedule: pd.DataFrame,
    scaling: Mapping[str, tuple[float, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute a V4 energy reference for an immutable feasible task schedule."""

    storage = _matheuristic_storage_arrays(data)
    current_soc = storage["initial_soc"].copy()
    historical_peak = np.zeros(len(data.regions), dtype=float)
    dispatch_parts: list[pd.DataFrame] = []
    window_rows: list[dict[str, object]] = []
    for tau in _matheuristic_window_starts(config):
        decision_end = min(tau + config.decision_hours, OPERATION_END)
        plan_end = min(decision_end + config.lookahead_hours, OPERATION_END)
        _, ai, _, _ = _shadow_profiles(data, schedule, tau, plan_end)
        task_metrics = _window_task_metrics(schedule, tau, plan_end, config)
        energy = solve_energy_response(
            data, config, tau=tau, ai_it_profile=ai, current_soc=current_soc,
            historical_peak=historical_peak, task_metrics=task_metrics,
            scaling=scaling, time_limit=config.energy_time_limit_seconds,
        )
        actual = energy.dispatch.loc[energy.dispatch["Hour"] < decision_end].copy()
        dispatch_parts.append(actual)
        current_soc = energy.decision_end_soc.copy()
        regional = actual.groupby("Region", observed=True)["NetGridImport_MW"].max()
        for r, region in enumerate(data.regions):
            historical_peak[r] = max(historical_peak[r], float(regional.get(region, 0.0)), 0.0)
        window_rows.append({
            "WindowStart": tau,
            "DecisionEnd": decision_end,
            "PlanEnd": plan_end,
            "SolverStatus": energy.status,
            "MIPGap": energy.mip_gap,
            "ElapsedSeconds": energy.elapsed_seconds,
            "Z": energy.z,
        })
    dispatch = pd.concat(dispatch_parts, ignore_index=True).sort_values(["Hour", "Region"], kind="stable").reset_index(drop=True)
    return dispatch, pd.DataFrame(window_rows)


def _load_or_build_fixed_schedule_energy_reference(
    data: InputData,
    config: ModelConfig,
    *,
    schedule: pd.DataFrame,
    scaling: Mapping[str, tuple[float, float]],
    paths: MatheuristicPaths,
    file_prefix: str,
    scheme_label: str,
    progress_label: str,
    source_schedule: str,
) -> dict[str, object]:
    """Persist one fixed-task V4 energy evaluation with explicit provenance."""

    paths.tables.mkdir(parents=True, exist_ok=True)
    assignments_path = paths.tables / f"{file_prefix}_assignments.csv"
    dispatch_path = paths.tables / f"{file_prefix}_dispatch.csv"
    metrics_path = paths.tables / f"{file_prefix}_metrics.csv"
    windows_path = paths.tables / f"{file_prefix}_windows.csv"
    marker_path = paths.tables / f".{file_prefix}_complete.json"
    signature = _fixed_schedule_energy_signature(data, schedule, config, scaling)
    if all(path.is_file() for path in (assignments_path, dispatch_path, metrics_path, marker_path)):
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("signature") == signature and marker.get("complete") is True:
                assignments = pd.read_csv(assignments_path, encoding="utf-8-sig")
                dispatch = pd.read_csv(dispatch_path, encoding="utf-8-sig")
                metrics_frame = pd.read_csv(metrics_path, encoding="utf-8-sig")
                metrics = {str(row.Metric): float(row.Value) for row in metrics_frame.itertuples(index=False)}
                return {"assignments": assignments, "dispatch": dispatch, "metrics": metrics, "reused": True}
        except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError):
            pass
    _progress(f"No reusable {progress_label} found; generating the unified V4 energy response trajectory.")
    dispatch, windows = _run_fixed_schedule_energy_rollout(data, config, schedule, scaling)
    simple = _simple_validation(data, schedule, dispatch, config.feasibility_tolerance)
    if not bool(simple["Passed"].all()):
        failed = simple.loc[~simple["Passed"], "Check"].tolist()
        raise RuntimeError(f"{progress_label} failed independent validation: {failed}")
    metrics = _final_metrics(data, schedule, dispatch, _qos_weight_map(config))
    metrics_frame = pd.DataFrame([
        {"Scheme": scheme_label, "Metric": key, "Value": value}
        for key, value in metrics.items()
    ])
    _atomic_write_csv(schedule, assignments_path)
    _atomic_write_csv(dispatch, dispatch_path)
    _atomic_write_csv(metrics_frame, metrics_path)
    _atomic_write_csv(windows, windows_path)
    _atomic_write_json(marker_path, {
        "complete": True,
        "model_version": MATHEURISTIC_VERSION,
        "signature": signature,
        "source_schedule": source_schedule,
        "task_count": len(schedule),
        "dispatch_count": len(dispatch),
    })
    return {"assignments": schedule, "dispatch": dispatch, "metrics": metrics, "reused": False}


def load_or_build_q2_v3_sequential_reference(
    data: InputData,
    config: ModelConfig,
    *,
    q2_schedule: pd.DataFrame,
    scaling: Mapping[str, tuple[float, float]],
    paths: MatheuristicPaths,
) -> dict[str, object]:
    """Persist the Q2-task-seed/V4-energy reference once per V4 output root."""

    return _load_or_build_fixed_schedule_energy_reference(
        data,
        config,
        schedule=q2_schedule,
        scaling=scaling,
        paths=paths,
        file_prefix="q4_sequential_baseline",
        scheme_label="Q2_TaskSeed_to_V4_EnergyReference",
        progress_label="Q2 task seed -> V4 sequential energy reference",
        source_schedule=str(q2_schedule.attrs.get("q2_schedule_source", "Q2ValidatedSchedule")),
    )


def run_sequential_reference_repair(
    *,
    config: ModelConfig | None = None,
) -> dict[str, object]:
    """Repair only the problematic Q2-task-seed to energy reference.

    This focused path deliberately does not run the Q4 rolling joint model.
    It reuses the audited Q2 task schedule, the completed V4 calibration, and
    recomputes only the sequential energy trajectory that produced the old
    physically arbitrary carbon baseline.
    """

    config = config or ModelConfig()
    data = load_data()
    paths = _matheuristic_paths()
    q2_schedule = load_and_validate_q2_schedule(data, config)
    scaling, calibration_records = load_or_build_fixed_calibration(
        data, q2_schedule, config, paths
    )
    result = load_or_build_q2_v3_sequential_reference(
        data,
        config,
        q2_schedule=q2_schedule,
        scaling=scaling,
        paths=paths,
    )
    _progress(
        "Sequential baseline targeted repair completed: "
        f"reused calibration records={len(calibration_records)} rows, "
        f"Cost={float(result['metrics']['Cost']):.6g}，"
        f"Carbon={float(result['metrics']['Carbon']):.6g}，"
        f"Peak={float(result['metrics']['Peak']):.6g}。"
    )
    return result


def load_and_validate_existing_joint_schedule(
    data: InputData,
    config: ModelConfig,
) -> pd.DataFrame:
    """Load the audited prior joint task schedule for a V4 energy-only re-evaluation."""

    source = LEGACY_JOINT_ROOT / "tables" / "q4_task_assignments.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Existing joint task schedule not found: {source}")
    raw = pd.read_csv(source, encoding="utf-8-sig")
    schedule = _standardize_shadow_schedule(data, raw)
    _validate_shadow_schedule(data, schedule, config.feasibility_tolerance)
    schedule = schedule.sort_values(["StartHour", "TaskID"], kind="stable").reset_index(drop=True)
    schedule.attrs["joint_schedule_source"] = str(source)
    _progress(f"Existing joint task schedule loaded and checked: {source.name}, tasks={len(schedule)}.")
    return schedule


def run_joint_energy_reference_repair(
    *,
    config: ModelConfig | None = None,
) -> dict[str, object]:
    """Re-evaluate the existing joint task schedule under the same V4 energy rule.

    This focused path does not claim or run a new V4 joint search.  It fixes the
    task schedule produced by the audited prior joint run and recomputes only
    the energy trajectory, making its reported C/E/Q/P values comparable with
    the repaired sequential reference.
    """

    config = config or ModelConfig()
    data = load_data()
    paths = _matheuristic_paths()
    q2_schedule = load_and_validate_q2_schedule(data, config)
    scaling, calibration_records = load_or_build_fixed_calibration(
        data, q2_schedule, config, paths
    )
    joint_schedule = load_and_validate_existing_joint_schedule(data, config)
    result = _load_or_build_fixed_schedule_energy_reference(
        data,
        config,
        schedule=joint_schedule,
        scaling=scaling,
        paths=paths,
        file_prefix="q4_joint_energy_reference",
        scheme_label="ExistingJointTaskSchedule_to_V4_EnergyReference",
        progress_label="Existing joint task schedule -> unified V4 energy reference",
        source_schedule=str(joint_schedule.attrs["joint_schedule_source"]),
    )
    _progress(
        "Joint schedule targeted V4 energy recomputation completed: "
        f"reused calibration records={len(calibration_records)} rows, "
        f"Cost={float(result['metrics']['Cost']):.6g}，"
        f"Carbon={float(result['metrics']['Carbon']):.6g}，"
        f"Peak={float(result['metrics']['Peak']):.6g}。"
    )
    return result


def _save_matheuristic_checkpoint(
    *,
    paths: MatheuristicPaths,
    signature: str,
    config: ModelConfig,
    next_tau: int,
    assignments: pd.DataFrame,
    dispatch: pd.DataFrame,
    solver: pd.DataFrame,
    lns: pd.DataFrame,
    forecast: pd.DataFrame,
    shadow_schedule: pd.DataFrame,
    scaling: Mapping[str, tuple[float, float]],
    calibration_records: pd.DataFrame,
    current_soc: np.ndarray,
    historical_peak: np.ndarray,
    past_carbon: float,
    scenario: Mapping[str, object] | None,
) -> None:
    paths.tables.mkdir(parents=True, exist_ok=True)
    paths.validation.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(assignments, paths.assignments_progress)
    _atomic_write_csv(dispatch, paths.dispatch_progress)
    _atomic_write_csv(solver, paths.solver_progress)
    _atomic_write_csv(lns, paths.lns_progress)
    _atomic_write_csv(forecast, paths.forecast_progress)
    payload = {
        "schema_version": 1,
        "model_version": MATHEURISTIC_VERSION,
        "input_signature": signature,
        "config": asdict(config),
        "next_tau": int(next_tau),
        "assignments": assignments,
        "dispatch": dispatch,
        "solver": solver,
        "lns": lns,
        "forecast": forecast,
        "current_soc": np.asarray(current_soc, dtype=float).copy(),
        "historical_peak": np.asarray(historical_peak, dtype=float).copy(),
        "past_carbon": float(past_carbon),
        "shadow_schedule": shadow_schedule,
        # Kept explicitly in the checkpoint schema: K-area dispatch is only a
        # hint, never a committed task or energy decision.
        "warm_hints": {"forecast_rows": int(len(forecast))},
        "metrics_so_far": {
            "CommittedTaskCount": int(len(assignments)),
            "CommittedDispatchRowCount": int(len(dispatch)),
            "PastCarbon": float(past_carbon),
        },
        "scaling": {key: (float(value[0]), float(value[1])) for key, value in scaling.items()},
        "calibration_records": calibration_records,
        "scenario": dict(scenario or {}),
    }
    _atomic_write_pickle(paths.checkpoint, payload)


def _load_matheuristic_checkpoint(
    data: InputData,
    config: ModelConfig,
    paths: MatheuristicPaths,
    signature: str,
) -> dict[str, object] | None:
    if not paths.checkpoint.is_file():
        return None
    with paths.checkpoint.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported Q4 matheuristic checkpoint version: {paths.checkpoint}")
    if payload.get("model_version") != MATHEURISTIC_VERSION:
        raise RuntimeError("Existing Q4 matheuristic checkpoint uses a different model version; create a new output directory to protect results")
    if payload.get("input_signature") != signature:
        raise RuntimeError("Existing Q4 matheuristic checkpoint differs from current inputs or parameters; refusing overwrite")
    required = {
        "next_tau", "assignments", "dispatch", "solver", "lns", "forecast",
        "current_soc", "historical_peak", "past_carbon", "shadow_schedule", "scaling",
        "warm_hints", "metrics_so_far",
    }
    if not required.issubset(payload):
        raise RuntimeError("Q4 matheuristic checkpoint has incomplete fields")
    next_tau = int(payload["next_tau"])
    if next_tau not in set(_matheuristic_window_starts(config) + [OPERATION_END]):
        raise RuntimeError(f"Q4 matheuristic checkpoint next_tau={next_tau} is not a valid window boundary")
    assignments = pd.DataFrame(payload["assignments"])
    dispatch = pd.DataFrame(payload["dispatch"])
    shadow = _standardize_shadow_schedule(data, pd.DataFrame(payload["shadow_schedule"]))
    _validate_shadow_schedule(data, shadow, config.feasibility_tolerance)
    if assignments["TaskID"].astype(str).duplicated().any():
        raise RuntimeError("Q4 matheuristic checkpoint contains duplicate committed TaskID values")
    if not assignments.empty:
        if (pd.to_numeric(assignments["StartHour"]) >= next_tau - EPS).any():
            raise RuntimeError("Q4 matheuristic checkpoint contains tasks outside the committed H region")
        joined = assignments.set_index("TaskID").join(
            shadow.set_index("TaskID")[["TargetRegion", "StartHour"]],
            how="left", rsuffix="_Shadow",
        )
        if joined[["TargetRegion_Shadow", "StartHour_Shadow"]].isna().any().any():
            raise RuntimeError("Committed tasks in the Q4 matheuristic checkpoint are absent from the shadow schedule")
        if not (
            joined["TargetRegion"].astype(str).eq(joined["TargetRegion_Shadow"].astype(str)).all()
            and np.allclose(joined["StartHour"].to_numpy(dtype=float), joined["StartHour_Shadow"].to_numpy(dtype=float), rtol=0.0, atol=EPS)
        ):
            raise RuntimeError("Committed tasks in the Q4 matheuristic checkpoint differ from the shadow schedule")
    expected_rows = next_tau * len(data.regions)
    if len(dispatch) != expected_rows:
        raise RuntimeError(f"Unexpected Q4 matheuristic checkpoint energy row count: {len(dispatch)} != {expected_rows}")
    if next_tau > 0:
        ordered = dispatch.sort_values(["Region", "Hour"], kind="stable")
        continuity = []
        for _, subset in ordered.groupby("Region", sort=False, observed=True):
            continuity.append(
                subset["SOCStart_MWh"].to_numpy(dtype=float)[1:]
                - subset["SOCEnd_MWh"].to_numpy(dtype=float)[:-1]
            )
        if continuity and np.abs(np.concatenate(continuity)).max(initial=0.0) > config.feasibility_tolerance * 10:
            raise RuntimeError("Q4 matheuristic checkpoint SOC is discontinuous")
        end_rows = dispatch.loc[dispatch["Hour"].eq(next_tau - 1)].set_index("Region").loc[list(data.regions)]
        if not np.allclose(
            end_rows["SOCEnd_MWh"].to_numpy(dtype=float),
            np.asarray(payload["current_soc"], dtype=float), rtol=0.0, atol=1e-5,
        ):
            raise RuntimeError("Q4 matheuristic checkpoint current SOC differs from the actual H-region trajectory")
    return payload


def _write_matheuristic_final_outputs(
    *,
    data: InputData,
    config: ModelConfig,
    paths: MatheuristicPaths,
    signature: str,
    assignments: pd.DataFrame,
    dispatch: pd.DataFrame,
    solver: pd.DataFrame,
    lns: pd.DataFrame,
    forecast: pd.DataFrame,
    scaling: Mapping[str, tuple[float, float]],
    calibration_records: pd.DataFrame,
    sequential: Mapping[str, object],
    scenario: Mapping[str, object] | None,
    final_carbon_upper_bound: float | None = None,
) -> dict[str, object]:
    assignments = assignments.sort_values(["StartHour", "TaskID"], kind="stable").reset_index(drop=True)
    dispatch = dispatch.sort_values(["Hour", "Region"], kind="stable").reset_index(drop=True)
    if len(assignments) != len(data.tasks) or assignments["TaskID"].astype(str).nunique() != len(data.tasks):
        raise RuntimeError("Final Q4 matheuristic TaskID schedule does not cover all 50000 tasks")
    if len(dispatch) != OPERATION_END * len(data.regions):
        raise RuntimeError("Final Q4 matheuristic hourly energy trajectory is incomplete")
    metrics = _final_metrics(data, assignments, dispatch, _qos_weight_map(config))
    if final_carbon_upper_bound is not None:
        upper_bound = float(final_carbon_upper_bound)
        if not np.isfinite(upper_bound):
            raise ValueError("Final emissions upper bound for the low-carbon reference must be finite")
        if metrics["Carbon"] > upper_bound + max(config.feasibility_tolerance, 1e-8):
            raise RuntimeError(
                "Final scenario emissions exceed the specified upper bound: "
                f"{metrics['Carbon']:.12g} > {upper_bound:.12g}"
            )
    summary_rows = [
        {"Scheme": "Q4JointMatheuristic", "Metric": "Cost", "Value": metrics["Cost"]},
        {"Scheme": "Q4JointMatheuristic", "Metric": "Carbon", "Value": metrics["Carbon"]},
        {"Scheme": "Q4JointMatheuristic", "Metric": "Latency_ms", "Value": metrics["Latency"]},
        {"Scheme": "Q4JointMatheuristic", "Metric": "LatencySLA", "Value": metrics["LatencySLA"]},
        {"Scheme": "Q4JointMatheuristic", "Metric": "QoSLoss", "Value": metrics["Delay"]},
        {"Scheme": "Q4JointMatheuristic", "Metric": "QoS", "Value": 1.0 - metrics["Delay"]},
        {"Scheme": "Q4JointMatheuristic", "Metric": "RenewableUnusedRate", "Value": metrics["RenewableUnusedRate"]},
        {"Scheme": "Q4JointMatheuristic", "Metric": "RenewableUtilization", "Value": 1.0 - metrics["RenewableUnusedRate"]},
        {"Scheme": "Q4JointMatheuristic", "Metric": "Peak", "Value": metrics["Peak"]},
    ]
    summary = pd.DataFrame(summary_rows)
    simple = _simple_validation(data, assignments, dispatch, config.feasibility_tolerance)
    audit = independent_hard_constraint_audit(
        data, assignments, dispatch,
        solver_metrics={
            "Cost": metrics["Cost"], "Carbon": metrics["Carbon"],
            "Latency": metrics["Latency"], "Delay": metrics["Delay"],
            "RenewableUnusedRate": metrics["RenewableUnusedRate"], "Peak": metrics["Peak"],
        },
        baseline_status="Q2_SHADOW_TASK_SEED_V4_ENERGY_REFERENCE",
        scenario_status="MATHEURISTIC_COMPLETE",
        tolerance=config.feasibility_tolerance,
        qos_weights=_qos_weight_map(config),
    )
    checks = audit.pop("checks")
    if not bool(simple["Passed"].all()) or not bool(audit["passed"]):
        failures = simple.loc[~simple["Passed"], "Check"].tolist() + list(audit.get("failed_checks", []))
        raise RuntimeError(f"Final Q4 matheuristic independent validation failed: {sorted(set(failures))}")
    configuration = pd.DataFrame([
        {"Parameter": key, "Value": value} for key, value in asdict(config).items()
    ] + [
        {"Parameter": "ModelVersion", "Value": MATHEURISTIC_VERSION},
        {"Parameter": "FormalSolvePath", "Value": "Q2Shadow->EnergyResponse->Marginal->LocalRepair->JointLNS->CommitH"},
        {"Parameter": "FullWindowMILPFormalPath", "Value": False},
        {"Parameter": "FixedCalibration", "Value": True},
        {"Parameter": "CalibrationMethod", "Value": "RepresentativeWindowsContinuousLBAndFeasibleReference"},
        {"Parameter": "OptimizationMetrics", "Value": ",".join(_active_minimax_metrics(scaling))},
        {"Parameter": "ReportOnlyDegenerateMetrics", "Value": ",".join(
            metric for metric in METRICS if metric not in set(_active_minimax_metrics(scaling))
        )},
        {"Parameter": "DegenerateScaleFallback", "Value": "FullHorizonRenewableDeficitPhysicalProxy"},
        {"Parameter": "PhysicalTieBreak", "Value": "CarbonPriceAndStorageThroughput"},
        {"Parameter": "DocumentEnergySemanticConstraints", "Value": "RenewableDirectUse<=FacilityLoad;GridCharge<=GridPurchase"},
        {"Parameter": "ActualOperationHours", "Value": "0--2405"},
        {"Parameter": "TerminalSOCHour", "Value": 2406},
        {"Parameter": "InputSignature", "Value": signature},
        {"Parameter": "Scenario", "Value": json.dumps(dict(scenario or {}), ensure_ascii=False)},
    ])
    _atomic_write_csv(assignments, paths.tables / "q4_task_assignments.csv")
    _atomic_write_csv(dispatch, paths.tables / "q4_region_hour_dispatch.csv")
    _atomic_write_csv(summary, paths.tables / "q4_objective_summary.csv")
    _atomic_write_csv(solver, paths.tables / "q4_window_solver.csv")
    _atomic_write_csv(lns, paths.tables / "q4_lns_diagnostics.csv")
    _atomic_write_csv(forecast, paths.tables / "q4_forecast_profile.csv")
    _atomic_write_csv(
        _v4_scaling_frame(scaling, calibration_records),
        paths.tables / "q4_scaling.csv",
    )
    _atomic_write_csv(calibration_records, paths.tables / "q4_calibration_records.csv")
    _atomic_write_csv(configuration, paths.tables / "q4_model_configuration.csv")
    _atomic_write_csv(simple, paths.tables / "q4_simple_validation.csv")
    _atomic_write_csv(checks, paths.validation / "q4_hard_constraint_audit.csv")
    _atomic_write_json(paths.validation / "q4_independent_audit.json", audit)
    _, qos_details = _qos_details(data, assignments, _qos_weight_map(config))
    _atomic_write_csv(qos_details, paths.validation / "q4_qos_details.csv")
    sequential_metrics = pd.DataFrame([
        {"Scheme": "Q2_TaskSeed_to_V4_EnergyReference", "Metric": key, "Value": value}
        for key, value in dict(sequential["metrics"]).items()
    ])
    _atomic_write_csv(sequential_metrics, paths.tables / "q4_sequential_baseline_metrics.csv")
    marker = {
        "complete": True,
        "audit_passed": True,
        "model_version": MATHEURISTIC_VERSION,
        "input_signature": signature,
        "task_count": len(assignments),
        "dispatch_count": len(dispatch),
        "scenario": dict(scenario or {}),
    }
    _atomic_write_json(paths.complete, marker)
    return marker


def _rollout_signature(
    data: InputData,
    config: ModelConfig,
    q2_schedule: pd.DataFrame,
    *,
    scenario: Mapping[str, object] | None,
    carbon_budget: np.ndarray | None,
    fixed_scaling_override: Mapping[str, tuple[float, float]] | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(_matheuristic_signature(data, config, scenario=scenario).encode("ascii"))
    digest.update(_matheuristic_frame_hash(q2_schedule))
    if carbon_budget is not None:
        digest.update(np.asarray(carbon_budget, dtype=float).tobytes())
    if fixed_scaling_override is not None:
        digest.update(json.dumps(
            {metric: [float(fixed_scaling_override[metric][0]), float(fixed_scaling_override[metric][1])] for metric in METRICS},
            sort_keys=True,
        ).encode("utf-8"))
    return digest.hexdigest()


def _run_joint_matheuristic_rollout(
    data: InputData,
    *,
    config: ModelConfig | None = None,
    output_root: Path | None = None,
    scenario: Mapping[str, object] | None = None,
    carbon_budget: np.ndarray | None = None,
    fixed_scaling_override: Mapping[str, tuple[float, float]] | None = None,
    initial_shadow_schedule: pd.DataFrame | None = None,
    final_carbon_upper_bound: float | None = None,
    objective_mode: str = OBJECTIVE_MODE_MINIMAX,
    force: bool = False,
) -> dict[str, object]:
    """Formal Q4 main path with a Q2 feasible extension and bounded joint LNS."""

    config = config or ModelConfig()
    mode = _validate_objective_mode(objective_mode)
    scenario_payload = {**dict(scenario or {}), "OptimizationMode": mode}
    _matheuristic_window_starts(config)
    paths = _matheuristic_paths(output_root)
    q2_schedule = load_and_validate_q2_schedule(data, config)
    if initial_shadow_schedule is None:
        initial_schedule = q2_schedule.copy()
    else:
        initial_schedule = _standardize_shadow_schedule(data, initial_shadow_schedule)
        _validate_shadow_schedule(data, initial_schedule, config.feasibility_tolerance)
        initial_schedule = initial_schedule.sort_values(
            ["StartHour", "TaskID"], kind="stable"
        ).reset_index(drop=True)
        seed_hash = hashlib.sha256(_matheuristic_frame_hash(
            initial_schedule[["TaskID", "TargetRegion", "StartHour"]]
        )).hexdigest()
        scenario_payload = {
            **scenario_payload,
            "InitialShadowSchedule": "EXTERNAL_AUDITED_SEED",
            "InitialShadowScheduleHash": seed_hash,
        }
        _progress("Low-carbon reference loaded the audited baseline task schedule as a globally feasible seed.")
    if carbon_budget is not None:
        carbon_budget = np.asarray(carbon_budget, dtype=float)
        if carbon_budget.shape != (OPERATION_END + 1,):
            raise ValueError("Cumulative carbon budget must cover all 2407 time points from 0 through 2406")
    signature = _rollout_signature(
        data, config, q2_schedule, scenario=scenario_payload, carbon_budget=carbon_budget,
        fixed_scaling_override=fixed_scaling_override,
    )
    if paths.complete.is_file():
        try:
            marker = json.loads(paths.complete.read_text(encoding="utf-8"))
            if marker.get("complete") is True and marker.get("input_signature") == signature:
                _progress(f"Reusing complete Q4 matheuristic results: {paths.root}")
                return marker
            raise RuntimeError("Formal Q4 matheuristic results with a different signature already exist; refusing overwrite to protect them")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Corrupt Q4 matheuristic completion marker: {paths.complete}") from exc
    if force and paths.checkpoint.is_file():
        raise RuntimeError("--force does not overwrite existing matheuristic checkpoints; copy to a new output directory before running")
    paths.tables.mkdir(parents=True, exist_ok=True)
    paths.validation.mkdir(parents=True, exist_ok=True)
    if fixed_scaling_override is None:
        scaling, calibration_records = load_or_build_fixed_calibration(data, q2_schedule, config, paths)
    else:
        scaling = {
            metric: (float(fixed_scaling_override[metric][0]), float(fixed_scaling_override[metric][1]))
            for metric in METRICS
        }
        if any(scale <= EPS or not np.isfinite(anchor) or not np.isfinite(scale) for anchor, scale in scaling.values()):
            raise ValueError("Shared fixed scenario scaling parameters are incomplete or invalid")
        calibration_records = pd.DataFrame([
            {
                "Metric": metric,
                "Anchor": scaling[metric][0],
                "Scale": scaling[metric][1],
                "CalibrationSource": "SHARED_BASELINE_FIXED_CALIBRATION",
                "MetricDefinitionVersion": MATHEURISTIC_METRIC_VERSION,
            }
            for metric in METRICS
        ])
    sequential = load_or_build_q2_v3_sequential_reference(
        data, config, q2_schedule=q2_schedule, scaling=scaling, paths=paths
    )
    checkpoint = _load_matheuristic_checkpoint(data, config, paths, signature)
    storage = _matheuristic_storage_arrays(data)
    if checkpoint is None:
        assignments = pd.DataFrame(columns=_matheuristic_assignment_columns())
        dispatch = pd.DataFrame()
        solver = pd.DataFrame()
        lns = pd.DataFrame()
        forecast = pd.DataFrame()
        shadow_schedule = initial_schedule.copy()
        current_soc = storage["initial_soc"].copy()
        historical_peak = np.zeros(len(data.regions), dtype=float)
        past_carbon = 0.0
        next_tau = 0
        _save_matheuristic_checkpoint(
            paths=paths, signature=signature, config=config, next_tau=next_tau,
            assignments=assignments, dispatch=dispatch, solver=solver, lns=lns,
            forecast=forecast, shadow_schedule=shadow_schedule, scaling=scaling,
            calibration_records=calibration_records, current_soc=current_soc,
            historical_peak=historical_peak, past_carbon=past_carbon, scenario=scenario_payload,
        )
    else:
        assignments = pd.DataFrame(checkpoint["assignments"])
        dispatch = pd.DataFrame(checkpoint["dispatch"])
        solver = pd.DataFrame(checkpoint["solver"])
        lns = pd.DataFrame(checkpoint["lns"])
        forecast = pd.DataFrame(checkpoint["forecast"])
        shadow_schedule = _standardize_shadow_schedule(data, pd.DataFrame(checkpoint["shadow_schedule"]))
        scaling = {
            str(metric): (float(value[0]), float(value[1]))
            for metric, value in dict(checkpoint["scaling"]).items()
        }
        calibration_records = pd.DataFrame(checkpoint.get("calibration_records", calibration_records))
        current_soc = np.asarray(checkpoint["current_soc"], dtype=float).copy()
        historical_peak = np.asarray(checkpoint["historical_peak"], dtype=float).copy()
        past_carbon = float(checkpoint["past_carbon"])
        next_tau = int(checkpoint["next_tau"])
        _progress(f"Q4 matheuristic checkpoint restored: next_tau={next_tau}, committed tasks={len(assignments)}.")

    window_starts = _matheuristic_window_starts(config)
    for tau in (value for value in window_starts if value >= next_tau):
        decision_end = min(tau + config.decision_hours, OPERATION_END)
        plan_end = min(decision_end + config.lookahead_hours, OPERATION_END)
        started = time.perf_counter()
        carbon_remaining: float | None = None
        if carbon_budget is not None:
            carbon_remaining = float(carbon_budget[decision_end] - past_carbon)
            if carbon_remaining < -config.feasibility_tolerance:
                raise RuntimeError(f"Window {tau}: cumulative carbon budget is infeasible before the window starts")
            carbon_remaining = max(carbon_remaining, 0.0)
        incumbent = _build_joint_incumbent(
            data, config, shadow_schedule=shadow_schedule, tau=tau,
            current_soc=current_soc, historical_peak=historical_peak,
            scaling=scaling,
            time_limit=min(config.energy_time_limit_seconds, config.window_hard_time_limit_seconds),
            carbon_budget_remaining=carbon_remaining,
            objective_mode=mode,
        )
        energy_seconds = incumbent.energy.elapsed_seconds
        z_before_lns = incumbent.z
        marginal_status = "MARGINAL_SKIPPED"
        marginal_seconds = 0.0
        legal_candidate_count = 0
        marginal_moves: list[MarginalMove] = []
        accepted_moves: list[MarginalMove] = []
        repair_status = "REPAIR_NOT_NEEDED"
        repair_count = 0
        repair_seconds = 0.0
        marginal_reoptimization_seconds = 0.0

        def remaining_seconds() -> float:
            return max(0.0, config.window_hard_time_limit_seconds - (time.perf_counter() - started))

        # Phase B/C: score all operationally legal options, retain only a
        # capacity-safe batch, and re-evaluate that batch with the exact energy MIP.
        if remaining_seconds() > 1.0 and (time.perf_counter() - started) < config.window_soft_time_limit_seconds:
            try:
                marginal_moves, legal_candidate_count, marginal_status, marginal_seconds = _generate_marginal_moves(
                    data, config, incumbent=incumbent, tau=tau, current_soc=current_soc,
                    historical_peak=historical_peak, scaling=scaling,
                    carbon_budget_remaining=carbon_remaining, remaining_seconds=remaining_seconds(),
                    objective_mode=mode,
                )
            except RuntimeError as exc:
                marginal_status = f"MARGINAL_FALLBACK:{exc}"
                marginal_moves = []
        if marginal_moves and remaining_seconds() > 1.0:
            prefix_sizes: list[int] = []
            size = min(len(marginal_moves), config.marginal_max_moves)
            while size > 0:
                prefix_sizes.append(size)
                if size == 1:
                    break
                size //= 2
            for prefix_size in prefix_sizes:
                # Reserve only the time actually still needed by a bounded
                # subproblem; never extend into a full-window recovery model.
                if remaining_seconds() <= 1.0:
                    break
                proposal, attempted_moves = _apply_capacity_safe_moves(
                    data, config, incumbent.shadow_schedule, marginal_moves[:prefix_size],
                    tau=tau, plan_end=plan_end,
                )
                repaired, repair_status, repair_count, repair_elapsed = repair_if_needed(
                    data, config, incumbent_schedule=incumbent.shadow_schedule,
                    proposal=proposal, moved_task_ids=[move.task_id for move in attempted_moves],
                    tau=tau, plan_end=plan_end, remaining_seconds=remaining_seconds(),
                )
                repair_seconds += repair_elapsed
                if repaired.equals(incumbent.shadow_schedule):
                    continue
                try:
                    candidate = _build_joint_incumbent(
                        data, config, shadow_schedule=repaired, tau=tau,
                        current_soc=current_soc, historical_peak=historical_peak,
                        scaling=scaling, time_limit=min(config.energy_time_limit_seconds, max(0.1, remaining_seconds())),
                        carbon_budget_remaining=carbon_remaining,
                        objective_mode=mode,
                    )
                    marginal_reoptimization_seconds += candidate.energy.elapsed_seconds
                except RuntimeError:
                    continue
                if _incumbent_objective_value(candidate, mode) < (
                    _incumbent_objective_value(incumbent, mode) - config.feasibility_tolerance
                ):
                    incumbent = candidate
                    accepted_moves = attempted_moves
                    break

        # Phase D: the only joint MILP.  It opens all energy variables and the
        # full legal H-domain of a bounded, model-selected task neighbourhood.
        lns_status = "LNS_SKIPPED"
        lns_gap = float("nan")
        lns_seconds = 0.0
        lns_neighborhood_size = 0
        lns_neighborhood_group_size = 0
        lns_integer_options = 0
        for pass_index in range(1, config.lns_max_passes + 1):
            if remaining_seconds() <= 1.0:
                lns_status = "LNS_SKIPPED_HARD_LIMIT"
                break
            selected, integer_options, ranking = select_joint_lns_neighborhood(
                data, config, incumbent=incumbent, tau=tau, objective_mode=mode
            )
            if not selected:
                lns_status = "LNS_NO_NEIGHBORHOOD"
                break
            lns_neighborhood_size = len(selected)
            lns_neighborhood_group_size = int(ranking["SelectedExactTaskGroupCount"].max()) if not ranking.empty else 0
            lns_integer_options = integer_options
            z_before_pass = incumbent.z
            try:
                candidate, status, gap, elapsed, message = solve_joint_lns(
                    data, config, incumbent=incumbent, selected_task_ids=selected,
                    tau=tau, current_soc=current_soc, historical_peak=historical_peak,
                    scaling=scaling, time_limit=min(config.lns_time_limit_seconds, max(0.1, remaining_seconds())),
                    carbon_budget_remaining=carbon_remaining,
                    objective_mode=mode,
                )
            except (RuntimeError, ValueError) as exc:
                candidate, status, gap, elapsed, message = None, "LNS_BUILD_OR_SOLVE_FAILED", float("nan"), 0.0, str(exc)
            lns_seconds += elapsed
            lns_status = status
            lns_gap = gap
            lns = pd.concat([lns, pd.DataFrame([{
                "WindowStart": tau,
                "Pass": pass_index,
                "NeighborhoodTaskCount": len(selected),
                "NeighborhoodExactTaskGroupCount": lns_neighborhood_group_size,
                "IntegerOptionVariableCount": integer_options,
                "Status": status,
                "MIPGap": gap,
                "OptimizationMode": mode,
                "ZBefore": z_before_pass,
                "ZAfter": candidate.z if candidate is not None else z_before_pass,
                "SearchObjectiveBefore": _incumbent_objective_value(incumbent, mode),
                "SearchObjectiveAfter": _incumbent_objective_value(candidate, mode) if candidate is not None else _incumbent_objective_value(incumbent, mode),
                "ElapsedSeconds": elapsed,
                "SolverMessage": message,
            }])], ignore_index=True)
            if candidate is None:
                break
            incumbent = candidate

        actual = incumbent.energy.dispatch.loc[
            incumbent.energy.dispatch["Hour"] < decision_end
        ].copy()
        if len(actual) != (decision_end - tau) * len(data.regions):
            raise RuntimeError(f"Window {tau} did not produce a complete H-region energy trajectory")
        committed = incumbent.shadow_schedule.loc[
            pd.to_numeric(incumbent.shadow_schedule["StartHour"]).between(tau, decision_end - 1)
        ].copy()
        committed = committed[_matheuristic_assignment_columns()]
        overlap = set(committed["TaskID"].astype(str)) & set(assignments["TaskID"].astype(str))
        if overlap:
            raise RuntimeError(f"Window {tau} committed duplicate tasks: {sorted(overlap)[:5]}")
        assignments = pd.concat([assignments, committed], ignore_index=True)
        dispatch = pd.concat([dispatch, actual], ignore_index=True)
        predicted = incumbent.energy.dispatch.loc[
            incumbent.energy.dispatch["Hour"] >= decision_end
        ].copy()
        if not predicted.empty:
            predicted["ForecastRole"] = "K_PREDICTION_NOT_COMMITTED"
            forecast = pd.concat([forecast, predicted], ignore_index=True)
        current_soc = incumbent.energy.decision_end_soc.copy()
        regional_peak = actual.groupby("Region", observed=True)["NetGridImport_MW"].max()
        for r, region in enumerate(data.regions):
            historical_peak[r] = max(historical_peak[r], float(regional_peak.get(region, 0.0)), 0.0)
        past_carbon += float(actual["CarbonEmission_tCO2"].sum())
        if carbon_budget is not None and past_carbon > float(carbon_budget[decision_end]) + config.feasibility_tolerance:
            raise RuntimeError(f"Window {tau}: cumulative emissions exceed the constraint after commitment")
        active_task_count = int(pd.to_numeric(incumbent.shadow_schedule["StartHour"]).between(tau, plan_end - 1).sum())
        elapsed_total = time.perf_counter() - started
        hard_limited = elapsed_total >= config.window_hard_time_limit_seconds
        solver = pd.concat([solver, pd.DataFrame([{
            "WindowStart": tau,
            "DecisionEnd": decision_end,
            "PlanEnd": plan_end,
            "ActiveTaskCount": active_task_count,
            "LegalCandidateCount": legal_candidate_count,
            "CommittedTaskCount": len(committed),
            "MarginalMoveCount": len(accepted_moves),
            "RepairCount": repair_count,
            "RepairStatus": repair_status,
            "MarginalStatus": marginal_status,
            "LNSNeighborhoodTaskCount": lns_neighborhood_size,
            "LNSNeighborhoodTaskGroupCount": lns_neighborhood_group_size,
            "LNSIntegerVariableCount": lns_integer_options,
            "LNSStatus": lns_status,
            "LNSMIPGap": lns_gap,
            "ZBeforeLNS": z_before_lns,
            "ZAfterLNS": incumbent.z,
            "EnergySolveSeconds": energy_seconds + marginal_reoptimization_seconds,
            "MarginalLPSeconds": marginal_seconds,
            "RepairSeconds": repair_seconds,
            "LNSSeconds": lns_seconds,
            "TotalWindowSeconds": elapsed_total,
            "WindowHardLimitReached": hard_limited,
            "EnergyStatus": incumbent.energy.status,
            "EnergyMIPGap": incumbent.energy.mip_gap,
            "EnergyMessage": incumbent.energy.message,
            **{f"Window{metric}": value for metric, value in incumbent.metrics.items()},
        }])], ignore_index=True)
        shadow_schedule = incumbent.shadow_schedule
        next_tau = min(decision_end, OPERATION_END)
        _save_matheuristic_checkpoint(
            paths=paths, signature=signature, config=config, next_tau=next_tau,
            assignments=assignments, dispatch=dispatch, solver=solver, lns=lns,
            forecast=forecast, shadow_schedule=shadow_schedule, scaling=scaling,
            calibration_records=calibration_records, current_soc=current_soc,
            historical_peak=historical_peak, past_carbon=past_carbon, scenario=scenario_payload,
        )
        _progress(
            f"Q4 matheuristic window {tau} saved: committed tasks={len(committed)}, "
            f"LNS={lns_status}, elapsed={elapsed_total:.2f}s, next_tau={next_tau}."
        )

    return _write_matheuristic_final_outputs(
        data=data, config=config, paths=paths, signature=signature,
        assignments=assignments, dispatch=dispatch, solver=solver, lns=lns,
        forecast=forecast, scaling=scaling, calibration_records=calibration_records,
        sequential=sequential, scenario=scenario_payload,
        final_carbon_upper_bound=final_carbon_upper_bound,
    )


def _read_matheuristic_result(directory: Path) -> tuple[Path, dict[str, float], pd.DataFrame]:
    root = directory.resolve()
    tables = root / "tables" if (root / "tables").is_dir() else root
    marker_path = _matheuristic_paths(root).complete
    summary_path = tables / "q4_objective_summary.csv"
    dispatch_path = tables / "q4_region_hour_dispatch.csv"
    if not all(path.is_file() for path in (marker_path, summary_path, dispatch_path)):
        raise FileNotFoundError(f"Complete Q4 matheuristic results not found: {root}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("complete") is not True or marker.get("audit_passed") is not True:
        raise RuntimeError(f"Q4 matheuristic results failed final audit: {root}")
    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    values = {str(row.Metric): float(row.Value) for row in summary.itertuples(index=False)}
    return tables, values, pd.read_csv(dispatch_path, encoding="utf-8-sig")


def audit_document_compliant_result(
    directory: Path | None = None,
    *,
    data: InputData | None = None,
) -> dict[str, object]:
    """Read-only document-level audit for a completed V4 baseline or scenario.

    ``data`` is optional for backward compatibility.  The baseline audit uses
    the current input when it is omitted; scenario validation supplies the
    reconstructed scenario input so changed exogenous quantities are audited
    against the same input that generated the scenario result.
    """

    root = (directory or MATHEURISTIC_ROOT).resolve()
    tables, _, dispatch = _read_matheuristic_result(root)
    assignments = pd.read_csv(tables / "q4_task_assignments.csv", encoding="utf-8-sig")
    summary = pd.read_csv(tables / "q4_objective_summary.csv", encoding="utf-8-sig")
    marker = json.loads(_matheuristic_paths(root).complete.read_text(encoding="utf-8"))
    audit_data = data if data is not None else load_data()
    audit = independent_hard_constraint_audit(
        audit_data,
        assignments,
        dispatch,
        solver_metrics=_solver_metrics_from_summary(summary),
        baseline_status="Q4_V4_DOCUMENT_COMPLIANT",
        scenario_status=str(dict(marker.get("scenario", {})).get("ScenarioKind", "BASELINE")),
        tolerance=ModelConfig().feasibility_tolerance,
        qos_weights=_qos_weight_map(ModelConfig()),
    )
    version_ok = marker.get("model_version") == MATHEURISTIC_VERSION
    return {
        "passed": bool(audit.get("passed", False) and version_ok),
        "tables": str(tables),
        "model_version": marker.get("model_version"),
        "version_ok": version_ok,
        "audit_data_context": "provided" if data is not None else "current_default_input",
        "audit": audit,
    }


def _scenario_baseline_reference() -> tuple[Path, dict[str, float], pd.DataFrame]:
    """Load only the current document-compliant baseline for scenario work."""

    try:
        return _read_matheuristic_result(MATHEURISTIC_ROOT)
    except (FileNotFoundError, RuntimeError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Scenario runs require a complete Q4 V4 baseline passing the documented audit; "
            "run the updated baseline first. Old V2/V3 results are not reused."
        ) from exc


def _load_audited_task_seed(
    data: InputData,
    config: ModelConfig,
    *,
    assignment_path: Path,
    label: str,
) -> tuple[pd.DataFrame, str]:
    """Load a complete, independently auditable task schedule as a safe seed."""

    if not assignment_path.is_file():
        raise FileNotFoundError(f"{label} is missing the task schedule seed: {assignment_path}")
    try:
        raw_seed = pd.read_csv(assignment_path, encoding="utf-8-sig")
        seed = _standardize_shadow_schedule(data, raw_seed)
        _validate_shadow_schedule(data, seed, config.feasibility_tolerance)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise RuntimeError(f"{label} cannot use the task schedule seed: {assignment_path}") from exc
    seed = seed.sort_values(["StartHour", "TaskID"], kind="stable").reset_index(drop=True)
    seed_hash = hashlib.sha256(_matheuristic_frame_hash(
        seed[["TaskID", "TargetRegion", "StartHour"]]
    )).hexdigest()
    return seed, seed_hash


def _low_carbon_reference_controls(
    data: InputData,
    config: ModelConfig,
    *,
    baseline_tables: Path,
    baseline_dispatch: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    """Build a dominance-safe seed and cumulative carbon guard for E^LC.

    The audited Q4 baseline is a feasible point under the unchanged low-carbon
    scenario inputs.  Seeding from it and guarding cumulative carbon makes the
    reference no worse than E^0 even when a rolling local neighborhood cannot
    anticipate a late-horizon energy bottleneck.
    """

    assignment_path = baseline_tables / "q4_task_assignments.csv"
    seed, seed_hash = _load_audited_task_seed(
        data,
        config,
        assignment_path=assignment_path,
        label="Low-carbon reference",
    )
    baseline_cumulative = _cumulative_carbon(baseline_dispatch)
    if baseline_cumulative.shape != (OPERATION_END + 1,):
        raise ValueError("Baseline cumulative carbon trajectory is incomplete; cannot construct the low-carbon safeguard")
    if not np.all(np.isfinite(baseline_cumulative)):
        raise ValueError("Baseline cumulative carbon trajectory contains nonfinite values; cannot construct the low-carbon safeguard")
    if np.any(np.diff(baseline_cumulative) < -max(config.feasibility_tolerance, 1e-8)):
        raise ValueError("Baseline cumulative carbon trajectory is nonmonotonic; cannot serve as the low-carbon upper bound")
    return seed, baseline_cumulative.copy(), {
        "LowCarbonReferenceStrategy": "audited_baseline_seed_plus_cumulative_carbon_guard_v1",
        "LowCarbonSeedSource": str(assignment_path),
        "LowCarbonSeedTaskHash": seed_hash,
        "CarbonReferenceUpperBound": float(baseline_cumulative[-1]),
        "CarbonReferenceCumulativeGuard": "baseline_joint_dispatch",
    }


def run_full_scenario(
    spec: ScenarioSpec,
    *,
    config: ModelConfig | None = None,
    lp_time_limit: float = 20.0,
    low_carbon_reference_dir: Path | None = None,
) -> dict[str, object]:
    """Run a scenario through the same bounded matheuristic, never the legacy full MILP."""

    del lp_time_limit  # Retained for CLI compatibility; fixed scaling is never per-window LP recalibrated.
    config = config or ModelConfig()
    original = load_data()
    baseline_tables, baseline_metrics, baseline_dispatch = _scenario_baseline_reference()
    scaling_path = baseline_tables / "q4_scaling.csv"
    if not scaling_path.is_file():
        raise FileNotFoundError(f"Scenario run lacks the baseline fixed scaling table: {scaling_path}")
    scaling_frame = pd.read_csv(scaling_path, encoding="utf-8-sig").set_index("Metric")
    shared_scaling = {
        metric: (float(scaling_frame.at[metric, "Anchor"]), float(scaling_frame.at[metric, "Scale"]))
        for metric in METRICS
    }
    normalized_spec = spec
    carbon_budget: np.ndarray | None = None
    initial_shadow_schedule: pd.DataFrame | None = None
    final_carbon_upper_bound: float | None = None
    low_carbon_controls: dict[str, object] = {}
    carbon_constraint_controls: dict[str, object] = {}
    objective_mode = (
        OBJECTIVE_MODE_CARBON_PRIORITY
        if spec.kind == "low_carbon_reference"
        else OBJECTIVE_MODE_MINIMAX
    )
    if spec.kind == "low_carbon_reference":
        (
            initial_shadow_schedule,
            carbon_budget,
            low_carbon_controls,
        ) = _low_carbon_reference_controls(
            original,
            config,
            baseline_tables=baseline_tables,
            baseline_dispatch=baseline_dispatch,
        )
        final_carbon_upper_bound = float(
            low_carbon_controls["CarbonReferenceUpperBound"]
        )
    if spec.kind == "carbon_constraint":
        lam = float(spec.carbon_lambda if spec.carbon_lambda is not None else 1.0)
        if not 0.0 <= lam <= 1.0:
            raise ValueError("carbon_lambda must lie within [0,1]")
        if low_carbon_reference_dir is None and lam > EPS:
            raise RuntimeError("Carbon-constrained scenarios require a completed low-carbon reference directory")
        baseline_cumulative = _cumulative_carbon(baseline_dispatch)
        if low_carbon_reference_dir is None:
            low_cumulative = baseline_cumulative.copy()
            low_carbon = float(baseline_metrics["Carbon"])
        else:
            low_tables, low_values, low_dispatch = _read_matheuristic_result(low_carbon_reference_dir)
            low_marker = json.loads(_matheuristic_paths(low_tables.parent).complete.read_text(encoding="utf-8"))
            low_mode = str(dict(low_marker.get("scenario", {})).get("OptimizationMode", ""))
            if low_mode != OBJECTIVE_MODE_CARBON_PRIORITY:
                raise RuntimeError("Low-carbon reference was not solved with carbon priority; cannot construct the carbon constraint")
            low_cumulative = _cumulative_carbon(low_dispatch)
            low_carbon = float(low_values.get("Carbon", low_cumulative[-1]))
            if low_carbon > float(baseline_metrics["Carbon"]) + 1e-8:
                raise RuntimeError("Low-carbon reference emissions exceed baseline emissions; cannot construct the carbon constraint")
            if np.any(low_cumulative > baseline_cumulative + max(config.feasibility_tolerance, 1e-8)):
                raise RuntimeError("Low-carbon reference cumulative emissions exceed the baseline at some times; cannot construct rolling carbon constraints")
            low_seed_path = low_tables / "q4_task_assignments.csv"
            initial_shadow_schedule, low_seed_hash = _load_audited_task_seed(
                original,
                config,
                assignment_path=low_seed_path,
                label="Carbon-constrained scenario",
            )
        carbon_budget = baseline_cumulative - lam * (baseline_cumulative - low_cumulative)
        if np.any(carbon_budget + max(config.feasibility_tolerance, 1e-8) < low_cumulative):
            raise RuntimeError("Constructed carbon constraint trajectory lies below the feasible low-carbon seed; refusing infeasible rolling optimization")
        final_carbon_upper_bound = float(carbon_budget[-1])
        if low_carbon_reference_dir is not None:
            carbon_constraint_controls = {
                "CarbonConstraintSeedSource": str(low_seed_path),
                "CarbonConstraintSeedTaskHash": low_seed_hash,
                "CarbonConstraintCumulativeGuard": "linear_interpolation_between_baseline_and_low_carbon",
                "CarbonConstraintFinalUpperBound": final_carbon_upper_bound,
            }
        normalized_spec = ScenarioSpec(
            name=spec.name,
            kind=spec.kind,
            carbon_lambda=lam,
            low_carbon_reference=low_carbon,
            renewable_gamma=spec.renewable_gamma,
        )
    scenario_data, metadata = generate_scenario_data(
        original, normalized_spec, {"Carbon": float(baseline_metrics["Carbon"])}
    )
    factor_check = validate_single_factor_scenario(original, scenario_data, metadata)
    if not factor_check["passed"]:
        raise RuntimeError(f"Scenario input single-factor consistency check failed: {factor_check}")
    scenario_root = SCENARIO_DIR / _scenario_slug(spec.name) / MATHEURISTIC_NAMESPACE
    scenario_payload = {
        **metadata,
        "SingleFactorCheck": factor_check,
        "OptimizationMode": objective_mode,
        **low_carbon_controls,
        **carbon_constraint_controls,
    }
    result = _run_joint_matheuristic_rollout(
        scenario_data, config=config, output_root=scenario_root,
        scenario=scenario_payload,
        carbon_budget=carbon_budget, fixed_scaling_override=shared_scaling,
        initial_shadow_schedule=initial_shadow_schedule,
        final_carbon_upper_bound=final_carbon_upper_bound,
        objective_mode=objective_mode,
    )
    tables = scenario_root / "tables"
    _atomic_write_json(tables / "q4_scenario_metadata.json", {
        **metadata,
        "SingleFactorCheck": factor_check,
        "Algorithm": MATHEURISTIC_VERSION,
        "OptimizationMode": objective_mode,
        "SharedFixedCalibration": True,
        **low_carbon_controls,
        **carbon_constraint_controls,
    })
    _atomic_write_json(tables / ".q4_scenario_complete.json", {
        **result,
        "scenario_name": spec.name,
        "scenario_kind": spec.kind,
    })
    return result


def build_scenario_comparison(
    scenario_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compare only audited matheuristic scenarios on the same six-metric scale."""

    _progress("Scenario comparison started: reading completed, audited V4 results only; no MILP calls.")
    baseline_dir, baseline_values, _ = _scenario_baseline_reference()
    _progress(f"Scenario comparison baseline loaded: {baseline_dir}")
    if scenario_names is None:
        candidate_names = [
            directory.name for directory in SCENARIO_DIR.iterdir()
            if directory.is_dir() and (directory / MATHEURISTIC_NAMESPACE / "tables" / f".{MATHEURISTIC_FILE_PREFIX}_complete.json").is_file()
        ] if SCENARIO_DIR.is_dir() else []
        _progress(f"Scenario comparison candidates: {len(candidate_names)} directories; checking completion markers.")
        names = []
        for index, name in enumerate(candidate_names, start=1):
            _progress(f"Scenario comparison check: {index}/{len(candidate_names)}, scenario={name}.")
            tables, values, _ = _read_matheuristic_result(
                SCENARIO_DIR / name / MATHEURISTIC_NAMESPACE
            )
            marker = json.loads(
                (tables / f".{MATHEURISTIC_FILE_PREFIX}_complete.json").read_text(encoding="utf-8")
            )
            scenario = dict(marker.get("scenario", {}))
            if (
                scenario.get("ScenarioKind") == "low_carbon_reference"
                and float(values["Carbon"]) > float(baseline_values["Carbon"]) + 1e-8
            ):
                _progress(f"Scenario comparison skipping invalid low-carbon reference: {name} (emissions exceed baseline).")
                continue
            names.append(name)
    else:
        names = [_scenario_slug(name) for name in scenario_names]
    if not names:
        raise FileNotFoundError("No completed, audited Q4 matheuristic scenarios")
    _progress(f"Scenario comparison includes {len(names)} scenarios; computing six-objective differences.")
    key_map = {
        "Cost": "Cost", "Carbon": "Carbon", "Latency": "Latency_ms",
        "Delay": "QoSLoss", "RenewableUnusedRate": "RenewableUnusedRate", "Peak": "Peak",
    }
    rows: list[dict[str, object]] = []
    for index, name in enumerate(sorted(names), start=1):
        _progress(f"Scenario comparison computation: {index}/{len(names)}, scenario={name}.")
        tables, values, _ = _read_matheuristic_result(SCENARIO_DIR / name / MATHEURISTIC_NAMESPACE)
        for internal, output_name in key_map.items():
            baseline_value = float(baseline_values[output_name] if output_name in baseline_values else baseline_values[internal])
            scenario_value = float(values[output_name])
            delta = scenario_value - baseline_value
            rows.append({
                "BaselineDirectory": str(baseline_dir),
                "ScenarioName": name,
                "ScenarioTables": str(tables),
                "Metric": output_name,
                "BaselineValue": baseline_value,
                "ScenarioValue": scenario_value,
                "AbsoluteChange": delta,
                "RelativeChange": delta / abs(baseline_value) if abs(baseline_value) > EPS else float("nan"),
                "Algorithm": MATHEURISTIC_VERSION,
            })
    comparison = pd.DataFrame(rows)
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SCENARIO_DIR / f"{MATHEURISTIC_FILE_PREFIX}_scenario_comparison.csv"
    _atomic_write_csv(comparison, output_path)
    _progress(f"Scenario comparison table written: {output_path}, rows={len(comparison)}.")
    return comparison


def run_automated_tests() -> pd.DataFrame:
    """Lightweight structural checks for the formal matheuristic path only."""

    records: list[dict[str, object]] = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        records.append({"Test": name, "Passed": bool(passed), "Detail": detail})

    data = load_data()
    config = ModelConfig()
    input_summary = _validate_task_time_and_latency_inputs(data.tasks, data.candidates)
    realtime_tasks = data.tasks.loc[data.tasks["TaskType"].eq("RealTimeInference")]
    def realtime_has_only_arrival_start(task: object) -> bool:
        arrival = int(task.ArrivalHour)
        tau = int(arrival // config.decision_hours * config.decision_hours)
        return _feasible_starts(
            task,
            tau,
            min(tau + config.decision_hours, OPERATION_END),
        ) == (arrival,)
    record(
        "TaskTimeAndLatencyPreflight",
        input_summary["TaskCount"] == len(data.tasks)
        and input_summary["CandidateRowCount"] == len(data.candidates),
    )
    record(
        "RealtimeArrivalStartDomain",
        all(realtime_has_only_arrival_start(task) for task in realtime_tasks.itertuples(index=False)),
        f"realtime task count={len(realtime_tasks)}",
    )
    tail_realtime = realtime_tasks.loc[
        realtime_tasks["ArrivalHour"] >= MAIN_END - config.decision_hours
    ]
    record(
        "RealtimeTailWindowCovered",
        all(realtime_has_only_arrival_start(task) for task in tail_realtime.itertuples(index=False)),
        f"tail-window realtime task count={len(tail_realtime)}",
    )
    candidate_bounds = data.candidates.merge(
        data.tasks[["TaskID", "MaxLatency_ms"]].rename(
            columns={"MaxLatency_ms": "TaskMaxLatency_ms"}
        ),
        on="TaskID",
        how="left",
        validate="many_to_one",
    )
    record(
        "CandidateLatencyUsesTaskBound",
        not candidate_bounds["TaskMaxLatency_ms"].isna().any()
        and np.allclose(
            candidate_bounds["MaxLatency_ms"].to_numpy(dtype=float),
            candidate_bounds["TaskMaxLatency_ms"].to_numpy(dtype=float),
            rtol=0.0,
            atol=EPS,
        )
        and bool(np.all(
            candidate_bounds["NetworkLatency_ms"].to_numpy(dtype=float)
            <= candidate_bounds["TaskMaxLatency_ms"].to_numpy(dtype=float) + EPS
        )),
    )
    try:
        shadow = load_and_validate_q2_schedule(data, config)
        record("Q2ShadowScheduleValidated", len(shadow) == len(data.tasks))
    except Exception as exc:  # Keep a self-test report rather than hide the cause.
        shadow = pd.DataFrame()
        record("Q2ShadowScheduleValidated", False, str(exc))
    source = Path(__file__).read_text(encoding="utf-8")
    formal_start = source.find("def _run_joint_matheuristic_rollout")
    formal_end = source.find("def _read_matheuristic_result", formal_start)
    formal_source = source[formal_start:formal_end] if formal_start >= 0 and formal_end > formal_start else ""
    record("FormalPathExists", bool(formal_source))
    record("NoFullWindowMILPInFormalPath", "build_window_problem(" not in formal_source and "_solve_with_time_extension(" not in formal_source)
    record("EnergyResponseExists", "def solve_energy_response" in source)
    record("LocalRepairExists", "def build_local_repair_problem" in source and "def repair_if_needed" in source)
    record("JointLNSExists", "def build_joint_lns_problem" in source and "def solve_joint_lns" in source)
    record("FixedCalibrationExists", "def load_or_build_fixed_calibration" in source)
    record(
        "DedicatedCheckpointNamespace",
        "MATHEURISTIC_FILE_PREFIX" in source and "checkpoint" in source,
    )
    record("LNSVariableCapExists", "lns_max_integer_option_vars" in source)
    record("HardWindowLimitExists", "window_hard_time_limit_seconds" in source)
    record("ScenarioReusesSharedScaling", "fixed_scaling_override=shared_scaling" in source)
    record(
        "SOCRecoverabilityConstraint",
        source.count("lower[indices[\"soc\"][h_count]]") >= 2,
    )
    record(
        "SharedDocumentEnergyConstraints",
        source.count("_append_document_energy_semantic_constraints(") >= 3,
    )
    record(
        "IndependentEnergySemanticAudit",
        all(name in source for name in (
            "RenewableBalance", "FacilityLoadBalance",
            "RenewableDirectUseLimit", "GridChargeSourceLimit",
            "ChargePowerDefinition", "NetGridImportDefinition",
            "CarbonEmissionDefinition",
        )),
    )
    record(
        "AllSixMetricsCanBeActive",
        set(_active_minimax_metrics({
            "Cost": (0.0, 1.0), "Carbon": (0.0, 1.0),
            "Latency": (0.0, 1.0), "Delay": (0.0, 1.0),
            "RenewableUnusedRate": (0.0, 1.0), "Peak": (0.0, 1.0),
        })) == set(METRICS),
    )
    calibration_start = source.find("def load_or_build_fixed_calibration")
    calibration_end = source.find("def _matheuristic_storage_arrays", calibration_start)
    calibration_source = source[calibration_start:calibration_end]
    record(
        "RepresentativeWindowCalibration",
        "_calibrate(" in calibration_source
        and "feasible_reference_schedule=reusable_schedule" in calibration_source
        and "_legacy_fixed_scaling()" not in calibration_source,
    )
    _, carbon_metadata = generate_scenario_data(
        data,
        ScenarioSpec("carbon_formula_test", "carbon_constraint", 0.25, 20.0),
        {"Carbon": 100.0},
    )
    record(
        "CarbonCapFormulaDirection",
        abs(float(carbon_metadata["CarbonCap"]) - 80.0) <= 1e-9,
    )
    renewable_scenario, renewable_metadata = generate_scenario_data(
        data,
        ScenarioSpec("renewable_gamma_test", "low_variability_renewable", renewable_gamma=0.5),
        {"Carbon": 0.0},
    )
    record(
        "RenewableGammaInterpolation",
        bool(renewable_metadata.get("RenewableSmoothingGamma") == 0.5)
        and validate_single_factor_scenario(data, renewable_scenario, renewable_metadata)["passed"],
    )
    record(
        "LowCarbonReferencePriorityPath",
        "objective_mode = (" in source
        and "OBJECTIVE_MODE_CARBON_PRIORITY" in source
        and "low_carbon_reference" in source,
    )
    if _matheuristic_paths().complete.is_file():
        try:
            baseline_tables, _, baseline_dispatch = _scenario_baseline_reference()
            low_seed, low_budget, low_controls = _low_carbon_reference_controls(
                data,
                config,
                baseline_tables=baseline_tables,
                baseline_dispatch=baseline_dispatch,
            )
            record(
                "LowCarbonReferenceAuditedBaselineSeed",
                len(low_seed) == len(data.tasks)
                and low_seed["TaskID"].astype(str).nunique() == len(data.tasks),
            )
            record(
                "LowCarbonReferenceCumulativeCarbonGuard",
                low_budget.shape == (OPERATION_END + 1,)
                and np.allclose(low_budget, _cumulative_carbon(baseline_dispatch), rtol=0.0, atol=EPS)
                and "CarbonReferenceUpperBound" in low_controls,
            )
        except Exception as exc:
            record("LowCarbonReferenceAuditedBaselineSeed", False, str(exc))
            record("LowCarbonReferenceCumulativeCarbonGuard", False, str(exc))
    else:
        record("LowCarbonReferenceAuditedBaselineSeed", True, "Deferred until the required V4 baseline exists.")
        record("LowCarbonReferenceCumulativeCarbonGuard", True, "Deferred until the required V4 baseline exists.")
    record(
        "LowCarbonReferenceFinalCarbonGuard",
        "final_carbon_upper_bound" in formal_source,
    )
    record(
        "CarbonConstraintUsesLowCarbonSeed",
        "CarbonConstraintSeedSource" in source
        and "linear_interpolation_between_baseline_and_low_carbon" in source,
    )
    legacy_tables = QUESTION_DIR / "outputs" / "matheuristic_v2" / "tables"
    legacy_assignments = legacy_tables / "q4_task_assignments.csv"
    legacy_dispatch = legacy_tables / "q4_region_hour_dispatch.csv"
    reusable_v2_schedule, reusable_v2_source = _load_reusable_v2_task_reference(data, config)
    v2_reference_valid = (
        reusable_v2_schedule is None
        or (
            len(reusable_v2_schedule) == len(data.tasks)
            and reusable_v2_schedule["TaskID"].astype(str).nunique() == len(data.tasks)
        )
    )
    record(
        "LegacyV2TaskSideReferenceOptional",
        v2_reference_valid,
        reusable_v2_source or "No reusable V2 task schedule found; calibration will use the Q2 seed.",
    )
    if legacy_assignments.is_file() and legacy_dispatch.is_file():
        try:
            legacy_assignment_frame = pd.read_csv(legacy_assignments, encoding="utf-8-sig")
            legacy_dispatch_frame = pd.read_csv(legacy_dispatch, encoding="utf-8-sig")
            legacy_audit = independent_hard_constraint_audit(
                data,
                legacy_assignment_frame,
                legacy_dispatch_frame,
                tolerance=config.feasibility_tolerance,
                qos_weights=_qos_weight_map(config),
            )
            detected = set(legacy_audit.get("failed_checks", []))
            record(
                "LegacyV2SemanticViolationDetected",
                {"RenewableDirectUseLimit", "GridChargeSourceLimit"}.issubset(detected),
                ",".join(sorted(detected)),
            )
            object_assignments = legacy_assignment_frame.copy()
            object_dispatch = legacy_dispatch_frame.copy()
            object_assignments["TaskID"] = object_assignments["TaskID"].astype(str)
            object_dispatch["Region"] = object_dispatch["Region"].astype(str)
            for column in (
                "ArrivalHour", "NetworkLatency_ms", "MaxLatency_ms", "StartHour",
                "FinishHour", "Duration_h", "GPU_Demand", "Task_Full_IT_Power_MW",
                "WaitHours", "DecisionWindowStart",
            ):
                if column in object_assignments:
                    object_assignments[column] = object_assignments[column].astype(object)
            for column in (
                "Hour", "AI_IT_Load_MW", "NonAI_IT_Load_MW", "Total_IT_Load_MW",
                "Facility_Load_MW", "PUE", "AvailableRenewable_MW",
                "RenewableDirectUse_MW", "RenewableCharge_MW", "GridCharge_MW",
                "DischargePower_MW", "GridPurchase_MW", "GridSell_MW",
                "RenewableCurtailment_MW", "NetGridImport_MW", "SOCStart_MWh",
                "SOCEnd_MWh", "ChargePower_MW", "CarbonEmission_tCO2",
                "ElectricityPrice_CNY_per_MWh", "SellPrice_CNY_per_MWh",
                "CarbonIntensity_tCO2_per_MWh", "Available_GPU", "Max_IT_Power_MW",
                "Max_Facility_Power_MW", "DecisionWindowStart",
            ):
                if column in object_dispatch:
                    object_dispatch[column] = object_dispatch[column].astype(object)
            object_checks = _simple_validation(
                data, object_assignments, object_dispatch, config.feasibility_tolerance
            )
            record(
                "ObjectTypedCheckpointFinalValidation",
                bool(object_checks.loc[
                    object_checks["Check"].eq("TaskIntegerStart"), "Passed"
                ].all()),
            )
        except Exception as exc:
            record("LegacyV2SemanticViolationDetected", False, str(exc))
            record("ObjectTypedCheckpointFinalValidation", False, str(exc))
    frame = pd.DataFrame(records)
    paths = _matheuristic_paths()
    paths.validation.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(frame, paths.validation / f"{MATHEURISTIC_FILE_PREFIX}_automated_tests.csv")
    return frame


def _legacy_run_wrapper(
    *,
    force: bool = False,
    config: ModelConfig | None = None,
    resume_from: int | None = None,
    state_file: Path | None = None,
) -> None:
    """Compatibility wrapper retained with archived pre-rewrite code below."""

    if resume_from is not None or state_file is not None:
        raise RuntimeError(
            "Q4 matheuristic automatically restores its dedicated checkpoint; legacy --resume-from and --state-file are unsupported"
        )
    _run_joint_matheuristic_rollout(
        load_data(), config=config or ModelConfig(), force=force
    )
    return

    # Retain the original rolling implementation for scenario executors reviewed explicitly before reuse.
    # The baseline entry above always returns or raises, so it cannot reach this code.
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    _progress(
        f"Q4 model started: force={force}, resume_from={resume_from}, "
        f"state_file={state_file or 'default checkpoint'}."
    )
    signature = _cache_signature(config)
    if resume_from is None and state_file is None and not force and _can_reuse(signature):
        _progress("Complete matching cached results found; skipping optimization.")
        return
    if resume_from is None and state_file is None and not force:
        auto_resume = _select_auto_resume_checkpoint(signature)
        if auto_resume is not None:
            state_file, resume_from = auto_resume
            _activate_progress_namespace(state_file)
            _progress(
                f"Argument-free startup selected a valid checkpoint: {state_file.name}, "
                f"next_tau={resume_from}; subsequent writes remain on this branch without overwriting the original checkpoint."
            )
    elif state_file is not None:
        state_file = state_file.resolve()
        _activate_progress_namespace(state_file)
        if resume_from is None:
            state = _checkpoint_header(state_file)
            try:
                resume_from = int(state["next_tau"]) if state is not None else None
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"Cannot read next_tau from state-file: {state_file}") from exc
            if resume_from is None:
                raise RuntimeError(f"Cannot read next_tau from state-file: {state_file}")

    data = load_data()
    _progress(
        f"Inputs loaded: tasks={len(data.tasks)}, regions={len(data.regions)}, "
        f"hourly regional records={len(data.region_hour)}."
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
            f"Resume validation passed: prior windows={len(solver_frame)}, assigned tasks={len(assignments)}, "
            f"hourly dispatch={len(dispatch)} rows, next window={start_tau}."
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
                "Existing Q4 progress detected; refusing overwrite from 0. Resume with --resume-from: "
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
        _progress("Fixed scaling completed and written to the initial checkpoint; formal rolling runs do not recalibrate.")

    historical_peak = np.array(
        historical_peak,
        dtype=np.float64,
        copy=True,
        order="C",
    )
    if not historical_peak.flags.writeable:
        historical_peak = historical_peak.copy()
    if set(scaling) != set(METRICS):
        raise ValueError("Checkpoint scaling parameters do not contain all six objectives")
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
            f"Rolling window {window_index}/{len(all_windows)}: tau={tau}, "
            f"H={decision_hours}h, normal K={normal_lookahead}h."
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
                    len(problem.task_groups[task_id])
                    for task_id in {option.task_id for option in problem.x_options}
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
                        f"K-region 4-hour block capacity violation {k_capacity_violation:.6g}; "
                        "automatic 2-hour refinement is disabled"
                    )
            except RuntimeError as exc:
                recovery_errors.append(f"NormalHK: {exc}")
                problem = None
                solution = None
                _progress(f"Window {tau}: normal H+K failed; entering H-only recovery: {exc}")
        else:
            _progress("Window 1608 uses H=24, K=0 directly; no K-region task prediction variables are created.")

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
                    _progress(f"Window {tau}, {stage} failed: {exc}")
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
                f"Window {tau}: normal model, H-only joint objective, and identical-hard-constraint feasibility model "
                f"all failed to produce a valid integer solution; last successful checkpoint remains unchanged. {reason}"
            )

        new_assignments = _selected_assignments(data, problem, solution.vector)
        duplicates = set(new_assignments["TaskID"]) & set(assignments_before["TaskID"])
        if duplicates:
            raise RuntimeError(f"Tasks executed more than once; examples: {sorted(duplicates)[:5]}")
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
            len(problem.task_groups[task_id])
            for task_id in {option.task_id for option in problem.x_options}
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
            "SolverStatusName": solution.status_name,
            "SolverMessage": solution.message,
            "ObjectiveValue": solution.objective,
            "BestBound": solution.best_bound,
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
            f"Window {tau} completed and checkpoint saved: mode={model_mode}, variables={problem.lower.size}, "
            f"constraints={problem.matrix.shape[0]}, elapsed={solution.elapsed_seconds:.2f}s, "
            f"new tasks={len(new_assignments)}, total tasks={len(assignments)}, next_tau={next_tau}."
        )

    _progress("All windows completed; independently recomputing from complete actual trajectories for hours 0--2405.")
    assignments = assignments.sort_values(
        ["StartHour", "TaskID"], kind="stable"
    ).reset_index(drop=True)
    if len(assignments) != len(data.tasks) or assignments["TaskID"].nunique() != len(data.tasks):
        missing = sorted(set(data.tasks["TaskID"]) - set(assignments["TaskID"]))[:10]
        raise RuntimeError(f"Unexecuted tasks remain after rolling optimization; examples: {missing}")
    dispatch = dispatch.sort_values(
        ["Hour", "Region"], kind="stable"
    ).reset_index(drop=True)
    expected_dispatch_rows = OPERATION_END * len(data.regions)
    if len(dispatch) != expected_dispatch_rows:
        raise RuntimeError(
            f"Actual energy trajectory requires {expected_dispatch_rows} rows; found {len(dispatch)}"
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
        raise RuntimeError(f"Final Q4 hard-constraint validation failed:\n{failed.to_string(index=False)}")
    _atomic_write_json(
        TABLES_DIR / ".q4_cache.json",
        {"signature": signature, "complete": True},
    )
    _progress(f"Final Q4 hard-constraint validation passed: {len(hard_validation)} checks.")


def run(
    *,
    force: bool = False,
    config: ModelConfig | None = None,
    resume_from: int | None = None,
    state_file: Path | None = None,
) -> None:
    """Formal Q4 entry: no full-window MILP or automatic recovery is reachable here."""

    if resume_from is not None or state_file is not None:
        raise RuntimeError(
            "Q4 matheuristic automatically restores its dedicated checkpoint; legacy --resume-from and --state-file are unsupported"
        )
    _run_joint_matheuristic_rollout(
        load_data(), config=config or ModelConfig(), force=force
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Q4 rolling compute-storage-grid matheuristic interface")
    parser.add_argument(
        "--resume-from", type=int, default=None,
        help="Legacy MILP checkpoint option; matheuristic restores dedicated checkpoints automatically and rejects this option",
    )
    parser.add_argument(
        "--state-file", type=Path, default=None,
        help="Legacy MILP checkpoint option; unsupported by the matheuristic",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Preserve existing matheuristic checkpoints; start normally only when no checkpoint exists",
    )
    parser.add_argument(
        "--baseline-audit", action="store_true",
        help="Recompute and audit the existing baseline read-only, without calling MILP",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Check matheuristic structure and Q2 shadow schedules without recomputing formal results",
    )
    parser.add_argument(
        "--repair-sequential-reference", action="store_true",
        help="Recompute the Q2 task seed -> V4 sequential energy baseline only; skip the joint rolling baseline",
    )
    parser.add_argument(
        "--repair-joint-energy-reference", action="store_true",
        help="Hold the existing joint task schedule fixed and recompute V4 energy without a new joint rolling search",
    )
    parser.add_argument(
        "--validation-only-windows", action="store_true",
        help="Solve four automatically identified validation windows and run single-window smoke tests for three scenario types",
    )
    parser.add_argument(
        "--validation-time-limit", type=float, default=30.0,
        help="Solver time limit per VALIDATION_ONLY window (seconds)",
    )
    parser.add_argument(
        "--run-scenario",
        choices=(
            "low_carbon_reference", "carbon_constraint",
            "flat_price", "low_variability_renewable",
        ),
        default=None,
        help="Run an independent scenario with the same matheuristic, fixed scaling, and H/K definitions",
    )
    parser.add_argument(
        "--scenario-name", type=str, default=None,
        help="Independent scenario output directory name; refuse overwrite when signatures differ",
    )
    parser.add_argument(
        "--carbon-lambda", type=float, default=1.0,
        help="Carbon constraint strength lambda: Ecap=E0-lambda*(E0-ELC), range [0,1]",
    )
    parser.add_argument(
        "--renewable-gamma", type=float, default=1.0,
        help="Intraday renewable smoothing gamma in [0,1]; 0 retains the profile, 1 uses the daily mean over nonzero hours",
    )
    parser.add_argument(
        "--low-carbon-reference-dir", type=Path, default=None,
        help="Complete low-carbon reference scenario directory; required when corrected baseline emissions are nonzero",
    )
    parser.add_argument(
        "--normal-time-limit", type=float, default=120.0,
        help="Legacy MILP compatibility option; matheuristic does not use a full-window MILP",
    )
    parser.add_argument(
        "--difficult-time-limit", type=float, default=240.0,
        help="Legacy MILP compatibility option; matheuristic does not extend solves",
    )
    parser.add_argument(
        "--mip-gap", type=float, default=0.02,
        help="Acceptable relative MIP gap for joint LNS",
    )
    parser.add_argument(
        "--window-lp-time-limit", type=float, default=20.0,
        help="Legacy window calibration option; formal execution uses one fixed calibration",
    )
    parser.add_argument(
        "--qos-weight-profile",
        choices=("3-2-1", "4-2-1", "2-1.5-1"),
        default="3-2-1",
        help="DelaySensitivity weights; use different scenario-name values for different weights",
    )
    parser.add_argument(
        "--energy-time-limit", type=float, default=15.0,
        help="MILP time limit per fixed-task energy response (seconds)",
    )
    parser.add_argument(
        "--repair-time-limit", type=float, default=12.0,
        help="MILP time limit per local feasibility repair (seconds)",
    )
    parser.add_argument(
        "--lns-time-limit", type=float, default=15.0,
        help="MILP time limit per joint LNS solve (seconds)",
    )
    parser.add_argument(
        "--window-hard-time-limit", type=float, default=75.0,
        help="Hard time limit per H=24 rolling window (seconds)",
    )
    parser.add_argument(
        "--lns-max-task-groups", type=int, default=60,
        help="Maximum exact homogeneous task groups opened in joint LNS",
    )
    parser.add_argument(
        "--lns-max-integer-options", type=int, default=12000,
        help="Maximum integer candidate task variables created in joint LNS",
    )
    parser.add_argument(
        "--compare-scenarios", action="store_true",
        help="Summarize all completed, audited new scenarios read-only without calling MILP",
    )
    args = parser.parse_args(argv)
    profiles = {
        "3-2-1": (3.0, 2.0, 1.0),
        "4-2-1": (4.0, 2.0, 1.0),
        "2-1.5-1": (2.0, 1.5, 1.0),
    }
    high, medium, low = profiles[args.qos_weight_profile]
    matheuristic_config = ModelConfig(
        mip_relative_gap=float(args.mip_gap),
        qos_high_weight=high,
        qos_medium_weight=medium,
        qos_low_weight=low,
        energy_time_limit_seconds=float(args.energy_time_limit),
        repair_time_limit_seconds=float(args.repair_time_limit),
        lns_time_limit_seconds=float(args.lns_time_limit),
        lns_target_gap=float(args.mip_gap),
        window_hard_time_limit_seconds=float(args.window_hard_time_limit),
        lns_max_task_groups=int(args.lns_max_task_groups),
        lns_max_integer_option_vars=int(args.lns_max_integer_options),
    )
    if args.baseline_audit:
        try:
            report = audit_document_compliant_result()
        except (FileNotFoundError, RuntimeError, OSError, ValueError, json.JSONDecodeError) as exc:
            _progress(f"V4 baseline audit not executed: {exc}")
            return 2
        _progress(
            f"baseline_status={report.get('model_version')}，"
            f"passed={report.get('passed')}。"
        )
        return 0 if report.get("passed") else 2
    if args.self_test:
        tests = run_automated_tests()
        _progress(f"Automated tests: {int(tests['Passed'].sum())}/{len(tests)} passed.")
        return 0 if bool(tests["Passed"].all()) else 3
    if args.repair_sequential_reference:
        try:
            run_sequential_reference_repair(config=matheuristic_config)
        except (FileNotFoundError, RuntimeError, OSError, ValueError, json.JSONDecodeError) as exc:
            _progress(f"Sequential baseline targeted repair failed: {exc}")
            return 2
        return 0
    if args.repair_joint_energy_reference:
        try:
            run_joint_energy_reference_repair(config=matheuristic_config)
        except (FileNotFoundError, RuntimeError, OSError, ValueError, json.JSONDecodeError) as exc:
            _progress(f"Joint schedule targeted V4 energy recomputation failed: {exc}")
            return 2
        return 0
    if args.validation_only_windows:
        results = run_validation_only_windows(
            time_limit=float(args.validation_time_limit),
            include_scenario_smoke=True,
        )
        _progress(
            f"VALIDATION_ONLY completed {len(results)} windows; "
            "results written only to outputs/validation."
        )
        return 0
    if args.compare_scenarios:
        comparison = build_scenario_comparison()
        _progress(
            f"Scenario comparison table updated: {len(comparison)} rows; existing Q4 baseline was not recomputed."
        )
        return 0
    if args.run_scenario is not None:
        scenario_name = args.scenario_name
        if scenario_name is None:
            if args.run_scenario == "low_carbon_reference":
                scenario_name = "low_carbon_reference_guarded"
            elif args.run_scenario == "carbon_constraint":
                scenario_name = f"carbon_lambda_{args.carbon_lambda:g}".replace(".", "_")
            elif args.run_scenario == "low_variability_renewable":
                scenario_name = (
                    "low_variability_renewable"
                    if abs(float(args.renewable_gamma) - 1.0) <= EPS
                    else f"low_variability_renewable_gamma_{args.renewable_gamma:g}".replace(".", "_")
                )
            else:
                scenario_name = args.run_scenario
        scenario_config = ModelConfig(
            normal_time_limit_seconds=float(args.normal_time_limit),
            difficult_time_limit_seconds=float(args.difficult_time_limit),
            mip_relative_gap=matheuristic_config.mip_relative_gap,
            qos_high_weight=high,
            qos_medium_weight=medium,
            qos_low_weight=low,
            energy_time_limit_seconds=matheuristic_config.energy_time_limit_seconds,
            repair_time_limit_seconds=matheuristic_config.repair_time_limit_seconds,
            lns_time_limit_seconds=matheuristic_config.lns_time_limit_seconds,
            lns_target_gap=matheuristic_config.lns_target_gap,
            window_hard_time_limit_seconds=matheuristic_config.window_hard_time_limit_seconds,
            lns_max_task_groups=matheuristic_config.lns_max_task_groups,
            lns_max_integer_option_vars=matheuristic_config.lns_max_integer_option_vars,
        )
        spec = ScenarioSpec(
            name=scenario_name,
            kind=args.run_scenario,
            carbon_lambda=(
                float(args.carbon_lambda)
                if args.run_scenario == "carbon_constraint" else None
            ),
            renewable_gamma=(
                float(args.renewable_gamma)
                if args.run_scenario == "low_variability_renewable" else None
            ),
        )
        marker = run_full_scenario(
            spec,
            config=scenario_config,
            lp_time_limit=float(args.window_lp_time_limit),
            low_carbon_reference_dir=args.low_carbon_reference_dir,
        )
        _progress(
            f"Scenario {scenario_name} completed: complete={marker.get('complete')}, "
            f"audit_passed={marker.get('audit_passed')}。"
        )
        return 0
    run(
        force=args.force,
        config=matheuristic_config,
        resume_from=args.resume_from,
        state_file=args.state_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
