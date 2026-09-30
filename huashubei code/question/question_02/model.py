"""问题2最终版：精确聚合 + 全局连续定标 + 滚动数学启发式多目标调度。

对应论文 Q2 重构模型，目标是在不删减合法区域/合法整数开工时刻的前提下，
通过“等价聚合、约束消元、全局定标、滚动边际构造 + 大邻域MILP精修”兼顾模型质量与求解速度。

核心口径
--------
1. 使用 0--2399 小时实际到达任务；2400--2405 仅用于结清；第 2406 小时不得占用。
2. 任务不可抢占、不可拆分；实时推理到达即开工；弹性任务在合法时间窗内错峰。
3. 仅合并在全部 Q2 有效属性上完全相同的任务；聚合变量是整数计数，属于严格等价重参数化。
4. GPU 约束独立保留；IT、设施功率、最大购电边界合并为逐时有效 AI IT 容量。
5. Q2 不启用储能；给定设施负荷后，购电、外送、弃电按题面无储能口径结算。
6. 纯算力基准优先采用解析零迁移快路；四个全局连续理想点采用紧凑精确列生成，不一次性展开全时域候选矩阵。
7. 均衡滚动采用 H=24、K=48：先按原 min-max 目标进行多目标边际贪心构造，再对关键任务做小规模大邻域MILP精修。
8. 最终结果回到 TaskID 层并独立复算全部硬约束、资源曲线和四项指标。

依赖
----
Python 3.10+
numpy, pandas, scipy

阶段4默认采用滚动数学启发式，不依赖全窗口大规模MILP；highspy仅为旧版
原生HiGHS兼容接口的可选依赖，新主线不要求安装。

8GB可用内存建议正式运行：
python model.py --decision-window 24 --lookahead 48 --mip-rel-gap 0.005 \
    --solver-time-limit 180 --max-solver-time-limit 900 --resume
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
import json
import logging
import math
import threading
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


# =============================================================================
# 0. 路径、时域与默认参数
# =============================================================================

QUESTION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = QUESTION_DIR.parents[1]
PROCESSED_DIR = QUESTION_DIR / "data" / "processed"
Q2_DIR = PROCESSED_DIR / "q2"
SHARED_DIR = PROJECT_DIR / "question" / "question_01" / "data" / "processed" / "shared"
TABLES_DIR = QUESTION_DIR / "outputs" / "tables"
LOGS_DIR = QUESTION_DIR / "outputs" / "logs"
CHECKPOINT_DIR = QUESTION_DIR / "outputs" / "checkpoints"
MODEL_LOG_PATH = LOGS_DIR / "question_02_model_refactored.log"

MAIN_START_HOUR = 0
MAIN_END_HOUR = 2399
TAIL_END_HOUR = 2405
TERMINAL_HOUR = 2406

DEFAULT_DECISION_WINDOW = 24
DEFAULT_LOOKAHEAD = 48
DEFAULT_MIP_REL_GAP = 5e-3
DEFAULT_SOLVER_TIME_LIMIT = 180.0
DEFAULT_MAX_SOLVER_TIME_LIMIT = 900.0
EMERGENCY_SOLVER_TIME_LIMIT = 900.0
BALANCED_MAX_ACCEPTABLE_MIP_GAP = 0.02

# 阶段4原生HiGHS/MIP Start。主检查点版本保持不变，保证已完成窗口直接续算。
BALANCED_WARMSTART_SCHEMA_VERSION = 1
BALANCED_WARMSTART_VALUE_TOL = 1e-7
BALANCED_WARMSTART_POSITIVE_TOL = 1e-9
HIGHS_MIP_MAX_START_NODES = 2000
HIGHS_PARALLEL = "choose"   # 8GB机器不强制并行，交给HiGHS按模型自行选择
HIGHS_THREADS = 0           # 0=自动
# 阶段4结构证书快路：不额外求LP。由 min-max 行的非负系数直接给出严格 z 下界，
# 再把 z 限制在满足目标 MIP gap 的窄带内，只做整数可行性搜索。
BALANCED_STRUCTURAL_FASTPATH = True
BALANCED_STRUCTURAL_FASTPATH_TIME_LIMIT = 180.0
BALANCED_STRUCTURAL_COEF_TOL = 1e-10

# 新阶段4：滚动数学启发式。保持 balanced v3 主检查点不变，因此此前已完成窗口
# 会原样恢复；从下一个未完成窗口开始改用快速构造 + 小规模LNS-MILP。
HEURISTIC_LNS_ENABLED = True
HEURISTIC_LNS_MAX_CLASSES = 60
HEURISTIC_LNS_TIME_LIMIT = 20.0
HEURISTIC_LNS_MIP_GAP = 0.05
HEURISTIC_REPAIR_MAX_CLASSES = 80
HEURISTIC_REPAIR_TIME_LIMIT = 15.0
HEURISTIC_REPAIR_EXPANDED_MAX_CLASSES = 160
HEURISTIC_REPAIR_EXPANDED_TIME_LIMIT = 25.0
HEURISTIC_CAPACITY_TOL = 1e-7
HEURISTIC_SCORE_TOL = 1e-10
DEFAULT_PROGRESS_INTERVAL = 30.0
FLOAT_EPS = 1e-8
INTEGER_TOL = 1e-5
LEXICOGRAPHIC_REL_TOL = 1e-7
CHECKPOINT_SCHEMA_VERSION = 3
CALIBRATION_SCHEMA_VERSION = 6

OBJECTIVE_NAMES = ("Cost", "Carbon", "MeanLatency", "RenewableUnusedRate")

CG_REDUCED_COST_TOL = 1e-7
CG_MAX_ITERATIONS = 400
CG_MAX_NEW_COLUMNS_PER_ITERATION = 2400
CG_COLUMNS_PER_CLASS = 2
CG_POSITIVE_COLUMN_TOL = 1e-9
CG_COLUMN_POOL_LIMIT = 60000
CG_POOL_NEAR_ZERO_KEEP = 10000
# RenewableUnusedRate 锚点存在大面积退化最优面，允许更大的工作列池与批量入列，
# 但最终仍由完整定价或严格的对偶下界证书控制误差。
CG_RENEWABLE_MAX_NEW_COLUMNS_PER_ITERATION = 5000
CG_RENEWABLE_COLUMNS_PER_CLASS = 3
CG_RENEWABLE_COLUMN_POOL_LIMIT = 80000
CG_RENEWABLE_CERTIFIED_ABS_GAP = 0.0025  # 未利用率绝对误差 <= 0.25 个百分点
CG_ITERATION_CHECKPOINT_EVERY = 1
MEMORY_SOFT_LIMIT_GB = 6.25
MEMORY_HARD_LIMIT_GB = 7.20

REGION_HOUR_COLUMNS = (
    "Hour",
    "Region",
    "TimeRole",
    "ElectricityPrice_CNY_per_MWh",
    "SellPrice_CNY_per_MWh",
    "CarbonIntensity_tCO2_per_MWh",
    "AvailableRenewable_MW",
    "NonAI_IT_Load_MW",
    "Available_GPU",
    "Max_IT_Power_MW",
    "PUE",
    "Max_Facility_Power_MW",
)
TASK_COLUMNS = (
    "TaskID",
    "TaskType",
    "ArrivalHour",
    "SourceRegion",
    "GPU_Demand",
    "EstimatedDuration_min",
    "MaxLatency_ms",
    "EarliestStartHour",
    "LatestFinishHour",
    "Duration_h",
    "Task_Full_IT_Power_MW",
)
CANDIDATE_COLUMNS = (
    "TaskID",
    "TargetRegion",
    "NetworkLatency_ms",
    "MaxLatency_ms",
)
BOUNDARY_COLUMNS = (
    "Region",
    "SellLimit_MW",
    "MaxGridImport_MW",
    "MaxGridExport_MW",
)


# =============================================================================
# 1. 数据结构
# =============================================================================

@dataclass(frozen=True, slots=True)
class CandidateRegion:
    target_region: str
    network_latency_ms: float
    max_latency_ms: float


@dataclass(slots=True)
class InputBundle:
    region_hour: pd.DataFrame
    tasks: pd.DataFrame
    candidates: pd.DataFrame
    candidate_map: dict[str, tuple[CandidateRegion, ...]]
    regions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExactTaskClass:
    class_id: int
    task_ids: tuple[str, ...]
    task_type: str
    arrival_hour: int
    source_region: str
    gpu_demand: float
    duration_h: float
    task_power_mw: float
    max_latency_ms: float
    earliest_start_hour: int
    latest_finish_hour: float
    candidate_regions: tuple[tuple[str, float], ...]

    @property
    def count(self) -> int:
        return len(self.task_ids)


@dataclass(frozen=True, slots=True)
class ClassDispatchOption:
    option_id: int
    class_id: int
    target_region: str
    network_latency_ms: float
    start_hour: int
    finish_hour: float


@dataclass(slots=True)
class AggregationAudit:
    original_task_count: int
    exact_class_count: int
    singleton_class_count: int
    max_class_size: int
    mean_class_size: float
    realtime_task_count: int
    fully_fixed_task_count: int
    original_option_count: int
    aggregated_option_count: int

    @property
    def task_compression_ratio(self) -> float:
        return (
            self.original_task_count / self.exact_class_count
            if self.exact_class_count > 0
            else float("nan")
        )

    @property
    def option_reduction_rate(self) -> float:
        if self.original_option_count <= 0:
            return 0.0
        return 1.0 - self.aggregated_option_count / self.original_option_count


@dataclass(slots=True)
class AggregatedModel:
    matrix: Any
    lower: np.ndarray
    upper: np.ndarray
    variable_lower: np.ndarray
    variable_upper: np.ndarray
    integrality: np.ndarray
    objective_vectors: dict[str, np.ndarray]
    objective_constants: dict[str, float]
    migration_objective: np.ndarray
    wait_objective: np.ndarray
    option_count: int
    options: list[ClassDispatchOption]
    class_ids: tuple[int, ...]
    class_row_count: int
    z_index: int | None
    decision_end: int
    objective_end: int
    resource_end: int
    global_task_count: int
    global_renewable_mwh: float


@dataclass(frozen=True, slots=True)
class GlobalCalibration:
    ideal_lb: dict[str, float]
    baseline_value: dict[str, float]
    reference_value: dict[str, float]
    scale: dict[str, float]
    active_metrics: tuple[str, ...]


@dataclass(slots=True)
class SolveResult:
    vector: np.ndarray
    status: int
    message: str
    mip_gap: float
    mip_dual_bound: float
    objective_value: float
    elapsed_seconds: float
    time_limit_used: float


@dataclass(frozen=True, slots=True)
class CGColumn:
    class_id: int
    target_region: str
    network_latency_ms: float
    start_hour: int
    finish_hour: float

    @property
    def key(self) -> tuple[int, str, int]:
        return (self.class_id, self.target_region, self.start_hour)


@dataclass(slots=True)
class CGAnchorResult:
    metric: str
    columns: list[CGColumn]
    weights: np.ndarray
    metrics: dict[str, float]
    iterations: int
    final_min_reduced_cost: float
    elapsed_seconds: float
    ideal_lower_bound: float
    certified_abs_gap: float
    convergence_status: str


# =============================================================================
# 2. 日志、读写与基础工具
# =============================================================================

def _configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"不支持的日志级别：{level_name}")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(MODEL_LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logging.basicConfig(level=level, handlers=[stream_handler, file_handler], force=True)


def _read_csv(path: Path, required_columns: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"缺少模型输入文件：{path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}缺少字段：{missing}")
    return frame


def _to_numeric(frame: pd.DataFrame, columns: Iterable[str], source_name: str) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any():
            raise ValueError(f"{source_name}的{column}存在无法转换为数值的记录")
    return result


def _write_table(frame: pd.DataFrame, filename: str) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        TABLES_DIR / filename,
        index=False,
        encoding="utf-8-sig",
        float_format="%.15g",
    )


def _task_signature(bundle: InputBundle) -> str:
    columns = [
        "TaskID",
        "TaskType",
        "ArrivalHour",
        "SourceRegion",
        "GPU_Demand",
        "Duration_h",
        "MaxLatency_ms",
        "EarliestStartHour",
        "LatestFinishHour",
        "Task_Full_IT_Power_MW",
    ]
    text = bundle.tasks.loc[:, columns].sort_values("TaskID", kind="stable").to_csv(index=False)
    candidate_text = bundle.candidates.sort_values(
        ["TaskID", "TargetRegion"], kind="stable"
    ).to_csv(index=False)
    return hashlib.sha256((text + "\n" + candidate_text).encode("utf-8")).hexdigest()


def _float_key(value: float) -> str:
    """使用读入后的浮点精确值做聚合键，不人为四舍五入。"""
    return float(value).hex()


def _overlap_by_hour(start_hour: float, duration_h: float) -> tuple[tuple[int, float], ...]:
    finish_hour = float(start_hour) + float(duration_h)
    if start_hour < MAIN_START_HOUR - FLOAT_EPS or finish_hour > TERMINAL_HOUR + FLOAT_EPS:
        return tuple()
    first_hour = max(MAIN_START_HOUR, int(math.floor(start_hour)))
    last_hour = min(TAIL_END_HOUR, int(math.ceil(finish_hour - FLOAT_EPS)))
    overlaps: list[tuple[int, float]] = []
    for hour in range(first_hour, last_hour + 1):
        overlap = max(
            0.0,
            min(finish_hour, float(hour + 1)) - max(float(start_hour), float(hour)),
        )
        if overlap > FLOAT_EPS:
            overlaps.append((hour, overlap))
    return tuple(overlaps)


def _sparse_from_rows(rows: list[dict[int, float]], n_variables: int):
    from scipy.sparse import coo_matrix

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for row_index, terms in enumerate(rows):
        for column, value in terms.items():
            if abs(value) <= FLOAT_EPS:
                continue
            row_indices.append(row_index)
            column_indices.append(int(column))
            values.append(float(value))
    return coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(rows), n_variables),
    ).tocsr()


def _linear_constraint_components(
    matrix: Any,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[Any | None, np.ndarray | None, Any | None, np.ndarray | None]:
    from scipy.sparse import csr_matrix, vstack

    constraint_matrix = matrix.tocsr() if hasattr(matrix, "tocsr") else csr_matrix(matrix)
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    equality_mask = (
        finite_lower
        & finite_upper
        & np.isclose(lower, upper, rtol=0.0, atol=FLOAT_EPS)
    )
    eq_idx = np.flatnonzero(equality_mask)
    a_eq = constraint_matrix[eq_idx] if len(eq_idx) else None
    b_eq = lower[eq_idx] if len(eq_idx) else None

    upper_idx = np.flatnonzero(finite_upper & ~equality_mask)
    lower_idx = np.flatnonzero(finite_lower & ~equality_mask)
    blocks = []
    bounds = []
    if len(upper_idx):
        blocks.append(constraint_matrix[upper_idx])
        bounds.append(upper[upper_idx])
    if len(lower_idx):
        blocks.append(-constraint_matrix[lower_idx])
        bounds.append(-lower[lower_idx])
    if blocks:
        a_ub = vstack(blocks, format="csr")
        b_ub = np.concatenate(bounds)
    else:
        a_ub = None
        b_ub = None
    return a_ub, b_ub, a_eq, b_eq


def _result_float(result: Any, name: str) -> float:
    value = getattr(result, name, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")



def _process_rss_gb() -> float:
    """尽量读取当前 Python 进程工作集；失败时返回 NaN，不引入强制新依赖。"""
    try:
        import psutil  # type: ignore
        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 ** 3)
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(counters),
                counters.cb,
            )
            if ok:
                return float(counters.WorkingSetSize) / (1024.0 ** 3)
        except Exception:
            pass
    return float("nan")


def _memory_guard(label: str, hard: bool = False) -> float:
    """8GB机器内存保护：软阈值只警告并回收，硬阈值才主动终止。"""
    rss = _process_rss_gb()
    if not math.isfinite(rss):
        return rss
    logging.info("内存状态：%s，当前进程 RSS=%.2f GB。", label, rss)
    if rss >= MEMORY_SOFT_LIMIT_GB:
        gc.collect()
        rss = _process_rss_gb()
        if math.isfinite(rss):
            logging.warning(
                "%s前内存已进入高水位：RSS=%.2f GB（软阈值%.2f GB）。",
                label, rss, MEMORY_SOFT_LIMIT_GB,
            )
    if hard and math.isfinite(rss) and rss >= MEMORY_HARD_LIMIT_GB:
        raise MemoryError(
            f"{label}前进程内存已达{rss:.2f} GB，超过硬保护阈值"
            f"{MEMORY_HARD_LIMIT_GB:.2f} GB；为避免系统死机主动停止。"
            "可使用已有检查点恢复。"
        )
    return rss

def _run_with_heartbeat(callable_obj, *, label: str, interval: float, **kwargs):
    if interval <= 0.0:
        return callable_obj(**kwargs)
    stop_event = threading.Event()
    started = time.perf_counter()

    def heartbeat() -> None:
        while not stop_event.wait(interval):
            logging.info("求解心跳：%s仍在运行，已耗时%.1fs。", label, time.perf_counter() - started)

    thread = threading.Thread(target=heartbeat, name="q2-solver-heartbeat", daemon=True)
    thread.start()
    try:
        return callable_obj(**kwargs)
    finally:
        stop_event.set()
        thread.join(timeout=max(1.0, min(interval, 5.0)))


# =============================================================================
# 3. 输入读取与严格校验
# =============================================================================

def _load_inputs() -> InputBundle:
    logging.info("阶段1：读取 Q2 输入。")
    q2_input = _read_csv(Q2_DIR / "q2_region_hour_input.csv", REGION_HOUR_COLUMNS)
    tasks = _read_csv(SHARED_DIR / "tasks_clean.csv", TASK_COLUMNS)
    candidates = _read_csv(SHARED_DIR / "task_candidate_regions.csv", CANDIDATE_COLUMNS)
    boundaries = _read_csv(SHARED_DIR / "storage_params.csv", BOUNDARY_COLUMNS)

    q2_input = _to_numeric(
        q2_input,
        [
            "Hour",
            "ElectricityPrice_CNY_per_MWh",
            "SellPrice_CNY_per_MWh",
            "CarbonIntensity_tCO2_per_MWh",
            "AvailableRenewable_MW",
            "NonAI_IT_Load_MW",
            "Available_GPU",
            "Max_IT_Power_MW",
            "PUE",
            "Max_Facility_Power_MW",
        ],
        "q2_region_hour_input.csv",
    )
    tasks = _to_numeric(
        tasks,
        [
            "ArrivalHour",
            "GPU_Demand",
            "EstimatedDuration_min",
            "MaxLatency_ms",
            "EarliestStartHour",
            "LatestFinishHour",
            "Duration_h",
            "Task_Full_IT_Power_MW",
        ],
        "tasks_clean.csv",
    )
    candidates = _to_numeric(
        candidates,
        ["NetworkLatency_ms", "MaxLatency_ms"],
        "task_candidate_regions.csv",
    )
    boundaries = _to_numeric(
        boundaries,
        ["SellLimit_MW", "MaxGridImport_MW", "MaxGridExport_MW"],
        "storage_params.csv",
    )

    q2_input["Hour"] = q2_input["Hour"].astype(int)
    q2_input["Region"] = q2_input["Region"].astype(str)
    q2_input["TimeRole"] = q2_input["TimeRole"].astype(str).str.strip().str.lower()
    tasks["TaskID"] = tasks["TaskID"].astype(str)
    tasks["TaskType"] = tasks["TaskType"].astype(str)
    tasks["SourceRegion"] = tasks["SourceRegion"].astype(str)
    candidates["TaskID"] = candidates["TaskID"].astype(str)
    candidates["TargetRegion"] = candidates["TargetRegion"].astype(str)
    boundaries["Region"] = boundaries["Region"].astype(str)

    if tasks["TaskID"].duplicated().any():
        raise ValueError("tasks_clean.csv 的 TaskID 必须唯一")
    if candidates.duplicated(["TaskID", "TargetRegion"]).any():
        raise ValueError("task_candidate_regions.csv 存在重复 TaskID×TargetRegion")
    if q2_input.duplicated(["Hour", "Region"]).any():
        raise ValueError("q2_region_hour_input.csv 存在重复 Hour×Region")
    if boundaries["Region"].duplicated().any():
        raise ValueError("storage_params.csv 的 Region 必须唯一")

    if not tasks["ArrivalHour"].between(MAIN_START_HOUR, MAIN_END_HOUR).all():
        raise ValueError("Q2 实际任务 ArrivalHour 必须位于 0--2399")
    if (tasks["LatestFinishHour"] > TERMINAL_HOUR + FLOAT_EPS).any():
        raise ValueError("存在 LatestFinishHour 超过 2406 的任务")
    if (
        tasks["LatestFinishHour"]
        < tasks["EarliestStartHour"] + tasks["Duration_h"] - FLOAT_EPS
    ).any():
        raise ValueError("存在没有合法开工时刻的任务")

    if not candidates["TaskID"].isin(set(tasks["TaskID"])).all():
        raise ValueError("候选区域表包含不存在的 TaskID")
    if (
        candidates["NetworkLatency_ms"]
        > candidates["MaxLatency_ms"] + FLOAT_EPS
    ).any():
        raise ValueError("候选区域表中存在超过 MaxLatency 的记录")

    if (q2_input["Hour"] == TERMINAL_HOUR).any():
        q2_input = q2_input.loc[q2_input["Hour"] <= TAIL_END_HOUR].copy()
    if not q2_input["Hour"].between(MAIN_START_HOUR, TAIL_END_HOUR).all():
        raise ValueError("Q2 逐时输入应覆盖 0--2405")

    main_mismatch = q2_input.loc[
        q2_input["Hour"].between(MAIN_START_HOUR, MAIN_END_HOUR), "TimeRole"
    ].ne("main").any()
    tail_mismatch = q2_input.loc[
        q2_input["Hour"].between(MAIN_END_HOUR + 1, TAIL_END_HOUR), "TimeRole"
    ].ne("tail").any()
    if main_mismatch or tail_mismatch:
        raise ValueError("TimeRole 与 0--2399 main、2400--2405 tail 的模型时域不一致")

    region_hour = q2_input.merge(
        boundaries.loc[:, list(BOUNDARY_COLUMNS)],
        how="left",
        on="Region",
        validate="many_to_one",
    )
    if region_hour[list(BOUNDARY_COLUMNS[1:])].isna().any().any():
        raise ValueError("storage_params.csv 无法覆盖所有区域")
    region_hour["ExportLimit_MW"] = np.minimum(
        region_hour["SellLimit_MW"], region_hour["MaxGridExport_MW"]
    )
    if (region_hour["ExportLimit_MW"] < -FLOAT_EPS).any():
        raise ValueError("新能源外送上限不能为负")

    # Q2 等价有效 AI IT 容量：IT、设施、最大购电三者取交集。
    effective_it = region_hour["Max_IT_Power_MW"] - region_hour["NonAI_IT_Load_MW"]
    effective_facility = (
        region_hour["Max_Facility_Power_MW"] / region_hour["PUE"]
        - region_hour["NonAI_IT_Load_MW"]
    )
    effective_grid = (
        (region_hour["AvailableRenewable_MW"] + region_hour["MaxGridImport_MW"])
        / region_hour["PUE"]
        - region_hour["NonAI_IT_Load_MW"]
    )
    region_hour["Effective_AI_IT_Capacity_MW_Exact"] = np.minimum.reduce(
        [
            effective_it.to_numpy(dtype=float),
            effective_facility.to_numpy(dtype=float),
            effective_grid.to_numpy(dtype=float),
        ]
    )
    if (region_hour["Effective_AI_IT_Capacity_MW_Exact"] < -FLOAT_EPS).any():
        bad = region_hour.loc[
            region_hour["Effective_AI_IT_Capacity_MW_Exact"] < -FLOAT_EPS,
            ["Hour", "Region", "Effective_AI_IT_Capacity_MW_Exact"],
        ].head()
        raise ValueError(f"固定 NonAI 负荷已导致有效 AI 容量为负：\n{bad}")

    # 紧凑能源上图线性化成立条件：购电边际价格不低于售电价，
    # 售电价与碳强度非负。满足时只需 B/W 两个下图约束即可精确表示
    # 无储能结算；若数据不满足则拒绝使用紧凑版，避免牺牲模型正确性。
    if (region_hour["SellPrice_CNY_per_MWh"] < -FLOAT_EPS).any():
        raise ValueError("存在负售电价，8GB紧凑能源线性化不适用")
    if (
        region_hour["ElectricityPrice_CNY_per_MWh"]
        < region_hour["SellPrice_CNY_per_MWh"] - FLOAT_EPS
    ).any():
        raise ValueError("存在购电价低于售电价，8GB紧凑能源线性化不适用")
    if (region_hour["CarbonIntensity_tCO2_per_MWh"] < -FLOAT_EPS).any():
        raise ValueError("存在负碳强度，8GB紧凑能源线性化不适用")

    regions = tuple(sorted(region_hour["Region"].unique().tolist()))
    expected_keys = pd.MultiIndex.from_product(
        [range(MAIN_START_HOUR, TAIL_END_HOUR + 1), regions],
        names=["Hour", "Region"],
    )
    actual_keys = pd.MultiIndex.from_frame(region_hour.loc[:, ["Hour", "Region"]])
    missing = expected_keys.difference(actual_keys)
    if len(missing):
        raise ValueError(f"逐时输入缺少 Hour×Region 记录，示例：{list(missing[:5])}")

    candidate_map: dict[str, tuple[CandidateRegion, ...]] = {}
    for task_id, group in candidates.groupby("TaskID", sort=False):
        rows = [
            CandidateRegion(
                target_region=str(row.TargetRegion),
                network_latency_ms=float(row.NetworkLatency_ms),
                max_latency_ms=float(row.MaxLatency_ms),
            )
            for row in group.sort_values("TargetRegion", kind="stable").itertuples(index=False)
        ]
        candidate_map[str(task_id)] = tuple(rows)
    missing_candidates = set(tasks["TaskID"]) - set(candidate_map)
    if missing_candidates:
        raise ValueError(f"存在没有合法执行区域的任务，示例：{sorted(missing_candidates)[:5]}")

    logging.info(
        "输入完成：任务=%d，候选区域记录=%d，区域=%d，逐时记录=%d。",
        len(tasks),
        len(candidates),
        len(regions),
        len(region_hour),
    )
    return InputBundle(
        region_hour=region_hour.sort_values(["Hour", "Region"], kind="stable").reset_index(drop=True),
        tasks=tasks.sort_values(["ArrivalHour", "TaskID"], kind="stable").reset_index(drop=True),
        candidates=candidates.sort_values(["TaskID", "TargetRegion"], kind="stable").reset_index(drop=True),
        candidate_map=candidate_map,
        regions=regions,
    )


# =============================================================================
# 4. 同质任务精确聚合
# =============================================================================

def _latest_start_for_class(task_class: ExactTaskClass) -> int:
    latest_finish = min(task_class.latest_finish_hour, float(TERMINAL_HOUR))
    return int(
        math.floor(
            min(float(TAIL_END_HOUR), latest_finish - task_class.duration_h) + FLOAT_EPS
        )
    )


def _all_valid_start_hours(task_class: ExactTaskClass) -> list[int]:
    if task_class.task_type == "RealTimeInference":
        return [task_class.arrival_hour]
    earliest = max(task_class.arrival_hour, task_class.earliest_start_hour)
    latest = _latest_start_for_class(task_class)
    if latest < earliest:
        return []
    return list(range(earliest, latest + 1))


def _window_valid_start_hours(
    task_class: ExactTaskClass,
    tau: int,
    decision_end: int,
    plan_end: int,
) -> list[int]:
    """与现行滚动逻辑一致：H 内当前/紧急任务，K 中只做可重优化前瞻。"""
    latest_start = _latest_start_for_class(task_class)
    if task_class.task_type == "RealTimeInference":
        start = task_class.arrival_hour
        return [start] if tau <= start < plan_end else []

    current = task_class.arrival_hour < decision_end
    urgent = latest_start < decision_end
    if urgent:
        first = max(tau, task_class.arrival_hour, task_class.earliest_start_hour)
        last = min(decision_end - 1, latest_start)
    elif current:
        first = max(tau, task_class.arrival_hour, task_class.earliest_start_hour)
        last = min(plan_end - 1, latest_start)
    else:
        first = max(decision_end, task_class.arrival_hour, task_class.earliest_start_hour)
        last = min(plan_end - 1, latest_start)
    if last < first:
        return []
    return list(range(first, last + 1))


def _build_exact_task_classes(
    bundle: InputBundle,
) -> tuple[list[ExactTaskClass], dict[str, int]]:
    task_lookup = bundle.tasks.set_index("TaskID", drop=False)
    groups: dict[tuple[Any, ...], list[str]] = {}
    for task in bundle.tasks.itertuples(index=False):
        task_id = str(task.TaskID)
        candidate_signature = tuple(
            sorted(
                (
                    candidate.target_region,
                    _float_key(candidate.network_latency_ms),
                    _float_key(candidate.max_latency_ms),
                )
                for candidate in bundle.candidate_map[task_id]
            )
        )
        key = (
            str(task.TaskType),
            int(task.ArrivalHour),
            str(task.SourceRegion),
            _float_key(task.GPU_Demand),
            _float_key(task.Duration_h),
            _float_key(task.Task_Full_IT_Power_MW),
            _float_key(task.MaxLatency_ms),
            int(task.EarliestStartHour),
            _float_key(task.LatestFinishHour),
            candidate_signature,
        )
        groups.setdefault(key, []).append(task_id)

    classes: list[ExactTaskClass] = []
    task_to_class: dict[str, int] = {}
    for class_id, task_ids in enumerate(groups.values()):
        task_ids = sorted(task_ids)
        row = task_lookup.loc[task_ids[0]]
        candidate_rows = bundle.candidate_map[task_ids[0]]
        task_class = ExactTaskClass(
            class_id=class_id,
            task_ids=tuple(task_ids),
            task_type=str(row.TaskType),
            arrival_hour=int(row.ArrivalHour),
            source_region=str(row.SourceRegion),
            gpu_demand=float(row.GPU_Demand),
            duration_h=float(row.Duration_h),
            task_power_mw=float(row.Task_Full_IT_Power_MW),
            max_latency_ms=float(row.MaxLatency_ms),
            earliest_start_hour=int(row.EarliestStartHour),
            latest_finish_hour=float(row.LatestFinishHour),
            candidate_regions=tuple(
                sorted(
                    (candidate.target_region, float(candidate.network_latency_ms))
                    for candidate in candidate_rows
                )
            ),
        )
        if not _all_valid_start_hours(task_class):
            raise ValueError(f"任务类{class_id}没有合法整数开工时刻")
        classes.append(task_class)
        for task_id in task_ids:
            task_to_class[task_id] = class_id
    return classes, task_to_class


def _build_aggregation_audit(classes: list[ExactTaskClass]) -> AggregationAudit:
    sizes = np.asarray([c.count for c in classes], dtype=int)
    original_options = 0
    aggregated_options = 0
    fully_fixed = 0
    realtime = 0
    for task_class in classes:
        starts = len(_all_valid_start_hours(task_class))
        regions = len(task_class.candidate_regions)
        original_options += task_class.count * starts * regions
        aggregated_options += starts * regions
        if starts == 1 and regions == 1:
            fully_fixed += task_class.count
        if task_class.task_type == "RealTimeInference":
            realtime += task_class.count
    return AggregationAudit(
        original_task_count=int(sizes.sum()),
        exact_class_count=len(classes),
        singleton_class_count=int(np.sum(sizes == 1)),
        max_class_size=int(sizes.max()) if sizes.size else 0,
        mean_class_size=float(sizes.mean()) if sizes.size else float("nan"),
        realtime_task_count=realtime,
        fully_fixed_task_count=fully_fixed,
        original_option_count=original_options,
        aggregated_option_count=aggregated_options,
    )


def _audit_frame(audit: AggregationAudit) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "OriginalTaskCount": audit.original_task_count,
                "ExactTaskClassCount": audit.exact_class_count,
                "SingletonClassCount": audit.singleton_class_count,
                "MaxClassSize": audit.max_class_size,
                "MeanClassSize": audit.mean_class_size,
                "TaskCompressionRatio": audit.task_compression_ratio,
                "RealtimeTaskCount": audit.realtime_task_count,
                "FullyFixedTaskCount": audit.fully_fixed_task_count,
                "OriginalTaskLevelOptionCount": audit.original_option_count,
                "AggregatedOptionCount": audit.aggregated_option_count,
                "OptionReductionRate": audit.option_reduction_rate,
                "AggregationType": "exact_attribute_equivalence",
            }
        ]
    )


# =============================================================================
# 5. 聚合窗口模型
# =============================================================================

def _fixed_loads(
    fixed_assignments: pd.DataFrame,
    keys: tuple[tuple[int, str], ...],
) -> tuple[dict[tuple[int, str], float], dict[tuple[int, str], float]]:
    gpu = {key: 0.0 for key in keys}
    ai = {key: 0.0 for key in keys}
    if fixed_assignments.empty:
        return gpu, ai
    key_set = set(keys)
    for row in fixed_assignments.itertuples(index=False):
        for hour, overlap in _overlap_by_hour(float(row.StartHour), float(row.Duration_h)):
            key = (int(hour), str(row.TargetRegion))
            if key not in key_set:
                continue
            gpu[key] += float(row.GPU_Demand) * overlap
            ai[key] += float(row.Task_Full_IT_Power_MW) * overlap
    return gpu, ai


def _active_classes_and_options(
    classes: list[ExactTaskClass],
    remaining_count: dict[int, int],
    tau: int,
    decision_end: int,
    plan_end: int,
) -> tuple[list[int], list[ClassDispatchOption]]:
    """生成当前滚动域内的完整合法候选。

    8GB 低内存版不在每个候选对象中缓存逐小时 overlap 元组；该信息由
    (class_id, start_hour, duration_h) 可完全恢复，矩阵构造和结果复算时按需计算。
    这只改变内存表示，不改变任何合法候选或约束系数。
    """
    class_ids: list[int] = []
    options: list[ClassDispatchOption] = []
    option_id = 0
    for task_class in classes:
        rem = int(remaining_count.get(task_class.class_id, 0))
        if rem <= 0 or task_class.arrival_hour >= plan_end:
            continue
        starts = _window_valid_start_hours(task_class, tau, decision_end, plan_end)
        if not starts:
            if _latest_start_for_class(task_class) < decision_end:
                raise RuntimeError(
                    f"任务类{task_class.class_id}在tau={tau}已必须启动但无合法 H 区时刻"
                )
            continue
        class_ids.append(task_class.class_id)
        for target_region, latency in task_class.candidate_regions:
            for start in starts:
                if not _overlap_by_hour(float(start), task_class.duration_h):
                    continue
                options.append(
                    ClassDispatchOption(
                        option_id=option_id,
                        class_id=task_class.class_id,
                        target_region=target_region,
                        network_latency_ms=latency,
                        start_hour=start,
                        finish_hour=float(start + task_class.duration_h),
                    )
                )
                option_id += 1
    return class_ids, options

def _build_aggregated_model(
    *,
    bundle: InputBundle,
    classes: list[ExactTaskClass],
    class_lookup: dict[int, ExactTaskClass],
    remaining_count: dict[int, int],
    fixed_assignments: pd.DataFrame,
    tau: int,
    decision_end: int,
    plan_end: int,
    include_z: bool,
    include_energy_variables: bool,
    relax_all_task_variables: bool = False,
    integerize_lookahead: bool = False,
) -> AggregatedModel:
    """8GB混合优化版聚合模型。

    与原数学模型等价，但能源线性化由 B/W/X 四组约束压缩为 B/W 两组下图约束。
    在购电价>=售电价>=0、碳强度>=0时，Cost/Carbon/Q 的最小化或 min-max
    会自动把 B、W 压到物理最小值：
        B=max(F-RE, 0), W=max(RE-F-ExportLimit, 0)
    因而不需要显式 X 约束，减少约一半能源行和大量非零元。
    """
    from scipy.sparse import coo_array

    class_ids, options = _active_classes_and_options(
        classes, remaining_count, tau, decision_end, plan_end
    )
    if not class_ids or not options:
        raise ValueError(f"tau={tau}没有进入当前模型的任务类/候选")

    max_finish = max(option.finish_hour for option in options)
    resource_end = min(
        TERMINAL_HOUR,
        max(plan_end, int(math.ceil(max_finish - FLOAT_EPS))),
    )
    objective_end = min(plan_end, TERMINAL_HOUR)

    resource_frame = bundle.region_hour.loc[
        bundle.region_hour["Hour"].between(tau, resource_end - 1)
    ].copy()
    resource_frame = resource_frame.sort_values(["Hour", "Region"], kind="stable")
    resource_rows = list(resource_frame.itertuples(index=False))
    resource_keys = tuple((int(row.Hour), str(row.Region)) for row in resource_rows)
    resource_lookup = {(int(row.Hour), str(row.Region)): row for row in resource_rows}
    fixed_gpu, fixed_ai = _fixed_loads(fixed_assignments, resource_keys)

    objective_keys = tuple(key for key in resource_keys if tau <= key[0] < objective_end)
    objective_pos = {key: i for i, key in enumerate(objective_keys)}
    resource_pos = {key: i for i, key in enumerate(resource_keys)}

    option_count = len(options)
    energy_count = len(objective_keys) if include_energy_variables else 0
    b_indices = np.arange(option_count, option_count + energy_count, dtype=np.int64)
    w_indices = np.arange(
        option_count + energy_count,
        option_count + 2 * energy_count,
        dtype=np.int64,
    )
    z_index = option_count + 2 * energy_count if include_z else None
    n_variables = option_count + 2 * energy_count + (1 if include_z else 0)

    variable_lower = np.zeros(n_variables, dtype=np.float64)
    variable_upper = np.full(n_variables, np.inf, dtype=np.float64)
    integrality = np.zeros(n_variables, dtype=np.uint8)

    for j, option in enumerate(options):
        variable_upper[j] = float(remaining_count[option.class_id])
        if not relax_all_task_variables and (
            integerize_lookahead or option.start_hour < decision_end
        ):
            integrality[j] = 1

    if include_energy_variables:
        for k, key in enumerate(objective_keys):
            row = resource_lookup[key]
            variable_upper[b_indices[k]] = max(0.0, float(row.MaxGridImport_MW))
            variable_upper[w_indices[k]] = max(0.0, float(row.AvailableRenewable_MW))
    if z_index is not None:
        variable_upper[z_index] = 1e6

    C = len(class_ids)
    R = len(resource_keys)
    O = len(objective_keys)
    class_row = {class_id: i for i, class_id in enumerate(class_ids)}
    gpu_offset = C
    ai_offset = gpu_offset + R
    energy_offset = ai_offset + R
    # 每个目标时空键仅两行：B>=F-RE；W>=RE-F-ExportLimit。
    n_rows = energy_offset + (2 * O if include_energy_variables else 0)

    lower = np.full(n_rows, -np.inf, dtype=np.float64)
    upper = np.full(n_rows, np.inf, dtype=np.float64)
    for class_id, row_idx in class_row.items():
        rem = float(remaining_count[class_id])
        lower[row_idx] = rem
        upper[row_idx] = rem

    for key, pos in resource_pos.items():
        row = resource_lookup[key]
        upper[gpu_offset + pos] = float(row.Available_GPU) - fixed_gpu[key]
        upper[ai_offset + pos] = (
            float(row.Effective_AI_IT_Capacity_MW_Exact) - fixed_ai[key]
        )

    rr = array("i")
    cc = array("i")
    vv = array("d")

    def add_coef(r: int, c: int, v: float) -> None:
        if abs(v) > FLOAT_EPS:
            rr.append(int(r))
            cc.append(int(c))
            vv.append(float(v))

    if include_energy_variables:
        for k, key in enumerate(objective_keys):
            row = resource_lookup[key]
            base_facility = (float(row.NonAI_IT_Load_MW) + fixed_ai[key]) * float(row.PUE)
            re = float(row.AvailableRenewable_MW)
            export_limit = max(0.0, float(row.ExportLimit_MW))
            r_b = energy_offset + 2 * k
            r_w = r_b + 1
            lower[r_b] = base_facility - re
            lower[r_w] = re - base_facility - export_limit
            add_coef(r_b, int(b_indices[k]), 1.0)
            add_coef(r_w, int(w_indices[k]), 1.0)

    global_task_count = int(sum(task_class.count for task_class in classes))
    global_renewable = float(
        bundle.region_hour.loc[
            bundle.region_hour["Hour"].between(MAIN_START_HOUR, TAIL_END_HOUR),
            "AvailableRenewable_MW",
        ].sum()
    )

    objective_vectors = {
        name: np.zeros(n_variables, dtype=np.float64) for name in OBJECTIVE_NAMES
    }
    objective_constants = {name: 0.0 for name in OBJECTIVE_NAMES}
    migration_objective = np.zeros(n_variables, dtype=np.float64)
    wait_objective = np.zeros(n_variables, dtype=np.float64)

    if include_energy_variables:
        for k, key in enumerate(objective_keys):
            row = resource_lookup[key]
            base_facility = (float(row.NonAI_IT_Load_MW) + fixed_ai[key]) * float(row.PUE)
            re = float(row.AvailableRenewable_MW)
            sell = float(row.SellPrice_CNY_per_MWh)
            objective_vectors["Cost"][b_indices[k]] = (
                float(row.ElectricityPrice_CNY_per_MWh) - sell
            )
            objective_vectors["Cost"][w_indices[k]] = sell
            objective_constants["Cost"] += sell * (base_facility - re)
            objective_vectors["Carbon"][b_indices[k]] = float(
                row.CarbonIntensity_tCO2_per_MWh
            )
            if global_renewable > FLOAT_EPS:
                objective_vectors["RenewableUnusedRate"][w_indices[k]] = 1.0 / global_renewable

    # 直接按持续时间模式写稀疏系数，避免为每个候选缓存 overlap 元组。
    for j, option in enumerate(options):
        task_class = class_lookup[option.class_id]
        add_coef(class_row[option.class_id], j, 1.0)
        objective_vectors["MeanLatency"][j] = (
            option.network_latency_ms / max(global_task_count, 1)
        )
        if option.target_region != task_class.source_region:
            migration_objective[j] = task_class.gpu_demand * task_class.duration_h
        if task_class.task_type != "RealTimeInference":
            wait_objective[j] = max(
                0.0, float(option.start_hour - task_class.earliest_start_hour)
            )

        q = int(math.floor(task_class.duration_h + FLOAT_EPS))
        frac = float(task_class.duration_h - q)
        if frac < FLOAT_EPS:
            frac = 0.0
        overlaps = [(option.start_hour + h, 1.0) for h in range(q)]
        if frac > 0.0:
            overlaps.append((option.start_hour + q, frac))

        for hour, overlap in overlaps:
            key = (int(hour), option.target_region)
            pos = resource_pos.get(key)
            if pos is None:
                raise ValueError(f"候选资源占用超出当前资源范围：{key}")
            gpu_contrib = task_class.gpu_demand * overlap
            ai_contrib = task_class.task_power_mw * overlap
            pue = float(resource_lookup[key].PUE)
            facility_contrib = ai_contrib * pue
            add_coef(gpu_offset + pos, j, gpu_contrib)
            add_coef(ai_offset + pos, j, ai_contrib)

            k = objective_pos.get(key)
            if include_energy_variables and k is not None:
                r_b = energy_offset + 2 * k
                r_w = r_b + 1
                add_coef(r_b, j, -facility_contrib)
                add_coef(r_w, j, facility_contrib)
                objective_vectors["Cost"][j] += (
                    float(resource_lookup[key].SellPrice_CNY_per_MWh) * facility_contrib
                )

    row_np = np.frombuffer(rr, dtype=np.int32)
    col_np = np.frombuffer(cc, dtype=np.int32)
    val_np = np.frombuffer(vv, dtype=np.float64)
    coo = coo_array(
        (val_np, (row_np, col_np)),
        shape=(n_rows, n_variables),
        dtype=np.float64,
    )
    matrix = coo.tocsr() if relax_all_task_variables else coo.tocsc()
    del coo, row_np, col_np, val_np, rr, cc, vv, resource_rows
    gc.collect()

    return AggregatedModel(
        matrix=matrix,
        lower=lower,
        upper=upper,
        variable_lower=variable_lower,
        variable_upper=variable_upper,
        integrality=integrality,
        objective_vectors=objective_vectors,
        objective_constants=objective_constants,
        migration_objective=migration_objective,
        wait_objective=wait_objective,
        option_count=option_count,
        options=options,
        class_ids=tuple(class_ids),
        class_row_count=C,
        z_index=z_index,
        decision_end=decision_end,
        objective_end=objective_end,
        resource_end=resource_end,
        global_task_count=global_task_count,
        global_renewable_mwh=global_renewable,
    )

def _objective_value(model: AggregatedModel, metric: str, vector: np.ndarray) -> float:
    return float(model.objective_constants[metric]) + float(
        np.dot(model.objective_vectors[metric], vector)
    )


# =============================================================================
# 6. LP/MILP 求解器封装
# =============================================================================

def _prepare_lp_problem(
    model: AggregatedModel,
) -> tuple[Any | None, np.ndarray | None, Any | None, np.ndarray | None, np.ndarray]:
    """一次性把统一双边约束转换为 linprog 形式，供四个全局锚点复用。"""
    a_ub, b_ub, a_eq, b_eq = _linear_constraint_components(
        model.matrix, model.lower, model.upper
    )
    # Nx2 ndarray 避免 list(zip(...)) 为每个变量创建 Python tuple。
    bounds = np.column_stack(
        (model.variable_lower, model.variable_upper)
    ).astype(np.float64, copy=False)
    return a_ub, b_ub, a_eq, b_eq, bounds


def _solve_lp_optimal_prepared(
    *,
    objective: np.ndarray,
    prepared: tuple[
        Any | None,
        np.ndarray | None,
        Any | None,
        np.ndarray | None,
        np.ndarray,
    ],
    label: str,
    progress_interval: float,
    time_limit: float | None,
) -> SolveResult:
    from scipy.optimize import linprog

    a_ub, b_ub, a_eq, b_eq, bounds = prepared
    options: dict[str, Any] = {"disp": False}
    if time_limit is not None and time_limit > 0:
        options["time_limit"] = float(time_limit)

    started = time.perf_counter()
    result = _run_with_heartbeat(
        linprog,
        label=label,
        interval=progress_interval,
        c=np.asarray(objective, dtype=np.float64),
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options=options,
    )
    elapsed = time.perf_counter() - started
    if int(result.status) != 0 or result.x is None:
        raise RuntimeError(
            f"{label}未证明连续 LP 最优，不能作为理想下界："
            f"status={result.status}; message={result.message}"
        )
    vector = np.asarray(result.x, dtype=np.float64)
    return SolveResult(
        vector=vector,
        status=int(result.status),
        message=str(result.message),
        mip_gap=float("nan"),
        mip_dual_bound=float("nan"),
        objective_value=float(np.dot(objective, vector)),
        elapsed_seconds=elapsed,
        time_limit_used=float(time_limit or 0.0),
    )


def _solve_lp_optimal(
    model: AggregatedModel,
    objective: np.ndarray,
    label: str,
    progress_interval: float,
    time_limit: float | None,
) -> SolveResult:
    """兼容普通LP调用；全局四锚点应使用 prepared 版本复用矩阵切分。"""
    prepared = _prepare_lp_problem(model)
    return _solve_lp_optimal_prepared(
        objective=objective,
        prepared=prepared,
        label=label,
        progress_interval=progress_interval,
        time_limit=time_limit,
    )

def _extra_linear_constraint(
    n_variables: int,
    extra_rows: list[tuple[dict[int, float], float, float]] | None,
):
    """把少量附加约束单独构造，避免 vstack 复制整张主矩阵。"""
    if not extra_rows:
        return None
    from scipy.optimize import LinearConstraint

    rows = [row for row, _, _ in extra_rows]
    matrix = _sparse_from_rows(rows, n_variables)
    lower = np.asarray([low for _, low, _ in extra_rows], dtype=np.float64)
    upper = np.asarray([high for _, _, high in extra_rows], dtype=np.float64)
    return LinearConstraint(matrix, lower, upper)


def _solve_milp_adaptive(
    model: AggregatedModel,
    objective: np.ndarray,
    *,
    label: str,
    mip_rel_gap: float,
    initial_time_limit: float,
    max_time_limit: float,
    progress_interval: float,
    extra_rows: list[tuple[dict[int, float], float, float]] | None = None,
    variable_upper: np.ndarray | None = None,
    allow_infeasible: bool = False,
    retry_emergency: bool = True,
) -> SolveResult:
    """单次 HiGHS MILP 求解。

    原版 180→360→720... 会在 SciPy ``milp`` 中从头重启，不能复用上一轮
    分支树；8GB版直接给最终保护限时，HiGHS 一旦达到目标 gap 会自行提前停止。
    因此数学精度不变，同时避免重复分支定界和重复内存分配。
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    ub = (
        model.variable_upper
        if variable_upper is None
        else np.asarray(variable_upper, dtype=np.float64)
    )
    main_constraint = LinearConstraint(
        model.matrix, model.lower, model.upper
    )
    extra_constraint = _extra_linear_constraint(
        len(model.variable_lower), extra_rows
    )
    constraints = (
        [main_constraint, extra_constraint]
        if extra_constraint is not None
        else main_constraint
    )

    # max_time_limit 只是保护上限；达到 mip_rel_gap 时 HiGHS 会提前结束。
    used_limit = float(max(max_time_limit, initial_time_limit))
    _memory_guard(f"{label}-MILP求解前", hard=True)
    logging.info(
        "%s开始：变量=%d，整数计数变量=%d，约束=%d，保护限时=%.0fs，gap=%.4g。",
        label,
        len(model.variable_lower),
        int(np.sum(model.integrality[: model.option_count] == 1)),
        model.matrix.shape[0] + (len(extra_rows) if extra_rows else 0),
        used_limit,
        mip_rel_gap,
    )
    started = time.perf_counter()
    while True:
        result = _run_with_heartbeat(
            milp,
            label=label,
            interval=progress_interval,
            c=np.asarray(objective, dtype=np.float64),
            integrality=model.integrality,
            bounds=Bounds(model.variable_lower, ub),
            constraints=constraints,
            options={
                "disp": False,
                "mip_rel_gap": float(mip_rel_gap),
                "time_limit": used_limit,
            },
        )
        if result.x is not None or int(result.status) != 1:
            break
        if (not retry_emergency) or used_limit >= EMERGENCY_SOLVER_TIME_LIMIT:
            break
        logging.warning(
            "%s在%.0fs限时内没有返回整数可行解；按文档自动延长到%.0fs重试。",
            label,
            used_limit,
            EMERGENCY_SOLVER_TIME_LIMIT,
        )
        used_limit = EMERGENCY_SOLVER_TIME_LIMIT
    elapsed = time.perf_counter() - started
    gap = _result_float(result, "mip_gap")

    if result.x is None:
        if allow_infeasible and int(result.status) == 2:
            return SolveResult(
                vector=np.full(len(model.variable_lower), np.nan),
                status=int(result.status),
                message=str(result.message),
                mip_gap=gap,
                mip_dual_bound=_result_float(result, "mip_dual_bound"),
                objective_value=float("nan"),
                elapsed_seconds=elapsed,
                time_limit_used=used_limit,
            )
        raise RuntimeError(
            f"{label}没有返回可行解：status={result.status}; message={result.message}"
        )

    if int(result.status) not in (0, 1):
        raise RuntimeError(
            f"{label}失败：status={result.status}; message={result.message}"
        )

    vector = np.asarray(result.x, dtype=np.float64)
    integer_mask = model.integrality[: model.option_count].astype(bool)
    if integer_mask.any():
        values = vector[: model.option_count][integer_mask]
        if np.max(np.abs(values - np.rint(values))) > INTEGER_TOL:
            raise RuntimeError(f"{label}返回的 H 区计数变量未形成整数解")

    if int(result.status) == 1 and not (
        math.isfinite(gap) and gap <= mip_rel_gap + FLOAT_EPS
    ):
        logging.warning(
            "%s达到保护限时，保留当前可行解；mip_gap=%s，目标gap=%g。",
            label,
            gap,
            mip_rel_gap,
        )

    return SolveResult(
        vector=vector,
        status=int(result.status),
        message=str(result.message),
        mip_gap=gap,
        mip_dual_bound=_result_float(result, "mip_dual_bound"),
        objective_value=float(np.dot(objective, vector)),
        elapsed_seconds=elapsed,
        time_limit_used=used_limit,
    )



def _require_highspy():
    """阶段4需要原生 highspy；阶段2/3缓存仍与原版本完全兼容。"""
    try:
        import highspy  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "阶段4原生HiGHS加速需要 highspy>=1.8。"
            "请在当前项目环境执行 `uv add highspy` 或 `uv pip install highspy`，"
            "然后使用同一条 --resume 命令继续；已有阶段2/3及阶段4检查点不会丢失。"
        ) from exc
    return highspy


def _to_highs_bounds(values: np.ndarray, highspy: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    arr[np.isposinf(arr)] = float(highspy.kHighsInf)
    arr[np.isneginf(arr)] = -float(highspy.kHighsInf)
    return arr


def _highs_status_is_error(status: Any, highspy: Any) -> bool:
    try:
        return status == highspy.HighsStatus.kError
    except Exception:
        return False


def _solve_milp_highspy(
    model: AggregatedModel,
    objective: np.ndarray,
    *,
    label: str,
    mip_rel_gap: float,
    initial_time_limit: float,
    max_time_limit: float,
    progress_interval: float,
    extra_rows: list[tuple[dict[int, float], float, float]] | None = None,
    variable_upper: np.ndarray | None = None,
    mip_start_indices: np.ndarray | None = None,
    mip_start_values: np.ndarray | None = None,
    allow_infeasible: bool = False,
) -> SolveResult:
    """用 highspy 原生 HiGHS 求解 MILP，并支持稀疏 partial MIP Start。

    数学模型、变量边界、整数性与 SciPy milp 版本完全一致。区别仅在求解接口：
    - 直接把 CSC 稀疏矩阵传给 HiGHS；
    - 可用 setSolution(num_entries, index, value) 注入上一窗口的部分解；
    - mip_max_start_nodes 控制补全 partial MIP start 的额外搜索成本。
    """
    highspy = _require_highspy()

    ub = (
        model.variable_upper
        if variable_upper is None
        else np.asarray(variable_upper, dtype=np.float64)
    )
    n_variables = len(model.variable_lower)
    matrix = model.matrix.tocsc(copy=False)
    matrix.sort_indices()

    lp = highspy.HighsLp()
    lp.num_col_ = int(n_variables)
    lp.num_row_ = int(matrix.shape[0])
    lp.col_cost_ = np.asarray(objective, dtype=np.float64)
    lp.col_lower_ = _to_highs_bounds(model.variable_lower, highspy)
    lp.col_upper_ = _to_highs_bounds(ub, highspy)
    lp.row_lower_ = _to_highs_bounds(model.lower, highspy)
    lp.row_upper_ = _to_highs_bounds(model.upper, highspy)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.num_col_ = int(matrix.shape[1])
    lp.a_matrix_.num_row_ = int(matrix.shape[0])
    lp.a_matrix_.start_ = np.asarray(matrix.indptr, dtype=np.int64)
    lp.a_matrix_.index_ = np.asarray(matrix.indices, dtype=np.int32)
    lp.a_matrix_.value_ = np.asarray(matrix.data, dtype=np.float64)
    lp.integrality_ = [
        highspy.HighsVarType.kInteger
        if int(value) != 0
        else highspy.HighsVarType.kContinuous
        for value in model.integrality
    ]

    highs = highspy.Highs()
    status = highs.passModel(lp)
    if _highs_status_is_error(status, highspy):
        raise RuntimeError(f"{label}向原生HiGHS传递模型失败：status={status}")

    # min-max 的少量附加目标约束直接 addRows，避免 vstack 复制整张主矩阵。
    if extra_rows:
        extra_matrix = _sparse_from_rows(
            [row for row, _, _ in extra_rows], n_variables
        ).tocsr()
        extra_lower = _to_highs_bounds(
            np.asarray([low for _, low, _ in extra_rows], dtype=np.float64),
            highspy,
        )
        extra_upper = _to_highs_bounds(
            np.asarray([high for _, _, high in extra_rows], dtype=np.float64),
            highspy,
        )
        add_status = highs.addRows(
            int(extra_matrix.shape[0]),
            extra_lower,
            extra_upper,
            int(extra_matrix.nnz),
            np.asarray(extra_matrix.indptr[:-1], dtype=np.int64),
            np.asarray(extra_matrix.indices, dtype=np.int32),
            np.asarray(extra_matrix.data, dtype=np.float64),
        )
        if _highs_status_is_error(add_status, highspy):
            raise RuntimeError(f"{label}向原生HiGHS追加min-max约束失败：status={add_status}")
        del extra_matrix, extra_lower, extra_upper

    used_limit = float(max(max_time_limit, initial_time_limit))
    option_values = {
        "output_flag": False,
        "log_to_console": False,
        "presolve": "on",
        "time_limit": used_limit,
        "mip_rel_gap": float(mip_rel_gap),
        "mip_max_start_nodes": int(HIGHS_MIP_MAX_START_NODES),
        "mip_heuristic_effort": 0.20,
        "parallel": HIGHS_PARALLEL,
        "threads": int(HIGHS_THREADS),
    }
    for option_name, option_value in option_values.items():
        option_status = highs.setOptionValue(option_name, option_value)
        if _highs_status_is_error(option_status, highspy):
            raise RuntimeError(
                f"{label}设置HiGHS参数失败：{option_name}={option_value!r}"
            )

    start_count = 0
    if mip_start_indices is not None and mip_start_values is not None:
        start_idx = np.asarray(mip_start_indices, dtype=np.int32)
        start_val = np.asarray(mip_start_values, dtype=np.float64)
        if len(start_idx) != len(start_val):
            raise ValueError("MIP Start索引和值长度不一致")
        valid = (
            (start_idx >= 0)
            & (start_idx < n_variables)
            & np.isfinite(start_val)
        )
        start_idx = start_idx[valid]
        start_val = start_val[valid]
        if len(start_idx):
            start_status = highs.setSolution(
                int(len(start_idx)),
                start_idx,
                start_val,
            )
            if _highs_status_is_error(start_status, highspy):
                logging.warning(
                    "%s的MIP Start被HiGHS拒绝，自动无热启动继续；条目=%d，status=%s。",
                    label, len(start_idx), start_status,
                )
            else:
                start_count = int(len(start_idx))
                logging.info(
                    "%s注入稀疏MIP Start：变量条目=%d，补全节点上限=%d。",
                    label, start_count, HIGHS_MIP_MAX_START_NODES,
                )

    _memory_guard(f"{label}-原生HiGHS求解前", hard=True)
    logging.info(
        "%s开始[highspy]：变量=%d，整数计数变量=%d，约束=%d，"
        "MIPStart=%d，保护限时=%.0fs，gap=%.4g。",
        label,
        n_variables,
        int(np.sum(model.integrality[: model.option_count] == 1)),
        int(matrix.shape[0] + (len(extra_rows) if extra_rows else 0)),
        start_count,
        used_limit,
        mip_rel_gap,
    )

    started = time.perf_counter()
    run_status = _run_with_heartbeat(
        highs.run,
        label=label,
        interval=progress_interval,
    )
    elapsed = time.perf_counter() - started
    if _highs_status_is_error(run_status, highspy):
        raise RuntimeError(f"{label}原生HiGHS运行失败：status={run_status}")

    model_status = highs.getModelStatus()
    status_text = str(highs.modelStatusToString(model_status))
    info = highs.getInfo()
    solution = highs.getSolution()
    primal_status_text = str(
        highs.solutionStatusToString(info.primal_solution_status)
    ).strip().lower()
    has_feasible_solution = (
        primal_status_text == "feasible"
        and len(solution.col_value) == n_variables
    )

    gap = _result_float(info, "mip_gap")
    dual_bound = _result_float(info, "mip_dual_bound")

    optimal_status = getattr(highspy.HighsModelStatus, "kOptimal", None)
    time_status = getattr(highspy.HighsModelStatus, "kTimeLimit", None)
    infeasible_status = getattr(highspy.HighsModelStatus, "kInfeasible", None)
    limit_statuses = {
        value
        for value in (
            time_status,
            getattr(highspy.HighsModelStatus, "kIterationLimit", None),
            getattr(highspy.HighsModelStatus, "kSolutionLimit", None),
            getattr(highspy.HighsModelStatus, "kObjectiveBound", None),
            getattr(highspy.HighsModelStatus, "kObjectiveTarget", None),
        )
        if value is not None
    }

    if model_status == infeasible_status:
        if allow_infeasible:
            return SolveResult(
                vector=np.full(n_variables, np.nan),
                status=2,
                message=status_text,
                mip_gap=gap,
                mip_dual_bound=dual_bound,
                objective_value=float("nan"),
                elapsed_seconds=elapsed,
                time_limit_used=used_limit,
            )
        raise RuntimeError(f"{label}被HiGHS证明不可行：{status_text}")

    if not has_feasible_solution:
        raise RuntimeError(
            f"{label}没有返回整数可行解：model_status={status_text}; "
            f"primal_status={primal_status_text}"
        )

    vector = np.asarray(list(solution.col_value), dtype=np.float64)
    integer_mask = model.integrality[: model.option_count].astype(bool)
    if integer_mask.any():
        values = vector[: model.option_count][integer_mask]
        if np.max(np.abs(values - np.rint(values))) > INTEGER_TOL:
            raise RuntimeError(f"{label}返回的H区计数变量未形成整数解")

    if model_status == optimal_status:
        scipy_like_status = 0
    elif model_status in limit_statuses:
        scipy_like_status = 1
    else:
        # 某些HiGHS版本可能用其他“有可行解但提前终止”状态；按限时解处理，
        # 后续仍由 balanced 的 gap 接受阈值决定是否允许提交。
        scipy_like_status = 1

    if scipy_like_status == 1 and not (
        math.isfinite(gap) and gap <= mip_rel_gap + FLOAT_EPS
    ):
        logging.warning(
            "%s原生HiGHS提前终止，保留当前可行解；model_status=%s，"
            "mip_gap=%s，目标gap=%g。",
            label, status_text, gap, mip_rel_gap,
        )

    objective_value = float(np.dot(objective, vector))
    del highs, lp
    gc.collect()
    return SolveResult(
        vector=vector,
        status=scipy_like_status,
        message=status_text,
        mip_gap=gap,
        mip_dual_bound=dual_bound,
        objective_value=objective_value,
        elapsed_seconds=elapsed,
        time_limit_used=used_limit,
    )



def _baseline_partial_mip_start_for_model(
    *,
    model: AggregatedModel,
    baseline_assignments: pd.DataFrame,
    pools: dict[int, list[str]],
    existing_indices: np.ndarray | None = None,
    existing_values: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, int]]:
    """用纯算力基准为当前窗口构造稀疏 partial MIP Start。

    只映射“当前仍未提交”的 TaskID，并且只使用当前模型中真实存在的
    (class, region, start) 变量。若上一窗口 K 区热启动已覆盖某任务类，
    则该类优先使用上一窗口信息，基准不再重复注入，避免类等式冲突。

    这不会固定任何变量；HiGHS 仍可完全修改该起始方案。
    """
    stats = {
        "BaselineMappedEntries": 0,
        "BaselineMappedTasks": 0,
        "ExistingCoveredClasses": 0,
    }
    option_index = {
        (int(option.class_id), str(option.target_region), int(option.start_hour)): j
        for j, option in enumerate(model.options)
    }

    covered_classes: set[int] = set()
    indices: list[int] = []
    values: list[float] = []
    if existing_indices is not None and existing_values is not None:
        for idx, value in zip(
            np.asarray(existing_indices, dtype=np.int64),
            np.asarray(existing_values, dtype=np.float64),
        ):
            if 0 <= int(idx) < model.option_count and math.isfinite(float(value)):
                indices.append(int(idx))
                values.append(float(value))
                covered_classes.add(int(model.options[int(idx)].class_id))
    stats["ExistingCoveredClasses"] = len(covered_classes)

    remaining_ids: set[str] = set()
    for class_id in model.class_ids:
        if int(class_id) in covered_classes:
            continue
        remaining_ids.update(str(x) for x in pools[int(class_id)])

    if remaining_ids:
        accum: dict[int, float] = {}
        mapped_tasks = 0
        for row in baseline_assignments.itertuples(index=False):
            task_id = str(row.TaskID)
            if task_id not in remaining_ids:
                continue
            class_id = int(row.ExactTaskClassID)
            if class_id in covered_classes:
                continue
            key = (
                class_id,
                str(row.TargetRegion),
                int(round(float(row.StartHour))),
            )
            j = option_index.get(key)
            if j is None:
                continue
            accum[j] = accum.get(j, 0.0) + 1.0
            mapped_tasks += 1

        for j, value in accum.items():
            upper = float(model.variable_upper[j])
            value = min(float(value), upper)
            if value <= BALANCED_WARMSTART_POSITIVE_TOL:
                continue
            # H 区必须给整数计数；基准映射本身就是整数任务计数。
            if model.integrality[j]:
                value = float(round(value))
            indices.append(int(j))
            values.append(float(value))

        stats["BaselineMappedEntries"] = len(accum)
        stats["BaselineMappedTasks"] = mapped_tasks

    if not indices:
        return None, None, stats

    # 相同变量若意外重复，只保留最后一个；正常情况下不会发生。
    merged: dict[int, float] = {}
    for idx, value in zip(indices, values):
        merged[int(idx)] = float(value)
    out_idx = np.asarray(sorted(merged), dtype=np.int32)
    out_val = np.asarray([merged[int(i)] for i in out_idx], dtype=np.float64)
    return out_idx, out_val, stats


def _balanced_warmstart_path() -> Path:
    return CHECKPOINT_DIR / (
        f"q2_refactored_balanced_v{CHECKPOINT_SCHEMA_VERSION}"
        f"_mipstart_v{BALANCED_WARMSTART_SCHEMA_VERSION}.npz"
    )


def _save_balanced_warmstart(
    *,
    bundle: InputBundle,
    model: AggregatedModel,
    vector: np.ndarray,
    source_window_id: int,
) -> None:
    """保存当前窗口 K 区的非零任务变量，供下一窗口作为 partial MIP Start。

    K 区在当前窗口是连续前瞻；下一窗口中其前24小时会进入整数H区。
    保存时不做取整，加载到下一窗口时：
    - 对已经进入H区的变量，仅注入数值上已接近整数的条目；
    - 对仍处于K区的连续变量，可直接注入原连续值。
    这不会固定任何决策，只给HiGHS一个可修复的起点。
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    region_code = {region: i for i, region in enumerate(bundle.regions)}

    class_ids: list[int] = []
    region_codes: list[int] = []
    starts: list[int] = []
    values: list[float] = []
    for j, option in enumerate(model.options):
        if option.start_hour < model.decision_end:
            continue
        value = float(vector[j])
        if not math.isfinite(value) or value <= BALANCED_WARMSTART_POSITIVE_TOL:
            continue
        class_ids.append(int(option.class_id))
        region_codes.append(int(region_code[option.target_region]))
        starts.append(int(option.start_hour))
        values.append(value)

    path = _balanced_warmstart_path()
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        schema=np.asarray([BALANCED_WARMSTART_SCHEMA_VERSION], dtype=np.int16),
        checkpoint_schema=np.asarray([CHECKPOINT_SCHEMA_VERSION], dtype=np.int16),
        signature=np.asarray([_task_signature(bundle)]),
        source_window_id=np.asarray([source_window_id], dtype=np.int32),
        target_tau=np.asarray([model.decision_end], dtype=np.int32),
        source_plan_end=np.asarray([model.objective_end], dtype=np.int32),
        class_id=np.asarray(class_ids, dtype=np.int32),
        region_code=np.asarray(region_codes, dtype=np.uint8),
        start=np.asarray(starts, dtype=np.int16),
        value=np.asarray(values, dtype=np.float64),
    )
    tmp.replace(path)
    logging.info(
        "均衡窗口%d已保存下一窗口MIP Start候选：K区非零条目=%d，目标tau=%d。",
        source_window_id, len(values), model.decision_end,
    )


def _load_balanced_warmstart_for_model(
    *,
    bundle: InputBundle,
    model: AggregatedModel,
    window_id: int,
    tau: int,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, int]]:
    """把上一窗口 K 区解映射到当前模型变量索引。

    主检查点不依赖此文件；文件缺失/陈旧/损坏时只是不热启动，不影响续算。
    """
    stats = {
        "SavedEntries": 0,
        "MappedEntries": 0,
        "MappedIntegerEntries": 0,
        "MappedContinuousEntries": 0,
        "SkippedFractionalIntegerEntries": 0,
    }
    path = _balanced_warmstart_path()
    if not path.is_file():
        return None, None, stats

    try:
        data = np.load(path, allow_pickle=False)
        if int(data["schema"][0]) != BALANCED_WARMSTART_SCHEMA_VERSION:
            return None, None, stats
        if int(data["checkpoint_schema"][0]) != CHECKPOINT_SCHEMA_VERSION:
            return None, None, stats
        if str(data["signature"][0]) != _task_signature(bundle):
            return None, None, stats
        if int(data["source_window_id"][0]) != window_id - 1:
            logging.info(
                "当前MIP Start来自窗口%d，而当前需要窗口%d；忽略陈旧热启动文件。",
                int(data["source_window_id"][0]), window_id - 1,
            )
            return None, None, stats
        if int(data["target_tau"][0]) != tau:
            logging.info(
                "当前MIP Start目标tau=%d，与当前tau=%d不一致；忽略。",
                int(data["target_tau"][0]), tau,
            )
            return None, None, stats

        class_ids = data["class_id"].astype(np.int64, copy=False)
        region_codes = data["region_code"].astype(np.int64, copy=False)
        starts = data["start"].astype(np.int64, copy=False)
        saved_values = data["value"].astype(np.float64, copy=False)
        stats["SavedEntries"] = int(len(saved_values))

        saved: dict[tuple[int, str, int], float] = {}
        for cid, rcode, start, value in zip(
            class_ids, region_codes, starts, saved_values
        ):
            if not math.isfinite(float(value)) or float(value) <= BALANCED_WARMSTART_POSITIVE_TOL:
                continue
            region = bundle.regions[int(rcode)]
            saved[(int(cid), region, int(start))] = float(value)

        indices: list[int] = []
        values: list[float] = []
        for j, option in enumerate(model.options):
            value = saved.get(
                (int(option.class_id), option.target_region, int(option.start_hour))
            )
            if value is None:
                continue
            upper = float(model.variable_upper[j])
            if model.integrality[j]:
                rounded = float(round(value))
                if (
                    abs(value - rounded) > BALANCED_WARMSTART_VALUE_TOL
                    or rounded <= 0.0
                    or rounded > upper + INTEGER_TOL
                ):
                    stats["SkippedFractionalIntegerEntries"] += 1
                    continue
                value_to_use = rounded
                stats["MappedIntegerEntries"] += 1
            else:
                value_to_use = min(max(float(value), 0.0), upper)
                if value_to_use <= BALANCED_WARMSTART_POSITIVE_TOL:
                    continue
                stats["MappedContinuousEntries"] += 1
            indices.append(j)
            values.append(value_to_use)

        stats["MappedEntries"] = len(indices)
        if not indices:
            return None, None, stats
        return (
            np.asarray(indices, dtype=np.int32),
            np.asarray(values, dtype=np.float64),
            stats,
        )
    except Exception as exc:
        logging.warning("均衡MIP Start读取/映射失败，忽略热启动并正常求解：%s", exc)
        return None, None, stats



# =============================================================================
# 7. TaskID 池、计数提交和全时域复算
# =============================================================================

def _make_task_pools(classes: list[ExactTaskClass]) -> dict[int, list[str]]:
    return {task_class.class_id: list(task_class.task_ids) for task_class in classes}


def _remove_committed_from_pools(
    pools: dict[int, list[str]],
    committed: pd.DataFrame,
) -> None:
    if committed.empty:
        return
    if "ExactTaskClassID" not in committed.columns:
        raise ValueError("检查点缺少 ExactTaskClassID，无法恢复聚合任务池")
    grouped = committed.groupby("ExactTaskClassID")["TaskID"].apply(list)
    for class_id_raw, task_ids in grouped.items():
        class_id = int(class_id_raw)
        existing = set(pools[class_id])
        remove = set(str(x) for x in task_ids)
        if not remove.issubset(existing):
            raise RuntimeError(f"检查点中任务类{class_id}包含无法恢复的 TaskID")
        pools[class_id] = [task_id for task_id in pools[class_id] if task_id not in remove]


def _remaining_count(pools: dict[int, list[str]]) -> dict[int, int]:
    return {class_id: len(task_ids) for class_id, task_ids in pools.items()}


def _commit_h_counts(
    *,
    model: AggregatedModel,
    vector: np.ndarray,
    class_lookup: dict[int, ExactTaskClass],
    pools: dict[int, list[str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for j, option in enumerate(model.options):
        if option.start_hour >= model.decision_end:
            continue
        count = int(round(float(vector[j])))
        if count <= 0:
            continue
        pool = pools[option.class_id]
        if len(pool) < count:
            raise RuntimeError(
                f"任务类{option.class_id}TaskID不足：需要{count}，剩余{len(pool)}"
            )
        chosen = pool[:count]
        del pool[:count]
        task_class = class_lookup[option.class_id]
        migrated = option.target_region != task_class.source_region
        for task_id in chosen:
            rows.append(
                {
                    "TaskID": task_id,
                    "TaskType": task_class.task_type,
                    "ArrivalHour": task_class.arrival_hour,
                    "SourceRegion": task_class.source_region,
                    "TargetRegion": option.target_region,
                    "NetworkLatency_ms": option.network_latency_ms,
                    "MaxLatency_ms": task_class.max_latency_ms,
                    "StartHour": option.start_hour,
                    "FinishHour": option.finish_hour,
                    "Duration_h": task_class.duration_h,
                    "GPU_Demand": task_class.gpu_demand,
                    "Task_Full_IT_Power_MW": task_class.task_power_mw,
                    "WaitHours": (
                        0.0
                        if task_class.task_type == "RealTimeInference"
                        else max(0.0, float(option.start_hour - task_class.earliest_start_hour))
                    ),
                    "Migration_GPU_Workload_GPUh": (
                        task_class.gpu_demand * task_class.duration_h if migrated else 0.0
                    ),
                    "IsMigrated": int(migrated),
                    "ExactTaskClassID": option.class_id,
                }
            )
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    result["TaskID_num"] = pd.to_numeric(result["TaskID"], errors="coerce")
    return (
        result.sort_values(["TaskID_num", "TaskID"], kind="stable")
        .drop(columns="TaskID_num")
        .reset_index(drop=True)
    )


def _schedule_profile(
    bundle: InputBundle,
    assignments: pd.DataFrame,
    hour_start: int = MAIN_START_HOUR,
    hour_end: int = TERMINAL_HOUR,
) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = bundle.region_hour.loc[
        bundle.region_hour["Hour"].between(hour_start, hour_end - 1)
    ].copy()
    frame = frame.sort_values(["Hour", "Region"], kind="stable").reset_index(drop=True)
    keys = [(int(row.Hour), str(row.Region)) for row in frame.itertuples(index=False)]
    key_to_index = {key: index for index, key in enumerate(keys)}
    scheduled_gpu = np.zeros(len(frame), dtype=float)
    scheduled_ai = np.zeros(len(frame), dtype=float)

    if not assignments.empty:
        for row in assignments.itertuples(index=False):
            for hour, overlap in _overlap_by_hour(float(row.StartHour), float(row.Duration_h)):
                index = key_to_index.get((hour, str(row.TargetRegion)))
                if index is None:
                    continue
                scheduled_gpu[index] += float(row.GPU_Demand) * overlap
                scheduled_ai[index] += float(row.Task_Full_IT_Power_MW) * overlap

    frame["Scheduled_GPU_Equivalent"] = scheduled_gpu
    frame["Scheduled_AI_IT_Load_MW"] = scheduled_ai
    frame["Total_IT_Load_MW"] = frame["NonAI_IT_Load_MW"] + frame["Scheduled_AI_IT_Load_MW"]
    frame["Total_Facility_Load_MW"] = frame["Total_IT_Load_MW"] * frame["PUE"]
    frame["GPU_Slack"] = frame["Available_GPU"] - frame["Scheduled_GPU_Equivalent"]
    frame["IT_Slack_MW"] = frame["Max_IT_Power_MW"] - frame["Total_IT_Load_MW"]
    frame["Facility_Slack_MW"] = frame["Max_Facility_Power_MW"] - frame["Total_Facility_Load_MW"]
    frame["EffectiveAI_Slack_MW"] = (
        frame["Effective_AI_IT_Capacity_MW_Exact"] - frame["Scheduled_AI_IT_Load_MW"]
    )
    frame["GPU_Utilization"] = np.divide(
        frame["Scheduled_GPU_Equivalent"],
        frame["Available_GPU"],
        out=np.zeros(len(frame), dtype=float),
        where=frame["Available_GPU"].to_numpy(dtype=float) > 0,
    )
    frame["AI_IT_Capacity_Utilization"] = np.divide(
        frame["Scheduled_AI_IT_Load_MW"],
        frame["Effective_AI_IT_Capacity_MW_Exact"],
        out=np.zeros(len(frame), dtype=float),
        where=frame["Effective_AI_IT_Capacity_MW_Exact"].to_numpy(dtype=float) > FLOAT_EPS,
    )

    frame["RenewableDirectUse_MW"] = np.minimum(
        frame["Total_Facility_Load_MW"], frame["AvailableRenewable_MW"]
    )
    frame["GridPurchase_MW"] = np.maximum(
        frame["Total_Facility_Load_MW"] - frame["AvailableRenewable_MW"], 0.0
    )
    frame["RenewableExport_MW"] = np.minimum(
        np.maximum(frame["AvailableRenewable_MW"] - frame["Total_Facility_Load_MW"], 0.0),
        frame["ExportLimit_MW"],
    )
    frame["RenewableCurtailment_MW"] = (
        frame["AvailableRenewable_MW"]
        - frame["RenewableDirectUse_MW"]
        - frame["RenewableExport_MW"]
    )
    frame["GridPurchaseViolation_MW"] = np.maximum(
        frame["GridPurchase_MW"] - frame["MaxGridImport_MW"], 0.0
    )
    frame["OperatingCost_CNY"] = (
        frame["ElectricityPrice_CNY_per_MWh"] * frame["GridPurchase_MW"]
        - frame["SellPrice_CNY_per_MWh"] * frame["RenewableExport_MW"]
    )
    frame["CarbonEmission_tCO2"] = (
        frame["CarbonIntensity_tCO2_per_MWh"] * frame["GridPurchase_MW"]
    )
    frame["EnergyBalanceResidual_MW"] = (
        frame["AvailableRenewable_MW"]
        + frame["GridPurchase_MW"]
        - frame["Total_Facility_Load_MW"]
        - frame["RenewableExport_MW"]
        - frame["RenewableCurtailment_MW"]
    )

    global_renewable = float(
        bundle.region_hour["AvailableRenewable_MW"].sum()
    )
    curtailment_global_ratio = (
        float(frame["RenewableCurtailment_MW"].sum()) / global_renewable
        if global_renewable > FLOAT_EPS
        else float("nan")
    )
    local_renewable = float(frame["AvailableRenewable_MW"].sum())
    local_unused_rate = (
        float(frame["RenewableCurtailment_MW"].sum()) / local_renewable
        if local_renewable > FLOAT_EPS
        else float("nan")
    )
    if assignments.empty:
        mean_latency = float("nan")
        mean_wait = float("nan")
        migration_ratio = float("nan")
        migration_workload = 0.0
    else:
        mean_latency = float(assignments["NetworkLatency_ms"].astype(float).mean())
        mean_wait = float(assignments["WaitHours"].astype(float).mean())
        migration_ratio = float(assignments["IsMigrated"].astype(float).mean())
        migration_workload = float(assignments["Migration_GPU_Workload_GPUh"].astype(float).sum())

    metrics = {
        "Cost": float(frame["OperatingCost_CNY"].sum()),
        "Carbon": float(frame["CarbonEmission_tCO2"].sum()),
        "MeanLatency": mean_latency,
        "RenewableUnusedRate": local_unused_rate,
        "RenewableUnusedContributionGlobal": curtailment_global_ratio,
        "AverageWaitHours": mean_wait,
        "MigratedTaskRatio": migration_ratio,
        "MigrationGPUWorkload_GPUh": migration_workload,
        "TaskCount": float(len(assignments)),
        "MaxGPUUtilization": float(frame["GPU_Utilization"].max()) if not frame.empty else float("nan"),
        "MaxAIITCapacityUtilization": float(frame["AI_IT_Capacity_Utilization"].max()) if not frame.empty else float("nan"),
        "MinGPUSlack": float(frame["GPU_Slack"].min()) if not frame.empty else float("nan"),
        "MinITSlack_MW": float(frame["IT_Slack_MW"].min()) if not frame.empty else float("nan"),
        "MinFacilitySlack_MW": float(frame["Facility_Slack_MW"].min()) if not frame.empty else float("nan"),
        "MinEffectiveAISlack_MW": float(frame["EffectiveAI_Slack_MW"].min()) if not frame.empty else float("nan"),
        "MaxGridPurchaseViolation_MW": float(frame["GridPurchaseViolation_MW"].max()) if not frame.empty else float("nan"),
        "MaxEnergyBalanceResidual_MW": float(frame["EnergyBalanceResidual_MW"].abs().max()) if not frame.empty else float("nan"),
    }
    return frame, metrics


def _energy_contribution(
    bundle: InputBundle,
    assignments: pd.DataFrame,
    hour_start: int,
    hour_end: int,
) -> dict[str, float]:
    if hour_end <= hour_start:
        return {"Cost": 0.0, "Carbon": 0.0, "RenewableUnusedRate": 0.0}
    profile, _ = _schedule_profile(bundle, assignments, hour_start, hour_end)
    global_renewable = float(bundle.region_hour["AvailableRenewable_MW"].sum())
    return {
        "Cost": float(profile["OperatingCost_CNY"].sum()),
        "Carbon": float(profile["CarbonEmission_tCO2"].sum()),
        "RenewableUnusedRate": (
            float(profile["RenewableCurtailment_MW"].sum()) / global_renewable
            if global_renewable > FLOAT_EPS
            else 0.0
        ),
    }


# =============================================================================
# 8. 聚合纯算力基准：全题只求一次
# =============================================================================


def _baseline_analytic_zero_migration_earliest(
    model: AggregatedModel,
    class_lookup: dict[int, ExactTaskClass],
) -> SolveResult | None:
    """若“全部本地 + 当前模型内最早合法启动”已可行，则直接得到字典序全局最优。

    一级迁移GPU工作量的理论下界是0；在保持0迁移条件下，每类任务选择最早
    合法时刻又逐项最小化等待时间。因此若该构造满足全部容量约束，无需调用MILP。
    """
    vector = np.zeros(len(model.variable_lower), dtype=np.float64)
    by_class: dict[int, list[int]] = {}
    for j, option in enumerate(model.options):
        task_class = class_lookup[option.class_id]
        if option.target_region == task_class.source_region:
            by_class.setdefault(option.class_id, []).append(j)

    for row_idx, class_id in enumerate(model.class_ids):
        candidates = by_class.get(class_id)
        if not candidates:
            return None
        earliest = min(model.options[j].start_hour for j in candidates)
        chosen = next(
            j for j in candidates if model.options[j].start_hour == earliest
        )
        vector[chosen] = float(model.lower[row_idx])

    lhs = np.asarray(model.matrix @ vector, dtype=np.float64).ravel()
    finite_lo = np.isfinite(model.lower)
    finite_hi = np.isfinite(model.upper)
    if finite_lo.any() and np.any(lhs[finite_lo] < model.lower[finite_lo] - 1e-7):
        return None
    if finite_hi.any() and np.any(lhs[finite_hi] > model.upper[finite_hi] + 1e-7):
        return None

    return SolveResult(
        vector=vector,
        status=0,
        message="analytic_zero_migration_earliest_start_optimum",
        mip_gap=0.0,
        mip_dual_bound=0.0,
        objective_value=float(np.dot(model.wait_objective, vector)),
        elapsed_seconds=0.0,
        time_limit_used=0.0,
    )

def _solve_baseline_window(
    model: AggregatedModel,
    *,
    class_lookup: dict[int, ExactTaskClass],
    mip_rel_gap: float,
    initial_time_limit: float,
    max_time_limit: float,
    progress_interval: float,
    window_id: int,
) -> tuple[SolveResult, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []

    analytic = _baseline_analytic_zero_migration_earliest(model, class_lookup)
    if analytic is not None:
        f2 = float(np.dot(model.wait_objective, analytic.vector))
        records.append(
            {
                "Mode": "baseline",
                "WindowID": window_id,
                "Stage": "baseline_analytic_zero_migration_earliest",
                "Status": 0,
                "Message": analytic.message,
                "MIPGap": 0.0,
                "MIPDualBound": 0.0,
                "ObjectiveValue": f2,
                "ElapsedSeconds": 0.0,
                "TimeLimitUsedSeconds": 0.0,
            }
        )
        logging.info(
            "基准窗口%d解析快路命中：零迁移且最早启动已满足全部容量约束，跳过2次MILP。",
            window_id,
        )
        return analytic, records

    stage1 = _solve_milp_adaptive(
        model,
        model.migration_objective,
        label=f"基准窗口{window_id}-一级迁移GPU工作量",
        mip_rel_gap=mip_rel_gap,
        initial_time_limit=initial_time_limit,
        max_time_limit=max_time_limit,
        progress_interval=progress_interval,
    )
    f1 = float(np.dot(model.migration_objective, stage1.vector))
    records.append(
        {
            "Mode": "baseline",
            "WindowID": window_id,
            "Stage": "baseline_migration",
            "Status": stage1.status,
            "Message": stage1.message,
            "MIPGap": stage1.mip_gap,
            "MIPDualBound": stage1.mip_dual_bound,
            "ObjectiveValue": f1,
            "ElapsedSeconds": stage1.elapsed_seconds,
            "TimeLimitUsedSeconds": stage1.time_limit_used,
        }
    )

    tol = max(1e-7, LEXICOGRAPHIC_REL_TOL * max(1.0, abs(f1)))
    f1_terms = {
        idx: float(value)
        for idx, value in enumerate(model.migration_objective)
        if abs(value) > FLOAT_EPS
    }
    stage2 = _solve_milp_adaptive(
        model,
        model.wait_objective,
        label=f"基准窗口{window_id}-二级等待时间",
        mip_rel_gap=mip_rel_gap,
        initial_time_limit=initial_time_limit,
        max_time_limit=max_time_limit,
        progress_interval=progress_interval,
        extra_rows=[(f1_terms, f1 - tol, f1 + tol)],
    )
    f2 = float(np.dot(model.wait_objective, stage2.vector))
    records.append(
        {
            "Mode": "baseline",
            "WindowID": window_id,
            "Stage": "baseline_wait",
            "Status": stage2.status,
            "Message": stage2.message,
            "MIPGap": stage2.mip_gap,
            "MIPDualBound": stage2.mip_dual_bound,
            "ObjectiveValue": f2,
            "ElapsedSeconds": stage2.elapsed_seconds,
            "TimeLimitUsedSeconds": stage2.time_limit_used,
        }
    )
    return stage2, records

def _checkpoint_paths(mode: str) -> dict[str, Path]:
    prefix = f"q2_refactored_{mode}_v{CHECKPOINT_SCHEMA_VERSION}"
    return {
        "state": CHECKPOINT_DIR / f"{prefix}_state.json",
        "assignments": CHECKPOINT_DIR / f"{prefix}_assignments.csv",
        "windows": CHECKPOINT_DIR / f"{prefix}_windows.csv",
        "solver": CHECKPOINT_DIR / f"{prefix}_solver.csv",
    }


def _write_checkpoint(
    *,
    bundle: InputBundle,
    mode: str,
    assignments: pd.DataFrame,
    windows: list[dict[str, Any]],
    solver: list[dict[str, Any]],
    next_tau: int,
    next_window_id: int,
    status: str,
    decision_window: int,
    lookahead: int,
) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    paths = _checkpoint_paths(mode)
    assignments.to_csv(paths["assignments"], index=False, encoding="utf-8-sig", float_format="%.15g")
    pd.DataFrame(windows).to_csv(paths["windows"], index=False, encoding="utf-8-sig", float_format="%.15g")
    pd.DataFrame(solver).to_csv(paths["solver"], index=False, encoding="utf-8-sig", float_format="%.15g")
    state = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "task_signature": _task_signature(bundle),
        "mode": mode,
        "status": status,
        "next_tau": int(next_tau),
        "next_window_id": int(next_window_id),
        "decision_window": int(decision_window),
        "lookahead": int(lookahead),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    paths["state"].write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_checkpoint(
    bundle: InputBundle,
    mode: str,
    decision_window: int,
    lookahead: int,
) -> dict[str, Any] | None:
    paths = _checkpoint_paths(mode)
    if not paths["state"].is_file():
        return None
    state = json.loads(paths["state"].read_text(encoding="utf-8"))
    if state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        return None
    if state.get("task_signature") != _task_signature(bundle):
        raise RuntimeError(f"{mode}检查点任务数据已变化，拒绝恢复")
    if int(state.get("decision_window", -1)) != decision_window or int(state.get("lookahead", -1)) != lookahead:
        raise RuntimeError(f"{mode}检查点的 H/K 与当前参数不同，拒绝恢复")

    def read_optional(path: Path) -> pd.DataFrame:
        if not path.is_file() or path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    return {
        "state": state,
        "assignments": read_optional(paths["assignments"]),
        "windows": read_optional(paths["windows"]).to_dict("records"),
        "solver": read_optional(paths["solver"]).to_dict("records"),
    }


def _run_baseline_rollout(
    *,
    bundle: InputBundle,
    classes: list[ExactTaskClass],
    class_lookup: dict[int, ExactTaskClass],
    decision_window: int,
    lookahead: int,
    mip_rel_gap: float,
    initial_time_limit: float,
    max_time_limit: float,
    progress_interval: float,
    resume: bool,
    max_windows: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    pools = _make_task_pools(classes)
    assignments = pd.DataFrame()
    window_records: list[dict[str, Any]] = []
    solver_records: list[dict[str, Any]] = []
    tau = 0
    window_id = 0

    if resume:
        checkpoint = _load_checkpoint(bundle, "baseline", decision_window, lookahead)
        if checkpoint is not None:
            assignments = checkpoint["assignments"]
            window_records = checkpoint["windows"]
            solver_records = checkpoint["solver"]
            tau = int(checkpoint["state"]["next_tau"])
            window_id = int(checkpoint["state"]["next_window_id"])
            _remove_committed_from_pools(pools, assignments)
            if checkpoint["state"].get("status") == "COMPLETED":
                profile, metrics = _schedule_profile(bundle, assignments)
                return assignments, pd.DataFrame(window_records), pd.DataFrame(solver_records), metrics
            logging.info("恢复纯算力基准：已完成任务=%d，下一窗口=%d，tau=%d。", len(assignments), window_id, tau)

    while any(pools.values()) and tau < TERMINAL_HOUR:
        if max_windows is not None and window_id >= max_windows:
            break
        decision_end = min(tau + decision_window, TERMINAL_HOUR)
        plan_end = min(decision_end + lookahead, TERMINAL_HOUR)
        remaining = _remaining_count(pools)
        try:
            model = _build_aggregated_model(
                bundle=bundle,
                classes=classes,
                class_lookup=class_lookup,
                remaining_count=remaining,
                fixed_assignments=assignments,
                tau=tau,
                decision_end=decision_end,
                plan_end=plan_end,
                include_z=False,
                include_energy_variables=False,
            )
        except ValueError as exc:
            if "没有进入当前模型" in str(exc):
                tau = decision_end
                window_id += 1
                continue
            raise

        started = time.perf_counter()
        result, local_solver = _solve_baseline_window(
            model,
            class_lookup=class_lookup,
            mip_rel_gap=mip_rel_gap,
            initial_time_limit=initial_time_limit,
            max_time_limit=max_time_limit,
            progress_interval=progress_interval,
            window_id=window_id,
        )
        committed = _commit_h_counts(
            model=model,
            vector=result.vector,
            class_lookup=class_lookup,
            pools=pools,
        )
        if not committed.empty:
            assignments = pd.concat([assignments, committed], ignore_index=True)
        remaining_after = sum(len(v) for v in pools.values())
        window_records.append(
            {
                "WindowID": window_id,
                "Mode": "baseline",
                "WindowStartHour": tau,
                "DecisionEndHourExclusive": decision_end,
                "PlanEndHourExclusive": plan_end,
                "ActiveClassCount": len(model.class_ids),
                "CandidateOptionCount": model.option_count,
                "DecisionIntegerOptionCount": int(np.sum(model.integrality[: model.option_count] == 1)),
                "LookaheadContinuousOptionCount": int(np.sum(model.integrality[: model.option_count] == 0)),
                "CommittedTaskCount": len(committed),
                "RemainingTaskCount": remaining_after,
                "ElapsedSeconds": time.perf_counter() - started,
            }
        )
        solver_records.extend(local_solver)
        candidate_count = model.option_count
        del result, model
        gc.collect()
        next_tau = decision_end
        _write_checkpoint(
            bundle=bundle,
            mode="baseline",
            assignments=assignments,
            windows=window_records,
            solver=solver_records,
            next_tau=next_tau,
            next_window_id=window_id + 1,
            status="RUNNING",
            decision_window=decision_window,
            lookahead=lookahead,
        )
        logging.info(
            "基准窗口%d完成：tau=%d，候选=%d，固定=%d，剩余=%d。",
            window_id, tau, candidate_count, len(committed), remaining_after,
        )
        tau = next_tau
        window_id += 1

    if any(pools.values()):
        if max_windows is not None:
            profile, metrics = _schedule_profile(bundle, assignments)
            return assignments, pd.DataFrame(window_records), pd.DataFrame(solver_records), metrics
        remaining_ids = [task_id for values in pools.values() for task_id in values]
        raise RuntimeError(f"纯算力基准滚动结束后仍有{len(remaining_ids)}个任务未固定")

    _write_checkpoint(
        bundle=bundle,
        mode="baseline",
        assignments=assignments,
        windows=window_records,
        solver=solver_records,
        next_tau=tau,
        next_window_id=window_id,
        status="COMPLETED",
        decision_window=decision_window,
        lookahead=lookahead,
    )
    profile, metrics = _schedule_profile(bundle, assignments)
    return assignments, pd.DataFrame(window_records), pd.DataFrame(solver_records), metrics


# =============================================================================
# 9. 全局连续理想点与统一定标
# =============================================================================



def _recompute_global_metrics_from_aggregated_vector(
    bundle: InputBundle,
    model: AggregatedModel,
    class_lookup: dict[int, ExactTaskClass],
    vector: np.ndarray,
) -> dict[str, float]:
    """仅由聚合任务变量重新做物理能源结算。

    全局连续理想点中，未进入当前单目标的 B/W 辅助变量可能存在多重最优，
    因而 payoff matrix 不能直接读取这些辅助变量；这里统一由 y 产生的设施负荷
    重新计算购电、外送与弃电，保证四个锚点的交叉指标具有确定物理含义。
    """

    frame = bundle.region_hour.copy().sort_values(["Hour", "Region"], kind="stable").reset_index(drop=True)
    keys = [(int(row.Hour), str(row.Region)) for row in frame.itertuples(index=False)]
    key_to_index = {key: idx for idx, key in enumerate(keys)}
    scheduled_ai = np.zeros(len(frame), dtype=float)
    latency_sum = 0.0
    for j, option in enumerate(model.options):
        weight = float(vector[j])
        if abs(weight) <= FLOAT_EPS:
            continue
        task_class = class_lookup[option.class_id]
        latency_sum += weight * option.network_latency_ms
        for hour, overlap in _overlap_by_hour(float(option.start_hour), task_class.duration_h):
            idx = key_to_index.get((int(hour), option.target_region))
            if idx is None:
                continue
            scheduled_ai[idx] += weight * task_class.task_power_mw * overlap

    facility = (frame["NonAI_IT_Load_MW"].to_numpy(dtype=float) + scheduled_ai) * frame["PUE"].to_numpy(dtype=float)
    renewable = frame["AvailableRenewable_MW"].to_numpy(dtype=float)
    export_limit = frame["ExportLimit_MW"].to_numpy(dtype=float)
    grid = np.maximum(facility - renewable, 0.0)
    export = np.minimum(np.maximum(renewable - facility, 0.0), export_limit)
    curtail = renewable - np.minimum(facility, renewable) - export
    cost = float(np.sum(frame["ElectricityPrice_CNY_per_MWh"].to_numpy(dtype=float) * grid - frame["SellPrice_CNY_per_MWh"].to_numpy(dtype=float) * export))
    carbon = float(np.sum(frame["CarbonIntensity_tCO2_per_MWh"].to_numpy(dtype=float) * grid))
    total_re = float(np.sum(renewable))
    return {
        "Cost": cost,
        "Carbon": carbon,
        "MeanLatency": latency_sum / max(model.global_task_count, 1),
        "RenewableUnusedRate": float(np.sum(curtail)) / total_re if total_re > FLOAT_EPS else float("nan"),
    }


def _calibration_cache_path() -> Path:
    return CHECKPOINT_DIR / f"q2_refactored_global_calibration_v{CALIBRATION_SCHEMA_VERSION}.json"



def _cg_seed_columns_from_baseline(
    baseline_assignments: pd.DataFrame,
    class_lookup: dict[int, ExactTaskClass],
) -> list[CGColumn]:
    """用已验证纯算力基准作为列生成受限主问题的可行初始列。"""
    required = {"ExactTaskClassID", "TargetRegion", "NetworkLatency_ms", "StartHour"}
    missing = required - set(baseline_assignments.columns)
    if missing:
        raise ValueError(f"纯算力基准缺少列生成初始列字段：{sorted(missing)}")
    columns: dict[tuple[int, str, int], CGColumn] = {}
    for row in baseline_assignments.itertuples(index=False):
        class_id = int(row.ExactTaskClassID)
        task_class = class_lookup[class_id]
        start = int(round(float(row.StartHour)))
        target = str(row.TargetRegion)
        key = (class_id, target, start)
        columns[key] = CGColumn(
            class_id=class_id,
            target_region=target,
            network_latency_ms=float(row.NetworkLatency_ms),
            start_hour=start,
            finish_hour=float(start + task_class.duration_h),
        )
    covered = {column.class_id for column in columns.values()}
    missing_classes = set(class_lookup) - covered
    if missing_classes:
        raise RuntimeError(
            f"纯算力基准未覆盖{len(missing_classes)}个任务类，无法作为全局LP列生成可行初始解"
        )
    return sorted(columns.values(), key=lambda c: c.key)


def _cg_full_resource_environment(bundle: InputBundle) -> dict[str, Any]:
    frame = bundle.region_hour.loc[
        bundle.region_hour["Hour"].between(MAIN_START_HOUR, TAIL_END_HOUR)
    ].copy()
    frame = frame.sort_values(["Hour", "Region"], kind="stable").reset_index(drop=True)
    region_array = frame["Region"].to_numpy(dtype=str)
    region_positions: dict[str, np.ndarray] = {}
    for region in bundle.regions:
        pos = np.flatnonzero(region_array == region).astype(np.int64, copy=False)
        hours = frame.loc[pos, "Hour"].to_numpy(dtype=np.int64)
        if not np.array_equal(hours, np.arange(MAIN_START_HOUR, TAIL_END_HOUR + 1)):
            raise RuntimeError(f"区域{region}全局列生成时域不完整")
        region_positions[region] = pos
    env = {
        "frame": frame,
        "region_positions": region_positions,
        "n_resource": len(frame),
        "global_renewable": float(frame["AvailableRenewable_MW"].sum()),
        "total_tasks": int(len(bundle.tasks)),
        "pue": frame["PUE"].to_numpy(dtype=np.float64),
        "sell": frame["SellPrice_CNY_per_MWh"].to_numpy(dtype=np.float64),
        "price": frame["ElectricityPrice_CNY_per_MWh"].to_numpy(dtype=np.float64),
        "carbon": frame["CarbonIntensity_tCO2_per_MWh"].to_numpy(dtype=np.float64),
        "renewable": frame["AvailableRenewable_MW"].to_numpy(dtype=np.float64),
        "non_ai": frame["NonAI_IT_Load_MW"].to_numpy(dtype=np.float64),
        "export_limit": frame["ExportLimit_MW"].to_numpy(dtype=np.float64),
        "available_gpu": frame["Available_GPU"].to_numpy(dtype=np.float64),
        "effective_ai": frame["Effective_AI_IT_Capacity_MW_Exact"].to_numpy(dtype=np.float64),
        "max_grid": frame["MaxGridImport_MW"].to_numpy(dtype=np.float64),
    }
    env["base_facility"] = env["non_ai"] * env["pue"]
    # 物理下界：即使每个时空单元都把 AI 功率推到有效容量上限，仍无法吸收的
    # 新能源也必然弃用。它不依赖任务分配，因此是 RenewableUnusedRate 的严格下界。
    max_facility = env["base_facility"] + env["pue"] * env["effective_ai"]
    unavoidable = np.maximum(
        env["renewable"] - max_facility - env["export_limit"], 0.0
    )
    env["renewable_physical_lb_rate"] = (
        float(np.sum(unavoidable)) / float(env["global_renewable"])
        if float(env["global_renewable"]) > FLOAT_EPS else 0.0
    )
    return env

def _cg_build_master(
    *,
    bundle: InputBundle,
    class_lookup: dict[int, ExactTaskClass],
    columns: list[CGColumn],
    metric: str,
    env: dict[str, Any],
):
    """构造低内存受限主LP；不同锚点只保留真正需要的能源上图变量/约束。

    Cost: GPU + AI + B + W
    Carbon: GPU + AI + B
    RenewableUnusedRate: GPU + AI + W
    MeanLatency: GPU + AI

    由于购电上限已严格并入 Effective_AI_IT_Capacity_MW_Exact，且题面能源结算
    满足单调性条件，这与完整 B/W/X 表达在 y 决策和最优值上等价。
    """
    from scipy.sparse import coo_array

    R = int(env["n_resource"])
    C = len(class_lookup)
    J = len(columns)

    if metric == "Cost":
        has_b, has_w = True, True
    elif metric == "Carbon":
        has_b, has_w = True, False
    elif metric == "RenewableUnusedRate":
        has_b, has_w = False, True
    elif metric == "MeanLatency":
        has_b, has_w = False, False
    else:
        raise ValueError(f"未知CG目标：{metric}")

    next_idx = J
    b_idx = None
    w_idx = None
    if has_b:
        b_idx = next_idx + np.arange(R, dtype=np.int64)
        next_idx += R
    if has_w:
        w_idx = next_idx + np.arange(R, dtype=np.int64)
        next_idx += R
    n_variables = next_idx

    # 行布局：GPU、AI，以及按目标需要的B/W下图。
    gpu_offset = 0
    ai_offset = R
    row_count = 2 * R
    b_offset = None
    w_offset = None
    if has_b:
        b_offset = row_count
        row_count += R
    if has_w:
        w_offset = row_count
        row_count += R

    rr_eq = array("i")
    cc_eq = array("i")
    vv_eq = array("d")
    rr_ub = array("i")
    cc_ub = array("i")
    vv_ub = array("d")

    b_ub = np.empty(row_count, dtype=np.float64)
    b_ub[gpu_offset:gpu_offset + R] = env["available_gpu"]
    b_ub[ai_offset:ai_offset + R] = env["effective_ai"]
    if has_b:
        # fac - B <= renewable - base
        b_ub[b_offset:b_offset + R] = env["renewable"] - env["base_facility"]
    if has_w:
        # -fac - W <= base + export - renewable
        b_ub[w_offset:w_offset + R] = (
            env["base_facility"] + env["export_limit"] - env["renewable"]
        )

    c = np.zeros(n_variables, dtype=np.float64)
    upper = np.full(n_variables, np.inf, dtype=np.float64)

    pue = env["pue"]
    sell = env["sell"]
    total_tasks = max(int(env["total_tasks"]), 1)
    region_positions = env["region_positions"]

    for j, column in enumerate(columns):
        task_class = class_lookup[column.class_id]
        rr_eq.append(column.class_id)
        cc_eq.append(j)
        vv_eq.append(1.0)

        if metric == "MeanLatency":
            c[j] = column.network_latency_ms / total_tasks

        pos_region = region_positions[column.target_region]
        start = int(column.start_hour)
        q = int(math.floor(task_class.duration_h + FLOAT_EPS))
        frac = float(task_class.duration_h - q)
        if frac < FLOAT_EPS:
            frac = 0.0

        # 直接根据区域连续Hour布局定位，避免字典查找与frame.iloc。
        for h in range(q):
            pos = int(pos_region[start + h])
            gpu = task_class.gpu_demand
            ai = task_class.task_power_mw
            fac = ai * pue[pos]
            rr_ub.extend((gpu_offset + pos, ai_offset + pos))
            cc_ub.extend((j, j))
            vv_ub.extend((gpu, ai))
            if has_b:
                rr_ub.append(int(b_offset + pos)); cc_ub.append(j); vv_ub.append(float(fac))
            if has_w:
                rr_ub.append(int(w_offset + pos)); cc_ub.append(j); vv_ub.append(float(-fac))
            if metric == "Cost":
                c[j] += sell[pos] * fac

        if frac > 0.0:
            pos = int(pos_region[start + q])
            gpu = task_class.gpu_demand * frac
            ai = task_class.task_power_mw * frac
            fac = ai * pue[pos]
            rr_ub.extend((gpu_offset + pos, ai_offset + pos))
            cc_ub.extend((j, j))
            vv_ub.extend((gpu, ai))
            if has_b:
                rr_ub.append(int(b_offset + pos)); cc_ub.append(j); vv_ub.append(float(fac))
            if has_w:
                rr_ub.append(int(w_offset + pos)); cc_ub.append(j); vv_ub.append(float(-fac))
            if metric == "Cost":
                c[j] += sell[pos] * fac

    if has_b:
        upper[b_idx] = np.maximum(env["max_grid"], 0.0)
        if metric == "Cost":
            c[b_idx] = env["price"] - env["sell"]
        elif metric == "Carbon":
            c[b_idx] = env["carbon"]
        for pos in range(R):
            rr_ub.append(int(b_offset + pos))
            cc_ub.append(int(b_idx[pos]))
            vv_ub.append(-1.0)

    if has_w:
        upper[w_idx] = np.maximum(env["renewable"], 0.0)
        if metric == "Cost":
            c[w_idx] = env["sell"]
        elif metric == "RenewableUnusedRate":
            # 乘以固定正数 TotalRenewable：由“未利用率”改为“未利用MWh”求解，
            # 最优解完全相同，但避免 1/TotalRenewable 导致的微小系数和对偶病态。
            c[w_idx] = 1.0
        for pos in range(R):
            rr_ub.append(int(w_offset + pos))
            cc_ub.append(int(w_idx[pos]))
            vv_ub.append(-1.0)

    row_eq = np.frombuffer(rr_eq, dtype=np.int32)
    col_eq = np.frombuffer(cc_eq, dtype=np.int32)
    val_eq = np.frombuffer(vv_eq, dtype=np.float64)
    row_ub = np.frombuffer(rr_ub, dtype=np.int32)
    col_ub = np.frombuffer(cc_ub, dtype=np.int32)
    val_ub = np.frombuffer(vv_ub, dtype=np.float64)

    A_eq = coo_array(
        (val_eq, (row_eq, col_eq)), shape=(C, n_variables), dtype=np.float64
    ).tocsr()
    A_ub = coo_array(
        (val_ub, (row_ub, col_ub)), shape=(row_count, n_variables), dtype=np.float64
    ).tocsr()
    b_eq = np.fromiter(
        (class_lookup[i].count for i in range(C)), dtype=np.float64, count=C
    )
    bounds = np.column_stack(
        (np.zeros(n_variables, dtype=np.float64), upper)
    )

    layout = {
        "R": R,
        "gpu_offset": gpu_offset,
        "ai_offset": ai_offset,
        "b_offset": b_offset,
        "w_offset": w_offset,
        "has_b": has_b,
        "has_w": has_w,
    }

    del rr_eq, cc_eq, vv_eq, rr_ub, cc_ub, vv_ub
    del row_eq, col_eq, val_eq, row_ub, col_ub, val_ub
    gc.collect()
    return c, A_ub, b_ub, A_eq, b_eq, bounds, layout

def _cg_duration_sum(
    prefix: np.ndarray,
    raw: np.ndarray,
    starts: np.ndarray,
    duration_h: float,
) -> np.ndarray:
    q = int(math.floor(duration_h + FLOAT_EPS))
    frac = float(duration_h - q)
    if frac < FLOAT_EPS:
        frac = 0.0
    result = prefix[starts + q] - prefix[starts]
    if frac > 0.0:
        result = result + frac * raw[starts + q]
    return result


def _cg_price_columns(
    *,
    bundle: InputBundle,
    class_lookup: dict[int, ExactTaskClass],
    active_keys: set[tuple[int, str, int]],
    metric: str,
    eq_dual: np.ndarray,
    ub_dual: np.ndarray,
    env: dict[str, Any],
    layout: dict[str, Any],
    max_new_columns: int,
    columns_per_class: int,
) -> tuple[list[CGColumn], float, float]:
    """完整候选流式定价。

    返回：(新增列, 最负遗漏列约化成本, 对偶可行修正量)。
    对每个任务类同时计算完整候选的最小约化成本 r_c。将受限主问题的
    类等式对偶变量下调 -min(0,r_c) 后即可得到完整主问题的可行对偶，故
    sum n_c*min(0,r_c) 是严格的目标下界修正量。
    """
    R = int(layout["R"])
    d_gpu = ub_dual[layout["gpu_offset"]:layout["gpu_offset"] + R]
    d_ai = ub_dual[layout["ai_offset"]:layout["ai_offset"] + R]
    d_b = (
        ub_dual[layout["b_offset"]:layout["b_offset"] + R]
        if layout["has_b"] else None
    )
    d_w = (
        ub_dual[layout["w_offset"]:layout["w_offset"] + R]
        if layout["has_w"] else None
    )

    pue_all = env["pue"]
    sell_all = env["sell"]
    region_positions = env["region_positions"]

    prefix_by_region: dict[str, tuple[np.ndarray, ...]] = {}
    for region, pos in region_positions.items():
        gpu_raw = d_gpu[pos]
        ai_raw = d_ai[pos]
        fac_dual_raw = np.zeros(len(pos), dtype=np.float64)
        if d_b is not None:
            fac_dual_raw += pue_all[pos] * d_b[pos]
        if d_w is not None:
            fac_dual_raw -= pue_all[pos] * d_w[pos]
        cost_raw = sell_all[pos] * pue_all[pos]

        def pref(x: np.ndarray) -> np.ndarray:
            out = np.empty(len(x) + 1, dtype=np.float64)
            out[0] = 0.0
            np.cumsum(x, dtype=np.float64, out=out[1:])
            return out

        prefix_by_region[region] = (
            pref(gpu_raw), gpu_raw,
            pref(ai_raw), ai_raw,
            pref(fac_dual_raw), fac_dual_raw,
            pref(cost_raw), cost_raw,
        )

    total_tasks = max(int(env["total_tasks"]), 1)
    active_by_class_region: dict[tuple[int, str], set[int]] = {}
    for active_class, active_region, active_start in active_keys:
        active_by_class_region.setdefault((active_class, active_region), set()).add(active_start)

    candidates: list[tuple[float, CGColumn]] = []
    global_min_omitted_rc = float("inf")
    dual_correction = 0.0

    for class_id in range(len(class_lookup)):
        task_class = class_lookup[class_id]
        earliest = (
            task_class.arrival_hour
            if task_class.task_type == "RealTimeInference"
            else max(task_class.arrival_hour, task_class.earliest_start_hour)
        )
        latest = _latest_start_for_class(task_class)
        if latest < earliest:
            continue
        starts = np.arange(earliest, latest + 1, dtype=np.int64)

        local_best: list[tuple[float, str, float, int]] = []
        class_min_rc = float("inf")
        for target_region, latency in task_class.candidate_regions:
            pg, rg, pa, ra, pf, rf, pc, rcost = prefix_by_region[target_region]
            resource_dual = (
                task_class.gpu_demand
                * _cg_duration_sum(pg, rg, starts, task_class.duration_h)
                + task_class.task_power_mw
                * _cg_duration_sum(pa, ra, starts, task_class.duration_h)
                + task_class.task_power_mw
                * _cg_duration_sum(pf, rf, starts, task_class.duration_h)
            )
            direct = np.zeros_like(resource_dual)
            if metric == "Cost":
                direct = (
                    task_class.task_power_mw
                    * _cg_duration_sum(pc, rcost, starts, task_class.duration_h)
                )
            elif metric == "MeanLatency":
                direct.fill(float(latency) / total_tasks)

            reduced_all = direct - float(eq_dual[class_id]) - resource_dual
            if reduced_all.size:
                class_min_rc = min(class_min_rc, float(np.min(reduced_all)))

            # 新列只从当前主问题之外挑选；对偶证书则使用上面的完整候选最小值。
            reduced = reduced_all.copy()
            for active_start in active_by_class_region.get((class_id, target_region), ()):
                idx = active_start - earliest
                if 0 <= idx < len(reduced):
                    reduced[idx] = np.inf

            finite = np.isfinite(reduced)
            if not finite.any():
                continue
            region_min = float(np.min(reduced[finite]))
            global_min_omitted_rc = min(global_min_omitted_rc, region_min)
            k = min(columns_per_class, int(finite.sum()))
            if k <= 0:
                continue
            idxs = (
                [int(np.argmin(reduced))]
                if k == 1
                else np.argpartition(reduced, k - 1)[:k].tolist()
            )
            for idx in idxs:
                value = float(reduced[idx])
                if value < -CG_REDUCED_COST_TOL:
                    local_best.append(
                        (value, target_region, float(latency), int(starts[idx]))
                    )

        if math.isfinite(class_min_rc):
            dual_correction += float(task_class.count) * min(0.0, class_min_rc)
        if local_best:
            local_best.sort(key=lambda x: x[0])
            for value, region, latency, start in local_best[:columns_per_class]:
                candidates.append(
                    (
                        value,
                        CGColumn(
                            class_id=class_id,
                            target_region=region,
                            network_latency_ms=latency,
                            start_hour=start,
                            finish_hour=float(start + task_class.duration_h),
                        ),
                    )
                )

    if not candidates:
        return [], global_min_omitted_rc, dual_correction
    candidates.sort(key=lambda x: x[0])
    return (
        [column for _, column in candidates[:max_new_columns]],
        global_min_omitted_rc,
        dual_correction,
    )

def _cg_recompute_metrics(
    *,
    bundle: InputBundle,
    class_lookup: dict[int, ExactTaskClass],
    columns: list[CGColumn],
    weights: np.ndarray,
) -> dict[str, float]:
    frame = bundle.region_hour.copy().sort_values(["Hour", "Region"], kind="stable").reset_index(drop=True)
    key_to_index = {
        (int(row.Hour), str(row.Region)): idx
        for idx, row in enumerate(frame.itertuples(index=False))
    }
    scheduled_ai = np.zeros(len(frame), dtype=np.float64)
    latency_sum = 0.0
    for j, column in enumerate(columns):
        weight = float(weights[j])
        if abs(weight) <= FLOAT_EPS:
            continue
        task_class = class_lookup[column.class_id]
        latency_sum += weight * column.network_latency_ms
        for hour, overlap in _overlap_by_hour(float(column.start_hour), task_class.duration_h):
            idx = key_to_index[(int(hour), column.target_region)]
            scheduled_ai[idx] += weight * task_class.task_power_mw * overlap
    facility = (
        frame["NonAI_IT_Load_MW"].to_numpy(dtype=np.float64) + scheduled_ai
    ) * frame["PUE"].to_numpy(dtype=np.float64)
    renewable = frame["AvailableRenewable_MW"].to_numpy(dtype=np.float64)
    export_limit = frame["ExportLimit_MW"].to_numpy(dtype=np.float64)
    grid = np.maximum(facility - renewable, 0.0)
    export = np.minimum(np.maximum(renewable - facility, 0.0), export_limit)
    curtail = renewable - np.minimum(facility, renewable) - export
    cost = float(np.sum(
        frame["ElectricityPrice_CNY_per_MWh"].to_numpy(dtype=np.float64) * grid
        - frame["SellPrice_CNY_per_MWh"].to_numpy(dtype=np.float64) * export
    ))
    carbon = float(np.sum(
        frame["CarbonIntensity_tCO2_per_MWh"].to_numpy(dtype=np.float64) * grid
    ))
    total_re = float(np.sum(renewable))
    total_tasks = max(sum(x.count for x in class_lookup.values()), 1)
    return {
        "Cost": cost,
        "Carbon": carbon,
        "MeanLatency": latency_sum / total_tasks,
        "RenewableUnusedRate": float(np.sum(curtail)) / total_re if total_re > FLOAT_EPS else float("nan"),
    }



def _cg_prune_column_pool(
    *,
    columns: list[CGColumn],
    result: Any,
    class_count: int,
    reserve_for_new: int,
    pool_limit: int,
) -> tuple[list[CGColumn], int]:
    """当列池过大时删除当前LP中无贡献的非基列。

    删除并不改变完整候选集：下一轮定价仍扫描全部合法列，任何再次变为负约化成本
    的已删列都会重新生成。因此这是内存管理，不是候选删减近似。
    """
    J = len(columns)
    dynamic_limit = max(pool_limit, class_count + 8000)
    target = max(class_count, dynamic_limit - max(0, reserve_for_new))
    if J <= target:
        return columns, 0

    weights = np.asarray(result.x[:J], dtype=np.float64)
    mandatory = set(np.flatnonzero(weights > CG_POSITIVE_COLUMN_TOL).tolist())

    # 数值保护：每个任务类至少保留当前权重最大的1列，确保删列后受限主问题仍可行。
    best_by_class: dict[int, tuple[float, int]] = {}
    for j, column in enumerate(columns):
        w = float(weights[j])
        prev = best_by_class.get(column.class_id)
        if prev is None or w > prev[0]:
            best_by_class[column.class_id] = (w, j)
    mandatory.update(idx for _, idx in best_by_class.values())

    keep_budget = max(target, len(mandatory))
    if len(mandatory) < keep_budget:
        try:
            marginal = np.asarray(result.lower.marginals[:J], dtype=np.float64)
            score = np.abs(marginal)
        except Exception:
            score = np.full(J, np.inf, dtype=np.float64)
        optional = np.asarray(
            [j for j in range(J) if j not in mandatory], dtype=np.int64
        )
        extra = min(keep_budget - len(mandatory), CG_POOL_NEAR_ZERO_KEEP, len(optional))
        if extra > 0:
            if extra < len(optional):
                chosen_local = np.argpartition(score[optional], extra - 1)[:extra]
                mandatory.update(optional[chosen_local].tolist())
            else:
                mandatory.update(optional.tolist())

    keep_idx = sorted(mandatory)
    pruned = [columns[j] for j in keep_idx]
    return pruned, J - len(pruned)

def _cg_iteration_checkpoint_path(metric: str) -> Path:
    safe = metric.replace("/", "_")
    return CHECKPOINT_DIR / f"q2_cg_iter_v{CALIBRATION_SCHEMA_VERSION}_{safe}.npz"


def _save_cg_iteration_checkpoint(
    *, bundle: InputBundle, metric: str, iteration: int, columns: list[CGColumn]
) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    region_code = {region: i for i, region in enumerate(bundle.regions)}
    path = _cg_iteration_checkpoint_path(metric)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        schema=np.asarray([CALIBRATION_SCHEMA_VERSION], dtype=np.int16),
        signature=np.asarray([_task_signature(bundle)]),
        iteration=np.asarray([iteration], dtype=np.int32),
        class_id=np.asarray([c.class_id for c in columns], dtype=np.int32),
        region_code=np.asarray([region_code[c.target_region] for c in columns], dtype=np.uint8),
        latency=np.asarray([c.network_latency_ms for c in columns], dtype=np.float64),
        start=np.asarray([c.start_hour for c in columns], dtype=np.int16),
    )
    tmp.replace(path)


def _load_cg_iteration_checkpoint(
    *, bundle: InputBundle, metric: str, class_lookup: dict[int, ExactTaskClass]
) -> tuple[int, list[CGColumn]] | None:
    path = _cg_iteration_checkpoint_path(metric)
    if not path.is_file():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        if int(data["schema"][0]) != CALIBRATION_SCHEMA_VERSION:
            return None
        if str(data["signature"][0]) != _task_signature(bundle):
            return None
        iteration = int(data["iteration"][0])
        class_ids = data["class_id"].astype(np.int64, copy=False)
        region_codes = data["region_code"].astype(np.int64, copy=False)
        latencies = data["latency"].astype(np.float64, copy=False)
        starts = data["start"].astype(np.int64, copy=False)
        columns: list[CGColumn] = []
        for cid, rcode, latency, start in zip(class_ids, region_codes, latencies, starts):
            task_class = class_lookup[int(cid)]
            columns.append(
                CGColumn(
                    class_id=int(cid),
                    target_region=bundle.regions[int(rcode)],
                    network_latency_ms=float(latency),
                    start_hour=int(start),
                    finish_hour=float(int(start) + task_class.duration_h),
                )
            )
        return iteration, columns
    except Exception as exc:
        logging.warning("列生成迭代检查点读取失败，忽略并重建：%s", exc)
        return None


def _clear_cg_iteration_checkpoint(metric: str) -> None:
    path = _cg_iteration_checkpoint_path(metric)
    if path.is_file():
        path.unlink()


def _cg_anchor_checkpoint_paths(metric: str) -> tuple[Path, Path]:
    safe = metric.replace("/", "_")
    return (
        CHECKPOINT_DIR / f"q2_cg_anchor_v{CALIBRATION_SCHEMA_VERSION}_{safe}.json",
        CHECKPOINT_DIR / f"q2_cg_anchor_v{CALIBRATION_SCHEMA_VERSION}_{safe}_positive.npz",
    )


def _save_cg_anchor_checkpoint(
    *, bundle: InputBundle, anchor: CGAnchorResult
) -> None:
    meta_path, col_path = _cg_anchor_checkpoint_paths(anchor.metric)
    meta_path.write_text(
        json.dumps(
            {
                "schema_version": CALIBRATION_SCHEMA_VERSION,
                "task_signature": _task_signature(bundle),
                "metric": anchor.metric,
                "metrics": anchor.metrics,
                "iterations": anchor.iterations,
                "final_min_reduced_cost": anchor.final_min_reduced_cost,
                "elapsed_seconds": anchor.elapsed_seconds,
                "ideal_lower_bound": anchor.ideal_lower_bound,
                "certified_abs_gap": anchor.certified_abs_gap,
                "convergence_status": anchor.convergence_status,
            },
            ensure_ascii=False, indent=2, allow_nan=False,
        ),
        encoding="utf-8",
    )
    region_code = {region: i for i, region in enumerate(bundle.regions)}
    positive = [
        c for c, w in zip(anchor.columns, anchor.weights)
        if float(w) > CG_POSITIVE_COLUMN_TOL
    ]
    np.savez(
        col_path,
        class_id=np.asarray([c.class_id for c in positive], dtype=np.int32),
        region_code=np.asarray([region_code[c.target_region] for c in positive], dtype=np.uint8),
        latency=np.asarray([c.network_latency_ms for c in positive], dtype=np.float64),
        start=np.asarray([c.start_hour for c in positive], dtype=np.int16),
    )


def _load_cg_anchor_checkpoint(
    *, bundle: InputBundle, metric: str, class_lookup: dict[int, ExactTaskClass]
) -> tuple[dict[str, Any], list[CGColumn]] | None:
    meta_path, col_path = _cg_anchor_checkpoint_paths(metric)
    if not meta_path.is_file() or not col_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
            return None
        if meta.get("task_signature") != _task_signature(bundle):
            return None
        data = np.load(col_path, allow_pickle=False)
        columns: list[CGColumn] = []
        for cid, rcode, latency, start in zip(
            data["class_id"], data["region_code"], data["latency"], data["start"]
        ):
            task_class = class_lookup[int(cid)]
            columns.append(
                CGColumn(
                    class_id=int(cid),
                    target_region=bundle.regions[int(rcode)],
                    network_latency_ms=float(latency),
                    start_hour=int(start),
                    finish_hour=float(int(start) + task_class.duration_h),
                )
            )
        return meta, columns
    except Exception as exc:
        logging.warning("锚点检查点读取失败，忽略：%s", exc)
        return None


def _solve_global_anchor_column_generation(
    *,
    bundle: InputBundle,
    class_lookup: dict[int, ExactTaskClass],
    initial_columns: list[CGColumn],
    metric: str,
    progress_interval: float,
    lp_time_limit: float | None,
) -> CGAnchorResult:
    from scipy.optimize import linprog

    env = _cg_full_resource_environment(bundle)
    checkpoint = _load_cg_iteration_checkpoint(
        bundle=bundle, metric=metric, class_lookup=class_lookup
    )
    if checkpoint is not None:
        saved_iteration, columns = checkpoint
        start_iteration = saved_iteration + 1
        logging.info(
            "恢复全局LP列生成 %s：从迭代%d后的列池继续，列=%d。",
            metric, saved_iteration, len(columns),
        )
    else:
        columns = list(initial_columns)
        start_iteration = 1
    columns = list({column.key: column for column in columns}.values())
    active_keys = {column.key for column in columns}
    started_all = time.perf_counter()
    last_min_rc = float("nan")

    renewable = metric == "RenewableUnusedRate"
    max_new = (
        CG_RENEWABLE_MAX_NEW_COLUMNS_PER_ITERATION
        if renewable else CG_MAX_NEW_COLUMNS_PER_ITERATION
    )
    per_class = CG_RENEWABLE_COLUMNS_PER_CLASS if renewable else CG_COLUMNS_PER_CLASS
    pool_limit = CG_RENEWABLE_COLUMN_POOL_LIMIT if renewable else CG_COLUMN_POOL_LIMIT

    for iteration in range(start_iteration, CG_MAX_ITERATIONS + 1):
        _memory_guard(f"CG-{metric}-迭代{iteration}-建模前")
        c, A_ub, b_ub, A_eq, b_eq, bounds, layout = _cg_build_master(
            bundle=bundle,
            class_lookup=class_lookup,
            columns=columns,
            metric=metric,
            env=env,
        )
        logging.info(
            "全局LP列生成 %s：迭代=%d，主问题列=%d，变量=%d，约束=%d，非零=%d。",
            metric, iteration, len(columns), len(c),
            A_ub.shape[0] + A_eq.shape[0], A_ub.nnz + A_eq.nnz,
        )
        _memory_guard(f"CG-{metric}-迭代{iteration}-求解前", hard=True)

        options: dict[str, Any] = {"disp": False, "presolve": True}
        if lp_time_limit is not None and lp_time_limit > 0:
            options["time_limit"] = float(lp_time_limit)

        label = f"Q2全局连续理想点-{metric}-CG{iteration}"
        result = _run_with_heartbeat(
            linprog, label=label, interval=progress_interval,
            c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method="highs", options=options,
        )
        if int(result.status) != 0 or result.x is None:
            raise RuntimeError(
                f"{label}受限主问题未证明最优：status={result.status}; message={result.message}"
            )

        eq_dual = np.asarray(result.eqlin.marginals, dtype=np.float64)
        ub_dual = np.asarray(result.ineqlin.marginals, dtype=np.float64)
        new_columns, min_rc, dual_correction = _cg_price_columns(
            bundle=bundle,
            class_lookup=class_lookup,
            active_keys=active_keys,
            metric=metric,
            eq_dual=eq_dual,
            ub_dual=ub_dual,
            env=env,
            layout=layout,
            max_new_columns=max_new,
            columns_per_class=per_class,
        )
        last_min_rc = float(min_rc)

        certified_lb_rate = float("nan")
        certified_gap = float("nan")
        if renewable:
            total_re = max(float(env["global_renewable"]), FLOAT_EPS)
            rmp_rate = float(result.fun) / total_re
            dual_lb_rate = (float(result.fun) + float(dual_correction)) / total_re
            certified_lb_rate = max(
                0.0, float(env["renewable_physical_lb_rate"]), dual_lb_rate
            )
            certified_lb_rate = min(certified_lb_rate, rmp_rate)
            certified_gap = max(0.0, rmp_rate - certified_lb_rate)
            logging.info(
                "RenewableUnusedRate证书：RMP=%.10g，LB=%.10g，abs_gap=%.3g，物理LB=%.10g。",
                rmp_rate, certified_lb_rate, certified_gap,
                float(env["renewable_physical_lb_rate"]),
            )

        logging.info(
            "全局LP列生成 %s：迭代=%d，活跃列=%d，新列=%d，最小约化成本=%.6g。",
            metric, iteration, len(columns), len(new_columns), last_min_rc,
        )

        exact_converged = not new_columns
        certified_converged = (
            renewable
            and math.isfinite(certified_gap)
            and certified_gap <= CG_RENEWABLE_CERTIFIED_ABS_GAP
        )
        if exact_converged or certified_converged:
            weights = np.asarray(result.x[: len(columns)], dtype=np.float64).copy()
            metrics = _cg_recompute_metrics(
                bundle=bundle, class_lookup=class_lookup, columns=columns, weights=weights
            )
            elapsed = time.perf_counter() - started_all
            if exact_converged:
                ideal_lb = float(metrics[metric])
                gap = 0.0
                status = "exact_reduced_cost_converged"
            else:
                ideal_lb = float(certified_lb_rate)
                gap = float(certified_gap)
                status = "certified_dual_gap"
                logging.info(
                    "RenewableUnusedRate以严格上下界证书停止：可行值=%.10g，下界=%.10g，"
                    "绝对差=%.3g <= %.3g。",
                    float(metrics[metric]), ideal_lb, gap, CG_RENEWABLE_CERTIFIED_ABS_GAP,
                )
            anchor = CGAnchorResult(
                metric=metric, columns=columns, weights=weights, metrics=metrics,
                iterations=iteration, final_min_reduced_cost=last_min_rc,
                elapsed_seconds=elapsed, ideal_lower_bound=ideal_lb,
                certified_abs_gap=gap, convergence_status=status,
            )
            _save_cg_anchor_checkpoint(bundle=bundle, anchor=anchor)
            _clear_cg_iteration_checkpoint(metric)
            del A_ub, A_eq, b_ub, b_eq, bounds, c, eq_dual, ub_dual, layout, result
            gc.collect()
            return anchor

        pruned_columns, removed = _cg_prune_column_pool(
            columns=columns, result=result, class_count=len(class_lookup),
            reserve_for_new=len(new_columns), pool_limit=pool_limit,
        )
        if removed:
            logging.info(
                "全局LP列池整理 %s：删除当前无贡献非基列=%d，%d→%d。",
                metric, removed, len(columns), len(pruned_columns),
            )
            columns = pruned_columns
            active_keys = {column.key for column in columns}

        for column in new_columns:
            if column.key not in active_keys:
                active_keys.add(column.key)
                columns.append(column)

        if iteration % CG_ITERATION_CHECKPOINT_EVERY == 0:
            _save_cg_iteration_checkpoint(
                bundle=bundle, metric=metric, iteration=iteration, columns=columns
            )

        del A_ub, A_eq, b_ub, b_eq, bounds, c, eq_dual, ub_dual, layout, result
        gc.collect()

    _save_cg_iteration_checkpoint(
        bundle=bundle, metric=metric, iteration=CG_MAX_ITERATIONS, columns=columns
    )
    raise RuntimeError(
        f"全局LP列生成-{metric}达到{CG_MAX_ITERATIONS}次迭代仍未满足完整定价/"
        f"对偶证书停止条件（min_rc={last_min_rc}）；已保存迭代列池，下次--resume可继续。"
    )

def _calibration_cache_path() -> Path:
    return CHECKPOINT_DIR / f"q2_refactored_global_calibration_v{CALIBRATION_SCHEMA_VERSION}.json"


def _compute_or_load_calibration(
    *,
    bundle: InputBundle,
    classes: list[ExactTaskClass],
    class_lookup: dict[int, ExactTaskClass],
    baseline_assignments: pd.DataFrame,
    baseline_metrics: dict[str, float],
    lp_time_limit: float | None,
    progress_interval: float,
    force_recalibrate: bool,
) -> tuple[GlobalCalibration, pd.DataFrame]:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _calibration_cache_path()
    signature = _task_signature(bundle)
    baseline_values = {metric: float(baseline_metrics[metric]) for metric in OBJECTIVE_NAMES}

    if path.is_file() and not force_recalibrate:
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            data.get("schema_version") == CALIBRATION_SCHEMA_VERSION
            and data.get("task_signature") == signature
            and all(
                math.isclose(
                    float(data["baseline_value"][metric]),
                    baseline_values[metric],
                    rel_tol=1e-10,
                    abs_tol=1e-8,
                )
                for metric in OBJECTIVE_NAMES
            )
        ):
            calibration = GlobalCalibration(
                ideal_lb={k: float(v) for k, v in data["ideal_lb"].items()},
                baseline_value={k: float(v) for k, v in data["baseline_value"].items()},
                reference_value={k: float(v) for k, v in data["reference_value"].items()},
                scale={k: float(v) for k, v in data["scale"].items()},
                active_metrics=tuple(data["active_metrics"]),
            )
            logging.info("复用全局连续定标缓存：%s", path)
            rows = [
                {
                    "Objective": metric,
                    "IdealLowerBound": calibration.ideal_lb[metric],
                    "BaselineValue": calibration.baseline_value[metric],
                    "ReferenceValue": calibration.reference_value[metric],
                    "Scale": calibration.scale[metric],
                    "ActiveInMinMax": int(metric in calibration.active_metrics),
                    "CalibrationMethod": "exact_column_generation",
                    "CacheReused": True,
                }
                for metric in OBJECTIVE_NAMES
            ]
            return calibration, pd.DataFrame(rows)

    logging.info(
        "阶段3：全局连续定标改用Dantzig-Wolfe列生成；完整候选流式定价，不再一次性展开全时域矩阵。"
    )
    baseline_seed = _cg_seed_columns_from_baseline(baseline_assignments, class_lookup)
    seed_map = {column.key: column for column in baseline_seed}
    ideal_lb: dict[str, float] = {}
    payoff: dict[str, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []

    for metric in OBJECTIVE_NAMES:
        cached_anchor = None if force_recalibrate else _load_cg_anchor_checkpoint(
            bundle=bundle, metric=metric, class_lookup=class_lookup
        )
        if cached_anchor is not None:
            meta, positive_columns = cached_anchor
            metrics = {k: float(v) for k, v in meta["metrics"].items()}
            ideal_lb[metric] = float(meta["ideal_lower_bound"])
            payoff[metric] = metrics
            rows.append(
                {
                    "AnchorObjective": metric,
                    "LPStatus": 0,
                    "LPMessage": str(meta.get("convergence_status", "anchor_cache")),
                    "ElapsedSeconds": float(meta.get("elapsed_seconds", 0.0)),
                    "CGIterations": int(meta.get("iterations", 0)),
                    "ActiveColumnCount": len(positive_columns),
                    "FinalMinReducedCost": float(meta.get("final_min_reduced_cost", 0.0)),
                    "CertifiedAbsGap": float(meta.get("certified_abs_gap", 0.0)),
                    "CalibrationMethod": "column_generation_with_dual_certificate",
                    "CacheReused": True,
                    **{f"Payoff_{name}": metrics[name] for name in OBJECTIVE_NAMES},
                }
            )
            for column in positive_columns:
                seed_map[column.key] = column
            logging.info(
                "复用阶段3锚点缓存 %s：ideal_lb=%.12g，正列=%d。",
                metric, ideal_lb[metric], len(positive_columns),
            )
            continue

        anchor = _solve_global_anchor_column_generation(
            bundle=bundle,
            class_lookup=class_lookup,
            initial_columns=list(seed_map.values()),
            metric=metric,
            progress_interval=progress_interval,
            lp_time_limit=lp_time_limit,
        )
        ideal_lb[metric] = float(anchor.ideal_lower_bound)
        payoff[metric] = dict(anchor.metrics)
        rows.append(
            {
                "AnchorObjective": metric,
                "LPStatus": 0,
                "LPMessage": anchor.convergence_status,
                "ElapsedSeconds": anchor.elapsed_seconds,
                "CGIterations": anchor.iterations,
                "ActiveColumnCount": len(anchor.columns),
                "FinalMinReducedCost": anchor.final_min_reduced_cost,
                "CertifiedAbsGap": anchor.certified_abs_gap,
                "CalibrationMethod": "column_generation_with_dual_certificate",
                "CacheReused": False,
                **{f"Payoff_{name}": anchor.metrics[name] for name in OBJECTIVE_NAMES},
            }
        )
        logging.info(
            "全局理想点 %s：下界=%.12g，可行锚点=%.12g，CG迭代=%d，状态=%s。",
            metric, ideal_lb[metric], float(anchor.metrics[metric]),
            anchor.iterations, anchor.convergence_status,
        )
        for column, weight in zip(anchor.columns, anchor.weights):
            if float(weight) > CG_POSITIVE_COLUMN_TOL:
                seed_map[column.key] = column
        # 锚点函数内部已落盘，之后即使后续目标失败也不会丢失。
        del anchor
        gc.collect()

    reference_value: dict[str, float] = {}
    scale: dict[str, float] = {}
    active: list[str] = []
    for metric in OBJECTIVE_NAMES:
        reference = max(
            baseline_values[metric],
            max(payoff[anchor][metric] for anchor in OBJECTIVE_NAMES),
        )
        reference_value[metric] = float(reference)
        diff = float(reference) - float(ideal_lb[metric])
        scale[metric] = diff
        if math.isfinite(diff) and diff > FLOAT_EPS:
            active.append(metric)
    if len(active) < 2:
        raise RuntimeError("全局定标后有效目标不足2个，请检查基准与连续 payoff matrix")

    calibration = GlobalCalibration(
        ideal_lb=ideal_lb,
        baseline_value=baseline_values,
        reference_value=reference_value,
        scale=scale,
        active_metrics=tuple(active),
    )
    path.write_text(
        json.dumps(
            {
                "schema_version": CALIBRATION_SCHEMA_VERSION,
                "task_signature": signature,
                "ideal_lb": ideal_lb,
                "baseline_value": baseline_values,
                "reference_value": reference_value,
                "scale": scale,
                "active_metrics": active,
                "payoff_matrix": payoff,
                "calibration_method": "column_generation_with_dual_certificate",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    table = pd.DataFrame(rows)
    for metric in OBJECTIVE_NAMES:
        table[f"IdealLB_{metric}"] = ideal_lb[metric]
        table[f"Baseline_{metric}"] = baseline_values[metric]
        table[f"Reference_{metric}"] = reference_value[metric]
        table[f"Scale_{metric}"] = scale[metric]
    return calibration, table


# =============================================================================
# 10. 均衡滚动：每窗口仅一次 min z
# =============================================================================

def _baseline_latency_lookup(baseline_assignments: pd.DataFrame) -> dict[str, float]:
    return {
        str(row.TaskID): float(row.NetworkLatency_ms)
        for row in baseline_assignments.itertuples(index=False)
    }


def _active_task_ids_from_model(
    model: AggregatedModel,
    pools: dict[int, list[str]],
) -> set[str]:
    ids: set[str] = set()
    for class_id in model.class_ids:
        ids.update(pools[class_id])
    return ids


def _past_delta(
    *,
    bundle: InputBundle,
    balanced_committed: pd.DataFrame,
    baseline_assignments: pd.DataFrame,
    baseline_latency: dict[str, float],
    tau: int,
    global_task_count: int,
) -> dict[str, float]:
    actual_energy = _energy_contribution(bundle, balanced_committed, 0, tau)
    baseline_energy = _energy_contribution(bundle, baseline_assignments, 0, tau)
    delta = {
        metric: actual_energy[metric] - baseline_energy[metric]
        for metric in ("Cost", "Carbon", "RenewableUnusedRate")
    }
    if balanced_committed.empty:
        delta["MeanLatency"] = 0.0
    else:
        actual_sum = float(balanced_committed["NetworkLatency_ms"].astype(float).sum())
        baseline_sum = float(
            sum(baseline_latency[str(task_id)] for task_id in balanced_committed["TaskID"].astype(str))
        )
        delta["MeanLatency"] = (actual_sum - baseline_sum) / max(global_task_count, 1)
    return delta


def _baseline_scope_value(
    *,
    bundle: InputBundle,
    baseline_assignments: pd.DataFrame,
    baseline_profile: pd.DataFrame,
    baseline_latency: dict[str, float],
    active_task_ids: set[str],
    tau: int,
    objective_end: int,
    global_task_count: int,
) -> dict[str, float]:
    scope = baseline_profile.loc[
        baseline_profile["Hour"].between(tau, objective_end - 1)
    ]
    global_re = float(bundle.region_hour["AvailableRenewable_MW"].sum())
    values = {
        "Cost": float(scope["OperatingCost_CNY"].sum()),
        "Carbon": float(scope["CarbonEmission_tCO2"].sum()),
        "RenewableUnusedRate": (
            float(scope["RenewableCurtailment_MW"].sum()) / global_re
            if global_re > FLOAT_EPS
            else 0.0
        ),
        "MeanLatency": float(
            sum(baseline_latency[task_id] for task_id in active_task_ids)
        ) / max(global_task_count, 1),
    }
    return values


def _solve_balanced_minmax_window(
    *,
    model: AggregatedModel,
    calibration: GlobalCalibration,
    past_delta: dict[str, float],
    baseline_scope: dict[str, float],
    mip_rel_gap: float,
    initial_time_limit: float,
    max_time_limit: float,
    progress_interval: float,
    window_id: int,
    mip_start_indices: np.ndarray | None = None,
    mip_start_values: np.ndarray | None = None,
    previous_k_warmstart_available: bool = False,
) -> tuple[SolveResult, dict[str, float], list[dict[str, Any]]]:
    if model.z_index is None:
        raise ValueError("均衡模型必须包含 z")
    extra_rows: list[tuple[dict[int, float], float, float]] = []
    estimate_constants: dict[str, float] = {}
    structural_lb_available = True

    # 未来未进入当前计划域的部分沿用纯算力基准，因此估计的全局指标为：
    # f_hat = f_baseline_global + past_delta + f_scope(y) - f_scope_baseline。
    # 对每个目标：
    # d_m = constant_m + a_m^T x <= z。
    # 当前Q2紧凑上图在已校验的 price>=sell>=0、carbon>=0 条件下，
    # Cost/Carbon/Latency/Renewable 的 raw objective 系数都非负。
    # 又因全部任务/能源变量下界为0，所以 d_m >= constant_m，
    # 从而 z >= max_m constant_m 是不依赖LP求解的严格结构下界。
    for metric in calibration.active_metrics:
        scale = calibration.scale[metric]
        raw_vector = model.objective_vectors[metric]
        raw_constant = float(model.objective_constants[metric])
        min_raw = float(np.min(raw_vector)) if len(raw_vector) else 0.0
        if min_raw < -BALANCED_STRUCTURAL_COEF_TOL:
            structural_lb_available = False

        terms = {
            idx: float(value / scale)
            for idx, value in enumerate(raw_vector)
            if abs(value) > FLOAT_EPS
        }
        terms[model.z_index] = terms.get(model.z_index, 0.0) - 1.0
        constant = (
            float(calibration.baseline_value[metric])
            + float(past_delta.get(metric, 0.0))
            + raw_constant
            - float(baseline_scope[metric])
            - float(calibration.ideal_lb[metric])
        ) / scale
        estimate_constants[metric] = constant
        extra_rows.append((terms, -np.inf, -constant))

    z_objective = np.zeros(len(model.variable_lower), dtype=float)
    z_objective[model.z_index] = 1.0
    solver_rows: list[dict[str, Any]] = []

    structural_lb = (
        max(estimate_constants.values())
        if structural_lb_available and estimate_constants
        else float("nan")
    )

    # 快路不再额外求大LP。若结构下界为正，则把z直接限制在能保证原目标gap的窄带，
    # 以零目标只寻找一个整数可行解。零目标下，一旦找到可行解，HiGHS无需继续
    # 为min-z做漫长的最优性证明；外部用严格结构下界完成gap证书。
    fast_result: SolveResult | None = None
    if (
        BALANCED_STRUCTURAL_FASTPATH
        and math.isfinite(structural_lb)
        and structural_lb > FLOAT_EPS
        and 0.0 <= mip_rel_gap < 1.0
    ):
        z_cap = structural_lb / max(1.0 - float(mip_rel_gap), 1e-12)
        z_cap = min(z_cap, float(model.variable_upper[model.z_index]))
        fast_upper = np.asarray(model.variable_upper, dtype=np.float64).copy()
        fast_upper[model.z_index] = z_cap
        zero_objective = np.zeros(len(model.variable_lower), dtype=np.float64)

        logging.info(
            "均衡窗口%d结构证书快路：严格z下界=%.12g；只需找到z<=%.12g的整数可行解"
            "即可证明gap<=%.4g；快路限时=%.0fs。",
            window_id,
            structural_lb,
            z_cap,
            mip_rel_gap,
            BALANCED_STRUCTURAL_FASTPATH_TIME_LIMIT,
        )
        try:
            trial = _solve_milp_highspy(
                model,
                zero_objective,
                label=f"均衡窗口{window_id}-结构下界目标带可行性",
                mip_rel_gap=0.0,
                initial_time_limit=BALANCED_STRUCTURAL_FASTPATH_TIME_LIMIT,
                max_time_limit=BALANCED_STRUCTURAL_FASTPATH_TIME_LIMIT,
                progress_interval=progress_interval,
                extra_rows=extra_rows,
                variable_upper=fast_upper,
                mip_start_indices=mip_start_indices,
                mip_start_values=mip_start_values,
            )
            z_value = float(trial.vector[model.z_index])
            certified_gap = max(
                0.0,
                (z_value - structural_lb) / max(abs(z_value), 1e-12),
            )
            solver_rows.append(
                {
                    "Mode": "balanced",
                    "WindowID": window_id,
                    "Stage": "balanced_structural_lb_feasibility",
                    "Status": 0,
                    "Message": (
                        f"structural_lb_certified; lb={structural_lb:.12g}; "
                        f"z={z_value:.12g}"
                    ),
                    "MIPGap": certified_gap,
                    "MIPDualBound": structural_lb,
                    "ObjectiveValue": z_value,
                    "ElapsedSeconds": trial.elapsed_seconds,
                    "TimeLimitUsedSeconds": trial.time_limit_used,
                }
            )
            if (
                z_value <= z_cap + 1e-7
                and certified_gap <= mip_rel_gap + 1e-7
            ):
                fast_result = SolveResult(
                    vector=trial.vector,
                    status=0,
                    message=(
                        "structural_lower_bound_feasibility_certified:"
                        f"lb={structural_lb:.12g};z={z_value:.12g};"
                        f"gap={certified_gap:.12g}"
                    ),
                    mip_gap=certified_gap,
                    mip_dual_bound=structural_lb,
                    objective_value=z_value,
                    elapsed_seconds=trial.elapsed_seconds,
                    time_limit_used=trial.time_limit_used,
                )
                logging.info(
                    "均衡窗口%d结构证书快路命中：z=%.12g，严格LB=%.12g，"
                    "证书gap=%.6g；无需继续完整min-z。",
                    window_id, z_value, structural_lb, certified_gap,
                )
        except RuntimeError as exc:
            logging.info(
                "均衡窗口%d结构证书快路未在%.0fs内找到目标带整数可行解，"
                "回退完整min-z：%s",
                window_id,
                BALANCED_STRUCTURAL_FASTPATH_TIME_LIMIT,
                exc,
            )

    if fast_result is not None:
        result = fast_result
    else:
        # 无跨窗口K区热启动时，优先用此前已实测稳定的SciPy milp作为bootstrap，
        # 避免原生highspy在首个无MIP Start窗口长时间只保留z=1e6的劣质incumbent。
        if not previous_k_warmstart_available:
            logging.info(
                "均衡窗口%d完整min-z无上一窗口K区热启动，使用SciPy/HiGHS bootstrap；"
                "完成后仍保存K区供下一窗口原生HiGHS热启动。",
                window_id,
            )
            result = _solve_milp_adaptive(
                model,
                z_objective,
                label=f"均衡窗口{window_id}-全局定标min-max-bootstrap",
                mip_rel_gap=mip_rel_gap,
                initial_time_limit=initial_time_limit,
                max_time_limit=max_time_limit,
                progress_interval=progress_interval,
                extra_rows=extra_rows,
            )
            backend_stage = "balanced_global_scaled_minmax_scipy_bootstrap"
        else:
            result = _solve_milp_highspy(
                model,
                z_objective,
                label=f"均衡窗口{window_id}-全局定标min-max",
                mip_rel_gap=mip_rel_gap,
                initial_time_limit=initial_time_limit,
                max_time_limit=max_time_limit,
                progress_interval=progress_interval,
                extra_rows=extra_rows,
                mip_start_indices=mip_start_indices,
                mip_start_values=mip_start_values,
            )
            backend_stage = "balanced_global_scaled_minmax_highspy_warmstart"

        solver_rows.append(
            {
                "Mode": "balanced",
                "WindowID": window_id,
                "Stage": backend_stage,
                "Status": result.status,
                "Message": result.message,
                "MIPGap": result.mip_gap,
                "MIPDualBound": result.mip_dual_bound,
                "ObjectiveValue": float(result.vector[model.z_index]),
                "ElapsedSeconds": result.elapsed_seconds,
                "TimeLimitUsedSeconds": result.time_limit_used,
            }
        )

    estimated_d: dict[str, float] = {}
    for metric in calibration.active_metrics:
        raw_value = _objective_value(model, metric, result.vector)
        estimated_global = (
            calibration.baseline_value[metric]
            + past_delta.get(metric, 0.0)
            + raw_value
            - baseline_scope[metric]
        )
        estimated_d[metric] = (
            estimated_global - calibration.ideal_lb[metric]
        ) / calibration.scale[metric]

    z_star = float(result.vector[model.z_index])
    metrics = {
        "ZStar": z_star,
        "StructuralZLowerBound": structural_lb,
        "EstimatedMaxNormalizedDeviation": max(estimated_d.values()) if estimated_d else float("nan"),
        **{f"EstimatedD_{metric}": estimated_d.get(metric, np.nan) for metric in OBJECTIVE_NAMES},
    }
    return result, metrics, solver_rows




def _heuristic_resource_environment(bundle: InputBundle) -> dict[str, Any]:
    """把逐时输入转成 [hour, region] 小矩阵，供快速构造反复 O(1) 查询。"""
    region_index = {region: idx for idx, region in enumerate(bundle.regions)}
    shape = (TERMINAL_HOUR, len(bundle.regions))
    names = (
        "available_gpu", "effective_ai", "non_ai", "pue", "renewable",
        "export_limit", "price", "sell", "carbon",
    )
    env: dict[str, Any] = {
        "region_index": region_index,
        "regions": bundle.regions,
        "global_renewable": float(bundle.region_hour["AvailableRenewable_MW"].sum()),
    }
    for name in names:
        env[name] = np.zeros(shape, dtype=np.float64)

    for row in bundle.region_hour.itertuples(index=False):
        h = int(row.Hour)
        r = region_index[str(row.Region)]
        env["available_gpu"][h, r] = float(row.Available_GPU)
        env["effective_ai"][h, r] = float(row.Effective_AI_IT_Capacity_MW_Exact)
        env["non_ai"][h, r] = float(row.NonAI_IT_Load_MW)
        env["pue"][h, r] = float(row.PUE)
        env["renewable"][h, r] = float(row.AvailableRenewable_MW)
        env["export_limit"][h, r] = float(row.ExportLimit_MW)
        env["price"][h, r] = float(row.ElectricityPrice_CNY_per_MWh)
        env["sell"][h, r] = float(row.SellPrice_CNY_per_MWh)
        env["carbon"][h, r] = float(row.CarbonIntensity_tCO2_per_MWh)
    return env


def _heuristic_add_row_load(
    row: Any,
    *,
    gpu: np.ndarray,
    ai: np.ndarray,
    region_index: dict[str, int],
    sign: float = 1.0,
) -> None:
    r = region_index[str(row["TargetRegion"] if isinstance(row, dict) else row.TargetRegion)]
    start = float(row["StartHour"] if isinstance(row, dict) else row.StartHour)
    duration = float(row["Duration_h"] if isinstance(row, dict) else row.Duration_h)
    gpu_demand = float(row["GPU_Demand"] if isinstance(row, dict) else row.GPU_Demand)
    power = float(
        row["Task_Full_IT_Power_MW"]
        if isinstance(row, dict)
        else row.Task_Full_IT_Power_MW
    )
    for h, overlap in _overlap_by_hour(start, duration):
        gpu[h, r] += sign * gpu_demand * overlap
        ai[h, r] += sign * power * overlap


def _heuristic_initial_committed_loads(
    assignments: pd.DataFrame,
    env: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    gpu = np.zeros_like(env["available_gpu"], dtype=np.float64)
    ai = np.zeros_like(env["effective_ai"], dtype=np.float64)
    if not assignments.empty:
        for row in assignments.itertuples(index=False):
            _heuristic_add_row_load(
                row,
                gpu=gpu,
                ai=ai,
                region_index=env["region_index"],
                sign=1.0,
            )
    return gpu, ai


def _heuristic_scope_energy(
    env: dict[str, Any],
    ai: np.ndarray,
    tau: int,
    objective_end: int,
) -> dict[str, float]:
    if objective_end <= tau:
        return {"Cost": 0.0, "Carbon": 0.0, "RenewableUnusedRate": 0.0}
    sl = slice(tau, objective_end)
    facility = (env["non_ai"][sl] + ai[sl]) * env["pue"][sl]
    renewable = env["renewable"][sl]
    grid = np.maximum(facility - renewable, 0.0)
    export = np.minimum(
        np.maximum(renewable - facility, 0.0),
        env["export_limit"][sl],
    )
    curtail = renewable - np.minimum(facility, renewable) - export
    global_re = max(float(env["global_renewable"]), FLOAT_EPS)
    return {
        "Cost": float(np.sum(env["price"][sl] * grid - env["sell"][sl] * export)),
        "Carbon": float(np.sum(env["carbon"][sl] * grid)),
        "RenewableUnusedRate": float(np.sum(curtail)) / global_re,
    }


def _heuristic_candidate_energy_delta(
    *,
    env: dict[str, Any],
    ai: np.ndarray,
    task_class: ExactTaskClass,
    target_region: str,
    start_hour: int,
    objective_end: int,
) -> dict[str, float]:
    r = env["region_index"][target_region]
    d_cost = 0.0
    d_carbon = 0.0
    d_unused = 0.0
    global_re = max(float(env["global_renewable"]), FLOAT_EPS)

    for h, overlap in _overlap_by_hour(float(start_hour), task_class.duration_h):
        if h >= objective_end:
            continue
        before_fac = (
            env["non_ai"][h, r] + ai[h, r]
        ) * env["pue"][h, r]
        after_fac = (
            env["non_ai"][h, r]
            + ai[h, r]
            + task_class.task_power_mw * overlap
        ) * env["pue"][h, r]
        re = env["renewable"][h, r]
        ex_lim = env["export_limit"][h, r]

        before_grid = max(before_fac - re, 0.0)
        after_grid = max(after_fac - re, 0.0)
        before_export = min(max(re - before_fac, 0.0), ex_lim)
        after_export = min(max(re - after_fac, 0.0), ex_lim)
        before_curtail = re - min(before_fac, re) - before_export
        after_curtail = re - min(after_fac, re) - after_export

        d_cost += (
            env["price"][h, r] * (after_grid - before_grid)
            - env["sell"][h, r] * (after_export - before_export)
        )
        d_carbon += env["carbon"][h, r] * (after_grid - before_grid)
        d_unused += (after_curtail - before_curtail) / global_re

    return {
        "Cost": float(d_cost),
        "Carbon": float(d_carbon),
        "RenewableUnusedRate": float(d_unused),
    }


def _heuristic_rolling_deviation(
    *,
    calibration: GlobalCalibration,
    past_delta: dict[str, float],
    baseline_scope: dict[str, float],
    scope_values: dict[str, float],
) -> tuple[float, dict[str, float], dict[str, float]]:
    estimated: dict[str, float] = {}
    deviation: dict[str, float] = {}
    for metric in OBJECTIVE_NAMES:
        estimated[metric] = (
            float(calibration.baseline_value[metric])
            + float(past_delta.get(metric, 0.0))
            + float(scope_values.get(metric, 0.0))
            - float(baseline_scope[metric])
        )
        if metric in calibration.active_metrics:
            deviation[metric] = (
                estimated[metric] - float(calibration.ideal_lb[metric])
            ) / float(calibration.scale[metric])
    z = max(deviation.values()) if deviation else float("nan")
    return float(z), estimated, deviation


def _heuristic_assignment_row(
    *,
    task_id: str,
    task_class: ExactTaskClass,
    target_region: str,
    network_latency_ms: float,
    start_hour: int,
    solve_method: str,
    heuristic_score: float = float("nan"),
) -> dict[str, Any]:
    migrated = target_region != task_class.source_region
    return {
        "TaskID": str(task_id),
        "TaskType": task_class.task_type,
        "ArrivalHour": task_class.arrival_hour,
        "SourceRegion": task_class.source_region,
        "TargetRegion": target_region,
        "NetworkLatency_ms": float(network_latency_ms),
        "MaxLatency_ms": task_class.max_latency_ms,
        "StartHour": int(start_hour),
        "FinishHour": float(start_hour + task_class.duration_h),
        "Duration_h": task_class.duration_h,
        "GPU_Demand": task_class.gpu_demand,
        "Task_Full_IT_Power_MW": task_class.task_power_mw,
        "WaitHours": (
            0.0
            if task_class.task_type == "RealTimeInference"
            else max(0.0, float(start_hour - task_class.earliest_start_hour))
        ),
        "Migration_GPU_Workload_GPUh": (
            task_class.gpu_demand * task_class.duration_h if migrated else 0.0
        ),
        "IsMigrated": int(migrated),
        "ExactTaskClassID": task_class.class_id,
        "SolveMethod": solve_method,
        "HeuristicScore": float(heuristic_score),
    }


def _heuristic_class_candidates(
    task_class: ExactTaskClass,
    *,
    tau: int,
    decision_end: int,
    plan_end: int,
) -> list[tuple[str, float, int, tuple[tuple[int, float], ...]]]:
    starts = _window_valid_start_hours(
        task_class, tau, decision_end, plan_end
    )
    result: list[tuple[str, float, int, tuple[tuple[int, float], ...]]] = []
    for region, latency in task_class.candidate_regions:
        for start in starts:
            overlaps = _overlap_by_hour(float(start), task_class.duration_h)
            if overlaps:
                result.append((region, float(latency), int(start), overlaps))
    return result


def _heuristic_candidate_feasible(
    *,
    env: dict[str, Any],
    gpu: np.ndarray,
    ai: np.ndarray,
    task_class: ExactTaskClass,
    target_region: str,
    overlaps: tuple[tuple[int, float], ...],
) -> tuple[bool, float]:
    r = env["region_index"][target_region]
    pressure = 0.0
    for h, overlap in overlaps:
        new_gpu = gpu[h, r] + task_class.gpu_demand * overlap
        new_ai = ai[h, r] + task_class.task_power_mw * overlap
        if (
            new_gpu > env["available_gpu"][h, r] + HEURISTIC_CAPACITY_TOL
            or new_ai > env["effective_ai"][h, r] + HEURISTIC_CAPACITY_TOL
        ):
            return False, float("inf")
        gpu_cap = max(env["available_gpu"][h, r], FLOAT_EPS)
        ai_cap = max(env["effective_ai"][h, r], FLOAT_EPS)
        pressure = max(
            pressure,
            new_gpu / gpu_cap,
            new_ai / ai_cap,
        )
    return True, float(pressure)


def _heuristic_place_row(
    row: dict[str, Any],
    *,
    placements: dict[str, dict[str, Any]],
    gpu: np.ndarray,
    ai: np.ndarray,
    slot_tasks: dict[tuple[int, int], set[str]],
    env: dict[str, Any],
) -> None:
    task_id = str(row["TaskID"])
    placements[task_id] = row
    _heuristic_add_row_load(
        row,
        gpu=gpu,
        ai=ai,
        region_index=env["region_index"],
        sign=1.0,
    )
    r = env["region_index"][str(row["TargetRegion"])]
    for h, _ in _overlap_by_hour(float(row["StartHour"]), float(row["Duration_h"])):
        slot_tasks.setdefault((h, r), set()).add(task_id)


def _heuristic_rebuild_state(
    *,
    committed_gpu: np.ndarray,
    committed_ai: np.ndarray,
    placements: dict[str, dict[str, Any]],
    env: dict[str, Any],
    tau: int,
    objective_end: int,
    global_task_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], set[str]], dict[str, float]]:
    gpu = committed_gpu.copy()
    ai = committed_ai.copy()
    slot_tasks: dict[tuple[int, int], set[str]] = {}
    latency_sum = 0.0
    for row in placements.values():
        _heuristic_add_row_load(
            row,
            gpu=gpu,
            ai=ai,
            region_index=env["region_index"],
            sign=1.0,
        )
        r = env["region_index"][str(row["TargetRegion"])]
        for h, _ in _overlap_by_hour(float(row["StartHour"]), float(row["Duration_h"])):
            slot_tasks.setdefault((h, r), set()).add(str(row["TaskID"]))
        latency_sum += float(row["NetworkLatency_ms"])
    scope = _heuristic_scope_energy(env, ai, tau, objective_end)
    scope["MeanLatency"] = latency_sum / max(global_task_count, 1)
    return gpu, ai, slot_tasks, scope


def _materialize_integer_model_plan(
    *,
    model: AggregatedModel,
    vector: np.ndarray,
    class_lookup: dict[int, ExactTaskClass],
    task_ids_by_class: dict[int, list[str]],
    solve_method: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    local_ids = {cid: list(ids) for cid, ids in task_ids_by_class.items()}
    assigned_count = {cid: 0 for cid in local_ids}
    for j, option in enumerate(model.options):
        value = float(vector[j])
        count = int(round(value))
        if count <= 0:
            continue
        if abs(value - count) > INTEGER_TOL:
            raise RuntimeError("局部整数模型返回了非整数任务计数")
        ids = local_ids.get(option.class_id, [])
        offset = assigned_count.get(option.class_id, 0)
        if offset + count > len(ids):
            raise RuntimeError(f"局部模型任务类{option.class_id}计数超过TaskID池")
        task_class = class_lookup[option.class_id]
        for task_id in ids[offset:offset + count]:
            rows.append(
                _heuristic_assignment_row(
                    task_id=task_id,
                    task_class=task_class,
                    target_region=option.target_region,
                    network_latency_ms=option.network_latency_ms,
                    start_hour=option.start_hour,
                    solve_method=solve_method,
                )
            )
        assigned_count[option.class_id] = offset + count

    for cid, ids in local_ids.items():
        if assigned_count.get(cid, 0) != len(ids):
            raise RuntimeError(
                f"局部模型任务类{cid}未完整落到TaskID："
                f"{assigned_count.get(cid, 0)}/{len(ids)}"
            )
    return pd.DataFrame(rows)


def _heuristic_repair_classes(
    *,
    task_class: ExactTaskClass,
    candidates: list[tuple[str, float, int, tuple[tuple[int, float], ...]]],
    placements: dict[str, dict[str, Any]],
    slot_tasks: dict[tuple[int, int], set[str]],
    gpu: np.ndarray,
    ai: np.ndarray,
    env: dict[str, Any],
    max_classes: int,
) -> set[int]:
    """从最接近可行的候选时空槽中提取真正造成容量冲突的任务类。"""
    ranked: list[tuple[float, set[str]]] = []
    for region, _lat, _start, overlaps in candidates:
        r = env["region_index"][region]
        violation = 0.0
        blockers: set[str] = set()
        for h, overlap in overlaps:
            new_gpu = gpu[h, r] + task_class.gpu_demand * overlap
            new_ai = ai[h, r] + task_class.task_power_mw * overlap
            gpu_cap = max(env["available_gpu"][h, r], FLOAT_EPS)
            ai_cap = max(env["effective_ai"][h, r], FLOAT_EPS)
            violation += max(0.0, new_gpu - gpu_cap) / gpu_cap
            violation += max(0.0, new_ai - ai_cap) / ai_cap
            if (
                new_gpu > gpu_cap + HEURISTIC_CAPACITY_TOL
                or new_ai > ai_cap + HEURISTIC_CAPACITY_TOL
            ):
                blockers.update(slot_tasks.get((h, r), set()))
        ranked.append((violation, blockers))
    ranked.sort(key=lambda x: x[0])

    frequency: dict[int, int] = {}
    for _, blockers in ranked[: min(16, len(ranked))]:
        for task_id in blockers:
            row = placements.get(str(task_id))
            if row is None:
                continue
            cid = int(row["ExactTaskClassID"])
            frequency[cid] = frequency.get(cid, 0) + 1

    ordered = sorted(frequency, key=lambda cid: (-frequency[cid], cid))
    selected = {task_class.class_id}
    selected.update(ordered[: max(0, max_classes - 1)])
    return selected


def _heuristic_repair_with_local_milp(
    *,
    bundle: InputBundle,
    classes: list[ExactTaskClass],
    class_lookup: dict[int, ExactTaskClass],
    pools: dict[int, list[str]],
    committed_assignments: pd.DataFrame,
    placements: dict[str, dict[str, Any]],
    selected_classes: set[int],
    tau: int,
    decision_end: int,
    plan_end: int,
    time_limit: float,
    progress_interval: float,
) -> pd.DataFrame | None:
    """只重开冲突任务类，以零目标小MILP做严格容量修复。"""
    remaining = {task_class.class_id: 0 for task_class in classes}
    task_ids_by_class: dict[int, list[str]] = {}
    for cid in selected_classes:
        ids = list(pools[cid])
        if not ids:
            continue
        remaining[cid] = len(ids)
        task_ids_by_class[cid] = ids
    if not task_ids_by_class:
        return None

    fixed_rows = [
        row for row in placements.values()
        if int(row["ExactTaskClassID"]) not in selected_classes
    ]
    fixed_df = committed_assignments
    if fixed_rows:
        fixed_df = pd.concat(
            [committed_assignments, pd.DataFrame(fixed_rows)],
            ignore_index=True,
        )

    try:
        model = _build_aggregated_model(
            bundle=bundle,
            classes=classes,
            class_lookup=class_lookup,
            remaining_count=remaining,
            fixed_assignments=fixed_df,
            tau=tau,
            decision_end=decision_end,
            plan_end=plan_end,
            include_z=False,
            include_energy_variables=False,
            integerize_lookahead=True,
        )
        objective = np.zeros(len(model.variable_lower), dtype=np.float64)
        result = _solve_milp_adaptive(
            model,
            objective,
            label=f"启发式容量修复-{len(task_ids_by_class)}类",
            mip_rel_gap=0.0,
            initial_time_limit=time_limit,
            max_time_limit=time_limit,
            progress_interval=progress_interval,
            retry_emergency=False,
        )
        repaired = _materialize_integer_model_plan(
            model=model,
            vector=result.vector,
            class_lookup=class_lookup,
            task_ids_by_class=task_ids_by_class,
            solve_method="heuristic_capacity_repair_milp",
        )
        del result, model
        gc.collect()
        return repaired
    except (RuntimeError, ValueError, MemoryError) as exc:
        logging.info(
            "启发式局部容量修复未成功：类=%d，限时=%.0fs，原因=%s",
            len(task_ids_by_class), time_limit, exc,
        )
        return None


def _construct_heuristic_window_plan(
    *,
    bundle: InputBundle,
    classes: list[ExactTaskClass],
    class_lookup: dict[int, ExactTaskClass],
    pools: dict[int, list[str]],
    committed_assignments: pd.DataFrame,
    committed_gpu: np.ndarray,
    committed_ai: np.ndarray,
    env: dict[str, Any],
    baseline_assignments: pd.DataFrame,
    calibration: GlobalCalibration,
    past_delta: dict[str, float],
    baseline_scope: dict[str, float],
    tau: int,
    decision_end: int,
    plan_end: int,
    progress_interval: float,
) -> tuple[pd.DataFrame, float, dict[str, float], dict[str, Any]]:
    """按原 min-max 标准化目标做紧迫度排序 + 边际代价快速构造。"""
    global_task_count = len(bundle.tasks)
    plan_gpu = committed_gpu.copy()
    plan_ai = committed_ai.copy()
    placements: dict[str, dict[str, Any]] = {}
    slot_tasks: dict[tuple[int, int], set[str]] = {}
    scope = _heuristic_scope_energy(env, plan_ai, tau, plan_end)
    scope["MeanLatency"] = 0.0

    baseline_lookup = {
        str(row.TaskID): (str(row.TargetRegion), int(round(float(row.StartHour))))
        for row in baseline_assignments.itertuples(index=False)
    }

    candidate_cache: dict[int, list[tuple[str, float, int, tuple[tuple[int, float], ...]]]] = {}
    entries: list[tuple[tuple[Any, ...], str, int]] = []
    candidate_option_count = 0
    for task_class in classes:
        cid = task_class.class_id
        if not pools[cid] or task_class.arrival_hour >= plan_end:
            continue
        candidates = _heuristic_class_candidates(
            task_class,
            tau=tau,
            decision_end=decision_end,
            plan_end=plan_end,
        )
        if not candidates:
            if _latest_start_for_class(task_class) < decision_end:
                raise RuntimeError(
                    f"启发式窗口tau={tau}：任务类{cid}已经必须开工但没有合法候选"
                )
            continue
        candidate_cache[cid] = candidates
        candidate_option_count += len(candidates) * len(pools[cid])

        latest_start = _latest_start_for_class(task_class)
        first_start = (
            task_class.arrival_hour
            if task_class.task_type == "RealTimeInference"
            else max(tau, task_class.arrival_hour, task_class.earliest_start_hour)
        )
        slack = max(0, latest_start - first_start)
        urgent = int(latest_start >= decision_end)
        future = int(task_class.arrival_hour >= decision_end)
        realtime = 0 if task_class.task_type == "RealTimeInference" else 1
        resource_work = task_class.gpu_demand * task_class.duration_h
        # current/urgent first；时间窗越窄、资源越大越优先。
        priority = (
            future,
            urgent,
            realtime,
            slack,
            len(candidates),
            -resource_work,
            cid,
        )
        for task_id in pools[cid]:
            entries.append((priority, str(task_id), cid))
    entries.sort(key=lambda x: x[0])

    repair_count = 0
    placed_direct = 0
    for index, (_priority, task_id, cid) in enumerate(entries, start=1):
        if task_id in placements:
            continue
        task_class = class_lookup[cid]
        candidates = candidate_cache[cid]
        best_key: tuple[Any, ...] | None = None
        best_payload: tuple[str, float, int, tuple[tuple[int, float], ...], dict[str, float], float] | None = None

        for region, latency, start, overlaps in candidates:
            feasible, pressure = _heuristic_candidate_feasible(
                env=env,
                gpu=plan_gpu,
                ai=plan_ai,
                task_class=task_class,
                target_region=region,
                overlaps=overlaps,
            )
            if not feasible:
                continue
            energy_delta = _heuristic_candidate_energy_delta(
                env=env,
                ai=plan_ai,
                task_class=task_class,
                target_region=region,
                start_hour=start,
                objective_end=plan_end,
            )
            delta = {
                **energy_delta,
                "MeanLatency": float(latency) / max(global_task_count, 1),
            }
            deviations = []
            for metric in calibration.active_metrics:
                estimated = (
                    float(calibration.baseline_value[metric])
                    + float(past_delta.get(metric, 0.0))
                    + float(scope.get(metric, 0.0))
                    + float(delta.get(metric, 0.0))
                    - float(baseline_scope[metric])
                )
                deviations.append(
                    (estimated - float(calibration.ideal_lb[metric]))
                    / float(calibration.scale[metric])
                )
            score = max(deviations) if deviations else 0.0
            baseline_region, baseline_start = baseline_lookup.get(task_id, ("", -10**9))
            baseline_penalty = int(not (region == baseline_region and start == baseline_start))
            wait = (
                0.0 if task_class.task_type == "RealTimeInference"
                else max(0.0, start - task_class.earliest_start_hour)
            )
            migrated = int(region != task_class.source_region)
            key = (
                round(float(score), 12),
                round(float(pressure), 12),
                baseline_penalty,
                wait,
                migrated,
                start,
                region,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_payload = (
                    region, latency, start, overlaps, delta, float(score)
                )

        if best_payload is None:
            # 贪心被局部容量卡住时，不回退到几十万变量全窗口MILP；
            # 只提取真正占用冲突时空槽的任务类，做一个小规模零目标MILP修复。
            selected = _heuristic_repair_classes(
                task_class=task_class,
                candidates=candidates,
                placements=placements,
                slot_tasks=slot_tasks,
                gpu=plan_gpu,
                ai=plan_ai,
                env=env,
                max_classes=HEURISTIC_REPAIR_MAX_CLASSES,
            )
            repaired = _heuristic_repair_with_local_milp(
                bundle=bundle,
                classes=classes,
                class_lookup=class_lookup,
                pools=pools,
                committed_assignments=committed_assignments,
                placements=placements,
                selected_classes=selected,
                tau=tau,
                decision_end=decision_end,
                plan_end=plan_end,
                time_limit=HEURISTIC_REPAIR_TIME_LIMIT,
                progress_interval=progress_interval,
            )
            if repaired is None:
                selected = _heuristic_repair_classes(
                    task_class=task_class,
                    candidates=candidates,
                    placements=placements,
                    slot_tasks=slot_tasks,
                    gpu=plan_gpu,
                    ai=plan_ai,
                    env=env,
                    max_classes=HEURISTIC_REPAIR_EXPANDED_MAX_CLASSES,
                )
                repaired = _heuristic_repair_with_local_milp(
                    bundle=bundle,
                    classes=classes,
                    class_lookup=class_lookup,
                    pools=pools,
                    committed_assignments=committed_assignments,
                    placements=placements,
                    selected_classes=selected,
                    tau=tau,
                    decision_end=decision_end,
                    plan_end=plan_end,
                    time_limit=HEURISTIC_REPAIR_EXPANDED_TIME_LIMIT,
                    progress_interval=progress_interval,
                )
            if repaired is None:
                raise RuntimeError(
                    f"窗口tau={tau}快速构造在任务{task_id}/类{cid}处无法恢复可行性；"
                    "已尝试局部容量修复，不会修改上一检查点。"
                )

            # 删除修复邻域原先的安排，并用局部MILP结果替换；随后重建小型状态数组。
            for old_task_id, row in list(placements.items()):
                if int(row["ExactTaskClassID"]) in selected:
                    placements.pop(old_task_id, None)
            for row in repaired.to_dict("records"):
                placements[str(row["TaskID"])] = row
            plan_gpu, plan_ai, slot_tasks, scope = _heuristic_rebuild_state(
                committed_gpu=committed_gpu,
                committed_ai=committed_ai,
                placements=placements,
                env=env,
                tau=tau,
                objective_end=plan_end,
                global_task_count=global_task_count,
            )
            repair_count += 1
            continue

        region, latency, start, _overlaps, delta, score = best_payload
        row = _heuristic_assignment_row(
            task_id=task_id,
            task_class=task_class,
            target_region=region,
            network_latency_ms=latency,
            start_hour=start,
            solve_method="rolling_minmax_marginal_greedy",
            heuristic_score=score,
        )
        _heuristic_place_row(
            row,
            placements=placements,
            gpu=plan_gpu,
            ai=plan_ai,
            slot_tasks=slot_tasks,
            env=env,
        )
        for metric, value in delta.items():
            scope[metric] = float(scope.get(metric, 0.0)) + float(value)
        placed_direct += 1

        if index % 1000 == 0:
            logging.info(
                "启发式窗口tau=%d构造进度：%d/%d，直接放置=%d，容量修复=%d。",
                tau, index, len(entries), placed_direct, repair_count,
            )

    # 所有进入当前H+K规划域的任务都必须有一份临时计划。
    expected_ids = {task_id for _, task_id, _ in entries}
    missing = expected_ids - set(placements)
    if missing:
        raise RuntimeError(
            f"启发式窗口tau={tau}规划不完整：仍缺{len(missing)}个活动任务"
        )

    plan_df = pd.DataFrame(list(placements.values()))
    z, estimated, deviations = _heuristic_rolling_deviation(
        calibration=calibration,
        past_delta=past_delta,
        baseline_scope=baseline_scope,
        scope_values=scope,
    )
    stats = {
        "CandidateOptionCount": int(candidate_option_count),
        "ActiveTaskCount": len(expected_ids),
        "DirectGreedyPlaced": int(placed_direct),
        "RepairMILPCount": int(repair_count),
        "EstimatedZBeforeLNS": float(z),
        **{f"EstimatedBeforeLNS_{m}": estimated[m] for m in OBJECTIVE_NAMES},
        **{f"EstimatedDBeforeLNS_{m}": deviations.get(m, np.nan) for m in OBJECTIVE_NAMES},
    }
    return plan_df, float(z), scope, stats


def _lns_select_classes(
    *,
    plan: pd.DataFrame,
    class_lookup: dict[int, ExactTaskClass],
    env: dict[str, Any],
    gpu: np.ndarray,
    ai: np.ndarray,
    decision_end: int,
    max_classes: int,
) -> set[int]:
    if plan.empty or max_classes <= 0:
        return set()
    score_by_class: dict[int, float] = {}
    for row in plan.itertuples(index=False):
        cid = int(row.ExactTaskClassID)
        task_class = class_lookup[cid]
        # 实时任务的开始时刻不可改变，除非只允许区域迁移；仍可入邻域，但优先级较低。
        r = env["region_index"][str(row.TargetRegion)]
        pressure = 0.0
        for h, _overlap in _overlap_by_hour(float(row.StartHour), float(row.Duration_h)):
            pressure = max(
                pressure,
                gpu[h, r] / max(env["available_gpu"][h, r], FLOAT_EPS),
                ai[h, r] / max(env["effective_ai"][h, r], FLOAT_EPS),
            )
        h_bonus = 0.25 if float(row.StartHour) < decision_end else 0.0
        flexibility = max(
            0,
            _latest_start_for_class(task_class)
            - max(task_class.arrival_hour, task_class.earliest_start_hour),
        )
        flex_bonus = min(flexibility / 48.0, 1.0) * 0.10
        current = (
            pressure
            + h_bonus
            + flex_bonus
            + 0.02 * float(getattr(row, "IsMigrated", 0))
        )
        score_by_class[cid] = max(score_by_class.get(cid, -np.inf), current)
    ordered = sorted(score_by_class, key=lambda cid: (-score_by_class[cid], cid))
    return set(ordered[:max_classes])


def _solve_lns_refinement(
    *,
    bundle: InputBundle,
    classes: list[ExactTaskClass],
    class_lookup: dict[int, ExactTaskClass],
    pools: dict[int, list[str]],
    committed_assignments: pd.DataFrame,
    heuristic_plan: pd.DataFrame,
    selected_classes: set[int],
    calibration: GlobalCalibration,
    past_delta: dict[str, float],
    baseline_scope: dict[str, float],
    tau: int,
    decision_end: int,
    plan_end: int,
    progress_interval: float,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """固定邻域外启发式方案，仅对关键任务类求原 min-max 小MILP。"""
    if not selected_classes:
        return None, {"LNSStatus": "skipped_empty"}

    remaining = {task_class.class_id: 0 for task_class in classes}
    task_ids_by_class: dict[int, list[str]] = {}
    for cid in selected_classes:
        ids = list(pools[cid])
        if ids:
            remaining[cid] = len(ids)
            task_ids_by_class[cid] = ids
    if not task_ids_by_class:
        return None, {"LNSStatus": "skipped_no_tasks"}

    fixed_outside = heuristic_plan.loc[
        ~heuristic_plan["ExactTaskClassID"].astype(int).isin(selected_classes)
    ].copy()
    fixed_df = committed_assignments
    if not fixed_outside.empty:
        fixed_df = pd.concat([committed_assignments, fixed_outside], ignore_index=True)

    try:
        model = _build_aggregated_model(
            bundle=bundle,
            classes=classes,
            class_lookup=class_lookup,
            remaining_count=remaining,
            fixed_assignments=fixed_df,
            tau=tau,
            decision_end=decision_end,
            plan_end=plan_end,
            include_z=True,
            include_energy_variables=True,
            integerize_lookahead=True,
        )
        if model.z_index is None:
            return None, {"LNSStatus": "failed_no_z"}

        # 能源固定项已由 fixed_assignments 进入紧凑B/W模型；
        # 时延是可分项，需把邻域外活动任务的固定时延显式加入scope常数。
        model.objective_constants["MeanLatency"] += (
            float(fixed_outside["NetworkLatency_ms"].astype(float).sum())
            / max(len(bundle.tasks), 1)
            if not fixed_outside.empty else 0.0
        )

        extra_rows: list[tuple[dict[int, float], float, float]] = []
        for metric in calibration.active_metrics:
            scale = float(calibration.scale[metric])
            raw = model.objective_vectors[metric]
            terms = {
                idx: float(value / scale)
                for idx, value in enumerate(raw)
                if abs(value) > FLOAT_EPS
            }
            terms[model.z_index] = terms.get(model.z_index, 0.0) - 1.0
            constant = (
                float(calibration.baseline_value[metric])
                + float(past_delta.get(metric, 0.0))
                + float(model.objective_constants[metric])
                - float(baseline_scope[metric])
                - float(calibration.ideal_lb[metric])
            ) / scale
            extra_rows.append((terms, -np.inf, -constant))

        objective = np.zeros(len(model.variable_lower), dtype=np.float64)
        objective[model.z_index] = 1.0
        result = _solve_milp_adaptive(
            model,
            objective,
            label=f"窗口{tau//max(1, decision_end-tau)}-LNS精修-{len(task_ids_by_class)}类",
            mip_rel_gap=HEURISTIC_LNS_MIP_GAP,
            initial_time_limit=HEURISTIC_LNS_TIME_LIMIT,
            max_time_limit=HEURISTIC_LNS_TIME_LIMIT,
            progress_interval=progress_interval,
            retry_emergency=False,
            extra_rows=extra_rows,
        )
        refined_neighborhood = _materialize_integer_model_plan(
            model=model,
            vector=result.vector,
            class_lookup=class_lookup,
            task_ids_by_class=task_ids_by_class,
            solve_method="large_neighborhood_milp_refinement",
        )
        candidate = pd.concat(
            [fixed_outside, refined_neighborhood],
            ignore_index=True,
        )
        info = {
            "LNSStatus": "feasible",
            "LNSClassCount": len(task_ids_by_class),
            "LNSCandidateOptionCount": model.option_count,
            "LNSIntegerOptionCount": int(
                np.sum(model.integrality[:model.option_count] == 1)
            ),
            "LNSMIPGap": result.mip_gap,
            "LNSZSolver": float(result.vector[model.z_index]),
            "LNSElapsedSeconds": result.elapsed_seconds,
        }
        del result, model
        gc.collect()
        return candidate, info
    except (RuntimeError, ValueError, MemoryError) as exc:
        logging.info("大邻域MILP精修未得到可用解，保留贪心方案：%s", exc)
        return None, {
            "LNSStatus": "failed_keep_greedy",
            "LNSClassCount": len(task_ids_by_class),
            "LNSError": str(exc),
        }


def _evaluate_heuristic_plan(
    *,
    plan: pd.DataFrame,
    committed_gpu: np.ndarray,
    committed_ai: np.ndarray,
    env: dict[str, Any],
    calibration: GlobalCalibration,
    past_delta: dict[str, float],
    baseline_scope: dict[str, float],
    tau: int,
    plan_end: int,
    global_task_count: int,
) -> tuple[float, dict[str, float], dict[str, float], np.ndarray, np.ndarray]:
    if not plan.empty and plan["TaskID"].astype(str).duplicated().any():
        raise RuntimeError("启发式候选计划存在重复TaskID")
    gpu = committed_gpu.copy()
    ai = committed_ai.copy()
    latency_sum = 0.0
    if not plan.empty:
        for row in plan.itertuples(index=False):
            _heuristic_add_row_load(
                row,
                gpu=gpu,
                ai=ai,
                region_index=env["region_index"],
                sign=1.0,
            )
            latency_sum += float(row.NetworkLatency_ms)

    # 独立容量复核；不相信构造器/局部MILP内部状态。
    gpu_violation = float(np.maximum(gpu - env["available_gpu"], 0.0).max())
    ai_violation = float(np.maximum(ai - env["effective_ai"], 0.0).max())
    if gpu_violation > 1e-6 or ai_violation > 1e-6:
        raise RuntimeError(
            f"启发式候选独立容量复核失败：GPU={gpu_violation}, AI={ai_violation}"
        )

    scope = _heuristic_scope_energy(env, ai, tau, plan_end)
    scope["MeanLatency"] = latency_sum / max(global_task_count, 1)
    z, estimated, deviations = _heuristic_rolling_deviation(
        calibration=calibration,
        past_delta=past_delta,
        baseline_scope=baseline_scope,
        scope_values=scope,
    )
    return z, estimated, deviations, gpu, ai


def _commit_heuristic_h_plan(
    *,
    plan: pd.DataFrame,
    decision_end: int,
    pools: dict[int, list[str]],
) -> pd.DataFrame:
    if plan.empty:
        return pd.DataFrame()
    committed = plan.loc[
        plan["StartHour"].astype(float) < float(decision_end) - FLOAT_EPS
    ].copy()
    if committed.empty:
        return committed

    for row in committed.itertuples(index=False):
        cid = int(row.ExactTaskClassID)
        task_id = str(row.TaskID)
        if task_id not in pools[cid]:
            raise RuntimeError(f"启发式提交时TaskID {task_id}不在任务类{cid}剩余池")
        pools[cid].remove(task_id)

    committed["TaskID"] = committed["TaskID"].astype(str)
    committed["TaskID_num"] = pd.to_numeric(committed["TaskID"], errors="coerce")
    return (
        committed.sort_values(["TaskID_num", "TaskID"], kind="stable")
        .drop(columns="TaskID_num")
        .reset_index(drop=True)
    )


def _run_balanced_rollout(
    *,
    bundle: InputBundle,
    classes: list[ExactTaskClass],
    class_lookup: dict[int, ExactTaskClass],
    baseline_assignments: pd.DataFrame,
    baseline_profile: pd.DataFrame,
    calibration: GlobalCalibration,
    decision_window: int,
    lookahead: int,
    mip_rel_gap: float,
    initial_time_limit: float,
    max_time_limit: float,
    progress_interval: float,
    resume: bool,
    max_windows: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """新阶段4：恢复原 balanced v3 历史，然后从下一未完成窗口切换数学启发式。

    关键兼容性：
    - CHECKPOINT_SCHEMA_VERSION 保持3；
    - checkpoint文件名前缀仍为 q2_refactored_balanced_v3；
    - H/K 校验仍由 _load_checkpoint 原样执行；
    - 已提交窗口绝不重算、不删除，只把它们作为固定历史负荷。
    """
    pools = _make_task_pools(classes)
    assignments = pd.DataFrame()
    window_records: list[dict[str, Any]] = []
    solver_records: list[dict[str, Any]] = []
    tau = 0
    window_id = 0
    baseline_latency = _baseline_latency_lookup(baseline_assignments)
    global_task_count = len(bundle.tasks)

    if resume:
        checkpoint = _load_checkpoint(bundle, "balanced", decision_window, lookahead)
        if checkpoint is not None:
            assignments = checkpoint["assignments"]
            window_records = checkpoint["windows"]
            solver_records = checkpoint["solver"]
            tau = int(checkpoint["state"]["next_tau"])
            window_id = int(checkpoint["state"]["next_window_id"])
            _remove_committed_from_pools(pools, assignments)
            if checkpoint["state"].get("status") == "COMPLETED":
                profile, metrics = _schedule_profile(bundle, assignments)
                return assignments, pd.DataFrame(window_records), pd.DataFrame(solver_records), metrics
            logging.info(
                "恢复均衡滚动：已完成任务=%d，下一窗口=%d，tau=%d；"
                "既有窗口全部保留，从该窗口起改用滚动数学启发式。",
                len(assignments), window_id, tau,
            )

    env = _heuristic_resource_environment(bundle)
    committed_gpu, committed_ai = _heuristic_initial_committed_loads(assignments, env)

    while any(pools.values()) and tau < TERMINAL_HOUR:
        if max_windows is not None and window_id >= max_windows:
            break

        decision_end = min(tau + decision_window, TERMINAL_HOUR)
        plan_end = min(decision_end + lookahead, TERMINAL_HOUR)

        # 当前H+K真正进入规划域的任务集合，用于滚动全局指标修正。
        active_task_ids: set[str] = set()
        active_class_count = 0
        for task_class in classes:
            if not pools[task_class.class_id] or task_class.arrival_hour >= plan_end:
                continue
            starts = _window_valid_start_hours(
                task_class, tau, decision_end, plan_end
            )
            if starts:
                active_class_count += 1
                active_task_ids.update(str(x) for x in pools[task_class.class_id])

        if not active_task_ids:
            tau = decision_end
            window_id += 1
            continue

        past = _past_delta(
            bundle=bundle,
            balanced_committed=assignments,
            baseline_assignments=baseline_assignments,
            baseline_latency=baseline_latency,
            tau=tau,
            global_task_count=global_task_count,
        )
        base_scope = _baseline_scope_value(
            bundle=bundle,
            baseline_assignments=baseline_assignments,
            baseline_profile=baseline_profile,
            baseline_latency=baseline_latency,
            active_task_ids=active_task_ids,
            tau=tau,
            objective_end=plan_end,
            global_task_count=global_task_count,
        )

        started = time.perf_counter()
        heuristic_started = time.perf_counter()
        plan, z_greedy, _scope, construct_stats = _construct_heuristic_window_plan(
            bundle=bundle,
            classes=classes,
            class_lookup=class_lookup,
            pools=pools,
            committed_assignments=assignments,
            committed_gpu=committed_gpu,
            committed_ai=committed_ai,
            env=env,
            baseline_assignments=baseline_assignments,
            calibration=calibration,
            past_delta=past,
            baseline_scope=base_scope,
            tau=tau,
            decision_end=decision_end,
            plan_end=plan_end,
            progress_interval=progress_interval,
        )
        construct_elapsed = time.perf_counter() - heuristic_started

        z_verified, estimated_before, deviations_before, plan_gpu, plan_ai = (
            _evaluate_heuristic_plan(
                plan=plan,
                committed_gpu=committed_gpu,
                committed_ai=committed_ai,
                env=env,
                calibration=calibration,
                past_delta=past,
                baseline_scope=base_scope,
                tau=tau,
                plan_end=plan_end,
                global_task_count=global_task_count,
            )
        )
        if abs(z_verified - z_greedy) > 1e-6 * max(1.0, abs(z_verified)):
            logging.info(
                "窗口%d构造器增量z与独立复算略有差异：增量=%.9g，复算=%.9g；"
                "以后者为准。",
                window_id, z_greedy, z_verified,
            )
        z_before = z_verified

        lns_info: dict[str, Any] = {"LNSStatus": "disabled"}
        if HEURISTIC_LNS_ENABLED:
            selected = _lns_select_classes(
                plan=plan,
                class_lookup=class_lookup,
                env=env,
                gpu=plan_gpu,
                ai=plan_ai,
                decision_end=decision_end,
                max_classes=HEURISTIC_LNS_MAX_CLASSES,
            )
            refined, lns_info = _solve_lns_refinement(
                bundle=bundle,
                classes=classes,
                class_lookup=class_lookup,
                pools=pools,
                committed_assignments=assignments,
                heuristic_plan=plan,
                selected_classes=selected,
                calibration=calibration,
                past_delta=past,
                baseline_scope=base_scope,
                tau=tau,
                decision_end=decision_end,
                plan_end=plan_end,
                progress_interval=progress_interval,
            )
            if refined is not None:
                try:
                    z_refined, estimated_refined, deviations_refined, _, _ = (
                        _evaluate_heuristic_plan(
                            plan=refined,
                            committed_gpu=committed_gpu,
                            committed_ai=committed_ai,
                            env=env,
                            calibration=calibration,
                            past_delta=past,
                            baseline_scope=base_scope,
                            tau=tau,
                            plan_end=plan_end,
                            global_task_count=global_task_count,
                        )
                    )
                    if z_refined < z_before - HEURISTIC_SCORE_TOL:
                        logging.info(
                            "均衡窗口%d LNS精修接受：z %.9g → %.9g，邻域类=%s。",
                            window_id, z_before, z_refined,
                            lns_info.get("LNSClassCount", np.nan),
                        )
                        plan = refined
                        z_final = z_refined
                        estimated_final = estimated_refined
                        deviations_final = deviations_refined
                        lns_info["LNSAccepted"] = 1
                    else:
                        z_final = z_before
                        estimated_final = estimated_before
                        deviations_final = deviations_before
                        lns_info["LNSAccepted"] = 0
                        logging.info(
                            "均衡窗口%d LNS精修未改善独立复算z：候选=%.9g，原=%.9g；"
                            "保留快速构造方案。",
                            window_id, z_refined, z_before,
                        )
                except RuntimeError as exc:
                    z_final = z_before
                    estimated_final = estimated_before
                    deviations_final = deviations_before
                    lns_info["LNSAccepted"] = 0
                    lns_info["LNSIndependentCheckError"] = str(exc)
            else:
                z_final = z_before
                estimated_final = estimated_before
                deviations_final = deviations_before
        else:
            z_final = z_before
            estimated_final = estimated_before
            deviations_final = deviations_before

        # 最终窗口计划再次独立验容量，再提交H区。
        _evaluate_heuristic_plan(
            plan=plan,
            committed_gpu=committed_gpu,
            committed_ai=committed_ai,
            env=env,
            calibration=calibration,
            past_delta=past,
            baseline_scope=base_scope,
            tau=tau,
            plan_end=plan_end,
            global_task_count=global_task_count,
        )
        committed = _commit_heuristic_h_plan(
            plan=plan,
            decision_end=decision_end,
            pools=pools,
        )

        if committed.empty:
            # 若当前存在已到达或必须开工任务却没有提交，说明滚动策略发生停滞。
            must_start = any(
                pools[task_class.class_id]
                and task_class.arrival_hour < decision_end
                and _latest_start_for_class(task_class) < decision_end
                for task_class in classes
            )
            if must_start:
                raise RuntimeError(
                    f"均衡窗口{window_id}存在必须开工任务但H区没有提交，拒绝推进检查点"
                )

        if not committed.empty:
            assignments = pd.concat([assignments, committed], ignore_index=True)
            for row in committed.itertuples(index=False):
                _heuristic_add_row_load(
                    row,
                    gpu=committed_gpu,
                    ai=committed_ai,
                    region_index=env["region_index"],
                    sign=1.0,
                )

        remaining_after = sum(len(v) for v in pools.values())
        elapsed = time.perf_counter() - started
        solver_records.append(
            {
                "Mode": "balanced",
                "WindowID": window_id,
                "Stage": "rolling_minmax_marginal_greedy",
                "Status": 0,
                "Message": "strict_feasible_constructed",
                "MIPGap": np.nan,
                "MIPDualBound": np.nan,
                "ObjectiveValue": z_before,
                "ElapsedSeconds": construct_elapsed,
                "TimeLimitUsedSeconds": 0.0,
            }
        )
        if lns_info.get("LNSStatus") not in ("disabled", "skipped_empty", "skipped_no_tasks"):
            solver_records.append(
                {
                    "Mode": "balanced",
                    "WindowID": window_id,
                    "Stage": "large_neighborhood_milp_refinement",
                    "Status": 0 if lns_info.get("LNSStatus") == "feasible" else 1,
                    "Message": str(lns_info.get("LNSStatus")),
                    "MIPGap": lns_info.get("LNSMIPGap", np.nan),
                    "MIPDualBound": np.nan,
                    "ObjectiveValue": lns_info.get("LNSZSolver", np.nan),
                    "ElapsedSeconds": lns_info.get("LNSElapsedSeconds", 0.0),
                    "TimeLimitUsedSeconds": HEURISTIC_LNS_TIME_LIMIT,
                }
            )

        window_records.append(
            {
                "WindowID": window_id,
                "Mode": "balanced",
                "SolveMethod": "rolling_matheuristic_greedy_plus_lns_milp",
                "WindowStartHour": tau,
                "DecisionEndHourExclusive": decision_end,
                "PlanEndHourExclusive": plan_end,
                "ResourceEndHourExclusive": (
                    min(
                        TERMINAL_HOUR,
                        int(math.ceil(float(plan["FinishHour"].astype(float).max()) - FLOAT_EPS)),
                    )
                    if not plan.empty else plan_end
                ),
                "ActiveClassCount": active_class_count,
                "ActiveTaskCount": construct_stats["ActiveTaskCount"],
                "CandidateOptionCount": construct_stats["CandidateOptionCount"],
                "DecisionIntegerOptionCount": np.nan,
                "LookaheadContinuousOptionCount": np.nan,
                "CommittedTaskCount": len(committed),
                "RemainingTaskCount": remaining_after,
                "MIPGap": lns_info.get("LNSMIPGap", np.nan),
                "ElapsedSeconds": elapsed,
                "ZBeforeLNS": z_before,
                "ZStar": z_final,
                "EstimatedMaxNormalizedDeviation": z_final,
                "GreedyRepairMILPCount": construct_stats["RepairMILPCount"],
                "LNSStatus": lns_info.get("LNSStatus"),
                "LNSAccepted": lns_info.get("LNSAccepted", 0),
                "LNSClassCount": lns_info.get("LNSClassCount", 0),
                "LNSCandidateOptionCount": lns_info.get("LNSCandidateOptionCount", 0),
                **{f"EstimatedD_{m}": deviations_final.get(m, np.nan) for m in OBJECTIVE_NAMES},
                **{f"EstimatedGlobal_{m}": estimated_final.get(m, np.nan) for m in OBJECTIVE_NAMES},
                **{f"PastDelta_{m}": past.get(m, np.nan) for m in OBJECTIVE_NAMES},
                **{f"BaselineScope_{m}": base_scope.get(m, np.nan) for m in OBJECTIVE_NAMES},
            }
        )

        next_tau = decision_end
        # 仍写原 balanced v3 检查点；因此旧窗口与新窗口在同一连续状态链中。
        _write_checkpoint(
            bundle=bundle,
            mode="balanced",
            assignments=assignments,
            windows=window_records,
            solver=solver_records,
            next_tau=next_tau,
            next_window_id=window_id + 1,
            status="RUNNING",
            decision_window=decision_window,
            lookahead=lookahead,
        )
        logging.info(
            "均衡窗口%d完成[数学启发式]：tau=%d，活动任务=%d，候选=%d，"
            "固定=%d，剩余=%d，z=%.6g，耗时=%.1fs，LNS=%s。",
            window_id,
            tau,
            construct_stats["ActiveTaskCount"],
            construct_stats["CandidateOptionCount"],
            len(committed),
            remaining_after,
            z_final,
            elapsed,
            lns_info.get("LNSStatus"),
        )
        tau = next_tau
        window_id += 1
        gc.collect()

    if any(pools.values()):
        if max_windows is not None:
            profile, metrics = _schedule_profile(bundle, assignments)
            return assignments, pd.DataFrame(window_records), pd.DataFrame(solver_records), metrics
        remaining_ids = [task_id for values in pools.values() for task_id in values]
        raise RuntimeError(f"均衡滚动结束后仍有{len(remaining_ids)}个任务未固定")

    _write_checkpoint(
        bundle=bundle,
        mode="balanced",
        assignments=assignments,
        windows=window_records,
        solver=solver_records,
        next_tau=tau,
        next_window_id=window_id,
        status="COMPLETED",
        decision_window=decision_window,
        lookahead=lookahead,
    )
    profile, metrics = _schedule_profile(bundle, assignments)
    return assignments, pd.DataFrame(window_records), pd.DataFrame(solver_records), metrics


# =============================================================================
# 11. 最终独立检验
# =============================================================================

def _validate_assignments(bundle: InputBundle, assignments: pd.DataFrame, label: str) -> None:
    expected = set(bundle.tasks["TaskID"].astype(str))
    actual = set(assignments["TaskID"].astype(str))
    if expected != actual:
        raise RuntimeError(
            f"{label}任务覆盖不完整：缺失{len(expected - actual)}，多余{len(actual - expected)}"
        )
    if assignments["TaskID"].duplicated().any():
        raise RuntimeError(f"{label}存在重复 TaskID")
    if (
        assignments["StartHour"].astype(float)
        < assignments["ArrivalHour"].astype(float) - FLOAT_EPS
    ).any():
        raise RuntimeError(f"{label}存在早于 ArrivalHour 启动的任务")
    if (assignments["FinishHour"].astype(float) > TERMINAL_HOUR + FLOAT_EPS).any():
        raise RuntimeError(f"{label}存在占用第2406小时或更晚的任务")

    real = assignments["TaskType"].eq("RealTimeInference")
    if (
        assignments.loc[real, "StartHour"].astype(float)
        - assignments.loc[real, "ArrivalHour"].astype(float)
    ).abs().max() > FLOAT_EPS:
        raise RuntimeError(f"{label}存在实时推理未到达即开工")

    task_lookup = bundle.tasks.set_index("TaskID")
    joined = assignments.set_index("TaskID").join(
        task_lookup[["EarliestStartHour", "LatestFinishHour", "Duration_h"]].rename(
            columns={"Duration_h": "TaskDuration_h"}
        ),
        how="left",
        validate="one_to_one",
    )
    if (
        joined["StartHour"].astype(float)
        < joined["EarliestStartHour"].astype(float) - FLOAT_EPS
    ).any():
        raise RuntimeError(f"{label}存在早于 EarliestStartHour 启动")
    if (
        joined["FinishHour"].astype(float)
        > joined["LatestFinishHour"].astype(float) + FLOAT_EPS
    ).any():
        raise RuntimeError(f"{label}存在超过 LatestFinishHour 完成")
    duration_error = (
        joined["FinishHour"].astype(float)
        - joined["StartHour"].astype(float)
        - joined["TaskDuration_h"].astype(float)
    ).abs()
    if len(duration_error) and float(duration_error.max()) > 1e-7:
        raise RuntimeError(f"{label}任务持续时间与输入不一致")

    candidate_pairs = set(
        zip(
            bundle.candidates["TaskID"].astype(str),
            bundle.candidates["TargetRegion"].astype(str),
        )
    )
    bad_pairs = [
        (str(row.TaskID), str(row.TargetRegion))
        for row in assignments.itertuples(index=False)
        if (str(row.TaskID), str(row.TargetRegion)) not in candidate_pairs
    ]
    if bad_pairs:
        raise RuntimeError(f"{label}存在非合法候选区域安排，示例：{bad_pairs[:5]}")
    if (
        assignments["NetworkLatency_ms"].astype(float)
        > assignments["MaxLatency_ms"].astype(float) + FLOAT_EPS
    ).any():
        raise RuntimeError(f"{label}存在网络时延超过 MaxLatency")


def _validate_profile(profile: pd.DataFrame, label: str) -> None:
    if profile.empty:
        raise RuntimeError(f"{label}资源剖面为空")
    checks = {
        "GPU_Slack": "GPU容量",
        "IT_Slack_MW": "IT功率",
        "Facility_Slack_MW": "设施功率",
        "EffectiveAI_Slack_MW": "有效AI功率容量",
    }
    for column, description in checks.items():
        violation = float(np.maximum(-profile[column].to_numpy(dtype=float), 0.0).max())
        if violation > 1e-6:
            raise RuntimeError(f"{label}存在{description}违反量：{violation}")
    grid_violation = float(profile["GridPurchaseViolation_MW"].max())
    if grid_violation > 1e-6:
        raise RuntimeError(f"{label}存在购电上限违反量：{grid_violation}")
    balance_residual = float(profile["EnergyBalanceResidual_MW"].abs().max())
    if balance_residual > 1e-6:
        raise RuntimeError(f"{label}能源平衡残差过大：{balance_residual}")
    if (profile["RenewableExport_MW"] < -1e-8).any() or (
        profile["RenewableExport_MW"] - profile["ExportLimit_MW"] > 1e-6
    ).any():
        raise RuntimeError(f"{label}新能源外送边界违反")
    if (profile["RenewableCurtailment_MW"] < -1e-8).any():
        raise RuntimeError(f"{label}存在负弃电量")


# =============================================================================
# 12. 输出
# =============================================================================

def _comparison_table(
    balanced: dict[str, float], baseline: dict[str, float]
) -> pd.DataFrame:
    metrics = [
        ("OperatingCost_CNY", "Cost"),
        ("CarbonEmission_tCO2", "Carbon"),
        ("MeanNetworkLatency_ms", "MeanLatency"),
        ("RenewableUnusedRate", "RenewableUnusedRate"),
        ("MigratedTaskRatio", "MigratedTaskRatio"),
        ("MigrationGPUWorkload_GPUh", "MigrationGPUWorkload_GPUh"),
        ("AverageWaitHours", "AverageWaitHours"),
    ]
    rows = []
    for label, key in metrics:
        base = float(baseline.get(key, np.nan))
        value = float(balanced.get(key, np.nan))
        change = value - base
        relative = change / abs(base) if math.isfinite(base) and abs(base) > FLOAT_EPS else np.nan
        rows.append(
            {
                "Metric": label,
                "PureComputeBaseline": base,
                "Q2Balanced": value,
                "AbsoluteChange": change,
                "RelativeChange": relative,
            }
        )
    return pd.DataFrame(rows)


def _write_final_outputs(
    *,
    bundle: InputBundle,
    audit: AggregationAudit,
    calibration: GlobalCalibration,
    calibration_table: pd.DataFrame,
    baseline_assignments: pd.DataFrame,
    baseline_windows: pd.DataFrame,
    baseline_solver: pd.DataFrame,
    baseline_metrics: dict[str, float],
    balanced_assignments: pd.DataFrame,
    balanced_windows: pd.DataFrame,
    balanced_solver: pd.DataFrame,
    balanced_metrics: dict[str, float],
    decision_window: int,
    lookahead: int,
    mip_rel_gap: float,
    initial_time_limit: float,
    max_time_limit: float,
) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    _validate_assignments(bundle, baseline_assignments, "纯算力基准")
    _validate_assignments(bundle, balanced_assignments, "Q2均衡方案")
    baseline_profile, baseline_metrics_recomputed = _schedule_profile(bundle, baseline_assignments)
    balanced_profile, balanced_metrics_recomputed = _schedule_profile(bundle, balanced_assignments)
    _validate_profile(baseline_profile, "纯算力基准")
    _validate_profile(balanced_profile, "Q2均衡方案")

    # 最终指标以独立复算为准。
    baseline_metrics = baseline_metrics_recomputed
    balanced_metrics = balanced_metrics_recomputed

    _write_table(_audit_frame(audit), "q2_aggregation_audit.csv")
    _write_table(calibration_table, "q2_global_ideal_points.csv")
    _write_table(
        pd.DataFrame(
            [
                {
                    "Objective": metric,
                    "IdealLowerBound": calibration.ideal_lb[metric],
                    "PureComputeBaseline": calibration.baseline_value[metric],
                    "PayoffReference": calibration.reference_value[metric],
                    "NormalizationScale": calibration.scale[metric],
                    "ActiveInMinMax": int(metric in calibration.active_metrics),
                }
                for metric in OBJECTIVE_NAMES
            ]
        ),
        "q2_global_calibration.csv",
    )

    _write_table(baseline_assignments, "q2_baseline_assignments.csv")
    _write_table(balanced_assignments, "q2_balanced_assignments.csv")
    _write_table(balanced_assignments, "q2_assignments.csv")
    _write_table(baseline_profile, "q2_baseline_resource_profile.csv")
    _write_table(balanced_profile, "q2_resource_profile.csv")

    energy_columns = [
        "Hour",
        "Region",
        "TimeRole",
        "AvailableRenewable_MW",
        "Total_Facility_Load_MW",
        "RenewableDirectUse_MW",
        "GridPurchase_MW",
        "RenewableExport_MW",
        "RenewableCurtailment_MW",
        "OperatingCost_CNY",
        "CarbonEmission_tCO2",
        "MaxGridImport_MW",
        "ExportLimit_MW",
        "GridPurchaseViolation_MW",
        "EnergyBalanceResidual_MW",
    ]
    _write_table(baseline_profile.loc[:, energy_columns], "q2_baseline_energy_profile.csv")
    _write_table(balanced_profile.loc[:, energy_columns], "q2_energy_profile.csv")

    summary_rows = []
    for name, metrics in (
        ("Q1PureComputeBaseline", baseline_metrics),
        ("Q2Balanced", balanced_metrics),
    ):
        summary_rows.append(
            {
                "Solution": name,
                "OperatingCost_CNY": metrics.get("Cost", np.nan),
                "CarbonEmission_tCO2": metrics.get("Carbon", np.nan),
                "MeanNetworkLatency_ms": metrics.get("MeanLatency", np.nan),
                "RenewableUnusedRate": metrics.get("RenewableUnusedRate", np.nan),
                "RenewableUtilizationRate": (1.0 - metrics.get("RenewableUnusedRate", np.nan)) if math.isfinite(float(metrics.get("RenewableUnusedRate", np.nan))) else np.nan,
                "AverageWaitHours": metrics.get("AverageWaitHours", np.nan),
                "MigratedTaskRatio": metrics.get("MigratedTaskRatio", np.nan),
                "MigrationGPUWorkload_GPUh": metrics.get("MigrationGPUWorkload_GPUh", np.nan),
                "TaskCount": metrics.get("TaskCount", np.nan),
                "MaxGPUUtilization": metrics.get("MaxGPUUtilization", np.nan),
                "MaxAIITCapacityUtilization": metrics.get("MaxAIITCapacityUtilization", np.nan),
                "MinGPUSlack": metrics.get("MinGPUSlack", np.nan),
                "MinITSlack_MW": metrics.get("MinITSlack_MW", np.nan),
                "MinFacilitySlack_MW": metrics.get("MinFacilitySlack_MW", np.nan),
                "MinEffectiveAISlack_MW": metrics.get("MinEffectiveAISlack_MW", np.nan),
                "MaxGridPurchaseViolation_MW": metrics.get("MaxGridPurchaseViolation_MW", np.nan),
                "MaxEnergyBalanceResidual_MW": metrics.get("MaxEnergyBalanceResidual_MW", np.nan),
            }
        )
    _write_table(pd.DataFrame(summary_rows), "q2_objective_summary.csv")
    _write_table(_comparison_table(balanced_metrics, baseline_metrics), "q2_baseline_comparison.csv")
    _write_table(baseline_windows, "q2_baseline_rolling_windows.csv")
    _write_table(balanced_windows, "q2_balanced_rolling_windows.csv")
    _write_table(
        pd.concat([baseline_solver, balanced_solver], ignore_index=True),
        "q2_solver_log.csv",
    )
    _write_table(
        pd.DataFrame(
            [
                {
                    "DecisionWindowHours": decision_window,
                    "LookaheadHours": lookahead,
                    "MIPRelativeGap": mip_rel_gap,
                    "InitialSubproblemTimeLimitSeconds": initial_time_limit,
                    "MaxSubproblemTimeLimitSeconds": max_time_limit,
                    "CandidatePolicy": "all_legal_starts_no_topk_no_six_point",
                    "AggregationPolicy": "exact_task_equivalence_integer_counts",
                    "CalibrationPolicy": "four_global_continuous_ideal_points_compact_column_generation_dual_certificate",
                    "RollingPolicy": "H24_K48_minmax_marginal_greedy_plus_capacity_repair_plus_LNS_MILP",
                    "BaselinePolicy": "Q1_lexicographic_migration_then_wait_once",
                    "ValidationPolicy": "task_level_and_full_horizon_independent_recomputation",
                    "MILPBackend": "small_repair_and_LNS_subproblems_only_scipy_HiGHS",
                    "MIPStartPolicy": "not_required_in_matheuristic_mainline",
                    "HeuristicLNSMaxClasses": HEURISTIC_LNS_MAX_CLASSES,
                    "HeuristicLNSTimeLimitSeconds": HEURISTIC_LNS_TIME_LIMIT,
                    "HeuristicRepairMaxClasses": HEURISTIC_REPAIR_MAX_CLASSES,
                    "HeuristicRepairTimeLimitSeconds": HEURISTIC_REPAIR_TIME_LIMIT,
                }
            ]
        ),
        "q2_model_config.csv",
    )


# =============================================================================
# 13. 主程序
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Q2最终版：精确聚合+全局连续定标+滚动数学启发式多目标调度"
    )
    parser.add_argument("--decision-window", type=int, default=DEFAULT_DECISION_WINDOW)
    parser.add_argument("--lookahead", type=int, default=DEFAULT_LOOKAHEAD)
    parser.add_argument("--mip-rel-gap", type=float, default=DEFAULT_MIP_REL_GAP)
    parser.add_argument("--solver-time-limit", type=float, default=DEFAULT_SOLVER_TIME_LIMIT)
    parser.add_argument("--max-solver-time-limit", type=float, default=DEFAULT_MAX_SOLVER_TIME_LIMIT)
    parser.add_argument(
        "--lp-time-limit",
        type=float,
        default=0.0,
        help="全局连续理想点单个LP限时；0表示不设限，且必须status=0才接受",
    )
    parser.add_argument("--progress-interval", type=float, default=DEFAULT_PROGRESS_INTERVAL)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否从新版检查点恢复；默认开启，使用--no-resume才从头运行",
    )
    parser.add_argument("--force-recalibrate", action="store_true", help="忽略全局定标缓存并重算4个LP")
    parser.add_argument("--audit-only", action="store_true", help="只输出精确聚合规模审计，不求解")
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="调试：只运行前若干窗口；不会写正式完整结果",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args(argv)

    if args.decision_window <= 0 or args.lookahead < 0:
        raise SystemExit("--decision-window必须>0，--lookahead必须>=0")
    if not math.isfinite(args.mip_rel_gap) or args.mip_rel_gap < 0:
        raise SystemExit("--mip-rel-gap必须为非负有限数")
    if not math.isfinite(args.solver_time_limit) or args.solver_time_limit <= 0:
        raise SystemExit("--solver-time-limit必须为正有限数")
    if not math.isfinite(args.max_solver_time_limit) or args.max_solver_time_limit < args.solver_time_limit:
        raise SystemExit("--max-solver-time-limit必须>=--solver-time-limit")
    if not math.isfinite(args.lp_time_limit) or args.lp_time_limit < 0:
        raise SystemExit("--lp-time-limit必须>=0")
    if not math.isfinite(args.progress_interval) or args.progress_interval < 0:
        raise SystemExit("--progress-interval必须>=0")
    if args.max_windows is not None and args.max_windows <= 0:
        raise SystemExit("--max-windows必须为正")

    _configure_logging(args.log_level)
    total_started = time.perf_counter()
    logging.info(
        "Q2最终版启动：H=%d，K=%d，MIP gap=%.4g，MILP限时=%.0f→%.0fs，LP限时=%s。",
        args.decision_window,
        args.lookahead,
        args.mip_rel_gap,
        args.solver_time_limit,
        args.max_solver_time_limit,
        "无限" if args.lp_time_limit <= 0 else f"{args.lp_time_limit:.0f}s",
    )
    logging.info(
        "检查点恢复策略：resume=%s；无参数启动默认续算；如需从头运行请显式使用--no-resume。",
        args.resume,
    )

    bundle = _load_inputs()
    classes, _ = _build_exact_task_classes(bundle)
    class_lookup = {task_class.class_id: task_class for task_class in classes}
    audit = _build_aggregation_audit(classes)
    logging.info(
        "精确聚合：50000级任务=%d → 同质类=%d，任务压缩=%.2fx；候选理论减少=%.2f%%。",
        audit.original_task_count,
        audit.exact_class_count,
        audit.task_compression_ratio,
        100.0 * audit.option_reduction_rate,
    )
    _write_table(_audit_frame(audit), "q2_aggregation_audit.csv")
    if args.audit_only:
        logging.info("--audit-only 完成。")
        return 0

    logging.info("阶段2：求解一次全时域纯算力基准（滚动、聚合、字典序）。")
    baseline_assignments, baseline_windows, baseline_solver, baseline_metrics = _run_baseline_rollout(
        bundle=bundle,
        classes=classes,
        class_lookup=class_lookup,
        decision_window=args.decision_window,
        lookahead=args.lookahead,
        mip_rel_gap=args.mip_rel_gap,
        initial_time_limit=args.solver_time_limit,
        max_time_limit=args.max_solver_time_limit,
        progress_interval=args.progress_interval,
        resume=args.resume,
        max_windows=args.max_windows,
    )
    if args.max_windows is not None:
        logging.info("调试模式：基准完成指定窗口，停止，不执行全局定标和正式输出。")
        return 0
    _validate_assignments(bundle, baseline_assignments, "纯算力基准")
    baseline_profile, baseline_metrics = _schedule_profile(bundle, baseline_assignments)
    _validate_profile(baseline_profile, "纯算力基准")

    calibration, calibration_table = _compute_or_load_calibration(
        bundle=bundle,
        classes=classes,
        class_lookup=class_lookup,
        baseline_assignments=baseline_assignments,
        baseline_metrics=baseline_metrics,
        lp_time_limit=(None if args.lp_time_limit <= 0 else args.lp_time_limit),
        progress_interval=args.progress_interval,
        force_recalibrate=args.force_recalibrate,
    )
    _write_table(calibration_table, "q2_global_ideal_points.csv")

    logging.info(
        "阶段4：滚动数学启发式；恢复并保留 balanced v%d 已完成窗口；"
        "后续采用多目标边际贪心 + 容量局部修复 + 关键任务LNS-MILP精修。",
        CHECKPOINT_SCHEMA_VERSION,
    )
    balanced_assignments, balanced_windows, balanced_solver, balanced_metrics = _run_balanced_rollout(
        bundle=bundle,
        classes=classes,
        class_lookup=class_lookup,
        baseline_assignments=baseline_assignments,
        baseline_profile=baseline_profile,
        calibration=calibration,
        decision_window=args.decision_window,
        lookahead=args.lookahead,
        mip_rel_gap=args.mip_rel_gap,
        initial_time_limit=args.solver_time_limit,
        max_time_limit=args.max_solver_time_limit,
        progress_interval=args.progress_interval,
        resume=args.resume,
        max_windows=None,
    )

    logging.info("阶段5：独立复算、约束核验与正式输出。")
    _write_final_outputs(
        bundle=bundle,
        audit=audit,
        calibration=calibration,
        calibration_table=calibration_table,
        baseline_assignments=baseline_assignments,
        baseline_windows=baseline_windows,
        baseline_solver=baseline_solver,
        baseline_metrics=baseline_metrics,
        balanced_assignments=balanced_assignments,
        balanced_windows=balanced_windows,
        balanced_solver=balanced_solver,
        balanced_metrics=balanced_metrics,
        decision_window=args.decision_window,
        lookahead=args.lookahead,
        mip_rel_gap=args.mip_rel_gap,
        initial_time_limit=args.solver_time_limit,
        max_time_limit=args.max_solver_time_limit,
    )

    logging.info(
        "Q2最终版完成：Cost=%.8g，Carbon=%.8g，Latency=%.8g，UnusedRate=%.8g，总耗时=%.1fs。",
        balanced_metrics.get("Cost", np.nan),
        balanced_metrics.get("Carbon", np.nan),
        balanced_metrics.get("MeanLatency", np.nan),
        balanced_metrics.get("RenewableUnusedRate", np.nan),
        time.perf_counter() - total_started,
    )
    logging.info("结果目录：%s；日志：%s", TABLES_DIR, MODEL_LOG_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

