"""问题2最终模型检验。

当前Q2主模型的求解结构是：

    全局连续定标 + 多目标边际贪心 + 容量修复 + LNS-MILP

本脚本只负责检验，不导入或调用model.py，也不启动新的MILP。检验分为四层：

1. 全时域独立复算与能源守恒：从最终TaskID级安排重新构造0--2405小时、
   6区域的资源和能源状态；2406小时只作为完成边界，不作为占用时段。
2. 同口径基线与机制一致性：用同一份输入和同一套无储能能源结算函数重算
   Q1PureComputeBaseline与Q2Balanced，检查工作量守恒和指标差异。
3. 精确模型对照与LNS消融：读取单独验证目录中的同窗exact/heuristic结果；
   同时读取正式窗口日志中的LNS前后独立复算z，统计LNS改善窗口。
4. 滚动前瞻稳定性：读取正式K=48以及单独保存的K=24、K=72结果。只检查
   三种设置是否满足硬约束，以及成本下降、新能源利用率提高的方向是否反转。

缺少精确对照或K=24/K=72文件时，脚本会输出PENDING状态并明确缺口，
不会把“文件不存在”解释成模型通过。验证结果只写入q2_validation_*.csv，
不覆盖q2_assignments.csv、q2_resource_profile.csv等正式结果。
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


QUESTION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = QUESTION_DIR.parents[1]
SHARED_DIR = PROJECT_DIR / "question" / "question_01" / "data" / "processed" / "shared"
Q2_INPUT_DIR = QUESTION_DIR / "data" / "processed" / "q2"
OUTPUTS_DIR = QUESTION_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
LOGS_DIR = OUTPUTS_DIR / "logs"
VALIDATION_RUNS_DIR = OUTPUTS_DIR / "validation_runs"
VALIDATION_LOG_PATH = LOGS_DIR / "question_02_validation.log"

MAIN_START_HOUR = 0
MAIN_END_HOUR = 2399
TAIL_END_HOUR = 2405
TERMINAL_HOUR = 2406
FLOAT_EPS = 1e-8
DEFAULT_TOLERANCE = 1e-6
DEFAULT_MAX_SECONDS = 600.0
DEFAULT_PROGRESS_EVERY = 1000
EXACT_WINDOW_COUNT = 10
OBJECTIVE_NAMES = ("Cost", "Carbon", "MeanLatency", "RenewableUnusedRate")
OBJECTIVE_COLUMNS = {
    "Cost": "OperatingCost_CNY",
    "Carbon": "CarbonEmission_tCO2",
    "MeanLatency": "MeanNetworkLatency_ms",
    "RenewableUnusedRate": "RenewableUnusedRate",
}

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
    "MaxLatency_ms",
    "EarliestStartHour",
    "LatestFinishHour",
    "Duration_h",
    "Task_Full_IT_Power_MW",
)
CANDIDATE_COLUMNS = ("TaskID", "TargetRegion", "NetworkLatency_ms", "MaxLatency_ms")
BOUNDARY_COLUMNS = ("Region", "SellLimit_MW", "MaxGridImport_MW", "MaxGridExport_MW")
ASSIGNMENT_COLUMNS = (
    "TaskID",
    "TaskType",
    "ArrivalHour",
    "SourceRegion",
    "TargetRegion",
    "NetworkLatency_ms",
    "MaxLatency_ms",
    "StartHour",
    "FinishHour",
    "Duration_h",
    "GPU_Demand",
    "Task_Full_IT_Power_MW",
    "WaitHours",
    "Migration_GPU_Workload_GPUh",
    "IsMigrated",
)
PROFILE_COMPARE_COLUMNS = (
    "Scheduled_GPU_Equivalent",
    "Scheduled_AI_IT_Load_MW",
    "GridPurchase_MW",
    "RenewableExport_MW",
    "RenewableCurtailment_MW",
    "OperatingCost_CNY",
    "CarbonEmission_tCO2",
    "EnergyBalanceResidual_MW",
)


class ValidationTimeout(RuntimeError):
    """验证超过独立预算后的显式停止。"""


@dataclass
class Deadline:
    max_seconds: float
    started_at: float

    def check(self, label: str) -> None:
        elapsed = time.perf_counter() - self.started_at
        if elapsed > self.max_seconds:
            raise ValidationTimeout(
                f"{label}阶段超过验证预算{self.max_seconds:.1f}s，"
                f"已耗时{elapsed:.1f}s；结果未被伪造"
            )


@dataclass
class Inputs:
    region_hour: pd.DataFrame
    tasks: pd.DataFrame
    candidates: pd.DataFrame
    boundaries: pd.DataFrame
    raw_terminal_rows: int


@dataclass
class AuditResult:
    label: str
    assignments: pd.DataFrame
    assignment_path: Path
    logical_detail: pd.DataFrame
    logical_count: int
    joined: pd.DataFrame
    profile: pd.DataFrame
    continuous_detail: pd.DataFrame
    metrics: dict[str, float]
    workload: dict[str, float]
    vmax: float
    emax: float
    terminal_overlap_count: int
    profile_compare: pd.DataFrame


def _configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"不支持的日志级别：{level_name}")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    logging.basicConfig(
        level=level,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(VALIDATION_LOG_PATH, encoding="utf-8"),
        ],
        force=True,
    )
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)
    logging.info(
        "问题2新检验日志初始化：文件=%s，级别=%s。",
        VALIDATION_LOG_PATH,
        level_name.upper(),
    )


def _read_csv(path: Path, required: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"缺少文件：{path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}缺少字段：{missing}")
    return frame


def _to_numeric(frame: pd.DataFrame, columns: Iterable[str], source: str) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any():
            raise ValueError(f"{source}的{column}存在无法转换为数值的记录")
    return result


def _load_inputs() -> Inputs:
    logging.info("阶段1/4：开始读取Q2新模型检验输入。")
    region_hour_raw = _read_csv(Q2_INPUT_DIR / "q2_region_hour_input.csv", REGION_HOUR_COLUMNS)
    tasks = _read_csv(SHARED_DIR / "tasks_clean.csv", TASK_COLUMNS)
    candidates = _read_csv(SHARED_DIR / "task_candidate_regions.csv", CANDIDATE_COLUMNS)
    boundaries = _read_csv(SHARED_DIR / "storage_params.csv", BOUNDARY_COLUMNS)
    logging.info(
        "阶段1/4：原始输入读取完成：逐时=%d、任务=%d、候选=%d、区域边界=%d。",
        len(region_hour_raw),
        len(tasks),
        len(candidates),
        len(boundaries),
    )

    region_hour = _to_numeric(
        region_hour_raw,
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

    region_hour["Hour"] = region_hour["Hour"].astype(int)
    region_hour["Region"] = region_hour["Region"].astype(str)
    region_hour["TimeRole"] = region_hour["TimeRole"].astype(str).str.strip().str.lower()
    tasks["TaskID"] = tasks["TaskID"].astype(str)
    tasks["TaskType"] = tasks["TaskType"].astype(str)
    tasks["SourceRegion"] = tasks["SourceRegion"].astype(str)
    candidates["TaskID"] = candidates["TaskID"].astype(str)
    candidates["TargetRegion"] = candidates["TargetRegion"].astype(str)
    boundaries["Region"] = boundaries["Region"].astype(str)

    raw_terminal_rows = int((region_hour["Hour"] == TERMINAL_HOUR).sum())
    region_hour = region_hour.loc[
        region_hour["Hour"].between(MAIN_START_HOUR, TAIL_END_HOUR)
    ].copy()
    if raw_terminal_rows:
        logging.info(
            "时域口径：发现%d条Hour=2406输入记录；验证资源剖面只使用0--2405，"
            "Hour=2406不作为占用时段。",
            raw_terminal_rows,
        )
    if tasks["TaskID"].duplicated().any():
        raise ValueError("tasks_clean.csv的TaskID不唯一")
    if candidates.duplicated(["TaskID", "TargetRegion"]).any():
        raise ValueError("task_candidate_regions.csv存在重复TaskID×TargetRegion")
    if region_hour.duplicated(["Hour", "Region"]).any():
        raise ValueError("逐时输入存在重复Hour×Region")
    if boundaries["Region"].duplicated().any():
        raise ValueError("storage_params.csv的Region不唯一")
    if not tasks["ArrivalHour"].between(MAIN_START_HOUR, MAIN_END_HOUR).all():
        raise ValueError("任务ArrivalHour必须位于0--2399")
    if (tasks["LatestFinishHour"] > TERMINAL_HOUR + FLOAT_EPS).any():
        raise ValueError("存在LatestFinishHour超过2406的任务")
    if (
        tasks["LatestFinishHour"]
        < tasks["EarliestStartHour"] + tasks["Duration_h"] - FLOAT_EPS
    ).any():
        raise ValueError("存在任务时间窗无法容纳持续时间的记录")
    if not candidates["TaskID"].isin(set(tasks["TaskID"])).all():
        raise ValueError("候选区域表包含不存在的TaskID")
    if (
        candidates["NetworkLatency_ms"]
        > candidates["MaxLatency_ms"] + FLOAT_EPS
    ).any():
        raise ValueError("候选区域表存在超过MaxLatency的网络时延")

    region_hour = region_hour.merge(
        boundaries.loc[:, list(BOUNDARY_COLUMNS)],
        how="left",
        on="Region",
        validate="many_to_one",
    )
    if region_hour[list(BOUNDARY_COLUMNS[1:])].isna().any().any():
        raise ValueError("storage_params.csv无法覆盖全部逐时区域")
    region_hour["ExportLimit_MW"] = np.minimum(
        region_hour["SellLimit_MW"],
        region_hour["MaxGridExport_MW"],
    )
    effective_it = region_hour["Max_IT_Power_MW"] - region_hour["NonAI_IT_Load_MW"]
    effective_facility = (
        region_hour["Max_Facility_Power_MW"] / region_hour["PUE"]
        - region_hour["NonAI_IT_Load_MW"]
    )
    effective_grid = (
        (
            region_hour["AvailableRenewable_MW"]
            + region_hour["MaxGridImport_MW"]
        )
        / region_hour["PUE"]
        - region_hour["NonAI_IT_Load_MW"]
    )
    region_hour["Effective_AI_IT_Capacity_MW"] = np.minimum.reduce(
        [
            effective_it.to_numpy(dtype=float),
            effective_facility.to_numpy(dtype=float),
            effective_grid.to_numpy(dtype=float),
        ]
    )
    if (region_hour["Effective_AI_IT_Capacity_MW"] < -FLOAT_EPS).any():
        raise ValueError("固定NonAI负荷导致有效AI IT容量为负")

    regions = tuple(sorted(region_hour["Region"].unique().tolist()))
    expected = pd.MultiIndex.from_product(
        [range(MAIN_START_HOUR, TAIL_END_HOUR + 1), regions],
        names=["Hour", "Region"],
    )
    actual = pd.MultiIndex.from_frame(region_hour.loc[:, ["Hour", "Region"]])
    missing = expected.difference(actual)
    if len(missing):
        raise ValueError(f"逐时输入缺少Hour×Region记录，例如：{list(missing[:5])}")
    latest_2406 = int(
        (tasks["LatestFinishHour"].sub(TERMINAL_HOUR).abs() <= FLOAT_EPS).sum()
    )
    logging.info(
        "阶段1/4完成：任务=%d，候选=%d，区域=%d，资源键=%d，"
        "允许LatestFinishHour=2406的任务=%d。",
        len(tasks),
        len(candidates),
        len(regions),
        len(region_hour),
        latest_2406,
    )
    return Inputs(
        region_hour=region_hour.sort_values(["Hour", "Region"], kind="stable").reset_index(
            drop=True
        ),
        tasks=tasks.sort_values(["ArrivalHour", "TaskID"], kind="stable").reset_index(
            drop=True
        ),
        candidates=candidates.sort_values(
            ["TaskID", "TargetRegion"], kind="stable"
        ).reset_index(drop=True),
        boundaries=boundaries.sort_values("Region", kind="stable").reset_index(drop=True),
        raw_terminal_rows=raw_terminal_rows,
    )


def _locate_file(root: Path, filename: str) -> Path | None:
    candidates = (root / filename, root / "tables" / filename)
    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_assignments(root: Path, filenames: tuple[str, ...]) -> tuple[pd.DataFrame, Path]:
    path = None
    for filename in filenames:
        path = _locate_file(root, filename)
        if path is not None:
            break
    if path is None:
        raise FileNotFoundError(f"{root}中缺少任务安排：{filenames}")
    frame = _read_csv(path, ASSIGNMENT_COLUMNS)
    if frame["TaskID"].isna().any():
        raise ValueError(f"{path.name}存在空TaskID")
    for column in ("TaskID", "TaskType", "SourceRegion", "TargetRegion"):
        frame[column] = frame[column].astype(str)
    frame = _to_numeric(
        frame,
        [
            "ArrivalHour",
            "NetworkLatency_ms",
            "MaxLatency_ms",
            "StartHour",
            "FinishHour",
            "Duration_h",
            "GPU_Demand",
            "Task_Full_IT_Power_MW",
            "WaitHours",
            "Migration_GPU_Workload_GPUh",
            "IsMigrated",
        ],
        path.name,
    )
    logging.info("已读取%s：记录=%d。", path, len(frame))
    return frame, path


def _join_tasks(inputs: Inputs, assignments: pd.DataFrame) -> pd.DataFrame:
    return assignments.merge(
        inputs.tasks,
        how="left",
        on="TaskID",
        suffixes=("_assign", "_task"),
        indicator="_task_merge",
    )


def _add_check(
    records: list[dict[str, object]],
    name: str,
    count: int,
    description: str,
    label: str,
) -> None:
    count = int(count)
    records.append(
        {
            "Model": label,
            "Check": name,
            "ViolationCount": count,
            "Status": "PASS" if count == 0 else "FAIL",
            "Description": description,
        }
    )
    logging.info(
        "检验1/4-%s逻辑审计：%s完成，违规数=%d。",
        label,
        name,
        count,
    )


def _audit_logical(
    inputs: Inputs,
    assignments: pd.DataFrame,
    label: str,
    deadline: Deadline,
    tolerance: float,
) -> tuple[pd.DataFrame, int, pd.DataFrame]:
    logging.info(
        "检验1/4-%s：开始任务级硬约束审计，安排=%d、期望任务=%d。",
        label,
        len(assignments),
        len(inputs.tasks),
    )
    records: list[dict[str, object]] = []
    task_ids = set(inputs.tasks["TaskID"])
    counts = assignments.groupby("TaskID", sort=False).size()
    duplicate_count = int((counts - 1).clip(lower=0).sum())
    unknown_count = int((~assignments["TaskID"].isin(task_ids)).sum())
    missing_count = int((~inputs.tasks["TaskID"].isin(set(counts.index))).sum())
    _add_check(records, "任务重复执行", duplicate_count, "每个TaskID最多出现一次", label)
    _add_check(records, "任务未知ID", unknown_count, "安排中的TaskID必须来自任务表", label)
    _add_check(records, "任务漏排", missing_count, "每个任务必须恰好出现一次", label)

    joined = _join_tasks(inputs, assignments)
    known = joined.loc[joined["_task_merge"].eq("both")].copy()
    if known.empty:
        for name, description in (
            ("候选区域", "TargetRegion必须存在于候选区域表"),
            ("任务时间与资源字段", "安排字段必须与任务表一致"),
            ("时间窗", "Start/Finish必须满足到达、最早和最晚边界"),
            ("实时任务立即启动", "RealTimeInference的StartHour必须等于ArrivalHour"),
            ("网络时延", "网络时延必须满足候选和任务MaxLatency"),
            ("2406终止边界", "FinishHour=2406允许，但不得超过2406"),
            ("迁移与等待派生字段", "IsMigrated、WaitHours和迁移工作量应与安排一致"),
        ):
            _add_check(records, name, 0, description, label)
        deadline.check(f"{label}逻辑审计")
        detail = pd.DataFrame(records)
        return detail, int(detail["ViolationCount"].sum()), joined

    candidate_frame = inputs.candidates.rename(
        columns={
            "NetworkLatency_ms": "CandidateNetworkLatency_ms",
            "MaxLatency_ms": "CandidateMaxLatency_ms",
        }
    )
    candidate_join = known.merge(
        candidate_frame.loc[
            :,
            [
                "TaskID",
                "TargetRegion",
                "CandidateNetworkLatency_ms",
                "CandidateMaxLatency_ms",
            ],
        ],
        how="left",
        on=["TaskID", "TargetRegion"],
        indicator="_candidate_merge",
    )
    candidate_missing = int((~candidate_join["_candidate_merge"].eq("both")).sum())
    candidate_latency_bad = int(
        (
            candidate_join["_candidate_merge"].eq("both")
            & (
                (
                    candidate_join["NetworkLatency_ms"]
                    - candidate_join["CandidateNetworkLatency_ms"]
                ).abs()
                > tolerance
            )
        ).sum()
    )
    _add_check(records, "候选区域", candidate_missing, "TargetRegion必须存在于候选区域表", label)
    _add_check(
        records,
        "候选时延一致性",
        candidate_latency_bad,
        "安排网络时延必须与候选区域表一致",
        label,
    )

    def _bad_equal(left: str, right: str, tol: float = tolerance) -> int:
        return int(
            (
                (
                    pd.to_numeric(known[left], errors="coerce")
                    - pd.to_numeric(known[right], errors="coerce")
                ).abs()
                > tol
            ).sum()
        )

    task_field_pairs = (
        ("TaskType_assign", "TaskType_task"),
        ("SourceRegion_assign", "SourceRegion_task"),
        ("ArrivalHour_assign", "ArrivalHour_task"),
    )
    for left, right in task_field_pairs:
        if left in known.columns and right in known.columns:
            _add_check(
                records,
                f"任务字段一致性-{left}",
                int((known[left].astype(str) != known[right].astype(str)).sum()),
                "安排中的任务属性必须与任务表一致",
                label,
            )
    for field in ("GPU_Demand", "Duration_h", "Task_Full_IT_Power_MW"):
        left = f"{field}_assign"
        right = f"{field}_task"
        if left in known.columns and right in known.columns:
            _add_check(
                records,
                f"任务数值一致性-{field}",
                _bad_equal(left, right),
                "安排数值字段必须与任务表一致",
                label,
            )

    start_arrival = int(
        (known["StartHour"] < known["ArrivalHour_task"] - tolerance).sum()
    )
    start_earliest = int(
        (
            known["StartHour"]
            < known["EarliestStartHour"] - tolerance
        ).sum()
    )
    finish_latest = int(
        (
            known["FinishHour"]
            > known["LatestFinishHour"] + tolerance
        ).sum()
    )
    duration_bad = int(
        (
            (
                (known["FinishHour"] - known["StartHour"])
                - known["Duration_h_task"]
            ).abs()
            > tolerance
        ).sum()
    )
    assignment_duration_bad = _bad_equal("Duration_h_assign", "Duration_h_task")
    _add_check(records, "到达时刻边界", start_arrival, "StartHour不得早于ArrivalHour", label)
    _add_check(
        records,
        "最早开工边界",
        start_earliest,
        "StartHour不得早于EarliestStartHour",
        label,
    )
    _add_check(
        records,
        "最晚完成边界",
        finish_latest,
        "FinishHour不得超过LatestFinishHour",
        label,
    )
    _add_check(records, "持续时间一致性", duration_bad, "Finish-Start必须等于Duration_h", label)
    _add_check(
        records,
        "安排持续时间一致性",
        assignment_duration_bad,
        "安排Duration_h必须与任务Duration_h一致",
        label,
    )

    realtime_bad = int(
        (
            known["TaskType_task"].eq("RealTimeInference")
            & (
                known["StartHour"] - known["ArrivalHour_task"]
            ).abs().gt(tolerance)
        ).sum()
    )
    latency_bad = int(
        (
            (
                known["NetworkLatency_ms"]
                > known["MaxLatency_ms_assign"] + tolerance
            )
            | (
                known["NetworkLatency_ms"]
                > known["MaxLatency_ms_task"] + tolerance
            )
        ).sum()
    )
    terminal_bad = int(
        (known["FinishHour"] > TERMINAL_HOUR + tolerance).sum()
    )
    _add_check(
        records,
        "实时任务立即启动",
        realtime_bad,
        "RealTimeInference必须在到达小时开工",
        label,
    )
    _add_check(
        records,
        "网络时延上限",
        latency_bad,
        "NetworkLatency_ms不得超过任务或安排MaxLatency_ms",
        label,
    )
    _add_check(
        records,
        "2406终止边界",
        terminal_bad,
        "FinishHour=2406允许，但不得超过2406；2406不作为占用小时",
        label,
    )

    derived_checks = []
    if "WaitHours" in assignments.columns:
        expected_wait = np.maximum(
            known["StartHour"].to_numpy(dtype=float)
            - known["ArrivalHour_task"].to_numpy(dtype=float),
            0.0,
        )
        derived_checks.append(
            (
                "等待时间派生一致性",
                int(
                    (
                        np.abs(
                            known["WaitHours"].to_numpy(dtype=float) - expected_wait
                        )
                        > tolerance
                    ).sum()
                ),
                "WaitHours应等于max(StartHour-ArrivalHour,0)",
            )
        )
    if "IsMigrated" in assignments.columns:
        expected_migrated = (
            known["TargetRegion"].astype(str)
            != known["SourceRegion_task"].astype(str)
        ).astype(float)
        derived_checks.append(
            (
                "迁移标记派生一致性",
                int(
                    (
                        np.abs(
                            known["IsMigrated"].to_numpy(dtype=float)
                            - expected_migrated.to_numpy(dtype=float)
                        )
                        > tolerance
                    ).sum()
                ),
                "IsMigrated应等于TargetRegion!=SourceRegion",
            )
        )
    if "Migration_GPU_Workload_GPUh" in assignments.columns:
        expected_migration_work = (
            known["GPU_Demand_task"].to_numpy(dtype=float)
            * known["Duration_h_task"].to_numpy(dtype=float)
            * expected_migrated.to_numpy(dtype=float)
            if "IsMigrated" in assignments.columns
            else np.zeros(len(known), dtype=float)
        )
        derived_checks.append(
            (
                "迁移工作量派生一致性",
                int(
                    (
                        np.abs(
                            known["Migration_GPU_Workload_GPUh"].to_numpy(dtype=float)
                            - expected_migration_work
                        )
                        > tolerance
                    ).sum()
                ),
                "迁移GPU工作量应等于迁移任务GPU_Demand×Duration_h",
            )
        )
    for name, count, description in derived_checks:
        _add_check(records, name, count, description, label)

    deadline.check(f"{label}逻辑审计")
    detail = pd.DataFrame(records)
    logical_count = int(detail["ViolationCount"].sum())
    logging.info(
        "检验1/4-%s逻辑审计完成：检查项=%d，累计违规=%d。",
        label,
        len(detail),
        logical_count,
    )
    return detail, logical_count, joined


def _overlap_by_hour(start: float, finish: float) -> Iterable[tuple[int, float]]:
    if not math.isfinite(start) or not math.isfinite(finish) or finish <= start:
        return
    first = math.floor(start + FLOAT_EPS)
    last = math.ceil(finish - FLOAT_EPS) - 1
    for hour in range(first, last + 1):
        overlap = max(0.0, min(finish, hour + 1.0) - max(start, float(hour)))
        if overlap > FLOAT_EPS:
            yield hour, overlap


def _recompute_profile(
    inputs: Inputs,
    joined: pd.DataFrame,
    label: str,
    deadline: Deadline,
    progress_every: int,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, float], float, float, int]:
    profile = inputs.region_hour.copy()
    profile = profile.sort_values(["Hour", "Region"], kind="stable").reset_index(drop=True)
    key_to_index = {
        (int(row.Hour), str(row.Region)): index
        for index, row in enumerate(profile.itertuples(index=False))
    }
    scheduled_gpu = np.zeros(len(profile), dtype=float)
    scheduled_ai = np.zeros(len(profile), dtype=float)
    known = joined.loc[joined["_task_merge"].eq("both")].copy()
    total_known = len(known)
    progress_every = max(1, int(progress_every))
    terminal_overlap_count = 0
    invalid_row_count = 0
    logging.info(
        "检验1/4-%s资源重算开始：任务=%d，逐时键=%d，每%d条任务反馈一次。",
        label,
        total_known,
        len(profile),
        progress_every,
    )
    for row_number, row in enumerate(known.itertuples(index=False), start=1):
        if row_number % progress_every == 0 or row_number == total_known:
            deadline.check(f"{label}逐时资源重算")
            logging.info(
                "检验1/4-%s资源重算进度：%d/%d（%.1f%%），已耗时%.1fs。",
                label,
                row_number,
                total_known,
                100.0 * row_number / total_known if total_known else 100.0,
                time.perf_counter() - deadline.started_at,
            )
        start = float(row.StartHour)
        finish = float(row.FinishHour)
        if not math.isfinite(start) or not math.isfinite(finish) or finish <= start:
            invalid_row_count += 1
            continue
        if finish > TERMINAL_HOUR + tolerance:
            terminal_overlap_count += 1
        target_region = str(row.TargetRegion)
        gpu = float(row.GPU_Demand_task)
        power = float(row.Task_Full_IT_Power_MW_task)
        for hour, overlap in _overlap_by_hour(start, finish):
            if hour >= TERMINAL_HOUR:
                continue
            index = key_to_index.get((hour, target_region))
            if index is None:
                continue
            scheduled_gpu[index] += gpu * overlap
            scheduled_ai[index] += power * overlap

    profile["Scheduled_GPU_Equivalent"] = scheduled_gpu
    profile["Scheduled_AI_IT_Load_MW"] = scheduled_ai
    profile["Total_IT_Load_MW"] = profile["NonAI_IT_Load_MW"] + profile[
        "Scheduled_AI_IT_Load_MW"
    ]
    profile["Total_Facility_Load_MW"] = (
        profile["Total_IT_Load_MW"] * profile["PUE"]
    )
    profile["GPU_Slack"] = profile["Available_GPU"] - profile["Scheduled_GPU_Equivalent"]
    profile["IT_Slack_MW"] = profile["Max_IT_Power_MW"] - profile["Total_IT_Load_MW"]
    profile["Facility_Slack_MW"] = (
        profile["Max_Facility_Power_MW"] - profile["Total_Facility_Load_MW"]
    )
    profile["EffectiveAI_Slack_MW"] = (
        profile["Effective_AI_IT_Capacity_MW"]
        - profile["Scheduled_AI_IT_Load_MW"]
    )
    profile["GPU_Utilization"] = np.divide(
        profile["Scheduled_GPU_Equivalent"],
        profile["Available_GPU"],
        out=np.zeros(len(profile), dtype=float),
        where=profile["Available_GPU"].to_numpy(dtype=float) > FLOAT_EPS,
    )
    profile["AI_IT_Capacity_Utilization"] = np.divide(
        profile["Scheduled_AI_IT_Load_MW"],
        profile["Effective_AI_IT_Capacity_MW"],
        out=np.zeros(len(profile), dtype=float),
        where=profile["Effective_AI_IT_Capacity_MW"].to_numpy(dtype=float) > FLOAT_EPS,
    )
    profile["RenewableDirectUse_MW"] = np.minimum(
        profile["Total_Facility_Load_MW"],
        profile["AvailableRenewable_MW"],
    )
    profile["GridPurchase_MW"] = np.maximum(
        profile["Total_Facility_Load_MW"] - profile["AvailableRenewable_MW"],
        0.0,
    )
    profile["RenewableExport_MW"] = np.minimum(
        np.maximum(
            profile["AvailableRenewable_MW"] - profile["Total_Facility_Load_MW"],
            0.0,
        ),
        profile["ExportLimit_MW"],
    )
    profile["RenewableCurtailment_MW"] = np.maximum(
        profile["AvailableRenewable_MW"]
        - profile["RenewableDirectUse_MW"]
        - profile["RenewableExport_MW"],
        0.0,
    )
    profile["GridPurchaseViolation_MW"] = np.maximum(
        profile["GridPurchase_MW"] - profile["MaxGridImport_MW"],
        0.0,
    )
    profile["ExportViolation_MW"] = np.maximum(
        profile["RenewableExport_MW"] - profile["ExportLimit_MW"],
        0.0,
    )
    profile["GPU_Violation_MW"] = np.maximum(
        -profile["GPU_Slack"],
        0.0,
    )
    profile["IT_Violation_MW"] = np.maximum(-profile["IT_Slack_MW"], 0.0)
    profile["Facility_Violation_MW"] = np.maximum(
        -profile["Facility_Slack_MW"],
        0.0,
    )
    profile["EffectiveAI_Violation_MW"] = np.maximum(
        -profile["EffectiveAI_Slack_MW"],
        0.0,
    )
    profile["EnergyBalanceResidual_MW"] = (
        profile["AvailableRenewable_MW"]
        + profile["GridPurchase_MW"]
        - profile["Total_Facility_Load_MW"]
        - profile["RenewableExport_MW"]
        - profile["RenewableCurtailment_MW"]
    )
    profile["OperatingCost_CNY"] = (
        profile["ElectricityPrice_CNY_per_MWh"] * profile["GridPurchase_MW"]
        - profile["SellPrice_CNY_per_MWh"] * profile["RenewableExport_MW"]
    )
    profile["CarbonEmission_tCO2"] = (
        profile["CarbonIntensity_tCO2_per_MWh"] * profile["GridPurchase_MW"]
    )

    checks: list[dict[str, object]] = []
    for column, description in (
        ("GPU_Violation_MW", "GPU容量"),
        ("IT_Violation_MW", "IT功率"),
        ("Facility_Violation_MW", "设施功率"),
        ("EffectiveAI_Violation_MW", "有效AI IT容量"),
        ("GridPurchaseViolation_MW", "最大购电"),
        ("ExportViolation_MW", "新能源外送"),
    ):
        values = profile[column].to_numpy(dtype=float)
        maximum = float(values.max()) if len(values) else 0.0
        checks.append(
            {
                "Model": label,
                "Check": description,
                "MaximumViolation": maximum,
                "ViolationCount": int((values > tolerance).sum()),
                "Status": "PASS" if maximum <= tolerance else "FAIL",
            }
        )
    residual = profile["EnergyBalanceResidual_MW"].abs().to_numpy(dtype=float)
    residual_max = float(residual.max()) if len(residual) else 0.0
    checks.append(
        {
            "Model": label,
            "Check": "能源平衡残差",
            "MaximumViolation": residual_max,
            "ViolationCount": int((residual > tolerance).sum()),
            "Status": "PASS" if residual_max <= tolerance else "FAIL",
        }
    )
    checks.append(
        {
            "Model": label,
            "Check": "第2406小时正占用",
            "MaximumViolation": float(terminal_overlap_count),
            "ViolationCount": terminal_overlap_count,
            "Status": "PASS" if terminal_overlap_count == 0 else "FAIL",
        }
    )
    if invalid_row_count:
        checks.append(
            {
                "Model": label,
                "Check": "非法任务区间未参与资源重算",
                "MaximumViolation": float(invalid_row_count),
                "ViolationCount": invalid_row_count,
                "Status": "FAIL",
            }
        )
    detail = pd.DataFrame(checks)
    vmax = float(
        detail.loc[
            detail["Check"].ne("能源平衡残差"),
            "MaximumViolation",
        ].max()
    )
    emax = residual_max
    known_latency = (
        float(known["NetworkLatency_ms"].mean()) if len(known) else float("nan")
    )
    total_renewable = float(profile["AvailableRenewable_MW"].sum())
    total_direct_export = float(
        (
            profile["RenewableDirectUse_MW"]
            + profile["RenewableExport_MW"]
        ).sum()
    )
    metrics = {
        "Cost": float(profile["OperatingCost_CNY"].sum()),
        "Carbon": float(profile["CarbonEmission_tCO2"].sum()),
        "MeanLatency": known_latency,
        "RenewableUnusedRate": (
            float(profile["RenewableCurtailment_MW"].sum()) / total_renewable
            if total_renewable > FLOAT_EPS
            else float("nan")
        ),
        "RenewableUtilizationRate": (
            total_direct_export / total_renewable
            if total_renewable > FLOAT_EPS
            else float("nan")
        ),
        "AverageWaitHours": (
            float(known["WaitHours"].mean()) if "WaitHours" in known.columns else float("nan")
        ),
        "MigratedTaskRatio": (
            float(known["IsMigrated"].mean()) if "IsMigrated" in known.columns else float("nan")
        ),
    }
    task_gpu_work = float(
        (inputs.tasks["GPU_Demand"] * inputs.tasks["Duration_h"]).sum()
    )
    task_ai_energy = float(
        (
            inputs.tasks["Task_Full_IT_Power_MW"]
            * inputs.tasks["Duration_h"]
        ).sum()
    )
    assigned_gpu_work = float(
        (known["GPU_Demand_task"] * known["Duration_h_task"]).sum()
    ) if len(known) else 0.0
    assigned_ai_energy = float(
        (
            known["Task_Full_IT_Power_MW_task"]
            * known["Duration_h_task"]
        ).sum()
    ) if len(known) else 0.0
    workload = {
        "TaskGPUWorkload_GPUh": task_gpu_work,
        "AssignedGPUWorkload_GPUh": assigned_gpu_work,
        "TaskAIITEnergy_MWh": task_ai_energy,
        "AssignedAIITEnergy_MWh": assigned_ai_energy,
        "TaskCount": float(len(inputs.tasks)),
        "AssignedKnownTaskCount": float(len(known)),
    }
    deadline.check(f"{label}能源与资源独立复算")
    logging.info(
        "检验1/4-%s资源重算完成：V_max=%.6g，E_max=%.6g，2406正占用=%d，"
        "Cost=%.8g，UnusedRate=%.8g。",
        label,
        vmax,
        emax,
        terminal_overlap_count,
        metrics["Cost"],
        metrics["RenewableUnusedRate"],
    )
    return profile, detail, metrics, workload, vmax, emax, terminal_overlap_count


def _compare_exported_profile(
    recomputed: pd.DataFrame,
    exported_path: Path | None,
    label: str,
    tolerance: float,
) -> pd.DataFrame:
    if exported_path is None:
        return pd.DataFrame(
            [
                {
                    "Model": label,
                    "Check": "正式资源剖面文件",
                    "Status": "PENDING_NO_EXPORTED_PROFILE",
                    "Reason": "未找到q2_resource_profile.csv或对应基线剖面",
                }
            ]
        )
    exported = _read_csv(exported_path, ("Hour", "Region"))
    left = recomputed.loc[:, ["Hour", "Region", *PROFILE_COMPARE_COLUMNS]].copy()
    right_columns = [
        column for column in PROFILE_COMPARE_COLUMNS if column in exported.columns
    ]
    if not right_columns:
        return pd.DataFrame(
            [
                {
                    "Model": label,
                    "Check": "正式资源剖面字段",
                    "Status": "FAIL_EXPORTED_PROFILE_INCOMPLETE",
                    "Reason": "正式剖面缺少可比较的独立复算字段",
                }
            ]
        )
    right = exported.loc[:, ["Hour", "Region", *right_columns]].copy()
    merged = left.merge(
        right,
        how="outer",
        on=["Hour", "Region"],
        suffixes=("_recomputed", "_exported"),
        indicator=True,
    )
    rows: list[dict[str, object]] = []
    key_mismatch = int((merged["_merge"] != "both").sum())
    rows.append(
        {
            "Model": label,
            "Check": "正式剖面Hour×Region键",
            "MaximumDifference": float(key_mismatch),
            "DifferenceCount": key_mismatch,
            "Status": "PASS" if key_mismatch == 0 else "FAIL",
        }
    )
    for column in right_columns:
        left_column = f"{column}_recomputed"
        right_column = f"{column}_exported"
        if left_column not in merged.columns or right_column not in merged.columns:
            continue
        left_values = pd.to_numeric(merged[left_column], errors="coerce").fillna(0.0)
        right_values = pd.to_numeric(merged[right_column], errors="coerce").fillna(0.0)
        difference = (left_values - right_values).abs()
        maximum = float(difference.max()) if len(difference) else 0.0
        rows.append(
            {
                "Model": label,
                "Check": f"正式剖面字段-{column}",
                "MaximumDifference": maximum,
                "DifferenceCount": int((difference > tolerance).sum()),
                "Status": "PASS" if maximum <= tolerance else "FAIL",
            }
        )
    logging.info(
        "独立复算与正式%s资源剖面对比完成：比较字段=%d，键差异=%d。",
        label,
        len(right_columns),
        key_mismatch,
    )
    return pd.DataFrame(rows)


def _write_table(frame: pd.DataFrame, filename: str) -> Path:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / filename
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.15g")
    logging.info("验证结果写出：%s，行数=%d。", path, len(frame))
    return path


def _official_summary(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


def _metric_consistency_rows(
    independent: AuditResult,
    official_summary: pd.DataFrame | None,
    solution_name: str,
    tolerance: float,
) -> list[dict[str, object]]:
    if official_summary is None or "Solution" not in official_summary.columns:
        return [
            {
                "Model": independent.label,
                "Check": "正式目标汇总",
                "Status": "PENDING_NO_OBJECTIVE_SUMMARY",
                "Reason": "未找到q2_objective_summary.csv",
            }
        ]
    matched = official_summary.loc[
        official_summary["Solution"].astype(str).eq(solution_name)
    ]
    if matched.empty:
        return [
            {
                "Model": independent.label,
                "Check": "正式目标汇总",
                "Status": "FAIL_MISSING_SOLUTION_ROW",
                "Reason": f"缺少Solution={solution_name}",
            }
        ]
    row = matched.iloc[0]
    rows: list[dict[str, object]] = []
    for metric, column in OBJECTIVE_COLUMNS.items():
        if column not in row.index:
            rows.append(
                {
                    "Model": independent.label,
                    "Check": f"正式目标-{metric}",
                    "Status": "FAIL_MISSING_METRIC_COLUMN",
                }
            )
            continue
        expected = float(row[column])
        actual = float(independent.metrics[metric])
        difference = abs(actual - expected)
        allowed = max(tolerance, 1e-9 * max(1.0, abs(expected)))
        rows.append(
            {
                "Model": independent.label,
                "Check": f"正式目标-{metric}",
                "IndependentValue": actual,
                "OfficialValue": expected,
                "AbsoluteDifference": difference,
                "AllowedDifference": allowed,
                "Status": "PASS" if difference <= allowed else "FAIL",
            }
        )
    return rows


def _baseline_consistency(
    inputs: Inputs,
    baseline: AuditResult,
    balanced: AuditResult,
    official_summary: pd.DataFrame | None,
    tolerance: float,
) -> pd.DataFrame:
    logging.info("检验2/4：开始同口径基线与物理机制一致性检验。")
    rows: list[dict[str, object]] = []
    for metric in (
        "Cost",
        "Carbon",
        "MeanLatency",
        "RenewableUnusedRate",
        "RenewableUtilizationRate",
    ):
        base = baseline.metrics[metric]
        q2 = balanced.metrics[metric]
        if metric == "Cost":
            direction = "Q2成本更低" if q2 < base - tolerance else "Q2成本未降低"
        elif metric == "RenewableUtilizationRate":
            direction = (
                "Q2新能源利用率更高"
                if q2 > base + tolerance
                else "Q2新能源利用率未提高"
            )
        elif metric == "RenewableUnusedRate":
            direction = (
                "Q2弃新能源率更低"
                if q2 < base - tolerance
                else "Q2弃新能源率未降低"
            )
        else:
            direction = "按真实权衡报告，不预设改善方向"
        rows.append(
            {
                "Check": f"同口径指标-{metric}",
                "PureComputeBaseline": base,
                "Q2Balanced": q2,
                "AbsoluteChange_Q2MinusBaseline": q2 - base,
                "RelativeChange": (
                    (q2 - base) / max(abs(base), FLOAT_EPS)
                    if math.isfinite(base)
                    else float("nan")
                ),
                "DirectionInterpretation": direction,
                "Status": "REPORTED",
            }
        )

    for workload_name, base_key, q2_key in (
        ("GPU累计工作量", "AssignedGPUWorkload_GPUh", "AssignedGPUWorkload_GPUh"),
        ("AI IT累计能量", "AssignedAIITEnergy_MWh", "AssignedAIITEnergy_MWh"),
    ):
        base = baseline.workload[base_key]
        q2 = balanced.workload[q2_key]
        difference = abs(base - q2)
        allowed = max(tolerance, 1e-9 * max(1.0, abs(base)))
        rows.append(
            {
                "Check": workload_name,
                "PureComputeBaseline": base,
                "Q2Balanced": q2,
                "AbsoluteDifference": difference,
                "AllowedDifference": allowed,
                "Status": "PASS" if difference <= allowed else "FAIL",
                "Interpretation": "相同任务集合和执行时长下，调度只改变时空分布",
            }
        )
    rows.extend(
        _metric_consistency_rows(
            baseline,
            official_summary,
            "Q1PureComputeBaseline",
            tolerance,
        )
    )
    rows.extend(
        _metric_consistency_rows(
            balanced,
            official_summary,
            "Q2Balanced",
            tolerance,
        )
    )
    rows.append(
        {
            "Check": "能源结算函数口径",
            "Status": "PASS_SCOPE",
            "Interpretation": (
                "基线与Q2均由同一份0--2405输入、PUE、购售电边界和无储能分段函数独立重算"
            ),
        }
    )
    logging.info(
        "检验2/4完成：Q2成本变化=%.8g，新能源利用率变化=%.8g，"
        "GPU工作量差=%.6g，AI IT能量差=%.6g。",
        balanced.metrics["Cost"] - baseline.metrics["Cost"],
        balanced.metrics["RenewableUtilizationRate"]
        - baseline.metrics["RenewableUtilizationRate"],
        balanced.workload["AssignedGPUWorkload_GPUh"]
        - baseline.workload["AssignedGPUWorkload_GPUh"],
        balanced.workload["AssignedAIITEnergy_MWh"]
        - baseline.workload["AssignedAIITEnergy_MWh"],
    )
    return pd.DataFrame(rows)


def _read_rolling_windows(root: Path) -> tuple[pd.DataFrame | None, Path | None]:
    path = _locate_file(root, "q2_balanced_rolling_windows.csv")
    if path is None:
        return None, None
    return pd.read_csv(path, encoding="utf-8-sig"), path


def _lns_ablation(
    windows: pd.DataFrame | None,
    solver_log_path: Path | None,
    tolerance: float,
) -> pd.DataFrame:
    logging.info("检验3/4-LNS消融开始：读取正式窗口日志中的LNS前后独立复算z。")
    if windows is None:
        return pd.DataFrame(
            [
                {
                    "Check": "LNS消融",
                    "Status": "PENDING_NO_ROLLING_WINDOWS",
                    "Interpretation": "缺少q2_balanced_rolling_windows.csv",
                }
            ]
        )
    required = {"WindowID", "ZBeforeLNS", "ZStar", "LNSStatus"}
    if not required.issubset(windows.columns):
        return pd.DataFrame(
            [
                {
                    "Check": "LNS消融",
                    "Status": "PENDING_INCOMPLETE_LNS_FIELDS",
                    "Interpretation": f"缺少字段：{sorted(required - set(windows.columns))}",
                }
            ]
        )
    frame = windows.copy()
    for column in ("WindowID", "ZBeforeLNS", "ZStar", "LNSAccepted"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[
        frame["ZBeforeLNS"].notna()
        & frame["ZStar"].notna()
        & frame["LNSStatus"].astype(str).eq("feasible")
    ].copy()
    if frame.empty:
        return pd.DataFrame(
            [
                {
                    "Check": "LNS消融",
                    "Status": "PENDING_NO_FEASIBLE_LNS_RECORD",
                    "Interpretation": "没有可用于比较LNS前后z的窗口记录",
                }
            ]
        )
    solver = None
    if solver_log_path is not None and solver_log_path.is_file():
        solver = pd.read_csv(solver_log_path, encoding="utf-8-sig")
        if {"WindowID", "Stage"}.issubset(solver.columns):
            solver = solver.loc[
                solver["Stage"].astype(str).eq("large_neighborhood_milp_refinement"),
                ["WindowID", *[
                    column
                    for column in ("MIPGap", "ElapsedSeconds", "Status")
                    if column in solver.columns
                ]],
            ].copy()
            solver["WindowID"] = pd.to_numeric(solver["WindowID"], errors="coerce")
            frame = frame.merge(solver, how="left", on="WindowID", suffixes=("", "_solver"))
    frame["Improvement"] = frame["ZBeforeLNS"] - frame["ZStar"]
    frame["ImprovementRate"] = frame["Improvement"] / frame["ZBeforeLNS"].abs().clip(
        lower=max(tolerance, FLOAT_EPS)
    )
    frame["ImprovedByIndependentZ"] = np.where(
        frame["Improvement"] > tolerance,
        1,
        0,
    )
    if "LNSAccepted" in frame.columns:
        frame["AcceptanceFlagMismatch"] = (
            frame["ImprovedByIndependentZ"]
            != frame["LNSAccepted"].fillna(0).astype(int)
        ).astype(int)
    else:
        frame["AcceptanceFlagMismatch"] = np.nan
    output_rows = frame.to_dict(orient="records")
    accepted = frame.loc[frame["ImprovedByIndependentZ"].eq(1)]
    mismatch = int(frame["AcceptanceFlagMismatch"].fillna(0).sum())
    output_rows.append(
        {
            "Check": "LNS消融汇总",
            "HeuristicWindowCount": int(len(frame)),
            "LNSImprovedWindowCount": int(len(accepted)),
            "LNSImprovedWindowRate": float(len(accepted) / len(frame)),
            "MeanImprovementRateOnImprovedWindows": (
                float(accepted["ImprovementRate"].mean())
                if not accepted.empty
                else float("nan")
            ),
            "AcceptanceFlagMismatchCount": mismatch,
            "Status": "PASS_REPORTED" if mismatch == 0 else "FAIL_LOG_INCONSISTENCY",
            "Interpretation": (
                "ZBeforeLNS/ZStar来自模型对候选计划的独立容量与目标复算；"
                "本表检验LNS内部消融，不等同于全局精确最优性证明"
            ),
        }
    )
    logging.info(
        "检验3/4-LNS消融完成：可比较启发式窗口=%d，LNS改善=%d（%.1f%%），"
        "接受标记不一致=%d。",
        len(frame),
        len(accepted),
        100.0 * len(accepted) / len(frame),
        mismatch,
    )
    return pd.DataFrame(output_rows)


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def _discover_exact_pair(
    experiment_root: Path | None,
    exact_path: Path | None,
    heuristic_path: Path | None,
) -> tuple[Path | None, Path | None]:
    if exact_path is not None or heuristic_path is not None:
        return exact_path, heuristic_path
    if experiment_root is None:
        return None, None
    pair_specs = (
        (
            experiment_root / "exact_vs_heuristic" / "exact_windows.csv",
            experiment_root / "exact_vs_heuristic" / "heuristic_windows.csv",
        ),
        (
            experiment_root / "exact_vs_heuristic" / "exact" / "q2_balanced_rolling_windows.csv",
            experiment_root / "exact_vs_heuristic" / "heuristic" / "q2_balanced_rolling_windows.csv",
        ),
        (
            experiment_root / "exact" / "q2_balanced_rolling_windows.csv",
            experiment_root / "heuristic" / "q2_balanced_rolling_windows.csv",
        ),
        (
            experiment_root / "q2_exact_windows.csv",
            experiment_root / "q2_heuristic_windows.csv",
        ),
    )
    for exact_candidate, heuristic_candidate in pair_specs:
        if exact_candidate.is_file() and heuristic_candidate.is_file():
            return exact_candidate, heuristic_candidate
    return None, None


def _window_metric_column(
    frame: pd.DataFrame,
    metric: str,
) -> tuple[str | None, str]:
    aliases = {
        "Z": ("ZStar", "FinalZ", "EstimatedMaxNormalizedDeviation", "z"),
        "Cost": ("Cost", "OperatingCost_CNY", "WindowCost_CNY", "EstimatedGlobal_Cost"),
        "Carbon": (
            "Carbon",
            "CarbonEmission_tCO2",
            "WindowCarbon_tCO2",
            "EstimatedGlobal_Carbon",
        ),
        "MeanLatency": (
            "MeanLatency",
            "MeanNetworkLatency_ms",
            "WindowMeanLatency_ms",
            "EstimatedGlobal_MeanLatency",
        ),
        "RenewableUnusedRate": (
            "RenewableUnusedRate",
            "WindowRenewableUnusedRate",
            "EstimatedGlobal_RenewableUnusedRate",
        ),
        "ElapsedSeconds": ("ElapsedSeconds", "RuntimeSeconds", "TotalElapsedSeconds"),
    }
    for column in aliases[metric]:
        if column in frame.columns:
            source = "estimated" if column.startswith("EstimatedGlobal_") else "reported"
            return column, source
    return None, "missing"


def _exact_vs_heuristic(
    experiment_root: Path | None,
    exact_path: Path | None,
    heuristic_path: Path | None,
    tolerance: float,
) -> pd.DataFrame:
    logging.info("检验3/4-精确模型对照开始：目标窗口0--9。")
    exact_file, heuristic_file = _discover_exact_pair(
        experiment_root,
        exact_path,
        heuristic_path,
    )
    if exact_file is None or heuristic_file is None:
        logging.info(
            "检验3/4-精确模型对照待补：需要单独保存exact_windows.csv和heuristic_windows.csv；"
            "当前正式结果不能把前10个精确窗口与后91个启发式窗口错配比较。"
        )
        return pd.DataFrame(
            [
                {
                    "Check": "精确模型对照",
                    "Status": "PENDING_EXACT_HEURISTIC_ARTIFACT",
                    "ExactWindowsPath": str(exact_file) if exact_file else "",
                    "HeuristicWindowsPath": str(heuristic_file) if heuristic_file else "",
                    "Interpretation": (
                        "必须在相同历史状态、任务集合、H/K和定标参数下保存同一窗口的"
                        "exact与heuristic结果；当前正式表缺少同窗启发式对照"
                    ),
                }
            ]
        )
    exact = pd.read_csv(exact_file, encoding="utf-8-sig")
    heuristic = pd.read_csv(heuristic_file, encoding="utf-8-sig")
    if "WindowID" not in exact.columns or "WindowID" not in heuristic.columns:
        return pd.DataFrame(
            [
                {
                    "Check": "精确模型对照",
                    "Status": "PENDING_MISSING_WINDOW_ID",
                    "Interpretation": "exact与heuristic结果必须包含WindowID",
                }
            ]
        )
    exact = exact.copy()
    heuristic = heuristic.copy()
    exact["WindowID"] = pd.to_numeric(exact["WindowID"], errors="coerce")
    heuristic["WindowID"] = pd.to_numeric(heuristic["WindowID"], errors="coerce")
    exact = exact.loc[exact["WindowID"].between(0, EXACT_WINDOW_COUNT - 1)]
    heuristic = heuristic.loc[
        heuristic["WindowID"].between(0, EXACT_WINDOW_COUNT - 1)
    ]
    merged = exact.merge(
        heuristic,
        how="inner",
        on="WindowID",
        suffixes=("_exact", "_heuristic"),
    )
    if len(merged) < EXACT_WINDOW_COUNT:
        return pd.DataFrame(
            [
                {
                    "Check": "精确模型对照",
                    "Status": "PENDING_INCOMPLETE_COMMON_WINDOWS",
                    "CommonWindowCount": len(merged),
                    "RequiredWindowCount": EXACT_WINDOW_COUNT,
                    "Interpretation": "窗口0--9必须全部存在于两套同窗结果中",
                }
            ]
        )
    rows: list[dict[str, object]] = []
    all_delta_z: list[float] = []
    all_complete = True
    for row in merged.sort_values("WindowID").itertuples(index=False):
        row_dict = row._asdict()
        output: dict[str, object] = {
            "Check": "精确模型同窗对照",
            "WindowID": row_dict["WindowID"],
            "ExactPath": str(exact_file),
            "HeuristicPath": str(heuristic_file),
        }
        z_exact_column, z_exact_source = _window_metric_column(exact, "Z")
        z_heur_column, z_heur_source = _window_metric_column(heuristic, "Z")
        if z_exact_column is None or z_heur_column is None:
            all_complete = False
        else:
            z_exact = float(row_dict[f"{z_exact_column}_exact"])
            z_heur = float(row_dict[f"{z_heur_column}_heuristic"])
            delta_z = (z_heur - z_exact) / max(abs(z_exact), 1e-8)
            output.update(
                {
                    "ZExact": z_exact,
                    "ZHeuristic": z_heur,
                    "DeltaZRelative": delta_z,
                    "ZWithin5Percent": int(delta_z <= 0.05 + tolerance),
                    "ZMetricSourceExact": z_exact_source,
                    "ZMetricSourceHeuristic": z_heur_source,
                }
            )
            all_delta_z.append(delta_z)
        for metric in OBJECTIVE_NAMES:
            exact_column, exact_source = _window_metric_column(exact, metric)
            heuristic_column, heuristic_source = _window_metric_column(heuristic, metric)
            if exact_column is None or heuristic_column is None:
                all_complete = False
                continue
            exact_value = float(row_dict[f"{exact_column}_exact"])
            heuristic_value = float(row_dict[f"{heuristic_column}_heuristic"])
            difference = heuristic_value - exact_value
            relative = difference / max(abs(exact_value), 1e-8)
            output.update(
                {
                    f"{metric}Exact": exact_value,
                    f"{metric}Heuristic": heuristic_value,
                    f"Delta{metric}": difference,
                    f"RelativeDelta{metric}": relative,
                    f"{metric}MetricSourceExact": exact_source,
                    f"{metric}MetricSourceHeuristic": heuristic_source,
                }
            )
        elapsed_exact_column, _ = _window_metric_column(exact, "ElapsedSeconds")
        elapsed_heur_column, _ = _window_metric_column(heuristic, "ElapsedSeconds")
        if elapsed_exact_column and elapsed_heur_column:
            output["ElapsedExactSeconds"] = float(
                row_dict[f"{elapsed_exact_column}_exact"]
            )
            output["ElapsedHeuristicSeconds"] = float(
                row_dict[f"{elapsed_heur_column}_heuristic"]
            )
        output["Status"] = "REPORTED"
        rows.append(output)
    max_delta_z = max(all_delta_z) if all_delta_z else float("nan")
    rows.append(
        {
            "Check": "精确模型对照汇总",
            "CommonWindowCount": len(merged),
            "RequiredWindowCount": EXACT_WINDOW_COUNT,
            "MaxDeltaZRelative": max_delta_z,
            "AllZWithin5Percent": (
                int(all_delta_z and max_delta_z <= 0.05 + tolerance)
                if all_delta_z
                else 0
            ),
            "Status": (
                "PASS_WITHIN_5_PERCENT"
                if all_complete and all_delta_z and max_delta_z <= 0.05 + tolerance
                else "REPORTED_OVER_5_PERCENT"
                if all_complete and all_delta_z
                else "PENDING_INCOMPLETE_METRICS"
            ),
            "Interpretation": (
                "5%只作为与当前LNS局部精度同量级的报告界限，不是全局最优性证明"
            ),
        }
    )
    logging.info(
        "检验3/4-精确模型对照完成：共同窗口=%d，最大z相对差=%s。",
        len(merged),
        f"{max_delta_z:.6g}" if math.isfinite(max_delta_z) else "NA",
    )
    return pd.DataFrame(rows)


def _run_root_for_lookahead(
    experiment_root: Path | None,
    lookahead: int,
) -> Path | None:
    if experiment_root is None:
        return None
    names = (
        f"K{lookahead}",
        f"k{lookahead}",
        f"lookahead_K{lookahead}",
        f"lookahead_{lookahead}",
        f"K_{lookahead}",
    )
    for name in names:
        candidate = experiment_root / name
        if candidate.is_dir():
            return candidate
    return None


def _read_runtime_seconds(root: Path) -> float:
    log_candidates = (
        root / "question_02_model_refactored.log",
        root / "logs" / "question_02_model_refactored.log",
        root / "question_02_model.log",
        root / "logs" / "question_02_model.log",
    )
    path = _first_existing(log_candidates)
    if path is None:
        return float("nan")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return float("nan")
    matches = re.findall(r"总耗时=([0-9]+(?:\.[0-9]+)?)s", text)
    return float(matches[-1]) if matches else float("nan")


def _summary_row(root: Path) -> tuple[pd.DataFrame | None, pd.Series | None, pd.Series | None]:
    path = _locate_file(root, "q2_objective_summary.csv")
    if path is None:
        return None, None, None
    summary = pd.read_csv(path, encoding="utf-8-sig")
    if "Solution" not in summary.columns:
        return summary, None, None
    balanced = summary.loc[
        summary["Solution"].astype(str).eq("Q2Balanced")
    ]
    baseline = summary.loc[
        summary["Solution"].astype(str).eq("Q1PureComputeBaseline")
    ]
    return (
        summary,
        balanced.iloc[0] if not balanced.empty else None,
        baseline.iloc[0] if not baseline.empty else None,
    )


def _lookahead_stability(
    inputs: Inputs,
    experiment_root: Path | None,
    balanced_core: AuditResult,
    deadline: Deadline,
    tolerance: float,
    progress_every: int,
) -> pd.DataFrame:
    logging.info("检验4/4：开始滚动前瞻稳定性检验，目标K=24/48/72，H=24。")
    records: list[dict[str, object]] = []
    current_summary, current_balanced, current_baseline = _summary_row(TABLES_DIR)
    config_path = _locate_file(TABLES_DIR, "q2_model_config.csv")
    current_k = None
    if config_path is not None:
        config = pd.read_csv(config_path, encoding="utf-8-sig")
        if not config.empty and "LookaheadHours" in config.columns:
            current_k = int(float(config.iloc[0]["LookaheadHours"]))
    for lookahead in (24, 48, 72):
        deadline.check(f"K={lookahead}前瞻稳定性")
        if lookahead == current_k:
            root = TABLES_DIR
            balanced_row = current_balanced
            baseline_row = current_baseline
            source = "formal_outputs"
            hard_status = (
                "PASS"
                if balanced_core.logical_count == 0
                and balanced_core.vmax <= tolerance
                and balanced_core.emax <= tolerance
                else "FAIL"
            )
            runtime = _read_runtime_seconds(QUESTION_DIR / "outputs")
            logging.info("检验4/4：K=%d使用正式结果，硬约束状态=%s。", lookahead, hard_status)
        else:
            root = _run_root_for_lookahead(experiment_root, lookahead)
            source = str(root) if root is not None else ""
            summary, balanced_row, baseline_row = (
                _summary_row(root) if root is not None else (None, None, None)
            )
            runtime = _read_runtime_seconds(root) if root is not None else float("nan")
            hard_status = "NOT_AUDITED"
            if root is not None:
                assignment_path = _locate_file(root, "q2_assignments.csv")
                if assignment_path is None:
                    assignment_path = _locate_file(root, "q2_balanced_assignments.csv")
                if assignment_path is not None:
                    try:
                        alternative, path = _load_assignments(
                            root,
                            ("q2_assignments.csv", "q2_balanced_assignments.csv"),
                        )
                        logical_detail, logical_count, joined = _audit_logical(
                            inputs,
                            alternative,
                            f"K{lookahead}",
                            deadline,
                            tolerance,
                        )
                        (
                            _profile,
                            _continuous,
                            _metrics,
                            _workload,
                            vmax,
                            emax,
                            _terminal,
                        ) = _recompute_profile(
                            inputs,
                            joined,
                            f"K{lookahead}",
                            deadline,
                            progress_every,
                            tolerance,
                        )
                        hard_status = (
                            "PASS"
                            if logical_count == 0
                            and vmax <= tolerance
                            and emax <= tolerance
                            else "FAIL"
                        )
                    except (ValueError, KeyError, TypeError, pd.errors.ParserError) as exc:
                        hard_status = f"FAIL_AUDIT_ERROR:{exc}"
                else:
                    logging.info(
                        "检验4/4：K=%d只有指标文件，没有任务安排，硬约束暂不能审计。",
                        lookahead,
                    )
            else:
                logging.info(
                    "检验4/4：K=%d未找到独立运行目录，等待补跑结果。",
                    lookahead,
                )
        record: dict[str, object] = {
            "LookaheadHours": lookahead,
            "DecisionWindowHours": 24,
            "Source": source,
            "RuntimeSeconds": runtime,
            "HardConstraintStatus": hard_status,
        }
        if balanced_row is None or baseline_row is None:
            record.update(
                {
                    "Status": "PENDING_OBJECTIVE_SUMMARY",
                    "Interpretation": "需要同一K设置下同时保存Q2Balanced和Q1PureComputeBaseline",
                }
            )
            records.append(record)
            continue
        for metric, column in OBJECTIVE_COLUMNS.items():
            record[f"Q2_{metric}"] = float(balanced_row[column])
            record[f"Baseline_{metric}"] = float(baseline_row[column])
        baseline_util = (
            float(baseline_row["RenewableUtilizationRate"])
            if "RenewableUtilizationRate" in baseline_row.index
            else 1.0 - float(baseline_row[OBJECTIVE_COLUMNS["RenewableUnusedRate"]])
        )
        q2_util = (
            float(balanced_row["RenewableUtilizationRate"])
            if "RenewableUtilizationRate" in balanced_row.index
            else 1.0 - float(balanced_row[OBJECTIVE_COLUMNS["RenewableUnusedRate"]])
        )
        cost_lower = float(balanced_row[OBJECTIVE_COLUMNS["Cost"]]) < float(
            baseline_row[OBJECTIVE_COLUMNS["Cost"]]
        ) - tolerance
        renewable_higher = q2_util > baseline_util + tolerance
        record.update(
            {
                "Q2_RenewableUtilizationRate": q2_util,
                "Baseline_RenewableUtilizationRate": baseline_util,
                "CostDirection": "PASS" if cost_lower else "FAIL_DIRECTION",
                "RenewableDirection": "PASS" if renewable_higher else "FAIL_DIRECTION",
                "ConclusionDirection": (
                    "PASS"
                    if cost_lower and renewable_higher
                    else "FAIL_DIRECTION"
                ),
                "Status": (
                    "PASS_HARD_AND_DIRECTION"
                    if hard_status == "PASS" and cost_lower and renewable_higher
                    else "FAIL_HARD_CONSTRAINT"
                    if hard_status == "FAIL"
                    else "FAIL_DIRECTION"
                    if cost_lower is False or renewable_higher is False
                    else "PENDING_HARD_AUDIT"
                ),
                "Interpretation": (
                    "只判断成本下降和新能源利用率提高的方向，不设置人为百分比阈值"
                ),
            }
        )
        records.append(record)
        logging.info(
            "检验4/4：K=%d完成，硬约束=%s，成本方向=%s，新能源方向=%s。",
            lookahead,
            hard_status,
            record["CostDirection"],
            record["RenewableDirection"],
        )

    available = [
        record
        for record in records
        if record.get("Status") == "PASS_HARD_AND_DIRECTION"
    ]
    all_three = len(records) == 3 and len(available) == 3
    missing_lookahead = [
        int(record["LookaheadHours"])
        for record in records
        if str(record.get("Status", "")).startswith("PENDING")
    ]
    records.append(
        {
            "Check": "前瞻稳定性结论",
            "ComparedLookaheadHours": "24,48,72",
            "Status": (
                "PASS_DIRECTION_STABLE"
                if all_three
                else "PENDING_K24_K72_OR_HARD_AUDIT"
                if missing_lookahead or len(available) < 3
                else "FAIL_DIRECTION_OR_HARD_CONSTRAINT"
            ),
            "MissingOrPendingK": ",".join(map(str, missing_lookahead)),
            "Interpretation": (
                "三种K均满足硬约束且成本下降、新能源利用率提高时，"
                "才支持对前瞻长度稳定性的正文表述"
            ),
        }
    )
    logging.info(
        "检验4/4完成：K=24/48/72可通过记录=%d/3，待补K=%s。",
        len(available),
        missing_lookahead or "无",
    )
    return pd.DataFrame(records)


def _overview(
    balanced: AuditResult,
    baseline: AuditResult,
    exact: pd.DataFrame,
    lns: pd.DataFrame,
    rolling: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:
    def _status(frame: pd.DataFrame, check: str) -> str:
        if frame.empty or "Status" not in frame.columns:
            return "NO_OUTPUT"
        matches = frame.loc[
            frame.get("Check", pd.Series(dtype=str)).astype(str).str.contains(
                check,
                na=False,
            )
        ]
        if matches.empty:
            return "REPORTED"
        return ";".join(sorted(set(matches["Status"].astype(str))))

    core_pass = (
        balanced.logical_count == 0
        and balanced.vmax <= tolerance
        and balanced.emax <= tolerance
        and baseline.logical_count == 0
        and baseline.vmax <= tolerance
        and baseline.emax <= tolerance
    )
    return pd.DataFrame(
        [
            {
                "Layer": "全时域独立复算与守恒",
                "Status": "PASS" if core_pass else "FAIL",
                "Evidence": "q2_validation_constraint_audit.csv及独立重算资源剖面",
            },
            {
                "Layer": "同口径基线与机制一致性",
                "Status": "PASS_SCOPE_REPORTED",
                "Evidence": "q2_validation_baseline_consistency.csv",
            },
            {
                "Layer": "LNS消融",
                "Status": _status(lns, "LNS消融汇总"),
                "Evidence": "q2_validation_lns_ablation.csv",
            },
            {
                "Layer": "精确模型同窗对照",
                "Status": _status(exact, "精确模型对照"),
                "Evidence": "q2_validation_exact_vs_heuristic.csv",
            },
            {
                "Layer": "滚动前瞻K=24/48/72",
                "Status": _status(rolling, "前瞻稳定性结论"),
                "Evidence": "q2_validation_lookahead.csv",
            },
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="问题2新模型检验：独立复算、同口径基线、LNS消融、精确对照和K稳定性"
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=DEFAULT_MAX_SECONDS,
        help="全流程验证预算，默认600秒；超时会显式退出",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="硬约束和守恒残差容差，默认1e-6",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="独立资源重算每处理多少条任务反馈一次，默认1000",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=VALIDATION_RUNS_DIR,
        help="精确对照和K=24/72结果根目录，默认outputs/validation_runs",
    )
    parser.add_argument(
        "--exact-windows",
        type=Path,
        default=None,
        help="可选：exact同窗窗口CSV，优先于--experiment-root自动发现",
    )
    parser.add_argument(
        "--heuristic-windows",
        type=Path,
        default=None,
        help="可选：heuristic同窗窗口CSV，优先于--experiment-root自动发现",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="控制台和文件日志级别，默认INFO",
    )
    args = parser.parse_args(argv)
    if args.max_seconds <= 0:
        raise SystemExit("--max-seconds必须为正")
    if args.tolerance <= 0:
        raise SystemExit("--tolerance必须为正")
    if args.progress_every <= 0:
        raise SystemExit("--progress-every必须为正")

    _configure_logging(args.log_level)
    started_at = time.perf_counter()
    deadline = Deadline(args.max_seconds, started_at)
    logging.info(
        "问题2新模型检验启动：不调用model.py/MILP；预算=%.1fs；"
        "容差=%.3g；资源重算每%d条反馈；对照根目录=%s。",
        args.max_seconds,
        args.tolerance,
        args.progress_every,
        args.experiment_root,
    )

    try:
        inputs = _load_inputs()
        deadline.check("输入读取")
        balanced_assignments, balanced_path = _load_assignments(
            TABLES_DIR,
            ("q2_assignments.csv", "q2_balanced_assignments.csv"),
        )
        baseline_assignments, baseline_path = _load_assignments(
            TABLES_DIR,
            ("q2_baseline_assignments.csv",),
        )
        deadline.check("正式任务安排读取")

        logging.info("阶段2/4：开始Q2Balanced全时域独立审计。")
        balanced_logical, balanced_count, balanced_joined = _audit_logical(
            inputs,
            balanced_assignments,
            "Q2Balanced",
            deadline,
            args.tolerance,
        )
        (
            balanced_profile,
            balanced_continuous,
            balanced_metrics,
            balanced_workload,
            balanced_vmax,
            balanced_emax,
            balanced_terminal,
        ) = _recompute_profile(
            inputs,
            balanced_joined,
            "Q2Balanced",
            deadline,
            args.progress_every,
            args.tolerance,
        )
        balanced_profile_compare = _compare_exported_profile(
            balanced_profile,
            _locate_file(TABLES_DIR, "q2_resource_profile.csv"),
            "Q2Balanced",
            args.tolerance,
        )
        balanced = AuditResult(
            label="Q2Balanced",
            assignments=balanced_assignments,
            assignment_path=balanced_path,
            logical_detail=balanced_logical,
            logical_count=balanced_count,
            joined=balanced_joined,
            profile=balanced_profile,
            continuous_detail=balanced_continuous,
            metrics=balanced_metrics,
            workload=balanced_workload,
            vmax=balanced_vmax,
            emax=balanced_emax,
            terminal_overlap_count=balanced_terminal,
            profile_compare=balanced_profile_compare,
        )
        logging.info("阶段2/4：开始Q1PureComputeBaseline同口径独立审计。")
        baseline_logical, baseline_count, baseline_joined = _audit_logical(
            inputs,
            baseline_assignments,
            "Q1PureComputeBaseline",
            deadline,
            args.tolerance,
        )
        (
            baseline_profile,
            baseline_continuous,
            baseline_metrics,
            baseline_workload,
            baseline_vmax,
            baseline_emax,
            baseline_terminal,
        ) = _recompute_profile(
            inputs,
            baseline_joined,
            "Q1PureComputeBaseline",
            deadline,
            args.progress_every,
            args.tolerance,
        )
        baseline_profile_compare = _compare_exported_profile(
            baseline_profile,
            _locate_file(TABLES_DIR, "q2_baseline_resource_profile.csv"),
            "Q1PureComputeBaseline",
            args.tolerance,
        )
        baseline = AuditResult(
            label="Q1PureComputeBaseline",
            assignments=baseline_assignments,
            assignment_path=baseline_path,
            logical_detail=baseline_logical,
            logical_count=baseline_count,
            joined=baseline_joined,
            profile=baseline_profile,
            continuous_detail=baseline_continuous,
            metrics=baseline_metrics,
            workload=baseline_workload,
            vmax=baseline_vmax,
            emax=baseline_emax,
            terminal_overlap_count=baseline_terminal,
            profile_compare=baseline_profile_compare,
        )
        constraint_audit = pd.concat(
            [
                balanced.logical_detail,
                baseline.logical_detail,
                balanced.continuous_detail,
                baseline.continuous_detail,
                balanced.profile_compare,
                baseline.profile_compare,
            ],
            ignore_index=True,
            sort=False,
        )
        _write_table(constraint_audit, "q2_validation_constraint_audit.csv")
        _write_table(
            pd.concat(
                [balanced_profile, baseline_profile],
                keys=["Q2Balanced", "Q1PureComputeBaseline"],
                names=["Model", "Row"],
            ).reset_index(level=0),
            "q2_validation_recomputed_profiles.csv",
        )
        official_summary = _official_summary(TABLES_DIR / "q2_objective_summary.csv")
        baseline_consistency = _baseline_consistency(
            inputs,
            baseline,
            balanced,
            official_summary,
            args.tolerance,
        )
        _write_table(
            baseline_consistency,
            "q2_validation_baseline_consistency.csv",
        )
        logging.info(
            "阶段2/4完成：Q2逻辑违规=%d、Vmax=%.6g、Emax=%.6g；"
            "基线逻辑违规=%d、Vmax=%.6g、Emax=%.6g。",
            balanced.logical_count,
            balanced.vmax,
            balanced.emax,
            baseline.logical_count,
            baseline.vmax,
            baseline.emax,
        )

        logging.info("阶段3/4：开始LNS消融和精确模型同窗对照。")
        rolling_windows, rolling_path = _read_rolling_windows(TABLES_DIR)
        solver_path = _locate_file(TABLES_DIR, "q2_solver_log.csv")
        lns = _lns_ablation(rolling_windows, solver_path, args.tolerance)
        exact = _exact_vs_heuristic(
            args.experiment_root,
            args.exact_windows,
            args.heuristic_windows,
            args.tolerance,
        )
        _write_table(lns, "q2_validation_lns_ablation.csv")
        _write_table(exact, "q2_validation_exact_vs_heuristic.csv")
        logging.info(
            "阶段3/4完成：正式窗口=%d，LNS结果=%s，精确对照来源=%s。",
            len(rolling_windows) if rolling_windows is not None else 0,
            "已生成" if not lns.empty else "无",
            rolling_path if rolling_path else "待补",
        )

        logging.info("阶段4/4：开始K=24/48/72前瞻稳定性检验。")
        lookahead = _lookahead_stability(
            inputs,
            args.experiment_root,
            balanced,
            deadline,
            args.tolerance,
            args.progress_every,
        )
        _write_table(lookahead, "q2_validation_lookahead.csv")
        overview = _overview(
            balanced,
            baseline,
            exact,
            lns,
            lookahead,
            args.tolerance,
        )
        _write_table(overview, "q2_validation_overview.csv")
    except FileNotFoundError as exc:
        logging.error("问题2新模型检验缺少输入或正式结果：%s", exc)
        return 1
    except ValidationTimeout as exc:
        logging.error("问题2新模型检验安全停止：%s", exc)
        return 2
    except (ValueError, KeyError, TypeError, pd.errors.ParserError) as exc:
        logging.error("问题2新模型检验执行失败：%s", exc)
        return 1

    elapsed = time.perf_counter() - started_at
    core_pass = (
        balanced.logical_count == 0
        and baseline.logical_count == 0
        and balanced.vmax <= args.tolerance
        and baseline.vmax <= args.tolerance
        and balanced.emax <= args.tolerance
        and baseline.emax <= args.tolerance
    )
    pending_optional = (
        "PENDING" in ";".join(overview["Status"].astype(str).tolist())
    )
    logging.info(
        "问题2新模型检验完成：核心独立复算=%s；可选精确/K对照=%s；总耗时=%.2fs。",
        "PASS" if core_pass else "FAIL",
        "存在待补证据" if pending_optional else "已完成",
        elapsed,
    )
    return 0 if core_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
