"""C题第一问及全题共享数据的预处理入口。

本模块只生成两类结果：

* ``data/processed/shared``：Q1、Q2、Q3、Q4共同使用的确定性数据层；
* ``data/processed/q1``：第一问的需求预测面板和基础调度容量接口。

不在预处理阶段生成任务的目标区域、开工/完工时刻、逐小时运行矩阵或能源
优化结果。第2406小时只保留为终端结算时点，不生成任何任务执行记录。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


QUESTION_DIR = Path(__file__).resolve().parent
RAW_DIR = QUESTION_DIR / "data" / "raw"
PROCESSED_DIR = QUESTION_DIR / "data" / "processed"
SHARED_DIR = PROCESSED_DIR / "shared"
Q1_DIR = PROCESSED_DIR / "q1"

REGIONS: tuple[str, ...] = (
    "RegionA",
    "RegionB",
    "RegionC",
    "RegionD",
    "RegionE",
    "RegionF",
)
TASK_TYPES: tuple[str, ...] = (
    "RealTimeInference",
    "BatchInference",
    "AITraining",
)

WORKLOAD_START_HOUR = 0
WORKLOAD_END_HOUR = 2399
INITIAL_TRAIN_END_HOUR = 2351
VALIDATION_START_HOUR = 2352
VALIDATION_END_HOUR = 2375
TEST_START_HOUR = 2376
TEST_END_HOUR = 2399
TAIL_END_HOUR = 2405
TERMINAL_HOUR = 2406

TASK_SOURCE = "workload_trace.xlsx"
GPU_SOURCE = "GPU_information.xlsx"
LATENCY_SOURCE = "network_latency.xlsx"
POWER_MAPPING_SOURCE = "power_mapping.xlsx"
REGION_TIME_SOURCE = "region_time_data.xlsx"
STORAGE_SOURCE = "storage_information.xlsx"

TASK_COLUMNS: tuple[str, ...] = (
    "TaskID",
    "TaskType",
    "ArrivalHour",
    "SourceRegion",
    "GPU_Demand",
    "EstimatedDuration_min",
    "DelaySensitivity",
    "MaxLatency_ms",
    "EarliestStartHour",
    "LatestFinishHour",
    "ExecutionMode",
)
GPU_COLUMNS: tuple[str, ...] = (
    "Region",
    "Total_GPU",
    "Available_GPU",
    "Max_IT_Power_MW",
    "PUE",
    "Max_Facility_Power_MW",
)
LATENCY_COLUMNS: tuple[str, ...] = (
    "FromRegion",
    "ToRegion",
    "NetworkLatency_ms",
)
POWER_MAPPING_COLUMNS: tuple[str, ...] = (
    "TaskType",
    "GPU_Power_MW_per_EquivalentGPU",
)
REGION_TIME_EXOGENOUS_COLUMNS: tuple[str, ...] = (
    "Hour",
    "Region",
    "ElectricityPrice_CNY_per_MWh",
    "SellPrice_CNY_per_MWh",
    "CarbonIntensity_tCO2_per_MWh",
    "AvailableRenewable_MW",
    "NonAI_IT_Load_MW",
)
REGION_TIME_BASELINE_COLUMNS: tuple[str, ...] = (
    "Hour",
    "Region",
    "Baseline_AI_IT_Load_MW",
    "IT_Load_MW",
    "Total_Load_MW",
    "UsedRenewable_MW",
    "RenewableCharge_MW",
    "Curtailment_MW",
    "GridPurchase_MW",
    "GridCharge_MW",
    "GridSell_MW",
    "NetGridImport_MW",
    "CarbonEmission_tCO2",
    "SOC_MWh",
    "ChargePower_MW",
    "DischargePower_MW",
)
OPTIONAL_REGION_TIME_COLUMNS: tuple[str, ...] = ("PricePeriod", "DataPeriod")
STORAGE_COLUMNS: tuple[str, ...] = (
    "Region",
    "StorageCapacity_MWh",
    "MinSOC_MWh",
    "InitialSOC_MWh",
    "MaxChargePower_MW",
    "MaxDischargePower_MW",
    "ChargeEfficiency",
    "DischargeEfficiency",
    "SellLimit_MW",
    "MaxGridImport_MW",
    "MaxGridExport_MW",
)


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(columns=lambda value: str(value).strip())
    if renamed.columns.duplicated().any():
        duplicated = renamed.columns[renamed.columns.duplicated()].tolist()
        raise ValueError(f"工作表字段名重复：{duplicated}")
    return renamed


def _read_table(path: Path, required_columns: Sequence[str]) -> pd.DataFrame:
    """从带字段说明页的工作簿中定位主数据表。"""

    if not path.is_file():
        raise FileNotFoundError(f"找不到输入附件：{path}")

    try:
        workbook = pd.ExcelFile(path)
    except ImportError as exc:
        raise RuntimeError(
            f"无法读取 {path.name}：当前Python环境缺少Excel读取引擎；"
            "请在既有比赛环境中提供与pandas兼容的xlsx读取引擎。"
        ) from exc

    required = set(required_columns)
    sheets = ", ".join(workbook.sheet_names)
    try:
        for sheet_name in workbook.sheet_names:
            candidate = _normalise_columns(pd.read_excel(workbook, sheet_name=sheet_name))
            if required.issubset(candidate.columns):
                return candidate
    finally:
        workbook.close()

    raise ValueError(
        f"{path.name}中没有找到包含字段{sorted(required)}的主数据表；"
        f"已检查工作表：{sheets}"
    )


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], source_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source_name}缺少必要字段：{missing}")


def _require_no_missing(frame: pd.DataFrame, columns: Iterable[str], source_name: str) -> None:
    selected = list(columns)
    missing_counts = frame[selected].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        details = ", ".join(f"{column}={int(count)}" for column, count in missing_counts.items())
        raise ValueError(f"{source_name}必要字段存在缺失值：{details}")


def _coerce_numeric(frame: pd.DataFrame, columns: Iterable[str], source_name: str) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{source_name}字段{column}包含非有限数值或缺失值")


def _coerce_integer_columns(frame: pd.DataFrame, columns: Iterable[str], source_name: str) -> None:
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        if not np.isclose(values, np.round(values)).all():
            raise ValueError(f"{source_name}字段{column}包含非整数小时值")
        frame[column] = np.round(values).astype("int64")


def _normalise_text(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        frame[column] = frame[column].astype("string").str.strip()


def _validate_membership(values: pd.Series, allowed: Sequence[str], field_name: str) -> None:
    observed = set(values.dropna().astype(str))
    unexpected = sorted(observed.difference(allowed))
    if unexpected:
        raise ValueError(f"字段{field_name}出现题面未定义取值：{unexpected}")


def _validate_complete_region_hour(frame: pd.DataFrame, source_name: str) -> None:
    expected = {(hour, region) for hour in range(TERMINAL_HOUR + 1) for region in REGIONS}
    actual = set(zip(frame["Hour"], frame["Region"]))
    missing = sorted(expected.difference(actual))
    extra = sorted(actual.difference(expected))
    if missing or extra:
        raise ValueError(
            f"{source_name}必须完整覆盖Hour 0--{TERMINAL_HOUR}×六区域；"
            f"缺失示例={missing[:5]}，越界示例={extra[:5]}"
        )


def _validate_tasks(tasks: pd.DataFrame) -> None:
    _require_columns(tasks, TASK_COLUMNS, TASK_SOURCE)
    _require_no_missing(tasks, TASK_COLUMNS, TASK_SOURCE)
    _normalise_text(tasks, ("TaskID", "TaskType", "SourceRegion", "ExecutionMode", "DelaySensitivity"))

    if tasks["TaskID"].duplicated().any():
        duplicated = tasks.loc[tasks["TaskID"].duplicated(keep=False), "TaskID"].head(10).tolist()
        raise ValueError(f"{TASK_SOURCE}的TaskID不唯一，示例：{duplicated}")
    if (tasks["TaskID"] == "").any():
        raise ValueError(f"{TASK_SOURCE}存在空TaskID")

    _validate_membership(tasks["TaskType"], TASK_TYPES, "TaskType")
    _validate_membership(tasks["SourceRegion"], REGIONS, "SourceRegion")
    if set(tasks["ExecutionMode"].astype(str)) != {"NonPreemptive"}:
        modes = sorted(set(tasks["ExecutionMode"].astype(str)))
        raise ValueError(f"{TASK_SOURCE}包含非NonPreemptive执行方式：{modes}")

    numeric_columns = (
        "ArrivalHour",
        "GPU_Demand",
        "EstimatedDuration_min",
        "MaxLatency_ms",
        "EarliestStartHour",
        "LatestFinishHour",
    )
    _coerce_numeric(tasks, numeric_columns, TASK_SOURCE)
    _coerce_integer_columns(
        tasks,
        ("ArrivalHour", "EarliestStartHour", "LatestFinishHour"),
        TASK_SOURCE,
    )

    if not tasks["ArrivalHour"].between(WORKLOAD_START_HOUR, WORKLOAD_END_HOUR).all():
        raise ValueError(f"{TASK_SOURCE}的ArrivalHour必须位于0--2399")
    if not (tasks["GPU_Demand"] > 0).all():
        raise ValueError(f"{TASK_SOURCE}的GPU_Demand必须为正")
    if not (tasks["EstimatedDuration_min"] > 0).all():
        raise ValueError(f"{TASK_SOURCE}的EstimatedDuration_min必须为正")
    if not (tasks["MaxLatency_ms"] >= 0).all():
        raise ValueError(f"{TASK_SOURCE}的MaxLatency_ms不能为负")
    if not (tasks["EarliestStartHour"] >= tasks["ArrivalHour"]).all():
        raise ValueError(f"{TASK_SOURCE}存在EarliestStartHour早于ArrivalHour的任务")
    if not (tasks["LatestFinishHour"] >= tasks["EarliestStartHour"]).all():
        raise ValueError(f"{TASK_SOURCE}存在LatestFinishHour早于EarliestStartHour的任务")
    # 2406是允许出现在原始最晚完成边界中的终端时点，但不是可占用的任务小时。
    if not tasks["LatestFinishHour"].between(WORKLOAD_START_HOUR, TERMINAL_HOUR).all():
        raise ValueError(f"{TASK_SOURCE}的LatestFinishHour超出0--2406边界")


def _validate_gpu_information(gpu_info: pd.DataFrame) -> pd.DataFrame:
    _require_columns(gpu_info, GPU_COLUMNS, GPU_SOURCE)
    _require_no_missing(gpu_info, GPU_COLUMNS, GPU_SOURCE)
    _normalise_text(gpu_info, ("Region",))
    _validate_membership(gpu_info["Region"], REGIONS, "Region")
    if gpu_info["Region"].duplicated().any():
        raise ValueError(f"{GPU_SOURCE}的Region必须唯一")
    if set(gpu_info["Region"]) != set(REGIONS):
        raise ValueError(f"{GPU_SOURCE}必须完整覆盖六个区域：{REGIONS}")

    _coerce_numeric(
        gpu_info,
        ("Total_GPU", "Available_GPU", "Max_IT_Power_MW", "PUE", "Max_Facility_Power_MW"),
        GPU_SOURCE,
    )
    if not (gpu_info["Total_GPU"] >= 0).all():
        raise ValueError(f"{GPU_SOURCE}的Total_GPU不能为负")
    if not (gpu_info["Available_GPU"] >= 0).all():
        raise ValueError(f"{GPU_SOURCE}的Available_GPU不能为负")
    if not (gpu_info["Available_GPU"] <= gpu_info["Total_GPU"]).all():
        raise ValueError(f"{GPU_SOURCE}存在Available_GPU大于Total_GPU的区域")
    if not (gpu_info["Max_IT_Power_MW"] > 0).all():
        raise ValueError(f"{GPU_SOURCE}的Max_IT_Power_MW必须为正")
    if not (gpu_info["PUE"] > 0).all():
        raise ValueError(f"{GPU_SOURCE}的PUE必须为正")
    if not (gpu_info["Max_Facility_Power_MW"] > 0).all():
        raise ValueError(f"{GPU_SOURCE}的Max_Facility_Power_MW必须为正")
    return gpu_info.loc[:, list(GPU_COLUMNS)].copy()


def _validate_power_mapping(power_mapping: pd.DataFrame) -> pd.DataFrame:
    _require_columns(power_mapping, POWER_MAPPING_COLUMNS, POWER_MAPPING_SOURCE)
    _require_no_missing(power_mapping, POWER_MAPPING_COLUMNS, POWER_MAPPING_SOURCE)
    _normalise_text(power_mapping, ("TaskType",))
    _validate_membership(power_mapping["TaskType"], TASK_TYPES, "TaskType")
    if power_mapping["TaskType"].duplicated().any():
        raise ValueError(f"{POWER_MAPPING_SOURCE}的TaskType必须唯一")
    if set(power_mapping["TaskType"]) != set(TASK_TYPES):
        raise ValueError(f"{POWER_MAPPING_SOURCE}必须完整覆盖三类任务：{TASK_TYPES}")
    _coerce_numeric(power_mapping, ("GPU_Power_MW_per_EquivalentGPU",), POWER_MAPPING_SOURCE)
    if not (power_mapping["GPU_Power_MW_per_EquivalentGPU"] > 0).all():
        raise ValueError(f"{POWER_MAPPING_SOURCE}的单位GPU功率必须为正")
    return power_mapping.loc[:, list(POWER_MAPPING_COLUMNS)].copy()


def _validate_latency(network_latency: pd.DataFrame) -> pd.DataFrame:
    _require_columns(network_latency, LATENCY_COLUMNS, LATENCY_SOURCE)
    _require_no_missing(network_latency, LATENCY_COLUMNS, LATENCY_SOURCE)
    text_columns = ["FromRegion", "ToRegion"]
    if "LatencyClass" in network_latency.columns:
        text_columns.append("LatencyClass")
    _normalise_text(network_latency, text_columns)
    _validate_membership(network_latency["FromRegion"], REGIONS, "FromRegion")
    _validate_membership(network_latency["ToRegion"], REGIONS, "ToRegion")
    _coerce_numeric(network_latency, ("NetworkLatency_ms",), LATENCY_SOURCE)
    if not (network_latency["NetworkLatency_ms"] >= 0).all():
        raise ValueError(f"{LATENCY_SOURCE}的NetworkLatency_ms不能为负")
    if network_latency.duplicated(["FromRegion", "ToRegion"]).any():
        raise ValueError(f"{LATENCY_SOURCE}存在重复的区域有序对")

    expected_pairs = {(source, target) for source in REGIONS for target in REGIONS}
    actual_pairs = set(zip(network_latency["FromRegion"], network_latency["ToRegion"]))
    missing_pairs = sorted(expected_pairs.difference(actual_pairs))
    if missing_pairs:
        raise ValueError(f"{LATENCY_SOURCE}缺少区域有序对，示例：{missing_pairs[:10]}")

    output_columns = list(LATENCY_COLUMNS)
    if "LatencyClass" in network_latency.columns:
        output_columns.append("LatencyClass")
    return network_latency.loc[:, output_columns].copy()


def _validate_region_time(region_time: pd.DataFrame) -> pd.DataFrame:
    required_columns = tuple(
        dict.fromkeys((*REGION_TIME_EXOGENOUS_COLUMNS, *REGION_TIME_BASELINE_COLUMNS))
    )
    _require_columns(region_time, required_columns, REGION_TIME_SOURCE)
    _require_no_missing(region_time, required_columns, REGION_TIME_SOURCE)
    text_columns = ["Region"]
    text_columns.extend(column for column in OPTIONAL_REGION_TIME_COLUMNS if column in region_time.columns)
    _normalise_text(region_time, text_columns)
    _validate_membership(region_time["Region"], REGIONS, "Region")

    numeric_columns = [
        column
        for column in (*REGION_TIME_EXOGENOUS_COLUMNS, *REGION_TIME_BASELINE_COLUMNS)
        if column not in {"Hour", "Region"}
    ]
    _coerce_numeric(region_time, numeric_columns, REGION_TIME_SOURCE)
    _coerce_integer_columns(region_time, ("Hour",), REGION_TIME_SOURCE)
    if not region_time["Hour"].between(WORKLOAD_START_HOUR, TERMINAL_HOUR).all():
        raise ValueError(f"{REGION_TIME_SOURCE}的Hour超出0--2406边界")
    if not (region_time["AvailableRenewable_MW"] >= 0).all():
        raise ValueError(f"{REGION_TIME_SOURCE}的AvailableRenewable_MW不能为负")
    if not (region_time["NonAI_IT_Load_MW"] >= 0).all():
        raise ValueError(f"{REGION_TIME_SOURCE}的NonAI_IT_Load_MW不能为负")
    if region_time.duplicated(["Hour", "Region"]).any():
        raise ValueError(f"{REGION_TIME_SOURCE}存在重复的Hour×Region记录")
    _validate_complete_region_hour(region_time, REGION_TIME_SOURCE)
    return region_time


def _validate_storage_params(storage_params: pd.DataFrame) -> pd.DataFrame:
    _require_columns(storage_params, STORAGE_COLUMNS, STORAGE_SOURCE)
    _require_no_missing(storage_params, STORAGE_COLUMNS, STORAGE_SOURCE)
    _normalise_text(storage_params, ("Region",))
    _validate_membership(storage_params["Region"], REGIONS, "Region")
    if storage_params["Region"].duplicated().any():
        raise ValueError(f"{STORAGE_SOURCE}的Region必须唯一")
    if set(storage_params["Region"]) != set(REGIONS):
        raise ValueError(f"{STORAGE_SOURCE}必须完整覆盖六个区域：{REGIONS}")

    numeric_columns = [column for column in STORAGE_COLUMNS if column != "Region"]
    _coerce_numeric(storage_params, numeric_columns, STORAGE_SOURCE)
    if not (storage_params["StorageCapacity_MWh"] >= 0).all():
        raise ValueError(f"{STORAGE_SOURCE}的StorageCapacity_MWh不能为负")
    if not (
        (storage_params["MinSOC_MWh"] >= 0)
        & (storage_params["MinSOC_MWh"] <= storage_params["StorageCapacity_MWh"])
    ).all():
        raise ValueError(f"{STORAGE_SOURCE}存在越界的MinSOC_MWh")
    if not (
        (storage_params["InitialSOC_MWh"] >= 0)
        & (storage_params["InitialSOC_MWh"] <= storage_params["StorageCapacity_MWh"])
    ).all():
        raise ValueError(f"{STORAGE_SOURCE}存在越界的InitialSOC_MWh")
    for column in (
        "MaxChargePower_MW",
        "MaxDischargePower_MW",
        "SellLimit_MW",
        "MaxGridImport_MW",
        "MaxGridExport_MW",
    ):
        if not (storage_params[column] >= 0).all():
            raise ValueError(f"{STORAGE_SOURCE}的{column}不能为负")
    for column in ("ChargeEfficiency", "DischargeEfficiency"):
        if not storage_params[column].between(0, 1, inclusive="right").all():
            raise ValueError(f"{STORAGE_SOURCE}的{column}必须位于(0,1]")
    return storage_params.loc[:, list(STORAGE_COLUMNS)].copy()


def _build_tasks_clean(task_data: pd.DataFrame, power_mapping: pd.DataFrame) -> pd.DataFrame:
    tasks = task_data.loc[:, list(TASK_COLUMNS)].copy()
    _validate_tasks(tasks)
    tasks["Duration_h"] = tasks["EstimatedDuration_min"] / 60.0
    tasks["GPU_Workload_GPUh"] = tasks["GPU_Demand"] * tasks["Duration_h"]
    tasks = tasks.merge(power_mapping, how="left", on="TaskType", validate="many_to_one")
    _require_no_missing(tasks, ("GPU_Power_MW_per_EquivalentGPU",), POWER_MAPPING_SOURCE)
    tasks["Task_Full_IT_Power_MW"] = (
        tasks["GPU_Demand"] * tasks["GPU_Power_MW_per_EquivalentGPU"]
    )
    output_columns = [
        *TASK_COLUMNS,
        "Duration_h",
        "GPU_Workload_GPUh",
        "GPU_Power_MW_per_EquivalentGPU",
        "Task_Full_IT_Power_MW",
    ]
    return tasks.loc[:, output_columns].sort_values("TaskID", kind="stable").reset_index(drop=True)


def _build_task_candidate_regions(
    tasks_clean: pd.DataFrame,
    network_edges: pd.DataFrame,
) -> pd.DataFrame:
    tasks = tasks_clean.loc[:, ["TaskID", "TaskType", "SourceRegion", "MaxLatency_ms"]].copy()
    targets = pd.DataFrame({"TargetRegion": REGIONS})
    tasks["_join_key"] = 1
    targets["_join_key"] = 1
    candidates = tasks.merge(targets, how="inner", on="_join_key").drop(columns="_join_key")
    latency = network_edges.rename(
        columns={"FromRegion": "SourceRegion", "ToRegion": "TargetRegion"}
    )
    candidates = candidates.merge(
        latency,
        how="left",
        on=["SourceRegion", "TargetRegion"],
        validate="many_to_one",
    )
    _require_no_missing(candidates, ("NetworkLatency_ms",), LATENCY_SOURCE)
    candidates = candidates.loc[
        candidates["NetworkLatency_ms"] <= candidates["MaxLatency_ms"]
    ].copy()

    task_ids = set(tasks_clean["TaskID"])
    candidate_task_ids = set(candidates["TaskID"])
    no_candidate = sorted(task_ids.difference(candidate_task_ids))
    if no_candidate:
        raise ValueError(
            "存在没有任何满足任务级时延约束候选区域的任务："
            f"{no_candidate[:10]}"
        )

    return candidates.loc[
        :,
        [
            "TaskID",
            "TaskType",
            "SourceRegion",
            "TargetRegion",
            "NetworkLatency_ms",
            "MaxLatency_ms",
        ],
    ].sort_values(["TaskID", "TargetRegion"], kind="stable").reset_index(drop=True)


def _build_region_hour_exogenous(region_time: pd.DataFrame) -> pd.DataFrame:
    output_columns = list(REGION_TIME_EXOGENOUS_COLUMNS)
    output_columns.extend(
        column for column in OPTIONAL_REGION_TIME_COLUMNS if column in region_time.columns
    )
    return region_time.loc[:, output_columns].sort_values(["Hour", "Region"], kind="stable").reset_index(drop=True)


def _build_baseline_reference(region_time: pd.DataFrame) -> pd.DataFrame:
    return region_time.loc[:, list(REGION_TIME_BASELINE_COLUMNS)].sort_values(
        ["Hour", "Region"], kind="stable"
    ).reset_index(drop=True)


def _build_hourly_demand_panel(tasks_clean: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        tasks_clean.groupby(
            ["ArrivalHour", "SourceRegion", "TaskType"], as_index=False, sort=False
        )
        .agg(
            Task_Count=("TaskID", "size"),
            GPU_Demand_Arrival=("GPU_Demand", "sum"),
            GPU_Workload_Arrival_GPUh=("GPU_Workload_GPUh", "sum"),
        )
        .rename(columns={"ArrivalHour": "Hour"})
    )
    hours = range(WORKLOAD_START_HOUR, WORKLOAD_END_HOUR + 1)
    grid = pd.MultiIndex.from_product(
        [hours, REGIONS, TASK_TYPES], names=["Hour", "SourceRegion", "TaskType"]
    ).to_frame(index=False)
    panel = grid.merge(
        grouped,
        how="left",
        on=["Hour", "SourceRegion", "TaskType"],
        validate="one_to_one",
    )
    panel["Task_Count"] = panel["Task_Count"].fillna(0).astype("int64")
    panel["GPU_Demand_Arrival"] = panel["GPU_Demand_Arrival"].fillna(0.0)
    panel["GPU_Workload_Arrival_GPUh"] = panel["GPU_Workload_Arrival_GPUh"].fillna(0.0)
    panel["HourOfDay"] = panel["Hour"] % 24
    panel["DayIndex"] = panel["Hour"] // 24
    panel["Split"] = panel["Hour"].map(
        lambda hour: (
            "train"
            if hour <= INITIAL_TRAIN_END_HOUR
            else "validation"
            if hour <= VALIDATION_END_HOUR
            else "test"
        )
    )
    expected_rows = 2400 * len(REGIONS) * len(TASK_TYPES)
    if len(panel) != expected_rows:
        raise ValueError(f"hourly_demand_panel行数错误：期望{expected_rows}，实际{len(panel)}")
    return panel.loc[
        :,
        [
            "Hour",
            "SourceRegion",
            "TaskType",
            "Task_Count",
            "GPU_Demand_Arrival",
            "GPU_Workload_Arrival_GPUh",
            "HourOfDay",
            "DayIndex",
            "Split",
        ],
    ]


def _build_region_hour_capacity(
    region_static_compute: pd.DataFrame,
    region_hour_exogenous: pd.DataFrame,
) -> pd.DataFrame:
    hour_region_grid = pd.MultiIndex.from_product(
        [range(TEST_START_HOUR, TAIL_END_HOUR + 1), REGIONS], names=["Hour", "Region"]
    ).to_frame(index=False)
    selected_exogenous = region_hour_exogenous.loc[
        region_hour_exogenous["Hour"].between(TEST_START_HOUR, TAIL_END_HOUR),
        ["Hour", "Region", "NonAI_IT_Load_MW"],
    ]
    capacity = hour_region_grid.merge(
        region_static_compute,
        how="left",
        on="Region",
        validate="many_to_one",
    ).merge(
        selected_exogenous,
        how="left",
        on=["Hour", "Region"],
        validate="one_to_one",
    )
    _require_no_missing(
        capacity,
        (
            "Available_GPU",
            "Max_IT_Power_MW",
            "PUE",
            "Max_Facility_Power_MW",
            "NonAI_IT_Load_MW",
        ),
        "region_hour_capacity",
    )
    capacity["AI_IT_Margin_MW"] = (
        capacity["Max_IT_Power_MW"] - capacity["NonAI_IT_Load_MW"]
    )
    capacity["AI_Facility_Margin_MW"] = (
        capacity["Max_Facility_Power_MW"] / capacity["PUE"]
        - capacity["NonAI_IT_Load_MW"]
    )
    capacity["Effective_AI_IT_Capacity_MW"] = np.maximum(
        0.0,
        np.minimum(capacity["AI_IT_Margin_MW"], capacity["AI_Facility_Margin_MW"]),
    )
    output_columns = [
        "Hour",
        "Region",
        "Available_GPU",
        "NonAI_IT_Load_MW",
        "Max_IT_Power_MW",
        "PUE",
        "Max_Facility_Power_MW",
        "AI_IT_Margin_MW",
        "AI_Facility_Margin_MW",
        "Effective_AI_IT_Capacity_MW",
    ]
    return capacity.loc[:, output_columns].sort_values(
        ["Hour", "Region"], kind="stable"
    ).reset_index(drop=True)


def _write_csv(relative_path: str, frame: pd.DataFrame) -> None:
    root_name, filename = relative_path.split("/", 1)
    output_root = {"shared": SHARED_DIR, "q1": Q1_DIR}.get(root_name)
    if output_root is None:
        raise ValueError(f"不允许写入未声明的输出层：{relative_path}")
    output_path = output_root / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")


def build_processed_tables() -> dict[str, pd.DataFrame]:
    """读取六份原始附件，构造共享层和Q1专属接口。"""

    task_data = _read_table(RAW_DIR / TASK_SOURCE, TASK_COLUMNS)
    gpu_info = _validate_gpu_information(_read_table(RAW_DIR / GPU_SOURCE, GPU_COLUMNS))
    network_edges = _validate_latency(_read_table(RAW_DIR / LATENCY_SOURCE, LATENCY_COLUMNS))
    power_mapping = _validate_power_mapping(
        _read_table(RAW_DIR / POWER_MAPPING_SOURCE, POWER_MAPPING_COLUMNS)
    )
    region_time = _validate_region_time(
        _read_table(
            RAW_DIR / REGION_TIME_SOURCE,
            tuple(dict.fromkeys((*REGION_TIME_EXOGENOUS_COLUMNS, *REGION_TIME_BASELINE_COLUMNS))),
        )
    )
    storage_params = _validate_storage_params(
        _read_table(RAW_DIR / STORAGE_SOURCE, STORAGE_COLUMNS)
    )

    tasks_clean = _build_tasks_clean(task_data, power_mapping)
    region_static_compute = gpu_info
    task_candidate_regions = _build_task_candidate_regions(tasks_clean, network_edges)
    region_hour_exogenous = _build_region_hour_exogenous(region_time)
    baseline_reference = _build_baseline_reference(region_time)
    hourly_demand_panel = _build_hourly_demand_panel(tasks_clean)
    region_hour_capacity = _build_region_hour_capacity(
        region_static_compute, region_hour_exogenous
    )

    return {
        "shared/tasks_clean.csv": tasks_clean,
        "shared/region_static_compute.csv": region_static_compute,
        "shared/network_edges.csv": network_edges,
        "shared/task_candidate_regions.csv": task_candidate_regions,
        "shared/region_hour_exogenous.csv": region_hour_exogenous,
        "shared/baseline_reference_region_hour.csv": baseline_reference,
        "shared/storage_params.csv": storage_params,
        "q1/hourly_demand_panel.csv": hourly_demand_panel,
        "q1/region_hour_capacity.csv": region_hour_capacity,
    }


def main() -> int:
    tables = build_processed_tables()
    for relative_path, table in tables.items():
        _write_csv(relative_path, table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
