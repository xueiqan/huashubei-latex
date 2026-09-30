"""Preprocess Question 1 and shared Problem C inputs.

Generate only two output layers:

* ``data/processed/shared``: deterministic data shared by Q1--Q4.
* ``data/processed/q1``: the Q1 demand panel and compute-capacity interface.

Preprocessing does not assign task regions or start/finish times, create hourly
operation matrices, or optimize energy. Hour 2406 is a terminal settlement point
only, with no task execution records.
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
        raise ValueError(f"Duplicate worksheet column names: {duplicated}")
    return renamed


def _read_table(path: Path, required_columns: Sequence[str]) -> pd.DataFrame:
    """Locate the main data sheet in a workbook that also contains column documentation."""

    if not path.is_file():
        raise FileNotFoundError(f"Input attachment not found: {path}")

    try:
        workbook = pd.ExcelFile(path)
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot read {path.name}: the Python environment lacks an Excel reader; "
            "Provide a pandas-compatible xlsx reader in the existing competition environment."
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
        f"{path.name} has no main data sheet containing columns {sorted(required)}; "
        f"Worksheets checked: {sheets}"
    )


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], source_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")


def _require_no_missing(frame: pd.DataFrame, columns: Iterable[str], source_name: str) -> None:
    selected = list(columns)
    missing_counts = frame[selected].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        details = ", ".join(f"{column}={int(count)}" for column, count in missing_counts.items())
        raise ValueError(f"{source_name} contains missing values in required columns: {details}")


def _coerce_numeric(frame: pd.DataFrame, columns: Iterable[str], source_name: str) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{source_name} column {column} contains nonfinite or missing values")


def _coerce_integer_columns(frame: pd.DataFrame, columns: Iterable[str], source_name: str) -> None:
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        if not np.isclose(values, np.round(values)).all():
            raise ValueError(f"{source_name} column {column} contains noninteger hour values")
        frame[column] = np.round(values).astype("int64")


def _normalise_text(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        frame[column] = frame[column].astype("string").str.strip()


def _validate_membership(values: pd.Series, allowed: Sequence[str], field_name: str) -> None:
    observed = set(values.dropna().astype(str))
    unexpected = sorted(observed.difference(allowed))
    if unexpected:
        raise ValueError(f"Column {field_name} contains values undefined by the problem: {unexpected}")


def _validate_complete_region_hour(frame: pd.DataFrame, source_name: str) -> None:
    expected = {(hour, region) for hour in range(TERMINAL_HOUR + 1) for region in REGIONS}
    actual = set(zip(frame["Hour"], frame["Region"]))
    missing = sorted(expected.difference(actual))
    extra = sorted(actual.difference(expected))
    if missing or extra:
        raise ValueError(
            f"{source_name} must cover hours 0--{TERMINAL_HOUR} in all six regions; "
            f"missing examples={missing[:5]}; out-of-range examples={extra[:5]}"
        )


def _validate_tasks(tasks: pd.DataFrame) -> None:
    _require_columns(tasks, TASK_COLUMNS, TASK_SOURCE)
    _require_no_missing(tasks, TASK_COLUMNS, TASK_SOURCE)
    _normalise_text(tasks, ("TaskID", "TaskType", "SourceRegion", "ExecutionMode", "DelaySensitivity"))

    if tasks["TaskID"].duplicated().any():
        duplicated = tasks.loc[tasks["TaskID"].duplicated(keep=False), "TaskID"].head(10).tolist()
        raise ValueError(f"{TASK_SOURCE} contains duplicate TaskID values; examples: {duplicated}")
    if (tasks["TaskID"] == "").any():
        raise ValueError(f"{TASK_SOURCE} contains empty TaskID values")

    _validate_membership(tasks["TaskType"], TASK_TYPES, "TaskType")
    _validate_membership(tasks["SourceRegion"], REGIONS, "SourceRegion")
    if set(tasks["ExecutionMode"].astype(str)) != {"NonPreemptive"}:
        modes = sorted(set(tasks["ExecutionMode"].astype(str)))
        raise ValueError(f"{TASK_SOURCE} contains execution modes other than NonPreemptive: {modes}")

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
        raise ValueError(f"{TASK_SOURCE} ArrivalHour must be within 0--2399")
    if not (tasks["GPU_Demand"] > 0).all():
        raise ValueError(f"{TASK_SOURCE} GPU_Demand must be positive")
    if not (tasks["EstimatedDuration_min"] > 0).all():
        raise ValueError(f"{TASK_SOURCE} EstimatedDuration_min must be positive")
    if not (tasks["MaxLatency_ms"] >= 0).all():
        raise ValueError(f"{TASK_SOURCE} MaxLatency_ms must be nonnegative")
    if not (tasks["EarliestStartHour"] >= tasks["ArrivalHour"]).all():
        raise ValueError(f"{TASK_SOURCE} contains tasks with EarliestStartHour before ArrivalHour")
    if not (tasks["LatestFinishHour"] >= tasks["EarliestStartHour"]).all():
        raise ValueError(f"{TASK_SOURCE} contains tasks with LatestFinishHour before EarliestStartHour")
    # Hour 2406 is permitted as a raw latest-finish boundary but cannot be occupied by tasks.
    if not tasks["LatestFinishHour"].between(WORKLOAD_START_HOUR, TERMINAL_HOUR).all():
        raise ValueError(f"{TASK_SOURCE} LatestFinishHour exceeds the 0--2406 boundary")


def _validate_gpu_information(gpu_info: pd.DataFrame) -> pd.DataFrame:
    _require_columns(gpu_info, GPU_COLUMNS, GPU_SOURCE)
    _require_no_missing(gpu_info, GPU_COLUMNS, GPU_SOURCE)
    _normalise_text(gpu_info, ("Region",))
    _validate_membership(gpu_info["Region"], REGIONS, "Region")
    if gpu_info["Region"].duplicated().any():
        raise ValueError(f"{GPU_SOURCE} Region values must be unique")
    if set(gpu_info["Region"]) != set(REGIONS):
        raise ValueError(f"{GPU_SOURCE} must cover all six regions: {REGIONS}")

    _coerce_numeric(
        gpu_info,
        ("Total_GPU", "Available_GPU", "Max_IT_Power_MW", "PUE", "Max_Facility_Power_MW"),
        GPU_SOURCE,
    )
    if not (gpu_info["Total_GPU"] >= 0).all():
        raise ValueError(f"{GPU_SOURCE} Total_GPU must be nonnegative")
    if not (gpu_info["Available_GPU"] >= 0).all():
        raise ValueError(f"{GPU_SOURCE} Available_GPU must be nonnegative")
    if not (gpu_info["Available_GPU"] <= gpu_info["Total_GPU"]).all():
        raise ValueError(f"{GPU_SOURCE} contains regions where Available_GPU exceeds Total_GPU")
    if not (gpu_info["Max_IT_Power_MW"] > 0).all():
        raise ValueError(f"{GPU_SOURCE} Max_IT_Power_MW must be positive")
    if not (gpu_info["PUE"] > 0).all():
        raise ValueError(f"{GPU_SOURCE} PUE must be positive")
    if not (gpu_info["Max_Facility_Power_MW"] > 0).all():
        raise ValueError(f"{GPU_SOURCE} Max_Facility_Power_MW must be positive")
    return gpu_info.loc[:, list(GPU_COLUMNS)].copy()


def _validate_power_mapping(power_mapping: pd.DataFrame) -> pd.DataFrame:
    _require_columns(power_mapping, POWER_MAPPING_COLUMNS, POWER_MAPPING_SOURCE)
    _require_no_missing(power_mapping, POWER_MAPPING_COLUMNS, POWER_MAPPING_SOURCE)
    _normalise_text(power_mapping, ("TaskType",))
    _validate_membership(power_mapping["TaskType"], TASK_TYPES, "TaskType")
    if power_mapping["TaskType"].duplicated().any():
        raise ValueError(f"{POWER_MAPPING_SOURCE} TaskType values must be unique")
    if set(power_mapping["TaskType"]) != set(TASK_TYPES):
        raise ValueError(f"{POWER_MAPPING_SOURCE} must cover all three task types: {TASK_TYPES}")
    _coerce_numeric(power_mapping, ("GPU_Power_MW_per_EquivalentGPU",), POWER_MAPPING_SOURCE)
    if not (power_mapping["GPU_Power_MW_per_EquivalentGPU"] > 0).all():
        raise ValueError(f"{POWER_MAPPING_SOURCE} power per GPU must be positive")
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
        raise ValueError(f"{LATENCY_SOURCE} NetworkLatency_ms must be nonnegative")
    if network_latency.duplicated(["FromRegion", "ToRegion"]).any():
        raise ValueError(f"{LATENCY_SOURCE} contains duplicate ordered region pairs")

    expected_pairs = {(source, target) for source in REGIONS for target in REGIONS}
    actual_pairs = set(zip(network_latency["FromRegion"], network_latency["ToRegion"]))
    missing_pairs = sorted(expected_pairs.difference(actual_pairs))
    if missing_pairs:
        raise ValueError(f"{LATENCY_SOURCE} is missing ordered region pairs; examples: {missing_pairs[:10]}")

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
        raise ValueError(f"{REGION_TIME_SOURCE} Hour exceeds the 0--2406 boundary")
    if not (region_time["AvailableRenewable_MW"] >= 0).all():
        raise ValueError(f"{REGION_TIME_SOURCE} AvailableRenewable_MW must be nonnegative")
    if not (region_time["NonAI_IT_Load_MW"] >= 0).all():
        raise ValueError(f"{REGION_TIME_SOURCE} NonAI_IT_Load_MW must be nonnegative")
    if region_time.duplicated(["Hour", "Region"]).any():
        raise ValueError(f"{REGION_TIME_SOURCE} contains duplicate Hour x Region records")
    _validate_complete_region_hour(region_time, REGION_TIME_SOURCE)
    return region_time


def _validate_storage_params(storage_params: pd.DataFrame) -> pd.DataFrame:
    _require_columns(storage_params, STORAGE_COLUMNS, STORAGE_SOURCE)
    _require_no_missing(storage_params, STORAGE_COLUMNS, STORAGE_SOURCE)
    _normalise_text(storage_params, ("Region",))
    _validate_membership(storage_params["Region"], REGIONS, "Region")
    if storage_params["Region"].duplicated().any():
        raise ValueError(f"{STORAGE_SOURCE} Region values must be unique")
    if set(storage_params["Region"]) != set(REGIONS):
        raise ValueError(f"{STORAGE_SOURCE} must cover all six regions: {REGIONS}")

    numeric_columns = [column for column in STORAGE_COLUMNS if column != "Region"]
    _coerce_numeric(storage_params, numeric_columns, STORAGE_SOURCE)
    if not (storage_params["StorageCapacity_MWh"] >= 0).all():
        raise ValueError(f"{STORAGE_SOURCE} StorageCapacity_MWh must be nonnegative")
    if not (
        (storage_params["MinSOC_MWh"] >= 0)
        & (storage_params["MinSOC_MWh"] <= storage_params["StorageCapacity_MWh"])
    ).all():
        raise ValueError(f"{STORAGE_SOURCE} contains out-of-bounds MinSOC_MWh")
    if not (
        (storage_params["InitialSOC_MWh"] >= 0)
        & (storage_params["InitialSOC_MWh"] <= storage_params["StorageCapacity_MWh"])
    ).all():
        raise ValueError(f"{STORAGE_SOURCE} contains out-of-bounds InitialSOC_MWh")
    for column in (
        "MaxChargePower_MW",
        "MaxDischargePower_MW",
        "SellLimit_MW",
        "MaxGridImport_MW",
        "MaxGridExport_MW",
    ):
        if not (storage_params[column] >= 0).all():
            raise ValueError(f"{STORAGE_SOURCE} {column} must be nonnegative")
    for column in ("ChargeEfficiency", "DischargeEfficiency"):
        if not storage_params[column].between(0, 1, inclusive="right").all():
            raise ValueError(f"{STORAGE_SOURCE} {column} must be within (0,1]")
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
            "Tasks have no candidate region satisfying task-level latency constraints: "
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
        raise ValueError(f"Invalid hourly_demand_panel row count: expected {expected_rows}; actual {len(panel)}")
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
        raise ValueError(f"Writing to an undeclared output layer is forbidden: {relative_path}")
    output_path = output_root / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")


def build_processed_tables() -> dict[str, pd.DataFrame]:
    """Read six raw attachments and construct shared data and Q1-specific interfaces."""

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
