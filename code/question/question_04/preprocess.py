"""问题4预处理：读取问题1共享层并形成联合模型的逐时区域接口。

任务、候选区域、储能参数和基准参考均保持在第一问的shared目录中；本题不复制
它们，也不提前生成场景、服务质量或多目标结果表。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


QUESTION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = QUESTION_DIR.parents[1]
SHARED_DIR = (
    PROJECT_DIR / "question" / "question_01" / "data" / "processed" / "shared"
)
PROCESSED_DIR = QUESTION_DIR / "data" / "processed"
Q4_DIR = PROCESSED_DIR / "q4"

EXOGENOUS_COLUMNS = (
    "Hour",
    "Region",
    "ElectricityPrice_CNY_per_MWh",
    "SellPrice_CNY_per_MWh",
    "CarbonIntensity_tCO2_per_MWh",
    "AvailableRenewable_MW",
    "NonAI_IT_Load_MW",
)
STATIC_COLUMNS = (
    "Region",
    "Available_GPU",
    "Max_IT_Power_MW",
    "PUE",
    "Max_Facility_Power_MW",
)
STORAGE_COLUMNS = (
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


def _read_shared(filename: str, required_columns: tuple[str, ...]) -> pd.DataFrame:
    path = SHARED_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少问题1共享输入：{path}；请先运行 question_01/preprocess.py"
        )
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"共享文件{filename}缺少字段：{missing}")
    return frame


def _validate_hour_region(frame: pd.DataFrame, source_name: str) -> None:
    if frame[["Hour", "Region"]].isna().any().any():
        raise ValueError(f"{source_name}的Hour或Region存在缺失值")
    frame["Hour"] = pd.to_numeric(frame["Hour"], errors="coerce")
    if frame["Hour"].isna().any() or not frame["Hour"].eq(frame["Hour"].round()).all():
        raise ValueError(f"{source_name}的Hour不是有效整数")
    frame["Hour"] = frame["Hour"].astype("int64")
    if not frame["Hour"].between(0, 2406).all():
        raise ValueError(f"{source_name}的Hour必须位于0--2406")
    if frame.duplicated(["Hour", "Region"]).any():
        raise ValueError(f"{source_name}存在重复的Hour×Region记录")


def _time_role(hour: int) -> str:
    if hour <= 2399:
        return "main"
    if hour <= 2405:
        return "tail"
    return "terminal"


def _build_region_hour_input(
    region_hour_exogenous: pd.DataFrame,
    region_static_compute: pd.DataFrame,
) -> pd.DataFrame:
    exogenous = region_hour_exogenous.loc[:, list(EXOGENOUS_COLUMNS)].copy()
    static = region_static_compute.loc[:, list(STATIC_COLUMNS)].copy()
    _validate_hour_region(exogenous, "region_hour_exogenous")
    if static["Region"].duplicated().any():
        raise ValueError("region_static_compute的Region必须唯一")
    result = exogenous.merge(static, how="left", on="Region", validate="many_to_one")
    if result[list(STATIC_COLUMNS[1:])].isna().any().any():
        raise ValueError("region_static_compute无法覆盖所有逐时区域记录")
    result["TimeRole"] = result["Hour"].map(_time_role)
    output_columns = [
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
    ]
    return result.loc[:, output_columns].sort_values(
        ["Hour", "Region"], kind="stable"
    ).reset_index(drop=True)


def _check_shared_objects() -> None:
    tasks = _read_shared("tasks_clean.csv", ("TaskID", "TaskType", "ArrivalHour"))
    candidates = _read_shared(
        "task_candidate_regions.csv",
        ("TaskID", "TaskType", "SourceRegion", "TargetRegion", "NetworkLatency_ms"),
    )
    storage = _read_shared(
        "storage_params.csv",
        STORAGE_COLUMNS,
    )
    _read_shared("baseline_reference_region_hour.csv", ("Hour", "Region"))
    if tasks["TaskID"].duplicated().any():
        raise ValueError("共享tasks_clean的TaskID必须唯一")
    if not candidates["TaskID"].isin(tasks["TaskID"]).all():
        raise ValueError("共享task_candidate_regions包含不存在的TaskID")
    if storage["Region"].duplicated().any():
        raise ValueError("共享storage_params的Region必须唯一")


def main() -> int:
    _check_shared_objects()
    region_hour_exogenous = _read_shared(
        "region_hour_exogenous.csv", EXOGENOUS_COLUMNS
    )
    region_static_compute = _read_shared(
        "region_static_compute.csv", STATIC_COLUMNS
    )
    q4_region_hour_input = _build_region_hour_input(
        region_hour_exogenous, region_static_compute
    )
    Q4_DIR.mkdir(parents=True, exist_ok=True)
    q4_region_hour_input.to_csv(
        Q4_DIR / "q4_region_hour_input.csv", index=False, encoding="utf-8-sig"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
