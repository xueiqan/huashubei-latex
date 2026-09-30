"""问题3预处理：固定负荷接口与共享储能参数入口。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


QUESTION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = QUESTION_DIR.parents[1]
SHARED_DIR = (
    PROJECT_DIR / "question" / "question_01" / "data" / "processed" / "shared"
)
PROCESSED_DIR = QUESTION_DIR / "data" / "processed"
Q3_DIR = PROCESSED_DIR / "q3"

EXOGENOUS_COLUMNS = (
    "Hour",
    "Region",
    "ElectricityPrice_CNY_per_MWh",
    "SellPrice_CNY_per_MWh",
    "CarbonIntensity_tCO2_per_MWh",
    "AvailableRenewable_MW",
    "NonAI_IT_Load_MW",
)
BASELINE_COLUMNS = ("Hour", "Region", "Baseline_AI_IT_Load_MW")
STATIC_COLUMNS = ("Region", "PUE")
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


def main() -> int:
    region_hour_exogenous = _read_shared(
        "region_hour_exogenous.csv", EXOGENOUS_COLUMNS
    )
    baseline_reference = _read_shared(
        "baseline_reference_region_hour.csv", BASELINE_COLUMNS
    )
    region_static_compute = _read_shared(
        "region_static_compute.csv", STATIC_COLUMNS
    )
    storage_params = _read_shared("storage_params.csv", STORAGE_COLUMNS)

    exogenous = region_hour_exogenous.loc[:, list(EXOGENOUS_COLUMNS)].copy()
    baseline = baseline_reference.loc[:, list(BASELINE_COLUMNS)].copy()
    static = region_static_compute.loc[:, list(STATIC_COLUMNS)].copy()
    _validate_hour_region(exogenous, "region_hour_exogenous")
    _validate_hour_region(baseline, "baseline_reference_region_hour")
    if static["Region"].duplicated().any() or storage_params["Region"].duplicated().any():
        raise ValueError("共享静态参数的Region必须唯一")
    if set(storage_params["Region"]) != set(static["Region"]):
        raise ValueError("storage_params与region_static_compute的区域集合不一致")

    result = exogenous.merge(baseline, how="left", on=["Hour", "Region"], validate="one_to_one")
    result = result.merge(static, how="left", on="Region", validate="many_to_one")
    if result[["Baseline_AI_IT_Load_MW", "PUE"]].isna().any().any():
        raise ValueError("Q3共享输入无法覆盖完整的Hour×Region记录")

    result["TimeRole"] = result["Hour"].map(_time_role)
    result["Fixed_IT_Load_MW"] = (
        result["Baseline_AI_IT_Load_MW"] + result["NonAI_IT_Load_MW"]
    )
    result["Fixed_Facility_Load_MW"] = result["Fixed_IT_Load_MW"] * result["PUE"]
    output_columns = [
        "Hour",
        "Region",
        "TimeRole",
        "Baseline_AI_IT_Load_MW",
        "NonAI_IT_Load_MW",
        "Fixed_IT_Load_MW",
        "PUE",
        "Fixed_Facility_Load_MW",
        "ElectricityPrice_CNY_per_MWh",
        "SellPrice_CNY_per_MWh",
        "CarbonIntensity_tCO2_per_MWh",
        "AvailableRenewable_MW",
    ]
    result = result.loc[:, output_columns].sort_values(
        ["Hour", "Region"], kind="stable"
    ).reset_index(drop=True)

    Q3_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        Q3_DIR / "q3_fixed_energy_input.csv", index=False, encoding="utf-8-sig"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
