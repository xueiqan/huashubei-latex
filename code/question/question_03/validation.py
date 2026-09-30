"""Independent Question 3 validation: constraint closure, attachment-baseline reproduction, and multiobjective checks.

Do not resolve MILP or rely solely on model-provided constraint audits. Recompute
from ``q3_balanced_region_hour.csv``, attachment baseline states, and Q3 inputs:

1. Renewable balance, load/storage energy balance, SOC recurrence, terminal SOC,
   and all hard boundaries.
2. Cost, Carbon, Peak, Ramp, and Throughput for Balanced, BaselineReference,
   and NoStorage solutions.
3. Reproduction errors between hourly attachment states and unified model output.
4. Optimality, lexicographic ties, and dominance of four anchors and three balanced stages.

Write validation results to this question-specific ``outputs/tables``. Use only
existing dependencies; add no third-party packages.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


QUESTION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = QUESTION_DIR.parents[1]
PROCESSED_Q3_DIR = QUESTION_DIR / "data" / "processed" / "q3"
SHARED_DIR = PROJECT_DIR / "question" / "question_01" / "data" / "processed" / "shared"
TABLES_DIR = QUESTION_DIR / "outputs" / "tables"
LOGS_DIR = QUESTION_DIR / "outputs" / "logs"
VALIDATION_LOG_PATH = LOGS_DIR / "question_03_validation.log"

FLOAT_EPS = 1e-8
AUDIT_TOLERANCE = 1e-5
BALANCE_TOLERANCE = 1e-6
BALANCE_COMPARISON_EPS = 1e-10
BASELINE_RECONCILIATION_TOLERANCE = 2e-4
METRIC_ABS_TOLERANCE = 1e-5
METRIC_REL_TOLERANCE = 1e-8
OBJECTIVE_NAMES = ("Cost", "Carbon", "Peak", "Ramp")
SCHEMES = ("Balanced", "BaselineReference", "NoStorage")

Q3_INPUT_COLUMNS = (
    "Hour",
    "Region",
    "Fixed_Facility_Load_MW",
    "AvailableRenewable_MW",
    "ElectricityPrice_CNY_per_MWh",
    "SellPrice_CNY_per_MWh",
    "CarbonIntensity_tCO2_per_MWh",
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
BASELINE_COLUMNS = (
    "Hour",
    "Region",
    "UsedRenewable_MW",
    "RenewableCharge_MW",
    "Curtailment_MW",
    "GridPurchase_MW",
    "GridCharge_MW",
    "GridSell_MW",
    "NetGridImport_MW",
    "SOC_MWh",
    "ChargePower_MW",
    "DischargePower_MW",
)
PROFILE_COLUMNS = (
    "Scheme",
    "Hour",
    "Region",
    "TerminalStateOnly",
    "RenewableDirectUse_MW",
    "RenewableCharge_MW",
    "GridCharge_MW",
    "ChargePower_MW",
    "DischargePower_MW",
    "GridPurchase_MW",
    "RenewableExport_MW",
    "RenewableCurtailment_MW",
    "NetGridImport_MW",
    "SOC_MWh",
)


def _configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Unsupported log level: {level_name}")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(VALIDATION_LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logging.basicConfig(
        level=level,
        handlers=[stream_handler, file_handler],
        force=True,
    )


def _read_csv(path: Path, required_columns: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Validation input file is missing: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    return frame


def _to_numeric(frame: pd.DataFrame, columns: Iterable[str], source_name: str) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        values = result[column].to_numpy(dtype=float)
        if result[column].isna().any() or not np.isfinite(values).all():
            raise ValueError(f"{source_name} column {column} contains invalid numeric values")
    return result


def _validate_keys(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    result = frame.copy()
    if result[["Hour", "Region"]].isna().any().any():
        raise ValueError(f"{source_name} contains missing Hour or Region values")
    result["Hour"] = pd.to_numeric(result["Hour"], errors="coerce")
    if result["Hour"].isna().any() or not result["Hour"].eq(result["Hour"].round()).all():
        raise ValueError(f"{source_name} Hour values are not valid integers")
    result["Hour"] = result["Hour"].astype(int)
    result["Region"] = result["Region"].astype(str)
    if result.duplicated(["Hour", "Region"]).any():
        raise ValueError(f"{source_name} contains duplicate Hour x Region records")
    return result


def _complete_grid(frame: pd.DataFrame, keys: set[tuple[int, str]], source_name: str) -> None:
    actual = set(zip(frame["Hour"].astype(int), frame["Region"].astype(str)))
    missing = keys - actual
    extra = actual - keys
    if missing or extra:
        raise ValueError(
            f"{source_name} Hour x Region grid differs; missing examples={sorted(missing)[:5]}; extra examples={sorted(extra)[:5]}"
        )


def _metric_tolerance(value: float) -> float:
    return METRIC_ABS_TOLERANCE + METRIC_REL_TOLERANCE * max(1.0, abs(float(value)))


def _violation_status(value: float, tolerance: float = AUDIT_TOLERANCE) -> str:
    return "PASS" if float(value) <= tolerance else "FAIL"


def _profile_violation_status(scheme: str, value: float) -> str:
    status = _violation_status(value)
    if scheme == "BaselineReference" and status == "FAIL":
        return "WARN"
    return status


def _profile_terminal_flag(frame: pd.DataFrame) -> pd.Series:
    return frame["TerminalStateOnly"].astype(str).str.strip().str.lower().isin(
        ("true", "1", "yes")
    )


def _load_context() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    energy = _read_csv(PROCESSED_Q3_DIR / "q3_fixed_energy_input.csv", Q3_INPUT_COLUMNS)
    energy = _validate_keys(energy, "q3_fixed_energy_input.csv")
    energy = _to_numeric(
        energy,
        [column for column in Q3_INPUT_COLUMNS if column not in ("Region",)],
        "q3_fixed_energy_input.csv",
    )
    regions = tuple(sorted(energy["Region"].unique().tolist()))

    storage = _read_csv(SHARED_DIR / "storage_params.csv", STORAGE_COLUMNS)
    storage["Region"] = storage["Region"].astype(str)
    storage = _to_numeric(storage, STORAGE_COLUMNS[1:], "storage_params.csv")
    if storage["Region"].duplicated().any() or set(storage["Region"]) != set(regions):
        raise ValueError("storage_params.csv region set differs from Q3 inputs")
    storage = storage.sort_values("Region", kind="stable").reset_index(drop=True)

    baseline = _read_csv(SHARED_DIR / "baseline_reference_region_hour.csv", BASELINE_COLUMNS)
    baseline = _validate_keys(baseline, "baseline_reference_region_hour.csv")
    baseline = _to_numeric(
        baseline,
        [column for column in BASELINE_COLUMNS if column != "Region"],
        "baseline_reference_region_hour.csv",
    )
    return energy, storage, baseline, regions


def _load_profile(scheme: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    filename = {
        "Balanced": "q3_balanced_region_hour.csv",
        "BaselineReference": "q3_baseline_region_hour.csv",
        "NoStorage": "q3_no_storage_region_hour.csv",
    }[scheme]
    profile = _read_csv(TABLES_DIR / filename, PROFILE_COLUMNS)
    profile = _validate_keys(profile, filename)
    numeric_columns = [
        column
        for column in PROFILE_COLUMNS
        if column not in ("Scheme", "Region", "TerminalStateOnly")
    ]
    profile = _to_numeric(profile, numeric_columns, filename)
    terminal_flag = _profile_terminal_flag(profile)
    operating = profile.loc[~terminal_flag].copy()
    terminal = profile.loc[terminal_flag].copy()
    if operating.empty or terminal.empty:
        raise ValueError(f"{filename} must include both operating intervals and terminal-state rows")
    if operating["Scheme"].astype(str).nunique() != 1 or operating["Scheme"].iloc[0] != scheme:
        raise ValueError(f"{filename} Scheme field disagrees with its file purpose")
    if terminal["Scheme"].astype(str).nunique() != 1 or terminal["Scheme"].iloc[0] != scheme:
        raise ValueError(f"{filename} terminal Scheme field is inconsistent")
    operating = operating.sort_values(["Hour", "Region"], kind="stable").reset_index(drop=True)
    terminal = terminal.sort_values(["Hour", "Region"], kind="stable").reset_index(drop=True)
    if terminal["Hour"].nunique() != 1 or terminal["Hour"].iloc[0] != int(operating["Hour"].max()) + 1:
        raise ValueError(f"{filename} terminal hour is not the last operating hour plus one")
    return operating, terminal


def _merge_energy(profile: pd.DataFrame, energy: pd.DataFrame) -> pd.DataFrame:
    columns = list(Q3_INPUT_COLUMNS)
    result = profile.merge(
        energy.loc[:, columns],
        how="left",
        on=["Hour", "Region"],
        validate="one_to_one",
        suffixes=("", "_input"),
    )
    if result[["Fixed_Facility_Load_MW", "AvailableRenewable_MW"]].isna().any().any():
        raise ValueError("Hourly model results are not fully covered by Q3 inputs")
    return result


def _check_profile(
    scheme: str,
    operating: pd.DataFrame,
    terminal: pd.DataFrame,
    energy: pd.DataFrame,
    storage: pd.DataFrame,
) -> pd.DataFrame:
    profile = _merge_energy(operating, energy)
    regions = tuple(sorted(profile["Region"].astype(str).unique().tolist()))
    region_params = storage.set_index("Region").loc[list(regions)]
    profile = profile.sort_values(["Hour", "Region"], kind="stable").reset_index(drop=True)
    region_count = len(regions)
    time_count = profile["Hour"].nunique()
    if len(profile) != region_count * time_count:
        raise ValueError(f"{scheme} hourly results do not form a complete Hour x Region grid")

    def values(column: str) -> np.ndarray:
        return profile[column].to_numpy(dtype=float).reshape(time_count, region_count)

    direct = values("RenewableDirectUse_MW")
    renewable_charge = values("RenewableCharge_MW")
    grid_charge = values("GridCharge_MW")
    discharge = values("DischargePower_MW")
    grid_purchase = values("GridPurchase_MW")
    renewable_export = values("RenewableExport_MW")
    curtailment = values("RenewableCurtailment_MW")
    net = values("NetGridImport_MW")
    soc_end = values("SOC_MWh")
    fixed_load = values("Fixed_Facility_Load_MW")
    renewable = values("AvailableRenewable_MW")
    params = {
        column: region_params[column].to_numpy(dtype=float)
        for column in STORAGE_COLUMNS[1:]
    }
    soc_start = np.vstack([params["InitialSOC_MWh"], soc_end[:-1]])
    terminal_soc = terminal.sort_values("Region", kind="stable")["SOC_MWh"].to_numpy(dtype=float)
    terminal_flag = _profile_terminal_flag(terminal)

    residuals = {
        "RenewableBalanceResidual_MW": np.max(
            np.abs(renewable - direct - renewable_charge - renewable_export - curtailment)
        ),
        "EnergyBalanceResidual_MW": np.max(
            np.abs(grid_purchase + direct + discharge - fixed_load - grid_charge)
        ),
        "SOCRecurrenceResidual_MWh": np.max(
            np.abs(
                soc_end
                - soc_start
                - params["ChargeEfficiency"] * (renewable_charge + grid_charge)
                + discharge / params["DischargeEfficiency"]
            )
        ),
        "InitialSOCViolation_MWh": np.max(
            np.abs(soc_start[0] - params["InitialSOC_MWh"])
        ),
        "TerminalSOCDeficit_MWh": np.maximum(
            params["InitialSOC_MWh"] - terminal_soc, 0.0
        ).max(),
        "SOCLowerViolation_MWh": np.maximum(
            params["MinSOC_MWh"] - np.vstack([params["InitialSOC_MWh"], soc_end]), 0.0
        ).max(),
        "SOCUpperViolation_MWh": np.maximum(
            np.vstack([params["InitialSOC_MWh"], soc_end])
            - params["StorageCapacity_MWh"],
            0.0,
        ).max(),
        "ChargePowerViolation_MW": np.maximum(
            renewable_charge + grid_charge - params["MaxChargePower_MW"], 0.0
        ).max(),
        "DischargePowerViolation_MW": np.maximum(
            discharge - params["MaxDischargePower_MW"], 0.0
        ).max(),
        "ChargeDischargeOverlap_MW": np.minimum(
            renewable_charge + grid_charge, discharge
        ).max(),
        "GridImportViolation_MW": np.maximum(
            grid_purchase - params["MaxGridImport_MW"], 0.0
        ).max(),
        "SellLimitViolation_MW": np.maximum(
            renewable_export - params["SellLimit_MW"], 0.0
        ).max(),
        "GridExportViolation_MW": np.maximum(
            renewable_export - params["MaxGridExport_MW"], 0.0
        ).max(),
        "GridChargeSourceViolation_MW": np.maximum(
            grid_charge - grid_purchase, 0.0
        ).max(),
        "DirectRenewableLoadViolation_MW": np.maximum(
            direct - fixed_load, 0.0
        ).max(),
        "NetGridDefinitionResidual_MW": np.max(
            np.abs(net - grid_purchase + renewable_export)
        ),
        "ReportedChargePowerResidual_MW": np.max(
            np.abs(profile["ChargePower_MW"].to_numpy(dtype=float) - renewable_charge.reshape(-1) - grid_charge.reshape(-1))
        ),
        "TerminalFlagViolation": float((~terminal_flag).sum()),
        "TerminalDispatchNonzero_MW": float(
            np.abs(
                terminal[
                    [
                        "RenewableDirectUse_MW",
                        "RenewableCharge_MW",
                        "GridCharge_MW",
                        "ChargePower_MW",
                        "DischargePower_MW",
                        "GridPurchase_MW",
                        "RenewableExport_MW",
                        "RenewableCurtailment_MW",
                        "NetGridImport_MW",
                    ]
                ].to_numpy(dtype=float)
            ).max()
        ),
    }
    rows = []
    for check, value in residuals.items():
        rows.append(
            {
                "Scheme": scheme,
                "Check": check,
                "MaxViolation": float(value),
                "Tolerance": AUDIT_TOLERANCE,
                "Status": _profile_violation_status(scheme, float(value)),
            }
        )
    return pd.DataFrame(rows)


def _recompute_metrics(operating: pd.DataFrame, energy: pd.DataFrame) -> dict[str, float]:
    profile = _merge_energy(operating, energy).sort_values(["Hour", "Region"], kind="stable")
    net = profile["NetGridImport_MW"].to_numpy(dtype=float)
    peak = (
        profile.assign(_PositiveNet=np.maximum(net, 0.0))
        .groupby("Region", sort=True)["_PositiveNet"]
        .max()
        .sum()
    )
    ramp = profile.groupby("Region", sort=True)["NetGridImport_MW"].diff().abs().dropna()
    return {
        "Cost": float(
            np.sum(
                profile["ElectricityPrice_CNY_per_MWh"].to_numpy(dtype=float)
                * profile["GridPurchase_MW"].to_numpy(dtype=float)
                - profile["SellPrice_CNY_per_MWh"].to_numpy(dtype=float)
                * profile["RenewableExport_MW"].to_numpy(dtype=float)
            )
        ),
        "Carbon": float(
            np.sum(
                profile["CarbonIntensity_tCO2_per_MWh"].to_numpy(dtype=float)
                * profile["GridPurchase_MW"].to_numpy(dtype=float)
            )
        ),
        "Peak": float(peak),
        "Ramp": float(ramp.mean()) if not ramp.empty else 0.0,
        "Throughput": float(
            np.sum(
                profile["RenewableCharge_MW"].to_numpy(dtype=float)
                + profile["GridCharge_MW"].to_numpy(dtype=float)
                + profile["DischargePower_MW"].to_numpy(dtype=float)
            )
        ),
    }


def _metric_recompute_checks(
    profiles: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    energy: pd.DataFrame,
) -> pd.DataFrame:
    summary = _read_csv(
        TABLES_DIR / "q3_objective_summary.csv",
        ("Solution", "Cost", "Carbon", "Peak", "Ramp", "Throughput"),
    )
    rows: list[dict[str, object]] = []
    for scheme in SCHEMES:
        summary_row = summary.loc[summary["Solution"] == scheme]
        if len(summary_row) != 1:
            raise ValueError(f"q3_objective_summary.csv is missing a unique {scheme} row")
        recomputed = _recompute_metrics(profiles[scheme][0], energy)
        for metric, value in recomputed.items():
            reported = float(summary_row.iloc[0][metric])
            difference = value - reported
            tolerance = _metric_tolerance(reported)
            rows.append(
                {
                    "Scheme": scheme,
                    "Metric": metric,
                    "Reported": reported,
                    "Recomputed": value,
                    "Difference": difference,
                    "Tolerance": tolerance,
                    "Status": "PASS" if abs(difference) <= tolerance else "FAIL",
                }
            )
    return pd.DataFrame(rows)


def _baseline_reconciliation(
    baseline_profile: tuple[pd.DataFrame, pd.DataFrame],
    baseline_source: pd.DataFrame,
) -> pd.DataFrame:
    operating, terminal = baseline_profile
    op = operating.merge(
        baseline_source,
        how="left",
        on=["Hour", "Region"],
        validate="one_to_one",
        suffixes=("_out", "_source"),
    )
    if op[["UsedRenewable_MW", "GridPurchase_MW_source"]].isna().any().any():
        raise ValueError("Attachment baseline states do not fully cover the baseline solution in model output")
    mapping = {
        "RenewableDirectUse_MW": "UsedRenewable_MW",
        "RenewableCharge_MW": "RenewableCharge_MW",
        "RenewableCurtailment_MW": "Curtailment_MW",
        "GridPurchase_MW": "GridPurchase_MW",
        "GridCharge_MW": "GridCharge_MW",
        "RenewableExport_MW": "GridSell_MW",
        "NetGridImport_MW": "NetGridImport_MW",
        "SOC_MWh": "SOC_MWh",
        "ChargePower_MW": "ChargePower_MW",
        "DischargePower_MW": "DischargePower_MW",
    }
    rows: list[dict[str, object]] = []
    for output_column, source_column in mapping.items():
        output_name = f"{output_column}_out" if output_column in baseline_source.columns else output_column
        source_name = (
            f"{source_column}_source"
            if source_column in operating.columns
            else source_column
        )
        if output_name not in op.columns or source_name not in op.columns:
            raise ValueError(f"Baseline reproduction field does not exist: {output_column} / {source_column}")
        difference = np.abs(
            op[output_name].to_numpy(dtype=float) - op[source_name].to_numpy(dtype=float)
        )
        maximum = float(difference.max()) if len(difference) else 0.0
        rows.append(
            {
                "Scope": "operating",
                "Field": output_column,
                "MaxAbsoluteDifference": maximum,
                "Tolerance": BASELINE_RECONCILIATION_TOLERANCE,
                "Status": _violation_status(maximum, BASELINE_RECONCILIATION_TOLERANCE),
            }
        )

    source_terminal = baseline_source.loc[
        baseline_source["Hour"] == int(terminal["Hour"].iloc[0])
    ].sort_values("Region", kind="stable")
    output_terminal = terminal.sort_values("Region", kind="stable")
    if len(source_terminal) != len(output_terminal):
        raise ValueError("Baseline terminal-state regional row counts disagree")
    terminal_difference = np.abs(
        output_terminal["SOC_MWh"].to_numpy(dtype=float)
        - source_terminal["SOC_MWh"].to_numpy(dtype=float)
    )
    rows.append(
        {
            "Scope": "terminal",
            "Field": "SOC_MWh",
            "MaxAbsoluteDifference": float(terminal_difference.max()),
            "Tolerance": BASELINE_RECONCILIATION_TOLERANCE,
            "Status": _violation_status(
                float(terminal_difference.max()), BASELINE_RECONCILIATION_TOLERANCE
            ),
        }
    )
    return pd.DataFrame(rows)


def _multiobjective_checks() -> tuple[pd.DataFrame, pd.DataFrame]:
    anchors = _read_csv(
        TABLES_DIR / "q3_anchor_metrics.csv",
        ("Solution", "AnchorObjective", "Cost", "Carbon", "Peak", "Ramp", "Throughput"),
    )
    summary = _read_csv(TABLES_DIR / "q3_objective_summary.csv", ("Solution",))
    balanced = summary.loc[summary["Solution"] == "Balanced"]
    if len(balanced) != 1:
        raise ValueError("q3_objective_summary.csv is missing a unique Balanced row")
    balanced_row = balanced.iloc[0]
    rows: list[dict[str, object]] = []
    best: dict[str, float] = {}
    reference: dict[str, float] = {}
    for name in OBJECTIVE_NAMES:
        best_column = f"{name}Best"
        reference_column = f"{name}Reference"
        if best_column not in balanced or reference_column not in balanced:
            raise ValueError(f"q3_objective_summary.csv is missing {best_column}/{reference_column}")
        best[name] = float(balanced_row[best_column])
        reference[name] = float(balanced_row[reference_column])
        anchor_row = anchors.loc[anchors["AnchorObjective"] == name]
        if len(anchor_row) != 1:
            raise ValueError(f"q3_anchor_metrics.csv is missing a unique {name} anchor")
        anchor_value = float(anchor_row.iloc[0][name])
        all_values = anchors[name].to_numpy(dtype=float)
        minimum = float(np.min(all_values))
        diff = anchor_value - best[name]
        rows.append(
            {
                "Check": f"Anchor_{name}_reaches_best",
                "Value": anchor_value,
                "Reference": best[name],
                "Difference": diff,
                "Tolerance": _metric_tolerance(best[name]),
                "Status": "PASS" if abs(diff) <= _metric_tolerance(best[name]) else "FAIL",
            }
        )
        rows.append(
            {
                "Check": f"Anchor_{name}_is_minimum_among_anchors",
                "Value": anchor_value,
                "Reference": minimum,
                "Difference": anchor_value - minimum,
                "Tolerance": _metric_tolerance(minimum),
                "Status": "PASS" if anchor_value <= minimum + _metric_tolerance(minimum) else "FAIL",
            }
        )

    balanced_metrics = {name: float(balanced_row[name]) for name in OBJECTIVE_NAMES}
    for anchor_row in anchors.itertuples(index=False):
        anchor_metrics = {name: float(getattr(anchor_row, name)) for name in OBJECTIVE_NAMES}
        tolerances = {name: _metric_tolerance(balanced_metrics[name]) for name in OBJECTIVE_NAMES}
        dominates = all(
            anchor_metrics[name] <= balanced_metrics[name] + tolerances[name]
            for name in OBJECTIVE_NAMES
        ) and any(
            anchor_metrics[name] < balanced_metrics[name] - tolerances[name]
            for name in OBJECTIVE_NAMES
        )
        rows.append(
            {
                "Check": f"Balanced_not_dominated_by_{anchor_row.Solution}",
                "Value": int(dominates),
                "Reference": 0,
                "Difference": int(dominates),
                "Tolerance": 0,
                "Status": "FAIL" if dominates else "PASS",
            }
        )

    solver = _read_csv(
        TABLES_DIR / "q3_solver_log.csv",
        ("Stage", "Status", "Success", "Cost", "Carbon", "Peak", "Ramp", "Throughput"),
    )
    solver_rows: list[dict[str, object]] = []
    for row in solver.itertuples(index=False):
        status = int(row.Status)
        success = str(row.Success).lower() in ("true", "1", "yes")
        solver_rows.append(
            {
                "Stage": row.Stage,
                "Status": status,
                "Success": success,
                "CheckStatus": "PASS" if status == 0 and success else "WARN" if success else "FAIL",
            }
        )

    def normalized(row: pd.Series | object) -> tuple[float, float]:
        values = {
            name: float(row[name] if isinstance(row, pd.Series) else getattr(row, name))
            for name in OBJECTIVE_NAMES
        }
        deviations = [
            (values[name] - best[name]) / (reference[name] - best[name])
            for name in OBJECTIVE_NAMES
            if reference[name] - best[name] > FLOAT_EPS
        ]
        return (max(deviations) if deviations else 0.0, sum(deviations))

    stage_frames = {str(row.Stage): row for row in solver.itertuples(index=False)}
    required_stages = (
        "balanced_stage_1_minimax",
        "balanced_stage_2_sum_deviation",
        "balanced_stage_3_min_throughput",
    )
    if not all(stage in stage_frames for stage in required_stages):
        raise ValueError("q3_solver_log.csv is missing the three balanced stages")
    stage_one = stage_frames[required_stages[0]]
    stage_two = stage_frames[required_stages[1]]
    stage_three = stage_frames[required_stages[2]]
    z_one, _ = normalized(stage_one)
    _, phi_two = normalized(stage_two)
    z_three, phi_three = normalized(stage_three)
    balance_comparison_tolerance = BALANCE_TOLERANCE + BALANCE_COMPARISON_EPS
    rows.extend(
        [
            {
                "Check": "Stage3_preserves_minimax_z",
                "Value": z_three,
                "Reference": z_one,
                "Difference": z_three - z_one,
                "Tolerance": balance_comparison_tolerance,
                "Status": "PASS"
                if z_three <= z_one + balance_comparison_tolerance
                else "FAIL",
            },
            {
                "Check": "Stage3_preserves_sum_deviation",
                "Value": phi_three,
                "Reference": phi_two,
                "Difference": phi_three - phi_two,
                "Tolerance": balance_comparison_tolerance,
                "Status": "PASS"
                if phi_three <= phi_two + balance_comparison_tolerance
                else "FAIL",
            },
            {
                "Check": "Stage3_does_not_increase_throughput",
                "Value": float(stage_three.Throughput),
                "Reference": float(stage_two.Throughput),
                "Difference": float(stage_three.Throughput - stage_two.Throughput),
                "Tolerance": _metric_tolerance(float(stage_two.Throughput)),
                "Status": "PASS"
                if float(stage_three.Throughput) <= float(stage_two.Throughput) + _metric_tolerance(float(stage_two.Throughput))
                else "FAIL",
            },
        ]
    )
    return pd.DataFrame(rows), pd.DataFrame(solver_rows)


def _existing_model_audit() -> pd.DataFrame:
    audit = _read_csv(TABLES_DIR / "q3_constraint_audit.csv", ("Check", "MaxViolation", "Status"))
    result = audit.copy()
    result.insert(0, "Source", "model_self_audit")
    return result


def _write_table(frame: pd.DataFrame, filename: str) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES_DIR / filename, index=False, encoding="utf-8-sig", float_format="%.15g")


def _write_report(
    summary: pd.DataFrame,
    profile_checks: pd.DataFrame,
    metric_checks: pd.DataFrame,
    baseline_checks: pd.DataFrame,
    multiobjective_checks: pd.DataFrame,
    solver_checks: pd.DataFrame,
) -> None:
    counts = summary["Status"].value_counts().to_dict()
    lines = [
        "Q3 independent validation report",
        "",
        f"Overall status: {summary.attrs.get('OverallStatus', 'UNKNOWN')}",
        f"Check counts: {counts}",
        "",
        "Validation scope: independently recompute hourly model results, fixed Q3 energy inputs, shared storage parameters, and attachment baseline states; do not resolve MILP.",
        f"Constraint checks: {len(profile_checks)}; metric recomputation: {len(metric_checks)}; baseline reproduction: {len(baseline_checks)}; multiobjective checks: {len(multiobjective_checks)}; solver stages: {len(solver_checks)}.",
        "",
        "With FAIL results, do not directly use Q3 optimization conclusions in the paper. With WARN only, explain solver optimality limits in the paper or handoff record.",
    ]
    (TABLES_DIR / "q3_validation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent Question 3 validation")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Console/file log level, default INFO",
    )
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    logging.info("Independent Q3 validation started: result directory=%s.", TABLES_DIR)

    energy, storage, baseline, _ = _load_context()
    profiles = {scheme: _load_profile(scheme) for scheme in SCHEMES}
    operating_keys = set(zip(profiles["Balanced"][0]["Hour"], profiles["Balanced"][0]["Region"]))
    terminal_hour = int(profiles["Balanced"][1]["Hour"].iloc[0])
    expected_energy_keys = operating_keys | set(
        zip(profiles["Balanced"][1]["Hour"], profiles["Balanced"][1]["Region"])
    )
    _complete_grid(
        energy.loc[energy["Hour"] <= terminal_hour],
        expected_energy_keys,
        "Q3 inputs",
    )
    for scheme, (operating, terminal) in profiles.items():
        expected_operating = operating_keys
        actual_operating = set(zip(operating["Hour"], operating["Region"]))
        if actual_operating != expected_operating:
            raise ValueError(f"{scheme} operating Hour x Region grid differs from Balanced")
        if set(zip(terminal["Hour"], terminal["Region"])) != set(
            zip(profiles["Balanced"][1]["Hour"], profiles["Balanced"][1]["Region"])
        ):
            raise ValueError(f"{scheme} terminal Hour x Region grid differs from Balanced")

    profile_checks = pd.concat(
        [
            _check_profile(scheme, operating, terminal, energy, storage)
            for scheme, (operating, terminal) in profiles.items()
        ],
        ignore_index=True,
    )
    metric_checks = _metric_recompute_checks(profiles, energy)
    baseline_checks = _baseline_reconciliation(profiles["BaselineReference"], baseline)
    multiobjective_checks, solver_checks = _multiobjective_checks()
    model_audit = _existing_model_audit()

    profile_summary = profile_checks.loc[:, ["Scheme", "Check", "MaxViolation", "Tolerance", "Status"]].copy()
    profile_summary.insert(0, "Category", "independent_constraint_recheck")
    metric_summary = metric_checks.loc[:, ["Scheme", "Metric", "Difference", "Tolerance", "Status"]].copy()
    metric_summary = metric_summary.rename(columns={"Metric": "Check", "Difference": "MaxViolation"})
    metric_summary.insert(0, "Category", "metric_recompute")
    baseline_summary = baseline_checks.rename(
        columns={"Field": "Check", "MaxAbsoluteDifference": "MaxViolation"}
    ).copy()
    baseline_summary.insert(0, "Category", "baseline_reconciliation")
    baseline_summary.insert(0, "Scheme", "BaselineReference")
    multi_summary = multiobjective_checks.loc[:, ["Check", "Difference", "Tolerance", "Status"]].copy()
    multi_summary = multi_summary.rename(columns={"Difference": "MaxViolation"})
    multi_summary.insert(0, "Category", "multiobjective_structure")
    solver_summary = solver_checks.loc[:, ["Stage", "CheckStatus"]].rename(
        columns={"Stage": "Check", "CheckStatus": "Status"}
    ).copy()
    solver_summary["MaxViolation"] = np.nan
    solver_summary["Tolerance"] = np.nan
    solver_summary.insert(0, "Category", "solver_status")
    solver_summary.insert(0, "Scheme", "All")
    model_summary = model_audit.loc[:, ["Check", "MaxViolation", "Status"]].copy()
    model_summary["Tolerance"] = AUDIT_TOLERANCE
    model_summary.insert(0, "Category", "model_self_audit")
    model_summary.insert(0, "Scheme", "Balanced")

    summary = pd.concat(
        [
            profile_summary,
            metric_summary,
            baseline_summary,
            multi_summary,
            solver_summary.loc[:, ["Scheme", "Category", "Check", "MaxViolation", "Tolerance", "Status"]],
            model_summary.loc[:, ["Scheme", "Category", "Check", "MaxViolation", "Tolerance", "Status"]],
        ],
        ignore_index=True,
        sort=False,
    )
    summary["Status"] = summary["Status"].astype(str).str.upper()
    has_fail = (summary["Status"] == "FAIL").any()
    has_warn = (summary["Status"] == "WARN").any()
    overall = "FAIL" if has_fail else "PASS_WITH_WARNINGS" if has_warn else "PASS"
    summary.attrs["OverallStatus"] = overall
    _write_table(summary, "q3_validation_summary.csv")
    _write_table(profile_checks, "q3_validation_profile_checks.csv")
    _write_table(metric_checks, "q3_validation_metric_recompute.csv")
    _write_table(baseline_checks, "q3_validation_baseline_reconciliation.csv")
    _write_table(multiobjective_checks, "q3_validation_multiobjective.csv")
    _write_table(solver_checks, "q3_validation_solver_checks.csv")
    _write_report(
        summary,
        profile_checks,
        metric_checks,
        baseline_checks,
        multiobjective_checks,
        solver_checks,
    )

    logging.info(
        "Independent Q3 validation completed: overall=%s, PASS=%d, WARN=%d, FAIL=%d.",
        overall,
        int((summary["Status"] == "PASS").sum()),
        int((summary["Status"] == "WARN").sum()),
        int((summary["Status"] == "FAIL").sum()),
    )
    if has_fail:
        logging.error("FAIL results exist; see q3_validation_summary.csv.")
        return 1
    if has_warn:
        logging.warning("Validation passed with solver/evidence-limit warnings; see q3_validation_solver_checks.csv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
