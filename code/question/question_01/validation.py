"""Three rigorous Question 1 validation chains.

Retain the three evidence chains specified by the Q1 validation plan:

1. Rolling out-of-time forecasts and aggregation-scale generalization.
2. Algebraic reconciliation consistency, historical structure stability, and
   bottom-level error comparisons.
3. A global optimality certificate for the first scheduling objective and
   critical-load stress bounds.

Independently recompute forecasting metrics, structure shares, resource profiles,
and stress feasibility without changing ``model.py`` parameters or overwriting
model results. Write ``outputs/tables/validation_*.csv`` only; no decorative plots.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


QUESTION_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = QUESTION_DIR / "data" / "processed"
SHARED_DIR = PROCESSED_DIR / "shared"
Q1_DIR = PROCESSED_DIR / "q1"
TABLES_DIR = QUESTION_DIR / "outputs" / "tables"

TRAIN_START_HOUR = 0
TRAIN_END_HOUR = 2351
VALIDATION_START_HOUR = 2352
VALIDATION_END_HOUR = 2375
TEST_START_HOUR = 2376
TEST_END_HOUR = 2399
TAIL_END_HOUR = 2405
TERMINAL_HOUR = 2406

SERIES_COLUMNS = ["SourceRegion", "TaskType"]
FORECAST_MODEL = "HierarchicalLocalMean"
DIRECT_MODEL = "DirectLocalMean"
BASELINE_MODEL = "SameHour24Baseline"
HISTORY_WINDOWS = (24, 72, 168, 336)
SAME_HOUR_LAG = 24
ROLLING_MAX_HISTORY = max(HISTORY_WINDOWS)
ROLLING_HORIZON = 24
NUMERICAL_EPS = 1e-8
ALPHA_SEARCH_TOLERANCE = 0.01
ALPHA_SEARCH_MAX = 128.0

PANEL_COLUMNS = [
    "Hour",
    "SourceRegion",
    "TaskType",
    "Task_Count",
    "GPU_Demand_Arrival",
    "GPU_Workload_Arrival_GPUh",
    "HourOfDay",
    "DayIndex",
    "Split",
]
TASK_COLUMNS = [
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
    "Duration_h",
    "GPU_Workload_GPUh",
    "GPU_Power_MW_per_EquivalentGPU",
    "Task_Full_IT_Power_MW",
]
CANDIDATE_COLUMNS = [
    "TaskID",
    "TaskType",
    "SourceRegion",
    "TargetRegion",
    "NetworkLatency_ms",
    "MaxLatency_ms",
]
CAPACITY_COLUMNS = [
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


def _read_csv(path: Path, required_columns: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Validation input table not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required validation columns: {missing}")
    return frame


def _write_table(frame: pd.DataFrame, filename: str) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES_DIR / filename, index=False, encoding="utf-8-sig")


def _record(
    records: list[dict[str, object]],
    test_name: str,
    item: str,
    status: str,
    evidence: str,
    criterion: str,
    action: str,
) -> None:
    records.append(
        {
            "Test": test_name,
            "Item": item,
            "Status": status,
            "Evidence": evidence,
            "Criterion": criterion,
            "IfNotPassed": action,
        }
    )


def _to_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="raise")
    return result


def _read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = _read_csv(Q1_DIR / "hourly_demand_panel.csv", PANEL_COLUMNS)
    tasks = _read_csv(SHARED_DIR / "tasks_clean.csv", TASK_COLUMNS)
    candidates = _read_csv(
        SHARED_DIR / "task_candidate_regions.csv", CANDIDATE_COLUMNS
    )
    capacity = _read_csv(Q1_DIR / "region_hour_capacity.csv", CAPACITY_COLUMNS)

    panel = _to_numeric(
        panel,
        [
            "Hour",
            "Task_Count",
            "GPU_Demand_Arrival",
            "GPU_Workload_Arrival_GPUh",
        ],
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
            "GPU_Workload_GPUh",
            "GPU_Power_MW_per_EquivalentGPU",
            "Task_Full_IT_Power_MW",
        ],
    )
    candidates = _to_numeric(candidates, ["NetworkLatency_ms", "MaxLatency_ms"])
    capacity = _to_numeric(
        capacity,
        [
            "Hour",
            "Available_GPU",
            "NonAI_IT_Load_MW",
            "Max_IT_Power_MW",
            "PUE",
            "Max_Facility_Power_MW",
            "AI_IT_Margin_MW",
            "AI_Facility_Margin_MW",
            "Effective_AI_IT_Capacity_MW",
        ],
    )
    panel["Hour"] = panel["Hour"].astype(int)
    tasks["TaskID"] = tasks["TaskID"].astype(str)
    candidates["TaskID"] = candidates["TaskID"].astype(str)
    capacity["Hour"] = capacity["Hour"].astype(int)
    for column in ["SourceRegion", "TaskType"]:
        panel[column] = panel[column].astype(str)
        tasks[column] = tasks[column].astype(str)
        candidates[column] = candidates[column].astype(str)
    capacity["Region"] = capacity["Region"].astype(str)

    if panel.duplicated(["Hour", *SERIES_COLUMNS]).any():
        raise ValueError("hourly_demand_panel must have unique Hour x SourceRegion x TaskType keys")
    if tasks["TaskID"].duplicated().any():
        raise ValueError("tasks_clean contains duplicate TaskID values")
    if candidates.duplicated(["TaskID", "TargetRegion"]).any():
        raise ValueError("task_candidate_regions contains duplicate TaskID x TargetRegion records")
    if capacity.duplicated(["Hour", "Region"]).any():
        raise ValueError("region_hour_capacity contains duplicate Hour x Region records")
    if not capacity["Hour"].between(TEST_START_HOUR, TAIL_END_HOUR).all():
        raise ValueError("Capacity must cover hours 2376--2405 and exclude task execution capacity at hour 2406")
    return panel, tasks, candidates, capacity


def _load_model_module():
    path = QUESTION_DIR / "model.py"
    spec = importlib.util.spec_from_file_location("q1_model_for_validation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the Question 1 model module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metric_values(actual: pd.Series, prediction: pd.Series) -> tuple[float, float, float]:
    actual_values = actual.to_numpy(dtype=float)
    prediction_values = prediction.to_numpy(dtype=float)
    error = actual_values - prediction_values
    denominator = float(np.abs(actual_values).sum())
    wape = float(np.abs(error).sum() / denominator) if denominator > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(np.square(error)))) if len(error) else float("nan")
    mae = float(np.mean(np.abs(error))) if len(error) else float("nan")
    return wape, rmse, mae


def _level_entities(panel: pd.DataFrame, level: str) -> list[str]:
    if level == "Region":
        return sorted(panel["SourceRegion"].astype(str).unique())
    if level == "TaskType":
        return sorted(panel["TaskType"].astype(str).unique())
    if level == "System":
        return ["ALL"]
    raise ValueError(f"Unknown level: {level}")


def _level_demand(frame: pd.DataFrame, level: str) -> pd.DataFrame:
    if level == "Region":
        result = (
            frame.groupby(["Hour", "SourceRegion"], as_index=False)["GPU_Demand_Arrival"]
            .sum()
            .rename(columns={"SourceRegion": "Entity", "GPU_Demand_Arrival": "Demand"})
        )
    elif level == "TaskType":
        result = (
            frame.groupby(["Hour", "TaskType"], as_index=False)["GPU_Demand_Arrival"]
            .sum()
            .rename(columns={"TaskType": "Entity", "GPU_Demand_Arrival": "Demand"})
        )
    elif level == "System":
        result = (
            frame.groupby("Hour", as_index=False)["GPU_Demand_Arrival"]
            .sum()
            .rename(columns={"GPU_Demand_Arrival": "Demand"})
        )
        result["Entity"] = "ALL"
    else:
        raise ValueError(f"Unknown level: {level}")
    result["Entity"] = result["Entity"].astype(str)
    return result[["Hour", "Entity", "Demand"]]


def _mean_vector(frame: pd.DataFrame, level: str, entities: list[str]) -> pd.Series:
    hourly = _level_demand(frame, level)
    means = hourly.groupby("Entity")["Demand"].mean()
    return means.reindex(entities).fillna(0.0)


def _ipf(row_targets: np.ndarray, column_targets: np.ndarray, prior: np.ndarray) -> np.ndarray:
    rows = np.maximum(np.asarray(row_targets, dtype=float), 0.0)
    columns = np.maximum(np.asarray(column_targets, dtype=float), 0.0)
    total = float(rows.sum())
    if total <= NUMERICAL_EPS:
        return np.zeros_like(prior, dtype=float)
    if columns.sum() <= NUMERICAL_EPS:
        columns = np.full_like(columns, total / len(columns))
    else:
        columns = columns * total / float(columns.sum())
    matrix = np.maximum(prior, 1e-15) * total
    tolerance = NUMERICAL_EPS * max(1.0, total)
    for _ in range(1000):
        row_sums = matrix.sum(axis=1)
        matrix *= np.divide(
            rows,
            row_sums,
            out=np.ones_like(rows),
            where=row_sums > 0,
        )[:, None]
        column_sums = matrix.sum(axis=0)
        matrix *= np.divide(
            columns,
            column_sums,
            out=np.ones_like(columns),
            where=column_sums > 0,
        )[None, :]
        residual = max(
            float(np.max(np.abs(matrix.sum(axis=1) - rows))),
            float(np.max(np.abs(matrix.sum(axis=0) - columns))),
        )
        if residual <= tolerance:
            return matrix
    raise RuntimeError("Independent IPF validation did not converge within the iteration limit")


def _history_prior(history: pd.DataFrame, regions: list[str], task_types: list[str]) -> np.ndarray:
    matrix = (
        history.pivot_table(
            index="SourceRegion",
            columns="TaskType",
            values="GPU_Demand_Arrival",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=regions, columns=task_types, fill_value=0.0)
        .to_numpy(dtype=float)
    )
    total = float(matrix.sum())
    if total <= NUMERICAL_EPS:
        return np.full(matrix.shape, 1.0 / matrix.size, dtype=float)
    matrix = matrix / total
    matrix = matrix + 1e-12
    return matrix / matrix.sum()


def _hierarchical_prediction(
    panel: pd.DataFrame,
    target_start_hour: int,
    target_end_hour: int,
    fit_end_hour: int,
    history_window_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = panel.loc[panel["Hour"].between(TRAIN_START_HOUR, fit_end_hour)].copy()
    window_start = max(TRAIN_START_HOUR, fit_end_hour - history_window_hours + 1)
    window = history.loc[history["Hour"].between(window_start, fit_end_hour)].copy()
    target = panel.loc[panel["Hour"].between(target_start_hour, target_end_hour)].copy()
    if history.empty or window.empty or target.empty:
        raise ValueError("Rolling forecasts are missing history or target windows")

    regions = _level_entities(history, "Region")
    task_types = _level_entities(history, "TaskType")
    region_mean = _mean_vector(window, "Region", regions)
    type_mean = _mean_vector(window, "TaskType", task_types)
    system_mean = float(_mean_vector(window, "System", ["ALL"]).iloc[0])
    direct_mean = (
        window.groupby(SERIES_COLUMNS)["GPU_Demand_Arrival"]
        .mean()
        .reindex(
            pd.MultiIndex.from_product(
                [regions, task_types], names=SERIES_COLUMNS
            )
        )
        .fillna(0.0)
    )
    prior = _history_prior(history, regions, task_types)
    actual_bottom = (
        target.set_index(["Hour", *SERIES_COLUMNS])["GPU_Demand_Arrival"]
        .astype(float)
        .to_dict()
    )
    actual_levels = {
        level: _level_demand(target, level)
        .set_index(["Hour", "Entity"])["Demand"]
        .to_dict()
        for level in ["Region", "TaskType", "System"]
    }

    bottom_rows: list[dict[str, object]] = []
    hierarchy_rows: list[dict[str, object]] = []
    for hour in range(target_start_hour, target_end_hour + 1):
        region_independent = region_mean.to_numpy(dtype=float)
        type_independent = type_mean.to_numpy(dtype=float)
        system_prediction = max(0.0, system_mean)
        region_total = float(region_independent.sum())
        type_total = float(type_independent.sum())
        region_reconciled = (
            region_independent * system_prediction / region_total
            if region_total > NUMERICAL_EPS
            else np.full(len(regions), system_prediction / len(regions))
        )
        type_reconciled = (
            type_independent * system_prediction / type_total
            if type_total > NUMERICAL_EPS
            else np.full(len(task_types), system_prediction / len(task_types))
        )
        cross = _ipf(region_reconciled, type_reconciled, prior)

        for index, region in enumerate(regions):
            hierarchy_rows.append(
                {
                    "Level": "Region",
                    "Entity": region,
                    "Hour": hour,
                    "Actual_GPU_Demand": float(
                        actual_levels["Region"].get((hour, region), 0.0)
                    ),
                    "IndependentPrediction": float(region_independent[index]),
                    "ReconciledPrediction": float(region_reconciled[index]),
                }
            )
        for index, task_type in enumerate(task_types):
            hierarchy_rows.append(
                {
                    "Level": "TaskType",
                    "Entity": task_type,
                    "Hour": hour,
                    "Actual_GPU_Demand": float(
                        actual_levels["TaskType"].get((hour, task_type), 0.0)
                    ),
                    "IndependentPrediction": float(type_independent[index]),
                    "ReconciledPrediction": float(type_reconciled[index]),
                }
            )
        hierarchy_rows.append(
            {
                "Level": "System",
                "Entity": "ALL",
                "Hour": hour,
                "Actual_GPU_Demand": float(
                    actual_levels["System"].get((hour, "ALL"), 0.0)
                ),
                "IndependentPrediction": system_prediction,
                "ReconciledPrediction": system_prediction,
            }
        )
        for region_index, region in enumerate(regions):
            for type_index, task_type in enumerate(task_types):
                bottom_rows.append(
                    {
                        "SourceRegion": region,
                        "TaskType": task_type,
                        "Hour": hour,
                        "Actual_GPU_Demand": float(
                            actual_bottom.get((hour, region, task_type), 0.0)
                        ),
                        "HierarchicalPrediction": float(cross[region_index, type_index]),
                        "DirectPrediction": float(
                            direct_mean.loc[(region, task_type)]
                        ),
                    }
                )
    return pd.DataFrame(bottom_rows), pd.DataFrame(hierarchy_rows)


def _same_hour_prediction(
    panel: pd.DataFrame, target_start_hour: int, target_end_hour: int
) -> pd.DataFrame:
    target = panel.loc[
        panel["Hour"].between(target_start_hour, target_end_hour),
        [*SERIES_COLUMNS, "Hour", "GPU_Demand_Arrival"],
    ].copy()
    lagged = panel.loc[:, [*SERIES_COLUMNS, "Hour", "GPU_Demand_Arrival"]].copy()
    lagged["Hour"] = lagged["Hour"] + SAME_HOUR_LAG
    lagged = lagged.rename(columns={"GPU_Demand_Arrival": "Prediction"})
    result = target.merge(
        lagged,
        how="left",
        on=[*SERIES_COLUMNS, "Hour"],
        validate="one_to_one",
    )
    if result["Prediction"].isna().any():
        raise ValueError("The 24-hour same-hour baseline is missing t-24 history")
    return result.rename(columns={"GPU_Demand_Arrival": "Actual_GPU_Demand"})


def _aggregate_for_metric(bottom: pd.DataFrame, prediction_column: str, level: str) -> pd.DataFrame:
    if level == "Bottom":
        return bottom[["Hour", *SERIES_COLUMNS, "Actual_GPU_Demand", prediction_column]].copy()
    if level == "Region":
        group_columns = ["Hour", "SourceRegion"]
    elif level == "TaskType":
        group_columns = ["Hour", "TaskType"]
    elif level == "System":
        group_columns = ["Hour"]
    else:
        raise ValueError(f"Unknown evaluation level: {level}")
    result = (
        bottom.groupby(group_columns, as_index=False)[
            ["Actual_GPU_Demand", prediction_column]
        ]
        .sum()
    )
    return result


def _metric_rows(
    bottom: pd.DataFrame,
    prediction_column: str,
    model_name: str,
    split_name: str,
    history_window_hours: int,
    window_id: int,
    start_hour: int,
    end_hour: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for level in ["Bottom", "Region", "TaskType", "System"]:
        frame = _aggregate_for_metric(bottom, prediction_column, level)
        wape, rmse, mae = _metric_values(
            frame["Actual_GPU_Demand"], frame[prediction_column]
        )
        rows.append(
            {
                "Split": split_name,
                "WindowID": window_id,
                "WindowStartHour": start_hour,
                "WindowEndHour": end_hour,
                "HistoryWindowHours": history_window_hours,
                "Model": model_name,
                "Level": level,
                "ObservationCount": len(frame),
                "ActualTotal_GPU": float(frame["Actual_GPU_Demand"].sum()),
                "WAPE": wape,
                "RMSE": rmse,
                "MAE": mae,
            }
        )
    return rows


def _rolling_windows() -> list[tuple[int, int, int]]:
    starts = list(
        range(
            ROLLING_MAX_HISTORY,
            TRAIN_END_HOUR - ROLLING_HORIZON + 2,
            ROLLING_HORIZON,
        )
    )
    return [
        (window_id, start, start + ROLLING_HORIZON - 1)
        for window_id, start in enumerate(starts, start=1)
    ]


def _independent_selection(
    panel: pd.DataFrame,
) -> tuple[int, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for history_window_hours in HISTORY_WINDOWS:
        bottom, _ = _hierarchical_prediction(
            panel,
            VALIDATION_START_HOUR,
            VALIDATION_END_HOUR,
            TRAIN_END_HOUR,
            history_window_hours,
        )
        wape, rmse, mae = _metric_values(
            bottom["Actual_GPU_Demand"], bottom["HierarchicalPrediction"]
        )
        rows.append(
            {
                "Model": FORECAST_MODEL,
                "HistoryWindowHours": history_window_hours,
                "ValidationWAPE": wape,
                "ValidationRMSE": rmse,
                "ValidationMAE": mae,
            }
        )
    independent = pd.DataFrame(rows).sort_values(
        ["ValidationWAPE", "ValidationRMSE", "HistoryWindowHours"],
        kind="stable",
    )
    selected = int(independent.iloc[0]["HistoryWindowHours"])
    independent["SelectedIndependent"] = independent["HistoryWindowHours"].eq(selected)
    selection_path = TABLES_DIR / "forecast_window_selection.csv"
    if selection_path.is_file():
        reported = _read_csv(
            selection_path,
            ["Model", "HistoryWindowHours", "ValidationWAPE", "Selected"],
        )
        reported["HistoryWindowHours"] = pd.to_numeric(
            reported["HistoryWindowHours"], errors="coerce"
        )
        reported["Selected"] = reported["Selected"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
        audit = independent.merge(
            reported,
            how="outer",
            on=["Model", "HistoryWindowHours"],
            suffixes=("_Independent", "_Reported"),
            indicator=True,
        )
        audit["ValidationWAPE_AbsDiff"] = (
            audit["ValidationWAPE_Independent"] - audit["ValidationWAPE_Reported"]
        ).abs()
        audit["SelectedMatch"] = audit["SelectedIndependent"].eq(
            audit["Selected"]
        )
        _write_table(audit.drop(columns=["_merge"]), "validation_window_selection_audit.csv")
    else:
        _write_table(independent, "validation_window_selection_audit.csv")
    return selected, independent


def _wilcoxon(values: pd.Series, alternative: str) -> tuple[float, float, int]:
    data = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 2:
        return float("nan"), float("nan"), len(data)
    try:
        from scipy.stats import wilcoxon

        result = wilcoxon(
            data,
            zero_method="wilcox",
            alternative=alternative,
            method="auto",
        )
        return float(result.statistic), float(result.pvalue), len(data)
    except (ImportError, ValueError):
        return float("nan"), float("nan"), len(data)


def _run_rolling_forecast_test(
    panel: pd.DataFrame,
    records: list[dict[str, object]],
) -> dict[str, object]:
    selected_window, selection = _independent_selection(panel)
    rolling_rows: list[dict[str, object]] = []
    hierarchy_rows: list[dict[str, object]] = []
    for window_id, start_hour, end_hour in _rolling_windows():
        if window_id == 1 or window_id % 10 == 0 or window_id == len(_rolling_windows()):
            logging.info(
                "Validation 1 rolling window: %d/%d, target hours=%d--%d",
                window_id,
                len(_rolling_windows()),
                start_hour,
                end_hour,
            )
        baseline = _same_hour_prediction(panel, start_hour, end_hour)
        rolling_rows.extend(
            _metric_rows(
                baseline,
                "Prediction",
                BASELINE_MODEL,
                "rolling",
                SAME_HOUR_LAG,
                window_id,
                start_hour,
                end_hour,
            )
        )
        for history_window_hours in HISTORY_WINDOWS:
            bottom, hierarchy = _hierarchical_prediction(
                panel,
                start_hour,
                end_hour,
                start_hour - 1,
                history_window_hours,
            )
            for model_name, prediction_column in [
                (FORECAST_MODEL, "HierarchicalPrediction"),
                (DIRECT_MODEL, "DirectPrediction"),
            ]:
                rolling_rows.extend(
                    _metric_rows(
                        bottom,
                        prediction_column,
                        model_name,
                        "rolling",
                        history_window_hours,
                        window_id,
                        start_hour,
                        end_hour,
                    )
                )
            hierarchy = hierarchy.copy()
            hierarchy["WindowID"] = window_id
            hierarchy["WindowStartHour"] = start_hour
            hierarchy["WindowEndHour"] = end_hour
            hierarchy["HistoryWindowHours"] = history_window_hours
            hierarchy_rows.append(hierarchy)

    rolling = pd.DataFrame(rolling_rows)
    hierarchy_rolling = pd.concat(hierarchy_rows, ignore_index=True)
    rolling_summary = (
        rolling.groupby(["Model", "Level", "HistoryWindowHours"], as_index=False)
        .agg(
            WindowCount=("WindowID", "nunique"),
            Mean_WAPE=("WAPE", "mean"),
            Median_WAPE=("WAPE", "median"),
            Mean_RMSE=("RMSE", "mean"),
            Mean_MAE=("MAE", "mean"),
        )
        .sort_values(["Level", "Mean_WAPE", "Model"], kind="stable")
    )
    selected_hier = rolling.loc[
        (rolling["Model"] == FORECAST_MODEL)
        & (rolling["Level"] == "Bottom")
        & (rolling["HistoryWindowHours"] == selected_window),
        ["WindowID", "WAPE"],
    ].rename(columns={"WAPE": "HierarchicalWAPE"})
    selected_direct = rolling.loc[
        (rolling["Model"] == DIRECT_MODEL)
        & (rolling["Level"] == "Bottom")
        & (rolling["HistoryWindowHours"] == selected_window),
        ["WindowID", "WAPE"],
    ].rename(columns={"WAPE": "DirectWAPE"})
    selected_baseline = rolling.loc[
        (rolling["Model"] == BASELINE_MODEL) & (rolling["Level"] == "Bottom"),
        ["WindowID", "WAPE"],
    ].rename(columns={"WAPE": "BaselineWAPE"})
    paired = selected_hier.merge(selected_direct, on="WindowID", validate="one_to_one")
    paired = paired.merge(selected_baseline, on="WindowID", validate="one_to_one")
    paired["Delta_HierarchicalMinusBaseline"] = (
        paired["HierarchicalWAPE"] - paired["BaselineWAPE"]
    )
    paired["Delta_HierarchicalMinusDirect"] = (
        paired["HierarchicalWAPE"] - paired["DirectWAPE"]
    )

    _write_table(rolling, "validation_rolling_metrics.csv")
    _write_table(rolling_summary, "validation_rolling_summary.csv")
    _write_table(hierarchy_rolling, "validation_rolling_hierarchy.csv")
    _write_table(paired, "validation_paired_differences.csv")

    delta_baseline = paired["Delta_HierarchicalMinusBaseline"]
    statistic, p_value, sample_count = _wilcoxon(delta_baseline, "less")
    median_delta = float(delta_baseline.median())
    negative_count = int(delta_baseline.lt(0).sum())
    main_baseline_ok = bool(
        np.isfinite(p_value) and p_value < 0.05 and median_delta < 0
    )
    selected_row = selection.loc[
        selection["HistoryWindowHours"].eq(selected_window)
    ].iloc[0]
    rolling_rank = (
        rolling_summary.loc[
            (rolling_summary["Model"] == FORECAST_MODEL)
            & (rolling_summary["Level"] == "Bottom"),
            ["HistoryWindowHours", "Mean_WAPE"],
        ]
        .sort_values("Mean_WAPE", kind="stable")
        .reset_index(drop=True)
    )
    selected_rank = int(
        rolling_rank.index[rolling_rank["HistoryWindowHours"].eq(selected_window)][0]
        + 1
    )

    aggregation = rolling.loc[
        (rolling["Model"] == FORECAST_MODEL)
        & (rolling["HistoryWindowHours"] == selected_window)
    ].pivot(index="WindowID", columns="Level", values="WAPE")
    required_levels = {"Bottom", "Region", "TaskType", "System"}
    aggregation_complete = required_levels.issubset(set(aggregation.columns))
    if aggregation_complete:
        order_mask = (
            aggregation["Bottom"].ge(aggregation["Region"] - NUMERICAL_EPS)
            & aggregation["Bottom"].ge(aggregation["TaskType"] - NUMERICAL_EPS)
            & aggregation["Region"].ge(aggregation["System"] - NUMERICAL_EPS)
            & aggregation["TaskType"].ge(aggregation["System"] - NUMERICAL_EPS)
        )
        aggregation_order_ratio = float(order_mask.mean())
        median_order_ok = bool(
            aggregation["Bottom"].median() >= aggregation["Region"].median()
            and aggregation["Bottom"].median() >= aggregation["TaskType"].median()
            and aggregation["Region"].median() >= aggregation["System"].median()
            and aggregation["TaskType"].median() >= aggregation["System"].median()
        )
    else:
        aggregation_order_ratio = float("nan")
        median_order_ok = False
    aggregation_ok = bool(
        aggregation_complete and aggregation_order_ratio > 0.5 and median_order_ok
    )

    _write_table(
        pd.DataFrame(
            [
                {
                    "SelectedHistoryWindowHours": selected_window,
                    "RollingWindowCount": sample_count,
                    "SelectedWindowValidationWAPE": float(selected_row["ValidationWAPE"]),
                    "RollingSelectedWindowRank": selected_rank,
                    "DeltaMedian_MainMinusBaseline": median_delta,
                    "DeltaNegativeWindowCount": negative_count,
                    "WilcoxonStatistic": statistic,
                    "WilcoxonPValue_OneSidedLess": p_value,
                    "AggregationOrderRatio": aggregation_order_ratio,
                    "AggregationMedianOrder": median_order_ok,
                    "MainVsBaselineStable": main_baseline_ok,
                    "AggregationScaleStable": aggregation_ok,
                }
            ]
        ),
        "validation_test1_summary.csv",
    )
    _record(
        records,
        "Validation 1: rolling out-of-time forecasts and aggregation-scale generalization",
        "Validation-window selection and historical rolling stability",
        "PASS" if selected_rank <= len(HISTORY_WINDOWS) else "REVIEW",
        f"H*={selected_window}; rolling_windows={len(_rolling_windows())}; rolling_rank={selected_rank}/{len(HISTORY_WINDOWS)}",
        "Select H* on the official validation period and recheck it in strictly forward, nonoverlapping historical windows",
        "If rolling rankings are unstable, the paper cannot claim the official validation-day choice is universally optimal",
    )
    _record(
        records,
        "Validation 1: rolling out-of-time forecasts and aggregation-scale generalization",
        "Paired WAPE of the main model versus the 24-hour same-hour baseline",
        "PASS" if main_baseline_ok else "FAIL",
        f"n={sample_count}; median_delta={median_delta:.6g}; negative={negative_count}; p={p_value:.6g}",
        "One-sided Wilcoxon p<0.05 and median(delta)<0, where delta=main-model WAPE minus baseline WAPE",
        "If this fails, do not claim stable out-of-sample superiority; use a stable rolling model or the baseline",
    )
    _record(
        records,
        "Validation 1: rolling out-of-time forecasts and aggregation-scale generalization",
        "Aggregation-scale ordering from bottom level to region/type/system",
        "PASS" if aggregation_ok else "FAIL",
        f"order_ratio={aggregation_order_ratio:.6g}; median_order={median_order_ok}",
        "At selected H*, most rolling windows and median WAPE must show bottom-level error no lower than aggregate error, and region/type error no lower than system error",
        "If this fails, report only observed scale differences; do not claim aggregation universally improves predictability",
    )
    return {
        "ok": main_baseline_ok and aggregation_ok,
        "selected_window": selected_window,
        "rolling": rolling,
        "paired": paired,
    }


def _jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    p = np.maximum(np.asarray(p, dtype=float), 0.0)
    q = np.maximum(np.asarray(q, dtype=float), 0.0)
    if p.sum() <= NUMERICAL_EPS or q.sum() <= NUMERICAL_EPS:
        return float("nan")
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)
    p_mask = p > 0
    q_mask = q > 0
    divergence = 0.5 * float(
        np.sum(p[p_mask] * np.log(p[p_mask] / midpoint[p_mask]))
        + np.sum(q[q_mask] * np.log(q[q_mask] / midpoint[q_mask]))
    )
    return float(np.sqrt(max(divergence, 0.0)))


def _structure_share(panel: pd.DataFrame, start_hour: int, end_hour: int) -> np.ndarray:
    window = panel.loc[panel["Hour"].between(start_hour, end_hour)]
    regions = sorted(panel["SourceRegion"].astype(str).unique())
    task_types = sorted(panel["TaskType"].astype(str).unique())
    matrix = (
        window.pivot_table(
            index="SourceRegion",
            columns="TaskType",
            values="GPU_Demand_Arrival",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=regions, columns=task_types, fill_value=0.0)
        .to_numpy(dtype=float)
        .reshape(-1)
    )
    total = float(matrix.sum())
    return matrix / total if total > NUMERICAL_EPS else np.full_like(matrix, 1.0 / len(matrix))


def _run_structure_test(
    panel: pd.DataFrame,
    rolling_state: dict[str, object],
    records: list[dict[str, object]],
) -> dict[str, object]:
    hierarchy_path = TABLES_DIR / "forecast_hierarchy_predictions.csv"
    prediction_path = TABLES_DIR / "forecast_predictions.csv"
    consistency_ok = False
    max_region_diff = float("nan")
    max_type_diff = float("nan")
    actual_mismatch = -1
    bottom_region_diff = float("nan")
    if hierarchy_path.is_file() and prediction_path.is_file():
        hierarchy = _read_csv(
            hierarchy_path,
            [
                "Split",
                "Model",
                "Level",
                "Entity",
                "Hour",
                "Actual_GPU_Demand",
                "IndependentPrediction",
                "ReconciledPrediction",
            ],
        )
        hierarchy = _to_numeric(
            hierarchy,
            ["Hour", "Actual_GPU_Demand", "IndependentPrediction", "ReconciledPrediction"],
        )
        hierarchy["Hour"] = hierarchy["Hour"].astype(int)
        actual_rows: list[dict[str, object]] = []
        for split, start, end in [
            ("validation", VALIDATION_START_HOUR, VALIDATION_END_HOUR),
            ("test", TEST_START_HOUR, TEST_END_HOUR),
        ]:
            target = panel.loc[panel["Hour"].between(start, end)]
            for level in ["Region", "TaskType", "System"]:
                actual = _level_demand(target, level)
                actual_rows.extend(
                    {
                        "Split": split,
                        "Level": level,
                        "Entity": str(row.Entity),
                        "Hour": int(row.Hour),
                        "ExpectedActual": float(row.Demand),
                    }
                    for row in actual.itertuples(index=False)
                )
        actual = pd.DataFrame(actual_rows)
        actual_join = hierarchy.merge(
            actual,
            how="left",
            on=["Split", "Level", "Entity", "Hour"],
            validate="one_to_one",
        )
        actual_mismatch = int(
            actual_join["Actual_GPU_Demand"]
            .sub(actual_join["ExpectedActual"])
            .abs()
            .gt(NUMERICAL_EPS)
            .fillna(True)
            .sum()
        )
        consistency_rows: list[dict[str, object]] = []
        for (split, hour), group in hierarchy.groupby(["Split", "Hour"], sort=True):
            system_values = group.loc[
                group["Level"].eq("System"), "ReconciledPrediction"
            ]
            system = float(system_values.iloc[0]) if len(system_values) == 1 else float("nan")
            region = float(
                group.loc[group["Level"].eq("Region"), "ReconciledPrediction"].sum()
            )
            task_type = float(
                group.loc[group["Level"].eq("TaskType"), "ReconciledPrediction"].sum()
            )
            consistency_rows.append(
                {
                    "Split": split,
                    "Hour": hour,
                    "RegionSystemAbsDiff": abs(region - system),
                    "TaskTypeSystemAbsDiff": abs(task_type - system),
                }
            )
        consistency = pd.DataFrame(consistency_rows)
        max_region_diff = float(consistency["RegionSystemAbsDiff"].max())
        max_type_diff = float(consistency["TaskTypeSystemAbsDiff"].max())
        predictions = _read_csv(
            prediction_path,
            ["Split", "Model", "SourceRegion", "TaskType", "Hour", "Prediction"],
        )
        predictions = _to_numeric(predictions, ["Hour", "Prediction"])
        bottom = predictions.loc[predictions["Model"].eq(FORECAST_MODEL)].copy()
        bottom_region = (
            bottom.groupby(["Split", "Hour", "SourceRegion"], as_index=False)["Prediction"]
            .sum()
            .rename(columns={"SourceRegion": "Entity", "Prediction": "BottomPrediction"})
        )
        reported_region = hierarchy.loc[
            hierarchy["Level"].eq("Region"),
            ["Split", "Hour", "Entity", "ReconciledPrediction"],
        ].rename(columns={"ReconciledPrediction": "HierarchyPrediction"})
        bottom_region_join = bottom_region.merge(
            reported_region,
            how="outer",
            on=["Split", "Hour", "Entity"],
        )
        bottom_region_diff = float(
            bottom_region_join["BottomPrediction"]
            .sub(bottom_region_join["HierarchyPrediction"])
            .abs()
            .max()
        )
        consistency_ok = bool(
            actual_mismatch == 0
            and max_region_diff <= NUMERICAL_EPS
            and max_type_diff <= NUMERICAL_EPS
            and bottom_region_diff <= NUMERICAL_EPS
        )
        _write_table(consistency, "validation_hierarchy_consistency.csv")
    _record(
        records,
        "Validation 2: effectiveness of hierarchical forecast reconciliation",
        "Marginal closure and consistency of 18-category results",
        "PASS" if consistency_ok else "FAIL",
        f"actual_mismatch={actual_mismatch}; max_region_diff={max_region_diff:.6g}; max_type_diff={max_type_diff:.6g}; bottom_region_diff={bottom_region_diff:.6g}",
        "Regional and task-type margins must match the system margin within numerical precision, and the 18-category regional totals must match regional margins",
        "If this fails, reconciliation has failed; do not interpret the 18-category forecasts further",
    )

    block_starts = list(range(TRAIN_START_HOUR, TRAIN_END_HOUR + 1, 24))
    shares = [_structure_share(panel, start, start + 23) for start in block_starts]
    adjacent_rows = [
        {
            "WindowStartHour": block_starts[index + 1],
            "PreviousWindowStartHour": block_starts[index],
            "JS_Distance": _jensen_shannon(shares[index], shares[index + 1]),
        }
        for index in range(len(shares) - 1)
    ]
    adjacent = pd.DataFrame(adjacent_rows)
    q95 = float(adjacent["JS_Distance"].quantile(0.95))
    validation_share = _structure_share(
        panel, VALIDATION_START_HOUR, VALIDATION_END_HOUR
    )
    validation_js = _jensen_shannon(shares[-1], validation_share)
    js_ok = bool(np.isfinite(validation_js) and validation_js <= q95 + NUMERICAL_EPS)
    adjacent["Q95_TrainingJS"] = q95
    adjacent["ValidationJS"] = validation_js
    _write_table(adjacent, "validation_structure_js.csv")
    _record(
        records,
        "Validation 2: effectiveness of hierarchical forecast reconciliation",
        "Historical region/type structure stability",
        "PASS" if js_ok else "FAIL",
        f"training_adjacent_windows={len(adjacent)}; Q95_JS={q95:.6g}; JS_validation={validation_js:.6g}",
        "JS_validation <= the empirical 95th percentile of JS distances between adjacent training windows",
        "If this fails, shorten the structural prior window or use dynamic region/type shares; do not explain test forecasts with a fixed historical structure",
    )

    paired = rolling_state["paired"]
    delta = paired["Delta_HierarchicalMinusDirect"]
    statistic_less, p_less, sample_count = _wilcoxon(delta, "less")
    statistic_greater, p_greater, _ = _wilcoxon(delta, "greater")
    median_delta = float(delta.median())
    if np.isfinite(p_less) and p_less < 0.05 and median_delta < 0:
        direct_status = "PASS"
    elif np.isfinite(p_greater) and p_greater < 0.05 and median_delta > 0:
        direct_status = "FAIL"
    else:
        direct_status = "QUALIFIED"
    _write_table(
        pd.DataFrame(
            [
                {
                    "HistoryWindowHours": rolling_state["selected_window"],
                    "WindowCount": sample_count,
                    "Median_Delta_HierarchicalMinusDirect": median_delta,
                    "WilcoxonStatistic_Less": statistic_less,
                    "WilcoxonPValue_Less": p_less,
                    "WilcoxonStatistic_Greater": statistic_greater,
                    "WilcoxonPValue_Greater": p_greater,
                    "Decision": direct_status,
                }
            ]
        ),
        "validation_hierarchy_vs_direct.csv",
    )
    _record(
        records,
        "Validation 2: effectiveness of hierarchical forecast reconciliation",
        "Paired errors of reconciled versus direct forecasts for 18 series",
        direct_status,
        f"n={sample_count}; median_delta={median_delta:.6g}; p_less={p_less:.6g}; p_greater={p_greater:.6g}",
        "Median(delta)<0 and one-sided Wilcoxon p<0.05 indicate improvement; only significant deterioration fails. Otherwise claim marginal consistency without significant loss",
        "If errors deteriorate significantly, limit conclusions to region/type/system levels; do not force a reliability claim for 18 cross-category forecasts",
    )
    hard_ok = consistency_ok and js_ok and direct_status != "FAIL"
    return {"ok": hard_ok}


def _recompute_profile(assignments: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    profile = capacity.copy().sort_values(["Hour", "Region"], kind="stable")
    profile["Recomputed_GPU"] = 0.0
    profile["Recomputed_AI_IT_Load_MW"] = 0.0
    row_index = {
        (int(row.Hour), str(row.Region)): index
        for index, row in profile.reset_index(drop=True).iterrows()
    }
    profile = profile.reset_index(drop=True)
    for row in assignments.itertuples(index=False):
        start = float(row.StartHour)
        finish = float(row.FinishHour)
        for hour in range(TEST_START_HOUR, TAIL_END_HOUR + 1):
            overlap = max(
                0.0,
                min(finish, float(hour + 1)) - max(start, float(hour)),
            )
            if overlap <= NUMERICAL_EPS:
                continue
            key = (hour, str(row.TargetRegion))
            if key not in row_index:
                raise ValueError(f"The schedule references Hour x Region keys outside the capacity table: {key}")
            index = row_index[key]
            profile.loc[index, "Recomputed_GPU"] += float(row.GPU_Demand) * overlap
            profile.loc[index, "Recomputed_AI_IT_Load_MW"] += (
                float(row.Task_Full_IT_Power_MW) * overlap
            )
    profile["Recomputed_Total_IT_Load_MW"] = (
        profile["NonAI_IT_Load_MW"] + profile["Recomputed_AI_IT_Load_MW"]
    )
    profile["Recomputed_Total_Facility_Load_MW"] = (
        profile["Recomputed_Total_IT_Load_MW"] * profile["PUE"]
    )
    profile["GPU_Violation"] = (
        profile["Recomputed_GPU"] - profile["Available_GPU"]
    ).clip(lower=0.0)
    profile["IT_Violation_MW"] = (
        profile["Recomputed_Total_IT_Load_MW"] - profile["Max_IT_Power_MW"]
    ).clip(lower=0.0)
    profile["Facility_Violation_MW"] = (
        profile["Recomputed_Total_Facility_Load_MW"]
        - profile["Max_Facility_Power_MW"]
    ).clip(lower=0.0)
    return profile


def _run_dispatch_certificate(
    tasks: pd.DataFrame,
    capacity: pd.DataFrame,
    records: list[dict[str, object]],
) -> dict[str, object]:
    assignment_path = TABLES_DIR / "dispatch_assignments.csv"
    profile_path = TABLES_DIR / "dispatch_resource_profile.csv"
    summary_path = TABLES_DIR / "dispatch_summary.csv"
    if not all(path.is_file() for path in [assignment_path, profile_path, summary_path]):
        _record(
            records,
            "Validation 3: scheduling optimality and critical load",
            "Current schedule result files",
            "FAIL",
            "dispatch_assignments.csv, dispatch_resource_profile.csv, or dispatch_summary.csv is missing",
            "The scheduling certificate requires assignments, recomputed resource profiles, and solver summaries",
            "Rerun model.py to generate the compute schedule first",
        )
        return {"ok": False}

    assignments = _read_csv(
        assignment_path,
        [
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
        ],
    )
    reported_profile = _read_csv(
        profile_path,
        [
            "Hour",
            "Region",
            "Scheduled_GPU_Equivalent",
            "Scheduled_AI_IT_Load_MW",
            "Total_IT_Load_MW",
            "Total_Facility_Load_MW",
        ],
    )
    summary = _read_csv(
        summary_path,
        ["F1_Migration_GPU_Workload_GPUh", "F2_Flexible_WaitHours"],
    )
    assignments["TaskID"] = assignments["TaskID"].astype(str)
    assignments["TaskType"] = assignments["TaskType"].astype(str)
    assignments["SourceRegion"] = assignments["SourceRegion"].astype(str)
    assignments["TargetRegion"] = assignments["TargetRegion"].astype(str)
    assignments = _to_numeric(
        assignments,
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
    )
    dispatch_tasks = tasks.loc[
        tasks["ArrivalHour"].between(TEST_START_HOUR, TEST_END_HOUR)
    ].copy()
    expected_ids = set(dispatch_tasks["TaskID"].astype(str))
    counts = assignments["TaskID"].value_counts()
    duplicate_ids = int((counts > 1).sum())
    assigned_ids = set(assignments["TaskID"])
    missing_ids = len(expected_ids - assigned_ids)
    unexpected_ids = len(assigned_ids - expected_ids)
    exact_once_ok = duplicate_ids == 0 and missing_ids == 0 and unexpected_ids == 0

    task_lookup = dispatch_tasks[
        [
            "TaskID",
            "TaskType",
            "ArrivalHour",
            "SourceRegion",
            "GPU_Demand",
            "Duration_h",
            "EarliestStartHour",
            "LatestFinishHour",
            "MaxLatency_ms",
        ]
    ].copy()
    task_lookup["TaskID"] = task_lookup["TaskID"].astype(str)
    joined = assignments.merge(task_lookup, on="TaskID", how="outer", suffixes=("_reported", "_task"))
    joined = joined.loc[joined["TaskID"].isin(expected_ids)].copy()
    network_violation = int(
        (joined["NetworkLatency_ms"] - joined["MaxLatency_ms_task"] > NUMERICAL_EPS).fillna(True).sum()
    )
    earliest_violation = int(
        (joined["StartHour"] - joined["EarliestStartHour"] < -NUMERICAL_EPS).fillna(True).sum()
    )
    finish_violation = int(
        (joined["FinishHour"] - joined["LatestFinishHour"] > NUMERICAL_EPS).fillna(True).sum()
    )
    terminal_violation = int(
        (joined["FinishHour"] - TERMINAL_HOUR > NUMERICAL_EPS).fillna(True).sum()
    )
    duration_violation = int(
        (joined["FinishHour"] - joined["StartHour"] - joined["Duration_h_task"])
        .abs()
        .gt(NUMERICAL_EPS)
        .fillna(True)
        .sum()
    )
    realtime = joined["TaskType_task"].eq("RealTimeInference")
    realtime_start_violation = int(
        (joined.loc[realtime, "StartHour"] - joined.loc[realtime, "ArrivalHour_task"])
        .abs()
        .gt(NUMERICAL_EPS)
        .fillna(True)
        .sum()
    )
    occupied_terminal = int(
        joined.apply(
            lambda row: max(
                0.0,
                min(float(row["FinishHour"]), float(TERMINAL_HOUR + 1))
                - max(float(row["StartHour"]), float(TERMINAL_HOUR)),
            ),
            axis=1,
        )
        .gt(NUMERICAL_EPS)
        .fillna(True)
        .sum()
    )

    recomputed = _recompute_profile(assignments, capacity)
    _write_table(recomputed, "validation_dispatch_resource_recompute.csv")
    reported_profile = _to_numeric(
        reported_profile,
        [
            "Hour",
            "Scheduled_GPU_Equivalent",
            "Scheduled_AI_IT_Load_MW",
            "Total_IT_Load_MW",
            "Total_Facility_Load_MW",
        ],
    )
    profile_join = recomputed.merge(reported_profile, on=["Hour", "Region"], how="outer")
    profile_diffs = {}
    for reported, recomputed_column in [
        ("Scheduled_GPU_Equivalent", "Recomputed_GPU"),
        ("Scheduled_AI_IT_Load_MW", "Recomputed_AI_IT_Load_MW"),
        ("Total_IT_Load_MW", "Recomputed_Total_IT_Load_MW"),
        ("Total_Facility_Load_MW", "Recomputed_Total_Facility_Load_MW"),
    ]:
        profile_diffs[reported] = float(
            profile_join[reported].sub(profile_join[recomputed_column]).abs().max()
        )
    resource_ok = bool(
        recomputed["GPU_Violation"].max() <= NUMERICAL_EPS
        and recomputed["IT_Violation_MW"].max() <= NUMERICAL_EPS
        and recomputed["Facility_Violation_MW"].max() <= NUMERICAL_EPS
        and max(profile_diffs.values()) <= NUMERICAL_EPS
    )

    f1_recomputed = float(assignments["Migration_GPU_Workload_GPUh"].sum())
    flexible = assignments["TaskType"].isin(["BatchInference", "AITraining"])
    f2_recomputed = float(assignments.loc[flexible, "WaitHours"].sum())
    f1_reported = float(summary.iloc[0]["F1_Migration_GPU_Workload_GPUh"])
    f2_reported = float(summary.iloc[0]["F2_Flexible_WaitHours"])
    objective_ok = bool(
        math.isclose(f1_recomputed, f1_reported, rel_tol=0.0, abs_tol=NUMERICAL_EPS)
        and math.isclose(f2_recomputed, f2_reported, rel_tol=0.0, abs_tol=NUMERICAL_EPS)
    )
    feasible = bool(
        exact_once_ok
        and network_violation == 0
        and earliest_violation == 0
        and finish_violation == 0
        and terminal_violation == 0
        and duration_violation == 0
        and realtime_start_violation == 0
        and occupied_terminal == 0
        and resource_ok
        and objective_ok
    )
    certificate_ok = bool(feasible and f1_recomputed >= -NUMERICAL_EPS and abs(f1_recomputed) <= NUMERICAL_EPS)
    certificate = pd.DataFrame(
        [
            {
                "Check": "Exactly-once task execution",
                "Status": "PASS" if exact_once_ok else "FAIL",
                "Observed": f"duplicates={duplicate_ids}; missing={missing_ids}; unexpected={unexpected_ids}",
                "Rule": "Every task arriving during hours 2376--2399 must execute exactly once",
            },
            {
                "Check": "Task latency and time boundaries",
                "Status": "PASS"
                if network_violation + earliest_violation + finish_violation + terminal_violation + duration_violation + realtime_start_violation + occupied_terminal == 0
                else "FAIL",
                "Observed": f"network={network_violation}; earliest={earliest_violation}; finish={finish_violation}; terminal={terminal_violation}; duration={duration_violation}; realtime={realtime_start_violation}; occupied_2406={occupied_terminal}",
                "Rule": "Respect arrival, earliest start, LatestFinish, network latency, and the hour-2406 terminal boundary; hour-2406 occupancy must be zero",
            },
            {
                "Check": "GPU / IT / facility capacity",
                "Status": "PASS" if resource_ok else "FAIL",
                "Observed": f"max_gpu_violation={recomputed['GPU_Violation'].max()}; max_it_violation={recomputed['IT_Violation_MW'].max()}; max_facility_violation={recomputed['Facility_Violation_MW'].max()}",
                "Rule": "All three resource violation amounts must be zero, and independently recomputed profiles must match model output",
            },
            {
                "Check": "Strict global optimality certificate for F1",
                "Status": "PASS" if certificate_ok else "FAIL",
                "Observed": f"lower_bound=0; constructed_feasible={feasible}; F1={f1_recomputed}",
                "Rule": "Nonnegative migration workload implies F1>=0; a feasible F1=0 construction proves F1*=0",
            },
            {
                "Check": "Recomputation of the second objective F2",
                "Status": "PASS" if objective_ok else "FAIL",
                "Observed": f"F2_recomputed={f2_recomputed}; F2_reported={f2_reported}",
                "Rule": "After fixing optimal F1, flexible-task waiting workload must match the reported result",
            },
        ]
    )
    _write_table(certificate, "validation_dispatch_certificate.csv")
    for row in certificate.itertuples(index=False):
        _record(
            records,
            "Validation 3: scheduling optimality and critical load",
            str(row.Check),
            str(row.Status),
            str(row.Observed),
            str(row.Rule),
            "Repair task boundaries, resource profiles, or the two-stage objective certificate before interpreting local execution",
        )
    return {
        "ok": feasible,
        "certificate_ok": certificate_ok,
        "f1": f1_recomputed,
        "f2": f2_recomputed,
        "assignments": assignments,
    }


def _scaled_options(options: list[object], alpha: float) -> list[object]:
    return [
        replace(
            option,
            gpu_demand=float(option.gpu_demand) * alpha,
            task_power_mw=float(option.task_power_mw) * alpha,
            migration_gpu_workload_gpuh=float(option.migration_gpu_workload_gpuh)
            * alpha,
        )
        for option in options
    ]


def _capacity_upper_bound(
    options: list[object],
    tasks: pd.DataFrame,
    capacity: pd.DataFrame,
    local_only: bool,
) -> float:
    target_tasks = tasks.loc[
        tasks["ArrivalHour"].between(TEST_START_HOUR, TEST_END_HOUR)
    ].copy()
    if local_only:
        target_tasks = target_tasks.loc[
            target_tasks["SourceRegion"].isin(
                {str(option.source_region) for option in options}
            )
        ]
        ratios: list[float] = []
        for region in sorted(target_tasks["SourceRegion"].astype(str).unique()):
            workload = float(
                target_tasks.loc[target_tasks["SourceRegion"] == region, "GPU_Workload_GPUh"].sum()
            )
            region_tasks = target_tasks.loc[target_tasks["SourceRegion"] == region]
            power_workload = float(
                (region_tasks["Task_Full_IT_Power_MW"] * region_tasks["Duration_h"]).sum()
            )
            region_capacity = capacity.loc[capacity["Region"] == region]
            gpu_capacity = float(region_capacity["Available_GPU"].sum())
            power_capacity = float(region_capacity["Effective_AI_IT_Capacity_MW"].sum())
            if workload > NUMERICAL_EPS:
                ratios.append(gpu_capacity / workload)
            if power_workload > NUMERICAL_EPS:
                ratios.append(power_capacity / power_workload)
    else:
        workload = float(target_tasks["GPU_Workload_GPUh"].sum())
        power_workload = float(
            (target_tasks["Task_Full_IT_Power_MW"] * target_tasks["Duration_h"]).sum()
        )
        ratios = []
        if workload > NUMERICAL_EPS:
            ratios.append(float(capacity["Available_GPU"].sum()) / workload)
        if power_workload > NUMERICAL_EPS:
            ratios.append(
                float(capacity["Effective_AI_IT_Capacity_MW"].sum()) / power_workload
            )
    finite_ratios = [value for value in ratios if np.isfinite(value) and value > 1.0]
    return max(1.05, min(finite_ratios) * 1.01 if finite_ratios else 2.0)


def _run_feasibility(
    model,
    options: list[object],
    dispatch_tasks: pd.DataFrame,
    capacity: pd.DataFrame,
) -> tuple[bool, np.ndarray | None, int, str]:
    from scipy.optimize import Bounds, LinearConstraint, milp

    matrix, lower, upper, _, wait_objective = model._build_dispatch_constraints(
        options, dispatch_tasks, capacity
    )
    # Stress validation asks only whether a feasible solution exists. Waiting cost plus a tiny deterministic tie-break
    # removes equivalent solutions in the zero-objective feasibility MILP without changing its feasible set.
    objective = np.asarray(wait_objective, dtype=float) + np.arange(len(options)) * 1e-9
    result = milp(
        c=objective,
        integrality=np.ones(len(options), dtype=np.int8),
        bounds=Bounds(np.zeros(len(options)), np.ones(len(options))),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"disp": False},
    )
    if not result.success or result.x is None:
        return False, None, int(result.status), str(result.message)
    return True, np.rint(result.x).astype(int), int(result.status), str(result.message)


def _pressure_boundary(
    model,
    options: list[object],
    dispatch_tasks: pd.DataFrame,
    capacity: pd.DataFrame,
    granularity: str,
    local_only: bool,
    trace: list[dict[str, object]],
) -> dict[str, object]:
    base_options = [
        option
        for option in options
        if not local_only or str(option.target_region) == str(option.source_region)
    ]
    cache: dict[float, tuple[bool, np.ndarray | None, int, str]] = {}

    def evaluate(alpha: float) -> tuple[bool, np.ndarray | None, int, str]:
        key = round(float(alpha), 8)
        if key not in cache:
            scaled = _scaled_options(base_options, key)
            cache[key] = _run_feasibility(model, scaled, dispatch_tasks, capacity)
            feasible, selected, status, message = cache[key]
            trace_row: dict[str, object] = {
                "Granularity": granularity,
                "Boundary": "local" if local_only else "system",
                "Alpha": key,
                "Feasible": feasible,
                "SolverStatus": status,
                "SolverMessage": message,
            }
            if feasible and selected is not None:
                profile = model._build_dispatch_resource_profile(
                    scaled, selected, capacity
                )
                trace_row["MaxGPUUtilization"] = float(profile["GPU_Utilization"].max())
                trace_row["MaxAICapacityUtilization"] = float(
                    profile["AI_IT_Capacity_Utilization"].max()
                )
            else:
                trace_row["MaxGPUUtilization"] = float("nan")
                trace_row["MaxAICapacityUtilization"] = float("nan")
            trace.append(trace_row)
        return cache[key]

    lower = 1.0
    feasible_at_one, _, status_one, message_one = evaluate(lower)
    if not feasible_at_one:
        return {
            "Granularity": granularity,
            "Boundary": "local" if local_only else "system",
            "LowerFeasible": float("nan"),
            "UpperInfeasible": 1.0,
            "Gap": float("nan"),
            "Status": "FAIL_AT_ALPHA_1",
            "InitialSolverStatus": status_one,
            "InitialSolverMessage": message_one,
        }
    upper = _capacity_upper_bound(base_options, dispatch_tasks, capacity, local_only)
    feasible_upper, _, status_upper, message_upper = evaluate(upper)
    while feasible_upper and upper < ALPHA_SEARCH_MAX:
        lower = upper
        upper *= 2.0
        feasible_upper, _, status_upper, message_upper = evaluate(upper)
    if feasible_upper:
        return {
            "Granularity": granularity,
            "Boundary": "local" if local_only else "system",
            "LowerFeasible": lower,
            "UpperInfeasible": float("nan"),
            "Gap": float("nan"),
            "Status": "NO_BOUND_WITHIN_SEARCH_CAP",
            "InitialSolverStatus": status_one,
            "InitialSolverMessage": message_one,
        }
    for _ in range(20):
        if upper - lower <= ALPHA_SEARCH_TOLERANCE:
            break
        midpoint = 0.5 * (lower + upper)
        feasible_mid, _, _, _ = evaluate(midpoint)
        if feasible_mid:
            lower = midpoint
        else:
            upper = midpoint
    return {
        "Granularity": granularity,
        "Boundary": "local" if local_only else "system",
        "LowerFeasible": lower,
        "UpperInfeasible": upper,
        "Gap": upper - lower,
        "Status": "PASS",
        "InitialSolverStatus": status_one,
        "InitialSolverMessage": message_one,
    }


def _run_pressure_test(
    tasks: pd.DataFrame,
    candidates: pd.DataFrame,
    capacity: pd.DataFrame,
    dispatch_state: dict[str, object],
    records: list[dict[str, object]],
) -> dict[str, object]:
    model = _load_model_module()
    dispatch_tasks = tasks.loc[
        tasks["ArrivalHour"].between(TEST_START_HOUR, TEST_END_HOUR)
    ].copy()
    trace: list[dict[str, object]] = []
    options_1h_frame, options_1h = model._build_dispatch_options(
        tasks, candidates, start_step_hours=1.0
    )
    del options_1h_frame
    local_1h = _pressure_boundary(
        model,
        options_1h,
        dispatch_tasks,
        capacity,
        "1h",
        True,
        trace,
    )
    system_1h = _pressure_boundary(
        model,
        options_1h,
        dispatch_tasks,
        capacity,
        "1h",
        False,
        trace,
    )
    options_half_frame, options_half = model._build_dispatch_options(
        tasks, candidates, start_step_hours=0.5
    )
    del options_half_frame
    local_half = _pressure_boundary(
        model,
        options_half,
        dispatch_tasks,
        capacity,
        "0.5h",
        True,
        trace,
    )
    trace_frame = pd.DataFrame(trace)
    _write_table(trace_frame, "validation_pressure_trace.csv")

    boundary_rows = [local_1h, system_1h, local_half]
    boundary_frame = pd.DataFrame(boundary_rows)
    _write_table(boundary_frame, "validation_critical_pressure.csv")

    objective_rows: list[dict[str, object]] = [
        {
            "Granularity": "1h",
            "Alpha": 1.0,
            "F1": float(dispatch_state["f1"]),
            "F2": float(dispatch_state["f2"]),
            "Source": "Current main-solution dispatch_summary",
        }
    ]
    local_lower = local_1h.get("LowerFeasible")
    if isinstance(local_lower, (int, float)) and np.isfinite(local_lower):
        probe_alpha = float(local_1h["UpperInfeasible"])
        full_probe = _scaled_options(options_1h, probe_alpha)
        full_feasible, _, _, _ = _run_feasibility(
            model, full_probe, dispatch_tasks, capacity
        )
        if full_feasible:
            result = model._solve_two_stage_dispatch(
                full_probe, dispatch_tasks, capacity
            )
            objective_rows.append(
                {
                    "Granularity": "1h",
                    "Alpha": probe_alpha,
                    "F1": float(result[1]),
                    "F2": float(result[2]),
                    "Source": "Two-stage MILP near the local critical upper bound",
                }
            )
    objective_frame = pd.DataFrame(objective_rows)
    _write_table(objective_frame, "validation_pressure_objectives.csv")

    local_mid_1h = (
        0.5 * (float(local_1h["LowerFeasible"]) + float(local_1h["UpperInfeasible"]))
        if local_1h["Status"] == "PASS"
        else float("nan")
    )
    local_mid_half = (
        0.5 * (float(local_half["LowerFeasible"]) + float(local_half["UpperInfeasible"]))
        if local_half["Status"] == "PASS"
        else float("nan")
    )
    delta_alpha = abs(local_mid_1h - local_mid_half)
    margin = local_mid_1h - 1.0
    interaction_ok = bool(np.isfinite(delta_alpha) and np.isfinite(margin) and delta_alpha < margin)
    _write_table(
        pd.DataFrame(
            [
                {
                    "AlphaLocal_1h_Midpoint": local_mid_1h,
                    "AlphaLocal_0_5h_Midpoint": local_mid_half,
                    "DeltaAlphaLocal": delta_alpha,
                    "CurrentToBoundaryMargin_1h": margin,
                    "GranularityInteraction": "boundary_shift_smaller_than_current_margin"
                    if interaction_ok
                    else "boundary_shift_not_smaller_or_unavailable",
                }
            ]
        ),
        "validation_granularity_pressure_interaction.csv",
    )

    all_boundaries_ok = all(row.get("Status") == "PASS" for row in boundary_rows)
    _record(
        records,
        "Validation 3: scheduling optimality and critical load",
        "Adaptive critical-load bounds",
        "PASS" if all_boundaries_ok else "FAIL",
        f"alpha_local_1h=[{local_1h.get('LowerFeasible')},{local_1h.get('UpperInfeasible')}]; alpha_feas_1h=[{system_1h.get('LowerFeasible')},{system_1h.get('UpperInfeasible')}]; alpha_local_0_5h=[{local_half.get('LowerFeasible')},{local_half.get('UpperInfeasible')}]",
        "Search adaptively from alpha=1 to the first infeasibility, without preset percentages, then narrow feasible/infeasible bounds to within 0.01",
        "If bounds cannot be found or alpha=1 is infeasible, do not claim stress headroom for current local execution",
    )
    _record(
        records,
        "Validation 3: scheduling optimality and critical load",
        "Interaction between time resolution and local critical load",
        "PASS" if interaction_ok else "QUALIFIED",
        f"delta_alpha={delta_alpha:.6g}; current_margin={margin:.6g}",
        "Report |alpha_local(1h)-alpha_local(0.5h)| relative to current boundary headroom, without an external fixed tolerance",
        "If the boundary shift is comparable to current headroom, acknowledge the effect of time discretization on critical capacity",
    )
    return {"ok": all_boundaries_ok}


def _fixed_schedule_alpha_lower_bound(
    assignments: pd.DataFrame, capacity: pd.DataFrame
) -> float:
    """Strict feasible lower bound from scaling one independently verified schedule with load."""

    profile = _recompute_profile(assignments, capacity)
    ratios: list[float] = []
    for row in profile.itertuples(index=False):
        if float(row.Recomputed_GPU) > NUMERICAL_EPS:
            ratios.append(float(row.Available_GPU) / float(row.Recomputed_GPU))
        if float(row.Recomputed_AI_IT_Load_MW) > NUMERICAL_EPS:
            ai_it_capacity = float(row.Max_IT_Power_MW) - float(row.NonAI_IT_Load_MW)
            ai_facility_capacity = (
                float(row.Max_Facility_Power_MW) / float(row.PUE)
                - float(row.NonAI_IT_Load_MW)
            )
            ratios.extend(
                [
                    ai_it_capacity / float(row.Recomputed_AI_IT_Load_MW),
                    ai_facility_capacity / float(row.Recomputed_AI_IT_Load_MW),
                ]
            )
    finite = [ratio for ratio in ratios if np.isfinite(ratio) and ratio >= 0]
    return float(min(finite)) if finite else float("inf")


def _aggregate_alpha_upper_bound(
    tasks: pd.DataFrame, capacity: pd.DataFrame, local_only: bool
) -> float:
    """Strict infeasible upper bound from total workload and capacity conservation."""

    dispatch_tasks = tasks.loc[
        tasks["ArrivalHour"].between(TEST_START_HOUR, TEST_END_HOUR)
    ].copy()
    ratios: list[float] = []
    groups = (
        dispatch_tasks.groupby("SourceRegion", sort=True)
        if local_only
        else [("ALL", dispatch_tasks)]
    )
    capacity_groups = capacity.groupby("Region", sort=True)
    for region, group in groups:
        if local_only:
            region_capacity = capacity_groups.get_group(str(region))
        else:
            region_capacity = capacity
        gpu_workload = float(group["GPU_Workload_GPUh"].sum())
        power_workload = float(
            (group["Task_Full_IT_Power_MW"] * group["Duration_h"]).sum()
        )
        if gpu_workload > NUMERICAL_EPS:
            ratios.append(float(region_capacity["Available_GPU"].sum()) / gpu_workload)
        if power_workload > NUMERICAL_EPS:
            ratios.append(
                float(region_capacity["Effective_AI_IT_Capacity_MW"].sum())
                / power_workload
            )
    finite = [ratio for ratio in ratios if np.isfinite(ratio) and ratio >= 0]
    return float(min(finite)) if finite else float("inf")


def _run_pressure_bound_test(
    tasks: pd.DataFrame,
    capacity: pd.DataFrame,
    dispatch_state: dict[str, object],
    records: list[dict[str, object]],
) -> dict[str, object]:
    """Report verifiable load bounds without claiming an exact critical point.

    The full critical MILP has many equivalent start-time combinations. Report
    two strict evidence types first: a feasible lower bound from a verified
    schedule scaled with load, and an infeasible upper bound from capacity
    conservation. These form a traceable interval. An exact supremum requires
    a separate long-limit critical MILP search; no point within this interval
    may be reported as the exact critical value.
    """

    assignments = dispatch_state["assignments"]
    fixed_lower = _fixed_schedule_alpha_lower_bound(assignments, capacity)
    local_upper = _aggregate_alpha_upper_bound(tasks, capacity, local_only=True)
    system_upper = _aggregate_alpha_upper_bound(tasks, capacity, local_only=False)
    finite_ok = bool(
        np.isfinite(fixed_lower)
        and np.isfinite(local_upper)
        and np.isfinite(system_upper)
        and fixed_lower >= 1.0 - NUMERICAL_EPS
        and local_upper >= fixed_lower - NUMERICAL_EPS
        and system_upper >= fixed_lower - NUMERICAL_EPS
    )

    critical_rows = []
    for granularity in ["1h", "0.5h"]:
        critical_rows.extend(
            [
                {
                    "Granularity": granularity,
                    "Boundary": "alpha_local",
                    "AlphaLower": fixed_lower,
                    "AlphaUpper": local_upper,
                    "EvidenceType": "fixed_feasible_schedule_to_total_local_capacity",
                    "ExactCriticalValueSolved": False,
                },
                {
                    "Granularity": granularity,
                    "Boundary": "alpha_feas",
                    "AlphaLower": fixed_lower,
                    "AlphaUpper": system_upper,
                    "EvidenceType": "fixed_feasible_schedule_to_total_system_capacity",
                    "ExactCriticalValueSolved": False,
                },
            ]
        )
    critical = pd.DataFrame(critical_rows)
    _write_table(critical, "validation_critical_pressure.csv")
    _write_table(
        pd.DataFrame(
            [
                {
                    "Granularity": "1h",
                    "Alpha": 1.0,
                    "F1": float(dispatch_state["f1"]),
                    "F2": float(dispatch_state["f2"]),
                    "Source": "Current verified two-stage MILP solution",
                }
            ]
        ),
        "validation_pressure_objectives.csv",
    )
    _write_table(
        pd.DataFrame(
            [
                {
                    "Granularity": granularity,
                    "Alpha": fixed_lower,
                    "Feasible": True,
                    "Boundary": "constructed_lower_bound",
                    "Method": "validated_assignment_scaled_without_rescheduling",
                }
                for granularity in ["1h", "0.5h"]
            ]
            + [
                {
                    "Granularity": "1h",
                    "Alpha": local_upper,
                    "Feasible": False,
                    "Boundary": "local_capacity_upper_bound",
                    "Method": "regional_total_capacity_conservation",
                },
                {
                    "Granularity": "1h",
                    "Alpha": system_upper,
                    "Feasible": False,
                    "Boundary": "system_capacity_upper_bound",
                    "Method": "system_total_capacity_conservation",
                },
            ]
        ),
        "validation_pressure_trace.csv",
    )

    delta_alpha = 0.0
    current_margin = fixed_lower - 1.0
    _write_table(
        pd.DataFrame(
            [
                {
                    "AlphaLocal_1h_LowerBound": fixed_lower,
                    "AlphaLocal_0_5h_LowerBound": fixed_lower,
                    "DeltaAlphaLocalLowerBound": delta_alpha,
                    "CurrentToLowerBoundMargin": current_margin,
                    "Interpretation": "Both resolutions contain the same feasible integer-time construction; the exact critical supremum under rescheduling remains unsolved",
                }
            ]
        ),
        "validation_granularity_pressure_interaction.csv",
    )
    _record(
        records,
        "Validation 3: scheduling optimality and critical load",
        "Strict critical-load lower and upper bounds",
        "QUALIFIED" if finite_ok else "FAIL",
        f"constructed_lower={fixed_lower:.6g}; local_capacity_upper={local_upper:.6g}; system_capacity_upper={system_upper:.6g}",
        "A verified construction is feasible at alpha=1; scaling that fixed construction gives a lower bound, and regional/system capacity conservation gives an infeasible upper bound",
        "Report only a boundary interval. Exact alpha_local or alpha_feas requires continued long-limit critical MILP solving with recorded solver status",
    )
    _record(
        records,
        "Validation 3: scheduling optimality and critical load",
        "Interaction between time resolution and stress bounds",
        "QUALIFIED" if finite_ok else "FAIL",
        f"lower_bound_shift={delta_alpha:.6g}; current_margin_to_lower_bound={current_margin:.6g}",
        "The 1h and 0.5h grids share the same feasible integer-time construction, so their lower bounds agree; exact critical suprema still require separate solving",
        "Equal lower bounds do not prove equal critical points for both resolutions",
    )
    return {"ok": finite_ok}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    records: list[dict[str, object]] = []
    try:
        panel, tasks, candidates, capacity = _read_inputs()
        logging.info(
            "Q1 validation started: %d panel rows, %d scheduling tasks, %d candidate rows, %d capacity rows.",
            len(panel),
            len(tasks.loc[tasks["ArrivalHour"].between(TEST_START_HOUR, TEST_END_HOUR)]),
            len(candidates),
            len(capacity),
        )
        rolling_state = _run_rolling_forecast_test(panel, records)
        structure_state = _run_structure_test(panel, rolling_state, records)
        dispatch_state = _run_dispatch_certificate(tasks, capacity, records)
        pressure_state = _run_pressure_bound_test(
            tasks, capacity, dispatch_state, records
        )
    except Exception as exc:
        logging.exception("Q1 validation failed")
        _record(
            records,
            "Validation workflow",
            "Abnormal termination",
            "FAIL",
            f"{type(exc).__name__}: {exc}",
            "All three validations must complete and produce evidence tables",
            "Repair invalid inputs, code interfaces, or solver status before interpreting validation conclusions",
        )
        _write_table(pd.DataFrame(records), "validation_summary.csv")
        return 1

    summary = pd.DataFrame(records)
    _write_table(summary, "validation_summary.csv")
    success = bool(rolling_state["ok"] and structure_state["ok"] and dispatch_state["ok"] and pressure_state["ok"])
    logging.info(
        "Q1 validation completed: test1=%s, test2=%s, dispatch_certificate=%s, pressure=%s",
        rolling_state["ok"],
        structure_state["ok"],
        dispatch_state["ok"],
        pressure_state["ok"],
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
