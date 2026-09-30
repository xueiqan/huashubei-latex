"""Question 1: hierarchical short-term forecasting, baseline comparison, and compute scheduling.

Forecasting and scheduling are separate:

* Main forecast: predict regional and task-type margins and the system total,
  then reconcile the hierarchy.
* Optional baseline: a naive forecast from the same hour 24 hours earlier.
* Scheduling: use only actual arrivals during hours 2376--2399, never forecasts.

The binary scheduling model first minimizes interregional migrated GPU workload,
then minimizes flexible-task waiting time while fixing the first optimum. Exact
solving requires ``scipy.optimize.milp`` in the existing environment. This module
does not change project dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import logging
import math

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
SAME_HOUR_MODEL = "SameHour24Baseline"
SAME_HOUR_LAG = 24
HISTORY_WINDOW_CANDIDATES = (24, 72, 168, 336)
IPF_MAX_ITERATIONS = 1000
IPF_TOLERANCE = 1e-12
ROLLING_WINDOW_COUNT = 68
ROLLING_HISTORY_HOURS = 720
ROLLING_HORIZON_HOURS = 24

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
        raise FileNotFoundError(f"Model input table not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required model columns: {missing}")
    return frame


def _read_model_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = _read_csv(Q1_DIR / "hourly_demand_panel.csv", PANEL_COLUMNS)
    tasks = _read_csv(SHARED_DIR / "tasks_clean.csv", TASK_COLUMNS)
    candidates = _read_csv(SHARED_DIR / "task_candidate_regions.csv", CANDIDATE_COLUMNS)
    capacity = _read_csv(Q1_DIR / "region_hour_capacity.csv", CAPACITY_COLUMNS)

    if panel.duplicated(["Hour", *SERIES_COLUMNS]).any():
        raise ValueError("hourly_demand_panel must have unique Hour x SourceRegion x TaskType keys")
    if tasks["TaskID"].duplicated().any():
        raise ValueError("tasks_clean contains duplicate TaskID values")
    if candidates.duplicated(["TaskID", "TargetRegion"]).any():
        raise ValueError("task_candidate_regions contains duplicate TaskID x TargetRegion records")
    if capacity.duplicated(["Hour", "Region"]).any():
        raise ValueError("region_hour_capacity contains duplicate Hour x Region records")

    panel["Hour"] = pd.to_numeric(panel["Hour"], errors="raise").astype("int64")
    panel["Task_Count"] = pd.to_numeric(panel["Task_Count"], errors="raise")
    panel["GPU_Demand_Arrival"] = pd.to_numeric(
        panel["GPU_Demand_Arrival"], errors="raise"
    )
    panel["GPU_Workload_Arrival_GPUh"] = pd.to_numeric(
        panel["GPU_Workload_Arrival_GPUh"], errors="raise"
    )
    capacity["Hour"] = pd.to_numeric(capacity["Hour"], errors="raise").astype("int64")
    if not capacity["Hour"].between(TEST_START_HOUR, TAIL_END_HOUR).all():
        raise ValueError("Q1 capacity must cover hours 2376--2405 and exclude task execution capacity at hour 2406")
    return panel, tasks, candidates, capacity


HIERARCHY_LEVELS = ("Region", "TaskType", "System")


def _aggregate_hourly_demand(panel: pd.DataFrame, level: str) -> pd.DataFrame:
    """Aggregate the 18 bottom-level hourly series by region, task type, or system."""

    if level == "Region":
        result = (
            panel.groupby(["Hour", "SourceRegion"], as_index=False, sort=True)[
                "GPU_Demand_Arrival"
            ]
            .sum()
            .rename(columns={"SourceRegion": "Entity", "GPU_Demand_Arrival": "Demand"})
        )
    elif level == "TaskType":
        result = (
            panel.groupby(["Hour", "TaskType"], as_index=False, sort=True)[
                "GPU_Demand_Arrival"
            ]
            .sum()
            .rename(columns={"TaskType": "Entity", "GPU_Demand_Arrival": "Demand"})
        )
    elif level == "System":
        result = (
            panel.groupby("Hour", as_index=False, sort=True)["GPU_Demand_Arrival"]
            .sum()
            .rename(columns={"GPU_Demand_Arrival": "Demand"})
        )
        result["Entity"] = "ALL"
    else:
        raise ValueError(f"Unknown demand aggregation level: {level}")
    result["Entity"] = result["Entity"].astype(str)
    return result[["Hour", "Entity", "Demand"]]


def _hierarchy_entities(panel: pd.DataFrame, level: str) -> list[str]:
    if level == "Region":
        return sorted(panel["SourceRegion"].astype(str).unique())
    if level == "TaskType":
        return sorted(panel["TaskType"].astype(str).unique())
    if level == "System":
        return ["ALL"]
    raise ValueError(f"Unknown demand aggregation level: {level}")


def _history_mean_parameters(
    history: pd.DataFrame, fit_end_hour: int, history_window_hours: int
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    history_start_hour = max(TRAIN_START_HOUR, fit_end_hour - history_window_hours + 1)
    window = history.loc[history["Hour"].between(history_start_hour, fit_end_hour)].copy()
    if window.empty:
        raise ValueError("Local-mean forecasting has no valid historical window")

    parameters: list[dict[str, object]] = []
    level_means: dict[str, pd.DataFrame] = {}
    for level in HIERARCHY_LEVELS:
        hourly = _aggregate_hourly_demand(window, level)
        means = (
            hourly.groupby("Entity", as_index=False, sort=True)
            .agg(
                Mean_GPU_Demand=("Demand", "mean"),
                HistoryTotal_GPU_Demand=("Demand", "sum"),
                HistoryObservationCount=("Hour", "nunique"),
            )
        )
        expected_entities = _hierarchy_entities(history, level)
        means = means.set_index("Entity").reindex(expected_entities).reset_index()
        means["Mean_GPU_Demand"] = means["Mean_GPU_Demand"].fillna(0.0)
        means["HistoryTotal_GPU_Demand"] = means["HistoryTotal_GPU_Demand"].fillna(0.0)
        means["HistoryObservationCount"] = means["HistoryObservationCount"].fillna(0).astype(int)
        means["HistoryStartHour"] = history_start_hour
        means["FitEndHour"] = fit_end_hour
        means["HistoryWindowHours"] = history_window_hours
        level_means[level] = means
        for row in means.itertuples(index=False):
            parameters.append(
                {
                    "Model": FORECAST_MODEL,
                    "Level": level,
                    "Entity": str(row.Entity),
                    "FitEndHour": fit_end_hour,
                    "HistoryStartHour": history_start_hour,
                    "HistoryWindowHours": history_window_hours,
                    "Mean_GPU_Demand": float(row.Mean_GPU_Demand),
                    "HistoryTotal_GPU_Demand": float(row.HistoryTotal_GPU_Demand),
                    "HistoryObservationCount": int(row.HistoryObservationCount),
                }
            )
    return pd.DataFrame(parameters), level_means


def _build_cross_level_prior(history: pd.DataFrame) -> tuple[list[str], list[str], np.ndarray]:
    regions = _hierarchy_entities(history, "Region")
    task_types = _hierarchy_entities(history, "TaskType")
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
    if total <= 0:
        prior = np.full(matrix.shape, 1.0 / matrix.size, dtype=float)
    else:
        prior = matrix / total
        # Avoid zero historical region/type cells that prevent IPF from matching positive margins.
        prior = prior + 1e-12
        prior = prior / prior.sum()
    return regions, task_types, prior


def _ipf_reconcile(
    row_targets: np.ndarray,
    column_targets: np.ndarray,
    prior: np.ndarray,
) -> np.ndarray:
    """Use iterative proportional fitting for the least-change matrix matching both margins."""

    rows = np.maximum(np.asarray(row_targets, dtype=float), 0.0)
    columns = np.maximum(np.asarray(column_targets, dtype=float), 0.0)
    total = float(rows.sum())
    if total <= 1e-12:
        return np.zeros_like(prior, dtype=float)
    if columns.sum() <= 1e-12:
        columns = np.full_like(columns, total / len(columns))
    else:
        columns = columns * total / float(columns.sum())
    if rows.sum() <= 1e-12:
        rows = np.full_like(rows, total / len(rows))
    else:
        rows = rows * total / float(rows.sum())

    matrix = np.maximum(prior, 1e-15) * total
    tolerance = IPF_TOLERANCE * max(1.0, total)
    for _ in range(IPF_MAX_ITERATIONS):
        row_sums = matrix.sum(axis=1)
        matrix *= np.divide(rows, row_sums, out=np.ones_like(rows), where=row_sums > 0)[:, None]
        column_sums = matrix.sum(axis=0)
        matrix *= np.divide(columns, column_sums, out=np.ones_like(columns), where=column_sums > 0)[None, :]
        residual = max(
            float(np.max(np.abs(matrix.sum(axis=1) - rows))),
            float(np.max(np.abs(matrix.sum(axis=0) - columns))),
        )
        if residual <= tolerance:
            break
    else:
        raise RuntimeError("Hierarchical reconciliation IPF did not converge within the iteration limit")
    return matrix


def _hierarchical_local_mean_prediction(
    panel: pd.DataFrame,
    target_start_hour: int,
    target_end_hour: int,
    fit_end_hour: int,
    history_window_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Forecast regional, type, and system margins and recover 18 region/type cells."""

    history = panel.loc[panel["Hour"].between(TRAIN_START_HOUR, fit_end_hour)].copy()
    target = panel.loc[panel["Hour"].between(target_start_hour, target_end_hour)].copy()
    if history.empty or target.empty:
        raise ValueError("Hierarchical local-mean forecasting is missing history or target-hour data")

    parameters, level_means = _history_mean_parameters(
        history, fit_end_hour, history_window_hours
    )
    regions, task_types, prior = _build_cross_level_prior(history)
    actual_bottom = (
        target.set_index(["Hour", "SourceRegion", "TaskType"])["GPU_Demand_Arrival"]
        .astype(float)
        .to_dict()
    )
    actual_levels = {
        level: _aggregate_hourly_demand(target, level).set_index(["Hour", "Entity"])["Demand"].to_dict()
        for level in HIERARCHY_LEVELS
    }
    main_rows: list[dict[str, object]] = []
    hierarchy_rows: list[dict[str, object]] = []
    region_mean = level_means["Region"].set_index("Entity")["Mean_GPU_Demand"]
    type_mean = level_means["TaskType"].set_index("Entity")["Mean_GPU_Demand"]
    system_mean = float(level_means["System"].iloc[0]["Mean_GPU_Demand"])

    for hour in range(target_start_hour, target_end_hour + 1):
        region_independent = np.array([float(region_mean.get(region, 0.0)) for region in regions])
        type_independent = np.array([float(type_mean.get(task_type, 0.0)) for task_type in task_types])
        system_prediction = max(0.0, system_mean)
        region_sum = float(region_independent.sum())
        type_sum = float(type_independent.sum())
        region_reconciled = (
            region_independent * system_prediction / region_sum
            if region_sum > 1e-12
            else np.full(len(regions), system_prediction / len(regions))
        )
        type_reconciled = (
            type_independent * system_prediction / type_sum
            if type_sum > 1e-12
            else np.full(len(task_types), system_prediction / len(task_types))
        )
        cross = _ipf_reconcile(region_reconciled, type_reconciled, prior)

        for region_index, region in enumerate(regions):
            hierarchy_rows.append(
                {
                    "Model": FORECAST_MODEL,
                    "Level": "Region",
                    "Entity": region,
                    "Hour": hour,
                    "Actual_GPU_Demand": float(actual_levels["Region"].get((hour, region), 0.0)),
                    "IndependentPrediction": float(region_independent[region_index]),
                    "ReconciledPrediction": float(region_reconciled[region_index]),
                    "FitEndHour": fit_end_hour,
                    "HistoryWindowHours": history_window_hours,
                }
            )
        for task_index, task_type in enumerate(task_types):
            hierarchy_rows.append(
                {
                    "Model": FORECAST_MODEL,
                    "Level": "TaskType",
                    "Entity": task_type,
                    "Hour": hour,
                    "Actual_GPU_Demand": float(actual_levels["TaskType"].get((hour, task_type), 0.0)),
                    "IndependentPrediction": float(type_independent[task_index]),
                    "ReconciledPrediction": float(type_reconciled[task_index]),
                    "FitEndHour": fit_end_hour,
                    "HistoryWindowHours": history_window_hours,
                }
            )
        hierarchy_rows.append(
            {
                "Model": FORECAST_MODEL,
                "Level": "System",
                "Entity": "ALL",
                "Hour": hour,
                "Actual_GPU_Demand": float(actual_levels["System"].get((hour, "ALL"), 0.0)),
                "IndependentPrediction": system_prediction,
                "ReconciledPrediction": system_prediction,
                "FitEndHour": fit_end_hour,
                "HistoryWindowHours": history_window_hours,
            }
        )
        for region_index, region in enumerate(regions):
            for task_index, task_type in enumerate(task_types):
                main_rows.append(
                    {
                        "SourceRegion": region,
                        "TaskType": task_type,
                        "Hour": hour,
                        "Actual_GPU_Demand": float(actual_bottom.get((hour, region, task_type), 0.0)),
                        "Prediction": float(cross[region_index, task_index]),
                        "Model": FORECAST_MODEL,
                        "FitEndHour": fit_end_hour,
                        "HistoryWindowHours": history_window_hours,
                    }
                )
    return pd.DataFrame(main_rows), pd.DataFrame(hierarchy_rows), parameters


def _same_hour_prediction(
    panel: pd.DataFrame,
    target_start_hour: int,
    target_end_hour: int,
) -> pd.DataFrame:
    target = panel.loc[
        panel["Hour"].between(target_start_hour, target_end_hour),
        [*SERIES_COLUMNS, "Hour", "GPU_Demand_Arrival"],
    ].copy()
    lagged = panel.loc[:, [*SERIES_COLUMNS, "Hour", "GPU_Demand_Arrival"]].copy()
    lagged["Hour"] = lagged["Hour"] + SAME_HOUR_LAG
    lagged = lagged.rename(columns={"GPU_Demand_Arrival": "Prediction"})
    target = target.merge(
        lagged,
        how="left",
        on=[*SERIES_COLUMNS, "Hour"],
        validate="one_to_one",
    )
    if target["Prediction"].isna().any():
        raise ValueError("The 24-hour same-hour baseline is missing t-24 history")
    target["Prediction"] = target["Prediction"].clip(lower=0.0)
    target["Model"] = SAME_HOUR_MODEL
    target["FitEndHour"] = target_start_hour - SAME_HOUR_LAG
    target["HistoryWindowHours"] = SAME_HOUR_LAG
    return target.rename(columns={"GPU_Demand_Arrival": "Actual_GPU_Demand"})[
        [
            *SERIES_COLUMNS,
            "Hour",
            "Actual_GPU_Demand",
            "Prediction",
            "Model",
            "FitEndHour",
            "HistoryWindowHours",
        ]
    ]


def _forecast_window(
    panel: pd.DataFrame,
    target_start_hour: int,
    target_end_hour: int,
    fit_end_hour: int,
    history_window_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main, hierarchy, parameters = _hierarchical_local_mean_prediction(
        panel,
        target_start_hour,
        target_end_hour,
        fit_end_hour,
        history_window_hours,
    )
    baseline = _same_hour_prediction(panel, target_start_hour, target_end_hour)
    predictions = pd.concat([main, baseline], ignore_index=True)
    return predictions, hierarchy, parameters


def _metric_values(actual: pd.Series, prediction: pd.Series) -> tuple[float, float, float]:
    error = actual.to_numpy(dtype=float) - prediction.to_numpy(dtype=float)
    actual_values = actual.to_numpy(dtype=float)
    denominator = float(np.abs(actual_values).sum())
    wape = float(np.abs(error).sum() / denominator) if denominator > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mae = float(np.mean(np.abs(error)))
    return wape, rmse, mae


def _select_history_window(panel: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for history_window_hours in HISTORY_WINDOW_CANDIDATES:
        main, _, _ = _hierarchical_local_mean_prediction(
            panel,
            VALIDATION_START_HOUR,
            VALIDATION_END_HOUR,
            TRAIN_END_HOUR,
            history_window_hours,
        )
        wape, rmse, mae = _metric_values(main["Actual_GPU_Demand"], main["Prediction"])
        rows.append(
            {
                "Model": FORECAST_MODEL,
                "HistoryWindowHours": history_window_hours,
                "ValidationWAPE": wape,
                "ValidationRMSE": rmse,
                "ValidationMAE": mae,
            }
        )
    selection = pd.DataFrame(rows).sort_values(
        ["ValidationWAPE", "ValidationRMSE", "HistoryWindowHours"], kind="stable"
    ).reset_index(drop=True)
    selected = int(selection.iloc[0]["HistoryWindowHours"])
    selection["Selected"] = selection["HistoryWindowHours"].eq(selected)
    return selected, selection


def _build_metrics(predictions: pd.DataFrame, split_name: str | None = None) -> pd.DataFrame:
    data = predictions if split_name is None else predictions.loc[predictions["Split"] == split_name]
    rows: list[dict[str, object]] = []
    group_columns = ["Split", "Model", *SERIES_COLUMNS]
    for group_key, group in data.groupby(group_columns, sort=True):
        split, model, region, task_type = group_key
        wape, rmse, mae = _metric_values(group["Actual_GPU_Demand"], group["Prediction"])
        rows.append(
            {
                "Split": split,
                "Model": model,
                "SourceRegion": region,
                "TaskType": task_type,
                "ObservationCount": len(group),
                "ActualTotal_GPU": float(group["Actual_GPU_Demand"].sum()),
                "WAPE": wape,
                "RMSE": rmse,
                "MAE": mae,
            }
        )
    for (split, model), group in data.groupby(["Split", "Model"], sort=True):
        wape, rmse, mae = _metric_values(group["Actual_GPU_Demand"], group["Prediction"])
        rows.append(
            {
                "Split": split,
                "Model": model,
                "SourceRegion": "ALL",
                "TaskType": "ALL",
                "ObservationCount": len(group),
                "ActualTotal_GPU": float(group["Actual_GPU_Demand"].sum()),
                "WAPE": wape,
                "RMSE": rmse,
                "MAE": mae,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["Split", "SourceRegion", "TaskType", "Model"], kind="stable"
    ).reset_index(drop=True)


def _build_hierarchy_metrics(hierarchy_predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (split, model, level, entity), group in hierarchy_predictions.groupby(
        ["Split", "Model", "Level", "Entity"], sort=True
    ):
        wape, rmse, mae = _metric_values(
            group["Actual_GPU_Demand"], group["ReconciledPrediction"]
        )
        rows.append(
            {
                "Split": split,
                "Model": model,
                "Level": level,
                "Entity": entity,
                "ObservationCount": len(group),
                "ActualTotal_GPU": float(group["Actual_GPU_Demand"].sum()),
                "WAPE": wape,
                "RMSE": rmse,
                "MAE": mae,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["Split", "Level", "Entity"], kind="stable"
    ).reset_index(drop=True)


def _build_aggregation_sensitivity(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize hourly and daily system levels to show aggregation effects on error and sample size."""

    main = predictions.loc[predictions["Model"].eq(FORECAST_MODEL)].copy()
    main["DayIndex"] = (main["Hour"] // 24).astype(int)
    rows: list[dict[str, object]] = []
    for split, split_frame in main.groupby("Split", sort=True):
        aggregations = {
            "BottomHourly": split_frame.groupby(["Hour", *SERIES_COLUMNS], as_index=False)[
                ["Actual_GPU_Demand", "Prediction"]
            ].sum(),
            "RegionHourly": split_frame.groupby(["Hour", "SourceRegion"], as_index=False)[
                ["Actual_GPU_Demand", "Prediction"]
            ].sum().rename(columns={"SourceRegion": "Entity"}),
            "TaskTypeHourly": split_frame.groupby(["Hour", "TaskType"], as_index=False)[
                ["Actual_GPU_Demand", "Prediction"]
            ].sum().rename(columns={"TaskType": "Entity"}),
            "SystemHourly": split_frame.groupby(["Hour"], as_index=False)[
                ["Actual_GPU_Demand", "Prediction"]
            ].sum().assign(Entity="ALL"),
            "SystemDaily": split_frame.groupby(["DayIndex"], as_index=False)[
                ["Actual_GPU_Demand", "Prediction"]
            ].sum().assign(Entity="ALL"),
        }
        for level, frame in aggregations.items():
            if level == "BottomHourly":
                group_columns = ["SourceRegion", "TaskType"]
                frame["Entity"] = [
                    f"{region}×{task_type}"
                    for region, task_type in zip(frame["SourceRegion"], frame["TaskType"])
                ]
            elif level == "RegionHourly":
                group_columns = ["Entity"]
            elif level == "TaskTypeHourly":
                group_columns = ["Entity"]
            else:
                group_columns = ["Entity"]
            for entity, group in frame.groupby(group_columns if level == "BottomHourly" else ["Entity"], sort=True):
                if isinstance(entity, tuple):
                    entity_name = "×".join(map(str, entity))
                else:
                    entity_name = str(entity)
                wape, rmse, mae = _metric_values(
                    group["Actual_GPU_Demand"], group["Prediction"]
                )
                rows.append(
                    {
                        "Split": split,
                        "AggregationLevel": level,
                        "Entity": entity_name,
                        "ObservationCount": len(group),
                        "ActualTotal_GPU": float(group["Actual_GPU_Demand"].sum()),
                        "WAPE": wape,
                        "RMSE": rmse,
                        "MAE": mae,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["Split", "AggregationLevel", "Entity"], kind="stable"
    ).reset_index(drop=True)


def _build_forecast_parameters(
    validation_parameters: pd.DataFrame,
    test_parameters: pd.DataFrame,
) -> pd.DataFrame:
    validation_parameters = validation_parameters.copy()
    test_parameters = test_parameters.copy()
    validation_parameters["Split"] = "validation"
    test_parameters["Split"] = "test"
    return pd.concat([validation_parameters, test_parameters], ignore_index=True)[
        [
            "Split",
            "Model",
            "Level",
            "Entity",
            "FitEndHour",
            "HistoryStartHour",
            "HistoryWindowHours",
            "Mean_GPU_Demand",
            "HistoryTotal_GPU_Demand",
            "HistoryObservationCount",
        ]
    ].sort_values(["Split", "Level", "Entity"], kind="stable")


def _build_rolling_backtest(
    panel: pd.DataFrame, history_window_hours: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce 68 nonoverlapping historical 24-hour windows to assess model and baseline stability."""

    starts = [
        ROLLING_HISTORY_HOURS + ROLLING_HORIZON_HOURS * index
        for index in range(ROLLING_WINDOW_COUNT)
    ]
    expected_last_start = TRAIN_END_HOUR - ROLLING_HORIZON_HOURS + 1
    if starts[-1] != expected_last_start:
        raise ValueError(
            f"Invalid rolling-window configuration: expected last start {expected_last_start}; actual {starts[-1]}"
        )

    logging.info("Rolling backtest started: %d windows of 24 hours.", len(starts))
    rows: list[dict[str, object]] = []
    for window_id, start in enumerate(starts, start=1):
        if window_id == 1 or window_id % 10 == 0 or window_id == len(starts):
            logging.info(
                "Rolling backtest progress: %d/%d, target hours=%d--%d.",
                window_id,
                len(starts),
                start,
                start + ROLLING_HORIZON_HOURS - 1,
            )
        end = start + ROLLING_HORIZON_HOURS - 1
        window_predictions, _, _ = _forecast_window(
            panel, start, end, start - 1, history_window_hours
        )
        for model, group in window_predictions.groupby("Model", sort=True):
            wape, rmse, mae = _metric_values(
                group["Actual_GPU_Demand"], group["Prediction"]
            )
            rows.append(
                {
                    "WindowID": window_id,
                    "WindowStartHour": start,
                    "WindowEndHour": end,
                    "Model": model,
                    "WAPE": wape,
                    "RMSE": rmse,
                    "MAE": mae,
                    "HistoryWindowHours": history_window_hours
                    if model == FORECAST_MODEL
                    else SAME_HOUR_LAG,
                }
            )
    details = pd.DataFrame(rows)
    summary = (
        details.groupby("Model", as_index=False)
        .agg(
            WindowCount=("WindowID", "nunique"),
            Mean_WAPE=("WAPE", "mean"),
            Mean_RMSE=("RMSE", "mean"),
            Mean_MAE=("MAE", "mean"),
        )
        .sort_values("Mean_WAPE", kind="stable")
        .reset_index(drop=True)
    )
    logging.info(
        "Rolling backtest completed: %d detail rows, %d models.",
        len(details),
        summary["Model"].nunique(),
    )
    return details, summary


@dataclass(frozen=True)
class DispatchOption:
    option_id: int
    task_id: str
    task_type: str
    arrival_hour: int
    source_region: str
    target_region: str
    network_latency_ms: float
    max_latency_ms: float
    start_hour: float
    finish_hour: float
    duration_h: float
    gpu_demand: float
    task_power_mw: float
    wait_hours: float
    migration_gpu_workload_gpuh: float
    overlap_by_hour: tuple[tuple[int, float], ...]


def _task_start_hours(task: pd.Series, start_step_hours: float = 1.0) -> list[float]:
    if not math.isfinite(start_step_hours) or start_step_hours <= 0:
        raise ValueError("Start-time resolution must be positive")
    task_type = str(task["TaskType"])
    arrival = int(task["ArrivalHour"])
    earliest = int(task["EarliestStartHour"])
    latest_finish = min(float(task["LatestFinishHour"]), float(TERMINAL_HOUR))
    duration = float(task["Duration_h"])

    if task_type == "RealTimeInference":
        starts = [float(arrival)]
    else:
        latest_start = min(
            float(TAIL_END_HOUR),
            latest_finish - duration,
        )
        first_start = float(max(arrival, earliest))
        count = int(math.floor((latest_start - first_start) / start_step_hours + 1e-9))
        starts = [first_start + index * start_step_hours for index in range(count + 1)]

    valid = [
        start
        for start in starts
        if start >= earliest
        and start + duration <= latest_finish + 1e-9
        and start + duration <= TERMINAL_HOUR + 1e-9
    ]
    if not valid:
        raise ValueError(
            f"Task {task['TaskID']} has no start time satisfying real-time, earliest-start, latest-finish, and hour-2406 boundaries at "
            f"{start_step_hours}-hour resolution"
        )
    return valid


def _overlap_by_hour(start_hour: float, duration_h: float) -> tuple[tuple[int, float], ...]:
    end_hour = float(start_hour) + duration_h
    overlaps: list[tuple[int, float]] = []
    for hour in range(TEST_START_HOUR, TAIL_END_HOUR + 1):
        overlap = max(
            0.0,
            min(end_hour, float(hour + 1)) - max(float(start_hour), float(hour)),
        )
        if overlap > 1e-12:
            overlaps.append((hour, overlap))
    return tuple(overlaps)


def _build_dispatch_options(
    tasks: pd.DataFrame,
    candidates: pd.DataFrame,
    start_step_hours: float = 1.0,
) -> tuple[pd.DataFrame, list[DispatchOption]]:
    dispatch_tasks = tasks.loc[
        tasks["ArrivalHour"].between(TEST_START_HOUR, TEST_END_HOUR)
    ].copy()
    if dispatch_tasks.empty:
        raise ValueError("No actual arrivals during hours 2376--2399 are available for compute scheduling")

    task_ids = set(dispatch_tasks["TaskID"])
    candidate_groups = {
        task_id: group
        for task_id, group in candidates.loc[
            candidates["TaskID"].isin(task_ids)
        ].groupby("TaskID", sort=False)
    }
    options: list[DispatchOption] = []
    option_id = 0
    for _, task in dispatch_tasks.sort_values(
        ["LatestFinishHour", "EarliestStartHour", "TaskID"], kind="stable"
    ).iterrows():
        task_id = task["TaskID"]
        candidate_rows = candidate_groups.get(task_id)
        if candidate_rows is None or candidate_rows.empty:
            raise ValueError(f"Task {task_id} has no feasible candidate region")
        starts = _task_start_hours(task, start_step_hours=start_step_hours)
        duration = float(task["Duration_h"])
        for _, candidate in candidate_rows.sort_values("TargetRegion", kind="stable").iterrows():
            for start_hour in starts:
                overlaps = _overlap_by_hour(start_hour, duration)
                if not overlaps:
                    continue
                target_region = str(candidate["TargetRegion"])
                source_region = str(task["SourceRegion"])
                migrated = target_region != source_region
                options.append(
                    DispatchOption(
                        option_id=option_id,
                        task_id=str(task_id),
                        task_type=str(task["TaskType"]),
                        arrival_hour=int(task["ArrivalHour"]),
                        source_region=source_region,
                        target_region=target_region,
                        network_latency_ms=float(candidate["NetworkLatency_ms"]),
                        max_latency_ms=float(task["MaxLatency_ms"]),
                        start_hour=float(start_hour),
                        finish_hour=float(start_hour + duration),
                        duration_h=duration,
                        gpu_demand=float(task["GPU_Demand"]),
                        task_power_mw=float(task["Task_Full_IT_Power_MW"]),
                        wait_hours=float(
                            start_hour - int(task["EarliestStartHour"])
                            if str(task["TaskType"]) != "RealTimeInference"
                            else 0.0
                        ),
                        migration_gpu_workload_gpuh=float(
                            task["GPU_Demand"] * duration if migrated else 0.0
                        ),
                        overlap_by_hour=overlaps,
                    )
                )
                option_id += 1

    if not options:
        raise ValueError("No task/region/start-time candidate combinations were generated")
    option_frame = pd.DataFrame(
        {
            "TaskID": [option.task_id for option in options],
            "OptionID": [option.option_id for option in options],
        }
    )
    return option_frame, options


def _build_dispatch_constraints(
    options: list[DispatchOption],
    dispatch_tasks: pd.DataFrame,
    capacity: pd.DataFrame,
):
    try:
        from scipy.sparse import coo_matrix
    except ImportError as exc:
        raise RuntimeError(
            "Exact two-stage binary compute scheduling requires scipy, which is absent from this environment. "
            "Verify and install a Python-compatible scipy version in the competition environment first."
        ) from exc

    task_ids = [str(task_id) for task_id in dispatch_tasks["TaskID"]]
    task_row = {task_id: index for index, task_id in enumerate(task_ids)}
    capacity = capacity.sort_values(["Hour", "Region"], kind="stable").reset_index(drop=True)
    capacity_keys = list(zip(capacity["Hour"].astype(int), capacity["Region"].astype(str)))
    if len(set(capacity_keys)) != len(capacity_keys):
        raise ValueError("Capacity table Hour x Region keys must be unique")
    capacity_row = {key: index for index, key in enumerate(capacity_keys)}

    assignment_count = len(task_ids)
    capacity_count = len(capacity)
    gpu_row = {
        key: assignment_count + index for index, key in enumerate(capacity_keys)
    }
    power_row = {
        key: assignment_count + capacity_count + index
        for index, key in enumerate(capacity_keys)
    }
    row_count = assignment_count + 2 * capacity_count
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []

    for column, option in enumerate(options):
        if option.task_id not in task_row:
            raise ValueError(f"Candidates include TaskID values outside the scheduling task set: {option.task_id}")
        row_indices.append(task_row[option.task_id])
        column_indices.append(column)
        values.append(1.0)
        for hour, overlap in option.overlap_by_hour:
            key = (hour, option.target_region)
            if key not in capacity_row:
                raise ValueError(f"Candidates exceed the capacity-table time horizon or region set: {key}")
            row_indices.extend([gpu_row[key], power_row[key]])
            column_indices.extend([column, column])
            values.extend(
                [option.gpu_demand * overlap, option.task_power_mw * overlap]
            )

    matrix = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(row_count, len(options)),
    ).tocsr()
    lower = np.full(row_count, -np.inf, dtype=float)
    upper = np.full(row_count, np.inf, dtype=float)
    lower[:assignment_count] = 1.0
    upper[:assignment_count] = 1.0
    upper[assignment_count : assignment_count + capacity_count] = capacity[
        "Available_GPU"
    ].to_numpy(dtype=float)
    upper[assignment_count + capacity_count :] = capacity[
        "Effective_AI_IT_Capacity_MW"
    ].to_numpy(dtype=float)
    migration_objective = np.array(
        [option.migration_gpu_workload_gpuh for option in options], dtype=float
    )
    wait_objective = np.array([option.wait_hours for option in options], dtype=float)
    return matrix, lower, upper, migration_objective, wait_objective


def _solve_two_stage_dispatch(
    options: list[DispatchOption],
    dispatch_tasks: pd.DataFrame,
    capacity: pd.DataFrame,
) -> tuple[np.ndarray, float, float, int, int, str, str, float, float, float, float]:
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csr_matrix, vstack
    except ImportError as exc:
        raise RuntimeError(
            "Compute scheduling requires scipy.optimize.milp; scipy is absent from this environment."
        ) from exc

    matrix, lower, upper, migration_objective, wait_objective = _build_dispatch_constraints(
        options, dispatch_tasks, capacity
    )
    logging.info(
        "Scheduling candidates generated: %d tasks, %d combinations, %d constraint rows.",
        len(dispatch_tasks),
        len(options),
        len(upper),
    )
    variable_count = len(options)
    integrality = np.ones(variable_count, dtype=np.int8)
    bounds = Bounds(np.zeros(variable_count), np.ones(variable_count))
    base_constraint = LinearConstraint(matrix, lower, upper)

    logging.info("Scheduling stage 1 started: minimize migrated GPU workload F1.")
    stage_one = milp(
        c=migration_objective,
        integrality=integrality,
        bounds=bounds,
        constraints=base_constraint,
        options={"disp": False},
    )
    if not stage_one.success or stage_one.x is None:
        raise RuntimeError(f"Scheduling stage 1 failed: {stage_one.message}")
    f1_star = float(np.dot(migration_objective, np.rint(stage_one.x)))
    logging.info(
        "Scheduling stage 1 completed: status=%d, F1*=%.6f, gap=%s.",
        int(stage_one.status),
        f1_star,
        getattr(stage_one, "mip_gap", np.nan),
    )

    f1_row = csr_matrix(migration_objective.reshape(1, -1))
    stage_two_matrix = vstack([matrix, f1_row], format="csr")
    stage_two_lower = np.concatenate([lower, [f1_star - 1e-8]])
    stage_two_upper = np.concatenate([upper, [f1_star + 1e-8]])
    logging.info("Scheduling stage 2 started: fix F1=F1* and minimize flexible-task waiting F2.")
    stage_two = milp(
        c=wait_objective,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(
            stage_two_matrix, stage_two_lower, stage_two_upper
        ),
        options={"disp": False},
    )
    if not stage_two.success or stage_two.x is None:
        raise RuntimeError(f"Scheduling stage 2 failed: {stage_two.message}")
    selected = np.rint(stage_two.x).astype(int)
    f1_check = float(np.dot(migration_objective, selected))
    if not math.isclose(f1_check, f1_star, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(f"Stage 2 did not preserve the optimal stage-1 migration objective: {f1_check} != {f1_star}")
    f2_star = float(np.dot(wait_objective, selected))
    logging.info(
        "Scheduling stage 2 completed: status=%d, F2*=%.6f, gap=%s.",
        int(stage_two.status),
        f2_star,
        getattr(stage_two, "mip_gap", np.nan),
    )
    selected_indices = np.flatnonzero(selected == 1)
    counts = pd.Series([options[index].task_id for index in selected_indices]).value_counts()
    if len(counts) != len(dispatch_tasks) or not (counts == 1).all():
        raise RuntimeError("The schedule does not execute each task exactly once")
    def _result_value(result: object, name: str) -> float:
        value = getattr(result, name, np.nan)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    return (
        selected,
        f1_star,
        f2_star,
        int(stage_one.status),
        int(stage_two.status),
        str(stage_one.message),
        str(stage_two.message),
        _result_value(stage_one, "mip_gap"),
        _result_value(stage_two, "mip_gap"),
        _result_value(stage_one, "mip_dual_bound"),
        _result_value(stage_two, "mip_dual_bound"),
    )


def _build_dispatch_assignments(
    options: list[DispatchOption], selected: np.ndarray
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in np.flatnonzero(selected == 1):
        option = options[int(index)]
        rows.append(
            {
                "TaskID": option.task_id,
                "TaskType": option.task_type,
                "ArrivalHour": option.arrival_hour,
                "SourceRegion": option.source_region,
                "TargetRegion": option.target_region,
                "NetworkLatency_ms": option.network_latency_ms,
                "MaxLatency_ms": option.max_latency_ms,
                "StartHour": (
                    int(option.start_hour)
                    if float(option.start_hour).is_integer()
                    else option.start_hour
                ),
                "FinishHour": option.finish_hour,
                "Duration_h": option.duration_h,
                "GPU_Demand": option.gpu_demand,
                "Task_Full_IT_Power_MW": option.task_power_mw,
                "WaitHours": option.wait_hours,
                "Migration_GPU_Workload_GPUh": option.migration_gpu_workload_gpuh,
                "IsMigrated": int(option.target_region != option.source_region),
            }
        )
    return pd.DataFrame(rows).sort_values("TaskID", kind="stable").reset_index(drop=True)


def _build_dispatch_resource_profile(
    options: list[DispatchOption],
    selected: np.ndarray,
    capacity: pd.DataFrame,
) -> pd.DataFrame:
    profile = capacity.copy().sort_values(["Hour", "Region"], kind="stable")
    profile["Scheduled_GPU_Equivalent"] = 0.0
    profile["Scheduled_AI_IT_Load_MW"] = 0.0
    profile = profile.reset_index(drop=True)
    profile_index = {
        (int(row.Hour), str(row.Region)): index
        for index, row in profile.iterrows()
    }
    for index in np.flatnonzero(selected == 1):
        option = options[int(index)]
        for hour, overlap in option.overlap_by_hour:
            row_index = profile_index[(hour, option.target_region)]
            profile.loc[row_index, "Scheduled_GPU_Equivalent"] += (
                option.gpu_demand * overlap
            )
            profile.loc[row_index, "Scheduled_AI_IT_Load_MW"] += (
                option.task_power_mw * overlap
            )
    profile["Total_IT_Load_MW"] = (
        profile["NonAI_IT_Load_MW"] + profile["Scheduled_AI_IT_Load_MW"]
    )
    profile["Total_Facility_Load_MW"] = profile["Total_IT_Load_MW"] * profile["PUE"]
    profile["GPU_Slack"] = profile["Available_GPU"] - profile["Scheduled_GPU_Equivalent"]
    profile["IT_Slack_MW"] = profile["Max_IT_Power_MW"] - profile["Total_IT_Load_MW"]
    profile["Facility_Slack_MW"] = (
        profile["Max_Facility_Power_MW"] - profile["Total_Facility_Load_MW"]
    )
    profile["GPU_Utilization"] = np.divide(
        profile["Scheduled_GPU_Equivalent"],
        profile["Available_GPU"],
        out=np.zeros(len(profile), dtype=float),
        where=profile["Available_GPU"].to_numpy(dtype=float) > 0,
    )
    profile["AI_IT_Capacity_Utilization"] = np.divide(
        profile["Scheduled_AI_IT_Load_MW"],
        profile["Effective_AI_IT_Capacity_MW"],
        out=np.zeros(len(profile), dtype=float),
        where=profile["Effective_AI_IT_Capacity_MW"].to_numpy(dtype=float) > 0,
    )
    return profile


def _validate_dispatch_result(profile: pd.DataFrame, assignments: pd.DataFrame) -> None:
    if assignments["TaskID"].duplicated().any():
        raise RuntimeError("Scheduling output contains duplicate tasks")
    if not (assignments["NetworkLatency_ms"] <= assignments["MaxLatency_ms"] + 1e-9).all():
        raise RuntimeError("The schedule violates network latency constraints")
    if not (assignments["FinishHour"] <= TERMINAL_HOUR + 1e-9).all():
        raise RuntimeError("The schedule occupies or exceeds hour 2406")
    if (profile["GPU_Slack"] < -1e-7).any():
        raise RuntimeError("The schedule violates GPU capacity")
    if (profile["IT_Slack_MW"] < -1e-7).any():
        raise RuntimeError("The schedule violates IT power limits")
    if (profile["Facility_Slack_MW"] < -1e-7).any():
        raise RuntimeError("The schedule violates facility power limits")


def _build_dispatch_summary(
    assignments: pd.DataFrame,
    profile: pd.DataFrame,
    f1_star: float,
    f2_star: float,
    stage_one_status: int,
    stage_two_status: int,
    stage_one_message: str,
    stage_two_message: str,
    stage_one_mip_gap: float,
    stage_two_mip_gap: float,
    stage_one_mip_dual_bound: float,
    stage_two_mip_dual_bound: float,
) -> pd.DataFrame:
    flexible = assignments["TaskType"].isin(["BatchInference", "AITraining"])
    migrated = assignments["IsMigrated"].astype(bool)
    return pd.DataFrame(
        [
            {
                "Solver": "scipy.optimize.milp",
                "Stage1Status": stage_one_status,
                "Stage2Status": stage_two_status,
                "Stage1Message": stage_one_message,
                "Stage2Message": stage_two_message,
                "Stage1MIPGap": stage_one_mip_gap,
                "Stage2MIPGap": stage_two_mip_gap,
                "Stage1MIPDualBound": stage_one_mip_dual_bound,
                "Stage2MIPDualBound": stage_two_mip_dual_bound,
                "AssignedTaskCount": len(assignments),
                "MigratedTaskCount": int(migrated.sum()),
                "LocalTaskRatio": float(1.0 - migrated.mean()),
                "F1_Migration_GPU_Workload_GPUh": f1_star,
                "F2_Flexible_WaitHours": f2_star,
                "FlexibleTaskCount": int(flexible.sum()),
                "Max_GPU_Utilization": float(profile["GPU_Utilization"].max()),
                "Max_AI_IT_Capacity_Utilization": float(
                    profile["AI_IT_Capacity_Utilization"].max()
                ),
                "Min_GPU_Slack": float(profile["GPU_Slack"].min()),
                "Min_IT_Slack_MW": float(profile["IT_Slack_MW"].min()),
                "Min_Facility_Slack_MW": float(profile["Facility_Slack_MW"].min()),
            }
        ]
    )


def _write_table(frame: pd.DataFrame, filename: str) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    # Retain precision so CSV reaggregation does not mistake IPF rounding for reconciliation errors.
    frame.to_csv(
        TABLES_DIR / filename,
        index=False,
        encoding="utf-8-sig",
        float_format="%.15g",
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    logging.info("Q1 modeling started: reading shared tasks, candidate regions, demand panel, and capacity.")
    panel, tasks, candidates, capacity = _read_model_inputs()
    logging.info(
        "Inputs loaded: %d demand rows, %d tasks, %d candidate-region rows, %d capacity rows.",
        len(panel),
        len(tasks),
        len(candidates),
        len(capacity),
    )

    logging.info(
        "Forecast stage 1/7: compare local-mean windows %s on validation hours 2352--2375.",
        HISTORY_WINDOW_CANDIDATES,
    )
    selected_window_hours, window_selection = _select_history_window(panel)
    logging.info(
        "Window selected: H*=%d hours by overall validation WAPE; test data were excluded from selection.",
        selected_window_hours,
    )
    logging.info("Forecast stage 2/7: compute hierarchical and 24-hour baseline forecasts for validation hours 2352--2375 using H*.")
    validation_predictions, validation_hierarchy, validation_parameters = _forecast_window(
        panel,
        VALIDATION_START_HOUR,
        VALIDATION_END_HOUR,
        TRAIN_END_HOUR,
        selected_window_hours,
    )
    validation_predictions["Split"] = "validation"
    validation_hierarchy["Split"] = "validation"
    logging.info("Validation forecasts completed: %d rows.", len(validation_predictions))
    logging.info("Forecast stage 3/7: predict test hours 2376--2399 using information from hours 0--2375.")
    test_predictions, test_hierarchy, test_parameters = _forecast_window(
        panel,
        TEST_START_HOUR,
        TEST_END_HOUR,
        VALIDATION_END_HOUR,
        selected_window_hours,
    )
    test_predictions["Split"] = "test"
    test_hierarchy["Split"] = "test"
    logging.info("Test forecasts completed: %d rows.", len(test_predictions))
    predictions = pd.concat(
        [validation_predictions, test_predictions], ignore_index=True
    )
    predictions = predictions[
        [
            "Split",
            "Model",
            *SERIES_COLUMNS,
            "Hour",
            "Actual_GPU_Demand",
            "Prediction",
            "FitEndHour",
            "HistoryWindowHours",
        ]
    ].sort_values(["Split", *SERIES_COLUMNS, "Hour", "Model"], kind="stable")
    hierarchy_predictions = pd.concat(
        [validation_hierarchy, test_hierarchy], ignore_index=True
    ).sort_values(["Split", "Level", "Entity", "Hour"], kind="stable")
    logging.info("Forecast stage 4/7: aggregate WAPE, RMSE, and MAE at bottom, region, type, and system levels.")
    metrics = _build_metrics(predictions)
    hierarchy_metrics = _build_hierarchy_metrics(hierarchy_predictions)
    parameters = _build_forecast_parameters(validation_parameters, test_parameters)
    aggregation_sensitivity = _build_aggregation_sensitivity(predictions)
    logging.info("Forecast stage 5/7: run 68 historical 24-hour rolling backtest windows.")
    rolling_details, rolling_summary = _build_rolling_backtest(
        panel, selected_window_hours
    )
    _write_table(predictions, "forecast_predictions.csv")
    _write_table(metrics, "forecast_metrics.csv")
    _write_table(hierarchy_predictions, "forecast_hierarchy_predictions.csv")
    _write_table(hierarchy_metrics, "forecast_hierarchy_metrics.csv")
    _write_table(parameters, "forecast_parameters.csv")
    _write_table(window_selection, "forecast_window_selection.csv")
    _write_table(aggregation_sensitivity, "forecast_aggregation_sensitivity.csv")
    _write_table(rolling_details, "forecast_rolling_backtest.csv")
    _write_table(rolling_summary, "forecast_rolling_summary.csv")
    logging.info(
        "Forecast stage 6/7 completed: bottom forecasts, margins, window selection, scale summaries, and backtests saved."
    )

    logging.info("Scheduling stage 1/3: select actual arrivals during hours 2376--2399 and generate hourly candidates.")
    dispatch_tasks = tasks.loc[
        tasks["ArrivalHour"].between(TEST_START_HOUR, TEST_END_HOUR)
    ].copy()
    _, options = _build_dispatch_options(tasks, candidates)
    logging.info("Scheduling stage 2/3: invoke the two-stage MILP solver.")
    (
        selected,
        f1_star,
        f2_star,
        stage_one_status,
        stage_two_status,
        stage_one_message,
        stage_two_message,
        stage_one_mip_gap,
        stage_two_mip_gap,
        stage_one_mip_dual_bound,
        stage_two_mip_dual_bound,
    ) = _solve_two_stage_dispatch(options, dispatch_tasks, capacity)
    assignments = _build_dispatch_assignments(options, selected)
    profile = _build_dispatch_resource_profile(options, selected, capacity)
    _validate_dispatch_result(profile, assignments)
    summary = _build_dispatch_summary(
        assignments,
        profile,
        f1_star,
        f2_star,
        stage_one_status,
        stage_two_status,
        stage_one_message,
        stage_two_message,
        stage_one_mip_gap,
        stage_two_mip_gap,
        stage_one_mip_dual_bound,
        stage_two_mip_dual_bound,
    )
    _write_table(assignments, "dispatch_assignments.csv")
    _write_table(profile, "dispatch_resource_profile.csv")
    _write_table(summary, "dispatch_summary.csv")
    logging.info("Scheduling stage 3/3 completed: assignments, resource profiles, and solver summary saved.")
    logging.info("Forecast stage 7/7: Q1 hierarchical forecasts, baseline comparisons, and two-stage schedules are saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
