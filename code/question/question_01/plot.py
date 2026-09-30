"""Three groups of Question 1 paper figures.

Read existing input/result CSV files without refitting forecasts or solving
schedules. Produce three static PDFs:

1. Regional/task-type GPU demand structure.
2. Forecasts, schedules, and regional GPU utilization.
3. Rolling forecast validation, hierarchical reconciliation, and stress bounds.

Colors use the supplied plotting palette document, including its colorful and
blue-gradient series. Gray and black are excluded from data, text, axes, grids,
reference lines, and borders.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


QUESTION_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = QUESTION_DIR / "data" / "processed"
SHARED_DIR = PROCESSED_DIR / "shared"
Q1_DIR = PROCESSED_DIR / "q1"
TABLES_DIR = QUESTION_DIR / "outputs" / "tables"
FIGURES_DIR = QUESTION_DIR / "outputs" / "figures"

TRAIN_END_HOUR = 2351
VALIDATION_START_HOUR = 2352
VALIDATION_END_HOUR = 2375
TEST_START_HOUR = 2376
TEST_END_HOUR = 2399
TAIL_END_HOUR = 2405
TERMINAL_HOUR = 2406

TASK_TYPE_ORDER = ["RealTimeInference", "BatchInference", "AITraining"]
TASK_TYPE_LABELS = {
    "RealTimeInference": "Real-time inference",
    "BatchInference": "Batch inference",
    "AITraining": "AI training",
}
REGION_ORDER = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]

# All colors come from the user-supplied plotting palette document.
# The first six colors use palette 3 of the seven-color series; dark blue and light backgrounds use
# the six-color and three-color series. Gray, black, and near-gray colors are excluded.
PALETTE = {
    "red": "#D43F3A",
    "gold": "#EEA236",
    "green": "#5CB85C",
    "cyan": "#46B8DA",
    "blue": "#357EBD",
    "purple": "#9632B8",
    "deep_blue": "#003967",
    "light_blue": "#BADEFA",
    "lighter_blue": "#ACD6EC",
    "light_green": "#D4E6BC",
    "light_yellow": "#FFF8C4",
    "light_orange": "#F5D341",
    "pink": "#D77071",
    "white": "#FFFFFF",
}

DOCX_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "q1_docx_blue",
    [
        "#E3F2FD",
        "#BADEFA",
        "#90CAF8",
        "#64B4F6",
        "#41A5F4",
        "#2096F2",
        "#1E87E5",
        "#1976D2",
        "#1465BF",
        "#0C46A0",
    ],
)

REGION_COLORS = {
    "RegionA": PALETTE["red"],
    "RegionB": PALETTE["gold"],
    "RegionC": PALETTE["green"],
    "RegionD": PALETTE["cyan"],
    "RegionE": PALETTE["blue"],
    "RegionF": PALETTE["purple"],
}
TASK_COLORS = {
    "RealTimeInference": PALETTE["blue"],
    "BatchInference": PALETTE["green"],
    "AITraining": PALETTE["purple"],
}
MODEL_COLORS = {
    "actual": PALETTE["red"],
    "HierarchicalLocalMean": PALETTE["blue"],
    "SameHour24Baseline": PALETTE["gold"],
}
LEVEL_COLORS = {
    "Bottom": PALETTE["cyan"],
    "Region": PALETTE["green"],
    "TaskType": PALETTE["gold"],
    "System": PALETTE["purple"],
}

TEXT_COLOR = PALETTE["deep_blue"]
GRID_COLOR = PALETTE["lighter_blue"]
REFERENCE_COLOR = PALETTE["pink"]


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "text.color": TEXT_COLOR,
            "axes.edgecolor": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "axes.labelsize": 8.4,
            "axes.titlesize": 9.6,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "xtick.labelsize": 6.9,
            "ytick.labelsize": 6.9,
            "legend.labelcolor": TEXT_COLOR,
            "legend.fontsize": 6.5,
            "figure.facecolor": PALETTE["white"],
            "axes.facecolor": PALETTE["white"],
            "savefig.facecolor": PALETTE["white"],
        }
    )


def _read_csv(path: Path, required: tuple[str, ...] = ()) -> pd.DataFrame | None:
    if not path.is_file():
        logging.warning("Missing plot input; skipping: %s", path)
        return None
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        logging.warning("Missing plot columns; skipping %s: %s", path.name, missing)
        return None
    return frame


def _read_table(filename: str, required: tuple[str, ...] = ()) -> pd.DataFrame | None:
    return _read_csv(TABLES_DIR / filename, required)


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _region_color(region: str, index: int = 0) -> str:
    if region in REGION_COLORS:
        return REGION_COLORS[region]
    return list(REGION_COLORS.values())[index % len(REGION_COLORS)]


def _task_label(task_type: str) -> str:
    return TASK_TYPE_LABELS.get(str(task_type), str(task_type))


def _style_axis(axis: plt.Axes, grid_axis: str = "y") -> None:
    if grid_axis == "none":
        axis.grid(False)
    else:
        axis.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=0.65, alpha=0.75)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    axis.xaxis.label.set_color(TEXT_COLOR)
    axis.yaxis.label.set_color(TEXT_COLOR)
    axis.title.set_color(TEXT_COLOR)
    for spine in axis.spines.values():
        spine.set_color(TEXT_COLOR)
        spine.set_linewidth(0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _style_colorbar(colorbar: matplotlib.colorbar.Colorbar, label: str) -> None:
    colorbar.set_label(label, color=TEXT_COLOR)
    colorbar.ax.tick_params(colors=TEXT_COLOR)
    colorbar.outline.set_edgecolor(TEXT_COLOR)
    colorbar.outline.set_linewidth(0.7)


def _legend(axis: plt.Axes, **kwargs) -> None:
    legend = axis.legend(frameon=False, **kwargs)
    if legend is not None:
        for label in legend.get_texts():
            label.set_color(TEXT_COLOR)


def _panel_caption(
    axis: plt.Axes,
    text: str,
    *,
    y: float = -0.24,
    fontsize: float = 9.2,
    linespacing: float = 1.15,
) -> None:
    """Place subplot captions (a), (b), (c), etc. consistently below the axes."""
    axis.text(
        0.5,
        y,
        text,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        linespacing=linespacing,
        color=TEXT_COLOR,
        clip_on=False,
    )


def _save(figure: plt.Figure, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES_DIR / f"{stem}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)
    logging.info("Generated: %s.pdf", stem)


def _shade_forecast_axis(axis: plt.Axes) -> None:
    axis.axvspan(
        VALIDATION_START_HOUR - 0.5,
        VALIDATION_END_HOUR + 0.5,
        color=PALETTE["light_blue"],
        alpha=0.38,
        linewidth=0,
    )
    axis.axvspan(
        TEST_START_HOUR - 0.5,
        TEST_END_HOUR + 0.5,
        color=PALETTE["light_yellow"],
        alpha=0.55,
        linewidth=0,
    )
    axis.axvline(
        TEST_START_HOUR - 0.5,
        color=PALETTE["gold"],
        linestyle="--",
        linewidth=1.0,
    )


def _set_phase_labels(axis: plt.Axes, *, y: float = 0.90, fontsize: float = 8.0) -> None:
    axis.text(
        (VALIDATION_START_HOUR + VALIDATION_END_HOUR) / 2,
        y,
        "Validation",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=fontsize,
        color=PALETTE["blue"],
    )
    axis.text(
        (TEST_START_HOUR + TEST_END_HOUR) / 2,
        y,
        "Test",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=fontsize,
        color=PALETTE["gold"],
    )


def plot_group1_demand_structure() -> bool:
    """Group 1: demand heatmap, cumulative matrix, and validation/test forecast curves."""
    panel = _read_csv(
        Q1_DIR / "hourly_demand_panel.csv",
        ("Hour", "SourceRegion", "TaskType", "GPU_Demand_Arrival"),
    )
    if panel is None:
        return False
    panel = _numeric(panel, ("Hour", "GPU_Demand_Arrival"))
    panel = panel.loc[panel["Hour"].between(0, TEST_END_HOUR)].copy()
    if panel.empty:
        return False

    regions = [region for region in REGION_ORDER if region in set(panel["SourceRegion"].astype(str))]
    regions.extend(sorted(set(panel["SourceRegion"].astype(str)).difference(regions)))
    task_types = [task for task in TASK_TYPE_ORDER if task in set(panel["TaskType"].astype(str))]
    task_types.extend(sorted(set(panel["TaskType"].astype(str)).difference(task_types)))
    combinations = pd.MultiIndex.from_product(
        [regions, task_types], names=["SourceRegion", "TaskType"]
    )
    hours = pd.Index(range(TEST_END_HOUR + 1), name="Hour")
    hourly = (
        panel.groupby(["SourceRegion", "TaskType", "Hour"], as_index=True)["GPU_Demand_Arrival"]
        .sum()
        .unstack("Hour")
        .reindex(index=combinations, columns=hours, fill_value=0.0)
        .fillna(0.0)
    )
    labels = [f"{region}－{_task_label(task)}" for region, task in combinations]

    cumulative = (
        panel.groupby(["SourceRegion", "TaskType"])["GPU_Demand_Arrival"]
        .sum()
        .unstack("TaskType")
        .reindex(index=regions, columns=task_types, fill_value=0.0)
        .fillna(0.0)
    )

    # Approximately 17.5 cm wide for direct insertion into the A4 text area.
    figure = plt.figure(figsize=(6.9, 7.75))
    grid = figure.add_gridspec(
        3,
        1,
        height_ratios=[1.12, 0.84, 0.94],
        hspace=0.52,
    )
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[2, 0]),
    ]
    figure.subplots_adjust(left=0.10, right=0.965, bottom=0.095, top=0.985)

    image = axes[0].imshow(
        hourly.to_numpy(dtype=float),
        aspect="auto",
        interpolation="nearest",
        cmap=DOCX_BLUE_CMAP,
        extent=(-0.5, TEST_END_HOUR + 0.5, len(labels) - 0.5, -0.5),
    )
    axes[0].set_yticks(np.arange(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=7.0)
    axes[0].set_xticks([0, 600, 1200, 1800, 2399])
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Region / task type")
    _panel_caption(axes[0], "(a) Hourly GPU demand for 18 categories", y=-0.25, fontsize=9.4)
    _style_axis(axes[0], grid_axis="none")
    colorbar = figure.colorbar(image, ax=axes[0], fraction=0.026, pad=0.02)
    _style_colorbar(colorbar, "Absolute GPU demand")

    cumulative_image = axes[1].imshow(
        cumulative.to_numpy(dtype=float),
        aspect="auto",
        interpolation="nearest",
        cmap=DOCX_BLUE_CMAP,
    )
    axes[1].set_xticks(np.arange(len(task_types)))
    axes[1].set_xticklabels([_task_label(task) for task in task_types], rotation=14, ha="right", fontsize=7.0)
    axes[1].set_yticks(np.arange(len(regions)))
    axes[1].set_yticklabels(regions)
    axes[1].set_xlabel("Task type")
    axes[1].set_ylabel("Region")
    _panel_caption(axes[1], "(b) 6 x 3 cumulative GPU demand matrix", y=-0.36, fontsize=9.4)
    maximum = float(np.nanmax(cumulative.to_numpy(dtype=float))) if cumulative.size else 0.0
    threshold = maximum * 0.52
    for row_index, region in enumerate(regions):
        for column_index, task_type in enumerate(task_types):
            value = float(cumulative.loc[region, task_type])
            axes[1].text(
                column_index,
                row_index,
                f"{value:,.0f}",
                ha="center",
                va="center",
                fontsize=6.7,
                color=PALETTE["white"] if value >= threshold else TEXT_COLOR,
            )
    _style_axis(axes[1], grid_axis="none")
    colorbar = figure.colorbar(cumulative_image, ax=axes[1], fraction=0.046, pad=0.04)
    _style_colorbar(colorbar, "Cumulative GPU demand")

    _plot_forecast_panel(axes[2])
    _save(figure, "q1_group1_demand_structure")
    return True


def _forecast_total_series() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    predictions = _read_table(
        "forecast_predictions.csv",
        (
            "Split",
            "Model",
            "SourceRegion",
            "TaskType",
            "Hour",
            "Actual_GPU_Demand",
            "Prediction",
        ),
    )
    if predictions is None:
        return None
    predictions = _numeric(predictions, ("Hour", "Actual_GPU_Demand", "Prediction"))
    predictions = predictions.loc[
        predictions["Split"].astype(str).isin(["validation", "test"])
        & predictions["Hour"].between(VALIDATION_START_HOUR, TEST_END_HOUR)
    ].copy()
    if predictions.empty:
        return None
    actual = (
        predictions.drop_duplicates(["Split", "SourceRegion", "TaskType", "Hour"])
        .groupby("Hour", as_index=False)["Actual_GPU_Demand"]
        .sum()
        .rename(columns={"Actual_GPU_Demand": "Actual"})
    )
    model_totals = (
        predictions.groupby(["Model", "Hour"], as_index=False)["Prediction"]
        .sum()
        .rename(columns={"Prediction": "Predicted"})
    )
    return actual, model_totals


def _plot_forecast_panel(axis: plt.Axes) -> bool:
    series = _forecast_total_series()
    if series is None:
        axis.text(0.5, 0.5, "Test forecasts unavailable", ha="center", va="center", color=TEXT_COLOR)
        axis.set_axis_off()
        return False
    actual, model_totals = series
    _shade_forecast_axis(axis)
    _set_phase_labels(axis, y=0.87, fontsize=7.0)
    axis.plot(
        actual["Hour"],
        actual["Actual"],
        color=MODEL_COLORS["actual"],
        linewidth=1.7,
        marker="o",
        markersize=3.2,
        markerfacecolor=MODEL_COLORS["actual"],
        markeredgecolor=MODEL_COLORS["actual"],
        label="Actual",
        zorder=4,
    )
    for model in ("HierarchicalLocalMean", "SameHour24Baseline"):
        subset = model_totals.loc[model_totals["Model"] == model].sort_values("Hour")
        if subset.empty:
            continue
        axis.plot(
            subset["Hour"],
            subset["Predicted"],
            color=MODEL_COLORS[model],
            linewidth=1.8 if model == "HierarchicalLocalMean" else 1.35,
            linestyle="-" if model == "HierarchicalLocalMean" else "--",
            label="Hierarchical local mean" if model == "HierarchicalLocalMean" else "24-hour same-hour baseline",
            zorder=3,
        )
    axis.set_xlim(TEST_START_HOUR - 0.5, TEST_END_HOUR + 0.5)
    axis.set_xticks([2352, 2364, 2376, 2388, 2399])
    axis.set_xlabel("Hour")
    axis.set_ylabel("System GPU demand")
    _panel_caption(
        axis,
        "(c) System GPU forecasts, hours 2352--2399 (validation to test)",
        y=-0.27,
        fontsize=9.0,
    )
    metrics = _read_table(
        "forecast_metrics.csv",
        ("Split", "Model", "SourceRegion", "TaskType", "WAPE"),
    )
    if metrics is not None:
        metrics = _numeric(metrics, ("WAPE",))
        metrics = metrics.loc[
            (metrics["SourceRegion"] == "ALL") & (metrics["TaskType"] == "ALL")
        ]
        values: dict[tuple[str, str], float] = {}
        for row in metrics.itertuples(index=False):
            values[(str(row.Split), str(row.Model))] = float(row.WAPE)
        main_model = "HierarchicalLocalMean"
        baseline = "SameHour24Baseline"
        if all((split, model) in values for split in ("validation", "test") for model in (main_model, baseline)):
            metric_text = (
                "WAPE (hierarchical / 24-hour baseline)\n"
                f"Validation {values[('validation', main_model)]:.4f}/{values[('validation', baseline)]:.4f}\n"
                f"Test {values[('test', main_model)]:.4f}/{values[('test', baseline)]:.4f}"
            )
            axis.text(
                0.985,
                0.97,
                metric_text,
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=5.9,
                linespacing=1.10,
                color=PALETTE["deep_blue"],
                bbox={
                    "boxstyle": "round,pad=0.26",
                    "facecolor": PALETTE["white"],
                    "edgecolor": PALETTE["blue"],
                },
            )
    _legend(axis, loc="upper center", bbox_to_anchor=(0.50, 0.995), fontsize=5.8, ncol=3, columnspacing=1.05, handlelength=1.7)
    _style_axis(axis)
    return True


def _plot_gantt_panel(axis: plt.Axes) -> bool:
    assignments = _read_table(
        "dispatch_assignments.csv",
        (
            "TaskID",
            "TaskType",
            "SourceRegion",
            "TargetRegion",
            "StartHour",
            "FinishHour",
            "GPU_Demand",
        ),
    )
    if assignments is None:
        axis.text(0.5, 0.5, "Task schedules unavailable", ha="center", va="center", color=TEXT_COLOR)
        axis.set_axis_off()
        return False
    assignments = _numeric(
        assignments,
        ("StartHour", "FinishHour", "GPU_Demand", "Duration_h", "WaitHours"),
    )
    assignments["Duration_h"] = assignments["FinishHour"] - assignments["StartHour"]
    assignments["Workload_GPUh"] = assignments["GPU_Demand"] * assignments["Duration_h"]
    assignments = assignments.dropna(subset=["StartHour", "FinishHour", "Workload_GPUh"])
    if assignments.empty:
        axis.text(0.5, 0.5, "No scheduled tasks to plot", ha="center", va="center", color=TEXT_COLOR)
        axis.set_axis_off()
        return False
    # Sample representative tasks by type, retaining the temporal flexibility of all three types.
    # AI training remains the largest group. Rank each type by GPU-hours, then GPU demand.
    quotas = {"RealTimeInference": 6, "BatchInference": 6, "AITraining": 24}
    selected_parts: list[pd.DataFrame] = []
    for task_type in TASK_TYPE_ORDER:
        subset = assignments.loc[assignments["TaskType"].astype(str) == task_type].copy()
        if subset.empty:
            continue
        selected_parts.append(
            subset.sort_values(
                ["Workload_GPUh", "GPU_Demand", "StartHour"],
                ascending=[False, False, True],
                kind="stable",
            ).head(quotas[task_type])
        )
    display = pd.concat(selected_parts, ignore_index=True) if selected_parts else assignments.head(0)
    target_count = min(36, len(assignments))
    if len(display) < target_count:
        selected_ids = set(display["TaskID"].astype(str))
        remaining = assignments.loc[~assignments["TaskID"].astype(str).isin(selected_ids)]
        display = pd.concat(
            [
                display,
                remaining.sort_values(
                    ["Workload_GPUh", "GPU_Demand", "StartHour"],
                    ascending=[False, False, True],
                    kind="stable",
                ).head(target_count - len(display)),
            ],
            ignore_index=True,
        )
    display = display.sort_values(["StartHour", "FinishHour"], kind="stable").reset_index(drop=True)
    axis.axvspan(2400, TAIL_END_HOUR + 0.5, color=PALETTE["light_yellow"], alpha=0.55, linewidth=0)
    for row_index, row in enumerate(display.itertuples(index=False)):
        task_color = TASK_COLORS.get(str(row.TaskType), PALETTE["cyan"])
        axis.barh(
            row_index,
            float(row.FinishHour - row.StartHour),
            left=float(row.StartHour),
            height=0.76,
            color=task_color,
            edgecolor=task_color,
            linewidth=1.0,
        )
    axis.set_yticks(np.arange(len(display)))
    axis.set_yticklabels(
        [f"T{task_id}" if index % 2 == 0 else "" for index, task_id in enumerate(display["TaskID"].astype(str))],
        fontsize=6.0,
    )
    axis.invert_yaxis()
    # Reserve a narrow bottom strip for the legend to avoid crowding between subplots.
    axis.set_ylim(len(display) + 2.6, -0.8)
    axis.set_xlim(TEST_START_HOUR, TERMINAL_HOUR + 0.2)
    axis.set_xticks([2376, 2384, 2392, 2400, 2406])
    axis.set_xlabel("Hour")
    axis.set_ylabel("Representative tasks (36 sampled by type)", fontsize=8)
    _panel_caption(axis, "(a) Representative task Gantt chart, hours 2376--2406", y=-0.24, fontsize=9.3)
    axis.axvline(2400, color=PALETTE["gold"], linestyle=":", linewidth=1.0)
    axis.axvline(TERMINAL_HOUR, color=PALETTE["red"], linestyle="--", linewidth=1.0)
    axis.text(2400.15, 1.01, "Tail", transform=axis.get_xaxis_transform(), color=PALETTE["gold"], fontsize=8)
    axis.text(TERMINAL_HOUR - 0.1, 1.01, "Hour 2406 boundary", transform=axis.get_xaxis_transform(), ha="right", color=PALETTE["red"], fontsize=8)
    handles = [
        Patch(facecolor=TASK_COLORS[task], edgecolor=TASK_COLORS[task], label=_task_label(task))
        for task in TASK_TYPE_ORDER
    ]
    dispatch_summary = _read_table("dispatch_summary.csv", ("F2_Flexible_WaitHours",))
    if dispatch_summary is not None and not dispatch_summary.empty:
        f2 = float(pd.to_numeric(dispatch_summary.iloc[0]["F2_Flexible_WaitHours"], errors="coerce"))
        axis.text(
            0.745,
            0.965,
            f"F2*= {f2:.0f} h\nAll-local feasible",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            color=TEXT_COLOR,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": PALETTE["light_yellow"], "edgecolor": PALETTE["gold"]},
        )
    _legend(axis, handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=3, fontsize=6.1, columnspacing=1.35, handlelength=1.8)
    _style_axis(axis)
    return True


def _plot_utilization_panels(figure: plt.Figure, spec) -> bool:
    profile = _read_table(
        "dispatch_resource_profile.csv",
        (
            "Hour",
            "Region",
            "Available_GPU",
            "Scheduled_GPU_Equivalent",
        ),
    )
    if profile is None:
        return False
    profile = _numeric(
        profile,
        ("Hour", "Available_GPU", "Scheduled_GPU_Equivalent", "GPU_Utilization"),
    )
    if "GPU_Utilization" not in profile.columns:
        profile["GPU_Utilization"] = profile["Scheduled_GPU_Equivalent"] / profile["Available_GPU"].replace(0, np.nan)
    profile = profile.loc[profile["Hour"].between(TEST_START_HOUR, TAIL_END_HOUR)].copy()
    if profile.empty:
        return False
    regions = [region for region in REGION_ORDER if region in set(profile["Region"].astype(str))]
    regions.extend(sorted(set(profile["Region"].astype(str)).difference(regions)))
    if not regions:
        return False
    inner = spec.subgridspec(
        4, 2,
        height_ratios=[1, 1, 1, 0.12],
        hspace=0.22,
        wspace=0.20,
    )
    for index, region in enumerate(regions[:6]):
        row = index // 2
        col = index % 2
        axis = figure.add_subplot(inner[row, col])
        subset = profile.loc[profile["Region"].astype(str) == region].sort_values("Hour")
        utilization = subset["GPU_Utilization"].to_numpy(dtype=float)
        if np.nanmax(np.abs(utilization)) <= 1.5:
            utilization = utilization * 100.0
        if not subset.empty:
            axis.axvspan(
                2400, TAIL_END_HOUR + 0.5,
                color=PALETTE["light_yellow"], alpha=0.45, linewidth=0
            )
            axis.plot(
                subset["Hour"],
                utilization,
                color=_region_color(region, index),
                linewidth=1.35,
                marker="o",
                markersize=2.2,
            )
        axis.axhline(100, color=PALETTE["red"], linestyle="--", linewidth=0.8)
        axis.set_ylim(0, 105)
        axis.set_yticks([0, 50, 100])
        if col == 0:
            axis.set_yticklabels(["0%", "50%", "100%"], fontsize=6.1)
        else:
            axis.set_yticklabels([])
            axis.set_ylabel("")
        axis.set_xticks([2376, 2400, 2405])
        if row == 2:
            axis.set_xticklabels(["2376", "2400", "2405"], fontsize=6.1)
            axis.set_xlabel("Hour", fontsize=7.4)
        else:
            axis.set_xticklabels([])
            axis.set_xlabel("")
        axis.set_title(
            region, fontsize=7.2, fontweight="bold",
            color=_region_color(region, index), pad=2
        )
        _style_axis(axis)

    caption_axis = figure.add_subplot(inner[3, :])
    caption_axis.axis("off")
    caption_axis.text(
        0.5,
        0.56,
        "(c) Hourly GPU utilization in six regions",
        ha="center",
        va="top",
        fontsize=9.3,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    return True


def _plot_execution_matrix_panel(axis: plt.Axes) -> bool:
    assignments = _read_table(
        "dispatch_assignments.csv",
        (
            "TaskID",
            "SourceRegion",
            "TargetRegion",
            "StartHour",
            "FinishHour",
            "GPU_Demand",
        ),
    )
    if assignments is None:
        axis.text(0.5, 0.5, "Regional execution results unavailable", ha="center", va="center", color=TEXT_COLOR)
        axis.set_axis_off()
        return False
    assignments = _numeric(assignments, ("StartHour", "FinishHour", "GPU_Demand"))
    assignments["Workload_GPUh"] = assignments["GPU_Demand"] * (
        assignments["FinishHour"] - assignments["StartHour"]
    )
    assignments = assignments.dropna(subset=["Workload_GPUh"])
    if assignments.empty:
        axis.text(0.5, 0.5, "No regional execution results to plot", ha="center", va="center", color=TEXT_COLOR)
        axis.set_axis_off()
        return False
    regions = [region for region in REGION_ORDER if region in set(assignments["SourceRegion"].astype(str)) | set(assignments["TargetRegion"].astype(str))]
    all_regions = set(assignments["SourceRegion"].astype(str)) | set(assignments["TargetRegion"].astype(str))
    regions.extend(sorted(all_regions.difference(regions)))
    matrix = (
        assignments.assign(
            SourceRegion=assignments["SourceRegion"].astype(str),
            TargetRegion=assignments["TargetRegion"].astype(str),
        )
        .groupby(["SourceRegion", "TargetRegion"])["Workload_GPUh"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(index=regions, columns=regions, fill_value=0.0)
        .fillna(0.0)
    )
    image = axis.imshow(matrix.to_numpy(dtype=float), cmap=DOCX_BLUE_CMAP, aspect="auto")
    axis.set_xticks(np.arange(len(regions)))
    axis.set_xticklabels(regions, rotation=28, ha="right")
    axis.set_yticks(np.arange(len(regions)))
    axis.set_yticklabels(regions)
    axis.set_xlabel("Execution region")
    axis.set_ylabel("Source region")
    maximum = float(np.nanmax(matrix.to_numpy(dtype=float))) if matrix.size else 0.0
    for row_index, source in enumerate(regions):
        for column_index, target in enumerate(regions):
            value = float(matrix.loc[source, target])
            axis.text(
                column_index,
                row_index,
                f"{value:,.0f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=PALETTE["white"] if value >= maximum * 0.52 else TEXT_COLOR,
            )
    summary = _read_table(
        "dispatch_summary.csv",
        ("F1_Migration_GPU_Workload_GPUh", "LocalTaskRatio"),
    )
    if summary is not None and not summary.empty:
        f1 = float(pd.to_numeric(summary.iloc[0]["F1_Migration_GPU_Workload_GPUh"], errors="coerce"))
        local_ratio = float(pd.to_numeric(summary.iloc[0]["LocalTaskRatio"], errors="coerce"))
    else:
        f1 = float(assignments.loc[assignments["SourceRegion"].astype(str) != assignments["TargetRegion"].astype(str), "Workload_GPUh"].sum())
        local_ratio = float((assignments["SourceRegion"].astype(str) == assignments["TargetRegion"].astype(str)).mean())
    _panel_caption(
        axis,
        f"(b) Source-to-execution GPU workload matrix\n"
        f"F1*= {f1:.0f} GPU-hours; local execution share= {local_ratio:.0%}",
        y=-0.34,
        fontsize=8.5,
        linespacing=1.15,
    )
    _style_axis(axis, grid_axis="none")
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    _style_colorbar(colorbar, "")
    colorbar.ax.set_title("GPU·h", fontsize=6.5, color=TEXT_COLOR, pad=2)
    return True


def _plot_peak_resource_panel(axis: plt.Axes) -> bool:
    profile = _read_table(
        "dispatch_resource_profile.csv",
        (
            "Hour",
            "Region",
            "Available_GPU",
            "Scheduled_GPU_Equivalent",
            "Effective_AI_IT_Capacity_MW",
            "Scheduled_AI_IT_Load_MW",
            "Max_Facility_Power_MW",
            "Total_Facility_Load_MW",
        ),
    )
    if profile is None:
        axis.text(0.5, 0.5, "Resource profiles unavailable", ha="center", va="center", color=TEXT_COLOR)
        axis.set_axis_off()
        return False
    profile = _numeric(
        profile,
        (
            "Hour",
            "Available_GPU",
            "Scheduled_GPU_Equivalent",
            "GPU_Utilization",
            "AI_IT_Capacity_Utilization",
            "Effective_AI_IT_Capacity_MW",
            "Scheduled_AI_IT_Load_MW",
            "Max_Facility_Power_MW",
            "Total_Facility_Load_MW",
        ),
    )
    profile = profile.loc[profile["Hour"].between(TEST_START_HOUR, TAIL_END_HOUR)].copy()
    if profile.empty:
        return False
    if "GPU_Utilization" not in profile.columns:
        profile["GPU_Utilization"] = profile["Scheduled_GPU_Equivalent"] / profile["Available_GPU"].replace(0, np.nan)
    if "AI_IT_Capacity_Utilization" not in profile.columns:
        profile["AI_IT_Capacity_Utilization"] = profile["Scheduled_AI_IT_Load_MW"] / profile["Effective_AI_IT_Capacity_MW"].replace(0, np.nan)
    profile["Facility_Utilization"] = profile["Total_Facility_Load_MW"] / profile["Max_Facility_Power_MW"].replace(0, np.nan)
    summary = (
        profile.assign(
            GPU=profile["GPU_Utilization"],
            AI_IT=profile["AI_IT_Capacity_Utilization"],
            Facility=profile["Facility_Utilization"],
        )
        .groupby("Region", as_index=False)[["GPU", "AI_IT", "Facility"]]
        .max()
    )
    regions = [region for region in REGION_ORDER if region in set(summary["Region"].astype(str))]
    regions.extend(sorted(set(summary["Region"].astype(str)).difference(regions)))
    summary = summary.set_index("Region").reindex(regions)
    y = np.arange(len(regions), dtype=float)
    offsets = {"GPU": -0.18, "AI_IT": 0.0, "Facility": 0.18}
    resource_colors = {"GPU": PALETTE["blue"], "AI_IT": PALETTE["gold"], "Facility": PALETTE["purple"]}
    resource_labels = {"GPU": "GPU", "AI_IT": "AI IT capacity", "Facility": "Facility power"}
    for resource in ("GPU", "AI_IT", "Facility"):
        values = summary[resource].to_numpy(dtype=float) * 100.0
        axis.scatter(
            values,
            y + offsets[resource],
            s=40,
            color=resource_colors[resource],
            edgecolors=resource_colors[resource],
            label=resource_labels[resource],
            zorder=3,
        )
    axis.axvline(100, color=PALETTE["red"], linestyle="--", linewidth=1.0)
    axis.set_xlim(0, 105)
    axis.set_xticks([0, 25, 50, 75, 100])
    axis.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    axis.set_yticks(y)
    axis.set_yticklabels(regions)
    axis.invert_yaxis()
    axis.set_xlabel("Peak utilization / %")
    _panel_caption(axis, "(d) Regional peak utilization of three resources", y=-0.24, fontsize=9.2)
    _legend(axis, loc="upper left", bbox_to_anchor=(0.01, 0.99), fontsize=6.2, handletextpad=0.45, labelspacing=0.35)
    _style_axis(axis)
    return True


def plot_group2_core_results() -> bool:
    """Group 2: task Gantt chart, execution matrix, GPU utilization, and peak resource usage."""
    figure = plt.figure(figsize=(6.9, 8.45))
    grid = figure.add_gridspec(
        3,
        2,
        height_ratios=[1.00, 1.02, 1.60],
        width_ratios=[1.06, 1.0],
        hspace=0.50,
        wspace=0.27,
    )
    gantt_axis = figure.add_subplot(grid[0, :])
    matrix_axis = figure.add_subplot(grid[1, 0])
    gantt_ok = _plot_gantt_panel(gantt_axis)
    matrix_ok = _plot_execution_matrix_panel(matrix_axis)
    utilization_ok = _plot_utilization_panels(figure, grid[2, :])
    resource_axis = figure.add_subplot(grid[1, 1])
    resource_ok = _plot_peak_resource_panel(resource_axis)
    figure.subplots_adjust(left=0.075, right=0.98, bottom=0.055, top=0.988)
    _save(figure, "q1_group2_core_results")
    return gantt_ok or matrix_ok or utilization_ok or resource_ok


def _plot_wape_delta(axis: plt.Axes, paired: pd.DataFrame, summary: pd.DataFrame | None) -> bool:
    paired = _numeric(
        paired,
        ("WindowID", "Delta_HierarchicalMinusBaseline"),
    ).dropna(subset=["Delta_HierarchicalMinusBaseline"])
    if paired.empty:
        axis.text(0.5, 0.5, "Rolling WAPE differences unavailable", ha="center", va="center", color=TEXT_COLOR)
        axis.set_axis_off()
        return False
    paired = paired.sort_values("WindowID", kind="stable")
    x = paired["WindowID"].to_numpy(dtype=float)
    delta = paired["Delta_HierarchicalMinusBaseline"].to_numpy(dtype=float)
    axis.axhline(0.0, color=PALETTE["red"], linestyle="--", linewidth=1.0)
    axis.vlines(x, 0.0, delta, color=PALETTE["light_blue"], linewidth=0.55, alpha=0.75)
    axis.scatter(
        x,
        delta,
        s=19,
        color=PALETTE["blue"],
        edgecolors=PALETTE["deep_blue"],
        linewidths=0.3,
        zorder=3,
    )
    axis.set_xlim(float(x.min()) - 1, float(x.max()) + 1)
    axis.set_xlabel("Rolling window index", labelpad=2)
    axis.set_ylabel("WAPE difference\n(hierarchical model - 24-hour baseline)")
    n = len(paired)
    negative_count = int(np.sum(delta < 0.0))
    p_value = None
    if summary is not None and not summary.empty:
        p_column = "WilcoxonPValue_OneSidedLess"
        if p_column in summary.columns:
            p_value = float(pd.to_numeric(summary.iloc[0][p_column], errors="coerce"))
    if p_value is None or not np.isfinite(p_value):
        annotation = f"{negative_count}/{n}<0"
    else:
        annotation = f"{negative_count}/{n}<0\np={p_value:.2e}"
    axis.text(
        0.98,
        0.96,
        annotation,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.0,
        linespacing=1.05,
        color=TEXT_COLOR,
        bbox={"boxstyle": "round,pad=0.20", "facecolor": PALETTE["light_blue"], "edgecolor": PALETTE["blue"]},
    )
    _style_axis(axis)
    return True


def _plot_hierarchy_scatter(
    axis: plt.Axes,
    paired: pd.DataFrame,
    comparison: pd.DataFrame | None,
    consistency: pd.DataFrame | None,
    validation_summary: pd.DataFrame | None,
) -> bool:
    paired = _numeric(paired, ("HierarchicalWAPE", "DirectWAPE")).dropna(
        subset=["HierarchicalWAPE", "DirectWAPE"]
    ).copy()
    if paired.empty:
        axis.text(0.5, 0.5, "Hierarchical reconciliation comparisons unavailable", ha="center", va="center", color=TEXT_COLOR)
        axis.set_axis_off()
        return False
    x = paired["DirectWAPE"].to_numpy(dtype=float)
    y = paired["HierarchicalWAPE"].to_numpy(dtype=float)
    lower = float(np.nanmin(np.r_[x, y]))
    upper = float(np.nanmax(np.r_[x, y]))
    padding = max((upper - lower) * 0.06, 0.02)
    lower -= padding
    upper += padding
    axis.plot([lower, upper], [lower, upper], color=PALETTE["gold"], linestyle="--", linewidth=1.05, label="y=x")
    axis.scatter(
        x,
        y,
        s=20,
        color=PALETTE["cyan"],
        edgecolors=PALETTE["blue"],
        linewidths=0.35,
        alpha=0.72,
        label="Rolling window",
    )
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    # Use equal x/y ranges and aspect ratio to avoid visually distorting y=x.
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Direct forecast WAPE")
    axis.set_ylabel("Reconciled forecast WAPE")
    _panel_caption(axis, "(b) Hierarchical reconciliation versus direct forecasting", y=-0.25, fontsize=8.9)
    delta = float(np.median(y - x))
    p_value = None
    if comparison is not None and not comparison.empty:
        comparison = _numeric(
            comparison,
            ("Median_Delta_HierarchicalMinusDirect", "WilcoxonPValue_Less"),
        )
        if "Median_Delta_HierarchicalMinusDirect" in comparison.columns:
            delta = float(comparison.iloc[0]["Median_Delta_HierarchicalMinusDirect"])
        if "WilcoxonPValue_Less" in comparison.columns:
            p_value = float(comparison.iloc[0]["WilcoxonPValue_Less"])
    region_error = None
    task_error = None
    bottom_region_error = None
    if consistency is not None and not consistency.empty:
        consistency = _numeric(consistency, ("RegionSystemAbsDiff", "TaskTypeSystemAbsDiff"))
        region_error = float(consistency["RegionSystemAbsDiff"].max())
        task_error = float(consistency["TaskTypeSystemAbsDiff"].max())
    if validation_summary is not None and "Evidence" in validation_summary.columns:
        for evidence in validation_summary["Evidence"].astype(str):
            match = re.search(r"bottom_region_diff=([0-9.eE+-]+)", evidence)
            if match is not None:
                bottom_region_error = float(match.group(1))
                break
    annotation_lines = [f"Median difference={delta:.2e}"]
    if p_value is not None and np.isfinite(p_value):
        annotation_lines.append(f"p={p_value:.3f}")
    if region_error is not None and task_error is not None:
        annotation_lines.extend(
            [
                "Maximum closure error",
                f"Region={region_error:.2e}",
                f"Type={task_error:.2e}",
            ]
        )
    if bottom_region_error is not None:
        annotation_lines.append(f"18-category sum={bottom_region_error:.2e}")
    axis.text(
        0.04,
        0.96,
        "\n".join(annotation_lines),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color=TEXT_COLOR,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": PALETTE["light_blue"], "edgecolor": PALETTE["blue"]},
    )
    _legend(axis, loc="lower right", fontsize=5.8, handlelength=1.6, labelspacing=0.3)
    _style_axis(axis, grid_axis="both")
    return True


def _plot_pressure_interval(axis: plt.Axes, critical: pd.DataFrame | None) -> bool:
    if critical is None or critical.empty:
        axis.text(0.5, 0.5, "Stress bounds unavailable", ha="center", va="center", color=TEXT_COLOR)
        axis.set_axis_off()
        return False
    critical = _numeric(critical, ("AlphaLower", "AlphaUpper"))
    row_specs = [
        ("1h", "alpha_local", "1 h - all-local"),
        ("1h", "alpha_feas", "1 h - system feasible"),
        ("0.5h", "alpha_local", "0.5 h - all-local"),
        ("0.5h", "alpha_feas", "0.5 h - system feasible"),
    ]
    colors = {"1h": PALETTE["blue"], "0.5h": PALETTE["purple"]}
    y_values = np.arange(len(row_specs), dtype=float)
    max_alpha = 1.05
    plotted = False
    for y, (granularity, boundary, label) in zip(y_values, row_specs):
        subset = critical.loc[
            (critical["Granularity"].astype(str) == granularity)
            & (critical["Boundary"].astype(str) == boundary)
        ]
        if subset.empty:
            continue
        row = subset.iloc[0]
        lower = float(row["AlphaLower"])
        upper = float(row["AlphaUpper"])
        if not np.isfinite(lower) or not np.isfinite(upper):
            continue
        plotted = True
        max_alpha = max(max_alpha, upper)
        color = colors[granularity]
        axis.plot(
            [lower, upper],
            [y, y],
            color=PALETTE["lighter_blue"],
            linestyle="-",
            linewidth=1.35,
            solid_capstyle="round",
        )
        axis.scatter([lower], [y], s=42, color=color, edgecolors=color, zorder=4)
        axis.scatter(
            [upper],
            [y],
            s=46,
            facecolors=PALETTE["white"],
            edgecolors=color,
            linewidths=1.4,
            zorder=4,
        )
        axis.text(lower, y - 0.22, f"{lower:.4f}", ha="center", va="top", fontsize=7, color=color)
        axis.text(upper, y - 0.22, f"{upper:.4f}", ha="center", va="top", fontsize=7, color=color)

    if not plotted:
        axis.text(0.5, 0.5, "No stress-bound intervals to plot", ha="center", va="center", color=TEXT_COLOR)
        axis.set_axis_off()
        return False
    axis.axvline(1.0, color=PALETTE["red"], linestyle=":", linewidth=1.0)
    axis.text(
        1.02,
        0.04,
        "Current load alpha=1",
        transform=axis.get_xaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=6.3,
        color=PALETTE["red"],
    )
    axis.set_xlim(0.98, max_alpha + 0.12)
    axis.set_ylim(-0.55, len(row_specs) - 0.45)
    axis.set_yticks(y_values)
    axis.set_yticklabels([label for _, _, label in row_specs])
    axis.invert_yaxis()
    axis.set_xlabel("Load multiplier alpha")
    _panel_caption(axis, "(c) Stress-bound validation", y=-0.24, fontsize=8.9)
    legend_handles = [
        Line2D([0], [0], marker="o", color=PALETTE["blue"], markerfacecolor=PALETTE["blue"], markeredgecolor=PALETTE["blue"], linestyle="None", label="Filled: verified lower bound"),
        Line2D([0], [0], marker="o", color=PALETTE["blue"], markerfacecolor=PALETTE["white"], markeredgecolor=PALETTE["blue"], linestyle="None", label="Hollow: capacity upper bound"),
        Line2D([0], [0], color=PALETTE["lighter_blue"], linewidth=1.35, label="Interval: lower to capacity upper bound"),
    ]
    _legend(axis, handles=legend_handles, loc="upper center", bbox_to_anchor=(0.72, 0.995), ncol=1, fontsize=5.2, labelspacing=0.20, handlelength=1.6)
    _style_axis(axis)
    return True


def plot_group3_validation_robustness() -> bool:
    """Group 3: 84-window generalization, reconciliation consistency, and stress intervals."""
    paired = _read_table(
        "validation_paired_differences.csv",
        ("WindowID", "HierarchicalWAPE", "DirectWAPE", "Delta_HierarchicalMinusBaseline"),
    )
    test1_summary = _read_table(
        "validation_test1_summary.csv",
        ("WilcoxonPValue_OneSidedLess",),
    )
    comparison = _read_table(
        "validation_hierarchy_vs_direct.csv",
        ("Median_Delta_HierarchicalMinusDirect", "WilcoxonPValue_Less"),
    )
    consistency = _read_table(
        "validation_hierarchy_consistency.csv",
        ("RegionSystemAbsDiff", "TaskTypeSystemAbsDiff"),
    )
    validation_summary = _read_table("validation_summary.csv")
    critical = _read_table("validation_critical_pressure.csv", ("Granularity", "Boundary", "AlphaLower", "AlphaUpper"))
    if paired is None and critical is None:
        return False

    figure = plt.figure(figsize=(6.9, 5.82))
    grid = figure.add_gridspec(
        3, 5,
        height_ratios=[0.78, 0.10, 1.18],
        width_ratios=[1.0, 1.0, 0.12, 1.38, 1.38],
        hspace=0.22,
        wspace=0.42,
    )
    axes = [
        figure.add_subplot(grid[0, :]),
        figure.add_subplot(grid[2, 0:2]),
        figure.add_subplot(grid[2, 3:5]),
    ]

    # Place subplot (a) caption on its own line to avoid overlap with the rolling window label.
    caption_a = figure.add_subplot(grid[1, :])
    caption_a.axis("off")
    caption_a.text(
        0.5,
        0.48,
        "(a) WAPE differences across 84 rolling windows",
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color=TEXT_COLOR,
    )

    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.120, top=0.985)
    first_ok = _plot_wape_delta(axes[0], paired, test1_summary) if paired is not None else False
    second_ok = _plot_hierarchy_scatter(
        axes[1], paired, comparison, consistency, validation_summary
    ) if paired is not None else False
    third_ok = _plot_pressure_interval(axes[2], critical)
    _save(figure, "q1_group3_validation_robustness")
    return first_ok or second_ok or third_ok


def main() -> int:
    _configure_style()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    if not TABLES_DIR.is_dir() or not Q1_DIR.is_dir():
        logging.info("Question-specific plot input directory is missing; skipping plotting.")
        return 0
    figure_functions = [
        plot_group1_demand_structure,
        plot_group2_core_results,
        plot_group3_validation_robustness,
    ]
    generated = 0
    for function in figure_functions:
        logging.info("Plotting started: %s", function.__name__)
        if function():
            generated += 1
    logging.info("Q1 plotting completed: %d/3 figure groups generated in %s", generated, FIGURES_DIR)
    return 0 if generated == 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())