"""Three groups of Question 3 paper figures.

Read existing Q3 input/result CSV files without solving MILP or modifying model
results. Produce three static PDFs:

1. Storage time-shifting mechanisms.
2. Storage coordination effects and regional differences.
3. Physical closure, attachment-baseline diagnostics, and multiobjective selection.

Avoid overall figure titles and long in-figure explanations. Place each subplot
caption below its axes.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch


QUESTION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = QUESTION_DIR.parents[1]
SHARED_DIR = PROJECT_DIR / "question" / "question_01" / "data" / "processed" / "shared"
TABLES_DIR = QUESTION_DIR / "outputs" / "tables"
FIGURES_DIR = QUESTION_DIR / "outputs" / "figures"

REGION_ORDER = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
SCHEME_ORDER = ["NoStorage", "BaselineReference", "Balanced"]
SCHEME_LABELS = {
    "NoStorage": "No storage",
    "BaselineReference": "Attachment baseline",
    "Balanced": "Optimized solution",
}
# Use the Q1 plotting palette derived from the supplied document, mapped to Q3 semantics:
# green=renewables, blue=storage/purchases, orange-red=discharge/peaks/alerts, purple=attachment baseline.
SCHEME_COLORS = {
    "NoStorage": "#BADEFA",
    "BaselineReference": "#9632B8",
    "Balanced": "#357EBD",
}
TEXT_COLOR = "#003967"
GRID_COLOR = "#ACD6EC"
REFERENCE_COLOR = "#D77071"
WHITE = "#FFFFFF"

SIGNED_POWER_CMAP = LinearSegmentedColormap.from_list(
    "q3_signed_storage_power",
    ["#D77071", "#F7D9D4", WHITE, "#BADEFA", "#357EBD"],
)
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
            "axes.labelsize": 8.2,
            "axes.titlesize": 9.2,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.labelcolor": TEXT_COLOR,
            "legend.fontsize": 6.4,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
        }
    )


def _read_csv(path: Path, required: tuple[str, ...] = ()) -> pd.DataFrame | None:
    if not path.is_file():
        logging.error("Plot input missing: %s", path)
        return None
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        logging.error("%s is missing columns: %s", path.name, missing)
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


def _operating(frame: pd.DataFrame) -> pd.DataFrame:
    terminal = frame["TerminalStateOnly"].astype(str).str.strip().str.lower().isin(
        ("true", "1", "yes")
    )
    return frame.loc[~terminal].copy()


def _ordered_regions(values: pd.Series) -> list[str]:
    observed = set(values.astype(str))
    ordered = [region for region in REGION_ORDER if region in observed]
    ordered.extend(sorted(observed.difference(ordered)))
    return ordered


def _style_axis(axis: plt.Axes, grid_axis: str = "y") -> None:
    if grid_axis == "none":
        axis.grid(False)
    else:
        axis.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=0.6, alpha=0.75)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    axis.xaxis.label.set_color(TEXT_COLOR)
    axis.yaxis.label.set_color(TEXT_COLOR)
    for spine in axis.spines.values():
        spine.set_color(TEXT_COLOR)
        spine.set_linewidth(0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _legend(axis: plt.Axes, **kwargs) -> None:
    legend = axis.legend(frameon=False, **kwargs)
    if legend is not None:
        for label in legend.get_texts():
            label.set_color(TEXT_COLOR)


def _panel_caption(
    axis: plt.Axes,
    text: str,
    *,
    y: float = -0.25,
    fontsize: float = 9.0,
) -> None:
    """Place subplot captions below axes, separate from upper axis titles."""

    axis.text(
        0.5,
        y,
        text,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        color=TEXT_COLOR,
        clip_on=False,
    )


def _save(figure: plt.Figure, stem: str) -> None:
    """Save a PDF; if Windows preview locks the target, write *_new.pdf instead."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    target = FIGURES_DIR / f"{stem}.pdf"

    try:
        figure.savefig(
            target,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.03,
        )
        logging.info("Generated: %s", target.name)

    except PermissionError:
        fallback = FIGURES_DIR / f"{stem}_new.pdf"
        figure.savefig(
            fallback,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.03,
        )
        logging.warning(
            "%s is locked by another application; output redirected to %s",
            target.name,
            fallback.name,
        )

    finally:
        plt.close(figure)


def _load_balanced() -> pd.DataFrame | None:
    required = (
        "Hour",
        "Region",
        "RenewableCharge_MW",
        "GridCharge_MW",
        "ChargePower_MW",
        "DischargePower_MW",
        "SOC_MWh",
        "TerminalStateOnly",
    )
    frame = _read_table("q3_balanced_region_hour.csv", required)
    if frame is None:
        return None
    frame = _numeric(
        frame,
        (
            "Hour",
            "RenewableCharge_MW",
            "GridCharge_MW",
            "ChargePower_MW",
            "DischargePower_MW",
            "SOC_MWh",
        ),
    )
    return _operating(frame).sort_values(["Hour", "Region"], kind="stable")


def plot_group1_storage_mechanism() -> bool:
    """Figure 1: storage-power overview for six regions and SOC time shifts in active regions."""

    balanced = _load_balanced()
    storage = _read_csv(
        SHARED_DIR / "storage_params.csv",
        ("Region", "StorageCapacity_MWh", "MinSOC_MWh", "InitialSOC_MWh"),
    )
    if balanced is None or storage is None:
        return False

    balanced = balanced.dropna(
        subset=["Hour", "Region", "ChargePower_MW", "DischargePower_MW", "SOC_MWh"]
    )
    storage = _numeric(storage, ("StorageCapacity_MWh", "MinSOC_MWh", "InitialSOC_MWh"))
    regions = _ordered_regions(balanced["Region"])
    if not regions:
        return False
    params = storage.set_index("Region").reindex(regions)
    if params[["StorageCapacity_MWh", "MinSOC_MWh", "InitialSOC_MWh"]].isna().any().any():
        logging.error("storage_params.csv does not cover all Q3 regions")
        return False

    balanced = balanced.copy()
    balanced["SignedStoragePower_MW"] = (
        balanced["ChargePower_MW"] - balanced["DischargePower_MW"]
    )

    figure = plt.figure(figsize=(7.25, 7.20))
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=[1.06, 0.94],
        hspace=0.63,
    )

    # Keep only the heatmap and narrow colorbar in the upper area, without a statistics table.
    heat_grid = grid[0].subgridspec(
        1,
        2,
        width_ratios=[29.0, 1.0],
        wspace=0.08,
    )
    power_axis = figure.add_subplot(heat_grid[0, 0])
    colorbar_axis = figure.add_subplot(heat_grid[0, 1])

    power = (
        balanced.pivot(
            index="Region",
            columns="Hour",
            values="SignedStoragePower_MW",
        )
        .reindex(regions)
    )
    maximum = (
        float(np.nanmax(np.abs(power.to_numpy(dtype=float))))
        if power.size
        else 1.0
    )
    maximum = max(maximum, 1e-9)

    image = power_axis.imshow(
        power.to_numpy(dtype=float),
        aspect="auto",
        interpolation="nearest",
        cmap=SIGNED_POWER_CMAP,
        norm=TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum),
        extent=(-0.5, power.shape[1] - 0.5, len(regions) - 0.5, -0.5),
    )
    power_axis.set_yticks(np.arange(len(regions)))
    power_axis.set_yticklabels(regions)
    power_axis.set_xticks([0, 600, 1200, 1800, 2405])
    power_axis.set_xlabel("Time (t) / h", fontsize=10.2)
    power_axis.set_ylabel("Region", fontsize=10.2)
    _panel_caption(
        power_axis,
        "(a) Storage-power time series in six regions",
        y=-0.22,
        fontsize=9.2,
    )
    _style_axis(power_axis, grid_axis="none")

    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Net storage power / MW", color=TEXT_COLOR)
    colorbar.ax.tick_params(colors=TEXT_COLOR, labelsize=6.4)
    colorbar.outline.set_edgecolor(TEXT_COLOR)
    colorbar.outline.set_linewidth(0.6)

    # Retain only active RegionD--F in the lower area.
    active_regions = [
        region for region in ("RegionD", "RegionE", "RegionF")
        if region in regions
    ]
    if not active_regions:
        logging.error("Q3 results are missing active RegionD/RegionE/RegionF")
        plt.close(figure)
        return False

    soc_grid = grid[1].subgridspec(
        1,
        len(active_regions),
        wspace=0.28,
    )
    soc_axes = []

    for position, region in enumerate(active_regions):
        axis = figure.add_subplot(soc_grid[0, position])
        soc_axes.append(axis)

        subset = (
            balanced.loc[balanced["Region"].astype(str) == region]
            .sort_values("Hour")
        )
        axis.plot(
            subset["Hour"],
            subset["SOC_MWh"],
            color="#357EBD",
            linewidth=1.05,
            label="SOC",
        )

        initial = float(params.loc[region, "InitialSOC_MWh"])
        minimum = float(params.loc[region, "MinSOC_MWh"])
        capacity = float(params.loc[region, "StorageCapacity_MWh"])

        axis.axhline(
            initial,
            color=TEXT_COLOR,
            linestyle=":",
            linewidth=0.75,
            label="Initial SOC",
        )
        axis.axhline(
            minimum,
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=0.75,
            label="Minimum SOC",
        )
        axis.axhline(
            capacity,
            color="#5CB85C",
            linestyle="--",
            linewidth=0.75,
            label="Capacity limit",
        )

        axis.set_title(region, fontsize=8.3, pad=2.0)
        axis.set_xticks([0, 1200, 2405])
        axis.set_xlabel("Time t / h", fontsize=9.6)
        axis.set_ylabel("SOC / MWh", fontsize=9.6)
        _style_axis(axis)

        power_axis_twin = axis.twinx()
        signed_region = subset["SignedStoragePower_MW"].to_numpy(dtype=float)
        hours = subset["Hour"].to_numpy(dtype=float)
        power_axis_twin.plot(
            hours,
            signed_region,
            color="#D77071",
            linewidth=0.42,
            alpha=0.62,
            label="Signed power",
        )
        power_axis_twin.axhline(
            0.0,
            color="#D77071",
            linestyle="--",
            linewidth=0.42,
            alpha=0.65,
        )
        power_axis_twin.set_yticks([])
        power_axis_twin.spines["top"].set_visible(False)
        power_axis_twin.spines["right"].set_visible(False)

    # Use a shared lower-row legend to avoid overlap with dense SOC curves.
    legend_handles = [
        plt.Line2D([], [], color="#357EBD", linewidth=1.05, label="SOC"),
        plt.Line2D([], [], color=TEXT_COLOR, linestyle=":", linewidth=0.75, label="Initial SOC"),
        plt.Line2D([], [], color=REFERENCE_COLOR, linestyle="--", linewidth=0.75, label="Minimum SOC"),
        plt.Line2D([], [], color="#5CB85C", linestyle="--", linewidth=0.75, label="Capacity limit"),
        plt.Line2D([], [], color="#D77071", linewidth=0.55, label="Signed power"),
    ]
    legend = figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.475),
        ncol=5,
        frameon=False,
        fontsize=7.3,
        handlelength=1.85,
        columnspacing=1.35,
    )
    for label in legend.get_texts():
        label.set_color(TEXT_COLOR)

    anchor_axis = soc_axes[1 if len(soc_axes) > 1 else 0]
    _panel_caption(
        anchor_axis,
        "(b) RegionD--F SOC and charging/discharging time shifts",
        y=-0.31,
        fontsize=9.0,
    )

    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.09,
        top=0.975,
    )
    _save(figure, "q3_group1_storage_mechanism")
    return True


def _improvement_from_baseline(summary: pd.DataFrame, metric: str, schemes: tuple[str, ...]) -> dict[str, np.ndarray]:
    baseline_rows = summary.loc[summary["Solution"].astype(str) == "BaselineReference", metric]
    if baseline_rows.empty:
        return {scheme: np.asarray([np.nan]) for scheme in schemes}
    baseline = float(pd.to_numeric(baseline_rows.iloc[0], errors="coerce"))
    result: dict[str, np.ndarray] = {}
    for scheme in schemes:
        rows = summary.loc[summary["Solution"].astype(str) == scheme, metric]
        value = float(pd.to_numeric(rows.iloc[0], errors="coerce")) if not rows.empty else np.nan
        result[scheme] = np.asarray(
            [(baseline - value) / abs(baseline) * 100.0 if abs(baseline) > 1e-12 else np.nan]
        )
    return result


def _absolute_change_from_baseline(
    summary: pd.DataFrame,
    metric: str,
    schemes: tuple[str, ...],
) -> dict[str, float]:
    baseline_rows = summary.loc[summary["Solution"].astype(str) == "BaselineReference", metric]
    if baseline_rows.empty:
        return {scheme: np.nan for scheme in schemes}
    baseline = float(pd.to_numeric(baseline_rows.iloc[0], errors="coerce"))
    result: dict[str, float] = {}
    for scheme in schemes:
        rows = summary.loc[summary["Solution"].astype(str) == scheme, metric]
        value = float(pd.to_numeric(rows.iloc[0], errors="coerce")) if not rows.empty else np.nan
        result[scheme] = baseline - value
    return result


def _plot_cost_change(axis: plt.Axes, summary: pd.DataFrame) -> None:
    """Show absolute net-settlement changes; do not report a percentage when cost crosses zero."""

    schemes = ("NoStorage", "Balanced")
    changes = _absolute_change_from_baseline(summary, "Cost", schemes)
    values = np.asarray([changes[scheme] / 10000.0 for scheme in schemes], dtype=float)
    x = np.arange(len(schemes), dtype=float)
    bars = axis.bar(
        x,
        values,
        width=0.52,
        color=[SCHEME_COLORS[scheme] for scheme in schemes],
        edgecolor=["none", TEXT_COLOR],
        linewidth=0.35,
    )
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            offset = 0.018 * max(abs(value), 1.0)
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + offset if value >= 0 else value - offset,
                f"{value:+,.0f} x 10,000 CNY",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=6.2,
                color=TEXT_COLOR,
            )
    finite = values[np.isfinite(values)]
    if finite.size:
        low = min(0.0, float(finite.min()))
        high = max(0.0, float(finite.max()))
        span = max(high - low, 1.0)
        axis.set_ylim(low - 0.08 * span, high + 0.16 * span)
    axis.axhline(0.0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.9)
    axis.set_xticks(x)
    axis.set_xticklabels([SCHEME_LABELS[scheme] for scheme in schemes])
    axis.set_xlabel("Solution")
    axis.set_ylabel("Net-settlement change versus attachment baseline / 10,000 CNY", fontsize=9.1)
    _panel_caption(axis, "(a) Absolute net-settlement change (attachment baseline - solution)", y=-0.25, fontsize=9.4)
    _style_axis(axis)


def _plot_relative_improvement_metrics(axis: plt.Axes, summary: pd.DataFrame) -> None:
    metric_specs = [
        ("Carbon", "Carbon emissions"),
        ("Peak", "Peak net grid import"),
        ("Ramp", "Net-import fluctuation"),
    ]
    schemes = ("NoStorage", "Balanced")
    x = np.arange(len(metric_specs), dtype=float)
    width = 0.30
    all_values: list[float] = []
    for offset, scheme in enumerate(schemes):
        values = np.asarray(
            [
                _improvement_from_baseline(summary, metric, schemes)[scheme][0]
                for metric, _ in metric_specs
            ],
            dtype=float,
        )
        all_values.extend(values[np.isfinite(values)].tolist())
        axis.bar(
            x + (offset - 0.5) * width,
            values,
            width=width,
            color=SCHEME_COLORS[scheme],
            edgecolor=TEXT_COLOR if scheme == "Balanced" else "none",
            linewidth=0.35,
            label=SCHEME_LABELS[scheme],
        )
        for index, value in enumerate(values):
            if np.isfinite(value):
                axis.text(
                    x[index] + (offset - 0.5) * width,
                    value + 1.5 if value >= 0 else value - 1.5,
                    f"{value:.1f}%",
                    ha="right" if offset == 0 else "left",
                    va="bottom" if value >= 0 else "top",
                    fontsize=5.9,
                    color=TEXT_COLOR,
                )
    axis.axhline(0.0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.9, label="Attachment baseline=0%")
    if all_values:
        low = min(all_values)
        high = max(all_values)
        span = max(high - low, 10.0)
        axis.set_ylim(min(-5.0, low - 0.12 * span), high + 0.16 * span)
    axis.set_xticks(x)
    axis.set_xticklabels([label for _, label in metric_specs], fontsize=6.5)
    axis.set_xlabel("Metric")
    axis.set_ylabel("Improvement versus attachment baseline / %", fontsize=10.2)
    _panel_caption(axis, "(b) Three operating-metric improvements versus attachment baseline", y=-0.25, fontsize=9.4)
    _style_axis(axis)


def _regional_improvement(regional: pd.DataFrame, metric: str, scheme: str, regions: list[str]) -> np.ndarray:
    baseline = regional.loc[regional["Scheme"].astype(str) == "BaselineReference"].set_index("Region").reindex(regions)
    target = regional.loc[regional["Scheme"].astype(str) == scheme].set_index("Region").reindex(regions)
    baseline_values = pd.to_numeric(baseline[metric], errors="coerce").to_numpy(dtype=float)
    target_values = pd.to_numeric(target[metric], errors="coerce").to_numpy(dtype=float)
    return baseline_values - target_values


def _plot_regional_improvement(axis: plt.Axes, regional: pd.DataFrame, metric: str, ylabel: str, caption: str) -> None:
    regions = _ordered_regions(regional["Region"])
    schemes = ("NoStorage", "Balanced")
    x = np.arange(len(regions), dtype=float)
    width = 0.30
    for offset, scheme in enumerate(schemes):
        values = _regional_improvement(regional, metric, scheme, regions)
        axis.bar(
            x + (offset - 0.5) * width,
            values,
            width=width,
            color=SCHEME_COLORS[scheme],
            edgecolor=TEXT_COLOR if scheme == "Balanced" else "none",
            linewidth=0.35,
            label=SCHEME_LABELS[scheme],
        )
    axis.axhline(0.0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(regions)
    axis.set_xlabel("Region", fontsize=10.0)
    axis.set_ylabel(ylabel, fontsize=10.0)
    _panel_caption(axis, caption, y=-0.25, fontsize=9.6)
    _style_axis(axis)


def plot_group2_optimization_effect() -> bool:
    """Figure 2: absolute net-settlement changes, three relative improvements, and regional differences."""

    summary = _read_table("q3_objective_summary.csv", ("Solution", "Cost", "Carbon", "Peak", "Ramp"))
    regional = _read_table("q3_regional_metrics.csv", ("Scheme", "Region", "Peak", "Ramp"))
    if summary is None or regional is None:
        return False
    summary = _numeric(summary, ("Cost", "Carbon", "Peak", "Ramp"))
    regional = _numeric(regional, ("Peak", "Ramp"))
    if not set(SCHEME_ORDER).issubset(set(summary["Solution"].astype(str))):
        logging.error("q3_objective_summary.csv is missing the three solution types")
        return False

    figure = plt.figure(figsize=(6.85, 7.75))
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[1.03, 1.18],
        hspace=0.62,
        wspace=0.28,
    )
    cost_axis = figure.add_subplot(grid[0, 0])
    _plot_cost_change(cost_axis, summary)
    relative_axis = figure.add_subplot(grid[0, 1])
    _plot_relative_improvement_metrics(relative_axis, summary)
    peak_axis = figure.add_subplot(grid[1, 0])
    ramp_axis = figure.add_subplot(grid[1, 1])
    _plot_regional_improvement(
        peak_axis,
        regional,
        "Peak",
        "Peak net-import reduction / MW",
        "(c) Absolute peak net-import changes in six regions",
    )
    _plot_regional_improvement(
        ramp_axis,
        regional,
        "Ramp",
        "Net-import fluctuation reduction / MW",
        "(d) Absolute net-import fluctuation changes in six regions",
    )
    _legend(ramp_axis, loc="upper left", ncol=2, fontsize=6.0, handlelength=1.1, columnspacing=0.7)
    legend = figure.legend(
        handles=[
            plt.Line2D([], [], color=REFERENCE_COLOR, linestyle="--", linewidth=0.9, label="Attachment baseline=0%"),
            Patch(facecolor=SCHEME_COLORS["NoStorage"], label="No storage"),
            Patch(facecolor=SCHEME_COLORS["Balanced"], edgecolor=TEXT_COLOR, linewidth=0.35, label="Optimized solution"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        fontsize=6.1,
        handlelength=1.15,
        columnspacing=1.0,
    )
    for label in legend.get_texts():
        label.set_color(TEXT_COLOR)
    figure.subplots_adjust(left=0.086, right=0.978, bottom=0.095, top=0.89)
    _save(figure, "q3_group2_optimization_effect")
    return True


def _diagnostic_value(profile: pd.DataFrame, check: str) -> float:
    match = profile.loc[profile["Check"] == check, "MaxViolation"]
    return float(pd.to_numeric(match.iloc[0], errors="coerce")) if not match.empty else np.nan


def _plot_baseline_diagnostics(axis: plt.Axes, profile_checks: pd.DataFrame) -> None:
    checks = (
        ("RenewableBalanceResidual_MW", "Energy | renewable balance", "#5CB85C"),
        ("EnergyBalanceResidual_MW", "Energy | load balance", "#EEA236"),
        ("SOCRecurrenceResidual_MWh", "Storage | SOC recurrence", "#357EBD"),
        ("TerminalSOCDeficit_MWh", "Storage | terminal SOC gap", "#D43F3A"),
    )
    values = np.asarray([_diagnostic_value(profile_checks, check) for check, _, _ in checks], dtype=float)
    labels = [label for _, label, _ in checks]
    colors = [color for _, _, color in checks]
    finite = values[np.isfinite(values) & (values > 0)]
    if finite.size == 0:
        axis.text(0.5, 0.5, "Baseline diagnostics unavailable", ha="center", va="center", color=TEXT_COLOR)
        axis.set_axis_off()
        return
    y = np.asarray([0.0, 1.0, 3.0, 4.0])
    axis.barh(y, values, color=colors, height=0.55)
    axis.set_xscale("log")
    lower = min(float(finite.min()) / 3.0, 1e-8)
    upper = float(finite.max()) * 3.5
    axis.set_xlim(lower, upper)
    axis.axvline(1e-5, color=TEXT_COLOR, linestyle="--", linewidth=0.8, label="Validation tolerance")
    for position, value in zip(y, values):
        if np.isfinite(value) and value > 0:
            axis.text(value * 1.06, position, f"{value:.4g}", va="center", fontsize=6.1, color=TEXT_COLOR)
    axis.axhline(2.0, color=GRID_COLOR, linewidth=0.8)
    axis.set_yticks(y)
    axis.set_yticklabels(labels)
    axis.set_xlabel("Maximum deviation (log scale) / MW or MWh")
    # Place captions in the main layout so both groups align exactly.
    _legend(axis, loc="upper right", fontsize=5.6, handlelength=1.2)
    _style_axis(axis)


def _multiobjective_row(multiobjective: pd.DataFrame, check: str) -> tuple[float, float, float]:
    row = multiobjective.loc[multiobjective["Check"] == check]
    if row.empty:
        return np.nan, np.nan, np.nan
    value = float(pd.to_numeric(row.iloc[0]["Value"], errors="coerce"))
    reference = float(pd.to_numeric(row.iloc[0]["Reference"], errors="coerce"))
    difference = float(pd.to_numeric(row.iloc[0]["Difference"], errors="coerce"))
    tolerance = float(pd.to_numeric(row.iloc[0]["Tolerance"], errors="coerce"))
    if not np.isfinite(difference):
        difference = value - reference
    return abs(difference), tolerance, difference


def _plot_multiobjective_hold(
    error_axis: plt.Axes,
    throughput_axis: plt.Axes,
    multiobjective: pd.DataFrame,
) -> None:
    z_error, z_tolerance, _ = _multiobjective_row(multiobjective, "Stage3_preserves_minimax_z")
    phi_error, phi_tolerance, _ = _multiobjective_row(multiobjective, "Stage3_preserves_sum_deviation")
    _, _, throughput_difference = _multiobjective_row(
        multiobjective,
        "Stage3_does_not_increase_throughput",
    )
    errors = np.asarray([z_error, phi_error], dtype=float)
    tolerances = np.asarray([z_tolerance, phi_tolerance], dtype=float)
    x = np.arange(2, dtype=float)
    error_axis.bar(x, errors, width=0.48, color=["#357EBD", "#EEA236"])
    finite_errors = errors[np.isfinite(errors) & (errors > 0)]
    finite_tolerances = tolerances[np.isfinite(tolerances) & (tolerances > 0)]
    if finite_errors.size:
        lower = max(float(finite_errors.min()) / 3.0, 1e-12)
        upper = max(float(finite_errors.max()) * 5.0, 1e-5)
        error_axis.set_ylim(lower, upper)
    if finite_tolerances.size:
        error_axis.axhline(
            float(finite_tolerances.max()),
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=0.85,
            label="Solver tolerance",
        )
    for position, value in zip(x, errors):
        if np.isfinite(value):
            error_axis.text(position, value * 1.18, f"{value:.3g}", ha="center", va="bottom", fontsize=6.1, color=TEXT_COLOR)
    error_axis.set_yscale("log")
    error_axis.set_xticks(x)
    error_axis.set_xticklabels(["|Δz|", "|ΔΦ|"])
    error_axis.set_ylabel("Optimality preservation error / absolute value")
    error_axis.set_title("First two objectives", fontsize=7.2, pad=3.0, color=TEXT_COLOR)
    _legend(error_axis, loc="upper left", fontsize=5.5, handlelength=1.2)
    _style_axis(error_axis)

    throughput_axis.bar([0], [throughput_difference], width=0.48, color="#D43F3A")
    throughput_axis.axhline(0.0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.85, label="No increase")
    if np.isfinite(throughput_difference):
        offset = max(abs(throughput_difference) * 0.06, 1.0)
        throughput_axis.text(
            0,
            throughput_difference - offset if throughput_difference < 0 else throughput_difference + offset,
            f"{throughput_difference:+.2f}",
            ha="center",
            va="top" if throughput_difference < 0 else "bottom",
            fontsize=6.2,
            color=TEXT_COLOR,
        )
        throughput_axis.set_ylim(
            min(throughput_difference * 1.22, -1.0) if throughput_difference < 0 else -1.0,
            max(throughput_difference * 0.18, 1.0) if throughput_difference < 0 else throughput_difference * 1.22,
        )
    throughput_axis.set_xticks([0])
    throughput_axis.set_xticklabels(["ΔG = G(3)−G(2)"])
    throughput_axis.set_ylabel("Throughput change / MWh")
    throughput_axis.set_title("Final throughput stage", fontsize=7.2, pad=3.0, color=TEXT_COLOR)
    _legend(throughput_axis, loc="upper left", fontsize=5.5, handlelength=1.2)
    _style_axis(throughput_axis)
    # The right-hand panels share one group caption placed by the outer caption axis.


def plot_group3_validation_credibility() -> bool:
    """Figure 3: attachment-baseline diagnostics and three-stage selection preservation."""

    profile_checks = _read_table(
        "q3_validation_profile_checks.csv",
        ("Scheme", "Check", "MaxViolation", "Status"),
    )
    multiobjective = _read_table(
        "q3_validation_multiobjective.csv",
        ("Check", "Value", "Reference", "Difference", "Status"),
    )
    if profile_checks is None or multiobjective is None:
        return False

    profile_checks = _numeric(
        profile_checks,
        ("MaxViolation", "Tolerance"),
    )
    multiobjective = _numeric(
        multiobjective,
        ("Value", "Reference", "Difference", "Tolerance"),
    )
    baseline = profile_checks.loc[
        profile_checks["Scheme"] == "BaselineReference"
    ].copy()

    figure = plt.figure(figsize=(7.35, 4.82))

    # Reserve one outer caption row to align panels (a) and (b) exactly.
    outer = figure.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 0.12],
        width_ratios=[0.88, 1.52],
        hspace=0.02,
        wspace=0.24,
    )

    # Add a small lower margin inside the left panel to shorten its axes,
    # preventing x-labels, legends, and captions from crowding.
    left_grid = outer[0, 0].subgridspec(
        2,
        1,
        height_ratios=[0.88, 0.12],
        hspace=0.0,
    )
    diagnostics_axis = figure.add_subplot(left_grid[0, 0])
    left_spacer = figure.add_subplot(left_grid[1, 0])
    left_spacer.axis("off")

    # Use full height and slightly more width for the right validation panel.
    right_grid = outer[0, 1].subgridspec(
        1,
        2,
        width_ratios=[1.12, 1.00],
        wspace=0.34,
    )
    error_axis = figure.add_subplot(right_grid[0, 0])
    throughput_axis = figure.add_subplot(right_grid[0, 1])

    left_caption_axis = figure.add_subplot(outer[1, 0])
    right_caption_axis = figure.add_subplot(outer[1, 1])
    left_caption_axis.axis("off")
    right_caption_axis.axis("off")

    caption_y = 0.20
    left_caption_axis.text(
        0.5,
        caption_y,
        "(a) Attachment-baseline convention diagnostics",
        ha="center",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color=TEXT_COLOR,
        transform=left_caption_axis.transAxes,
    )
    right_caption_axis.text(
        0.5,
        caption_y,
        "(b) Three-stage selection preservation",
        ha="center",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color=TEXT_COLOR,
        transform=right_caption_axis.transAxes,
    )

    _plot_baseline_diagnostics(diagnostics_axis, baseline)
    _plot_multiobjective_hold(
        error_axis,
        throughput_axis,
        multiobjective,
    )

    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.135,
        top=0.95,
    )
    _save(figure, "q3_group3_validation_credibility")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot static Question 3 results")
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=("group1", "group2", "group3"),
        default=("group1", "group2", "group3"),
        help="Plot only the specified figure group; default plots all three",
    )
    args = parser.parse_args()
    _configure_style()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    if not TABLES_DIR.is_dir():
        logging.info("Q3 result directory is missing; skipping plotting: %s", TABLES_DIR)
        return 0
    figure_functions = {
        "group1": plot_group1_storage_mechanism,
        "group2": plot_group2_optimization_effect,
        "group3": plot_group3_validation_credibility,
    }
    selected_functions = [figure_functions[group] for group in args.groups]
    generated = 0
    for function in selected_functions:
        logging.info("Plotting started: %s", function.__name__)
        if function():
            generated += 1
    logging.info(
        "Q3 selected groups completed: %d/%d groups generated in %s",
        generated,
        len(selected_functions),
        FIGURES_DIR,
    )
    return 0 if generated == len(selected_functions) else 1


if __name__ == "__main__":
    raise SystemExit(main())
