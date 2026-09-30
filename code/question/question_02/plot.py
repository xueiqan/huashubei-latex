"""Three groups of Question 2 paper figures.

Read generated Q2 tables without solving models, invoking MILP, or modifying
official results. Produce:

1. q2_group1_core_results.pdf/png: normalized objectives and renewable usage.
2. q2_group2_dispatch_mechanism.pdf/png: migrated GPU workload matrix.
3. q2_group3_model_validation.pdf/png: exact same-window comparison and
   K=24/48/72 sensitivity.

Colors come from the supplied plotting palette:
- Two series: #5773cc, #ffb900.
- Three series: #eeca40, #fd763f, #23bac5.
- Four series: #dd5129, #0f7ba2, #43b284, #fab255.
- Heatmap: the blue-gradient series.
- Text, grids, and references: the dark-blue scheme used in Q1/Q3.

Missing exact comparisons or independent K results are explicitly marked as
pending evidence in the affected panels. Never substitute old or incompatible
results.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


QUESTION_DIR = Path(__file__).resolve().parent
TABLES_DIR = QUESTION_DIR / "outputs" / "tables"
VALIDATION_RUNS_DIR = QUESTION_DIR / "outputs" / "validation_runs"
FIGURES_DIR = QUESTION_DIR / "outputs" / "figures"

REGION_ORDER = [
    "RegionA",
    "RegionB",
    "RegionC",
    "RegionD",
    "RegionE",
    "RegionF",
]
REGION_LABELS = {region: region.replace("Region", "") for region in REGION_ORDER}

OBJECTIVE_SPECS = (
    ("Cost", "Operating cost", "OperatingCost_CNY"),
    ("Carbon", "Carbon emissions", "CarbonEmission_tCO2"),
    ("MeanLatency", "Mean network latency", "MeanNetworkLatency_ms"),
    ("RenewableUnusedRate", "Renewable unused rate", "RenewableUnusedRate"),
)

# All colors come from the user-supplied plotting palette.
PALETTE = {
    "series_blue": "#5773cc",
    "series_gold": "#ffb900",
    "series_cyan": "#23bac5",
    "series_orange": "#fd763f",
    "series_red": "#dd5129",
    "series_green": "#43b284",
    "series_deep_blue": "#0f7ba2",
    "text": "#003967",
    "grid": "#acd6ec",
    "reference": "#d77071",
    "diagonal": "#d3d5d4",
    "light_blue": "#badefa",
    "white": "#ffffff",
}

DOCX_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "q2_docx_blue",
    [
        "#e3f2fd",
        "#badefa",
        "#90caf8",
        "#64b4f6",
        "#41a5f4",
        "#2096f2",
        "#1e87e5",
        "#1976d2",
        "#1465bf",
        "#0c46a0",
    ],
)


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "text.color": PALETTE["text"],
            "axes.edgecolor": PALETTE["text"],
            "axes.labelcolor": PALETTE["text"],
            "axes.titlecolor": PALETTE["text"],
            "axes.labelsize": 8.2,
            "axes.titlesize": 9.2,
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.labelcolor": PALETTE["text"],
            "legend.fontsize": 6.7,
            "figure.facecolor": PALETTE["white"],
            "axes.facecolor": PALETTE["white"],
            "savefig.facecolor": PALETTE["white"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _read_csv(path: Path, required: tuple[str, ...] = ()) -> pd.DataFrame | None:
    if not path.is_file():
        logging.warning("Plot input missing: %s", path)
        return None
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        logging.warning("Cannot read plot input: %s; %s", path, exc)
        return None
    missing = [column for column in required if column not in frame.columns]
    if missing:
        logging.warning("%s is missing columns: %s", path.name, missing)
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


def _first_existing(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in columns:
            return column
    return None


def _style_axis(axis: plt.Axes, grid_axis: str = "y") -> None:
    if grid_axis == "none":
        axis.grid(False)
    else:
        axis.grid(
            True,
            axis=grid_axis,
            color=PALETTE["grid"],
            linewidth=0.65,
            alpha=0.72,
        )
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", colors=PALETTE["text"], labelcolor=PALETTE["text"])
    for spine in axis.spines.values():
        spine.set_color(PALETTE["text"])
        spine.set_linewidth(0.72)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _legend(axis: plt.Axes, **kwargs) -> None:
    legend = axis.legend(frameon=False, **kwargs)
    if legend is not None:
        for label in legend.get_texts():
            label.set_color(PALETTE["text"])


def _panel_caption(
    axis: plt.Axes,
    text: str,
    *,
    y: float = -0.22,
    fontsize: float = 9.0,
) -> None:
    axis.text(
        0.5,
        y,
        text,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        color=PALETTE["text"],
        clip_on=False,
    )


def _style_colorbar(colorbar: matplotlib.colorbar.Colorbar, label: str) -> None:
    colorbar.set_label(label, color=PALETTE["text"])
    colorbar.ax.tick_params(colors=PALETTE["text"], labelsize=6.8)
    colorbar.outline.set_edgecolor(PALETTE["text"])
    colorbar.outline.set_linewidth(0.7)


def _save(figure: plt.Figure, stem: str) -> None:
    """Export paper PDFs and 450 dpi PNG previews together."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / f"{stem}.pdf"
    png_path = FIGURES_DIR / f"{stem}.png"
    try:
        try:
            figure.savefig(
                pdf_path,
                format="pdf",
                bbox_inches="tight",
                pad_inches=0.06,
            )
        except PermissionError:
            fallback = FIGURES_DIR / f"{stem}_new.pdf"
            figure.savefig(
                fallback,
                format="pdf",
                bbox_inches="tight",
                pad_inches=0.06,
            )
            logging.warning("%s is locked; writing the PDF to %s instead.", pdf_path.name, fallback.name)
        figure.savefig(
            png_path,
            format="png",
            dpi=450,
            bbox_inches="tight",
            pad_inches=0.06,
        )
        logging.info("Generated: %s and %s.", pdf_path.name, png_path.name)
    finally:
        plt.close(figure)


def _solution_row(summary: pd.DataFrame, solution: str) -> pd.Series | None:
    if "Solution" not in summary.columns:
        return None
    rows = summary.loc[summary["Solution"].astype(str).eq(solution)]
    return rows.iloc[0] if not rows.empty else None


def _normalised_objectives() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    summary = _read_table(
        "q2_objective_summary.csv",
        ("Solution", "OperatingCost_CNY", "CarbonEmission_tCO2", "MeanNetworkLatency_ms", "RenewableUnusedRate"),
    )
    calibration = _read_table(
        "q2_global_calibration.csv",
        ("Objective", "IdealLowerBound", "NormalizationScale"),
    )
    if summary is None or calibration is None:
        return None
    summary = _numeric(
        summary,
        (
            "OperatingCost_CNY",
            "CarbonEmission_tCO2",
            "MeanNetworkLatency_ms",
            "RenewableUnusedRate",
        ),
    )
    calibration = _numeric(
        calibration,
        ("IdealLowerBound", "NormalizationScale"),
    ).set_index("Objective")
    baseline = _solution_row(summary, "Q1PureComputeBaseline")
    balanced = _solution_row(summary, "Q2Balanced")
    if baseline is None or balanced is None:
        logging.warning("q2_objective_summary.csv is missing the baseline or Q2Balanced row.")
        return None

    rows: list[dict[str, float | str]] = []
    for objective, label, value_column in OBJECTIVE_SPECS:
        if objective not in calibration.index:
            logging.warning("Global calibration table is missing objective: %s.", objective)
            return None
        ideal = float(calibration.loc[objective, "IdealLowerBound"])
        scale = float(calibration.loc[objective, "NormalizationScale"])
        if not math.isfinite(scale) or abs(scale) <= 1e-12:
            logging.warning("Objective %s has invalid NormalizationScale.", objective)
            return None
        base_value = float(baseline[value_column])
        balanced_value = float(balanced[value_column])
        rows.append(
            {
                "Objective": objective,
                "Label": label,
                "BaselineDeviation": (base_value - ideal) / scale,
                "BalancedDeviation": (balanced_value - ideal) / scale,
            }
        )
    return pd.DataFrame(rows), summary


def plot_group1_core_results() -> bool:
    """Figure 1: four normalized objective deviations and renewable-usage changes."""
    normalised = _normalised_objectives()
    balanced_energy = _read_table(
        "q2_energy_profile.csv",
        ("RenewableDirectUse_MW", "RenewableExport_MW", "RenewableCurtailment_MW"),
    )
    baseline_energy = _read_table(
        "q2_baseline_energy_profile.csv",
        ("RenewableDirectUse_MW", "RenewableExport_MW", "RenewableCurtailment_MW"),
    )
    if normalised is None or balanced_energy is None or baseline_energy is None:
        return False
    deviations, summary = normalised
    balanced_energy = _numeric(
        balanced_energy,
        ("RenewableDirectUse_MW", "RenewableExport_MW", "RenewableCurtailment_MW"),
    )
    baseline_energy = _numeric(
        baseline_energy,
        ("RenewableDirectUse_MW", "RenewableExport_MW", "RenewableCurtailment_MW"),
    )
    energy_columns = (
        "RenewableDirectUse_MW",
        "RenewableExport_MW",
        "RenewableCurtailment_MW",
    )
    delta = (
        balanced_energy[list(energy_columns)].sum()
        - baseline_energy[list(energy_columns)].sum()
    )

    figure = plt.figure(figsize=(7.05, 4.65))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.08, 0.92], wspace=0.42)
    ax_left = figure.add_subplot(grid[0, 0])
    ax_right = figure.add_subplot(grid[0, 1])
    figure.subplots_adjust(left=0.10, right=0.97, bottom=0.17, top=0.94)

    x = np.arange(len(deviations))
    width = 0.34
    base = deviations["BaselineDeviation"].to_numpy(dtype=float)
    balanced = deviations["BalancedDeviation"].to_numpy(dtype=float)
    bars_base = ax_left.bar(
        x - width / 2,
        base,
        width,
        label="Compute-only baseline",
        color=PALETTE["series_blue"],
        edgecolor=PALETTE["text"],
        linewidth=0.45,
    )
    bars_balanced = ax_left.bar(
        x + width / 2,
        balanced,
        width,
        label="Q2 multiobjective schedule",
        color=PALETTE["series_gold"],
        edgecolor=PALETTE["text"],
        linewidth=0.45,
    )
    q2_z = float(np.nanmax(balanced))
    ax_left.axhline(
        q2_z,
        color=PALETTE["reference"],
        linestyle="--",
        linewidth=1.0,
        label=f"Q2 maximum deviation z={q2_z:.2f}",
    )
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(["Cost", "Carbon emissions", "Mean\nnetwork latency", "Renewable\nunused rate"])
    ax_left.set_ylabel("Normalized objective deviation $d_m$")
    ax_left.set_title("Multiobjective comparison after global calibration")
    max_value = float(np.nanmax(np.r_[base, balanced, q2_z]))
    ax_left.set_ylim(0, max(1.0, max_value * 1.25))
    ax_left.axhline(0, color=PALETTE["text"], linewidth=0.7)
    _style_axis(ax_left, "y")
    _legend(ax_left, loc="upper left", ncol=1)
    for bars in (bars_base, bars_balanced):
        for bar in bars:
            value = float(bar.get_height())
            ax_left.text(
                bar.get_x() + bar.get_width() / 2,
                value + max_value * 0.025,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.4,
                color=PALETTE["text"],
            )
    _panel_caption(ax_left, "(a) Four normalized objective deviations", y=-0.22)

    energy_labels = ["Direct consumption", "Renewable export", "Renewable curtailment"]
    energy_values = delta.to_numpy(dtype=float)
    energy_colors = [
        PALETTE["series_cyan"],
        PALETTE["series_gold"],
        PALETTE["series_orange"],
    ]
    bars = ax_right.bar(
        np.arange(3),
        energy_values,
        color=energy_colors,
        width=0.58,
        edgecolor=PALETTE["text"],
        linewidth=0.45,
    )
    ax_right.axhline(0, color=PALETTE["text"], linewidth=0.8)
    ax_right.set_xticks(np.arange(3))
    ax_right.set_xticklabels(energy_labels)
    ax_right.set_ylabel("Q2 - baseline (MWh)")
    ax_right.set_title("Full-horizon total (0--2405 h)")
    energy_min = float(np.nanmin(energy_values))
    energy_max = float(np.nanmax(energy_values))
    energy_span = max(energy_max - energy_min, 1.0)
    ax_right.set_ylim(
        energy_min - 0.10 * energy_span,
        energy_max + 0.16 * energy_span,
    )
    _style_axis(ax_right, "y")
    for bar, value in zip(bars, energy_values):
        offset = max(float(np.nanmax(np.abs(energy_values))), 1.0) * 0.035
        y = value + offset if value >= 0 else value - offset
        ax_right.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{value:+,.0f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=6.5,
            color=PALETTE["text"],
        )
    base_row = _solution_row(summary, "Q1PureComputeBaseline")
    balanced_row = _solution_row(summary, "Q2Balanced")
    if base_row is not None and balanced_row is not None:
        cost_improvement_wan = (
            float(base_row["OperatingCost_CNY"])
            - float(balanced_row["OperatingCost_CNY"])
        ) / 1e4
        curtail_reduction = -float(delta["RenewableCurtailment_MW"])
        ax_right.text(
            0.02,
            0.08,
            f"Net operating cost reduction: {cost_improvement_wan:.2f} x 10,000 CNY\n"
            f"Renewable curtailment reduction: {curtail_reduction:,.0f} MWh",
            transform=ax_right.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            color=PALETTE["text"],
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": PALETTE["white"],
                "edgecolor": PALETTE["grid"],
                "linewidth": 0.6,
            },
        )
    _panel_caption(ax_right, "(b) Renewable-usage changes", y=-0.22)
    _save(figure, "q2_group1_core_results")
    return True


def _ordered_regions(frame: pd.DataFrame) -> list[str]:
    observed = set(frame["SourceRegion"].astype(str)) | set(frame["TargetRegion"].astype(str))
    ordered = [region for region in REGION_ORDER if region in observed]
    ordered.extend(sorted(observed.difference(ordered)))
    return ordered


def _compact_millions(value: float, _position: int) -> str:
    if abs(value) >= 1e6:
        return f"{value / 1e6:.1f}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.0f}k"
    return f"{value:.0f}"


def plot_group2_dispatch_mechanism() -> bool:
    """Figure 2: migrated GPU workload matrix with inbound/outbound marginal totals."""
    assignments = _read_table(
        "q2_balanced_assignments.csv",
        (
            "SourceRegion",
            "TargetRegion",
            "GPU_Demand",
            "Duration_h",
            "Migration_GPU_Workload_GPUh",
        ),
    )
    if assignments is None:
        return False
    assignments = _numeric(
        assignments,
        ("GPU_Demand", "Duration_h", "Migration_GPU_Workload_GPUh"),
    )
    assignments["SourceRegion"] = assignments["SourceRegion"].astype(str)
    assignments["TargetRegion"] = assignments["TargetRegion"].astype(str)
    assignments["_work_gpu_h"] = assignments["Migration_GPU_Workload_GPUh"]
    missing_work = assignments["_work_gpu_h"].isna()
    assignments.loc[missing_work, "_work_gpu_h"] = (
        assignments.loc[missing_work, "GPU_Demand"]
        * assignments.loc[missing_work, "Duration_h"]
    )
    assignments["_is_cross_region"] = (
        assignments["SourceRegion"] != assignments["TargetRegion"]
    )
    regions = _ordered_regions(assignments)
    if len(regions) < 2:
        logging.warning("Insufficient regions in the migration matrix.")
        return False

    work = (
        assignments.groupby(["SourceRegion", "TargetRegion"])["_work_gpu_h"]
        .sum()
        .unstack("TargetRegion")
        .reindex(index=regions, columns=regions, fill_value=0.0)
        .fillna(0.0)
    )
    counts = (
        assignments.groupby(["SourceRegion", "TargetRegion"])
        .size()
        .unstack("TargetRegion")
        .reindex(index=regions, columns=regions, fill_value=0)
        .fillna(0)
    )
    work_values = work.to_numpy(dtype=float)
    off_diagonal = ~np.eye(len(regions), dtype=bool)
    off_values = work_values[off_diagonal]
    max_work = float(np.nanmax(off_values)) if off_values.size else 0.0
    image_data = work_values.copy()
    image_data[~off_diagonal] = np.nan
    cmap = DOCX_BLUE_CMAP.copy()
    cmap.set_bad(PALETTE["diagonal"])
    norm = Normalize(vmin=0.0, vmax=max(max_work, 1.0))

    cross = assignments.loc[assignments["_is_cross_region"]].copy()
    incoming = cross.groupby("TargetRegion")["_work_gpu_h"].sum().reindex(regions, fill_value=0.0)
    outgoing = cross.groupby("SourceRegion")["_work_gpu_h"].sum().reindex(regions, fill_value=0.0)

    figure = plt.figure(figsize=(7.8, 6.35))
    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=[5.8, 0.32, 1.55],
        height_ratios=[1.10, 5.8],
        hspace=0.12,
        wspace=0.20,
    )
    ax_top = figure.add_subplot(grid[0, 0])
    ax_heat = figure.add_subplot(grid[1, 0])
    cax = figure.add_subplot(grid[1, 1])
    ax_right = figure.add_subplot(grid[1, 2], sharey=ax_heat)
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.97)

    ax_top.bar(
        np.arange(len(regions)),
        incoming.to_numpy(dtype=float),
        color=PALETTE["series_blue"],
        edgecolor=PALETTE["text"],
        linewidth=0.4,
    )
    ax_top.set_xticks(np.arange(len(regions)))
    ax_top.set_xticklabels([REGION_LABELS.get(r, r) for r in regions])
    ax_top.set_ylabel("GPU·h", labelpad=2)
    ax_top.set_title("Inbound migration", fontsize=8.5, pad=3)
    ax_top.yaxis.set_major_formatter(FuncFormatter(_compact_millions))
    _style_axis(ax_top, "y")
    ax_top.spines["bottom"].set_visible(False)
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)

    image = ax_heat.imshow(
        image_data,
        cmap=cmap,
        norm=norm,
        aspect="equal",
        interpolation="nearest",
    )
    ax_heat.set_xticks(np.arange(len(regions)))
    ax_heat.set_yticks(np.arange(len(regions)))
    ax_heat.set_xticklabels([REGION_LABELS.get(r, r) for r in regions])
    ax_heat.set_yticklabels([REGION_LABELS.get(r, r) for r in regions])
    ax_heat.set_xlabel("Target region")
    ax_heat.set_ylabel("Source region")
    ax_heat.set_xticks(np.arange(-0.5, len(regions), 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, len(regions), 1), minor=True)
    ax_heat.grid(which="minor", color=PALETTE["white"], linewidth=0.85)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    _style_axis(ax_heat, "none")
    for row_index, source in enumerate(regions):
        for column_index, target in enumerate(regions):
            value = float(work.loc[source, target])
            count = int(counts.loc[source, target])
            if row_index == column_index:
                label = f"Local\nN={count:,}"
                color = PALETTE["text"]
            elif value <= 0:
                label = "—"
                color = PALETTE["text"]
            else:
                label = f"{value:,.0f}\nN={count:,}"
                color = PALETTE["white"] if value >= max_work * 0.52 else PALETTE["text"]
            ax_heat.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=6.5,
                color=color,
                linespacing=1.08,
            )
    colorbar = figure.colorbar(image, cax=cax)
    _style_colorbar(colorbar, "GPU·h")

    ax_right.barh(
        np.arange(len(regions)),
        outgoing.to_numpy(dtype=float),
        color=PALETTE["series_gold"],
        edgecolor=PALETTE["text"],
        linewidth=0.4,
    )
    ax_right.set_yticks(np.arange(len(regions)))
    ax_right.set_yticklabels([])
    ax_right.set_xlabel("GPU·h", labelpad=2)
    ax_right.set_title("Outbound migration", fontsize=8.5, pad=3)
    ax_right.xaxis.set_major_formatter(FuncFormatter(_compact_millions))
    _style_axis(ax_right, "x")
    ax_right.spines["left"].set_visible(False)
    ax_right.tick_params(axis="y", left=False, labelleft=False)
    _panel_caption(
        ax_heat,
        "Q2 interregional migration and compute-workload redistribution\n"
        "Gray diagonal shows local task counts only; color scale shows migrated GPU-hours only",
        y=-0.20,
        fontsize=8.4,
    )
    _save(figure, "q2_group2_dispatch_mechanism")
    return True


def _pair_paths() -> tuple[Path, Path] | None:
    root = VALIDATION_RUNS_DIR / "exact_vs_heuristic"
    exact = root / "exact_windows.csv"
    heuristic = root / "heuristic_windows.csv"
    if exact.is_file() and heuristic.is_file():
        return exact, heuristic
    return None


def _z_column(frame: pd.DataFrame) -> str | None:
    return _first_existing(
        frame.columns,
        ("ZStar", "FinalZ", "EstimatedMaxNormalizedDeviation", "z"),
    )


def _load_pair() -> pd.DataFrame | None:
    paths = _pair_paths()
    if paths is None:
        return None
    exact = _read_csv(paths[0], ("WindowID",))
    heuristic = _read_csv(paths[1], ("WindowID",))
    if exact is None or heuristic is None:
        return None
    exact_z = _z_column(exact)
    heuristic_z = _z_column(heuristic)
    if exact_z is None or heuristic_z is None:
        logging.warning("Exact same-window files are missing Z metric columns.")
        return None
    exact = _numeric(exact, ("WindowID", exact_z, "ElapsedSeconds"))
    heuristic = _numeric(heuristic, ("WindowID", heuristic_z, "ElapsedSeconds"))
    exact = exact.loc[exact["WindowID"].between(0, 9)].copy()
    heuristic = heuristic.loc[heuristic["WindowID"].between(0, 9)].copy()
    exact = exact[["WindowID", exact_z] + (["ElapsedSeconds"] if "ElapsedSeconds" in exact else [])]
    heuristic = heuristic[["WindowID", heuristic_z] + (["ElapsedSeconds"] if "ElapsedSeconds" in heuristic else [])]
    exact = exact.rename(columns={exact_z: "ZExact", "ElapsedSeconds": "TimeExact"})
    heuristic = heuristic.rename(columns={heuristic_z: "ZHeuristic", "ElapsedSeconds": "TimeHeuristic"})
    merged = exact.merge(heuristic, on="WindowID", how="inner")
    if merged.empty:
        return None
    merged["RelativeDelta"] = (
        merged["ZHeuristic"] - merged["ZExact"]
    ) / merged["ZExact"].abs().clip(lower=1e-8)
    if "TimeExact" in merged.columns and "TimeHeuristic" in merged.columns:
        merged["Speedup"] = merged["TimeExact"] / merged["TimeHeuristic"].clip(lower=1e-8)
    return merged.sort_values("WindowID", kind="stable")


def _summary_path(root: Path) -> Path | None:
    candidates = (root / "q2_objective_summary.csv", root / "tables" / "q2_objective_summary.csv")
    return next((path for path in candidates if path.is_file()), None)


def _load_summary_root(root: Path) -> pd.DataFrame | None:
    path = _summary_path(root)
    if path is None:
        return None
    return _read_csv(path, ("Solution",))


def _lookahead_normalised() -> pd.DataFrame | None:
    calibration = _read_table(
        "q2_global_calibration.csv",
        ("Objective", "IdealLowerBound", "NormalizationScale"),
    )
    if calibration is None:
        return None
    calibration = _numeric(calibration, ("IdealLowerBound", "NormalizationScale")).set_index("Objective")
    roots = {
        24: VALIDATION_RUNS_DIR / "K24",
        48: TABLES_DIR,
        72: VALIDATION_RUNS_DIR / "K72",
    }
    value_columns = {objective: column for objective, _label, column in OBJECTIVE_SPECS}
    labels = {objective: label for objective, label, _column in OBJECTIVE_SPECS}
    rows: list[dict[str, float | int | str]] = []
    for lookahead, root in roots.items():
        summary = _load_summary_root(root)
        if summary is None:
            logging.info("Independent K=%d results are missing; omitting this sensitivity point.", lookahead)
            continue
        summary = _numeric(summary, tuple(value_columns.values()))
        balanced = _solution_row(summary, "Q2Balanced")
        if balanced is None:
            logging.warning("K=%d results are missing the Q2Balanced row.", lookahead)
            continue
        row: dict[str, float | int | str] = {"LookaheadHours": lookahead}
        for objective, value_column in value_columns.items():
            if objective not in calibration.index:
                logging.warning("Sensitivity calibration table is missing objective: %s.", objective)
                return None
            scale = float(calibration.loc[objective, "NormalizationScale"])
            ideal = float(calibration.loc[objective, "IdealLowerBound"])
            row[labels[objective]] = (
                float(balanced[value_column]) - ideal
            ) / scale
        rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("LookaheadHours", kind="stable")


def _validation_placeholder(axis: plt.Axes, message: str, detail: str) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.text(
        0.5,
        0.57,
        message,
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color=PALETTE["text"],
    )
    axis.text(
        0.5,
        0.40,
        detail,
        ha="center",
        va="center",
        fontsize=6.7,
        color=PALETTE["reference"],
        wrap=True,
    )


def plot_group3_model_validation() -> bool:
    """Figure 3: exact same-window quality comparison and rolling-lookahead sensitivity."""
    pair = _load_pair()
    lookahead = _lookahead_normalised()
    figure = plt.figure(figsize=(7.15, 4.75))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.42)
    ax_pair = figure.add_subplot(grid[0, 0])
    ax_k = figure.add_subplot(grid[0, 1])
    figure.subplots_adjust(left=0.10, right=0.97, bottom=0.17, top=0.94)

    if pair is None:
        _validation_placeholder(
            ax_pair,
            "Exact same-window evidence pending",
            "Requires exact_windows.csv and heuristic_windows.csv\nfor windows 0--9 under identical history",
        )
    else:
        ax_pair.scatter(
            pair["ZExact"],
            pair["ZHeuristic"],
            s=28,
            color=PALETTE["series_blue"],
            edgecolor=PALETTE["text"],
            linewidth=0.45,
            zorder=3,
        )
        pair = pair.copy()
        pair["_AbsRelativeDelta"] = pair["RelativeDelta"].abs()
        label_rows = pair.nlargest(
            min(3, len(pair)),
            "_AbsRelativeDelta",
        )
        for row in label_rows.itertuples(index=False):
            ax_pair.annotate(
                f"Window {int(row.WindowID)}",
                (float(row.ZExact), float(row.ZHeuristic)),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=6.0,
                color=PALETTE["text"],
                fontweight="bold",
            )
        minimum = float(np.nanmin(np.r_[pair["ZExact"], pair["ZHeuristic"]]))
        maximum = float(np.nanmax(np.r_[pair["ZExact"], pair["ZHeuristic"]]))
        padding = max((maximum - minimum) * 0.10, 0.05)
        lower = max(0.0, minimum - padding)
        upper = maximum + padding
        ax_pair.plot(
            [lower, upper],
            [lower, upper],
            linestyle="--",
            color=PALETTE["reference"],
            linewidth=0.9,
            label="y=x",
        )
        ax_pair.set_xlim(lower, upper)
        ax_pair.set_ylim(lower, upper)
        mean_relative = float(pair["RelativeDelta"].abs().mean())
        max_relative = float(pair["RelativeDelta"].abs().max())
        text = (
            f"Mean |relative deviation|: {mean_relative:.2%}\n"
            f"Maximum |relative deviation|: {max_relative:.2%}"
        )
        if "Speedup" in pair.columns:
            text += f"\nMedian speedup: {float(pair['Speedup'].median()):.1f}×"
        ax_pair.text(
            0.96,
            0.96,
            text,
            transform=ax_pair.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            color=PALETTE["text"],
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": PALETTE["white"],
                "edgecolor": PALETTE["grid"],
                "linewidth": 0.6,
            },
        )
        ax_pair.set_xlabel(r"Exact MILP $z_\tau$")
        ax_pair.set_ylabel(r"Heuristic + LNS $z_\tau$")
        ax_pair.set_title("Same-window comparison, windows 0--9")
        _style_axis(ax_pair, "both")
        _legend(ax_pair, loc="lower right")
    _panel_caption(ax_pair, "(a) Exact MILP versus matheuristic", y=-0.22, fontsize=8.8)

    if lookahead is None:
        _validation_placeholder(
            ax_k,
            "Lookahead sensitivity evidence pending",
            "Requires Q2Balanced results for K=24, K=48, and K=72",
        )
    else:
        colors = [
            PALETTE["series_red"],
            PALETTE["series_deep_blue"],
            PALETTE["series_green"],
            PALETTE["series_gold"],
        ]
        for index, (_objective, label, _value_column) in enumerate(OBJECTIVE_SPECS):
            if label not in lookahead.columns:
                continue
            data = lookahead.dropna(subset=[label])
            ax_k.plot(
                data["LookaheadHours"],
                data[label],
                marker="o",
                markersize=4.5,
                linewidth=1.15,
                color=colors[index],
                label=label,
            )
            if 48 in set(data["LookaheadHours"].astype(int)):
                central = data.loc[data["LookaheadHours"].astype(int).eq(48)].iloc[0]
                ax_k.scatter(
                    [48],
                    [float(central[label])],
                    s=48,
                    facecolor=PALETTE["white"],
                    edgecolor=colors[index],
                    linewidth=1.1,
                    zorder=4,
                )
        ax_k.axhline(0, color=PALETTE["reference"], linestyle="--", linewidth=0.85)
        ax_k.axvspan(47, 49, color=PALETTE["light_blue"], alpha=0.28, linewidth=0)
        ax_k.set_xticks([24, 48, 72])
        ax_k.set_xlim(20, 76)
        ax_k.set_xlabel(r"Lookahead length $K$ / h")
        ax_k.set_ylabel(r"Q2 normalized objective deviation $d_m$")
        ax_k.set_title("Objective structure at different lookahead lengths", pad=10)
        _style_axis(ax_k, "both")
        _legend(
            ax_k,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=2,
            columnspacing=1.2,
            handletextpad=0.5,
        )
        observed = set(lookahead["LookaheadHours"].astype(int))
        missing = [str(value) for value in (24, 48, 72) if value not in observed]
        if missing:
            ax_k.text(
                0.97,
                0.96,
                "Pending K=" + ",".join(missing),
                transform=ax_k.transAxes,
                ha="right",
                va="top",
                fontsize=6.5,
                color=PALETTE["reference"],
            )
    _panel_caption(ax_k, "(b) Rolling-lookahead sensitivity", y=-0.22, fontsize=8.8)
    _save(figure, "q2_group3_model_validation")
    return True


def main() -> int:
    _configure_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    results = {
        "group1": plot_group1_core_results(),
        "group2": plot_group2_dispatch_mechanism(),
        "group3": plot_group3_model_validation(),
    }
    logging.info("Q2 plotting completion status: %s.", results)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
