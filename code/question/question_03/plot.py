"""问题3三组论文图。

本文件只读取问题3已经生成的输入/结果CSV，不重新求解MILP，也不修改模型结果。
输出三份静态PDF：

1. 储能时移运行机制；
2. 储能协同优化效果与区域差异；
3. 物理闭环、附件基准诊断和多目标择优检验。

图内不放总图题和长段解释；每个子图的题目统一放在子图下方。
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
    "NoStorage": "无储能",
    "BaselineReference": "附件基准",
    "Balanced": "优化方案",
}
# 配色沿用问题1 plot.py中依据《绘图颜色搭配.docx》整理的色值；本次按Q3图示语义映射：
# 绿色=新能源，蓝色=储能/购电，橙红色=放电/峰值/告警，紫色=附件基准。
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
        logging.error("图表输入缺失：%s", path)
        return None
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        logging.error("%s缺少字段：%s", path.name, missing)
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
    """将子图题放在坐标轴下方，避免与坐标轴标题混在上方。"""

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
    """保存PDF；若原PDF正被Windows预览器占用，则自动写入 *_new.pdf。"""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    target = FIGURES_DIR / f"{stem}.pdf"

    try:
        figure.savefig(
            target,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.03,
        )
        logging.info("已生成：%s", target.name)

    except PermissionError:
        fallback = FIGURES_DIR / f"{stem}_new.pdf"
        figure.savefig(
            fallback,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.03,
        )
        logging.warning(
            "%s 正被其他程序占用，已改为输出：%s",
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
    """图1：六区域储能功率概览与活跃区域SOC时移特征。"""

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
        logging.error("storage_params.csv无法覆盖全部Q3区域")
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

    # 上半区只保留热力图主体和窄色条，不再放统计表。
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
    power_axis.set_xlabel("时段 (t) / h", fontsize=10.2)
    power_axis.set_ylabel("区域", fontsize=10.2)
    _panel_caption(
        power_axis,
        "(a) 六区域储能功率时序概览",
        y=-0.22,
        fontsize=9.2,
    )
    _style_axis(power_axis, grid_axis="none")

    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("储能净功率 / MW", color=TEXT_COLOR)
    colorbar.ax.tick_params(colors=TEXT_COLOR, labelsize=6.4)
    colorbar.outline.set_edgecolor(TEXT_COLOR)
    colorbar.outline.set_linewidth(0.6)

    # 下半区仅保留真正活跃的RegionD-F。
    active_regions = [
        region for region in ("RegionD", "RegionE", "RegionF")
        if region in regions
    ]
    if not active_regions:
        logging.error("Q3结果中缺少RegionD/RegionE/RegionF活跃区域")
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
            label="初始SOC",
        )
        axis.axhline(
            minimum,
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=0.75,
            label="SOC下限",
        )
        axis.axhline(
            capacity,
            color="#5CB85C",
            linestyle="--",
            linewidth=0.75,
            label="容量上限",
        )

        axis.set_title(region, fontsize=8.3, pad=2.0)
        axis.set_xticks([0, 1200, 2405])
        axis.set_xlabel("时段 t / h", fontsize=9.6)
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
            label="有符号功率",
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

    # 下排采用共享图例，避免图例被高密度SOC曲线遮住。
    legend_handles = [
        plt.Line2D([], [], color="#357EBD", linewidth=1.05, label="SOC"),
        plt.Line2D([], [], color=TEXT_COLOR, linestyle=":", linewidth=0.75, label="初始SOC"),
        plt.Line2D([], [], color=REFERENCE_COLOR, linestyle="--", linewidth=0.75, label="SOC下限"),
        plt.Line2D([], [], color="#5CB85C", linestyle="--", linewidth=0.75, label="容量上限"),
        plt.Line2D([], [], color="#D77071", linewidth=0.55, label="有符号功率"),
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
        "(b) RegionD–F SOC与充放电时移特征",
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
    """显示净运行结算的绝对变化，避免把跨过零点的成本写成改善率。"""

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
                f"{value:+,.0f}万元",
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
    axis.set_xlabel("方案")
    axis.set_ylabel("相对附件基准净运行结算变化 / 万元", fontsize=9.1)
    _panel_caption(axis, "(a) 净运行结算绝对变化（附件基准−方案）", y=-0.25, fontsize=9.4)
    _style_axis(axis)


def _plot_relative_improvement_metrics(axis: plt.Axes, summary: pd.DataFrame) -> None:
    metric_specs = [
        ("Carbon", "碳排放"),
        ("Peak", "峰值净购电"),
        ("Ramp", "净购电波动"),
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
    axis.axhline(0.0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.9, label="附件基准=0%")
    if all_values:
        low = min(all_values)
        high = max(all_values)
        span = max(high - low, 10.0)
        axis.set_ylim(min(-5.0, low - 0.12 * span), high + 0.16 * span)
    axis.set_xticks(x)
    axis.set_xticklabels([label for _, label in metric_specs], fontsize=6.5)
    axis.set_xlabel("指标")
    axis.set_ylabel("相对附件基准改善率 / %", fontsize=10.2)
    _panel_caption(axis, "(b) 三项运行指标相对附件基准改善率", y=-0.25, fontsize=9.4)
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
    axis.set_xlabel("区域", fontsize=10.0)
    axis.set_ylabel(ylabel, fontsize=10.0)
    _panel_caption(axis, caption, y=-0.25, fontsize=9.6)
    _style_axis(axis)


def plot_group2_optimization_effect() -> bool:
    """图2：净运行结算绝对变化、三项相对改善率与区域差异。"""

    summary = _read_table("q3_objective_summary.csv", ("Solution", "Cost", "Carbon", "Peak", "Ramp"))
    regional = _read_table("q3_regional_metrics.csv", ("Scheme", "Region", "Peak", "Ramp"))
    if summary is None or regional is None:
        return False
    summary = _numeric(summary, ("Cost", "Carbon", "Peak", "Ramp"))
    regional = _numeric(regional, ("Peak", "Ramp"))
    if not set(SCHEME_ORDER).issubset(set(summary["Solution"].astype(str))):
        logging.error("q3_objective_summary.csv缺少三类方案")
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
        "峰值净购电降低量 / MW",
        "(c) 六区域峰值净购电绝对变化",
    )
    _plot_regional_improvement(
        ramp_axis,
        regional,
        "Ramp",
        "净购电波动降低量 / MW",
        "(d) 六区域净购电波动绝对变化",
    )
    _legend(ramp_axis, loc="upper left", ncol=2, fontsize=6.0, handlelength=1.1, columnspacing=0.7)
    legend = figure.legend(
        handles=[
            plt.Line2D([], [], color=REFERENCE_COLOR, linestyle="--", linewidth=0.9, label="附件基准=0%"),
            Patch(facecolor=SCHEME_COLORS["NoStorage"], label="无储能"),
            Patch(facecolor=SCHEME_COLORS["Balanced"], edgecolor=TEXT_COLOR, linewidth=0.35, label="优化方案"),
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
        ("RenewableBalanceResidual_MW", "能源｜新能源平衡", "#5CB85C"),
        ("EnergyBalanceResidual_MW", "能源｜负荷平衡", "#EEA236"),
        ("SOCRecurrenceResidual_MWh", "储能｜SOC递推", "#357EBD"),
        ("TerminalSOCDeficit_MWh", "储能｜终端SOC缺口", "#D43F3A"),
    )
    values = np.asarray([_diagnostic_value(profile_checks, check) for check, _, _ in checks], dtype=float)
    labels = [label for _, label, _ in checks]
    colors = [color for _, _, color in checks]
    finite = values[np.isfinite(values) & (values > 0)]
    if finite.size == 0:
        axis.text(0.5, 0.5, "缺少基准诊断数据", ha="center", va="center", color=TEXT_COLOR)
        axis.set_axis_off()
        return
    y = np.asarray([0.0, 1.0, 3.0, 4.0])
    axis.barh(y, values, color=colors, height=0.55)
    axis.set_xscale("log")
    lower = min(float(finite.min()) / 3.0, 1e-8)
    upper = float(finite.max()) * 3.5
    axis.set_xlim(lower, upper)
    axis.axvline(1e-5, color=TEXT_COLOR, linestyle="--", linewidth=0.8, label="检验容差")
    for position, value in zip(y, values):
        if np.isfinite(value) and value > 0:
            axis.text(value * 1.06, position, f"{value:.4g}", va="center", fontsize=6.1, color=TEXT_COLOR)
    axis.axhline(2.0, color=GRID_COLOR, linewidth=0.8)
    axis.set_yticks(y)
    axis.set_yticklabels(labels)
    axis.set_xlabel("最大偏差（对数尺度） / MW 或 MWh")
    # 图题由主布局统一放置，使左右两组caption严格对齐。
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
            label="求解容差",
        )
    for position, value in zip(x, errors):
        if np.isfinite(value):
            error_axis.text(position, value * 1.18, f"{value:.3g}", ha="center", va="bottom", fontsize=6.1, color=TEXT_COLOR)
    error_axis.set_yscale("log")
    error_axis.set_xticks(x)
    error_axis.set_xticklabels(["|Δz|", "|ΔΦ|"])
    error_axis.set_ylabel("最优性保持误差 / 绝对值")
    error_axis.set_title("前两级目标", fontsize=7.2, pad=3.0, color=TEXT_COLOR)
    _legend(error_axis, loc="upper left", fontsize=5.5, handlelength=1.2)
    _style_axis(error_axis)

    throughput_axis.bar([0], [throughput_difference], width=0.48, color="#D43F3A")
    throughput_axis.axhline(0.0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.85, label="不增加")
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
    throughput_axis.set_ylabel("吞吐量变化 / MWh")
    throughput_axis.set_title("末级吞吐量", fontsize=7.2, pad=3.0, color=TEXT_COLOR)
    _legend(throughput_axis, loc="upper left", fontsize=5.5, handlelength=1.2)
    _style_axis(throughput_axis)
    # 右侧两幅图共用一个组题，统一由外层 caption 轴放置。


def plot_group3_validation_credibility() -> bool:
    """图3：附件基准口径诊断和三级择优保持性。"""

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

    # 外层统一保留一行图题，保证(a)(b)严格对齐。
    outer = figure.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 0.12],
        width_ratios=[0.88, 1.52],
        hspace=0.02,
        wspace=0.24,
    )

    # 左图内部再留一小段底部空白，使坐标轴更短，
    # 避免x轴标题、图例和子图题堆在一起。
    left_grid = outer[0, 0].subgridspec(
        2,
        1,
        height_ratios=[0.88, 0.12],
        hspace=0.0,
    )
    diagnostics_axis = figure.add_subplot(left_grid[0, 0])
    left_spacer = figure.add_subplot(left_grid[1, 0])
    left_spacer.axis("off")

    # 右侧检验图使用完整高度，并略加宽。
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
        "(a) 附件基准口径诊断",
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
        "(b) 三级择优保持性检验",
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
    parser = argparse.ArgumentParser(description="绘制问题3静态结果图")
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=("group1", "group2", "group3"),
        default=("group1", "group2", "group3"),
        help="只绘制指定图组；默认绘制全部三组",
    )
    args = parser.parse_args()
    _configure_style()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    if not TABLES_DIR.is_dir():
        logging.info("缺少问题3结果目录，跳过绘图：%s", TABLES_DIR)
        return 0
    figure_functions = {
        "group1": plot_group1_storage_mechanism,
        "group2": plot_group2_optimization_effect,
        "group3": plot_group3_validation_credibility,
    }
    selected_functions = [figure_functions[group] for group in args.groups]
    generated = 0
    for function in selected_functions:
        logging.info("开始绘制：%s", function.__name__)
        if function():
            generated += 1
    logging.info(
        "Q3指定图组绘制完成：生成%d/%d组图，输出目录：%s",
        generated,
        len(selected_functions),
        FIGURES_DIR,
    )
    return 0 if generated == len(selected_functions) else 1


if __name__ == "__main__":
    raise SystemExit(main())
