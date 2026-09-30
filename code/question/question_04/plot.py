"""Q4 publication plotting entry point.

Read existing CSV files and shared storage parameters without solving models or calling MILP.
Produce three figure groups with five panels: joint optimization, scenario responses,

MH/REF window checks, and full-horizon SOC. The first group compares sequential and
existing joint task schedules recomputed under the same V4 energy definitions. Read completed,
independently audited V4 scenarios from formal result directories. The third group uses
existing validation.py outputs; missing evidence is labeled explicitly without substituting rolling logs.
"""

from __future__ import annotations

import json
import logging
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


QUESTION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = QUESTION_DIR.parents[1]
RESULT_NAMESPACE = "matheuristic_v4_doccompliant"
RESULT_MARKER = ".q4_matheuristic_v4_complete.json"
JOINT_ENERGY_REFERENCE_MARKER = ".q4_joint_energy_reference_complete.json"
TABLES_DIR = QUESTION_DIR / "outputs" / RESULT_NAMESPACE / "tables"
LEGACY_JOINT_TABLES_DIR = QUESTION_DIR / "outputs" / "matheuristic_v3_doccompliant" / "tables"
SCENARIO_DIR = QUESTION_DIR / "outputs" / "scenarios"
VALIDATION_DIR = QUESTION_DIR / "outputs" / "validation"
FIGURES_DIR = QUESTION_DIR / "outputs" / "figures"
SHARED_DIR = PROJECT_DIR / "question" / "question_01" / "data" / "processed" / "shared"
MAIN_END = 2406

METRICS = ("Cost", "Carbon", "Latency", "Delay", "RenewableUnusedRate", "Peak")
GROUP1_ACTIVE_METRICS = ("Latency", "Delay", "RenewableUnusedRate")
METRIC_LABELS = {"Cost": "C", "Carbon": "E", "Latency": "L", "Delay": "J", "RenewableUnusedRate": "Q", "Peak": "P"}
SCENARIO_METRICS = ("Cost", "Carbon", "Latency", "QoS", "Utilization", "Peak")
SCENARIO_LABELS = {"Cost": "C", "Carbon": "E", "Latency": "L", "QoS": "QoS", "Utilization": "U", "Peak": "P"}
REGIONS = ("RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF")

PALETTE = {
    "blue": "#5773cc",
    "gold": "#ffb900",
    "cyan": "#23bac5",
    "orange": "#fd763f",
    "green": "#43b284",
    "deep_blue": "#0f7ba2",
    "text": "#003967",
    "grid": "#acd6ec",
    "reference": "#d77071",
    "neutral": "#7f8c8d",
    "white": "#ffffff",
}
REGION_COLORS = dict(zip(REGIONS, (PALETTE["blue"], PALETTE["gold"], PALETTE["cyan"], PALETTE["orange"], PALETTE["deep_blue"], PALETTE["green"])))
SCENARIO_CMAP = LinearSegmentedColormap.from_list("q4_diverging", [PALETTE["blue"], PALETTE["white"], PALETTE["orange"]])


def _configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "text.color": PALETTE["text"],
        "axes.edgecolor": PALETTE["text"],
        "axes.labelcolor": PALETTE["text"],
        "axes.titlecolor": PALETTE["text"],
        "xtick.color": PALETTE["text"],
        "ytick.color": PALETTE["text"],
        "legend.fontsize": 6.5,
        "figure.facecolor": PALETTE["white"],
        "axes.facecolor": PALETTE["white"],
        "savefig.facecolor": PALETTE["white"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _read_csv(path: Path, required: tuple[str, ...] = ()) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        logging.warning("Cannot read figure input: %s; %s", path, exc)
        return None
    missing = [column for column in required if column not in frame.columns]
    if missing:
        logging.warning("%s is missing columns: %s", path.name, missing)
        return None
    return frame


def _table(name: str, required: tuple[str, ...] = ()) -> pd.DataFrame | None:
    path = TABLES_DIR / name
    frame = _read_csv(path, required)
    if frame is None and not path.is_file():
        logging.info("Figure input pending: %s", name)
    return frame


def _joint_result_path(name: str) -> Path:
    """Use V4 joint output when present, otherwise the unchanged audited V3 joint result."""

    current = TABLES_DIR / name
    if current.is_file():
        return current
    legacy = LEGACY_JOINT_TABLES_DIR / name
    if legacy.is_file():
        logging.warning("V4 joint baseline has not been rerun; using existing joint results for group 1: %s.", legacy)
        return legacy
    return current


def _joint_table(name: str, required: tuple[str, ...] = ()) -> pd.DataFrame | None:
    return _read_csv(_joint_result_path(name), required)


def _token(value: object) -> str:
    return "".join(character.lower() for character in str(value) if character.isalnum())


def _column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    normalized = {_token(column): column for column in frame.columns}
    for alias in aliases:
        if _token(alias) in normalized:
            return normalized[_token(alias)]
    return None


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _style_axis(axis: plt.Axes, grid_axis: str = "y") -> None:
    if grid_axis == "none":
        axis.grid(False)
    else:
        axis.grid(True, axis=grid_axis, color=PALETTE["grid"], linewidth=0.65, alpha=0.72)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color(PALETTE["text"])
        spine.set_linewidth(0.72)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _legend(axis: plt.Axes, **kwargs) -> None:
    legend = axis.legend(frameon=False, **kwargs)
    if legend:
        for text in legend.get_texts():
            text.set_color(PALETTE["text"])


def _caption(axis: plt.Axes, text: str, y: float = -0.22) -> None:
    axis.text(0.5, y, text, transform=axis.transAxes, ha="center", va="top", fontsize=8.3, fontweight="bold", color=PALETTE["text"], clip_on=False)


def _placeholder(axis: plt.Axes, title: str, detail: str) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.text(0.5, 0.58, title, ha="center", va="center", fontsize=9, fontweight="bold", color=PALETTE["text"])
    axis.text(0.5, 0.38, detail, ha="center", va="center", fontsize=6.7, color=PALETTE["reference"], wrap=True)


def _save(figure: plt.Figure, stem: str) -> bool:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES_DIR / f"{stem}.pdf"
    png = FIGURES_DIR / f"{stem}.png"
    try:
        try:
            figure.savefig(pdf, format="pdf", bbox_inches="tight", pad_inches=0.06)
        except PermissionError:
            fallback = FIGURES_DIR / f"{stem}_new.pdf"
            figure.savefig(fallback, format="pdf", bbox_inches="tight", pad_inches=0.06)
            logging.warning("%s is locked; writing %s instead.", pdf.name, fallback.name)
        figure.savefig(png, format="png", dpi=450, bbox_inches="tight", pad_inches=0.06)
    except OSError as exc:
        logging.error("Cannot save figure: %s; %s", stem, exc)
        return False
    finally:
        plt.close(figure)
    logging.info("Figures written: %s, %s.", pdf, png)
    return True


def _metric_name(value: object) -> str | None:
    aliases = {
        "cost": "Cost", "operatingcost": "Cost", "operatingcostcny": "Cost",
        "carbon": "Carbon", "carbonemission": "Carbon", "carbonemissiontco2": "Carbon",
        "latency": "Latency", "latencyms": "Latency", "meannetworklatency": "Latency", "meannetworklatencyms": "Latency",
        "delay": "Delay", "qosloss": "Delay", "servicequalityloss": "Delay", "j": "Delay",
        "renewableunusedrate": "RenewableUnusedRate", "q": "RenewableUnusedRate",
        "peak": "Peak", "peaknetgridimport": "Peak", "peaknetgridimportmw": "Peak",
    }
    return aliases.get(_token(value))


def _row_for(frame: pd.DataFrame, preferred: tuple[str, ...]) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=object)
    selector = _column(frame, ("Scheme", "Solution", "Scenario", "Case", "Name"))
    if selector:
        labels = frame[selector].astype(str)
        for word in preferred:
            mask = labels.str.contains(word, case=False, regex=False, na=False)
            if mask.any():
                return frame.loc[mask].iloc[0]
    return frame.iloc[0]


def _metrics_from(frame: pd.DataFrame, preferred: tuple[str, ...] = ()) -> dict[str, float] | None:
    result: dict[str, float] = {}
    metric_column = _column(frame, ("Metric", "Objective", "Indicator"))
    value_column = _column(frame, ("Value", "MetricValue", "Result", "ObjectiveValue"))
    if metric_column and value_column:
        for _, row in frame.iterrows():
            name = _metric_name(row[metric_column])
            if name is None:
                continue
            value = pd.to_numeric(pd.Series([row[value_column]]), errors="coerce").iloc[0]
            if pd.notna(value):
                result[name] = float(value)
    if set(result) != set(METRICS):
        row = _row_for(frame, preferred)
        aliases = {
            "Cost": ("Cost", "OperatingCost_CNY", "OperatingCost"),
            "Carbon": ("Carbon", "CarbonEmission_tCO2", "CarbonEmission"),
            "Latency": ("Latency", "MeanNetworkLatency_ms", "MeanNetworkLatency"),
            "Delay": ("Delay", "ServiceQualityLoss", "J"),
            "RenewableUnusedRate": ("RenewableUnusedRate", "Q"),
            "Peak": ("Peak", "PeakNetGridImport_MW", "PeakNetGridImport"),
        }
        for name, candidates in aliases.items():
            column = _column(frame, candidates)
            if column is None:
                continue
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if pd.notna(value):
                result[name] = float(value)
    return result if set(result) == set(METRICS) else None


def _q4_metrics() -> dict[str, float] | None:
    frame = _joint_table("q4_objective_summary.csv", ("Metric", "Value"))
    return None if frame is None else _metrics_from(frame, ("Q4JointRolling", "Q4", "Joint"))


def _joint_energy_reference_metrics() -> dict[str, float] | None:
    marker_path = TABLES_DIR / JOINT_ENERGY_REFERENCE_MARKER
    if not marker_path.is_file():
        logging.warning("Missing V4 joint schedule energy recomputation marker: %s.", marker_path)
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        logging.warning("Cannot read V4 joint schedule energy recomputation marker: %s.", exc)
        return None
    if marker.get("complete") is not True:
        logging.warning("V4 joint schedule energy recomputation is incomplete.")
        return None
    frame = _table("q4_joint_energy_reference_metrics.csv", ("Metric", "Value"))
    return None if frame is None else _metrics_from(
        frame,
        ("ExistingJointTaskSchedule", "JointTaskSchedule", "Joint"),
    )


def _scaling() -> dict[str, tuple[float, float]] | None:
    frame = _table("q4_scaling.csv", ("Metric", "Anchor", "Scale"))
    if frame is None:
        return None
    frame = _numeric(frame, ["Anchor", "Scale"])
    result: dict[str, tuple[float, float]] = {}
    for _, row in frame.iterrows():
        name = _metric_name(row["Metric"])
        if name and pd.notna(row["Anchor"]) and pd.notna(row["Scale"]) and float(row["Scale"]) > 0:
            result[name] = (float(row["Anchor"]), float(row["Scale"]))
    return result if set(result) == set(METRICS) else None


def _active_metric_names() -> tuple[str, ...]:
    """Use explicit calibration roles; infer legacy roles without huge z values."""

    frame = _table("q4_scaling.csv", ("Metric", "Anchor", "Scale"))
    if frame is None:
        legacy_config = _read_csv(
            LEGACY_JOINT_TABLES_DIR / "q4_model_configuration.csv",
            ("Parameter", "Value"),
        )
        if legacy_config is not None:
            row = legacy_config.loc[
                legacy_config["Parameter"].astype(str).eq("OptimizationMetrics")
            ]
            if not row.empty:
                names = tuple(
                    name for name in (
                        _metric_name(token.strip())
                        for token in str(row.iloc[0]["Value"]).split(",")
                    )
                    if name is not None
                )
                if names:
                    return tuple(metric for metric in METRICS if metric in names)
        return METRICS
    frame = _numeric(frame, ["Anchor", "Scale"])
    active: list[str] = []
    for _, row in frame.iterrows():
        name = _metric_name(row["Metric"])
        if name is None or pd.isna(row["Scale"]) or float(row["Scale"]) <= 0:
            continue
        role = str(row.get("OptimizationRole", "")).strip().upper()
        if not role:
            anchor = float(row["Anchor"])
            scale = float(row["Scale"])
            role = (
                "REPORT_ONLY_DEGENERATE"
                if scale <= max(abs(anchor), 1.0) * 1e-5 + 1e-8
                else "MINIMAX_ACTIVE"
            )
        if role != "REPORT_ONLY_DEGENERATE":
            active.append(name)
    return tuple(metric for metric in METRICS if metric in active) or METRICS


def _optional_path(names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = TABLES_DIR / name
        if path.is_file():
            return path
    return None


SEQUENTIAL_METRIC_FILES = (
    "q4_sequential_objective_summary.csv",
    "q4_sequence_objective_summary.csv",
    "q4_sequential_summary.csv",
    "q4_sequential_baseline.csv",
    "q4_sequential_baseline_metrics.csv",
)
SEQUENTIAL_DISPATCH_FILES = (
    "q4_sequential_region_hour_dispatch.csv",
    "q4_sequential_dispatch.csv",
    "q4_sequential_energy_profile.csv",
    "q4_sequence_region_hour_dispatch.csv",
    "q4_sequential_baseline_dispatch.csv",
)


def _sequential_metrics() -> tuple[dict[str, float], str] | None:
    path = _optional_path(SEQUENTIAL_METRIC_FILES)
    if path is None:
        logging.warning("Q2 -> Q3 sequential baseline metrics not found: %s.", ", ".join(SEQUENTIAL_METRIC_FILES))
        return None
    frame = _read_csv(path)
    values = None if frame is None else _metrics_from(frame, ("Sequential", "Sequence", "Baseline"))
    if values is None:
        logging.warning("Sequential baseline table %s lacks complete six-objective metrics.", path.name)
        return None
    return values, path.name


def _dispatch(path: Path) -> pd.DataFrame | None:
    frame = _read_csv(path)
    if frame is None:
        return None
    hour = _column(frame, ("Hour",))
    region = _column(frame, ("Region",))
    if not hour or not region:
        return None
    frame = frame.copy()
    frame["_hour"] = pd.to_numeric(frame[hour], errors="coerce")
    frame["_region"] = frame[region].astype(str)
    return frame.loc[frame["_hour"].between(0, MAIN_END - 1)].copy()


def _aggregate(frame: pd.DataFrame) -> dict[str, float] | None:
    names = {
        "direct": ("RenewableDirectUse_MW",),
        "storage": ("RenewableCharge_MW",),
        "export": ("GridSell_MW", "RenewableExport_MW"),
        "curtailment": ("RenewableCurtailment_MW",),
        "available": ("AvailableRenewable_MW",),
        "purchase": ("GridPurchase_MW",),
    }
    columns = {key: _column(frame, value) for key, value in names.items()}
    if any(value is None for value in columns.values()):
        return None
    result = frame.copy()
    for column in columns.values():
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[list(columns.values())].isna().any().any():
        return None
    purchase = result[columns["purchase"]]
    export = result[columns["export"]]
    net_column = _column(frame, ("NetGridImport_MW",))
    net = pd.to_numeric(result[net_column], errors="coerce") if net_column else purchase - export
    peak = float(pd.DataFrame({"Region": result["_region"], "Net": net}).groupby("Region", observed=True)["Net"].max().clip(lower=0).sum())
    available = float(result[columns["available"]].sum())
    curtailment = float(result[columns["curtailment"]].sum())
    cost_column = _column(frame, ("OperatingCost_CNY",))
    if cost_column:
        cost = float(pd.to_numeric(result[cost_column], errors="coerce").sum())
    else:
        price = _column(frame, ("ElectricityPrice_CNY_per_MWh",))
        sell_price = _column(frame, ("SellPrice_CNY_per_MWh",))
        cost = float((pd.to_numeric(result[price], errors="coerce") * purchase - pd.to_numeric(result[sell_price], errors="coerce") * export).sum()) if price and sell_price else float("nan")
    carbon_column = _column(frame, ("CarbonEmission_tCO2",))
    if carbon_column:
        carbon = float(pd.to_numeric(result[carbon_column], errors="coerce").sum())
    else:
        intensity = _column(frame, ("CarbonIntensity_tCO2_per_MWh",))
        carbon = float((pd.to_numeric(result[intensity], errors="coerce") * purchase).sum()) if intensity else float("nan")
    charge = _column(frame, ("ChargePower_MW",))
    grid_charge = _column(frame, ("GridCharge_MW",))
    discharge = _column(frame, ("DischargePower_MW",))
    throughput = float("nan")
    if charge and discharge:
        charge_series = pd.to_numeric(result[charge], errors="coerce")
        if grid_charge:
            charge_series = charge_series + pd.to_numeric(result[grid_charge], errors="coerce").fillna(0)
        throughput = float((charge_series + pd.to_numeric(result[discharge], errors="coerce")).sum())
    return {
        "direct": float(result[columns["direct"]].sum()),
        "storage": float(result[columns["storage"]].sum()),
        "export": float(result[columns["export"]].sum()),
        "curtailment": curtailment,
        "available": available,
        "Cost": cost,
        "Carbon": carbon,
        "Utilization": 1 - curtailment / available if available > 1e-12 else float("nan"),
        "Peak": peak,
        "throughput": throughput,
    }


def _sequential_dispatch() -> tuple[pd.DataFrame, str] | None:
    path = _optional_path(SEQUENTIAL_DISPATCH_FILES)
    if path is None:
        logging.warning("Q2 -> Q3 sequential energy trajectory not found: %s.", ", ".join(SEQUENTIAL_DISPATCH_FILES))
        return None
    frame = _dispatch(path)
    return None if frame is None else (frame, path.name)


def _deviation(
    values: dict[str, float],
    scaling: dict[str, tuple[float, float]],
    metrics: tuple[str, ...] = METRICS,
) -> np.ndarray:
    return np.asarray(
        [max((values[name] - scaling[name][0]) / scaling[name][1], 0) for name in metrics],
        dtype=float,
    )


def _compact(value: float) -> str:
    if abs(value) >= 1e6:
        return f"{value / 1e6:+.2f}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:+.2f}k"
    return f"{value:+.3g}"


def plot_group1_joint_optimization_results() -> bool:
    q4 = _joint_energy_reference_metrics()
    scaling = _scaling()
    seq = _sequential_metrics()
    q4_dispatch = _dispatch(TABLES_DIR / "q4_joint_energy_reference_dispatch.csv")
    seq_dispatch = _sequential_dispatch()
    figure, axes = plt.subplots(1, 2, figsize=(7.35, 3.75), gridspec_kw={"width_ratios": [1.04, 0.96]})
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.80, wspace=0.36)
    ax_metric, ax_energy = axes
    if q4 is None or scaling is None or seq is None:
        _placeholder(ax_metric, "Six-objective comparison pending", "Formal sequential six-objective metrics required\nInputs are incomplete; fixed-input Q3 baseline was not substituted")
    else:
        # The retained joint task schedule was produced by the audited search
        # whose actual minimax set was L/J/Q.  C/E/P are re-evaluated below as
        # report-only outcomes and must not be retroactively presented as
        # objectives of a V4 joint search that was never run.
        active_metrics = GROUP1_ACTIVE_METRICS
        sequence = _deviation(seq[0], scaling, active_metrics)
        joint = _deviation(q4, scaling, active_metrics)
        x = np.arange(len(active_metrics))
        width = 0.34
        ax_metric.bar(x - width / 2, sequence, width, color=PALETTE["gold"], label="Sequential task schedule", edgecolor=PALETTE["text"], linewidth=0.35)
        ax_metric.bar(x + width / 2, joint, width, color=PALETTE["blue"], label="Joint task schedule", edgecolor=PALETTE["text"], linewidth=0.35)
        ymax = max(float(np.max(np.r_[sequence, joint])), 0.08)
        for values, offset in ((sequence, -width / 2), (joint, width / 2)):
            for index, value in enumerate(values):
                ax_metric.text(index + offset, value + ymax * 0.025, f"{value:.2g}", ha="center", va="bottom", fontsize=5.8)
        z_seq, z_joint = float(np.max(sequence)), float(np.max(joint))
        ax_metric.axhline(z_seq, color=PALETTE["gold"], linestyle="--", linewidth=0.9, label=r"$z^{SEQ}$")
        ax_metric.axhline(z_joint, color=PALETTE["blue"], linestyle=":", linewidth=1.0, label=r"$z^{JOINT}$")
        change = (z_seq - z_joint) / z_seq if z_seq > 1e-12 else 0
        ax_metric.text(0.98, 0.96, f"z_SEQ={z_seq:.3g}\nz_JOINT={z_joint:.3g}\nMaximum deviation reduction={change:+.1%}", transform=ax_metric.transAxes, ha="right", va="top", fontsize=6.5, bbox={"boxstyle": "round,pad=0.25", "facecolor": PALETTE["white"], "edgecolor": PALETTE["grid"], "linewidth": 0.6})
        ax_metric.set_xticks(x, [METRIC_LABELS[name] for name in active_metrics])
        ax_metric.set_xlabel("Active metrics in minimax optimization")
        ax_metric.set_ylabel(r"Normalized deviation $d_m$")
        _style_axis(ax_metric)
        handles, labels = ax_metric.get_legend_handles_labels()
        _legend(ax_metric, handles=handles, labels=labels, loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=2, columnspacing=0.65, handletextpad=0.35, borderaxespad=0.0)
    _caption(ax_metric, "(a) Active metric deviations under unified V4 definitions")
    if q4_dispatch is None or seq_dispatch is None:
        _placeholder(ax_energy, "Energy comparison pending", "Complete sequential energy trajectory required\nThe Q4 trajectory cannot be combined directly with the fixed-input baseline")
    else:
        current = _aggregate(q4_dispatch)
        sequence = _aggregate(seq_dispatch[0])
        if current is None or sequence is None:
            _placeholder(ax_energy, "Energy fields pending", "Both sequential and Q4 trajectories must contain direct use, storage, export, and curtailment")
        else:
            keys = ("direct", "storage", "export", "curtailment")
            labels = ("Direct use", "Storage charging", "Renewable export", "Renewable curtailment")
            delta = np.asarray([current[key] - sequence[key] for key in keys])
            x = np.arange(len(keys))
            colors = [PALETTE["blue"] if value >= 0 else PALETTE["orange"] for value in delta]
            ax_energy.bar(x, delta, color=colors, edgecolor=PALETTE["text"], linewidth=0.4)
            ax_energy.axhline(0, color=PALETTE["text"], linewidth=0.75)
            bound = max(float(np.max(np.abs(delta))), 1)
            for index, value in enumerate(delta):
                ax_energy.text(index, value + (0.035 * bound if value >= 0 else -0.035 * bound), _compact(float(value)), ha="center", va="bottom" if value >= 0 else "top", fontsize=6.2)
            lower = min(float(np.min(delta)), 0.0)
            upper = max(float(np.max(delta)), 0.0)
            label_padding = max(0.18 * bound, 1.0)
            ax_energy.set_ylim(lower - label_padding, upper + label_padding)
            callout = f"ΔC={_compact(current['Cost'] - sequence['Cost'])} CNY\nΔU={(current['Utilization'] - sequence['Utilization']) * 100:+.2f} pp\nΔP={_compact(current['Peak'] - sequence['Peak'])} MW\nΔThroughput={_compact(current['throughput'] - sequence['throughput'])} MWh"
            ax_energy.text(0.98, 0.96, callout, transform=ax_energy.transAxes, ha="right", va="top", fontsize=6.2, bbox={"boxstyle": "round,pad=0.25", "facecolor": PALETTE["white"], "edgecolor": PALETTE["grid"], "linewidth": 0.6})
            ax_energy.set_xticks(x, labels, rotation=18, ha="right")
            ax_energy.set_ylabel("Cumulative change (joint - sequential) / MWh")
            _style_axis(ax_energy)
    _caption(ax_energy, "(b) Renewable and storage allocation changes under unified V4 energy definitions")
    return _save(figure, "q4_group1_joint_optimization_results")


def _scenario_metric(value: object) -> tuple[str, bool] | None:
    token = _token(value)
    if token in {"cost", "operatingcost", "operatingcostcny", "c"}: return "Cost", False
    if token in {"carbon", "carbonemission", "carbonemissiontco2", "e"}: return "Carbon", False
    if token in {"latency", "latencyms", "meannetworklatency", "meannetworklatencyms", "l"}: return "Latency", False
    if token in {"qos", "servicequality", "servicequalityscore"}: return "QoS", False
    if token in {"qosloss", "j", "serviceloss", "servicequalityloss"}: return "QoS", True
    if token in {"utilization", "renewableutilization", "renewableutilizationrate", "u"}: return "Utilization", False
    if token in {"renewableunusedrate", "q"}: return "Utilization", True
    if token in {"peak", "peaknetgridimport", "peaknetgridimportmw", "p"}: return "Peak", False
    return None


SCENARIO_FILES = ("q4_scenario_metrics.csv", "q4_scenarios.csv", "q4_scenario_results.csv", "q4_scenario_summary.csv")


def _scenario_display_name(directory_name: str, scenario: dict[str, object]) -> str:
    kind = str(scenario.get("ScenarioKind", "")).strip().lower()
    if kind == "low_carbon_reference":
        return "Low-carbon reference"
    if kind == "carbon_constraint":
        value = scenario.get("CarbonLambda")
        return f"Carbon constraint λ={float(value):g}" if value is not None else "Carbon constraint"
    if kind == "flat_price":
        return "Parity electricity prices"
    if kind == "low_variability_renewable":
        value = scenario.get("RenewableSmoothingGamma")
        return f"Renewable smoothing γ={float(value):g}" if value is not None else "Renewable smoothing"
    return directory_name


def _scenario_metrics_from_summary(frame: pd.DataFrame) -> dict[str, float] | None:
    metric_column = _column(frame, ("Metric", "Objective", "Indicator"))
    value_column = _column(frame, ("Value", "MetricValue", "Result", "ObjectiveValue"))
    if metric_column is None or value_column is None:
        return None
    result: dict[str, float] = {}
    for _, row in frame.iterrows():
        parsed = _scenario_metric(row[metric_column])
        if parsed is None:
            continue
        metric, reverse = parsed
        value = pd.to_numeric(pd.Series([row[value_column]]), errors="coerce").iloc[0]
        if pd.notna(value):
            result[metric] = float(1.0 - value if reverse else value)
    return result if set(result) == set(SCENARIO_METRICS) else None


def _load_v4_scenarios() -> tuple[pd.DataFrame, str, str] | None:
    """Load the baseline and completed V4 scenario directories directly."""

    roots: list[tuple[str, Path, dict[str, object]]] = [("Q4 00 baseline", TABLES_DIR, {})]
    if SCENARIO_DIR.is_dir():
        for directory in sorted(SCENARIO_DIR.iterdir(), key=lambda path: path.name):
            if not directory.is_dir():
                continue
            tables = directory / RESULT_NAMESPACE / "tables"
            marker_path = tables / RESULT_MARKER
            summary_path = tables / "q4_objective_summary.csv"
            metadata_path = tables / "q4_scenario_metadata.json"
            if not (marker_path.is_file() and summary_path.is_file()):
                continue
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
                logging.warning("Cannot read V4 scenario marker: %s; %s", marker_path, exc)
                continue
            if marker.get("complete") is not True or marker.get("audit_passed") is not True:
                logging.warning("Skipping incomplete or unaudited V4 scenario: %s.", directory.name)
                continue
            scenario = dict(marker.get("scenario", {}))
            if metadata_path.is_file():
                try:
                    scenario.update(json.loads(metadata_path.read_text(encoding="utf-8")))
                except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
                    logging.warning("Cannot read V4 scenario metadata; using completion marker: %s; %s", metadata_path, exc)
            roots.append((_scenario_display_name(directory.name, scenario), tables, scenario))

    records: list[dict[str, object]] = []
    for label, tables, _ in roots:
        summary_path = tables / "q4_objective_summary.csv"
        frame = _read_csv(summary_path, ("Metric", "Value"))
        values = None if frame is None else _scenario_metrics_from_summary(frame)
        if values is None:
            logging.warning("V4 scenario summary lacks six plotting metrics: %s.", summary_path)
            continue
        records.append({"Scenario": label, **values})

    if len(records) < 2:
        return None
    wide = pd.DataFrame(records).set_index("Scenario")
    wide = wide.loc[:, list(SCENARIO_METRICS)].dropna(how="any")
    baseline = "Q4 00 baseline"
    if baseline not in wide.index or wide.empty:
        logging.warning("V4 scenario plotting input lacks the Q4 00 baseline or complete metrics.")
        return None
    logging.info("V4 scenario plotting inputs loaded: %d groups, baseline=%s.", len(wide), baseline)
    return wide, baseline, "Formal V4 scenario result directories"


def _load_scenarios() -> tuple[pd.DataFrame, str, str] | None:
    v4 = _load_v4_scenarios()
    if v4 is not None:
        return v4
    path = _optional_path(SCENARIO_FILES)
    if path is None:
        logging.warning("Q4 scenario result table not found: %s.", ", ".join(SCENARIO_FILES))
        return None
    frame = _read_csv(path)
    if frame is None:
        return None
    scenario_column = _column(frame, ("Scenario", "ScenarioName", "Case", "Name"))
    if scenario_column is None:
        return None
    metric_column = _column(frame, ("Metric", "Objective", "Indicator"))
    value_column = _column(frame, ("Value", "MetricValue", "Result", "ObjectiveValue"))
    records: list[dict[str, object]] = []
    if metric_column and value_column:
        for _, row in frame.iterrows():
            parsed = _scenario_metric(row[metric_column])
            if parsed is None:
                continue
            metric, reverse = parsed
            value = pd.to_numeric(pd.Series([row[value_column]]), errors="coerce").iloc[0]
            if pd.notna(value):
                records.append({"Scenario": str(row[scenario_column]), "Metric": metric, "Value": float(1 - value if reverse else value)})
        long = pd.DataFrame(records)
        if long.empty:
            return None
        wide = long.pivot_table(index="Scenario", columns="Metric", values="Value", aggfunc="mean")
    else:
        wide = pd.DataFrame(index=frame[scenario_column].astype(str))
        aliases = {
            "Cost": ("Cost", "OperatingCost_CNY", "C"),
            "Carbon": ("Carbon", "CarbonEmission_tCO2", "E"),
            "Latency": ("Latency", "MeanNetworkLatency_ms", "L"),
            "QoS": ("QoS", "ServiceQuality", "J"),
            "Utilization": ("RenewableUtilizationRate", "RenewableUtilization", "U", "RenewableUnusedRate", "Q"),
            "Peak": ("Peak", "PeakNetGridImport_MW", "P"),
        }
        for metric, candidates in aliases.items():
            column = _column(frame, candidates)
            if column is None:
                continue
            series = pd.to_numeric(frame[column], errors="coerce")
            if metric == "QoS" and _token(column) in {"j", "serviceloss", "servicequalityloss"}:
                series = 1 - series
            if metric == "Utilization" and _token(column) in {"renewableunusedrate", "q"}:
                series = 1 - series
            wide[metric] = series.to_numpy()
    if any(metric not in wide.columns for metric in SCENARIO_METRICS):
        return None
    wide = wide.loc[:, list(SCENARIO_METRICS)].dropna(how="any")
    baseline = next((str(label) for label in wide.index if any(word in str(label).lower() for word in ("base", "baseline", "default", "\u57fa\u51c6", "\u9ed8\u8ba4"))), None)
    if baseline is None or wide.empty:
        logging.warning("Scenario table %s lacks an explicit baseline or complete metrics.", path.name)
        return None
    return wide, baseline, path.name


def _change_text(value: float, mode: str) -> str:
    if not np.isfinite(value):
        return "—"
    if mode == "pp":
        return f"{value * 100:+.1f}pp"
    if mode == "relative":
        return f"{value * 100:+.1f}%"
    return f"{value:+.3g}"


def plot_group2_scenario_response() -> bool:
    loaded = _load_scenarios()
    figure, axis = plt.subplots(figsize=(7.15, 3.65))
    figure.subplots_adjust(left=0.15, right=0.87, bottom=0.28, top=0.86)
    if loaded is None:
        _placeholder(axis, "Q4 scenario responses pending", "Formal V4 baseline and at least one complete audited V4 scenario required\nNo usable V4 summary found; λ values and scenario counts are not fabricated")
        _caption(axis, "Strategy response matrix across scenario mechanisms")
        return _save(figure, "q4_group2_scenario_response")
    wide, baseline, source = loaded
    base = wide.loc[baseline]
    changes = np.zeros((len(wide), len(SCENARIO_METRICS)))
    modes: list[str] = []
    for column_index, metric in enumerate(SCENARIO_METRICS):
        base_value = float(base[metric])
        mode = "pp" if metric in {"QoS", "Utilization"} else ("absolute" if abs(base_value) <= 1e-12 else "relative")
        modes.append(mode)
        for row_index, value in enumerate(wide[metric].to_numpy(dtype=float)):
            changes[row_index, column_index] = value - base_value if mode != "relative" else (value - base_value) / abs(base_value)
    limit = max(float(np.max(np.abs(changes))), 1e-8)
    image = axis.imshow(changes, aspect="auto", cmap=SCENARIO_CMAP, norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    axis.set_xticks(np.arange(len(SCENARIO_METRICS)), [SCENARIO_LABELS[metric] for metric in SCENARIO_METRICS])
    axis.set_yticks(np.arange(len(wide)), wide.index.astype(str))
    axis.set_xlabel("Final metrics; QoS=1-J, U=1-Q")
    for row_index in range(len(wide)):
        for column_index in range(len(SCENARIO_METRICS)):
            value = changes[row_index, column_index]
            color = PALETTE["white"] if abs(value) > limit * 0.54 else PALETTE["text"]
            axis.text(column_index, row_index, _change_text(value, modes[column_index]), ha="center", va="center", fontsize=6.2, color=color)
    axis.set_xticks(np.arange(-0.5, len(SCENARIO_METRICS), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(wide), 1), minor=True)
    axis.grid(which="minor", color=PALETTE["white"], linewidth=1)
    axis.tick_params(which="minor", bottom=False, left=False)
    _style_axis(axis, "none")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.035)
    colorbar.set_label("Relative change: fraction / pp / absolute change for zero baseline", fontsize=7, color=PALETTE["text"])
    colorbar.ax.tick_params(colors=PALETTE["text"], labelsize=6.5)
    _caption(axis, "Scenario metric responses relative to baseline; positive values indicate increases", y=-0.27)
    return _save(figure, "q4_group2_scenario_response")


VALIDATION_WINDOW_FILES = ("q4_reference_window_comparison.csv", "q4_validation_window_summary.csv")


def _validation_report_ready() -> bool:
    """Only read validation tables after validation has atomically completed."""

    report_path = VALIDATION_DIR / "q4_validation_report.json"
    if not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not report.get("completed_at"):
            return False
        report_mtime = report_path.stat().st_mtime
        # A selection/summary newer than the report means another validation run
        # may still be writing.  Do not plot historical output as current output.
        for name in VALIDATION_WINDOW_FILES + ("q4_validation_summary.csv", "q4_reference_window_selection.csv"):
            path = VALIDATION_DIR / name
            if path.is_file() and path.stat().st_mtime > report_mtime + 1.0:
                return False
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _numeric_series(frame: pd.DataFrame, aliases: tuple[str, ...]) -> pd.Series:
    column = _column(frame, aliases)
    if column is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "passed", "\u901a\u8fc7"}


def _validation_window_label(row: pd.Series, window: float) -> str:
    role = str(row.get("ReferenceRole", row.get("ScenarioName", ""))).upper()
    if "MEDIAN" in role:
        return "Median-runtime window"
    if "LONGEST" in role:
        return "Longest-runtime window"
    if "GPU" in role or "COMPUTE" in role:
        return "GPU/compute pressure"
    if "DEADLINE" in role:
        return "Dense deadlines"
    if "SOC" in role or "RENEWABLE" in role:
        return "SOC/renewable pressure"
    if "ORDINARY" in role:
        return "Ordinary window"
    return f"Window {int(window)}"


def _validation_window_evidence() -> tuple[pd.DataFrame, str, str] | None:
    """Load completed validation output; never start a solver from the plot entry."""

    if not _validation_report_ready():
        logging.info("validation.py has no stable completion report; group 3 will not read historical validation tables.")
        return None
    for name in VALIDATION_WINDOW_FILES:
        path = VALIDATION_DIR / name
        frame = _read_csv(path)
        if frame is None or frame.empty:
            continue
        window_column = _column(frame, ("WindowStart", "Window", "Tau", "WindowID"))
        if window_column is None:
            continue
        result = pd.DataFrame(index=frame.index)
        result["Window"] = pd.to_numeric(frame[window_column], errors="coerce")
        result = result.loc[result["Window"].notna()].copy()
        if result.empty:
            continue
        source_frame = frame.loc[result.index]
        result["Label"] = [
            _validation_window_label(row, float(window))
            for (_, row), window in zip(source_frame.iterrows(), result["Window"])
        ]
        status_column = _column(source_frame, ("SolverStatus", "MILPAttemptStatus", "Status"))
        reference_status_column = _column(source_frame, ("ReferenceStatus", "Status"))
        audit_column = _column(source_frame, ("IndependentAuditPassed", "AuditPassed", "Passed"))
        result["SolverStatus"] = source_frame[status_column].astype(str).to_numpy() if status_column else "UNKNOWN"
        result["ReferenceStatus"] = source_frame[reference_status_column].astype(str).to_numpy() if reference_status_column else "UNKNOWN"
        result["AuditPassed"] = source_frame[audit_column].map(_bool_value).to_numpy() if audit_column else False
        result["MIPGap"] = _numeric_series(source_frame, ("MIPGap", "Gap")).to_numpy()
        result["MIPGapTarget"] = _numeric_series(source_frame, ("ReferenceMIPGapTarget", "MIPGapTarget", "TargetMIPGap")).to_numpy()
        result["ElapsedSeconds"] = _numeric_series(source_frame, ("ElapsedSeconds", "Elapsed", "TimeSeconds")).to_numpy()
        result["MaxEnergyBalanceError"] = _numeric_series(source_frame, ("MaxEnergyBalanceError", "MaxResidual", "EnergyResidual")).to_numpy()
        mode = "reference" if name == "q4_reference_window_comparison.csv" else "audit"
        logging.info("Validation plotting inputs loaded: %d windows, source=%s.", len(result), name)
        return result.reset_index(drop=True), name, mode
    logging.warning("Completed validation window table not found: %s.", ", ".join(VALIDATION_WINDOW_FILES))
    return None


def _plot_validation_window_evidence(axis: plt.Axes, loaded: tuple[pd.DataFrame, str, str]) -> None:
    frame, source, mode = loaded
    if mode == "reference" and frame["MIPGap"].notna().any():
        local = frame.loc[frame["MIPGap"].notna()].copy()
        x = np.arange(len(local))
        gaps = local["MIPGap"].to_numpy(dtype=float) * 100.0
        targets = local["MIPGapTarget"].fillna(0.05).to_numpy(dtype=float) * 100.0
        colors = [PALETTE["blue"] if gap <= target + 1e-9 else PALETTE["orange"] for gap, target in zip(gaps, targets)]
        axis.bar(x, gaps, color=colors, edgecolor=PALETTE["text"], linewidth=0.4, label="Actual MIP gap")
        target = float(np.nanmedian(targets))
        axis.axhline(target, color=PALETTE["reference"], linestyle="--", linewidth=0.9, label=f"Target MIP gap={target:g}%")
        ymax = max(float(np.max(gaps)), target, 1.0)
        for index, (_, row) in enumerate(local.iterrows()):
            status = str(row["SolverStatus"]).replace("_", " ")
            status = status.replace("TIME LIMIT FEASIBLE", "TIME LIMIT")
            audit = "Audit passed" if bool(row["AuditPassed"]) else "Audit failed"
            axis.text(index, gaps[index] + ymax * 0.04, f"{gaps[index]:.2f}%\n{status}\n{audit}", ha="center", va="bottom", fontsize=5.8)
        axis.set_xticks(x, local["Label"].astype(str))
        axis.set_ylabel("MIP gap / %")
        axis.set_ylim(0, ymax * 1.32)
        _style_axis(axis)
        _legend(axis, loc="upper right", bbox_to_anchor=(0.98, 0.79))
        return
    checks = ("AuditPassed",)
    check_labels = ("Independent window audit",)
    matrix = frame.loc[:, list(checks)].astype(float).to_numpy()
    image = axis.imshow(matrix, aspect="auto", cmap=LinearSegmentedColormap.from_list("q4_audit", [PALETTE["white"], PALETTE["blue"]]), vmin=0.0, vmax=1.0)
    axis.set_xticks(np.arange(len(checks)), check_labels)
    axis.set_yticks(np.arange(len(frame)), frame["Label"].astype(str))
    for row_index in range(len(frame)):
        for column_index in range(len(checks)):
            passed = bool(matrix[row_index, column_index])
            axis.text(column_index, row_index, "PASS" if passed else "FAIL", ha="center", va="center", fontsize=6.0, color=PALETTE["text"] if passed else PALETTE["reference"])
    axis.set_xlabel("Independent audit status")
    axis.tick_params(axis="x", bottom=False)
    _style_axis(axis, "none")
    axis.text(0.98, 0.02, "Source: representative-window validation results", transform=axis.transAxes, ha="right", va="bottom", fontsize=5.8, color=PALETTE["text"])


def plot_group3_model_validation() -> bool:
    validation = _validation_window_evidence()
    soc = _soc_trajectory()
    figure, axes = plt.subplots(1, 2, figsize=(7.35, 4.05), gridspec_kw={"width_ratios": [0.92, 1.45]})
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.79, wspace=0.34)
    ax_pair, ax_soc = axes
    if validation is None:
        _placeholder(ax_pair, "Representative validation pending", "Complete representative-window validation to obtain its result table")
    else:
        _plot_validation_window_evidence(ax_pair, validation)
    _caption(ax_pair, "(a) Representative-window MIP gaps and independent audits")
    if soc is None:
        _placeholder(ax_soc, "SOC trajectory pending", "Complete full-horizon dispatch and shared storage parameters required")
    else:
        trajectory, terminal_ok = soc
        for region in REGIONS:
            local = trajectory.loc[trajectory["Region"].eq(region)]
            if local.empty:
                continue
            color = REGION_COLORS[region]
            ax_soc.plot(local["Hour"], local["SOCNorm"], linewidth=0.75, color=color, label=region.replace("Region", "Region"))
            ax_soc.scatter([0], [local["InitialNorm"].iloc[0]], s=18, facecolor=PALETTE["white"], edgecolor=color, linewidth=0.8, zorder=4)
            ax_soc.scatter([MAIN_END], [local.loc[local["Hour"].eq(MAIN_END), "SOCNorm"].iloc[0]], s=18, facecolor=color, edgecolor=PALETTE["text"], linewidth=0.45, zorder=4)
        ax_soc.axhline(0, color=PALETTE["reference"], linestyle="--", linewidth=0.8, label="SOC lower bound")
        ax_soc.axhline(1, color=PALETTE["reference"], linestyle=":", linewidth=0.9, label="SOC upper bound")
        ax_soc.text(0.98, 0.12, r"$S_{r,2406}\geq InitialSOC_r$" + (": all satisfied" if terminal_ok else ": violations exist"), transform=ax_soc.transAxes, ha="right", va="bottom", fontsize=6.5, bbox={"boxstyle": "round,pad=0.25", "facecolor": PALETTE["white"], "edgecolor": PALETTE["grid"], "linewidth": 0.6})
        ax_soc.set_xlim(0, MAIN_END)
        ax_soc.set_ylim(-0.035, 1.035)
        ax_soc.set_xlabel("Time t / h")
        ax_soc.set_ylabel(r"Normalized storage state $SOC^{norm}_{rt}$")
        _style_axis(ax_soc, "both")
        handles, labels = ax_soc.get_legend_handles_labels()
        _legend(ax_soc, handles=handles, labels=labels, loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=4, columnspacing=0.7, handletextpad=0.35, borderaxespad=0.0)
    _caption(ax_soc, "(b) SOC feasibility and terminal recovery in six regions")
    return _save(figure, "q4_group3_model_validation")


def _soc_trajectory() -> tuple[pd.DataFrame, bool] | None:
    dispatch = _table("q4_joint_energy_reference_dispatch.csv")
    storage = _read_csv(SHARED_DIR / "storage_params.csv", ("Region", "StorageCapacity_MWh", "MinSOC_MWh", "InitialSOC_MWh"))
    if dispatch is None or storage is None:
        return None
    required = {"Hour", "Region", "SOCStart_MWh", "SOCEnd_MWh"}
    if not required.issubset(dispatch.columns):
        return None
    dispatch = _numeric(dispatch, ["Hour", "SOCStart_MWh", "SOCEnd_MWh"])
    storage = _numeric(storage, ["StorageCapacity_MWh", "MinSOC_MWh", "InitialSOC_MWh"]).set_index("Region")
    rows: list[dict[str, object]] = []
    terminal_ok = True
    available_regions = set(dispatch["Region"].astype(str))
    for region in REGIONS:
        if region not in storage.index or region not in available_regions:
            continue
        local = dispatch.loc[dispatch["Region"].astype(str).eq(region)].sort_values("Hour", kind="stable")
        local = local.loc[local["Hour"].between(0, MAIN_END - 1)]
        if len(local) != MAIN_END or not np.array_equal(local["Hour"].to_numpy(dtype=int), np.arange(MAIN_END)):
            return None
        minimum = float(storage.loc[region, "MinSOC_MWh"])
        capacity = float(storage.loc[region, "StorageCapacity_MWh"])
        initial = float(storage.loc[region, "InitialSOC_MWh"])
        if capacity <= minimum:
            return None
        soc = np.r_[float(local.iloc[0]["SOCStart_MWh"]), local["SOCEnd_MWh"].to_numpy(dtype=float)]
        normalized = (soc - minimum) / (capacity - minimum)
        initial_norm = (initial - minimum) / (capacity - minimum)
        terminal_ok = terminal_ok and bool(normalized[-1] >= initial_norm - 1e-6)
        rows.extend({"Region": region, "Hour": hour, "SOCNorm": value, "InitialNorm": initial_norm} for hour, value in enumerate(normalized))
    trajectory = pd.DataFrame(rows)
    return (trajectory, terminal_ok) if not trajectory.empty else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Q4 publication plotting entry point")
    parser.add_argument(
        "--groups", nargs="+", choices=("group1", "group2", "group3"),
        default=("group1", "group2", "group3"),
        help="Plot the selected figure groups only; default: all three groups",
    )
    args = parser.parse_args()
    _configure_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    plotters = {
        "group1": plot_group1_joint_optimization_results,
        "group2": plot_group2_scenario_response,
        "group3": plot_group3_model_validation,
    }
    result = {name: plotters[name]() for name in args.groups}
    logging.info("Q4 plotting completion status: %s.", result)
    return 0 if all(result.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
