"""问题4独立检验入口。

本脚本只读取正式Q4 V4结果和已经生成的验证产物，不修改正式模型结果。
默认执行三条证据链：

1. 对V4主结果做全时域独立硬约束审计和指标重算；
2. 检查两个代表窗口的完整联合MILP对照是否已经形成；
3. 独立复核Q2任务种子→V4能源参考顺序基准，并检查同版本Q4情景结果。

完整联合MILP窗口对照计算量较大，但正式检验默认会实际运行两个代表窗口，
每个窗口默认上限300秒。若只想做快速文件级检查，可显式增加
``--skip-reference-windows``。
验证结果写入 ``outputs/validation/q4_validation_*.csv/json``，不覆盖
``matheuristic_v4_doccompliant``中的正式结果表。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


QUESTION_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = QUESTION_DIR / "outputs"
V4_ROOT = OUTPUTS_DIR / "matheuristic_v4_doccompliant"
V4_TABLES_DIR = V4_ROOT / "tables"
V4_VALIDATION_DIR = V4_ROOT / "validation"
SCENARIO_DIR = OUTPUTS_DIR / "scenarios"
VALIDATION_DIR = OUTPUTS_DIR / "validation"
LOG_DIR = OUTPUTS_DIR / "logs"
LOG_PATH = LOG_DIR / "question_04_validation.log"

MODEL_VERSION = "Q4_MATHEURISTIC_V4_PHYSICAL_TIEBREAK_DOCUMENT_COMPLIANT"
DEFAULT_TOLERANCE = 1e-6
REFERENCE_TIME_LIMIT_SECONDS = 600.0
REFERENCE_MIP_GAP = 0.05

try:
    import model as q4_model
except Exception as exc:  # pragma: no cover - exercised only by broken environments
    q4_model = None
    MODEL_IMPORT_ERROR = exc
else:
    MODEL_IMPORT_ERROR = None


def _configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"不支持的日志级别：{level_name}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logging.basicConfig(
        level=level,
        handlers=[stream, file_handler],
        force=True,
    )


def _running_q4_model_processes() -> tuple[list[dict[str, object]], str | None]:
    """Fail closed when a Q4 model/scenario process is still running.

    A completed marker from an earlier run is not evidence for the current run.
    The check deliberately uses a fixed PowerShell query and does not interpolate
    any user input into the command line.
    """

    if os.name != "nt":
        return [], None
    query = r"""
$items = @(Get-CimInstance -ClassName Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe' OR Name = 'uv.exe'" -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -in @('python.exe', 'pythonw.exe', 'uv.exe') -and
        $_.CommandLine -and
        $_.CommandLine -match 'question_04[\\/].*model\.py'
    } |
    Select-Object ProcessId, Name, CreationDate, CommandLine)
$items | ConvertTo-Json -Compress -Depth 4
"""
    try:
        completed = subprocess.run(
            [
                shutil.which("pwsh.exe") or shutil.which("powershell.exe") or "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                query,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [], f"无法检查Q4进程状态：{exc!r}"
    if completed.returncode != 0:
        return [], (
            "检查Q4进程状态的PowerShell返回失败："
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    output = completed.stdout.strip()
    if not output:
        return [], None
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError as exc:
        return [], f"Q4进程状态输出无法解析：{exc!r}"
    if isinstance(parsed, dict):
        return [parsed], None
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)], None
    return [], "Q4进程状态输出类型异常"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"无法序列化类型：{type(value)!r}")


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _read_csv(path: Path, required: tuple[str, ...] = ()) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"缺少检验输入文件：{path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name}缺少字段：{missing}")
    return frame


def _parse_bool_series(series: pd.Series) -> pd.Series:
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "pass": True,
        "false": False,
        "0": False,
        "no": False,
        "fail": False,
    }
    return series.astype(str).str.strip().str.lower().map(mapping)


def _record(
    records: list[dict[str, object]],
    test: str,
    item: str,
    status: str,
    evidence: str,
    criterion: str,
    action: str = "",
    value: object = None,
    tolerance: object = None,
) -> None:
    records.append(
        {
            "Test": test,
            "Item": item,
            "Status": status,
            "Value": value,
            "Tolerance": tolerance,
            "Evidence": evidence,
            "Criterion": criterion,
            "IfNotPassed": action,
        }
    )


def _strip_checks(payload: dict[str, object]) -> tuple[dict[str, object], pd.DataFrame | None]:
    clean = dict(payload)
    checks = clean.pop("checks", None)
    return clean, checks if isinstance(checks, pd.DataFrame) else None


def _summary_metrics(summary: pd.DataFrame) -> dict[str, float]:
    """把不同输出版本的指标列名统一到独立复算口径。"""

    if not {"Metric", "Value"}.issubset(summary.columns):
        return {}
    aliases = {
        "Cost": "Cost",
        "Carbon": "Carbon",
        "Latency_ms": "Latency",
        "Latency": "Latency",
        "QoSLoss": "Delay",
        "Delay": "Delay",
        "RenewableUnusedRate": "RenewableUnusedRate",
        "Peak": "Peak",
    }
    values: dict[str, float] = {}
    for row in summary.itertuples(index=False):
        source = str(row.Metric)
        target = aliases.get(source)
        if target is None:
            continue
        number = float(row.Value)
        if math.isfinite(number):
            values[target] = number
    return values


def _metric_rows(
    scheme: str,
    independent: dict[str, float],
    reported: dict[str, float] | None = None,
    *,
    tolerance: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric in q4_model.METRICS:
        independent_value = independent.get(metric)
        reported_value = (reported or {}).get(metric)
        difference = (
            float(independent_value) - float(reported_value)
            if independent_value is not None and reported_value is not None
            else float("nan")
        )
        rows.append(
            {
                "Scheme": scheme,
                "Metric": metric,
                "IndependentValue": independent_value,
                "ReportedValue": reported_value,
                "AbsoluteDifference": abs(difference) if math.isfinite(difference) else np.nan,
                "Passed": bool(math.isfinite(difference) and abs(difference) <= tolerance),
            }
        )
    return rows


def _validate_fixed_energy_reference_only(
    tolerance: float,
    *,
    input_prefix: str,
    output_prefix: str,
    scheme: str,
    baseline_status: str,
    scenario_status: str,
    scope: str,
) -> dict[str, object]:
    """Independently audit one fixed-task V4 energy evaluation."""

    if q4_model is None:
        raise RuntimeError(f"Q4模型模块导入失败：{MODEL_IMPORT_ERROR!r}")
    config = q4_model.ModelConfig()
    data = q4_model.load_data()
    paths = q4_model._matheuristic_paths()
    assignments = _read_csv(
        paths.tables / f"{input_prefix}_assignments.csv",
        ("TaskID", "TargetRegion", "StartHour", "FinishHour"),
    )
    dispatch = _read_csv(
        paths.tables / f"{input_prefix}_dispatch.csv",
        ("Hour", "Region", "GridPurchase_MW", "SOCEnd_MWh"),
    )
    reported_frame = _read_csv(
        paths.tables / f"{input_prefix}_metrics.csv",
        ("Metric", "Value"),
    )
    reported = _summary_metrics(reported_frame)
    audit = q4_model.independent_hard_constraint_audit(
        data,
        assignments,
        dispatch,
        solver_metrics=reported,
        baseline_status=baseline_status,
        scenario_status=scenario_status,
        tolerance=tolerance,
        qos_weights=q4_model._qos_weight_map(config),
    )
    independent = dict(audit.get("independent_metrics", {}))
    metric_rows = _metric_rows(
        scheme,
        independent,
        reported,
        tolerance=tolerance,
    )
    checks = audit.pop("checks", pd.DataFrame())
    audit_checks_path = V4_VALIDATION_DIR / f"{output_prefix}_audit.csv"
    audit_json_path = V4_VALIDATION_DIR / f"{output_prefix}.json"
    metric_path = V4_VALIDATION_DIR / f"{output_prefix}_metrics.csv"
    _write_csv(checks, audit_checks_path)
    _write_csv(pd.DataFrame(metric_rows), metric_path)
    audit_payload = dict(audit)
    audit_payload["model_version"] = MODEL_VERSION
    audit_payload["scope"] = scope
    audit_payload["independent_metrics"] = independent
    audit_payload["reported_metrics"] = reported
    audit_payload["audit_checks"] = str(audit_checks_path)
    audit_payload["metric_checks"] = str(metric_path)
    _atomic_write_json(audit_json_path, audit_payload)
    return {
        "passed": bool(audit.get("passed")) and all(row["Passed"] for row in metric_rows),
        "audit": audit_payload,
        "metrics": independent,
        "paths": {
            "audit": audit_checks_path,
            "report": audit_json_path,
            "metrics": metric_path,
        },
    }


def _validate_sequential_reference_only(tolerance: float) -> dict[str, object]:
    """Audit only the repaired Q2-task-seed to V4-energy reference."""

    return _validate_fixed_energy_reference_only(
        tolerance,
        input_prefix="q4_sequential_baseline",
        output_prefix="q4_validation_v4_sequential_repair",
        scheme="Q2_TASK_SEED_TO_V4_ENERGY_REFERENCE",
        baseline_status="Q2_TASK_SEED_TO_V4_ENERGY_REFERENCE_REPAIRED",
        scenario_status="SEQUENTIAL_REFERENCE_ONLY",
        scope="sequential_reference_only",
    )


def _validate_comparison_references_only(tolerance: float) -> dict[str, object]:
    """Audit both task schedules evaluated under the common V4 energy rule."""

    sequential = _validate_sequential_reference_only(tolerance)
    joint = _validate_fixed_energy_reference_only(
        tolerance,
        input_prefix="q4_joint_energy_reference",
        output_prefix="q4_validation_v4_joint_energy_reference",
        scheme="EXISTING_JOINT_TASK_SCHEDULE_TO_V4_ENERGY_REFERENCE",
        baseline_status="EXISTING_JOINT_TASK_SCHEDULE_TO_V4_ENERGY_REFERENCE",
        scenario_status="JOINT_TASK_SCHEDULE_ENERGY_REEVALUATION_ONLY",
        scope="joint_task_schedule_energy_reference_only",
    )
    rows: list[dict[str, object]] = []
    for label, result in (("SequentialTaskSchedule", sequential), ("JointTaskSchedule", joint)):
        for metric, value in result["metrics"].items():
            rows.append({"Scheme": label, "Metric": metric, "IndependentValue": value})
    summary_path = V4_VALIDATION_DIR / "q4_validation_v4_comparison_reference_metrics.csv"
    _write_csv(pd.DataFrame(rows), summary_path)
    return {
        "passed": bool(sequential["passed"] and joint["passed"]),
        "sequential": sequential,
        "joint": joint,
        "summary": summary_path,
    }


def _load_v4_context(
    data: Any,
    records: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    tolerance: float,
) -> dict[str, object] | None:
    required = (
        V4_TABLES_DIR / ".q4_matheuristic_v4_complete.json",
        V4_TABLES_DIR / "q4_task_assignments.csv",
        V4_TABLES_DIR / "q4_region_hour_dispatch.csv",
        V4_TABLES_DIR / "q4_objective_summary.csv",
        V4_TABLES_DIR / "q4_simple_validation.csv",
        V4_TABLES_DIR / "q4_window_solver.csv",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        _record(
            records,
            "全时域独立审计",
            "V4正式结果完整性",
            "FAIL",
            "; ".join(missing),
            "完成标记、任务表、能源表、指标表、简单复核表和窗口表均存在",
            "先完成Q4 V4正式运行",
        )
        return None

    marker = json.loads(required[0].read_text(encoding="utf-8"))
    marker_ok = (
        marker.get("complete") is True
        and marker.get("audit_passed") is True
        and marker.get("model_version") == MODEL_VERSION
    )
    _record(
        records,
        "全时域独立审计",
        "V4完成标记和版本",
        "PASS" if marker_ok else "FAIL",
        str(required[0]),
        "complete=true、audit_passed=true且model_version为正式V4",
        "拒绝把旧版本或未完成断点作为正式结果",
        value={key: marker.get(key) for key in ("complete", "audit_passed", "model_version")},
    )

    assignments = _read_csv(V4_TABLES_DIR / "q4_task_assignments.csv")
    dispatch = _read_csv(V4_TABLES_DIR / "q4_region_hour_dispatch.csv")
    summary = _read_csv(V4_TABLES_DIR / "q4_objective_summary.csv")
    simple = _read_csv(V4_TABLES_DIR / "q4_simple_validation.csv", ("Check", "Passed"))
    simple_passed = _parse_bool_series(simple["Passed"])
    simple_ok = not simple_passed.isna().any() and bool(simple_passed.all())
    _record(
        records,
        "全时域独立审计",
        "模型生成的简单复核表",
        "PASS" if simple_ok else "FAIL",
        str(V4_TABLES_DIR / "q4_simple_validation.csv"),
        "所有简单复核项的Passed均为True",
        "检查失败项后重新生成正式结果",
        value={"rows": int(len(simple)), "failed": int((~simple_passed.fillna(False)).sum())},
    )

    row_count_ok = len(assignments) == len(data.tasks) and len(dispatch) == q4_model.OPERATION_END * len(data.regions)
    _record(
        records,
        "全时域独立审计",
        "任务与逐时能源轨迹覆盖",
        "PASS" if row_count_ok else "FAIL",
        f"assignments={len(assignments)}, dispatch={len(dispatch)}",
        f"任务数={len(data.tasks)}且能源轨迹={q4_model.OPERATION_END}×{len(data.regions)}",
        "补齐缺失任务或小时区域记录",
        value={"task_rows": int(len(assignments)), "dispatch_rows": int(len(dispatch))},
    )

    try:
        document_audit = q4_model.audit_document_compliant_result(V4_ROOT)
        audit = dict(document_audit["audit"])
        clean_audit, audit_checks = _strip_checks(audit)
        if audit_checks is not None:
            _write_csv(audit_checks, VALIDATION_DIR / "q4_validation_v4_hard_constraint_audit.csv")
        _atomic_write_json(
            VALIDATION_DIR / "q4_validation_v4_independent_audit.json",
            clean_audit,
        )
        audit_ok = bool(document_audit.get("passed") and audit.get("passed"))
        _record(
            records,
            "全时域独立审计",
            "独立硬约束复算",
            "PASS" if audit_ok else "FAIL",
            str(VALIDATION_DIR / "q4_validation_v4_hard_constraint_audit.csv"),
            "所有独立硬约束检查通过且最大违差不超过容差",
            "查看failed_checks并修正结果或输入口径",
            value={
                "failed_checks": audit.get("failed_checks", []),
                "max_violation": audit.get("max_violation"),
            },
            tolerance=tolerance,
        )

        reported = _summary_metrics(summary)
        independent = {
            str(key): float(value)
            for key, value in dict(audit.get("independent_metrics", {})).items()
            if isinstance(value, (int, float, np.integer, np.floating))
        }
        rows = _metric_rows("Q4_V4", independent, reported, tolerance=tolerance)
        metric_rows.extend(rows)
        metric_ok = all(bool(row["Passed"]) for row in rows)
        _record(
            records,
            "全时域独立审计",
            "六项目标独立重算与汇总表一致性",
            "PASS" if metric_ok else "FAIL",
            str(V4_TABLES_DIR / "q4_objective_summary.csv"),
            "Cost、Carbon、Latency、Delay、RenewableUnusedRate、Peak均在容差内一致",
            "不要直接引用内部累计值，先核对明细表和指标口径",
            value={"max_absolute_difference": max((float(row["AbsoluteDifference"]) for row in rows if pd.notna(row["AbsoluteDifference"])), default=float("nan"))},
            tolerance=tolerance,
        )
    except Exception as exc:
        logging.exception("V4独立审计异常")
        _record(
            records,
            "全时域独立审计",
            "独立审计执行",
            "FAIL",
            repr(exc),
            "验证函数能够从V4明细重新计算审计结果",
            "检查模型输入、输出字段和运行环境",
        )
        return None

    return {
        "marker": marker,
        "assignments": assignments,
        "dispatch": dispatch,
        "summary": summary,
        "audit": audit,
        "reported_metrics": reported,
        "independent_metrics": independent,
        "tables": V4_TABLES_DIR,
    }


def _validate_sequential_baseline(
    data: Any,
    context: dict[str, object],
    records: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    tolerance: float,
) -> dict[str, object] | None:
    paths = {
        "marker": V4_TABLES_DIR / ".q4_sequential_baseline_complete.json",
        "assignments": V4_TABLES_DIR / "q4_sequential_baseline_assignments.csv",
        "dispatch": V4_TABLES_DIR / "q4_sequential_baseline_dispatch.csv",
        "metrics": V4_TABLES_DIR / "q4_sequential_baseline_metrics.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        _record(
            records,
            "顺序基准与协同检验",
            "Q2任务种子→V4能源顺序基准",
            "WARN",
            "; ".join(missing),
            "顺序基准完整标记、任务表、能源表和指标表均存在",
            "生成同口径顺序基准后重新运行检验",
        )
        return None

    marker = json.loads(paths["marker"].read_text(encoding="utf-8"))
    assignments = _read_csv(paths["assignments"])
    dispatch = _read_csv(paths["dispatch"])
    metrics_frame = _read_csv(paths["metrics"], ("Metric", "Value"))
    marker_ok = (
        marker.get("complete") is True
        and int(marker.get("task_count", -1)) == len(data.tasks)
        and int(marker.get("dispatch_count", -1)) == q4_model.OPERATION_END * len(data.regions)
    )
    try:
        audit = q4_model.independent_hard_constraint_audit(
            data,
            assignments,
            dispatch,
            baseline_status="Q2_TASK_SEED_TO_V4_ENERGY_REFERENCE",
            scenario_status="SEQUENTIAL_BASELINE",
            tolerance=tolerance,
            qos_weights=q4_model._qos_weight_map(q4_model.ModelConfig()),
        )
        clean_audit, audit_checks = _strip_checks(dict(audit))
        if audit_checks is not None:
            _write_csv(
                audit_checks,
                VALIDATION_DIR / "q4_validation_sequential_hard_constraint_audit.csv",
            )
        _atomic_write_json(
            VALIDATION_DIR / "q4_validation_sequential_independent_audit.json",
            clean_audit,
        )
    except Exception as exc:
        logging.exception("顺序基准独立审计异常")
        _record(
            records,
            "顺序基准与协同检验",
            "顺序基准独立审计",
            "FAIL",
            repr(exc),
            "顺序基准也必须满足同一套任务和能源硬约束",
            "修正顺序基准结果后重算",
        )
        return None

    reported = _summary_metrics(metrics_frame)
    independent = {
        str(key): float(value)
        for key, value in dict(audit.get("independent_metrics", {})).items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }
    rows = _metric_rows("SequentialBaseline", independent, reported, tolerance=tolerance)
    metric_rows.extend(rows)
    metric_ok = all(bool(row["Passed"]) for row in rows)
    audit_ok = bool(audit.get("passed"))
    all_ok = marker_ok and audit_ok and metric_ok
    _record(
        records,
        "顺序基准与协同检验",
        "顺序基准完成标记",
        "PASS" if marker_ok else "FAIL",
        str(paths["marker"]),
        "complete=true且任务/能源记录数与当前输入一致",
        "拒绝混用其他版本顺序基准",
        value=marker,
    )
    _record(
        records,
        "顺序基准与协同检验",
        "顺序基准硬约束和指标复算",
        "PASS" if all_ok else "FAIL",
        str(VALIDATION_DIR / "q4_validation_sequential_independent_audit.json"),
        "顺序基准审计通过且指标文件与独立重算一致",
        "检查顺序基准任务表、能源轨迹和指标文件",
        value={"audit_passed": audit_ok, "metric_passed": metric_ok},
        tolerance=tolerance,
    )
    return {
        "marker": marker,
        "audit": audit,
        "reported_metrics": reported,
        "independent_metrics": independent,
    }


def _select_reference_windows(solver: pd.DataFrame) -> pd.DataFrame:
    required = {"WindowStart", "TotalWindowSeconds"}
    if not required.issubset(solver.columns):
        raise ValueError(f"q4_window_solver.csv缺少代表窗口选择字段：{sorted(required - set(solver.columns))}")
    frame = solver.copy()
    frame["WindowStart"] = pd.to_numeric(frame["WindowStart"], errors="coerce")
    frame["TotalWindowSeconds"] = pd.to_numeric(frame["TotalWindowSeconds"], errors="coerce")
    frame = frame.dropna(subset=["WindowStart", "TotalWindowSeconds"]).drop_duplicates("WindowStart")
    if frame.empty:
        raise ValueError("q4_window_solver.csv没有可用窗口")
    median_time = float(frame["TotalWindowSeconds"].median())
    median_row = frame.iloc[(frame["TotalWindowSeconds"] - median_time).abs().argsort().iloc[0]]
    longest_row = frame.sort_values(["TotalWindowSeconds", "WindowStart"], ascending=[False, True]).iloc[0]
    selected = pd.DataFrame(
        [
            {
                "Role": "MEDIAN_TIME",
                "WindowStart": int(median_row["WindowStart"]),
                "FormalWindowSeconds": float(median_row["TotalWindowSeconds"]),
                "SelectionCriterion": "closest to formal window-time median",
            },
            {
                "Role": "LONGEST_TIME",
                "WindowStart": int(longest_row["WindowStart"]),
                "FormalWindowSeconds": float(longest_row["TotalWindowSeconds"]),
                "SelectionCriterion": "maximum formal window time",
            },
        ]
    )
    return selected.drop_duplicates("WindowStart", keep="first").reset_index(drop=True)


def _run_reference_windows(
    data: Any,
    context: dict[str, object],
    selection: pd.DataFrame,
    *,
    time_limit: float,
    mip_gap: float,
) -> pd.DataFrame:
    baseline = {
        "q4_task_assignments.csv": context["assignments"],
        "q4_region_hour_dispatch.csv": context["dispatch"],
        "q4_objective_summary.csv": context["summary"],
        "q4_scaling.csv": _read_csv(V4_TABLES_DIR / "q4_scaling.csv"),
    }
    config = q4_model.ModelConfig(
        mip_relative_gap=mip_gap,
        normal_time_limit_seconds=time_limit,
        difficult_time_limit_seconds=time_limit,
        window_hard_time_limit_seconds=time_limit,
    )
    rows: list[dict[str, object]] = []
    logging.info(
        "代表窗口完整MILP开始：窗口数=%d，单窗口上限=%.1fs，目标MIP gap=%.4f。",
        len(selection),
        time_limit,
        mip_gap,
    )
    for index, selected in enumerate(selection.itertuples(index=False), start=1):
        tau = int(selected.WindowStart)
        role = str(selected.Role)
        logging.info(
            "代表窗口完整MILP：%d/%d开始，role=%s，WindowStart=%d。",
            index,
            len(selection),
            role,
            tau,
        )
        try:
            summary, assignments, dispatch = q4_model._run_validation_window(
                data,
                baseline,
                tau,
                config,
                scenario_name=f"REFERENCE_{role}",
                time_limit=time_limit,
            )
            result = dict(summary)
            result.update(
                {
                    "ReferenceRole": role,
                    "FormalWindowSeconds": float(selected.FormalWindowSeconds),
                    "ReferenceTimeLimitSeconds": time_limit,
                    "ReferenceMIPGapTarget": mip_gap,
                    "ReferenceStatus": (
                        "PASS"
                        if result.get("FinalConfirmationModel") == "ORIGINAL_Q4_FULL_HK_MILP"
                        and bool(result.get("IndependentAuditPassed"))
                        else "WARN"
                    ),
                }
            )
            logging.info(
                "代表窗口完整MILP：WindowStart=%d完成，ReferenceStatus=%s，"
                "FinalConfirmationModel=%s，IndependentAuditPassed=%s，耗时=%s。",
                tau,
                result.get("ReferenceStatus"),
                result.get("FinalConfirmationModel"),
                result.get("IndependentAuditPassed"),
                result.get("ElapsedSeconds"),
            )
            _write_csv(
                assignments,
                VALIDATION_DIR / f"q4_reference_window_{tau}_assignments.csv",
            )
            _write_csv(
                dispatch,
                VALIDATION_DIR / f"q4_reference_window_{tau}_dispatch.csv",
            )
        except Exception as exc:
            result = {
                "ReferenceRole": role,
                "WindowStart": tau,
                "FormalWindowSeconds": float(selected.FormalWindowSeconds),
                "ReferenceTimeLimitSeconds": time_limit,
                "ReferenceMIPGapTarget": mip_gap,
                "ReferenceStatus": "FAIL",
                "FinalConfirmationModel": "EXCEPTION",
                "Error": repr(exc),
            }
            logging.exception(
                "代表窗口完整MILP失败：WindowStart=%d，role=%s。",
                tau,
                role,
            )
        rows.append(result)
    result_frame = pd.DataFrame(rows)
    _write_csv(result_frame, VALIDATION_DIR / "q4_reference_window_comparison.csv")
    logging.info(
        "代表窗口完整MILP汇总已写出：%s。",
        VALIDATION_DIR / "q4_reference_window_comparison.csv",
    )
    return result_frame


def _validate_reference_windows(
    data: Any,
    context: dict[str, object],
    records: list[dict[str, object]],
    *,
    run_reference: bool,
    time_limit: float,
    mip_gap: float,
) -> dict[str, object]:
    solver = _read_csv(V4_TABLES_DIR / "q4_window_solver.csv")
    selection = _select_reference_windows(solver)
    _write_csv(selection, VALIDATION_DIR / "q4_reference_window_selection.csv")
    _record(
        records,
        "代表窗口完整MILP对照",
        "代表窗口选择",
        "PASS",
        str(VALIDATION_DIR / "q4_reference_window_selection.csv"),
        "选择正式窗口耗时接近中位数和耗时最长的两个窗口",
        "检查正式窗口日志是否完整",
        value=selection.to_dict(orient="records"),
    )
    logging.info(
        "代表窗口已选择：%s。",
        selection[["Role", "WindowStart", "FormalWindowSeconds"]].to_dict(orient="records"),
    )

    comparison: pd.DataFrame | None = None
    if run_reference:
        logging.info("正式检验：开始运行两个代表窗口的完整联合MILP。")
        comparison = _run_reference_windows(
            data,
            context,
            selection,
            time_limit=time_limit,
            mip_gap=mip_gap,
        )
        completed = (
            comparison["ReferenceStatus"].eq("PASS").all()
            if "ReferenceStatus" in comparison.columns else False
        )
        _record(
            records,
            "代表窗口完整MILP对照",
            "两个完整联合MILP窗口",
            "PASS" if completed else "FAIL",
            str(VALIDATION_DIR / "q4_reference_window_comparison.csv"),
            "两个窗口均实际调用原始Q4完整H+K MILP且窗口审计通过",
            "检查求解状态、可行解、下界和窗口审计",
            value=comparison.to_dict(orient="records"),
            tolerance=mip_gap,
        )
        return {
            "selection": selection,
            "comparison": comparison,
            "status": "PASS" if completed else "FAIL",
        }

    logging.info("已跳过代表窗口完整MILP，仅检查已有验证文件。")
    existing = VALIDATION_DIR / "q4_reference_window_comparison.csv"
    legacy = VALIDATION_DIR / "q4_validation_window_summary.csv"
    evidence_path = existing if existing.is_file() else legacy if legacy.is_file() else None
    if evidence_path is None:
        status = "WARN"
        detail = "尚未找到完整联合MILP窗口结果；默认模式不启动300秒求解。"
    else:
        comparison = _read_csv(evidence_path)
        if "FinalConfirmationModel" in comparison.columns:
            full_milp = comparison["FinalConfirmationModel"].astype(str).eq("ORIGINAL_Q4_FULL_HK_MILP")
            audit_ok = (
                _parse_bool_series(comparison["IndependentAuditPassed"]).fillna(False)
                if "IndependentAuditPassed" in comparison.columns
                else pd.Series(False, index=comparison.index)
            )
            status = "PASS" if len(comparison) >= 2 and bool((full_milp & audit_ok).all()) else "WARN"
            detail = "已有窗口文件，但需确认每个窗口均为完整MILP并通过窗口审计。"
        else:
            status = "WARN"
            detail = "已有验证性窗口文件，但没有完整MILP确认字段。"
    _record(
        records,
        "代表窗口完整MILP对照",
        "完整联合MILP结果",
        status,
        str(evidence_path) if evidence_path else "未找到结果文件",
        "两个代表窗口均有ORIGINAL_Q4_FULL_HK_MILP结果且IndependentAuditPassed为True",
        "使用--run-reference-windows实际运行两个300秒窗口",
        value=detail,
    )
    return {
        "selection": selection,
        "comparison": comparison,
        "status": status,
        "evidence": str(evidence_path) if evidence_path else None,
    }


def _scenario_formal_root(directory: Path) -> Path:
    return directory / "matheuristic_v4_doccompliant"


def _reconstruct_scenario_input(
    data: Any,
    baseline_context: dict[str, object],
    scenario: dict[str, object],
    directory: Path,
) -> tuple[Any, dict[str, object]]:
    """Rebuild the external inputs used by one completed V4 scenario.

    The formal scenario output is generated from the base input plus the
    scenario-specific exogenous transformation.  Independent auditing must
    use that transformed input; auditing every scenario against ``load_data``
    would incorrectly reject flat-price and renewable-smoothing outputs.
    """

    kind = str(scenario.get("ScenarioKind", "")).strip().lower()
    allowed_kinds = {
        "low_carbon_reference",
        "carbon_constraint",
        "flat_price",
        "low_variability_renewable",
    }
    if kind not in allowed_kinds:
        raise ValueError(f"情景{directory.name}的ScenarioKind无效：{kind!r}")

    name = str(scenario.get("ScenarioName") or directory.name)
    baseline_metrics = {
        str(key): float(value)
        for key, value in dict(baseline_context.get("independent_metrics", {})).items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }
    if "Carbon" not in baseline_metrics:
        raise ValueError(f"情景{directory.name}缺少可用于重建碳约束的基准Carbon指标")

    kwargs: dict[str, object] = {"name": name, "kind": kind}
    if kind == "carbon_constraint":
        for key in ("CarbonLambda", "LowCarbonReference"):
            value = scenario.get(key)
            if value is None:
                raise ValueError(f"情景{directory.name}缺少{key}元数据")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"情景{directory.name}的{key}不是有限数值：{value!r}")
            kwargs[
                "carbon_lambda" if key == "CarbonLambda" else "low_carbon_reference"
            ] = number
    elif kind == "low_variability_renewable":
        value = scenario.get("RenewableSmoothingGamma")
        if value is None:
            raise ValueError(f"情景{directory.name}缺少RenewableSmoothingGamma元数据")
        gamma = float(value)
        if not math.isfinite(gamma):
            raise ValueError(f"情景{directory.name}的RenewableSmoothingGamma不是有限数值：{value!r}")
        kwargs["renewable_gamma"] = gamma

    spec = q4_model.ScenarioSpec(**kwargs)
    scenario_data, generated_metadata = q4_model.generate_scenario_data(
        data, spec, baseline_metrics
    )
    factor_check = q4_model.validate_single_factor_scenario(
        data, scenario_data, generated_metadata
    )
    if not factor_check.get("passed"):
        raise ValueError(
            f"情景{directory.name}重建后的单因素输入检查失败：{factor_check}"
        )

    expected_changed = sorted(str(item) for item in scenario.get("ChangedColumns", []))
    actual_changed = sorted(str(item) for item in generated_metadata.get("ChangedColumns", []))
    if expected_changed != actual_changed:
        raise ValueError(
            f"情景{directory.name}ChangedColumns不一致："
            f"marker={expected_changed}，重建={actual_changed}"
        )
    return scenario_data, {
        "metadata": generated_metadata,
        "factor_check": factor_check,
        "changed_columns": actual_changed,
    }


def _validate_scenarios(
    data: Any,
    baseline_context: dict[str, object],
    records: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    tolerance: float,
) -> dict[str, object]:
    required_groups = {
        "flat_price": {"flat_price"},
        "low_carbon": {"low_carbon_reference", "carbon_constraint"},
        "renewable_variability": {"low_variability_renewable"},
    }
    rows: list[dict[str, object]] = []
    formal_kinds: set[str] = set()
    legacy_names: list[str] = []
    failed_formal = False
    if SCENARIO_DIR.is_dir():
        directories = sorted(path for path in SCENARIO_DIR.iterdir() if path.is_dir())
    else:
        directories = []
    for directory in directories:
        formal_root = _scenario_formal_root(directory)
        tables = formal_root / "tables"
        marker_path = tables / ".q4_matheuristic_v4_complete.json"
        if not marker_path.is_file():
            if (directory / "matheuristic_v2").is_dir() or (directory / "tables").is_dir():
                legacy_names.append(directory.name)
            continue
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            scenario = dict(marker.get("scenario", {}))
            kind = str(scenario.get("ScenarioKind", ""))
            scenario_data, scenario_input = _reconstruct_scenario_input(
                data, baseline_context, scenario, directory
            )
            logging.info(
                "情景独立审计：%s，使用情景输入，kind=%s，改变列=%s。",
                directory.name,
                kind,
                scenario_input["changed_columns"],
            )
            audit_report = q4_model.audit_document_compliant_result(
                formal_root,
                data=scenario_data,
            )
            audit = dict(audit_report["audit"])
            clean_audit, audit_checks = _strip_checks(audit)
            safe_name = directory.name.replace("/", "_").replace("\\", "_")
            if audit_checks is not None:
                _write_csv(
                    audit_checks,
                    VALIDATION_DIR / f"q4_validation_scenario_{safe_name}_audit.csv",
                )
            scenario_status = bool(audit_report.get("passed") and marker.get("complete") is True and marker.get("audit_passed") is True)
            if not scenario_status:
                failed_formal = True
            formal_kinds.add(kind)
            summary = _read_csv(tables / "q4_objective_summary.csv")
            reported = _summary_metrics(summary)
            independent = {
                str(key): float(value)
                for key, value in dict(audit.get("independent_metrics", {})).items()
                if isinstance(value, (int, float, np.integer, np.floating))
            }
            metric_rows.extend(_metric_rows(f"Scenario:{directory.name}", independent, reported, tolerance=tolerance))
            rows.append(
                {
                    "ScenarioName": directory.name,
                    "ScenarioKind": kind,
                    "ModelVersion": marker.get("model_version"),
                    "Complete": marker.get("complete"),
                    "AuditPassed": marker.get("audit_passed"),
                    "IndependentAuditPassed": bool(audit_report.get("passed")),
                    "AuditDataContext": audit_report.get("audit_data_context"),
                    "Carbon": independent.get("Carbon"),
                    "Cost": independent.get("Cost"),
                    "Status": "PASS" if scenario_status else "FAIL",
                }
            )
        except Exception as exc:
            failed_formal = True
            rows.append(
                {
                    "ScenarioName": directory.name,
                    "ScenarioKind": "",
                    "ModelVersion": "",
                    "Complete": False,
                    "AuditPassed": False,
                    "IndependentAuditPassed": False,
                    "Status": "FAIL",
                    "Error": repr(exc),
                }
            )
    scenario_frame = pd.DataFrame(rows)
    _write_csv(scenario_frame, VALIDATION_DIR / "q4_validation_scenario_summary.csv")

    missing_groups = [
        group for group, kinds in required_groups.items()
        if not formal_kinds.intersection(kinds)
    ]
    if failed_formal:
        status = "FAIL"
    elif missing_groups:
        status = "WARN"
    else:
        status = "PASS"
    _record(
        records,
        "顺序基准与题设情景一致性",
        "同版本Q4情景结果",
        status,
        str(VALIDATION_DIR / "q4_validation_scenario_summary.csv"),
        "平价、低碳约束和新能源波动情景均使用V4同版本完成标记并通过独立审计",
        "不要混用matheuristic_v2；补齐缺失情景后重跑检验",
        value={
            "formal_scenario_kinds": sorted(formal_kinds),
            "missing_groups": missing_groups,
            "legacy_scenario_names": legacy_names,
        },
    )
    return {
        "rows": scenario_frame,
        "status": status,
        "formal_kinds": sorted(formal_kinds),
        "missing_groups": missing_groups,
        "legacy_names": legacy_names,
    }


def _overall_status(records: list[dict[str, object]]) -> tuple[str, dict[str, int]]:
    counts = {
        status: sum(1 for row in records if row.get("Status") == status)
        for status in ("PASS", "WARN", "FAIL")
    }
    if counts["FAIL"]:
        return "FAIL", counts
    if counts["WARN"]:
        return "WARN", counts
    return "PASS", counts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="问题4独立模型检验")
    reference_group = parser.add_mutually_exclusive_group()
    reference_group.add_argument(
        "--run-reference-windows",
        dest="run_reference_windows",
        action="store_true",
        default=True,
        help="实际运行两个代表窗口的完整联合MILP（默认开启）；每个窗口默认300秒",
    )
    reference_group.add_argument(
        "--skip-reference-windows",
        dest="run_reference_windows",
        action="store_false",
        help="跳过两个完整MILP窗口，仅做已有文件和结果审计",
    )
    parser.add_argument(
        "--sequential-only",
        action="store_true",
        help="只审计已经修复的Q2任务种子到V4能源顺序基准，不检查Q4联合基准和情景",
    )
    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help="只独立核验按同一V4能源口径复算的顺序任务方案和既有联合任务方案",
    )
    parser.add_argument(
        "--reference-time-limit",
        type=float,
        default=REFERENCE_TIME_LIMIT_SECONDS,
        help="代表窗口完整MILP单窗口时间上限（秒）",
    )
    parser.add_argument(
        "--reference-mip-gap",
        type=float,
        default=REFERENCE_MIP_GAP,
        help="代表窗口完整MILP相对间隙",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="硬约束和指标重算的绝对容差",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.tolerance <= 0.0:
        raise ValueError("--tolerance必须为正数")
    if args.reference_time_limit <= 0.0:
        raise ValueError("--reference-time-limit必须为正数")
    if not 0.0 <= args.reference_mip_gap <= 1.0:
        raise ValueError("--reference-mip-gap必须位于[0,1]")
    _configure_logging(args.log_level)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    report: dict[str, object] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "question": "Q4",
        "model_version": MODEL_VERSION,
        "tolerance": args.tolerance,
        "reference_windows_requested": bool(args.run_reference_windows),
        "reference_time_limit_seconds": args.reference_time_limit,
    }

    running_processes, process_check_error = _running_q4_model_processes()
    if process_check_error:
        _record(
            records,
            "运行状态",
            "检查Q4模型进程",
            "FAIL",
            process_check_error,
            "无法确认Q4是否仍在运行时，禁止读取历史完成标记",
            "修复进程检查环境后重新运行",
        )
        report["blocked"] = True
        report["block_reason"] = process_check_error
    elif running_processes:
        _record(
            records,
            "运行状态",
            "Q4模型尚未完成",
            "WARN",
            "检测到仍在运行的model.py进程",
            "只有不存在Q4模型进程时，才允许读取完成标记和正式结果",
            "等待Q4模型/情景运行结束后重新运行validation.py",
            value=running_processes,
        )
        report["blocked"] = True
        report["block_reason"] = "Q4模型进程仍在运行，未读取任何历史正式结果"
    elif q4_model is None:
        _record(
            records,
            "运行环境",
            "导入Q4模型模块",
            "FAIL",
            repr(MODEL_IMPORT_ERROR),
            "validation.py可以导入同目录model.py及其既有依赖",
            "检查项目虚拟环境和依赖安装",
        )
    else:
        if args.comparison_only:
            try:
                result = _validate_comparison_references_only(args.tolerance)
                logging.info(
                    "V4统一口径对比定点检验完成：overall=%s，summary=%s。",
                    "PASS" if result["passed"] else "FAIL",
                    result["summary"],
                )
                return 0 if result["passed"] else 2
            except Exception as exc:
                logging.exception("V4统一口径对比定点检验异常")
                logging.error("V4统一口径对比定点检验失败：%s", exc)
                return 2
        if args.sequential_only:
            try:
                result = _validate_sequential_reference_only(args.tolerance)
                logging.info(
                    "顺序基准定点检验完成：overall=%s，metrics=%s。",
                    "PASS" if result["passed"] else "FAIL",
                    result["metrics"],
                )
                return 0 if result["passed"] else 2
            except Exception as exc:
                logging.exception("顺序基准定点检验异常")
                logging.error("顺序基准定点检验失败：%s", exc)
                return 2
        try:
            data = q4_model.load_data()
            context = _load_v4_context(data, records, metric_rows, args.tolerance)
            if context is not None:
                sequential = _validate_sequential_baseline(
                    data, context, records, metric_rows, args.tolerance
                )
                reference = _validate_reference_windows(
                    data,
                    context,
                    records,
                    run_reference=args.run_reference_windows,
                    time_limit=args.reference_time_limit,
                    mip_gap=args.reference_mip_gap,
                )
                scenarios = _validate_scenarios(
                    data, context, records, metric_rows, args.tolerance
                )
                report.update(
                    {
                        "v4": {
                            "tables": str(context["tables"]),
                            "independent_metrics": context["independent_metrics"],
                        },
                        "sequential": {
                            "available": sequential is not None,
                            "independent_metrics": sequential.get("independent_metrics", {}) if sequential else {},
                        },
                        "reference_windows": {
                            "status": reference.get("status"),
                            "selection": reference["selection"].to_dict(orient="records"),
                            "evidence": reference.get("evidence"),
                        },
                        "scenarios": {
                            "status": scenarios["status"],
                            "formal_kinds": scenarios["formal_kinds"],
                            "missing_groups": scenarios["missing_groups"],
                            "legacy_names": scenarios["legacy_names"],
                        },
                    }
                )
        except Exception as exc:
            logging.exception("Q4检验执行异常")
            _record(
                records,
                "运行环境",
                "检验主流程",
                "FAIL",
                repr(exc),
                "检验流程应在异常输入下明确失败而不是伪造通过",
                "根据错误信息修正输入或代码",
            )

    overall, counts = _overall_status(records)
    summary = pd.DataFrame(records)
    _write_csv(summary, VALIDATION_DIR / "q4_validation_summary.csv")
    _write_csv(pd.DataFrame(metric_rows), VALIDATION_DIR / "q4_validation_metric_comparison.csv")
    report.update(
        {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "overall": overall,
            "counts": counts,
            "checks": records,
            "metric_comparison": metric_rows,
        }
    )
    _atomic_write_json(report_path := VALIDATION_DIR / "q4_validation_report.json", report)
    logging.info(
        "Q4独立检验完成：overall=%s，PASS=%d，WARN=%d，FAIL=%d。",
        overall,
        counts["PASS"],
        counts["WARN"],
        counts["FAIL"],
    )
    logging.info("检验汇总：%s", VALIDATION_DIR / "q4_validation_summary.csv")
    logging.info("检验报告：%s", report_path)
    return 0 if overall == "PASS" else 1 if overall == "WARN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
