"""单独重算Q2精确同窗异常窗口的启发式结果。

默认目标为WindowID=9（第10个窗口，tau=216）。本入口读取已完成的精确
同窗对照结果，只固定窗口0--8的精确历史安排，然后调用“边际贪心+LNS
精修”启发式重算目标窗口；不会调用精确MILP，也不会覆盖正式Q2结果或
原有exact_vs_heuristic对照文件。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd

import exact_heuristic_benchmark as benchmark
import model as q2


QUESTION_DIR = Path(__file__).resolve().parent
VALIDATION_ROOT = QUESTION_DIR / "outputs" / "validation_runs"
DEFAULT_HISTORY_ROOT = VALIDATION_ROOT / "exact_vs_heuristic"
DEFAULT_OUTPUT_ROOT = VALIDATION_ROOT / "heuristic_window9"
DECISION_WINDOW = 24
LOOKAHEAD = 48
DEFAULT_WINDOW_ID = 9


def _configure_logging(output_root: Path, level_name: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"不支持的日志级别：{level_name}")
    logging.basicConfig(
        level=level,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                output_root / "q2_heuristic_window9.log",
                encoding="utf-8",
            ),
        ],
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _load_exact_history(
    history_root: Path,
    *,
    window_id: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    windows_path = history_root / "exact_windows.csv"
    assignments_path = history_root / "exact_assignments.csv"
    if not windows_path.is_file():
        raise FileNotFoundError(f"缺少精确窗口结果：{windows_path}")
    if not assignments_path.is_file():
        raise FileNotFoundError(f"缺少精确历史任务安排：{assignments_path}")

    windows = pd.read_csv(windows_path, encoding="utf-8-sig")
    assignments = pd.read_csv(assignments_path, encoding="utf-8-sig")
    required_windows = {
        "WindowID",
        "CommittedTaskCount",
        "MIPStatus",
        "ZStar",
        "ElapsedSeconds",
    }
    required_assignments = {
        "TaskID",
        "ExactTaskClassID",
        "StartHour",
    }
    missing_windows = sorted(required_windows.difference(windows.columns))
    missing_assignments = sorted(required_assignments.difference(assignments.columns))
    if missing_windows:
        raise RuntimeError(f"exact_windows.csv缺少字段：{missing_windows}")
    if missing_assignments:
        raise RuntimeError(f"exact_assignments.csv缺少字段：{missing_assignments}")

    windows = windows.copy()
    windows["WindowID"] = pd.to_numeric(windows["WindowID"], errors="coerce")
    windows["CommittedTaskCount"] = pd.to_numeric(
        windows["CommittedTaskCount"], errors="coerce"
    )
    windows["MIPStatus"] = pd.to_numeric(windows["MIPStatus"], errors="coerce")
    target_rows = windows.loc[windows["WindowID"].eq(window_id)].copy()
    if len(target_rows) != 1:
        raise RuntimeError(
            f"exact_windows.csv中WindowID={window_id}应恰有1行，实际{len(target_rows)}行"
        )
    previous = windows.loc[windows["WindowID"].between(0, window_id - 1)].copy()
    expected_previous = set(range(window_id))
    actual_previous = set(previous["WindowID"].dropna().astype(int))
    if actual_previous != expected_previous:
        raise RuntimeError(
            f"窗口{window_id}需要完整窗口0--{window_id - 1}历史，"
            f"实际={sorted(actual_previous)}"
        )
    if previous["MIPStatus"].ne(0).any() or target_rows["MIPStatus"].ne(0).any():
        raise RuntimeError("窗口0--目标窗口必须全部为status=0的精确MILP结果")
    if previous["CommittedTaskCount"].isna().any():
        raise RuntimeError("窗口0--目标窗口存在无效CommittedTaskCount")

    tau = window_id * DECISION_WINDOW
    assignments = assignments.copy()
    assignments["StartHour"] = pd.to_numeric(
        assignments["StartHour"], errors="coerce"
    )
    if assignments["StartHour"].isna().any():
        raise RuntimeError("exact_assignments.csv存在无效StartHour")
    history = assignments.loc[
        assignments["StartHour"] < float(tau) - q2.FLOAT_EPS
    ].copy()
    expected_count = int(round(float(previous["CommittedTaskCount"].sum())))
    if len(history) != expected_count:
        raise RuntimeError(
            f"窗口0--{window_id - 1}历史任务数不一致："
            f"按窗口记录应为{expected_count}，按StartHour<{tau}筛得{len(history)}"
        )
    if history["TaskID"].astype(str).duplicated().any():
        raise RuntimeError("窗口0--目标窗口历史安排存在重复TaskID")
    if not history.empty and float(history["StartHour"].max()) >= tau:
        raise RuntimeError("历史安排包含目标窗口及之后的StartHour")
    history["TaskID"] = history["TaskID"].astype(str)
    return history.reset_index(drop=True), target_rows.iloc[0], previous


def _write_result(
    output_root: Path,
    *,
    result: dict[str, Any],
    history: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    window_id = int(result["WindowID"])
    pd.DataFrame([result]).to_csv(
        output_root / f"heuristic_window{window_id}.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.15g",
    )
    history.to_csv(
        output_root / f"exact_history_before_window{window_id}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    history_root = args.history_root.resolve()
    _configure_logging(output_root, args.log_level)
    started = time.perf_counter()
    window_id = int(args.window_id)
    tau = window_id * DECISION_WINDOW
    decision_end = tau + DECISION_WINDOW
    plan_end = decision_end + LOOKAHEAD

    logging.info(
        "窗口%d启发式单独重算启动：tau=%d，决策区间=%d--%d，前瞻结束=%d；"
        "只运行边际贪心+LNS，不运行精确MILP。",
        window_id,
        tau,
        tau,
        decision_end,
        plan_end,
    )
    bundle = q2._load_inputs()
    classes, _ = q2._build_exact_task_classes(bundle)
    class_lookup = {task_class.class_id: task_class for task_class in classes}
    baseline_assignments, baseline_profile, baseline_metrics = benchmark._load_baseline(
        bundle
    )
    calibration = benchmark._read_calibration(bundle, baseline_metrics)
    baseline_latency = q2._baseline_latency_lookup(baseline_assignments)
    history, exact_row, previous = _load_exact_history(
        history_root,
        window_id=window_id,
    )
    pools = q2._make_task_pools(classes)
    q2._remove_committed_from_pools(pools, history)
    active_task_ids, active_class_count = benchmark._active_context(
        classes,
        pools,
        tau,
        decision_end,
        plan_end,
    )
    if not active_task_ids:
        raise RuntimeError(f"窗口{window_id}没有可供启发式重算的活动任务")

    logging.info(
        "历史状态锁定：窗口0--%d精确任务=%d，活动任务=%d，活动类=%d。",
        window_id - 1,
        len(history),
        len(active_task_ids),
        active_class_count,
    )
    heuristic_row = benchmark._heuristic_record(
        bundle=bundle,
        classes=classes,
        class_lookup=class_lookup,
        pools=pools,
        committed_assignments=history,
        baseline_assignments=baseline_assignments,
        baseline_profile=baseline_profile,
        baseline_latency=baseline_latency,
        calibration=calibration,
        tau=tau,
        decision_end=decision_end,
        plan_end=plan_end,
        active_class_count=active_class_count,
        active_task_count=len(active_task_ids),
        progress_interval=float(args.progress_interval),
    )

    z_exact = float(exact_row["ZStar"])
    z_heuristic = float(heuristic_row["ZStar"])
    relative_delta = (z_heuristic - z_exact) / max(abs(z_exact), 1e-8)
    result: dict[str, Any] = {
        **heuristic_row,
        "ReferenceExactZ": z_exact,
        "ReferenceExactElapsedSeconds": float(exact_row["ElapsedSeconds"]),
        "RelativeDeltaToExact": relative_delta,
        "HistoryWindowCount": window_id,
        "HistoryTaskCount": len(history),
    }
    metadata = {
        "status": "COMPLETED",
        "window_id": window_id,
        "tau": tau,
        "decision_window": DECISION_WINDOW,
        "lookahead": LOOKAHEAD,
        "method": "greedy_plus_lns_only",
        "exact_milp_called": False,
        "history_root": str(history_root),
        "history_windows": list(range(window_id)),
        "history_task_count": len(history),
        "task_signature": q2._task_signature(bundle),
        "reference_exact_z": z_exact,
        "reference_heuristic_z": z_heuristic,
        "relative_delta": relative_delta,
        "elapsed_seconds": time.perf_counter() - started,
        "source_exact_row": {
            "WindowID": int(exact_row["WindowID"]),
            "ZStar": z_exact,
            "ElapsedSeconds": float(exact_row["ElapsedSeconds"]),
        },
    }
    _write_result(
        output_root,
        result=result,
        history=history,
        metadata=metadata,
    )
    logging.info(
        "窗口%d启发式单独重算完成：z_exact=%.9g，z_heuristic=%.9g，"
        "相对偏差=%.4f%%，启发式耗时=%.2fs，输出=%s。",
        window_id,
        z_exact,
        z_heuristic,
        relative_delta * 100.0,
        float(heuristic_row["ElapsedSeconds"]),
        output_root,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="只重算Q2精确同窗异常窗口的启发式+LNS结果"
    )
    parser.add_argument("--window-id", type=int, default=DEFAULT_WINDOW_ID)
    parser.add_argument(
        "--history-root",
        type=Path,
        default=DEFAULT_HISTORY_ROOT,
        help="已完成精确同窗结果目录",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="单独输出目录，不覆盖正式对照结果",
    )
    parser.add_argument("--progress-interval", type=float, default=30.0)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args()
    if not 0 <= args.window_id <= 9:
        raise SystemExit("--window-id当前必须在0--9之间")
    if not math.isfinite(args.progress_interval) or args.progress_interval < 0:
        raise SystemExit("--progress-interval必须为非负有限数")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
