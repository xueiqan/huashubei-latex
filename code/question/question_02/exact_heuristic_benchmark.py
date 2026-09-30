"""Question 2 exact same-window comparison: exact MILP versus heuristic plus LNS.

This validation-only script never calls model.py main or writes official Q2
tables/checkpoints. Each window first builds a heuristic candidate against the
current exact history, then solves the exact min-max model against the same
history. Only the exact solution is committed to the next window. Both methods
therefore share H=24, K=48, task pools, historical loads, and global calibration
parameters in windows 0--9.

If an exact window does not return status=0, do not label a time-limited solution
as exact comparison evidence. Preserve completed windows for resumption with
a larger solver time limit.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import model as q2


QUESTION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = QUESTION_DIR.parents[1]
FORMAL_TABLES_DIR = QUESTION_DIR / "outputs" / "tables"
FORMAL_CHECKPOINT_DIR = QUESTION_DIR / "outputs" / "checkpoints"
DEFAULT_OUTPUT_ROOT = QUESTION_DIR / "outputs" / "validation_runs" / "exact_vs_heuristic"
WINDOW_COUNT = 10
DECISION_WINDOW = 24
LOOKAHEAD = 48


def _configure_logging(output_root: Path, level_name: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Unsupported log level: {level_name}")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    logging.basicConfig(
        level=level,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                output_root / "q2_exact_heuristic_benchmark.log",
                encoding="utf-8",
            ),
        ],
        format=formatter._fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _read_calibration(
    bundle: q2.InputBundle,
    baseline_metrics: dict[str, float],
) -> q2.GlobalCalibration:
    path = FORMAL_CHECKPOINT_DIR / "q2_refactored_global_calibration_v6.json"
    if not path.is_file():
        raise FileNotFoundError(f"Official global calibration cache is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != q2.CALIBRATION_SCHEMA_VERSION:
        raise RuntimeError(
            f"Calibration cache version mismatch: actual={data.get('schema_version')}，"
            f"expected={q2.CALIBRATION_SCHEMA_VERSION}"
        )
    signature = q2._task_signature(bundle)
    if data.get("task_signature") != signature:
        raise RuntimeError("Calibration cache does not match the current Q2 task-input signature; comparison refused")
    for metric in q2.OBJECTIVE_NAMES:
        cached = float(data["baseline_value"][metric])
        current = float(baseline_metrics[metric])
        if not math.isclose(cached, current, rel_tol=1e-10, abs_tol=1e-8):
            raise RuntimeError(
                f"Calibration cache baseline mismatch: {metric}; cached={cached}; current={current}"
            )
    return q2.GlobalCalibration(
        ideal_lb={key: float(value) for key, value in data["ideal_lb"].items()},
        baseline_value={
            key: float(value) for key, value in data["baseline_value"].items()
        },
        reference_value={
            key: float(value) for key, value in data["reference_value"].items()
        },
        scale={key: float(value) for key, value in data["scale"].items()},
        active_metrics=tuple(str(value) for value in data["active_metrics"]),
    )


def _load_baseline(
    bundle: q2.InputBundle,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    path = FORMAL_TABLES_DIR / "q2_baseline_assignments.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Official Q2 compute-only baseline assignments are missing: {path}")
    baseline = pd.read_csv(path, encoding="utf-8-sig")
    q2._validate_assignments(bundle, baseline, "Official compute-only baseline")
    profile, metrics = q2._schedule_profile(bundle, baseline)
    q2._validate_profile(profile, "Official compute-only baseline")
    return baseline, profile, metrics


def _active_context(
    classes: list[q2.ExactTaskClass],
    pools: dict[int, list[str]],
    tau: int,
    decision_end: int,
    plan_end: int,
) -> tuple[set[str], int]:
    active_task_ids: set[str] = set()
    active_class_count = 0
    for task_class in classes:
        if not pools[task_class.class_id] or task_class.arrival_hour >= plan_end:
            continue
        starts = q2._window_valid_start_hours(
            task_class,
            tau,
            decision_end,
            plan_end,
        )
        if starts:
            active_class_count += 1
            active_task_ids.update(pools[task_class.class_id])
    return active_task_ids, active_class_count


def _heuristic_record(
    *,
    bundle: q2.InputBundle,
    classes: list[q2.ExactTaskClass],
    class_lookup: dict[int, q2.ExactTaskClass],
    pools: dict[int, list[str]],
    committed_assignments: pd.DataFrame,
    baseline_assignments: pd.DataFrame,
    baseline_profile: pd.DataFrame,
    baseline_latency: dict[str, float],
    calibration: q2.GlobalCalibration,
    tau: int,
    decision_end: int,
    plan_end: int,
    active_class_count: int,
    active_task_count: int,
    progress_interval: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    env = q2._heuristic_resource_environment(bundle)
    committed_gpu, committed_ai = q2._heuristic_initial_committed_loads(
        committed_assignments,
        env,
    )
    past_delta = q2._past_delta(
        bundle=bundle,
        balanced_committed=committed_assignments,
        baseline_assignments=baseline_assignments,
        baseline_latency=baseline_latency,
        tau=tau,
        global_task_count=len(bundle.tasks),
    )
    active_task_ids, _ = _active_context(
        classes,
        pools,
        tau,
        decision_end,
        plan_end,
    )
    baseline_scope = q2._baseline_scope_value(
        bundle=bundle,
        baseline_assignments=baseline_assignments,
        baseline_profile=baseline_profile,
        baseline_latency=baseline_latency,
        active_task_ids=active_task_ids,
        tau=tau,
        objective_end=plan_end,
        global_task_count=len(bundle.tasks),
    )

    plan, z_greedy, _scope, construct_stats = q2._construct_heuristic_window_plan(
        bundle=bundle,
        classes=classes,
        class_lookup=class_lookup,
        pools=pools,
        committed_assignments=committed_assignments,
        committed_gpu=committed_gpu,
        committed_ai=committed_ai,
        env=env,
        baseline_assignments=baseline_assignments,
        calibration=calibration,
        past_delta=past_delta,
        baseline_scope=baseline_scope,
        tau=tau,
        decision_end=decision_end,
        plan_end=plan_end,
        progress_interval=progress_interval,
    )
    z_before, estimated_before, _deviations_before, plan_gpu, plan_ai = (
        q2._evaluate_heuristic_plan(
            plan=plan,
            committed_gpu=committed_gpu,
            committed_ai=committed_ai,
            env=env,
            calibration=calibration,
            past_delta=past_delta,
            baseline_scope=baseline_scope,
            tau=tau,
            plan_end=plan_end,
            global_task_count=len(bundle.tasks),
        )
    )
    if abs(z_before - z_greedy) > 1e-6 * max(1.0, abs(z_before)):
        logging.info(
            "Window %d heuristic z differs slightly: incremental=%.9g, independent=%.9g; use independent recomputation.",
            tau // DECISION_WINDOW,
            z_greedy,
            z_before,
        )

    lns_info: dict[str, Any] = {"LNSStatus": "disabled", "LNSAccepted": 0}
    z_final = z_before
    estimated_final = estimated_before
    if q2.HEURISTIC_LNS_ENABLED:
        selected = q2._lns_select_classes(
            plan=plan,
            class_lookup=class_lookup,
            env=env,
            gpu=plan_gpu,
            ai=plan_ai,
            decision_end=decision_end,
            max_classes=q2.HEURISTIC_LNS_MAX_CLASSES,
        )
        refined, lns_info = q2._solve_lns_refinement(
            bundle=bundle,
            classes=classes,
            class_lookup=class_lookup,
            pools=pools,
            committed_assignments=committed_assignments,
            heuristic_plan=plan,
            selected_classes=selected,
            calibration=calibration,
            past_delta=past_delta,
            baseline_scope=baseline_scope,
            tau=tau,
            decision_end=decision_end,
            plan_end=plan_end,
            progress_interval=progress_interval,
        )
        if refined is not None:
            try:
                z_refined, estimated_refined, _deviations_refined, _, _ = (
                    q2._evaluate_heuristic_plan(
                        plan=refined,
                        committed_gpu=committed_gpu,
                        committed_ai=committed_ai,
                        env=env,
                        calibration=calibration,
                        past_delta=past_delta,
                        baseline_scope=baseline_scope,
                        tau=tau,
                        plan_end=plan_end,
                        global_task_count=len(bundle.tasks),
                    )
                )
                if z_refined < z_before - q2.HEURISTIC_SCORE_TOL:
                    plan = refined
                    z_final = z_refined
                    estimated_final = estimated_refined
                    lns_info["LNSAccepted"] = 1
            except RuntimeError as exc:
                lns_info["LNSIndependentCheckError"] = str(exc)

    q2._evaluate_heuristic_plan(
        plan=plan,
        committed_gpu=committed_gpu,
        committed_ai=committed_ai,
        env=env,
        calibration=calibration,
        past_delta=past_delta,
        baseline_scope=baseline_scope,
        tau=tau,
        plan_end=plan_end,
        global_task_count=len(bundle.tasks),
    )
    committed_count = int(
        (plan["StartHour"].astype(float) < float(decision_end) - q2.FLOAT_EPS).sum()
    )
    elapsed = time.perf_counter() - started
    return {
        "WindowID": tau // DECISION_WINDOW,
        "Mode": "balanced",
        "SolveMethod": "validation_same_exact_history_greedy_plus_lns",
        "WindowStartHour": tau,
        "DecisionEndHourExclusive": decision_end,
        "PlanEndHourExclusive": plan_end,
        "ActiveClassCount": active_class_count,
        "ActiveTaskCount": active_task_count,
        "CandidateOptionCount": construct_stats["CandidateOptionCount"],
        "CommittedTaskCount": committed_count,
        "ElapsedSeconds": elapsed,
        "ZBeforeLNS": z_before,
        "ZStar": z_final,
        "EstimatedMaxNormalizedDeviation": z_final,
        "LNSStatus": lns_info.get("LNSStatus", ""),
        "LNSAccepted": lns_info.get("LNSAccepted", 0),
        "LNSClassCount": lns_info.get("LNSClassCount", 0),
        "LNSCandidateOptionCount": lns_info.get("LNSCandidateOptionCount", 0),
        "MIPGap": lns_info.get("LNSMIPGap", np.nan),
        **{
            f"EstimatedGlobal_{metric}": estimated_final.get(metric, np.nan)
            for metric in q2.OBJECTIVE_NAMES
        },
    }


def _exact_record(
    *,
    model: q2.AggregatedModel,
    result: q2.SolveResult,
    metrics: dict[str, float],
    calibration: q2.GlobalCalibration,
    past_delta: dict[str, float],
    baseline_scope: dict[str, float],
    committed_count: int,
    active_class_count: int,
    active_task_count: int,
    tau: int,
    decision_end: int,
    plan_end: int,
) -> dict[str, Any]:
    estimated_global: dict[str, float] = {}
    for metric in q2.OBJECTIVE_NAMES:
        raw_value = q2._objective_value(model, metric, result.vector)
        estimated_global[metric] = (
            float(calibration.baseline_value[metric])
            + float(past_delta.get(metric, 0.0))
            + raw_value
            - float(baseline_scope[metric])
        )
    return {
        "WindowID": tau // DECISION_WINDOW,
        "Mode": "balanced",
        "SolveMethod": "validation_exact_minmax_milp",
        "WindowStartHour": tau,
        "DecisionEndHourExclusive": decision_end,
        "PlanEndHourExclusive": plan_end,
        "ActiveClassCount": active_class_count,
        "ActiveTaskCount": active_task_count,
        "CandidateOptionCount": model.option_count,
        "DecisionIntegerOptionCount": int(
            np.sum(model.integrality[: model.option_count] == 1)
        ),
        "LookaheadContinuousOptionCount": int(
            np.sum(model.integrality[: model.option_count] == 0)
        ),
        "CommittedTaskCount": committed_count,
        "ElapsedSeconds": result.elapsed_seconds,
        "MIPGap": result.mip_gap,
        "MIPStatus": result.status,
        "MIPMessage": result.message,
        "ZStar": metrics.get("ZStar", np.nan),
        "EstimatedMaxNormalizedDeviation": metrics.get(
            "EstimatedMaxNormalizedDeviation", np.nan
        ),
        **{
            f"EstimatedGlobal_{metric}": estimated_global[metric]
            for metric in q2.OBJECTIVE_NAMES
        },
    }


def _write_progress(
    output_root: Path,
    exact_rows: list[dict[str, Any]],
    heuristic_rows: list[dict[str, Any]],
    assignments: pd.DataFrame,
    next_window_id: int,
    status: str,
    error: str = "",
) -> None:
    pd.DataFrame(exact_rows).to_csv(
        output_root / "exact_windows.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.15g",
    )
    pd.DataFrame(heuristic_rows).to_csv(
        output_root / "heuristic_windows.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.15g",
    )
    assignments.to_csv(
        output_root / "exact_assignments.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.15g",
    )
    state = {
        "schema_version": 1,
        "status": status,
        "error": error,
        "decision_window": DECISION_WINDOW,
        "lookahead": LOOKAHEAD,
        "required_window_count": WINDOW_COUNT,
        "next_window_id": next_window_id,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "formal_outputs_untouched": True,
    }
    (output_root / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_progress(
    output_root: Path,
    resume: bool,
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    state_path = output_root / "state.json"
    if not state_path.is_file():
        return 0, [], [], pd.DataFrame()
    if not resume:
        raise RuntimeError(
            f"{output_root} already contains comparison progress; use another --output-root or resume to avoid overwriting."
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if int(state.get("decision_window", -1)) != DECISION_WINDOW or int(
        state.get("lookahead", -1)
    ) != LOOKAHEAD:
        raise RuntimeError("Existing comparison H/K does not match this script")
    exact_path = output_root / "exact_windows.csv"
    heuristic_path = output_root / "heuristic_windows.csv"
    assignments_path = output_root / "exact_assignments.csv"
    exact = (
        pd.read_csv(exact_path, encoding="utf-8-sig").to_dict("records")
        if exact_path.is_file()
        else []
    )
    heuristic = (
        pd.read_csv(heuristic_path, encoding="utf-8-sig").to_dict("records")
        if heuristic_path.is_file()
        else []
    )
    assignments = (
        pd.read_csv(assignments_path, encoding="utf-8-sig")
        if assignments_path.is_file()
        else pd.DataFrame()
    )
    next_window_id = int(state.get("next_window_id", len(exact)))
    if next_window_id != len(exact) or len(exact) != len(heuristic):
        raise RuntimeError("Comparison progress file counts disagree; continuation refused to avoid mismatched windows")
    return next_window_id, exact, heuristic, assignments


def run(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    _configure_logging(output_root, args.log_level)
    next_window_id, exact_rows, heuristic_rows, exact_assignments = _load_progress(
        output_root,
        args.resume,
    )
    if next_window_id >= args.windows:
        logging.info("Exact same-window comparison already completed: windows 0--%d.", next_window_id - 1)
        return 0

    logging.info(
        "Exact same-window comparison started: target windows 0--%d, H=%d, K=%d, MIP gap=%.4g, "
        "per-window time limit=%.0f->%.0fs.",
        args.windows - 1,
        DECISION_WINDOW,
        LOOKAHEAD,
        args.mip_rel_gap,
        args.solver_time_limit,
        args.max_solver_time_limit,
    )
    bundle = q2._load_inputs()
    classes, _ = q2._build_exact_task_classes(bundle)
    class_lookup = {task_class.class_id: task_class for task_class in classes}
    baseline_assignments, baseline_profile, baseline_metrics = _load_baseline(bundle)
    calibration = _read_calibration(bundle, baseline_metrics)
    baseline_latency = q2._baseline_latency_lookup(baseline_assignments)

    pools = q2._make_task_pools(classes)
    if not exact_assignments.empty:
        q2._remove_committed_from_pools(pools, exact_assignments)
    logging.info(
        "Inputs and official baseline locked: tasks=%d, homogeneous classes=%d, completed comparison windows=%d, fixed tasks=%d.",
        len(bundle.tasks),
        len(classes),
        next_window_id,
        len(exact_assignments),
    )

    while next_window_id < args.windows:
        tau = next_window_id * DECISION_WINDOW
        decision_end = min(tau + DECISION_WINDOW, q2.TERMINAL_HOUR)
        plan_end = min(decision_end + LOOKAHEAD, q2.TERMINAL_HOUR)
        active_task_ids, active_class_count = _active_context(
            classes,
            pools,
            tau,
            decision_end,
            plan_end,
        )
        if not active_task_ids:
            raise RuntimeError(f"Window {next_window_id} has no active tasks for same-window comparison")
        logging.info(
            "Window %d started: tau=%d, active tasks=%d, active classes=%d; heuristic first, exact MILP second.",
            next_window_id,
            tau,
            len(active_task_ids),
            active_class_count,
        )

        remaining_count = q2._remaining_count(pools)
        past_delta = q2._past_delta(
            bundle=bundle,
            balanced_committed=exact_assignments,
            baseline_assignments=baseline_assignments,
            baseline_latency=baseline_latency,
            tau=tau,
            global_task_count=len(bundle.tasks),
        )
        baseline_scope = q2._baseline_scope_value(
            bundle=bundle,
            baseline_assignments=baseline_assignments,
            baseline_profile=baseline_profile,
            baseline_latency=baseline_latency,
            active_task_ids=active_task_ids,
            tau=tau,
            objective_end=plan_end,
            global_task_count=len(bundle.tasks),
        )

        heuristic_row = _heuristic_record(
            bundle=bundle,
            classes=classes,
            class_lookup=class_lookup,
            pools=pools,
            committed_assignments=exact_assignments,
            baseline_assignments=baseline_assignments,
            baseline_profile=baseline_profile,
            baseline_latency=baseline_latency,
            calibration=calibration,
            tau=tau,
            decision_end=decision_end,
            plan_end=plan_end,
            active_class_count=active_class_count,
            active_task_count=len(active_task_ids),
            progress_interval=args.progress_interval,
        )
        logging.info(
            "Window %d heuristic completed: zBefore=%.9g, zFinal=%.9g, LNS=%s, elapsed=%.1fs.",
            next_window_id,
            heuristic_row["ZBeforeLNS"],
            heuristic_row["ZStar"],
            heuristic_row.get("LNSStatus", ""),
            heuristic_row["ElapsedSeconds"],
        )

        model = q2._build_aggregated_model(
            bundle=bundle,
            classes=classes,
            class_lookup=class_lookup,
            remaining_count=remaining_count,
            fixed_assignments=exact_assignments,
            tau=tau,
            decision_end=decision_end,
            plan_end=plan_end,
            include_z=True,
            include_energy_variables=True,
            integerize_lookahead=False,
        )
        exact_started = time.perf_counter()
        try:
            result, exact_metrics, _solver_rows = q2._solve_balanced_minmax_window(
                model=model,
                calibration=calibration,
                past_delta=past_delta,
                baseline_scope=baseline_scope,
                mip_rel_gap=args.mip_rel_gap,
                initial_time_limit=args.solver_time_limit,
                max_time_limit=args.max_solver_time_limit,
                progress_interval=args.progress_interval,
                window_id=next_window_id,
                previous_k_warmstart_available=False,
            )
        except Exception:
            del model
            gc.collect()
            raise
        if result.status != 0:
            message = (
                f"Window {next_window_id} exact MILP did not return status=0: "
                f"status={result.status}，message={result.message}，"
                f"elapsed={time.perf_counter() - exact_started:.1f}s。"
                "Increase --max-solver-time-limit before resuming; this comparison is not saved as exact evidence."
            )
            del result, model
            gc.collect()
            raise RuntimeError(message)

        committed = q2._commit_h_counts(
            model=model,
            vector=result.vector,
            class_lookup=class_lookup,
            pools=pools,
        )
        if committed.empty:
            del result, model
            gc.collect()
            raise RuntimeError(f"Window {next_window_id} exact MILP committed no tasks in the H interval")
        exact_assignments = pd.concat(
            [exact_assignments, committed],
            ignore_index=True,
        )
        exact_row = _exact_record(
            model=model,
            result=result,
            metrics=exact_metrics,
            calibration=calibration,
            past_delta=past_delta,
            baseline_scope=baseline_scope,
            committed_count=len(committed),
            active_class_count=active_class_count,
            active_task_count=len(active_task_ids),
            tau=tau,
            decision_end=decision_end,
            plan_end=plan_end,
        )
        exact_rows.append(exact_row)
        heuristic_rows.append(heuristic_row)
        next_window_id += 1
        _write_progress(
            output_root,
            exact_rows,
            heuristic_rows,
            exact_assignments,
            next_window_id,
            "RUNNING" if next_window_id < args.windows else "COMPLETED",
        )
        logging.info(
            "Window %d exact MILP completed: z=%.9g, committed=%d, status=%d, gap=%s, elapsed=%.1fs; "
            "progress=%d/%d.",
            next_window_id - 1,
            exact_row["ZStar"],
            len(committed),
            result.status,
            f"{result.mip_gap:.6g}" if math.isfinite(result.mip_gap) else "NA",
            result.elapsed_seconds,
            next_window_id,
            args.windows,
        )
        del result, model, committed
        gc.collect()

    logging.info("Exact same-window comparison completed: exact_windows.csv and heuristic_windows.csv generated.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Q2 windows 0--9: exact MILP versus heuristic plus LNS under identical history"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Comparison directory; default outputs/validation_runs/exact_vs_heuristic",
    )
    parser.add_argument("--windows", type=int, default=WINDOW_COUNT)
    parser.add_argument("--mip-rel-gap", type=float, default=5e-3)
    parser.add_argument("--solver-time-limit", type=float, default=180.0)
    parser.add_argument("--max-solver-time-limit", type=float, default=900.0)
    parser.add_argument("--progress-interval", type=float, default=30.0)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume saved windows; enabled by default",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args()
    if not 1 <= args.windows <= WINDOW_COUNT:
        raise SystemExit(f"--windows must be within 1--{WINDOW_COUNT} inclusive")
    if not math.isfinite(args.mip_rel_gap) or args.mip_rel_gap < 0:
        raise SystemExit("--mip-rel-gap must be finite and nonnegative")
    if not math.isfinite(args.solver_time_limit) or args.solver_time_limit <= 0:
        raise SystemExit("--solver-time-limit must be finite and positive")
    if (
        not math.isfinite(args.max_solver_time_limit)
        or args.max_solver_time_limit < args.solver_time_limit
    ):
        raise SystemExit("--max-solver-time-limit must be >= --solver-time-limit")
    if not math.isfinite(args.progress_interval) or args.progress_interval < 0:
        raise SystemExit("--progress-interval must be >= 0")
    try:
        return run(args)
    except Exception as exc:
        output_root = args.output_root.resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        try:
            existing_state_path = output_root / "state.json"
            existing_state = (
                json.loads(existing_state_path.read_text(encoding="utf-8"))
                if existing_state_path.is_file()
                else {}
            )
            exact_path = output_root / "exact_windows.csv"
            heuristic_path = output_root / "heuristic_windows.csv"
            assignments_path = output_root / "exact_assignments.csv"
            exact_rows = (
                pd.read_csv(exact_path, encoding="utf-8-sig").to_dict("records")
                if exact_path.is_file()
                else []
            )
            heuristic_rows = (
                pd.read_csv(heuristic_path, encoding="utf-8-sig").to_dict("records")
                if heuristic_path.is_file()
                else []
            )
            assignments = (
                pd.read_csv(assignments_path, encoding="utf-8-sig")
                if assignments_path.is_file()
                else pd.DataFrame()
            )
            _write_progress(
                output_root,
                exact_rows,
                heuristic_rows,
                assignments,
                int(existing_state.get("next_window_id", len(exact_rows))),
                "FAILED",
                str(exc),
            )
        except Exception:
            pass
        logging.exception("Exact same-window comparison failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
