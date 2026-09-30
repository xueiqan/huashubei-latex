"""准备和归档问题2的隔离验证运行。

本文件不启动长时间求解。它只完成两件事：

1. prepare：建立K=24或K=72的最小独立运行目录，并打印用户应执行的model.py命令；
2. collect：检查独立运行已完成后，把tables和logs复制到正式验证专用目录。

正式K=48的tables、logs和checkpoints不在本文件的写入范围内。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


QUESTION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = QUESTION_DIR.parents[1]
FORMAL_OUTPUT_DIR = QUESTION_DIR / "outputs"
EXPERIMENT_PROJECTS_DIR = PROJECT_DIR.parent / f"{PROJECT_DIR.name}_q2_validation_runs"

MODEL_SOURCE = QUESTION_DIR / "model.py"
Q2_INPUT_SOURCE = QUESTION_DIR / "data" / "processed" / "q2" / "q2_region_hour_input.csv"
SHARED_SOURCE_DIR = (
    PROJECT_DIR
    / "question"
    / "question_01"
    / "data"
    / "processed"
    / "shared"
)
CHECKPOINT_SOURCE_DIR = QUESTION_DIR / "outputs" / "checkpoints"

SHARED_FILES = (
    "tasks_clean.csv",
    "task_candidate_regions.csv",
    "storage_params.csv",
)
CALIBRATION_FILES = (
    "q2_refactored_global_calibration_v6.json",
    "q2_cg_anchor_v6_Cost.json",
    "q2_cg_anchor_v6_Cost_positive.npz",
    "q2_cg_anchor_v6_Carbon.json",
    "q2_cg_anchor_v6_Carbon_positive.npz",
    "q2_cg_anchor_v6_MeanLatency.json",
    "q2_cg_anchor_v6_MeanLatency_positive.npz",
    "q2_cg_anchor_v6_RenewableUnusedRate.json",
    "q2_cg_anchor_v6_RenewableUnusedRate_positive.npz",
)


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label}不存在：{path}")


def _copy_file(source: Path, target: Path, *, overwrite: bool) -> None:
    _require_file(source, "源文件")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        return
    shutil.copy2(source, target)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _experiment_root(lookahead: int) -> Path:
    if lookahead not in (24, 72):
        raise ValueError("lookahead只能取24或72")
    return EXPERIMENT_PROJECTS_DIR / f"K{lookahead}"


def _state_path(experiment_root: Path) -> Path:
    return (
        experiment_root
        / "question"
        / "question_02"
        / "outputs"
        / "checkpoints"
        / "q2_refactored_balanced_v3_state.json"
    )


def _read_state(experiment_root: Path) -> dict[str, object] | None:
    path = _state_path(experiment_root)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取隔离运行状态：{path}；{exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"隔离运行状态不是JSON对象：{path}")
    return data


def prepare(lookahead: int) -> Path:
    """建立一个不含正式结果的独立Q2运行目录。"""
    experiment_root = _experiment_root(lookahead)
    if experiment_root.resolve() == PROJECT_DIR.resolve():
        raise RuntimeError("拒绝把正式项目目录作为隔离实验目录")

    existing_state = _read_state(experiment_root)
    if existing_state and str(existing_state.get("status", "")).upper() == "RUNNING":
        raise RuntimeError(
            f"{experiment_root}当前状态为RUNNING，拒绝覆盖运行中的实验；"
            "请先让它完成或停止后再执行prepare。"
        )

    q2_question_dir = experiment_root / "question" / "question_02"
    q2_input_target = q2_question_dir / "data" / "processed" / "q2" / Q2_INPUT_SOURCE.name
    shared_target_dir = (
        experiment_root
        / "question"
        / "question_01"
        / "data"
        / "processed"
        / "shared"
    )
    checkpoint_target_dir = q2_question_dir / "outputs" / "checkpoints"

    _copy_file(MODEL_SOURCE, q2_question_dir / MODEL_SOURCE.name, overwrite=True)
    _copy_file(Q2_INPUT_SOURCE, q2_input_target, overwrite=True)
    for name in SHARED_FILES:
        _copy_file(
            SHARED_SOURCE_DIR / name,
            shared_target_dir / name,
            overwrite=True,
        )
    for name in CALIBRATION_FILES:
        _copy_file(
            CHECKPOINT_SOURCE_DIR / name,
            checkpoint_target_dir / name,
            overwrite=existing_state is None,
        )

    (q2_question_dir / "outputs" / "tables").mkdir(parents=True, exist_ok=True)
    (q2_question_dir / "outputs" / "logs").mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_project": str(PROJECT_DIR),
        "experiment_project": str(experiment_root),
        "decision_window": 24,
        "lookahead": lookahead,
        "source_model_sha256": _sha256(MODEL_SOURCE),
        "source_q2_input_sha256": _sha256(Q2_INPUT_SOURCE),
        "formal_outputs_untouched": True,
    }
    (experiment_root / "validation_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return experiment_root


def print_run_command(lookahead: int, experiment_root: Path) -> None:
    python_exe = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
    model_path = experiment_root / "question" / "question_02" / "model.py"
    print("请在新的PowerShell窗口执行以下命令：")
    print(
        f'& "{python_exe}" "{model_path}" '
        f"--decision-window 24 --lookahead {lookahead} --no-resume "
        "--progress-interval 30 --solver-time-limit 180 "
        "--max-solver-time-limit 900"
    )
    print("运行日志：")
    print(
        experiment_root
        / "question"
        / "question_02"
        / "outputs"
        / "logs"
        / "question_02_model_refactored.log"
    )
    print("完成后执行collect，把结果归档到正式验证目录。")


def collect(lookahead: int) -> Path:
    """只复制已完成隔离实验的tables/logs，不复制检查点。"""
    experiment_root = _experiment_root(lookahead)
    state = _read_state(experiment_root)
    if state is None:
        raise RuntimeError(f"尚未找到K={lookahead}的运行状态：{_state_path(experiment_root)}")
    if str(state.get("status", "")).upper() != "COMPLETED":
        raise RuntimeError(
            f"K={lookahead}尚未完成：status={state.get('status')}；"
            "不要把未完成结果纳入稳定性检验。"
        )
    if int(state.get("decision_window", -1)) != 24:
        raise RuntimeError("隔离实验的decision_window不是24，拒绝归档")
    if int(state.get("lookahead", -1)) != lookahead:
        raise RuntimeError("隔离实验的lookahead与归档目标不一致，拒绝归档")

    source_outputs = experiment_root / "question" / "question_02" / "outputs"
    source_tables = source_outputs / "tables"
    source_logs = source_outputs / "logs"
    required = ("q2_assignments.csv", "q2_objective_summary.csv")
    for name in required:
        _require_file(source_tables / name, f"K={lookahead}结果")

    target_root = FORMAL_OUTPUT_DIR / "validation_runs" / f"K{lookahead}"
    target_tables = target_root / "tables"
    target_logs = target_root / "logs"
    for directory in (target_tables, target_logs):
        directory.mkdir(parents=True, exist_ok=True)

    for source in source_tables.glob("*.csv"):
        shutil.copy2(source, target_tables / source.name)
    if source_logs.is_dir():
        for source in source_logs.iterdir():
            if source.is_file() and source.suffix.lower() in {".log", ".txt"}:
                shutil.copy2(source, target_logs / source.name)

    manifest = {
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(experiment_root),
        "target_root": str(target_root),
        "decision_window": 24,
        "lookahead": lookahead,
        "state": state,
        "checkpoints_copied": False,
        "formal_outputs_untouched": True,
    }
    (target_root / "collection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target_root


def status(lookahead: int) -> None:
    experiment_root = _experiment_root(lookahead)
    state = _read_state(experiment_root)
    print(f"实验目录：{experiment_root}")
    if state is None:
        print("状态：未启动或尚未生成检查点")
        return
    for key in ("status", "next_window_id", "next_tau", "decision_window", "lookahead"):
        print(f"{key}: {state.get(key)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="准备/归档Q2 K=24、K=72隔离验证运行")
    parser.add_argument("action", choices=("prepare", "collect", "status"))
    parser.add_argument("--lookahead", type=int, required=True, choices=(24, 72))
    args = parser.parse_args()

    if args.action == "prepare":
        root = prepare(args.lookahead)
        print(f"隔离目录已准备：{root}")
        print_run_command(args.lookahead, root)
    elif args.action == "collect":
        root = collect(args.lookahead)
        print(f"归档完成：{root}")
    else:
        status(args.lookahead)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
