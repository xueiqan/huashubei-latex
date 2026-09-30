"""问题1独立流程入口。

执行顺序固定为：本题预处理 → 模型求解 → 模型检验 → 结果绘图。
每个阶段都使用独立脚本，并在阶段结束后检查关键输出是否落盘；任一阶段失败
或关键结果缺失，入口返回非零状态，避免把不完整结果交给建模手或论文写作。
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path


QUESTION_DIR = Path(__file__).resolve().parent
QUESTION_NAME = QUESTION_DIR.name
PROCESSED_DIR = QUESTION_DIR / "data" / "processed"
OUTPUTS_DIR = QUESTION_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
LOGS_DIR = OUTPUTS_DIR / "logs"

STAGES = ("preprocess", "model", "validation", "plot")
STAGE_ARTIFACTS: dict[str, tuple[Path, ...]] = {
    "preprocess": (
        PROCESSED_DIR / "shared" / "tasks_clean.csv",
        PROCESSED_DIR / "shared" / "task_candidate_regions.csv",
        PROCESSED_DIR / "q1" / "hourly_demand_panel.csv",
        PROCESSED_DIR / "q1" / "region_hour_capacity.csv",
    ),
    "model": (
        TABLES_DIR / "forecast_predictions.csv",
        TABLES_DIR / "forecast_metrics.csv",
        TABLES_DIR / "dispatch_assignments.csv",
        TABLES_DIR / "dispatch_resource_profile.csv",
        TABLES_DIR / "dispatch_summary.csv",
    ),
    "validation": (
        TABLES_DIR / "validation_summary.csv",
        TABLES_DIR / "validation_critical_pressure.csv",
        TABLES_DIR / "validation_dispatch_certificate.csv",
    ),
    "plot": (
        FIGURES_DIR / "q1_group1_demand_structure.pdf",
        FIGURES_DIR / "q1_group2_core_results.pdf",
        FIGURES_DIR / "q1_group3_validation_robustness.pdf",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="华数杯C题问题1：预处理、建模、检验和绘图完整入口"
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="跳过本题预处理；仅当data/processed中的关键输入已存在时使用",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="跳过模型求解；仅当outputs/tables中的模型结果已存在时使用",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="跳过模型检验；不建议用于论文最终取数",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="跳过结果绘图；不影响模型表和检验表生成",
    )
    return parser.parse_args()


def _configure_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / f"{QUESTION_NAME}_run.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def _missing_artifacts(stage: str) -> list[Path]:
    return [path for path in STAGE_ARTIFACTS[stage] if not path.is_file()]


def _check_artifacts(stage: str) -> bool:
    missing = _missing_artifacts(stage)
    if missing:
        logging.error(
            "阶段%s关键输出缺失：%s",
            stage,
            "; ".join(str(path) for path in missing),
        )
        return False
    logging.info("阶段%s关键输出已确认：%d项。", stage, len(STAGE_ARTIFACTS[stage]))
    return True


def run_stage(stage: str) -> bool:
    script = QUESTION_DIR / f"{stage}.py"
    if not script.is_file():
        logging.error("阶段脚本不存在：%s", script)
        return False

    logging.info("========== Q1阶段开始：%s ==========", stage)
    try:
        process = subprocess.Popen(
            [sys.executable, str(script)],
            cwd=QUESTION_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            logging.error("阶段%s没有可读取的标准输出。", stage)
            process.kill()
            process.wait()
            return False
        for line in process.stdout:
            message = line.rstrip()
            if message:
                logging.info("[%s] %s", stage, message)
        return_code = process.wait()
    except OSError:
        logging.exception("阶段%s无法启动。", stage)
        return False

    if return_code != 0:
        logging.error("阶段%s失败，退出码=%d。", stage, return_code)
        return False
    if not _check_artifacts(stage):
        logging.error("阶段%s虽正常退出，但关键结果不完整。", stage)
        return False
    logging.info("========== Q1阶段完成：%s ==========", stage)
    return True


def _skip_requested_stage(stage: str, skip_flags: dict[str, bool]) -> bool:
    logging.info("跳过阶段%s。", stage)
    later_requested = any(not skip_flags[later] for later in STAGES[STAGES.index(stage) + 1 :])
    if later_requested and not _check_artifacts(stage):
        logging.error("后续阶段需要阶段%s的结果，但现有结果不完整。", stage)
        return False
    return True


def main() -> int:
    args = parse_args()
    _configure_logging()
    for directory in (PROCESSED_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    skip_flags = {
        "preprocess": args.skip_preprocess,
        "model": args.skip_model,
        "validation": args.skip_validation,
        "plot": args.skip_plot,
    }
    logging.info("Q1流程入口启动：目录=%s。", QUESTION_DIR)

    for stage in STAGES:
        if skip_flags[stage]:
            if not _skip_requested_stage(stage, skip_flags):
                return 1
            continue
        if not run_stage(stage):
            logging.error("Q1流程在阶段%s停止。", stage)
            return 1

    logging.info("Q1流程完成：预处理、模型、检验、绘图阶段均已按请求处理。")
    logging.info("结果表目录：%s", TABLES_DIR)
    logging.info("结果图目录：%s", FIGURES_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
