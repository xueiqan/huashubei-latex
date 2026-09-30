"""Independent Question 1 workflow.

Run preprocessing, modeling, validation, and plotting in that order. Each stage
uses its own script and checks its required artifacts. A failed stage or missing
artifact produces a nonzero exit status to prevent incomplete results from
being used for modeling or paper preparation.
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
        description="Huashu Cup Problem C, Question 1: preprocessing, modeling, validation, and plotting"
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing only when the required data/processed inputs already exist",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip solving only when the model results in outputs/tables already exist",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation; discouraged when collecting final paper results",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip plotting; model and validation tables are still generated",
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
            "Stage %s is missing required artifacts: %s",
            stage,
            "; ".join(str(path) for path in missing),
        )
        return False
    logging.info("Stage %s artifacts confirmed: %d items.", stage, len(STAGE_ARTIFACTS[stage]))
    return True


def run_stage(stage: str) -> bool:
    script = QUESTION_DIR / f"{stage}.py"
    if not script.is_file():
        logging.error("Stage script does not exist: %s", script)
        return False

    logging.info("========== Q1 stage started: %s ==========", stage)
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
            logging.error("Stage %s has no readable standard output.", stage)
            process.kill()
            process.wait()
            return False
        for line in process.stdout:
            message = line.rstrip()
            if message:
                logging.info("[%s] %s", stage, message)
        return_code = process.wait()
    except OSError:
        logging.exception("Stage %s could not start.", stage)
        return False

    if return_code != 0:
        logging.error("Stage %s failed with exit code %d.", stage, return_code)
        return False
    if not _check_artifacts(stage):
        logging.error("Stage %s exited normally but required artifacts are incomplete.", stage)
        return False
    logging.info("========== Q1 stage completed: %s ==========", stage)
    return True


def _skip_requested_stage(stage: str, skip_flags: dict[str, bool]) -> bool:
    logging.info("Skipping stage %s.", stage)
    later_requested = any(not skip_flags[later] for later in STAGES[STAGES.index(stage) + 1 :])
    if later_requested and not _check_artifacts(stage):
        logging.error("Later stages require stage %s, but its existing results are incomplete.", stage)
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
    logging.info("Q1 workflow started: directory=%s.", QUESTION_DIR)

    for stage in STAGES:
        if skip_flags[stage]:
            if not _skip_requested_stage(stage, skip_flags):
                return 1
            continue
        if not run_stage(stage):
            logging.error("Q1 workflow stopped at stage %s.", stage)
            return 1

    logging.info("Q1 workflow completed: all requested preprocessing, modeling, validation, and plotting stages were handled.")
    logging.info("Table directory: %s", TABLES_DIR)
    logging.info("Figure directory: %s", FIGURES_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
