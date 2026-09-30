"""Run preprocessing, modeling, validation, and plotting within this question directory."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path


QUESTION_DIR = Path(__file__).resolve().parent
QUESTION_NAME = QUESTION_DIR.name
PROCESSED_DIR = QUESTION_DIR / "data" / "processed"
OUTPUTS_DIR = QUESTION_DIR / "outputs"


def has_model_input() -> bool:
    return any(path.is_file() for path in PROCESSED_DIR.rglob("*"))


def run_stage(name: str) -> bool:
    script = QUESTION_DIR / f"{name}.py"
    result = subprocess.run([sys.executable, str(script)], cwd=QUESTION_DIR, text=True, capture_output=True)
    if result.stdout:
        logging.info("%s", result.stdout.strip())
    if result.stderr:
        logging.warning("%s", result.stderr.strip())
    return result.returncode == 0


def main() -> int:
    for directory in (PROCESSED_DIR, OUTPUTS_DIR / "tables", OUTPUTS_DIR / "figures", OUTPUTS_DIR / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(OUTPUTS_DIR / "logs" / f"{QUESTION_NAME}_run.log", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    if not run_stage("preprocess"):
        return 1
    if not has_model_input():
        logging.info("No processed model input is available; skipping subsequent stages.")
        return 0
    if not run_stage("model"):
        return 1
    if not run_stage("validation"):
        logging.warning("Validation failed; inspect this question's validation.py.")
    if not run_stage("plot"):
        logging.warning("Plotting failed; inspect this question's plot.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
