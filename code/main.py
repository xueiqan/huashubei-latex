"""全题入口：按题目运行各自的完整流程。"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
QUESTION_NAMES = ("question_01", "question_02", "question_03", "question_04")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="华数杯全题运行入口")
    parser.add_argument("--question", choices=("all", "1", "2", "3", "4"), default="all")
    parser.add_argument("--skip-shared-preprocess", action="store_true")
    return parser.parse_args()


def run_script(script_path: Path) -> bool:
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=script_path.parent,
        text=True,
        capture_output=True,
    )
    if result.stdout:
        logging.info("%s", result.stdout.strip())
    if result.stderr:
        logging.warning("%s", result.stderr.strip())
    if result.returncode:
        logging.error("失败：%s", script_path)
        return False
    return True


def main() -> int:
    args = parse_args()
    log_dir = ROOT_DIR / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_dir / "run.log", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )

    if not args.skip_shared_preprocess:
        if not run_script(ROOT_DIR / "preprocess.py"):
            return 1

    selected = QUESTION_NAMES if args.question == "all" else (f"question_0{args.question}",)
    failed = [name for name in selected if not run_script(ROOT_DIR / "question" / name / "main.py")]
    logging.info("完成问题：%s", ", ".join(name for name in selected if name not in failed))
    if failed:
        logging.error("失败问题：%s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
