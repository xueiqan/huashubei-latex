"""Check that Q2 validation reads both historical and English runtime logs."""

import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "question/question_02"))
from validation import _read_runtime_seconds


def check_runtime_logs():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        log = root / "question_02_model.log"
        assert math.isnan(_read_runtime_seconds(root))
        for message in ("total elapsed=12.5s", "\u603b\u8017\u65f6=12.5s"):
            log.write_text(message, encoding="utf-8")
            assert _read_runtime_seconds(root) == 12.5
        log.write_text("total elapsed=1s\ntotal elapsed=23.75s", encoding="utf-8")
        assert _read_runtime_seconds(root) == 23.75
        log.write_text("No completion record", encoding="utf-8")
        assert math.isnan(_read_runtime_seconds(root))


if __name__ == "__main__":
    check_runtime_logs()
    print("Historical and English runtime-log checks passed.")
