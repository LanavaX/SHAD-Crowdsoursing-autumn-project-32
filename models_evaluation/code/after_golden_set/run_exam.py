"""Run a model on the exam images (cross-checks / fallback submissions).

Usage: HF_HOME=/home/hf_cache python code/after_golden_set/run_exam.py <model>
Writes runs/exam/<model>.jsonl (with scores where available).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import io
from lib.runner import run_dataset

EXAM = io.PROJECT_ROOT / "data" / "exam" / "images"


def main():
    name = sys.argv[1]
    photos = [(int(p.stem), p) for p in sorted(EXAM.iterdir())
              if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    print(f"exam: {len(photos)} photos")
    run_dataset([name], photos, io.RUNS_DIR / "exam")


if __name__ == "__main__":
    main()
