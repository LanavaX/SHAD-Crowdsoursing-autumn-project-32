"""Inference of all models on the real golden set -> runs/golden/<model>.jsonl.

Expects data/golden/golden.jsonl (contest format) and images under
data/golden/images/ (override with --images DIR).

Usage: HF_HOME=/home/hf_cache python code/after_golden_set/run_golden.py [--images DIR] [model ...]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import io, models
from lib.runner import run_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(io.GOLDEN_DIR))
    ap.add_argument("models", nargs="*")
    args = ap.parse_args()

    gt = io.read_jsonl(io.GOLDEN_DIR / "golden.jsonl")
    photos = [(pid, io.image_path(pid, args.images)) for pid in gt]
    print(f"golden set: {len(photos)} photos from {args.images}")
    names = args.models or models.available()
    run_dataset(names, photos, io.RUNS_DIR / "golden")


if __name__ == "__main__":
    main()
