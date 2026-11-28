"""Run all detectors on the smoke set -> runs/smoke/<model>.jsonl (+scores, low threshold).

Usage: python code/before_golden_set/run_smoke.py [model_name ...]   (default: all)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import io, models
from lib.runner import run_dataset


def main():
    args = sys.argv[1:]
    out_dir = io.RUNS_DIR / "smoke"
    if args and args[0] == "--out":  # usage: run_smoke.py --out runs/smoke_v2 [model ...]
        out_dir = io.RUNS_DIR / args[1]
        args = args[2:]
    manifest = io.read_manifest()
    photos = [(int(r["photo_id"]), io.image_path(int(r["photo_id"]))) for r in manifest]
    names = args or models.available()
    run_dataset(names, photos, out_dir)


if __name__ == "__main__":
    main()
