"""Sweep score thresholds on smoke predictions against the pseudo-GT.

Orientation only — final thresholds are tuned on the real golden set.
Usage: python code/before_golden_set/scout_thresholds.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import io
from lib.metrics import evaluate

THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 13)]  # 0.05 .. 0.60


def main():
    runs_dir = io.RUNS_DIR / (sys.argv[1] if len(sys.argv) > 1 else "smoke")
    gt = io.read_jsonl(io.SMOKE_DIR / "pseudo_gt.jsonl")
    lines = []
    for f in sorted(runs_dir.glob("*.jsonl")):
        preds = io.read_jsonl(f)
        has_scores = any(b.get("score") is not None
                         for p in preds.values() for b in p["boxes"])
        if not has_scores:
            lines.append(f"{f.stem}: no scores (VLM) — threshold not applicable")
            continue
        best = None
        for t in THRESHOLDS:
            filt = {pid: {"boxes": [b for b in p["boxes"] if b.get("score", 1.0) >= t],
                          "error": p["error"]}
                    for pid, p in preds.items()}
            r = evaluate(gt, filt)
            lines.append(f"{f.stem} thr={t:.2f} F1={r['f1']:.3f} "
                         f"P={r['precision']:.3f} R={r['recall']:.3f} "
                         f"FPempty={r['fp_on_empty']} FP={r['fp']} FN={r['fn']}")
            if best is None or r["f1"] > best[1]:
                best = (t, r["f1"])
        lines.append(f"BEST {f.stem}: thr={best[0]:.2f} F1={best[1]:.3f}")
        lines.append("")
    report = "\n".join(lines)
    print(report)
    out = runs_dir / "scout_report.txt"
    out.write_text(report)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
