"""Contact sheets (PNG) of predictions vs pseudo-GT for human eyeballing — for the user,
not for the model context. One sheet per model, 12 photos (8 butterfly / 2 moth / 2 plant).

Usage: python code/before_golden_set/contact_sheets.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import io, visualize


def main():
    runs_dir = io.RUNS_DIR / (sys.argv[1] if len(sys.argv) > 1 else "smoke")
    gt = io.read_jsonl(io.SMOKE_DIR / "pseudo_gt.jsonl")
    by_group = {}
    for row in io.read_manifest():
        by_group.setdefault(row["group"], []).append(int(row["photo_id"]))
    sample = (by_group["butterfly"][:8] + by_group["moth"][:2] + by_group["plant"][:2])

    out_dir = runs_dir / "sheets"
    for f in sorted(runs_dir.glob("*.jsonl")):
        if f.stem == "pseudo_gt":
            continue
        preds = io.read_jsonl(f)
        # operating threshold from the golden-phase config, for a fair picture
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "after_golden_set"))
            from config import apply_threshold
            preds = apply_threshold(preds, f.stem)
        except Exception:
            pass
        entries = [(io.image_path(pid),
                    gt[pid]["boxes"],
                    preds.get(pid, {}).get("boxes", []),
                    f"{pid} ({next(r['group'] for r in io.read_manifest() if int(r['photo_id']) == pid)})")
                   for pid in sample]
        out = out_dir / f"{f.stem}.png"
        visualize.contact_sheet(entries, out, title=f.stem)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
