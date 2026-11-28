"""Final contest submission file from the chosen model with frozen settings.

Usage: HF_HOME=/home/hf_cache python code/after_golden_set/make_submission.py <model> <images_dir> <out.jsonl>

Validates: every photo_id exactly once, boxes normalized, error null, file <= 5 MiB.
Technical errors are journaled (runs/submission_errors.log) and abort the submission —
the task forbids substituting an error with an empty box list.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import io
from lib.runner import sanitize
from config import CFG, apply_threshold


def main():
    model, images_dir, out_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    cfg = CFG[model]
    from lib.models import Detector
    overrides = {k: v for k, v in cfg.items() if k != "box_threshold"}  # threshold is a post-filter
    det = Detector(model, **overrides)

    images = sorted([p for p in images_dir.iterdir()
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    print(f"{len(images)} exam images from {images_dir}")

    raw, journal = {}, []
    for img_path in images:
        photo_id = int(img_path.stem)
        try:
            boxes = sanitize(det.predict(io.load_image(img_path)))
            raw[photo_id] = {"boxes": boxes, "error": None}
        except Exception as e:
            journal.append(f"{photo_id}: {type(e).__name__}: {e}")
            raw[photo_id] = {"boxes": [], "error": f"{type(e).__name__}: {e}"}

    if journal:
        log = io.RUNS_DIR / "submission_errors.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as f:
            f.write(f"--- {time.strftime('%Y-%m-%d %H:%M')} {model}\n" + "\n".join(journal) + "\n")
        print(f"ABORT: {len(journal)} technical errors — fix them before submitting (see {log})")
        sys.exit(1)

    final = apply_threshold(raw, model)
    with open(out_path, "w") as f:
        for photo_id in sorted(final):
            boxes, seen = [], set()
            for b in final[photo_id]["boxes"]:
                key = (round(b["left"], 6), round(b["top"], 6),
                       round(b["width"], 6), round(b["height"], 6))
                if key not in seen:  # exact duplicates are guaranteed FP — Qwen repeats boxes sometimes
                    seen.add(key)
                    boxes.append({"left": key[0], "top": key[1],
                                  "width": key[2], "height": key[3]})
            f.write(json.dumps({"photo_id": photo_id, "boxes": boxes, "error": None}) + "\n")

    size = out_path.stat().st_size
    assert size <= 5 * 1024 * 1024, f"file too large: {size}"
    ids = [json.loads(l)["photo_id"] for l in open(out_path)]
    assert len(ids) == len(set(ids)) == len(images), "photo_id mismatch"
    print(f"OK: {out_path} ({len(ids)} photos, {size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
