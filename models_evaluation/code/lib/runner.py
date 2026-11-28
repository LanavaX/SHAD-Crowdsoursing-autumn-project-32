"""Shared inference engine: run detectors over a photo set, save contest JSONL + timing."""
import csv
import time
from pathlib import Path

from lib import io


def sanitize(boxes):
    """Clamp to [0,1], drop degenerate boxes, keep score if present."""
    out = []
    for b in boxes:
        l = min(max(b["left"], 0.0), 1.0)
        t = min(max(b["top"], 0.0), 1.0)
        w = min(max(b["width"], 0.0), 1.0 - l)
        h = min(max(b["height"], 0.0), 1.0 - t)
        if w > 1e-6 and h > 1e-6:
            r = {"left": l, "top": t, "width": w, "height": h}
            if "score" in b:
                r["score"] = b["score"]
            out.append(r)
    return out


def run_dataset(names, photos, out_dir):
    """photos: [(photo_id, image_path)]; writes out_dir/<name>.jsonl and out_dir/timing.csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from lib.models import Detector

    timing_rows = []
    for name in names:
        print(f"=== {name} ===", flush=True)
        det = Detector(name)
        records, per_img, errors = {}, [], 0
        for photo_id, img_path in photos:
            try:
                img = io.load_image(img_path)
                t0 = time.time()
                boxes = sanitize(det.predict(img))
                per_img.append(time.time() - t0)
                records[photo_id] = {"boxes": boxes, "error": None}
            except Exception as e:
                errors += 1
                records[photo_id] = {"boxes": [], "error": f"{type(e).__name__}: {e}"}
        io.write_jsonl(out_dir / f"{name}.jsonl", records)
        timing_rows.append({
            "model": name, "n_photos": len(photos), "errors": errors,
            "total_s": round(sum(per_img), 1), "per_img_s": round(sum(per_img) / max(len(per_img), 1), 3),
        })
        print(f"{name}: done, {errors} errors, {timing_rows[-1]['per_img_s']}s/img", flush=True)
        del det

    # merge with existing timing.csv so per-model reruns don't clobber other rows
    timing_path = out_dir / "timing.csv"
    old = {}
    if timing_path.exists():
        with open(timing_path) as f:
            old = {r["model"]: r for r in csv.DictReader(f)}
    for r in timing_rows:
        old[r["model"]] = r
    with open(timing_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "n_photos", "errors", "total_s", "per_img_s"])
        w.writeheader()
        for r in old.values():
            w.writerow(r)
