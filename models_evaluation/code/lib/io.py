"""Contest JSONL I/O and project paths.

Record format (contest): {"photo_id": int, "boxes": [{"left","top","width","height"}, ...], "error": null}
Coordinates normalized 0..1. Internal records may carry an extra "score" per box;
make_submission.py strips it before the contest upload.
"""
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
# Smoke data lives inside the disposable before_golden_set folder; the whole
# folder is dropped once the real golden set arrives.
SMOKE_DIR = PROJECT_ROOT / "code" / "before_golden_set" / "data"
GOLDEN_DIR = DATA_DIR / "golden"
RUNS_DIR = PROJECT_ROOT / "runs"


def read_jsonl(path):
    """-> {photo_id: {"boxes": [...], "error": str|None}}"""
    records = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records[r["photo_id"]] = {"boxes": r.get("boxes", []), "error": r.get("error")}
    return records


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for photo_id in sorted(records):
            r = records[photo_id]
            f.write(json.dumps({
                "photo_id": photo_id,
                "boxes": r["boxes"],
                "error": r.get("error"),
            }) + "\n")


def read_manifest(path=None):
    with open(path or SMOKE_DIR / "manifest.csv") as f:
        return list(csv.DictReader(f))


def image_path(photo_id, root=SMOKE_DIR):
    """Find <photo_id>.<jpg|jpeg|png> under root/images/ or directly under root/."""
    root = Path(root)
    for ext in (".jpg", ".jpeg", ".png"):
        for p in (root / "images" / f"{photo_id}{ext}", root / f"{photo_id}{ext}"):
            if p.exists():
                return p
    raise FileNotFoundError(f"image for photo_id={photo_id} under {root}")


def load_image(path):
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")
