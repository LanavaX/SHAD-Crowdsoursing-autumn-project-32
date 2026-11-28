"""Draw submission boxes on exam images -> framed_test_for_user/ (user-side visual check).

with_boxes/{photo_id}.jpg — фото с предсказанными рамками (красные)
empty/{photo_id}.jpg — фото, где модель сказала «бабочек нет»

Usage: python code/after_golden_set/frame_exam.py
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import io

OUT = io.PROJECT_ROOT / "framed_test_for_user"


def main():
    sub = {json.loads(l)["photo_id"]: json.loads(l)["boxes"]
           for l in open(io.PROJECT_ROOT / "submission.jsonl")}
    n_boxed = n_empty = 0
    for pid, boxes in sub.items():
        img = io.load_image(io.image_path(pid, io.PROJECT_ROOT / "data" / "exam"))
        img.thumbnail((1200, 1200))
        if boxes:
            d = ImageDraw.Draw(img)
            w_line = max(3, img.width // 300)
            for b in boxes:
                x, y = b["left"] * img.width, b["top"] * img.height
                d.rectangle([x, y, x + b["width"] * img.width, y + b["height"] * img.height],
                            outline=(255, 0, 0), width=w_line)
        dest = OUT / ("with_boxes" if boxes else "empty")
        dest.mkdir(parents=True, exist_ok=True)
        img.save(dest / f"{pid}.jpg", quality=88)
        n_boxed += bool(boxes)
        n_empty += not boxes
    print(f"framed_test_for_user: {n_boxed} с рамками, {n_empty} пустых")


if __name__ == "__main__":
    main()
