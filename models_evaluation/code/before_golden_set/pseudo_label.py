"""Pseudo ground truth for the smoke set: consensus of models.

Per photo, boxes from all models are clustered greedily (IoU >= 0.5); a cluster
becomes a pseudo-GT box if >= ceil(n_models/2) distinct models contributed to it;
the box is the coordinate-wise median of members. Debug-only — never a substitute
for the real golden set (disclosed in the report).

Usage: python code/before_golden_set/pseudo_label.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import io
from lib.metrics import iou_matrix

RUNS = io.RUNS_DIR / "smoke"


def main():
    model_files = sorted(RUNS.glob("*.jsonl"))
    assert model_files, f"no prediction files in {RUNS}; run run_smoke.py first"
    preds = {f.stem: io.read_jsonl(f) for f in model_files}
    all_ids = sorted({pid for p in preds.values() for pid in p})
    n_models = len(preds)
    quorum = -(-n_models // 2)  # ceil

    out, stats = {}, {"photos": len(all_ids), "boxes": 0, "rejected_clusters": 0}
    for pid in all_ids:
        voting = [m for m in preds if pid in preds[m] and not preds[m][pid].get("error")]
        # all boxes of this photo as (score, box, model)
        items = []
        for m in voting:
            for b in preds[m][pid]["boxes"]:
                items.append((b.get("score", 1.0), [b["left"], b["top"], b["width"], b["height"]], m))
        items.sort(key=lambda x: -x[0])
        clusters = []  # list of dict(members=[box...], models=set())
        for score, box, m in items:
            for cl in clusters:
                if any(iou_matrix([box], [mb])[0, 0] >= 0.5 for mb in cl["members"]):
                    cl["members"].append(box)
                    cl["models"].add(m)
                    break
            else:
                clusters.append({"members": [box], "models": {m}})
        boxes = []
        for cl in clusters:
            if len(cl["models"]) >= quorum:
                med = np.median(np.array(cl["members"]), axis=0)
                boxes.append({"left": med[0], "top": med[1], "width": med[2], "height": med[3]})
            else:
                stats["rejected_clusters"] += 1
        out[pid] = {"boxes": boxes, "error": None}
        stats["boxes"] += len(boxes)

    io.write_jsonl(io.SMOKE_DIR / "pseudo_gt.jsonl", out)
    print(f"models={n_models} quorum={quorum} {stats}")
    print(f"-> {io.SMOKE_DIR / 'pseudo_gt.jsonl'}")


if __name__ == "__main__":
    main()
