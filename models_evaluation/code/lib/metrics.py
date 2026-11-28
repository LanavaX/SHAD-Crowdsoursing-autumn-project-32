"""IoU, maximum bipartite box matching and micro-F1 — mirrors the contest metric.

Contest rule: per image, the maximum number of GT-prediction pairs with IoU >= 0.5,
each box in at most one pair; TP/FP/FN summed over the whole dataset, then
F1 = 2*TP / (2*TP + FP + FN).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def _lists(boxes):
    """Accept [{"left":..}, ...] dicts or [[l,t,w,h], ...] lists -> list of lists."""
    if boxes and isinstance(boxes[0], dict):
        return [[b["left"], b["top"], b["width"], b["height"]] for b in boxes]
    return boxes


def _xyxy(boxes):
    """[left, top, width, height] -> (n, 4) array of [x1, y1, x2, y2]."""
    b = np.asarray(_lists(boxes), dtype=float)
    if b.size == 0:
        return b.reshape(0, 4)
    out = np.empty_like(b)
    out[:, 0] = b[:, 0]
    out[:, 1] = b[:, 1]
    out[:, 2] = b[:, 0] + b[:, 2]
    out[:, 3] = b[:, 1] + b[:, 3]
    return out


def iou_matrix(gt_boxes, pred_boxes):
    """(len(gt), len(pred)) pairwise IoU between two lists of [l,t,w,h] boxes."""
    gt, pr = _xyxy(gt_boxes), _xyxy(pred_boxes)
    if len(gt) == 0 or len(pr) == 0:
        return np.zeros((len(gt), len(pr)))
    ix1 = np.maximum(gt[:, None, 0], pr[None, :, 0])
    iy1 = np.maximum(gt[:, None, 1], pr[None, :, 1])
    ix2 = np.minimum(gt[:, None, 2], pr[None, :, 2])
    iy2 = np.minimum(gt[:, None, 3], pr[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_g = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    area_p = (pr[:, 2] - pr[:, 0]) * (pr[:, 3] - pr[:, 1])
    union = area_g[:, None] + area_p[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_counts(gt_boxes, pred_boxes, iou_thr):
    """Maximum-cardinality matching of pairs with IoU >= iou_thr -> (tp, fp, fn)."""
    m = iou_matrix(gt_boxes, pred_boxes)
    if m.size == 0:
        return 0, len(pred_boxes), len(gt_boxes)
    rows, cols = linear_sum_assignment(-m)
    tp = int(sum(m[r, c] >= iou_thr for r, c in zip(rows, cols)))
    return tp, len(pred_boxes) - tp, len(gt_boxes) - tp


def evaluate(gt_records, pred_records, iou_thr=0.5):
    """gt/pred: {photo_id: {"boxes": [...], "error": str|None}}, normalized l,t,w,h.

    Photos present in gt but missing from pred count all their GT boxes as FN.
    A pred record with a non-null error counts as an error and as no boxes.
    """
    tp = fp = fn = errors = fp_on_empty = 0
    for photo_id, gt in gt_records.items():
        pred = pred_records.get(photo_id)
        if pred is None:
            fn += len(gt["boxes"])
            continue
        if pred.get("error"):
            errors += 1
        t, f, n = match_counts(gt["boxes"], pred["boxes"], iou_thr)
        tp += t
        fp += f
        fn += n
        if not gt["boxes"] and pred["boxes"]:
            fp_on_empty += len(pred["boxes"])
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1,
        "fp_on_empty": fp_on_empty, "errors": errors,
        "n_photos": len(gt_records),
    }
