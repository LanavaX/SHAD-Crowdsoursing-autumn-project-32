"""Synthetic-box tests for the contest metric. No images needed."""
from lib.metrics import evaluate, iou_matrix, match_counts


def test_iou_known_values():
    # identical boxes -> 1; disjoint -> 0; half overlap of two 1x1 in a 2x1 canvas
    assert abs(iou_matrix([[0, 0, 1, 1]], [[0, 0, 1, 1]])[0, 0] - 1.0) < 1e-9
    assert iou_matrix([[0, 0, 1, 1]], [[2, 2, 1, 1]])[0, 0] == 0.0
    # two 1x1 squares shifted by 0.5: inter 0.25, union 1.75
    assert abs(iou_matrix([[0, 0, 1, 1]], [[0.5, 0.5, 1, 1]])[0, 0] - 0.25 / 1.75) < 1e-9


def test_perfect_match_is_f1_one():
    gt = {1: {"boxes": [[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.3, 0.3]], "error": None}}
    pred = {1: {"boxes": [[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.3, 0.3]], "error": None}}
    r = evaluate(gt, pred)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 0, 0)
    assert abs(r["f1"] - 1.0) < 1e-9


def test_duplicate_predictions_counted_as_fp():
    gt = {1: {"boxes": [[0.1, 0.1, 0.5, 0.5]], "error": None}}
    pred = {1: {"boxes": [[0.1, 0.1, 0.5, 0.5], [0.11, 0.11, 0.5, 0.5]], "error": None}}
    r = evaluate(gt, pred)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 0)
    assert abs(r["f1"] - 2 / 3) < 1e-9


def test_prediction_on_empty_photo_is_pure_fp():
    gt = {1: {"boxes": [], "error": None}, 2: {"boxes": [[0.2, 0.2, 0.4, 0.4]], "error": None}}
    pred = {1: {"boxes": [[0.5, 0.5, 0.2, 0.2]], "error": None}, 2: {"boxes": [], "error": None}}
    r = evaluate(gt, pred)
    assert r["fp_on_empty"] == 1
    assert (r["tp"], r["fp"], r["fn"]) == (0, 1, 1)


def test_matching_is_maximum_not_greedy():
    # pred1 (full frame) has IoU exactly 0.5 with both strips; pred2 matches only gt1.
    # Greedy pairing pred1->gt1 leaves pred2 unmatched; maximum matching finds 2 pairs.
    gt = [[0, 0, 0.5, 1], [0.5, 0, 0.5, 1]]
    preds = [[0, 0, 1, 1], [0, 0, 0.5, 1]]
    assert match_counts(gt, preds, 0.5) == (2, 0, 0)


def test_below_threshold_is_not_matched():
    tp, fp, fn = match_counts([[0, 0, 1, 1]], [[0.6, 0.6, 1, 1]], 0.5)  # IoU ~0.05
    assert (tp, fp, fn) == (0, 1, 1)


def test_missing_photo_in_pred_counts_as_fn():
    gt = {1: {"boxes": [[0.1, 0.1, 0.3, 0.3]], "error": None}}
    assert evaluate(gt, {})["fn"] == 1


def test_error_record_counts_as_error_and_no_boxes():
    gt = {1: {"boxes": [[0.1, 0.1, 0.3, 0.3]], "error": None}}
    pred = {1: {"boxes": [], "error": "parse failure"}}
    r = evaluate(gt, pred)
    assert r["errors"] == 1 and r["fn"] == 1 and r["f1"] == 0.0
