"""Unit tests for eval/metrics.py — synthetic boxes only, no real images."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

from metrics import iou, match_boxes, precision_recall, recall_by_size_bucket, size_bucket


def test_iou_identical_boxes_is_one():
    box = (10, 10, 20, 20)
    assert iou(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_iou_known_partial_overlap():
    # Two 10x10 boxes overlapping in a 5x10 strip: intersection 50, union 150.
    a = (0, 0, 10, 10)
    b = (5, 0, 10, 10)
    assert math.isclose(iou(a, b), 50 / 150, rel_tol=1e-9)


def test_size_bucket_thresholds():
    assert size_bucket((0, 0, 20, 20)) == "small"
    assert size_bucket((0, 0, 32, 10)) == "medium"
    assert size_bucket((0, 0, 50, 60)) == "medium"
    assert size_bucket((0, 0, 96, 10)) == "large"
    assert size_bucket((0, 0, 200, 200)) == "large"


def test_match_boxes_perfect_match():
    gt = [(0, 0, 10, 10), (100, 100, 10, 10)]
    pred = [(0, 0, 10, 10), (100, 100, 10, 10)]
    matches, unmatched_gt, unmatched_pred = match_boxes(gt, pred)
    assert len(matches) == 2
    assert unmatched_gt == []
    assert unmatched_pred == []


def test_match_boxes_one_miss_one_false_positive():
    gt = [(0, 0, 10, 10), (100, 100, 10, 10)]
    pred = [(0, 0, 10, 10), (500, 500, 10, 10)]
    matches, unmatched_gt, unmatched_pred = match_boxes(gt, pred)
    assert len(matches) == 1
    assert unmatched_gt == [1]
    assert unmatched_pred == [1]


def test_match_boxes_does_not_double_assign():
    # One prediction overlaps both GT boxes; it can only match one.
    gt = [(0, 0, 10, 10), (2, 2, 10, 10)]
    pred = [(1, 1, 10, 10)]
    matches, unmatched_gt, unmatched_pred = match_boxes(gt, pred, iou_threshold=0.3)
    assert len(matches) == 1
    assert len(unmatched_gt) == 1
    assert unmatched_pred == []


def test_precision_recall_perfect_dataset():
    all_gt = [[(0, 0, 10, 10)], [(5, 5, 10, 10)]]
    all_pred = [[(0, 0, 10, 10)], [(5, 5, 10, 10)]]
    result = precision_recall(all_gt, all_pred)
    assert result == {"tp": 2, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0}


def test_precision_recall_with_miss_and_false_positive():
    all_gt = [[(0, 0, 10, 10), (100, 100, 10, 10)]]
    all_pred = [[(0, 0, 10, 10), (500, 500, 10, 10)]]
    result = precision_recall(all_gt, all_pred)
    assert result["tp"] == 1
    assert result["fn"] == 1
    assert result["fp"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5


def test_precision_recall_no_predictions_no_gt_is_nan():
    result = precision_recall([[]], [[]])
    assert math.isnan(result["precision"])
    assert math.isnan(result["recall"])


def test_precision_recall_rejects_mismatched_lengths():
    try:
        precision_recall([[]], [[], []])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_recall_by_size_bucket():
    all_gt = [[(0, 0, 10, 10), (0, 0, 200, 200)]]  # one small, one large
    all_pred = [[(0, 0, 10, 10)]]  # only the small one is detected
    result = recall_by_size_bucket(all_gt, all_pred)
    assert result["small"]["n_gt"] == 1
    assert result["small"]["recall"] == 1.0
    assert result["large"]["n_gt"] == 1
    assert result["large"]["recall"] == 0.0
    assert result["medium"]["n_gt"] == 0
    assert math.isnan(result["medium"]["recall"])
