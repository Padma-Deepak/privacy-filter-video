"""
metrics.py — detection metrics used by eval/run_baseline.py and later phases.

Definitions follow CLAUDE.md:
  - IoU match threshold: 0.5
  - Face recall: matched ground-truth boxes / all ground-truth boxes
  - Size buckets: based on max(box width, box height) in pixels —
      small  : < 32px
      medium : 32px to 96px
      large  : > 96px
    (Matches the small/medium/large convention used in COCO-style detection
    evaluation, applied here to face box size instead of area.)

All boxes are (x, y, w, h) in pixel coordinates. No I/O, no third-party
dependencies — kept pure so it's cheap to unit test with synthetic boxes.
"""

from __future__ import annotations

Box = tuple[float, float, float, float]

SIZE_BUCKETS = ("small", "medium", "large")
_SMALL_MAX = 32.0
_MEDIUM_MAX = 96.0


def iou(box_a: Box, box_b: Box) -> float:
    """Intersection-over-union of two (x, y, w, h) boxes."""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def size_bucket(box: Box) -> str:
    """Bucket a box by its longer side, per the thresholds documented above."""
    _, _, w, h = box
    longer = max(w, h)
    if longer < _SMALL_MAX:
        return "small"
    if longer < _MEDIUM_MAX:
        return "medium"
    return "large"


def match_boxes(
    gt_boxes: list[Box], pred_boxes: list[Box], iou_threshold: float = 0.5
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Greedily match predictions to ground truth by descending IoU.

    Returns (matches, unmatched_gt_indices, unmatched_pred_indices), where
    matches is a list of (gt_index, pred_index) pairs. Each gt and each
    pred is used in at most one match. Detectors here (Haar, contour, YOLO)
    don't all expose a comparable confidence score, so ranking by IoU is the
    same greedy strategy the standard mAP matching would use when scores are
    tied.
    """
    candidates = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            score = iou(gt, pred)
            if score >= iou_threshold:
                candidates.append((score, gi, pi))
    candidates.sort(key=lambda c: c[0], reverse=True)

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matches.append((gi, pi))

    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    unmatched_pred = [i for i in range(len(pred_boxes)) if i not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


def precision_recall(
    all_gt: list[list[Box]], all_pred: list[list[Box]], iou_threshold: float = 0.5
) -> dict:
    """
    Aggregate precision/recall (IoU >= iou_threshold) across a dataset.

    all_gt and all_pred are parallel lists, one entry (a list of boxes) per
    image. Returns a dict with tp, fp, fn, precision, recall.
    """
    if len(all_gt) != len(all_pred):
        raise ValueError("all_gt and all_pred must have the same number of images")

    tp = fp = fn = 0
    for gt_boxes, pred_boxes in zip(all_gt, all_pred):
        matches, unmatched_gt, unmatched_pred = match_boxes(gt_boxes, pred_boxes, iou_threshold)
        tp += len(matches)
        fn += len(unmatched_gt)
        fp += len(unmatched_pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}


def recall_by_size_bucket(
    all_gt: list[list[Box]], all_pred: list[list[Box]], iou_threshold: float = 0.5
) -> dict:
    """
    Recall (IoU >= iou_threshold) broken down by ground-truth box size bucket.

    Returns {"small": {"recall": .., "n_gt": ..}, "medium": {...}, "large": {...}}.
    A bucket with zero ground-truth boxes reports recall as NaN rather than 0,
    so an empty bucket is never mistaken for a detector failing on that bucket.
    """
    if len(all_gt) != len(all_pred):
        raise ValueError("all_gt and all_pred must have the same number of images")

    hits = {b: 0 for b in SIZE_BUCKETS}
    totals = {b: 0 for b in SIZE_BUCKETS}

    for gt_boxes, pred_boxes in zip(all_gt, all_pred):
        matches, unmatched_gt, _ = match_boxes(gt_boxes, pred_boxes, iou_threshold)
        matched_gt_idx = {gi for gi, _ in matches}
        for gi, gt in enumerate(gt_boxes):
            bucket = size_bucket(gt)
            totals[bucket] += 1
            if gi in matched_gt_idx:
                hits[bucket] += 1

    return {
        b: {
            "recall": (hits[b] / totals[b]) if totals[b] > 0 else float("nan"),
            "n_gt": totals[b],
        }
        for b in SIZE_BUCKETS
    }
