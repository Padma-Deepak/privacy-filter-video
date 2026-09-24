"""
tracking.py — assigns persistent track IDs to per-frame detections and adds
gap-filling and smoothing on top, WITHOUT ever gating redaction.

Non-negotiable (CLAUDE.md #1, privacy safety over speed): every raw
detection passed to ClassTracker.update() is included in that call's return
value, unconditionally. supervision.ByteTrack is used only as a best-effort
ID-assignment aid across frames — never as a filter on whether something
gets redacted.

This was a deliberate design choice, not an assumption: ByteTrack has an
internal, non-configurable confidence floor (confirmed empirically —
detections with confidence below roughly 0.1 are dropped entirely on their
first frame, regardless of the `track_activation_threshold` constructor
argument) that can silently drop a detection ByteTrack decides not to trust.
Relying on its return value as the source of truth for redaction would
violate the "redact on first sight" guarantee. So ClassTracker treats
ByteTrack purely as an oracle for "which past track does this box belong
to", with a local IoU-based fallback for anything it doesn't confirm, and
keeps its own gap-fill/smoothing state independent of ByteTrack's internal
lost-track handling.

One ClassTracker instance = one object class (faces, plates, screens, ...)
for ONE video-processing job. All state is instance attributes — never a
module-level tracker, since Flask may process multiple jobs concurrently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv

Box = tuple[int, int, int, int]  # x, y, w, h, pixel coordinates


@dataclass
class TrackedBox:
    track_id: int
    box: Box             # final redaction region: padded union(raw, smoothed) — always covers the raw detection
    source: str           # "detected" (a raw detection matched this frame) or "gap_fill"
    raw_box: "Box | None"  # the exact raw detection this frame, if any (None for gap_fill)


def _to_xyxy(box: Box) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return (float(x), float(y), float(x + w), float(y + h))


def _to_xywh(xyxy) -> Box:
    x1, y1, x2, y2 = xyxy
    return (int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1)))


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two (x, y, w, h) boxes."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def union_box(a: Box, b: Box) -> tuple[float, float, float, float]:
    """Smallest axis-aligned box containing both a and b."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    x2, y2 = max(ax2, bx2), max(ay2, by2)
    return (x1, y1, x2 - x1, y2 - y1)


def pad_box(box, pad_pct: float, frame_w: int, frame_h: int) -> Box:
    """Pad a box by pad_pct of its own size on each side, clamped to the frame."""
    x, y, w, h = box
    pad_x, pad_y = w * pad_pct, h * pad_pct
    x1, y1 = x - pad_x, y - pad_y
    x2, y2 = x + w + pad_x, y + h + pad_y
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(frame_w), x2), min(float(frame_h), y2)
    return (int(round(x1)), int(round(y1)), int(round(max(0.0, x2 - x1))), int(round(max(0.0, y2 - y1))))


class ClassTracker:
    """Tracks one object class across frames for one video-processing job."""

    def __init__(
        self,
        track_buffer: int = 30,
        smoothing_alpha: float = 0.5,
        pad_pct: float = 0.15,
        frame_rate: int = 30,
        extrapolate_velocity: bool = True,
        match_iou_threshold: float = 0.3,
    ):
        self._bytetrack = sv.ByteTrack(
            track_activation_threshold=0.0,
            minimum_consecutive_frames=1,
            lost_track_buffer=track_buffer,
            frame_rate=frame_rate,
        )
        self._track_buffer = track_buffer
        self._smoothing_alpha = smoothing_alpha
        self._pad_pct = pad_pct
        self._extrapolate_velocity = extrapolate_velocity
        self._match_iou_threshold = match_iou_threshold
        # track_id -> {"smoothed": Box, "raw": Box, "prev_raw": Box|None, "age": int}
        self._history: dict[int, dict] = {}
        self._next_fallback_id = -1

    def update(self, raw_boxes, frame_w: int, frame_h: int) -> list[TrackedBox]:
        """
        Advance the tracker by one frame.

        raw_boxes: this frame's raw detections for this class, (x, y, w, h).
        Returns one TrackedBox per raw detection (source="detected", always
        present, regardless of what the underlying tracker does with it) plus
        one TrackedBox per track that's within its gap-fill window but wasn't
        matched this frame (source="gap_fill").
        """
        raw_boxes = list(raw_boxes)
        assigned_ids = self._assign_ids(raw_boxes)

        results: list[TrackedBox] = []
        touched_ids = set()

        for raw_box, track_id in zip(raw_boxes, assigned_ids):
            touched_ids.add(track_id)
            prev = self._history.get(track_id)
            if prev is None:
                smoothed = tuple(float(v) for v in raw_box)
                prev_raw = None
            else:
                smoothed = self._ema(prev["smoothed"], raw_box)
                prev_raw = prev["raw"]

            self._history[track_id] = {
                "smoothed": smoothed,
                "raw": raw_box,
                "prev_raw": prev_raw,
                "age": 0,
            }

            # Smoothing must never shrink or lag the redaction region below
            # the raw detection: pad the UNION of raw and smoothed, not the
            # smoothed box alone. Padding only ever grows a box, so this
            # guarantees full raw-box coverage regardless of pad_pct.
            region = union_box(raw_box, smoothed)
            region = pad_box(region, self._pad_pct, frame_w, frame_h)
            results.append(TrackedBox(track_id=track_id, box=region, source="detected", raw_box=raw_box))

        for track_id in list(self._history.keys()):
            if track_id in touched_ids:
                continue
            state = self._history[track_id]
            state["age"] += 1
            if state["age"] > self._track_buffer:
                del self._history[track_id]
                continue

            box = state["smoothed"]
            if self._extrapolate_velocity and state["prev_raw"] is not None:
                vx = state["raw"][0] - state["prev_raw"][0]
                vy = state["raw"][1] - state["prev_raw"][1]
                x, y, w, h = box
                box = (x + vx * state["age"], y + vy * state["age"], w, h)

            region = pad_box(box, self._pad_pct, frame_w, frame_h)
            results.append(TrackedBox(track_id=track_id, box=region, source="gap_fill", raw_box=None))

        return results

    def _ema(self, prev_smoothed, raw: Box) -> tuple:
        a = self._smoothing_alpha
        return tuple(a * raw[i] + (1 - a) * prev_smoothed[i] for i in range(4))

    def _assign_ids(self, raw_boxes: list) -> list:
        """
        Best-effort persistent ID per raw box.

        Tries supervision.ByteTrack first (real motion-model-based
        matching). Anything ByteTrack doesn't confirm (its internal
        confidence floor, or a genuinely new/ambiguous box) falls back to
        our own IoU match against last-known positions, then a fresh
        negative ID if that also fails. IDs are never reused twice within
        the same frame.
        """
        if not raw_boxes:
            self._bytetrack.update_with_detections(sv.Detections.empty())
            return []

        xyxy = np.array([_to_xyxy(b) for b in raw_boxes], dtype=np.float32)
        confidence = np.ones(len(raw_boxes), dtype=np.float32)
        class_id = np.zeros(len(raw_boxes), dtype=int)
        detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        tracked = self._bytetrack.update_with_detections(detections)

        ids: list = [None] * len(raw_boxes)

        pairs = []
        for ri, raw_box in enumerate(raw_boxes):
            for ti in range(len(tracked)):
                score = iou(raw_box, _to_xywh(tracked.xyxy[ti]))
                if score >= self._match_iou_threshold:
                    pairs.append((score, ri, ti))
        pairs.sort(key=lambda p: p[0], reverse=True)
        matched_raw, used_tracked = set(), set()
        for score, ri, ti in pairs:
            if ri in matched_raw or ti in used_tracked:
                continue
            matched_raw.add(ri)
            used_tracked.add(ti)
            ids[ri] = int(tracked.tracker_id[ti])

        assigned_this_frame = {i for i in ids if i is not None}
        for ri, raw_box in enumerate(raw_boxes):
            if ids[ri] is not None:
                continue
            best_id, best_iou = None, self._match_iou_threshold
            for track_id, state in self._history.items():
                if track_id in assigned_this_frame:
                    continue
                score = iou(raw_box, state["raw"])
                if score > best_iou:
                    best_id, best_iou = track_id, score
            if best_id is not None:
                ids[ri] = best_id
            else:
                ids[ri] = self._next_fallback_id
                self._next_fallback_id -= 1
            assigned_this_frame.add(ids[ri])

        return ids
