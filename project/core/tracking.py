"""
tracking.py — assigns persistent track IDs to per-frame detections and adds
gap-filling and smoothing on top, WITHOUT ever gating redaction.

Non-negotiable (CLAUDE.md #1, privacy safety over speed): every raw
detection passed to ClassTracker.update() is included in that call's return
value, unconditionally. supervision.ByteTrack is used only as a best-effort
aid for seeding brand-new track IDs — never as a filter on whether something
gets redacted, and never as the authority on whether an existing track
continues (see below).

Two things were verified empirically here, not assumed:

1. ByteTrack has an internal, non-configurable confidence floor (detections
   below roughly 0.1 confidence are dropped entirely on their first frame,
   regardless of the `track_activation_threshold` constructor argument).
   Relying on its return value as the source of truth for redaction would
   violate the "redact on first sight" guarantee.

2. ByteTrack's OWN frame-to-frame continuity matching uses a much stricter
   threshold by default (`minimum_matching_threshold=0.8`) than what this
   module needs for stable IDs. Under fast or erratic motion, ByteTrack can
   lose its own internal continuity for a still-visible face and silently
   hand out a NEW internal id for it — which would make a track's id churn
   for reasons that have nothing to do with the face actually changing.
   So id continuity here is decided by OUR OWN velocity-aware IoU match
   against each track's last-known position FIRST; ByteTrack's suggestion is
   only used to seed an id for a box that doesn't match anything already in
   our own history (i.e. a genuinely new appearance, or the fallback case
   from point 1). This way an id can only ever change when the box's
   position genuinely stops overlapping where we last saw it — not because
   of what ByteTrack's internal bookkeeping decided to do on its own.

The numeric ids described above (ByteTrack-sourced, always positive; or our
own fallback, always negative — so the two spaces can never collide) are
purely an internal matching key. TrackedBox.track_id is a SEPARATE,
per-instance renumbering (1, 2, 3, ... in order of first appearance) applied
on top, because ByteTrack's internal counter is a process-wide class
attribute (STrack._external_count) shared across every ClassTracker in the
process — without renumbering, a job's first track could come out as id 743
depending on what ran before it. Phase 3's click-to-select needs ids that
are both stable across a clip AND start fresh per job.

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
    track_id: int          # stable, per-job renumbered id (1, 2, 3, ... in order of first appearance)
    box: Box                # final redaction region: padded union(raw, smoothed) — always covers the raw detection
    source: str              # "detected" (a raw detection matched this frame) or "gap_fill"
    raw_box: "Box | None"    # the exact raw detection this frame, if any (None for gap_fill)
    tracker_box: Box         # the tracker's own unpadded estimate: smoothed box (detected) or extrapolated
                              # last-known box (gap_fill) — before the raw-box union and padding are applied.
                              # Exposed for scripts/debug_tracking.py; redaction always uses `box`, never this.


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
        # internal_id -> {"smoothed": Box, "raw": Box, "prev_raw": Box|None, "age": int}
        self._history: dict[int, dict] = {}
        self._next_fallback_id = -1
        # internal_id -> external_id (1, 2, 3, ... in order of first appearance)
        self._external_ids: dict[int, int] = {}
        self._next_external_id = 1

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

        for raw_box, internal_id in zip(raw_boxes, assigned_ids):
            touched_ids.add(internal_id)
            prev = self._history.get(internal_id)
            if prev is None:
                smoothed = tuple(float(v) for v in raw_box)
                prev_raw = None
            else:
                smoothed = self._ema(prev["smoothed"], raw_box)
                prev_raw = prev["raw"]

            self._history[internal_id] = {
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
            smoothed_int = tuple(int(round(v)) for v in smoothed)
            results.append(TrackedBox(
                track_id=self._external_id(internal_id), box=region, source="detected",
                raw_box=raw_box, tracker_box=smoothed_int,
            ))

        for internal_id in list(self._history.keys()):
            if internal_id in touched_ids:
                continue
            state = self._history[internal_id]
            state["age"] += 1
            if state["age"] > self._track_buffer:
                del self._history[internal_id]
                continue

            box = state["smoothed"]
            if self._extrapolate_velocity and state["prev_raw"] is not None:
                vx = state["raw"][0] - state["prev_raw"][0]
                vy = state["raw"][1] - state["prev_raw"][1]
                x, y, w, h = box
                box = (x + vx * state["age"], y + vy * state["age"], w, h)

            region = pad_box(box, self._pad_pct, frame_w, frame_h)
            box_int = tuple(int(round(v)) for v in box)
            results.append(TrackedBox(
                track_id=self._external_id(internal_id), box=region, source="gap_fill",
                raw_box=None, tracker_box=box_int,
            ))

        return results

    def _external_id(self, internal_id: int) -> int:
        """Renumber an internal id to 1, 2, 3, ... in order of first appearance."""
        if internal_id not in self._external_ids:
            self._external_ids[internal_id] = self._next_external_id
            self._next_external_id += 1
        return self._external_ids[internal_id]

    def _ema(self, prev_smoothed, raw: Box) -> tuple:
        a = self._smoothing_alpha
        return tuple(a * raw[i] + (1 - a) * prev_smoothed[i] for i in range(4))

    def _predict(self, internal_id: int, state: dict) -> Box:
        """One-frame-ahead predicted position for continuity matching."""
        box = state["smoothed"]
        if self._extrapolate_velocity and state["prev_raw"] is not None:
            vx = state["raw"][0] - state["prev_raw"][0]
            vy = state["raw"][1] - state["prev_raw"][1]
            x, y, w, h = box
            return (x + vx, y + vy, w, h)
        return box

    def _assign_ids(self, raw_boxes: list) -> list:
        """
        Persistent internal id per raw box.

        Priority 1 — OUR OWN continuity: match each raw box against every
        existing track's predicted (velocity-extrapolated) position. This is
        authoritative for id stability; see the module docstring for why it
        must not defer to ByteTrack's own internal matching.

        Priority 2 — ByteTrack's suggestion, but ONLY to seed a box that
        didn't match anything in our own history (a genuinely new box, or
        one ByteTrack's confidence floor dropped last frame too) — and only
        if that suggested id isn't already claimed by something else this
        frame, so it can never steal an unrelated track's identity.

        Priority 3 — a fresh negative fallback id.

        No id is ever reused twice within the same frame.
        """
        if not raw_boxes:
            self._bytetrack.update_with_detections(sv.Detections.empty())
            return []

        # ByteTrack still runs every frame (keeps its own internal state
        # consistent, and gives us candidate ids for new tracks) but its
        # output is consulted only after our own continuity match fails.
        xyxy = np.array([_to_xyxy(b) for b in raw_boxes], dtype=np.float32)
        confidence = np.ones(len(raw_boxes), dtype=np.float32)
        class_id = np.zeros(len(raw_boxes), dtype=int)
        detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        tracked = self._bytetrack.update_with_detections(detections)

        bytetrack_suggestion: list = [None] * len(raw_boxes)
        bt_pairs = []
        for ri, raw_box in enumerate(raw_boxes):
            for ti in range(len(tracked)):
                score = iou(raw_box, _to_xywh(tracked.xyxy[ti]))
                if score >= self._match_iou_threshold:
                    bt_pairs.append((score, ri, ti))
        bt_pairs.sort(key=lambda p: p[0], reverse=True)
        matched_raw, used_tracked = set(), set()
        for score, ri, ti in bt_pairs:
            if ri in matched_raw or ti in used_tracked:
                continue
            matched_raw.add(ri)
            used_tracked.add(ti)
            bytetrack_suggestion[ri] = int(tracked.tracker_id[ti])

        # Priority 1: our own continuity match against predicted positions.
        ids: list = [None] * len(raw_boxes)
        assigned_this_frame = set()

        predicted = {tid: self._predict(tid, state) for tid, state in self._history.items()}
        own_pairs = []
        for ri, raw_box in enumerate(raw_boxes):
            for internal_id, pred_box in predicted.items():
                score = iou(raw_box, pred_box)
                if score >= self._match_iou_threshold:
                    own_pairs.append((score, ri, internal_id))
        own_pairs.sort(key=lambda p: p[0], reverse=True)
        matched_raw2, used_ids2 = set(), set()
        for score, ri, internal_id in own_pairs:
            if ri in matched_raw2 or internal_id in used_ids2:
                continue
            matched_raw2.add(ri)
            used_ids2.add(internal_id)
            ids[ri] = internal_id
            assigned_this_frame.add(internal_id)

        # Priority 2 & 3: anything left unmatched by our own history.
        for ri, raw_box in enumerate(raw_boxes):
            if ids[ri] is not None:
                continue
            suggestion = bytetrack_suggestion[ri]
            if (
                suggestion is not None
                and suggestion not in assigned_this_frame
                and suggestion not in self._history
            ):
                # Only trust a brand-new ByteTrack id (nothing in our own
                # history already owns it) — otherwise we'd risk silently
                # stealing another track's identity.
                ids[ri] = suggestion
            else:
                ids[ri] = self._next_fallback_id
                self._next_fallback_id -= 1
            assigned_this_frame.add(ids[ri])

        return ids
