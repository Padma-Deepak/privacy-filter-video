"""
Tests for project/core/tracking.py.

Synthetic box sequences only — no real video/detector involved. Covers the
Stage 2 conditions: unconditional redaction of raw detections (first frame
and mid-clip), smoothed region always covering the raw box, gap-fill with a
configurable buffer and velocity extrapolation, per-class tracker isolation,
and no cross-job state leakage.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))

from core.tracking import ClassTracker, pad_box, union_box  # noqa: E402

FRAME_W, FRAME_H = 1000, 1000


def _contains(outer, inner) -> bool:
    ox1, oy1, ox2, oy2 = outer[0], outer[1], outer[0] + outer[2], outer[1] + outer[3]
    ix1, iy1, ix2, iy2 = inner[0], inner[1], inner[0] + inner[2], inner[1] + inner[3]
    return ox1 <= ix1 and oy1 <= iy1 and ox2 >= ix2 and oy2 >= iy2


def _by_source(results, source):
    return [r for r in results if r.source == source]


# ── Condition 1: redaction is never gated by the tracker ────────────────────

def test_face_on_very_first_frame_is_redacted_immediately():
    tracker = ClassTracker()
    results = tracker.update([(100, 100, 50, 50)], FRAME_W, FRAME_H)

    detected = _by_source(results, "detected")
    assert len(detected) == 1
    assert detected[0].raw_box == (100, 100, 50, 50)
    assert _contains(detected[0].box, detected[0].raw_box)


def test_face_appearing_mid_clip_is_redacted_on_its_first_detected_frame():
    tracker = ClassTracker()
    for _ in range(5):
        results = tracker.update([], FRAME_W, FRAME_H)
        assert results == []

    results = tracker.update([(200, 200, 40, 40)], FRAME_W, FRAME_H)
    detected = _by_source(results, "detected")
    assert len(detected) == 1
    assert detected[0].raw_box == (200, 200, 40, 40)
    assert _contains(detected[0].box, detected[0].raw_box)


def test_every_raw_box_appears_every_frame_regardless_of_count():
    tracker = ClassTracker()
    boxes = [(10, 10, 20, 20), (500, 500, 30, 30), (800, 100, 25, 25)]
    results = tracker.update(boxes, FRAME_W, FRAME_H)
    detected = _by_source(results, "detected")
    assert len(detected) == len(boxes)
    assert {r.raw_box for r in detected} == set(boxes)


# ── ID stability: a continuously-visible face keeps ONE id (review round 2) ─
#
# Our own velocity-aware continuity match against last-known position is
# authoritative for id assignment; ByteTrack's suggestion only seeds a
# brand-new id when nothing in our own history matches. This is what
# guarantees stability even when ByteTrack's own internal continuity (a much
# stricter matching threshold, 0.8 by default vs our 0.3) would have lost
# the track and silently started a new internal id for it — see the module
# docstring.

def test_continuously_visible_face_keeps_one_stable_id_across_a_clip():
    tracker = ClassTracker(smoothing_alpha=0.4)
    ids = []
    x = 100
    for _ in range(60):
        results = tracker.update([(x, 200, 40, 40)], FRAME_W, FRAME_H)
        ids.append(_by_source(results, "detected")[0].track_id)
        x += 5  # steady, moderate motion across the whole clip
    assert len(set(ids)) == 1


def test_id_stays_stable_even_when_bytetrack_loses_internal_continuity(monkeypatch):
    """
    Directly forces the scenario the id-stability bug came from: ByteTrack
    fails to link a frame's detection to its own internal track (simulated
    here by making update_with_detections return nothing, standing in for
    ByteTrack's stricter 0.8 matching threshold losing continuity under
    motion, or its confidence floor dropping the detection). Our own
    continuity match must still recognize the box as the same face — via its
    last-known position — and keep the same external id regardless.
    """
    tracker = ClassTracker(smoothing_alpha=0.4)
    real_update = tracker._bytetrack.update_with_detections

    r1 = tracker.update([(100, 100, 40, 40)], FRAME_W, FRAME_H)
    id_before = _by_source(r1, "detected")[0].track_id

    import supervision as sv
    monkeypatch.setattr(tracker._bytetrack, "update_with_detections", lambda dets: sv.Detections.empty())
    r2 = tracker.update([(108, 100, 40, 40)], FRAME_W, FRAME_H)  # ByteTrack "loses" this frame
    id_during = _by_source(r2, "detected")[0].track_id

    monkeypatch.setattr(tracker._bytetrack, "update_with_detections", real_update)
    r3 = tracker.update([(116, 100, 40, 40)], FRAME_W, FRAME_H)  # ByteTrack "recovers"
    id_after = _by_source(r3, "detected")[0].track_id

    assert id_before == id_during == id_after


def test_id_stays_stable_when_bytetrack_hands_back_a_conflicting_new_id(monkeypatch):
    """
    The precise failure mode this design exists to prevent: ByteTrack
    doesn't just go silent — it can also internally decide a still-visible,
    spatially-continuous box is a NEW track and hand back a different, but
    otherwise perfectly valid-looking, tracker_id for it (its own frame-to-
    frame matching threshold is 0.8, stricter than ours). If our code ever
    trusted that suggestion over our own continuity match, this would fail.
    """
    import numpy as np
    import supervision as sv

    tracker = ClassTracker(smoothing_alpha=0.4)
    r1 = tracker.update([(100, 100, 40, 40)], FRAME_W, FRAME_H)
    id_before = _by_source(r1, "detected")[0].track_id

    def conflicting_suggestion(_dets):
        # A box at the same continuous position, but claiming a brand-new,
        # never-before-seen ByteTrack id — exactly what internal continuity
        # loss inside ByteTrack would look like from the outside.
        return sv.Detections(
            xyxy=np.array([[108.0, 100.0, 148.0, 140.0]], dtype=np.float32),
            confidence=np.array([1.0], dtype=np.float32),
            class_id=np.array([0]),
            tracker_id=np.array([999999]),
        )

    monkeypatch.setattr(tracker._bytetrack, "update_with_detections", conflicting_suggestion)
    r2 = tracker.update([(108, 100, 40, 40)], FRAME_W, FRAME_H)
    id_during = _by_source(r2, "detected")[0].track_id

    assert id_during == id_before, "our own continuity match must win over ByteTrack's conflicting suggestion"


# ── Per-job id renumbering ───────────────────────────────────────────────────

def test_track_ids_are_renumbered_1_2_3_in_order_of_first_appearance():
    tracker = ClassTracker()
    r1 = tracker.update([(0, 0, 10, 10)], FRAME_W, FRAME_H)
    assert _by_source(r1, "detected")[0].track_id == 1

    r2 = tracker.update([(0, 0, 10, 10), (500, 500, 10, 10)], FRAME_W, FRAME_H)
    ids_by_pos = {r.raw_box: r.track_id for r in _by_source(r2, "detected")}
    assert ids_by_pos[(0, 0, 10, 10)] == 1     # same face as before -> same external id
    assert ids_by_pos[(500, 500, 10, 10)] == 2  # newly appeared -> next external id

    r3 = tracker.update([(0, 0, 10, 10), (500, 500, 10, 10), (900, 900, 10, 10)], FRAME_W, FRAME_H)
    ids_by_pos3 = {r.raw_box: r.track_id for r in _by_source(r3, "detected")}
    assert ids_by_pos3[(900, 900, 10, 10)] == 3


def test_external_ids_start_at_1_for_every_fresh_job_regardless_of_process_history():
    # Run one tracker through several tracks first, to advance both
    # supervision.ByteTrack's process-wide internal counter and this
    # tracker's own external counter.
    warm_up = ClassTracker()
    for i in range(5):
        warm_up.update([(i * 100, i * 100, 10, 10)], FRAME_W, FRAME_H)

    # A brand new job's tracker must still start its own external ids at 1,
    # no matter what the shared internal ByteTrack counter is doing.
    fresh = ClassTracker()
    result = fresh.update([(0, 0, 10, 10)], FRAME_W, FRAME_H)
    assert _by_source(result, "detected")[0].track_id == 1


# ── Condition 2: smoothing must never shrink/lag below the raw box ──────────

def test_raw_box_always_fully_covered_for_fast_moving_object():
    # A wide frame so a realistic detector-style box (always within frame
    # bounds — a detector never reports coordinates outside the pixels it
    # was given) never runs off the edge; that's a separate, already-covered
    # concern (see test_pad_box_clamps_to_frame_bounds).
    wide_frame_w = 5000
    tracker = ClassTracker(smoothing_alpha=0.2, pad_pct=0.0)  # low alpha = heavy lag, pad=0 = no safety margin from padding
    x = 0
    for _ in range(30):
        raw = (x, 100, 40, 40)
        results = tracker.update([raw], wide_frame_w, FRAME_H)
        detected = _by_source(results, "detected")[0]
        assert _contains(detected.box, raw), f"raw box {raw} not fully covered by region {detected.box}"
        x += 60  # large jump each frame — heavy smoothing lag if not unioned with raw


def test_smoothing_still_reduces_jitter_around_a_stable_box():
    # Same box every frame except small per-frame noise — smoothed box should
    # converge tightly around it (not just always equal the noisy raw box).
    # Only one object is tracked in this test, so there's exactly one entry
    # in _history at any time — read it directly rather than by id (ids are
    # internal-bookkeeping keys, not what TrackedBox.track_id returns; see
    # the external-id renumbering test below for that distinction).
    import random
    rng = random.Random(0)
    tracker = ClassTracker(smoothing_alpha=0.3, pad_pct=0.0)
    widths = []
    for _ in range(20):
        noise = rng.randint(-5, 5)
        raw = (100, 100, 50 + noise, 50 + noise)
        tracker.update([raw], FRAME_W, FRAME_H)
        widths.append(list(tracker._history.values())[0]["smoothed"][2])
    # Smoothed width should vary less than the raw noise range (±5 around 50).
    assert max(widths) - min(widths) < 10 - 1e-9


# ── Condition 3: gap-fill with configurable buffer + velocity extrapolation ─

def test_gap_fill_persists_then_expires_after_track_buffer():
    tracker = ClassTracker(track_buffer=3, extrapolate_velocity=False)
    tracker.update([(100, 100, 40, 40)], FRAME_W, FRAME_H)  # frame 1: establish track

    for i in range(3):  # frames 2-4: within buffer
        results = tracker.update([], FRAME_W, FRAME_H)
        gap = _by_source(results, "gap_fill")
        assert len(gap) == 1, f"expected gap-fill on gap frame {i + 1}"
        assert gap[0].raw_box is None

    results = tracker.update([], FRAME_W, FRAME_H)  # frame 5: buffer exceeded
    assert results == []


def test_gap_fill_and_detected_coexist_and_are_labelled_separately():
    tracker = ClassTracker(track_buffer=5)
    tracker.update([(100, 100, 40, 40), (500, 500, 40, 40)], FRAME_W, FRAME_H)
    # Second object stops appearing; first keeps being detected.
    results = tracker.update([(105, 100, 40, 40)], FRAME_W, FRAME_H)

    detected = _by_source(results, "detected")
    gap = _by_source(results, "gap_fill")
    assert len(detected) == 1
    assert len(gap) == 1
    assert detected[0].raw_box == (105, 100, 40, 40)
    assert gap[0].raw_box is None


def test_velocity_extrapolation_moves_gap_fill_in_last_known_direction():
    tracker = ClassTracker(track_buffer=5, extrapolate_velocity=True, smoothing_alpha=1.0)
    tracker.update([(100, 100, 40, 40)], FRAME_W, FRAME_H)
    tracker.update([(120, 100, 40, 40)], FRAME_W, FRAME_H)  # moving +20/frame in x

    results = tracker.update([], FRAME_W, FRAME_H)  # gap frame 1
    gap_x_1 = _by_source(results, "gap_fill")[0].box[0]
    results = tracker.update([], FRAME_W, FRAME_H)  # gap frame 2
    gap_x_2 = _by_source(results, "gap_fill")[0].box[0]

    assert gap_x_2 > gap_x_1 > 120  # keeps extrapolating forward, not frozen at last position


def test_no_extrapolation_keeps_gap_fill_static():
    tracker = ClassTracker(track_buffer=5, extrapolate_velocity=False)
    tracker.update([(100, 100, 40, 40)], FRAME_W, FRAME_H)
    tracker.update([(120, 100, 40, 40)], FRAME_W, FRAME_H)

    results = tracker.update([], FRAME_W, FRAME_H)
    gap_1 = _by_source(results, "gap_fill")[0].box
    results = tracker.update([], FRAME_W, FRAME_H)
    gap_2 = _by_source(results, "gap_fill")[0].box

    assert gap_1 == gap_2


# ── Condition 4: one tracker instance per class, IDs never cross ────────────

def test_separate_class_trackers_have_independent_internal_state():
    # NOTE: supervision.ByteTrack's numeric track_id is drawn from a
    # process-wide counter (STrack._external_count is a class attribute) —
    # so two fresh trackers do NOT both start at id=1, and this test does
    # not assume they do. What actually matters for "IDs never cross
    # classes" is that a face-track and a plate-track can never be confused
    # as the same track: they live in fully separate ClassTracker instances
    # with their own history/ByteTrack state, so even if their numeric IDs
    # ever coincided, nothing in this module would ever compare a face
    # track_id to a plate track_id.
    faces = ClassTracker()
    plates = ClassTracker()

    face_result = faces.update([(0, 0, 10, 10)], FRAME_W, FRAME_H)
    plate_result = plates.update([(50, 50, 10, 10)], FRAME_W, FRAME_H)

    assert faces._history is not plates._history
    assert faces._bytetrack is not plates._bytetrack
    assert faces._bytetrack.tracked_tracks is not plates._bytetrack.tracked_tracks
    # Each tracker only knows about the box it was actually given.
    assert list(faces._history.values())[0]["raw"] == (0, 0, 10, 10)
    assert list(plates._history.values())[0]["raw"] == (50, 50, 10, 10)
    assert face_result[0].raw_box != plate_result[0].raw_box


# ── Condition 5: tracker state is per-job, never shared across instances ────

def test_two_trackers_do_not_share_state_when_run_interleaved():
    """
    Two ClassTracker instances standing in for two concurrent Flask video
    jobs. Calls are interleaved frame-by-frame with unrelated box sequences.
    What must hold: neither tracker's matching/gap-fill logic is ever
    influenced by the other's boxes, regardless of call interleaving.
    (Exact numeric track_id equality with an isolated run is NOT asserted —
    supervision.ByteTrack's process-wide id counter means interleaving
    changes which numbers get handed out, without changing behaviour.)
    """
    tracker_a = ClassTracker()
    tracker_b = ClassTracker()

    a_positions = [(10, 10), (20, 10), (30, 10), (40, 10)]  # smooth motion, small region
    b_positions = [(2000, 2000), (2500, 2200), (1800, 2600), (2300, 1900)]  # jumps around, far region

    a_track_ids, b_track_ids = [], []
    for i in range(4):
        ra = tracker_a.update([(a_positions[i][0], a_positions[i][1], 20, 20)], FRAME_W, FRAME_H)
        rb = tracker_b.update([(b_positions[i][0], b_positions[i][1], 20, 20)], FRAME_W, FRAME_H)
        a_track_ids.append(_by_source(ra, "detected")[0].track_id)
        b_track_ids.append(_by_source(rb, "detected")[0].track_id)

    # A's smoothly-moving box is recognized as one continuous track.
    assert len(set(a_track_ids)) == 1
    # Neither tracker's history contains the other's boxes.
    a_raw_boxes = {state["raw"] for state in tracker_a._history.values()}
    b_raw_boxes = {state["raw"] for state in tracker_b._history.values()}
    assert a_raw_boxes.isdisjoint(b_raw_boxes)
    assert all(box[0] < 100 for box in a_raw_boxes)
    assert all(box[0] >= 1000 for box in b_raw_boxes)
    # Genuinely separate objects, not aliases of shared module-level state.
    assert tracker_a._history is not tracker_b._history
    assert tracker_a._bytetrack is not tracker_b._bytetrack
    assert tracker_a._bytetrack.tracked_tracks is not tracker_b._bytetrack.tracked_tracks


# ── Helper function unit tests ───────────────────────────────────────────────

def test_union_box_known_values():
    assert union_box((0, 0, 10, 10), (5, 5, 10, 10)) == (0, 0, 15, 15)
    assert union_box((0, 0, 10, 10), (0, 0, 10, 10)) == (0, 0, 10, 10)


def test_pad_box_grows_by_percentage():
    result = pad_box((100, 100, 40, 40), 0.5, FRAME_W, FRAME_H)
    # 50% of 40 = 20px padding on EACH side -> width grows by 40 total (20+20)
    assert result == (80, 80, 80, 80)


def test_pad_box_clamps_to_frame_bounds():
    result = pad_box((5, 5, 20, 20), 0.5, FRAME_W, FRAME_H)
    assert result[0] == 0 and result[1] == 0  # clamped at top-left edge

    result2 = pad_box((FRAME_W - 20, FRAME_H - 20, 20, 20), 0.5, FRAME_W, FRAME_H)
    assert result2[0] + result2[2] == FRAME_W
    assert result2[1] + result2[3] == FRAME_H
