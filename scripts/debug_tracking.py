"""
debug_tracking.py — visualize the tracking pipeline on a real clip.

Draws three layers per tracked object, without redacting anything, so the
gap between "what the detector saw," "what the tracker thinks," and "what
actually gets blurred" is visible frame by frame:

    GREEN  = raw detection box this frame (exactly what the detector
             returned — absent on gap-fill frames, since there's no raw
             detection then)
    YELLOW = the tracker's own estimate: the smoothed box (detected frames)
             or the extrapolated last-known box (gap-fill frames) — before
             the raw-box union and padding
    RED    = the final redaction region actually used by the real pipeline
             (padded union of raw + smoothed) — this is what
             project/core/video.py blurs/masks/pixelates

Each box is labelled with its renumbered, per-job-stable track ID (see
project/core/tracking.py — 1, 2, 3, ... in order of first appearance, not
supervision.ByteTrack's internal process-wide id).

Usage:
    python scripts/debug_tracking.py <input_video> [--out PATH] [--max-seconds N]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import cv2

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "project")
sys.path.insert(0, PROJECT_DIR)

import detector  # noqa: E402
from core.tracking import ClassTracker  # noqa: E402
from core.video import (  # noqa: E402
    apply_rotation, check_ffmpeg_available, detection_scale_factor,
    probe_video, rotated_dimensions, scale_boxes,
)

GREEN = (0, 200, 0)     # BGR
YELLOW = (0, 210, 255)
RED = (0, 0, 255)
LABEL_COLOR = (255, 255, 255)

CLASS_SHORT_NAME = {"faces": "face", "plates": "plate", "screens": "screen"}


def _draw_box(frame, box, color, thickness=2):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_video")
    parser.add_argument("--out", default=None, help="output path (default: scripts/debug_output/<name>_debug.mp4)")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--models-dir", default=os.path.join(PROJECT_DIR, "models"))
    args = parser.parse_args()

    check_ffmpeg_available()

    out_dir = os.path.dirname(args.out) if args.out else os.path.join(os.path.dirname(__file__), "debug_output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.out or os.path.join(
        out_dir, os.path.splitext(os.path.basename(args.input_video))[0] + "_debug.mp4"
    )
    raw_path = out_path + ".raw.mp4"

    probe = probe_video(args.input_video)
    out_w, out_h = rotated_dimensions(probe.width, probe.height, probe.rotation_degrees)

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        sys.exit(f"Could not open {args.input_video}")

    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), probe.nominal_fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        sys.exit("Could not open video writer")

    trackers = {cls: ClassTracker() for cls in ("faces", "plates", "screens")}
    scale = detection_scale_factor(out_w, 960)
    max_frames = int(args.max_seconds * probe.nominal_fps) if args.max_seconds else None

    counts_by_source = {"detected": 0, "gap_fill": 0}
    frame_idx = 0
    try:
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frame = apply_rotation(frame, probe.rotation_degrees)

            if scale != 1.0:
                small = cv2.resize(
                    frame, (max(1, int(out_w * scale)), max(1, int(out_h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
                detections = detector.detect_all(small, args.models_dir)
                detections = {cls: scale_boxes(boxes, scale) for cls, boxes in detections.items()}
            else:
                detections = detector.detect_all(frame, args.models_dir)

            for cls, boxes in detections.items():
                tracked = trackers[cls].update(boxes, out_w, out_h)
                for t in tracked:
                    counts_by_source[t.source] += 1
                    if t.raw_box is not None:
                        _draw_box(frame, t.raw_box, GREEN)
                    _draw_box(frame, t.tracker_box, YELLOW)
                    _draw_box(frame, t.box, RED)
                    label = f"{CLASS_SHORT_NAME[cls]}#{t.track_id}"
                    label_pos = (t.box[0], max(15, t.box[1] - 8))
                    cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, LABEL_COLOR, 1, cv2.LINE_AA)

            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    if frame_idx == 0:
        os.remove(raw_path)
        sys.exit("No frames read — nothing to write")

    # Re-encode for broad playability (see docs/AUDIT.md finding #3 — raw
    # mp4v from cv2.VideoWriter isn't reliably playable in browsers).
    subprocess.run(
        ["ffmpeg", "-y", "-i", raw_path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-loglevel", "error", out_path],
        check=True,
    )
    os.remove(raw_path)

    print(f"Processed {frame_idx} frames")
    print(f"  detected boxes drawn: {counts_by_source['detected']}")
    print(f"  gap-fill boxes drawn: {counts_by_source['gap_fill']}")
    print(f"Legend: GREEN=raw detection  YELLOW=tracker estimate (unpadded)  RED=final redaction region")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
