"""
video.py — streaming video pipeline: read frames one at a time (never the
whole clip in memory), detect on an optionally-downscaled copy but redact at
the original resolution, route detections through per-class ClassTracker
instances, and mux the result with ffmpeg (real H.264, original audio).

Replaces detector.process_video() for the video upload path. The image path
(detector.process_image()) is untouched.

Duration cap: behind the MAX_CLIP_SECONDS environment variable (read by
project/app.py, passed in as `max_clip_seconds`), default unlimited — see
PHASES.md Phase 1. The old hardcoded 12s/960px-forced-downscale defaults are
gone from this path; 960px is now only a detection-speed optimization that
never touches the output resolution.

VFR (variable frame rate) handling — a real tradeoff, not an oversight: this
pipeline does not preserve exact per-frame timestamps. It reads every frame
via OpenCV (frame-accurate, but OpenCV's VideoWriter only supports writing
at a single constant rate — there's no per-frame PTS control without adding
a dependency like PyAV), then normalizes to constant frame rate at the
source's ACTUAL average fps: frame_count_actually_read / source_duration
(both measured, not assumed). That average is applied as the final output's
frame rate during the ffmpeg encode step — not the provisional rate used for
the silent intermediate — which makes output duration equal source duration
by construction (frame_count / (frame_count / duration) == duration, up to
rounding), so it holds regardless of how uneven the source's real frame
timing was. What's lost: if the source had large timing swings (e.g. long
pauses on some frames), motion in those stretches will look evenly-paced in
the output rather than preserving the original unevenness. Audio is muxed
from the original file, which has the same source duration, so it stays in
sync with the now-duration-matched video track.

Rotation — also a documented gap, not silently ignored: rotation is read via
ffprobe (checking both side_data_list "Display Matrix" rotation and the
legacy tags.rotate) and compensated with cv2.rotate() on every frame, rather
than trusting OpenCV's CAP_PROP_ORIENTATION_AUTO (see docs/AUDIT.md and the
Stage 1/2 summaries for why: this environment's ffmpeg build couldn't be
made to write real rotation metadata into a test fixture despite trying five
documented methods, so this logic is unit-tested against a mocked ffprobe
rotation value, not a real rotated file — check real portrait phone clips
manually, per README's Testing section).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass

import cv2

from core.tracking import ClassTracker

logger = logging.getLogger(__name__)

DETECTION_MAX_WIDTH_DEFAULT = 960


@dataclass
class VideoProbe:
    width: int
    height: int
    rotation_degrees: int   # normalized to one of 0, 90, 180, 270
    duration_sec: float
    has_audio: bool
    nominal_fps: float       # container-reported fps; a provisional value only, see module docstring


class FFmpegNotFoundError(RuntimeError):
    pass


def check_ffmpeg_available() -> None:
    """Raise a clear, actionable error if ffmpeg/ffprobe aren't on PATH."""
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise FFmpegNotFoundError(
            f"{' and '.join(missing)} not found on PATH. Video processing requires ffmpeg "
            f"(install via e.g. `brew install ffmpeg` on macOS, or your OS package manager)."
        )


def _run_ffprobe(input_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise ValueError(f"ffprobe failed on {input_path}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def _extract_rotation_degrees(video_stream: dict) -> int:
    """
    Check both places rotation shows up: the modern side_data_list "Display
    Matrix" rotation field, and the legacy tags.rotate tag. Normalized to a
    positive value in {0, 90, 180, 270} — the "clockwise rotation needed to
    display correctly" convention used by ffmpeg/QuickTime/most players.
    """
    for side_data in video_stream.get("side_data_list", []):
        if "rotation" in side_data:
            try:
                degrees = -float(side_data["rotation"])  # side_data rotation is counter-clockwise
                return int(round(degrees)) % 360
            except (TypeError, ValueError):
                pass

    rotate_tag = video_stream.get("tags", {}).get("rotate")
    if rotate_tag is not None:
        try:
            return int(round(float(rotate_tag))) % 360
        except (TypeError, ValueError):
            pass

    return 0


def probe_video(input_path: str) -> VideoProbe:
    """Read container metadata needed before streaming frames."""
    data = _run_ffprobe(input_path)
    video_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if not video_streams:
        raise ValueError(f"No video stream found in {input_path}")
    video_stream = video_streams[0]
    has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))

    width = int(video_stream["width"])
    height = int(video_stream["height"])
    rotation = _extract_rotation_degrees(video_stream)

    duration_sec = None
    for source in (video_stream.get("duration"), data.get("format", {}).get("duration")):
        if source is not None:
            try:
                duration_sec = float(source)
                break
            except (TypeError, ValueError):
                continue
    if duration_sec is None or duration_sec <= 0:
        raise ValueError(f"Could not determine duration for {input_path}")

    nominal_fps = 24.0
    rate_str = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")
    if rate_str and rate_str != "0/0":
        try:
            num, den = rate_str.split("/")
            if float(den) > 0:
                nominal_fps = float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            pass

    return VideoProbe(
        width=width, height=height, rotation_degrees=rotation,
        duration_sec=duration_sec, has_audio=has_audio, nominal_fps=nominal_fps,
    )


def rotated_dimensions(width: int, height: int, rotation_degrees: int) -> tuple:
    """Display-correct (width, height) after compensating for rotation_degrees."""
    if rotation_degrees in (90, 270):
        return height, width
    return width, height


def rotation_cv2_flag(rotation_degrees: int):
    """Map a normalized rotation angle to a cv2.rotate() flag, or None for 0."""
    return {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }.get(rotation_degrees % 360)


def apply_rotation(frame, rotation_degrees: int):
    """Return frame rotated to display-correct orientation, or frame unchanged if rotation_degrees is 0."""
    flag = rotation_cv2_flag(rotation_degrees)
    if flag is None:
        return frame
    return cv2.rotate(frame, flag)


def detection_scale_factor(width: int, max_width: int) -> float:
    """<=1.0 factor to shrink a frame to max_width for detection only; 1.0 if already small enough."""
    if width <= max_width or max_width <= 0:
        return 1.0
    return max_width / width


def scale_boxes(boxes, factor: float):
    """Rescale (x, y, w, h) boxes by 1/factor — used to map detection-space boxes back to full resolution."""
    if factor == 1.0:
        return list(boxes)
    inv = 1.0 / factor
    return [
        (int(round(x * inv)), int(round(y * inv)), int(round(w * inv)), int(round(h * inv)))
        for (x, y, w, h) in boxes
    ]
