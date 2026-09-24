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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2

import detector
from core.tracking import ClassTracker

logger = logging.getLogger(__name__)

DETECTION_MAX_WIDTH_DEFAULT = 960

_FILTER_FOR_CLASS = {
    "faces": detector.apply_gaussian_blur,
    "plates": detector.apply_black_mask,
    "screens": detector.apply_pixelation,
}


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

    # format-level duration is computed from actual container/packet
    # timestamps and is preferred; the video STREAM's own `duration` field
    # can be stale/derived (nb_frames / avg_frame_rate using an avg_frame_rate
    # that doesn't reflect real irregular timing) — confirmed empirically on
    # a genuinely variable-frame-rate test fixture, where the stream-level
    # duration was wrong by 0.8s out of 2.8s while format-level was correct.
    duration_sec = None
    for source in (data.get("format", {}).get("duration"), video_stream.get("duration")):
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


def _safe_remove(path) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            logger.warning("Could not remove temp file %s", path, exc_info=True)


def _mux_with_ffmpeg(silent_path: str, original_input_path: str, output_path: str,
                      fps: float, mux_audio: bool, duration_sec: float) -> None:
    """
    Re-encode the silent intermediate as real H.264 (libx264/yuv420p/+faststart)
    and mux the original audio back in. `-r fps` before `-i silent_path`
    reinterprets its frame timing so frame_count/fps == duration_sec exactly
    (see module docstring for why fps is a measured, corrected value here,
    not the provisional rate the intermediate was written with). `-t` clamps
    both streams to duration_sec as a rounding safety net.
    """
    cmd = ["ffmpeg", "-y", "-r", f"{fps:.6f}", "-i", silent_path]
    if mux_audio:
        cmd += ["-i", original_input_path]
    cmd += ["-map", "0:v:0"]
    if mux_audio:
        cmd += ["-map", "1:a:0", "-c:a", "aac"]
    cmd += [
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-t", f"{duration_sec:.6f}", "-loglevel", "error", output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg encode/mux failed: {result.stderr.strip()}")


def process_video_streaming(
    input_path: str,
    outputs_dir: str,
    models_dir: str,
    max_clip_seconds: "float | None" = None,
    detection_max_width: int = DETECTION_MAX_WIDTH_DEFAULT,
    track_buffer: int = 30,
    smoothing_alpha: float = 0.5,
    pad_pct: float = 0.15,
) -> dict:
    """
    Stream a video frame-by-frame (never the whole clip in memory), detect on
    an optionally-downscaled copy but redact at the original resolution,
    track faces/plates/screens for gap-fill and smoothing, and mux a real
    H.264 output with the original audio via ffmpeg.

    max_clip_seconds: None (default) = unlimited, matching PHASES.md Phase 1
    ("remove the 12-second cap ... behind an environment variable
    MAX_CLIP_SECONDS, default unlimited"). project/app.py reads that env var
    and passes it through.

    Temp files (the silent pre-mux intermediate always, and the final output
    too if anything fails before returning) are cleaned up on every exit
    path — success, error, or an interrupt (e.g. Ctrl-C / KeyboardInterrupt)
    mid-loop, since try/finally runs regardless of how the frame reads exit.
    There is no separate "cancel" code path yet — that needs the background
    job system planned for Phase 3; this makes the current synchronous path
    exception-safe for any interruption.

    Returns dict: output_path, faces_found, plates_found, screens_found,
    frames_processed, truncated, warnings.
    """
    check_ffmpeg_available()
    probe = probe_video(input_path)
    out_w, out_h = rotated_dimensions(probe.width, probe.height, probe.rotation_degrees)

    os.makedirs(outputs_dir, exist_ok=True)
    job_id = uuid.uuid4().hex
    silent_path = os.path.join(outputs_dir, f"{job_id}_silent.mp4")
    final_path = os.path.join(outputs_dir, f"{job_id}.mp4")

    cap = None
    writer = None
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not read video: {input_path}")

        writer = cv2.VideoWriter(
            silent_path, cv2.VideoWriter_fourcc(*"mp4v"), probe.nominal_fps, (out_w, out_h)
        )
        if not writer.isOpened():
            raise ValueError("Could not open video writer for output.")

        trackers = {
            cls: ClassTracker(
                track_buffer=track_buffer, smoothing_alpha=smoothing_alpha,
                pad_pct=pad_pct, frame_rate=max(1, round(probe.nominal_fps)),
            )
            for cls in ("faces", "plates", "screens")
        }

        scale = detection_scale_factor(out_w, detection_max_width)
        max_frames = int(max_clip_seconds * probe.nominal_fps) if max_clip_seconds else None

        totals = {"faces": 0, "plates": 0, "screens": 0}
        frame_idx = 0
        truncated = False

        with ThreadPoolExecutor(max_workers=3) as pool:
            while True:
                if max_frames is not None and frame_idx >= max_frames:
                    truncated = True
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
                    detections = detector.detect_all(small, models_dir, executor=pool)
                    detections = {cls: scale_boxes(boxes, scale) for cls, boxes in detections.items()}
                else:
                    detections = detector.detect_all(frame, models_dir, executor=pool)

                for cls, boxes in detections.items():
                    totals[cls] += len(boxes)
                    tracked = trackers[cls].update(boxes, out_w, out_h)
                    filter_fn = _FILTER_FOR_CLASS[cls]
                    for t in tracked:
                        x, y, w, h = t.box
                        filter_fn(frame, x, y, w, h)

                writer.write(frame)
                frame_idx += 1

        cap.release()
        writer.release()
        cap = writer = None

        if frame_idx == 0:
            raise ValueError("No readable frames found in video.")

        # elapsed_sec: how much of the source this output actually spans.
        # NOTE: cv2.CAP_PROP_POS_MSEC was tried here first and found
        # unreliable for variable-frame-rate sources on this OpenCV/ffmpeg
        # backend combination — it drifted substantially from ffprobe's own
        # (decode-verified) duration on a real VFR test fixture. ffprobe's
        # probed duration is trusted instead for a full read; only the
        # truncated case (where we deliberately stopped partway through and
        # ffprobe's full-source duration no longer applies) falls back to
        # frame_count / nominal_fps.
        elapsed_sec = probe.duration_sec if not truncated else frame_idx / probe.nominal_fps
        corrected_fps = frame_idx / elapsed_sec if elapsed_sec > 0 else probe.nominal_fps
        output_duration_sec = frame_idx / corrected_fps

        _mux_with_ffmpeg(
            silent_path, input_path, final_path,
            fps=corrected_fps, mux_audio=probe.has_audio, duration_sec=output_duration_sec,
        )

        return {
            "output_path": final_path,
            "faces_found": totals["faces"],
            "plates_found": totals["plates"],
            "screens_found": totals["screens"],
            "frames_processed": frame_idx,
            "truncated": truncated,
            "warnings": detector.get_detector_warnings(),
        }
    except BaseException:
        # BaseException, not Exception: a KeyboardInterrupt mid-loop must
        # still clean up temp files, not just a caught application error.
        _safe_remove(final_path)
        raise
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        _safe_remove(silent_path)
