"""Analyse once, review stable boxes, then render without re-running detection.

Decoded originals and thumbnails live in memory only. The uploaded source copy
exists for the lifetime of a review job; jobs.py owns its deletion.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path

import cv2
import detector
from core.profiles import redact
from core.faces import detect_faces
from core.selection import is_hidden
from core.tracking import ClassTracker, pad_box
from core.video import (apply_rotation, check_ffmpeg_available, detection_scale_factor,
                        probe_video, rotated_dimensions, scale_boxes)

LIMITATIONS = [
    "Detectors can miss faces and plates or produce false positives. Review every frame.",
    "Track IDs are motion-based, not verified identities; crossings and re-entry can switch or split IDs.",
    "Keep-visible and range choices may expose a different person if an identity switch occurs.",
    "Voice, clothing, gait, background and context can identify people despite face redaction.",
    "Variable frame timing is normalized to an average frame rate; local timing is not preserved.",
    "YuNet can miss small or occluded faces. Journalist changes redaction strength, not identity guarantees.",
]


class Cancelled(Exception):
    """Cooperative cancellation between frames and while FFmpeg runs."""


def open_capture(path: str):
    cap = cv2.VideoCapture(path)
    # Rotation is explicitly applied from ffprobe. Prevent double rotation.
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    if not cap.isOpened():
        cap.release()
        raise ValueError("Cannot decode this video")
    return cap


@detector.synchronized
def detect(frame, models_dir: str, profile: dict) -> dict:
    """Serialize model use within the single job worker; do not make network calls."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = {}
    if "faces" in profile["classes"]:
        result["faces"] = detect_faces(frame, models_dir, profile["face_confidence"])
    if "plates" in profile["classes"]:
        result["plates"] = detector.detect_plates(gray)
    if "screens" in profile["classes"]:
        result["screens"] = detector.detect_screens(frame, conf=profile["screen_confidence"])
    return result


def thumbnail(frame, box) -> str:
    x, y, w, h = box
    height, width = frame.shape[:2]
    crop = frame[max(0, y):min(height, y + h), max(0, x):min(width, x + w)]
    if crop.size == 0:
        return ""
    scale = min(1, 80 / max(crop.shape[:2]))
    crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))))
    ok, encoded = cv2.imencode(".jpg", crop)
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode() if ok else ""


def analyse(path: str, is_video: bool, models_dir: str, profile: dict,
            progress, check_cancel, max_frames: int = 18000) -> dict:
    """Return track geometry and in-memory representative crops, never frame files."""
    cap = None
    if is_video:
        check_ffmpeg_available()
        probe = probe_video(path)
        width, height = rotated_dimensions(probe.width, probe.height, probe.rotation_degrees)
        cap = open_capture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rotation, duration = probe.rotation_degrees, probe.duration_sec
        still = None
    else:
        still = cv2.imread(path)
        if still is None:
            raise ValueError("Cannot decode this image")
        height, width = still.shape[:2]
        rotation, duration, total = 0, 0, 1
    trackers = {cls: ClassTracker(track_buffer=profile["track_buffers"][cls],
                                 pad_pct=profile["padding"]) for cls in profile["classes"]}
    frames, tracks, timestamps = [], {}, []
    records = 0
    started = time.monotonic()
    try:
        while True:
            check_cancel()
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = apply_rotation(frame, rotation)
                timestamps.append(max(0, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000))
            elif not frames:
                frame = still
            else:
                break
            if len(frames) >= max_frames:
                raise ValueError(f"Review is limited to {max_frames} frames; use a shorter clip")
            if frame.shape[:2] != (height, width):
                raise ValueError("Decoded dimensions disagree with video metadata")
            scale = detection_scale_factor(width, 960)
            small = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale)))) if scale != 1 else frame
            detections = detect(small, models_dir, profile)
            frame_boxes = []
            index = len(frames)
            for cls, tracker in trackers.items():
                boxes = scale_boxes(detections.get(cls, []), scale)
                for tracked in tracker.update(boxes, width, height):
                    key = f"{cls}:{tracked.track_id}"
                    if key not in tracks:
                        tracks[key] = {"id": key, "class": cls, "first": index, "last": index,
                                       "thumbnail": thumbnail(frame, tracked.raw_box or tracked.box),
                                       "gap_frames": 0}
                    tracks[key]["last"] = index
                    tracks[key]["gap_frames"] += int(tracked.source == "gap_fill")
                    frame_boxes.append({"id": key, "class": cls, "box": list(tracked.box),
                                        "raw_box": list(tracked.raw_box) if tracked.raw_box else None,
                                        "source": tracked.source})
            records += len(frame_boxes)
            if records > 500000 or len(tracks) > 5000:
                raise ValueError("Too many detections for interactive review; use a shorter clip")
            frames.append(frame_boxes)
            progress(len(frames), max(total, len(frames)))
        if not frames:
            raise ValueError("No readable frames")
        return {"width": width, "height": height, "rotation": rotation,
                "duration": duration, "fps": len(frames) / duration if duration else 1,
                "frame_count": len(frames), "is_video": is_video, "frames": frames,
                "timestamps": timestamps,
                "tracks": tracks, "warnings": detector.get_detector_warnings(),
                "analysis_seconds": time.monotonic() - started}
    finally:
        if cap is not None:
            cap.release()


def preview_frame(path: str, analysis: dict, index: int) -> bytes:
    """Decode only the requested frame, encode JPEG in memory."""
    if analysis["is_video"]:
        cap = open_capture(path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Cannot decode preview frame")
            frame = apply_rotation(frame, analysis["rotation"])
        finally:
            cap.release()
    else:
        frame = cv2.imread(path)
    if frame is None:
        raise ValueError("Cannot decode preview")
    scale = min(1, 1000 / max(frame.shape[:2]))
    if scale < 1:
        frame = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("Cannot encode preview")
    return encoded.tobytes()


def render_frame(frame, index: int, analysis: dict, choices: dict, manual: list, profile: dict):
    for record in analysis["frames"][index]:
        if is_hidden(record["id"], index, choices):
            redact(frame, record["box"], record["class"], profile)
    for region in manual:
        if region["start"] <= index <= region["end"]:
            # Manual fixes always hide, including overlap with a keep-visible track.
            box = pad_box(region["box"], profile["padding"], analysis["width"], analysis["height"])
            redact(frame, box, "faces", profile)
    return frame


def run_ffmpeg(command: list, check_cancel) -> None:
    """Poll a child process so cancel/timeout can terminate encoding promptly."""
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        while process.poll() is None:
            check_cancel()
            time.sleep(0.1)
        if process.returncode:
            raise ValueError("Video encoding failed; check FFmpeg and available disk space")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def export(path: str, directory: Path, analysis: dict, profile: dict, choices: dict,
           manual: list, audio: str, progress, check_cancel) -> Path:
    """Render the recorded geometry. Audio uses only the first source audio stream."""
    if not analysis["is_video"]:
        frame = cv2.imread(path)
        if frame is None:
            raise ValueError("Cannot decode image")
        check_cancel()
        render_frame(frame, 0, analysis, choices, manual, profile)
        output = directory / "redacted.png"
        if not cv2.imwrite(str(output), frame):
            raise ValueError("Cannot save image")
        progress(1, 1)
        return output
    silent, output = directory / "silent.mp4", directory / "redacted.mp4"
    cap, writer = None, None
    try:
        cap = open_capture(path)
        writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), analysis["fps"],
                                 (analysis["width"], analysis["height"]))
        if not writer.isOpened():
            raise ValueError("Cannot open video writer")
        for index in range(analysis["frame_count"]):
            check_cancel()
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Source ended before all reviewed frames were exported")
            frame = apply_rotation(frame, analysis["rotation"])
            writer.write(render_frame(frame, index, analysis, choices, manual, profile))
            progress(index + 1, analysis["frame_count"])
        cap.release()
        writer.release()
        cap = writer = None
        command = ["ffmpeg", "-y", "-i", str(silent)]
        if audio == "keep":
            command += ["-i", path]
        command += ["-map", "0:v:0"]
        if audio == "keep":
            command += ["-map", "1:a:0?", "-c:a", "aac"]
        else:
            command += ["-an"]
        # Never copy source metadata or extra data streams. Generic codec tags may remain.
        command += ["-map_metadata", "-1", "-map_chapters", "-1",
                    "-metadata", "encoder=Privacy Filter", "-metadata:s:v", "rotate=0",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-t", str(analysis["duration"]), "-loglevel", "error", str(output)]
        run_ffmpeg(command, check_cancel)
        return output
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        silent.unlink(missing_ok=True)


def build_report(path: str, analysis: dict, profile: dict, choices: dict, manual: list,
                 audio: str, elapsed: float) -> dict:
    """Audit information only: no filenames, thumbnails, pixels or original metadata."""
    with open(path, "rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    counts = {key: 0 for key in analysis["tracks"]}
    for index, frame in enumerate(analysis["frames"]):
        for record in frame:
            if is_hidden(record["id"], index, choices):
                counts[record["id"]] += 1
    tracks = []
    for key, track in analysis["tracks"].items():
        tracks.append({k: v for k, v in track.items() if k != "thumbnail"} |
                      {"choice": choices.get(key, {"mode": "hide"}),
                       "hidden_frames": counts[key]})
    return {"schema_version": 1, "input_sha256": digest, "profile": profile,
            "audio": audio, "metadata": "Source metadata and chapters removed; generic codec tags may remain",
            "frame_count": analysis["frame_count"], "tracks": tracks,
            "track_counts": {cls: sum(t["class"] == cls for t in tracks) for cls in profile["classes"]},
            "manual_regions": manual, "warnings": analysis["warnings"], "limitations": LIMITATIONS,
            "versions": {pkg: importlib.metadata.version(pkg) for pkg in
                         ("opencv-python", "ultralytics", "supervision")},
            "timing": {"analysis_seconds": analysis["analysis_seconds"], "export_seconds": elapsed,
                       "pipeline_fps": analysis["frame_count"] / max(0.001, elapsed + analysis["analysis_seconds"]),
                       "excludes": "upload, user review and downloads"}}
