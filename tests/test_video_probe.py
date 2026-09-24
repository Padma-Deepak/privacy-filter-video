"""
Tests for project/core/video.py's probing and pure geometry helpers.
Real tiny ffmpeg-generated clips for probing; pure unit tests for rotation
math, detection-scale math, and the ffmpeg-availability check.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "project")
sys.path.insert(0, PROJECT_DIR)

import core.video as video  # noqa: E402


def _make_clip(path, duration=2, size="320x240", fps=10, with_audio=False):
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=duration={duration}:size={size}:rate={fps}"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path), "-loglevel", "error"]
    subprocess.run(cmd, check=True)


@pytest.fixture
def plain_clip(tmp_path):
    path = tmp_path / "plain.mp4"
    _make_clip(path, duration=2, size="320x240", fps=10, with_audio=False)
    return str(path)


@pytest.fixture
def audio_clip(tmp_path):
    path = tmp_path / "audio.mp4"
    _make_clip(path, duration=2, size="320x240", fps=10, with_audio=True)
    return str(path)


# ── probe_video (real generated clips) ───────────────────────────────────────

def test_probe_video_reads_dimensions_and_duration(plain_clip):
    probe = video.probe_video(plain_clip)
    assert probe.width == 320
    assert probe.height == 240
    assert abs(probe.duration_sec - 2.0) < 0.2
    assert probe.rotation_degrees == 0
    assert probe.has_audio is False


def test_probe_video_detects_audio(audio_clip):
    probe = video.probe_video(audio_clip)
    assert probe.has_audio is True


def test_probe_video_raises_clear_error_for_bad_path(tmp_path):
    with pytest.raises(ValueError):
        video.probe_video(str(tmp_path / "does_not_exist.mp4"))


# ── rotation helpers (pure logic, mocked angles — see README Testing note) ──

def test_rotation_cv2_flag_mapping():
    import cv2
    assert video.rotation_cv2_flag(0) is None
    assert video.rotation_cv2_flag(90) == cv2.ROTATE_90_CLOCKWISE
    assert video.rotation_cv2_flag(180) == cv2.ROTATE_180
    assert video.rotation_cv2_flag(270) == cv2.ROTATE_90_COUNTERCLOCKWISE


def test_rotated_dimensions_swaps_for_90_and_270():
    assert video.rotated_dimensions(640, 480, 0) == (640, 480)
    assert video.rotated_dimensions(640, 480, 90) == (480, 640)
    assert video.rotated_dimensions(640, 480, 180) == (640, 480)
    assert video.rotated_dimensions(640, 480, 270) == (480, 640)


def test_apply_rotation_actually_swaps_frame_shape():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)  # h=480, w=640
    rotated = video.apply_rotation(frame, 90)
    assert rotated.shape[:2] == (640, 480)  # h/w swapped

    unrotated = video.apply_rotation(frame, 0)
    assert unrotated.shape[:2] == (480, 640)


def test_extract_rotation_degrees_from_side_data():
    stream = {"side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90.0}]}
    assert video._extract_rotation_degrees(stream) == 90


def test_extract_rotation_degrees_from_legacy_tag():
    stream = {"tags": {"rotate": "180"}}
    assert video._extract_rotation_degrees(stream) == 180


def test_extract_rotation_degrees_defaults_to_zero():
    assert video._extract_rotation_degrees({}) == 0


# ── resolution decoupling ────────────────────────────────────────────────────

def test_detection_scale_factor_no_scaling_when_already_small():
    assert video.detection_scale_factor(800, 960) == 1.0
    assert video.detection_scale_factor(960, 960) == 1.0


def test_detection_scale_factor_shrinks_large_frames():
    factor = video.detection_scale_factor(1920, 960)
    assert factor == 0.5


def test_scale_boxes_maps_back_to_full_resolution():
    boxes = [(10, 20, 30, 40)]
    scaled = video.scale_boxes(boxes, 0.5)  # detected at half-res -> scale up by 2x
    assert scaled == [(20, 40, 60, 80)]


def test_scale_boxes_noop_at_factor_one():
    boxes = [(10, 20, 30, 40)]
    assert video.scale_boxes(boxes, 1.0) == boxes


# ── ffmpeg availability ──────────────────────────────────────────────────────

def test_check_ffmpeg_available_raises_clear_error_when_missing(monkeypatch):
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    with pytest.raises(video.FFmpegNotFoundError, match="ffmpeg"):
        video.check_ffmpeg_available()


def test_check_ffmpeg_available_passes_when_present():
    video.check_ffmpeg_available()  # should not raise on this machine
