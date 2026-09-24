"""
Integration tests for project/core/video.py's process_video_streaming().

Real tiny ffmpeg-generated clips (a few seconds, small resolution) — no
large media files, per CLAUDE.md. Covers: basic success, audio preservation
and sync, duration-matching (including a variable-frame-rate source),
resolution decoupling (output stays full-res even when detection downscales),
the MAX_CLIP_SECONDS truncation path, and cleanup on both success and error.
"""

import json
import os
import subprocess
import sys

import pytest

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "project")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
sys.path.insert(0, PROJECT_DIR)

import core.video as video  # noqa: E402


def _ffprobe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def _make_clip(path, duration=2, size="320x240", fps=10, with_audio=False):
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=duration={duration}:size={size}:rate={fps}"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path), "-loglevel", "error"]
    subprocess.run(cmd, check=True)


def _make_vfr_clip(path, tmp_dir, size="320x240", with_audio=False):
    """
    Genuinely variable frame timing: two segments at different constant
    rates (20fps then 5fps), concatenated into one continuous timeline. A
    single-source `-fps_mode vfr` doesn't produce irregular timing on its
    own — testsrc itself has no irregularity for "don't force CFR" to
    preserve; concatenating different rates does.
    """
    seg_a = tmp_dir / "seg_a.mp4"
    seg_b = tmp_dir / "seg_b.mp4"
    _make_clip(seg_a, duration=1.5, size=size, fps=20)
    _make_clip(seg_b, duration=1.5, size=size, fps=5)
    concat_list = tmp_dir / "concat.txt"
    concat_list.write_text(f"file '{seg_a.name}'\nfile '{seg_b.name}'\n")

    video_only = tmp_dir / "vfr_video_only.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video_only), "-loglevel", "error"],
        check=True, cwd=str(tmp_dir),
    )
    if not with_audio:
        video_only.rename(path)
        return

    duration = float(_ffprobe(str(video_only))["format"]["duration"])
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_only), "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
         "-c:v", "copy", "-c:a", "aac", "-shortest", str(path), "-loglevel", "error"],
        check=True,
    )


@pytest.fixture
def outputs_dir(tmp_path):
    d = tmp_path / "outputs"
    d.mkdir()
    return str(d)


@pytest.fixture
def plain_clip(tmp_path):
    path = tmp_path / "plain.mp4"
    _make_clip(path, duration=2, size="320x240", fps=10)
    return str(path)


@pytest.fixture
def audio_clip(tmp_path):
    path = tmp_path / "audio.mp4"
    _make_clip(path, duration=2, size="320x240", fps=10, with_audio=True)
    return str(path)


# ── basic success ────────────────────────────────────────────────────────────

def test_basic_success_produces_playable_output(plain_clip, outputs_dir):
    result = video.process_video_streaming(plain_clip, outputs_dir, MODELS_DIR)
    assert os.path.exists(result["output_path"])
    assert result["frames_processed"] > 0
    assert result["truncated"] is False
    assert result["warnings"] == []

    probe = _ffprobe(result["output_path"])
    video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert video_stream["codec_name"] == "h264"


def test_output_resolution_matches_input_not_downscaled(plain_clip, outputs_dir):
    result = video.process_video_streaming(plain_clip, outputs_dir, MODELS_DIR, detection_max_width=100)
    probe = _ffprobe(result["output_path"])
    video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    # detection_max_width=100 forces internal downscaling for detection, but
    # the OUTPUT must stay at the source's native 320x240 — resolution
    # decoupling (detect small, redact/write at full resolution).
    assert video_stream["width"] == 320
    assert video_stream["height"] == 240


# ── audio ─────────────────────────────────────────────────────────────────────

def test_audio_is_preserved_when_source_has_it(audio_clip, outputs_dir):
    result = video.process_video_streaming(audio_clip, outputs_dir, MODELS_DIR)
    probe = _ffprobe(result["output_path"])
    assert any(s["codec_type"] == "audio" for s in probe["streams"])


def test_no_audio_track_when_source_has_none(plain_clip, outputs_dir):
    result = video.process_video_streaming(plain_clip, outputs_dir, MODELS_DIR)
    probe = _ffprobe(result["output_path"])
    assert not any(s["codec_type"] == "audio" for s in probe["streams"])


# ── duration matching (incl. variable frame rate) ────────────────────────────

def test_output_duration_matches_input_within_one_frame(plain_clip, outputs_dir):
    input_probe = _ffprobe(plain_clip)
    input_duration = float(input_probe["format"]["duration"])

    result = video.process_video_streaming(plain_clip, outputs_dir, MODELS_DIR)
    output_probe = _ffprobe(result["output_path"])
    output_duration = float(output_probe["format"]["duration"])

    frame_duration = input_duration / result["frames_processed"]
    assert abs(output_duration - input_duration) <= frame_duration + 0.05  # small encoder-overhead margin


def test_variable_frame_rate_source_duration_and_audio_stay_in_sync(tmp_path, outputs_dir):
    vfr_clip = tmp_path / "vfr.mp4"
    _make_vfr_clip(vfr_clip, tmp_path, size="320x240", with_audio=True)

    input_probe = _ffprobe(str(vfr_clip))
    input_duration = float(input_probe["format"]["duration"])

    result = video.process_video_streaming(str(vfr_clip), outputs_dir, MODELS_DIR)
    output_probe = _ffprobe(result["output_path"])
    output_duration = float(output_probe["format"]["duration"])
    video_stream = next(s for s in output_probe["streams"] if s["codec_type"] == "video")
    audio_stream = next(s for s in output_probe["streams"] if s["codec_type"] == "audio")

    frame_duration = input_duration / result["frames_processed"]
    assert abs(output_duration - input_duration) <= frame_duration + 0.1
    # audio and video streams end within a frame of each other -> "in sync"
    assert abs(float(video_stream["duration"]) - float(audio_stream["duration"])) <= frame_duration + 0.1


# ── portrait dimensions (rotation's externally-observable half — see README
#    Testing section for why real rotation *metadata* isn't fixture-tested) ─

def test_physically_portrait_clip_keeps_portrait_output_dimensions(tmp_path, outputs_dir):
    portrait_clip = tmp_path / "portrait.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=10",
        "-vf", "transpose=1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(portrait_clip), "-loglevel", "error",
    ], check=True)

    probe = video.probe_video(str(portrait_clip))
    assert probe.rotation_degrees == 0  # no metadata — physically transposed pixels
    assert probe.width == 240 and probe.height == 320  # genuinely portrait

    result = video.process_video_streaming(str(portrait_clip), outputs_dir, MODELS_DIR)
    output_probe = _ffprobe(result["output_path"])
    video_stream = next(s for s in output_probe["streams"] if s["codec_type"] == "video")
    assert video_stream["width"] == 240
    assert video_stream["height"] == 320


# ── MAX_CLIP_SECONDS truncation ──────────────────────────────────────────────

def test_max_clip_seconds_truncates_and_reports_it(tmp_path, outputs_dir):
    long_clip = tmp_path / "long.mp4"
    _make_clip(long_clip, duration=4, size="320x240", fps=10)

    result = video.process_video_streaming(str(long_clip), outputs_dir, MODELS_DIR, max_clip_seconds=1.5)
    assert result["truncated"] is True
    assert result["frames_processed"] < 40  # well under the full 4s*10fps=40 frames

    output_probe = _ffprobe(result["output_path"])
    output_duration = float(output_probe["format"]["duration"])
    assert output_duration < 2.5  # truncated well short of the full 4s


def test_no_cap_processes_the_full_clip(plain_clip, outputs_dir):
    result = video.process_video_streaming(plain_clip, outputs_dir, MODELS_DIR, max_clip_seconds=None)
    assert result["truncated"] is False


# ── cleanup ───────────────────────────────────────────────────────────────────

def test_silent_intermediate_is_removed_after_success(plain_clip, outputs_dir):
    video.process_video_streaming(plain_clip, outputs_dir, MODELS_DIR)
    leftovers = [f for f in os.listdir(outputs_dir) if "_silent" in f]
    assert leftovers == []


def test_temp_files_removed_when_ffmpeg_mux_fails(plain_clip, outputs_dir, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("simulated ffmpeg mux failure")

    monkeypatch.setattr(video, "_mux_with_ffmpeg", boom)

    with pytest.raises(RuntimeError, match="simulated ffmpeg mux failure"):
        video.process_video_streaming(plain_clip, outputs_dir, MODELS_DIR)

    assert os.listdir(outputs_dir) == []  # both silent intermediate and any partial final file gone


def test_temp_files_removed_when_frame_loop_raises(plain_clip, outputs_dir, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("simulated detector crash")

    monkeypatch.setattr(video.detector, "detect_all", boom)

    with pytest.raises(RuntimeError, match="simulated detector crash"):
        video.process_video_streaming(plain_clip, outputs_dir, MODELS_DIR)

    assert os.listdir(outputs_dir) == []


def test_ffmpeg_missing_raises_before_any_file_is_created(plain_clip, outputs_dir, monkeypatch):
    monkeypatch.setattr(video.shutil, "which", lambda name: None)

    with pytest.raises(video.FFmpegNotFoundError):
        video.process_video_streaming(plain_clip, outputs_dir, MODELS_DIR)

    assert os.listdir(outputs_dir) == []


# ── detection_stride (fast mode) ─────────────────────────────────────────────
#
# Found during real-clip profiling: face+plate detection dominate per-frame
# cost (~55% and ~26% of sequential time respectively on a real portrait
# clip), while tracking/redaction are cheap. Detecting every Kth frame and
# letting the tracker's existing gap-fill bridge the rest cuts detector calls
# roughly K-fold without touching the "every raw detection is redacted the
# frame it's found" guarantee — that guarantee only ever applies to frames
# detection actually runs on; a new object first appearing on a skipped
# frame is exposed for up to (K-1) frames, which is a deliberate, documented
# trade-off, not a violation of it.

def test_detection_stride_one_calls_detector_every_frame(plain_clip, outputs_dir, monkeypatch):
    calls = []
    real_detect_all = video.detector.detect_all
    monkeypatch.setattr(video.detector, "detect_all", lambda *a, **k: calls.append(1) or real_detect_all(*a, **k))

    result = video.process_video_streaming(plain_clip, outputs_dir, MODELS_DIR, detection_stride=1)
    assert len(calls) == result["frames_processed"]


def test_detection_stride_three_calls_detector_a_third_as_often(plain_clip, outputs_dir, monkeypatch):
    calls = []
    real_detect_all = video.detector.detect_all
    monkeypatch.setattr(video.detector, "detect_all", lambda *a, **k: calls.append(1) or real_detect_all(*a, **k))

    result = video.process_video_streaming(plain_clip, outputs_dir, MODELS_DIR, detection_stride=3)
    n = result["frames_processed"]
    assert len(calls) == -(-n // 3)  # ceil(n / 3): frames 0, 3, 6, ...


def test_raw_detection_guarantee_holds_on_every_detection_frame_with_stride(plain_clip, outputs_dir, monkeypatch):
    """
    With stride=2, a fixed synthetic box fed on every ACTUAL detection call
    must still be redacted (blurred, since it's fed as a "face") immediately
    on each frame detection runs — the stride skips detector CALLS, it must
    never delay redaction on the frames detection does run on. Proven by
    comparing actual output pixels against a second run with detection
    entirely disabled: frame 0 (always a detection frame, 0 % stride == 0
    for any stride) must differ at the detected region.
    """
    fixed_box = {"faces": [(10, 10, 30, 30)], "plates": [], "screens": []}
    monkeypatch.setattr(video.detector, "detect_all", lambda *a, **k: fixed_box)
    result = video.process_video_streaming(plain_clip, outputs_dir, MODELS_DIR, detection_stride=2)
    detection_frames = -(-result["frames_processed"] // 2)
    assert result["faces_found"] == detection_frames  # one box counted per actual detection call

    cap = video.cv2.VideoCapture(result["output_path"])
    ok, redacted_frame0 = cap.read()
    cap.release()
    assert ok

    no_detections = {"faces": [], "plates": [], "screens": []}
    monkeypatch.setattr(video.detector, "detect_all", lambda *a, **k: no_detections)
    result2 = video.process_video_streaming(plain_clip, outputs_dir, MODELS_DIR, detection_stride=2)
    cap2 = video.cv2.VideoCapture(result2["output_path"])
    ok2, unredacted_frame0 = cap2.read()
    cap2.release()
    assert ok2

    roi_redacted = redacted_frame0[10:40, 10:40]
    roi_unredacted = unredacted_frame0[10:40, 10:40]
    assert not (roi_redacted == roi_unredacted).all(), "the detected region must actually differ once blurred"


def test_detection_stride_default_is_one():
    import inspect
    sig = inspect.signature(video.process_video_streaming)
    assert sig.parameters["detection_stride"].default == 1
