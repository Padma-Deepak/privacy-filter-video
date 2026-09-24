"""
Tests for project/app.py's _preview_data_url() — the fix for a real bug
found on an actual iPhone upload: the "Original" panel didn't load in
Chrome. Root cause: .mov uploads were embedded as data:video/quicktime,
which Chrome (and most non-Safari browsers) don't reliably play inline via
a <video> tag. Fix: remux (not re-encode) non-web-playable containers to a
clean MP4 for the preview only.
"""

import importlib
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))


def _reload_app(monkeypatch):
    for key in ("FLASK_DEBUG", "FLASK_HOST", "FLASK_PORT"):
        monkeypatch.delenv(key, raising=False)
    import app as app_module
    importlib.reload(app_module)
    return app_module


def _make_mov_clip(path):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=5",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         str(path), "-loglevel", "error"],
        check=True,
    )


def test_mp4_preview_is_not_remuxed(monkeypatch, tmp_path):
    app_module = _reload_app(monkeypatch)
    clip = tmp_path / "clip.mp4"
    _make_mov_clip(clip)  # container extension is what matters here, not actual codec

    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a) or subprocess.CompletedProcess(a, 0))
    url = app_module._preview_data_url(str(clip), "mp4", is_video=True)

    assert calls == []  # never touched ffmpeg
    assert url.startswith("data:video/mp4;base64,")


def test_mov_preview_is_remuxed_to_mp4(monkeypatch, tmp_path):
    app_module = _reload_app(monkeypatch)
    clip = tmp_path / "clip.mov"
    _make_mov_clip(clip)

    url = app_module._preview_data_url(str(clip), "mov", is_video=True)

    assert url.startswith("data:video/mp4;base64,")  # not video/quicktime
    assert "video/quicktime" not in url


def test_mov_preview_cleans_up_temp_remux_file(monkeypatch, tmp_path):
    app_module = _reload_app(monkeypatch)
    clip = tmp_path / "clip.mov"
    _make_mov_clip(clip)

    app_module._preview_data_url(str(clip), "mov", is_video=True)

    leftovers = [f for f in os.listdir(tmp_path) if f != "clip.mov"]
    assert leftovers == []


def test_preview_falls_back_to_raw_file_when_remux_fails(monkeypatch, tmp_path):
    app_module = _reload_app(monkeypatch)
    clip = tmp_path / "clip.mov"
    _make_mov_clip(clip)

    def boom(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "ffmpeg")

    monkeypatch.setattr(subprocess, "run", boom)
    url = app_module._preview_data_url(str(clip), "mov", is_video=True)

    assert url.startswith("data:video/quicktime;base64,")  # fell back to the raw file's own mime


def test_image_preview_never_touches_ffmpeg(monkeypatch, tmp_path):
    app_module = _reload_app(monkeypatch)
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"not a real jpeg but that's fine, we never read the bytes as an image here")

    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))
    url = app_module._preview_data_url(str(img), "jpg", is_video=False)

    assert calls == []
    assert url.startswith("data:image/jpeg;base64,")
