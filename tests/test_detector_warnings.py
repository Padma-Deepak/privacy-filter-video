"""
Tests for project/detector.py's detector-load-failure handling:
  - DNN model files simply being absent stays quiet (expected, documented
    fallback) — no warning, no error logged.
  - DNN model files present but failing to load is a real failure — logged
    as an error and surfaced via get_detector_warnings().
  - get_detector_warnings() aggregates whatever is currently set.
"""

import os
import sys

import pytest

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "project")
sys.path.insert(0, PROJECT_DIR)

# The module eagerly loads YOLO at import time using a path relative to its
# own file location (not CWD), so this works regardless of where pytest is
# invoked from — see detector.py's _YOLO_WEIGHTS_PATH.
import detector  # noqa: E402


@pytest.fixture(autouse=True)
def reset_dnn_singleton():
    """Each test gets a clean slate for the DNN loader's tri-state cache."""
    detector._dnn_net = None
    detector._dnn_warning = None
    yield
    detector._dnn_net = None
    detector._dnn_warning = None


def test_get_dnn_returns_none_quietly_when_files_absent(tmp_path):
    result = detector._get_dnn(str(tmp_path))
    assert result is None
    assert detector.get_detector_warnings() == []


def test_get_dnn_logs_and_warns_when_files_are_corrupt(tmp_path, caplog):
    (tmp_path / "deploy.prototxt").write_text("this is not a real prototxt")
    (tmp_path / "res10_300x300_ssd_iter_140000.caffemodel").write_bytes(b"not a real caffemodel")

    with caplog.at_level("ERROR", logger="detector"):
        result = detector._get_dnn(str(tmp_path))

    assert result is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "ERROR"

    warnings = detector.get_detector_warnings()
    assert len(warnings) == 1
    assert "Haar" in warnings[0]


def test_get_dnn_caches_failure_without_retrying(tmp_path, caplog):
    (tmp_path / "deploy.prototxt").write_text("garbage")
    (tmp_path / "res10_300x300_ssd_iter_140000.caffemodel").write_bytes(b"garbage")

    with caplog.at_level("ERROR", logger="detector"):
        detector._get_dnn(str(tmp_path))
        detector._get_dnn(str(tmp_path))
        detector._get_dnn(str(tmp_path))

    # Logged once, not once per call — otherwise a whole video's worth of
    # frames would each re-attempt and re-log the same failure.
    assert len(caplog.records) == 1


def test_get_detector_warnings_aggregates_dnn_and_yolo(monkeypatch):
    monkeypatch.setattr(detector, "_dnn_warning", "dnn broke")
    monkeypatch.setattr(detector, "_yolo_warning", "yolo broke")
    assert set(detector.get_detector_warnings()) == {"dnn broke", "yolo broke"}


def test_get_detector_warnings_empty_when_nothing_failed(monkeypatch):
    monkeypatch.setattr(detector, "_dnn_warning", None)
    monkeypatch.setattr(detector, "_yolo_warning", None)
    assert detector.get_detector_warnings() == []


def test_yolo_weights_path_is_independent_of_cwd():
    assert os.path.isabs(detector._YOLO_WEIGHTS_PATH)
    assert os.path.dirname(detector._YOLO_WEIGHTS_PATH) == os.path.dirname(os.path.abspath(detector.__file__))
