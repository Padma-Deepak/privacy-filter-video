"""Review must use only the configured face detector, never noisy fallbacks."""
import numpy as np
import pytest
from core import review
from core.faces import detect_faces
from core.profiles import load_profile


def test_review_defaults_are_faces_only(monkeypatch):
    expected = [(10, 20, 30, 40)]
    monkeypatch.setattr(review, 'detect_faces', lambda *args: expected)
    def forbidden(*args, **kwargs):
        pytest.fail('Legacy or non-face detector used in face-only review')
    for name in ('_detect_faces_haar', '_detect_faces_dnn', 'detect_plates', 'detect_screens'):
        monkeypatch.setattr(review.detector, name, forbidden)
    for name in ('creator', 'journalist'):
        profile = load_profile(name)
        assert profile['classes'] == ['faces']
        assert profile['detector'] == 'yunet'
        assert review.detect(np.zeros((100,100,3), dtype=np.uint8), '.', profile) == {'faces': expected}


def test_missing_yunet_does_not_silently_fallback(tmp_path):
    with pytest.raises(ValueError, match='download_yunet.py'):
        detect_faces(np.zeros((100,100,3), dtype=np.uint8), tmp_path)
