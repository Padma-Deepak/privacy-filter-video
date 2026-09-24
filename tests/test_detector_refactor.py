"""
Regression test for extracting detect_all() out of process_frame().

This is a pure refactor (Phase 1 Stage 1) meant to change nothing about
image-path behaviour — project/core/video.py needs the raw boxes before
redaction so it can route them through tracking, but process_frame() and
process_image() must keep detecting and redacting in one step exactly as
before. These tests prove that split is behaviourally invisible.
"""

import os
import sys

import cv2
import numpy as np

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "project")
sys.path.insert(0, PROJECT_DIR)

import detector  # noqa: E402

SAMPLE_IMAGE = os.path.join(os.path.dirname(__file__), "..", "images_CV_AAT", "images (1).jpg")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")


def test_detect_all_matches_individual_detector_calls():
    image = cv2.imread(SAMPLE_IMAGE)
    assert image is not None, f"could not read sample image {SAMPLE_IMAGE}"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    expected_faces = detector.detect_faces(image, gray, MODELS_DIR)
    expected_plates = detector.detect_plates(gray)
    expected_screens = detector.detect_screens(image)

    result = detector.detect_all(image, MODELS_DIR)

    assert sorted(result["faces"]) == sorted(expected_faces)
    assert sorted(result["plates"]) == sorted(expected_plates)
    assert sorted(result["screens"]) == sorted(expected_screens)


def test_process_frame_output_unchanged_by_the_split():
    image = cv2.imread(SAMPLE_IMAGE)
    assert image is not None

    output, counts = detector.process_frame(image, MODELS_DIR)

    detections = detector.detect_all(image, MODELS_DIR)
    expected = image.copy()
    for (x, y, w, h) in detections["faces"]:
        detector.apply_gaussian_blur(expected, x, y, w, h)
    for (x, y, w, h) in detections["plates"]:
        detector.apply_black_mask(expected, x, y, w, h)
    for (x, y, w, h) in detections["screens"]:
        detector.apply_pixelation(expected, x, y, w, h)

    assert counts == {
        "faces": len(detections["faces"]),
        "plates": len(detections["plates"]),
        "screens": len(detections["screens"]),
    }
    assert np.array_equal(output, expected)


def test_detect_all_works_with_shared_executor():
    from concurrent.futures import ThreadPoolExecutor

    image = cv2.imread(SAMPLE_IMAGE)
    assert image is not None

    direct = detector.detect_all(image, MODELS_DIR)
    with ThreadPoolExecutor(max_workers=3) as pool:
        via_pool = detector.detect_all(image, MODELS_DIR, executor=pool)

    assert sorted(direct["faces"]) == sorted(via_pool["faces"])
    assert sorted(direct["plates"]) == sorted(via_pool["plates"])
    assert sorted(direct["screens"]) == sorted(via_pool["screens"])
