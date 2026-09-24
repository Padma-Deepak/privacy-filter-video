"""
download_models.py — fetch the optional DNN face detector model that
project/detector.py uses if present (see detector.py's _get_dnn()).

Files are verified against a pinned SHA-256 checksum after download and
placed in project/models/, which is git-ignored — see the licence note
below for why these aren't committed to the repo directly.

    deploy.prototxt                              — network architecture only
    res10_300x300_ssd_iter_140000.caffemodel     — trained weights, ~10.2MB

Source: OpenCV's official face-detector sample
  https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
  https://github.com/opencv/opencv_3rdparty/tree/dnn_samples_face_detector_20170830

Licence — checked, and recorded here honestly rather than assumed:
  - deploy.prototxt lives in the main github.com/opencv/opencv repository,
    which is Apache License 2.0.
  - res10_300x300_ssd_iter_140000.caffemodel (the trained weights) lives in
    github.com/opencv/opencv_3rdparty, which has NO LICENSE file at all
    (confirmed: a GitHub API content listing on that branch returns no
    LICENSE, and the repo root has none either). The OpenCV community has
    publicly flagged this as an unresolved gap — see
    https://answers.opencv.org/question/212903/license-for-trained-dnn-face-detector-models/
    ("There is no license ... which renders them useless for commercial
    use"). Training data provenance is likewise undocumented upstream.
  This model is used here only as one candidate in Phase 0/1's local,
  non-commercial evaluation. Phase 2 explicitly re-evaluates face detectors
  by measured recall/FPS and picks defaults from that data (CLAUDE.md) — if
  this model is kept as a shipped default rather than a comparison point,
  its licence gap should be resolved first (e.g. swap for YuNet, which
  ships under OpenCV's own Apache-2.0 license with clear provenance).

Usage:
    python scripts/download_models.py
"""

from __future__ import annotations

import hashlib
import ssl
import sys
import urllib.request
from pathlib import Path

import certifi

urllib.request.install_opener(
    urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=certifi.where()))
    )
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "project" / "models"

FILES = [
    {
        "name": "deploy.prototxt",
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "sha256": "dcd661dc48fc9de0a341db1f666a2164ea63a67265c7f779bc12d6b3f2fa67e9",
    },
    {
        "name": "res10_300x300_ssd_iter_140000.caffemodel",
        "url": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "sha256": "2a56a11a57a4a295956b0660b4a3d76bbdca2206c4961cea8efe7d95c7cb2f2d",
    },
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(name: str, url: str, expected_sha256: str) -> None:
    dest = MODELS_DIR / name
    if dest.exists() and _sha256(dest) == expected_sha256:
        print(f"  {name}: already present and checksum matches, skipping")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {name} from {url}")
    urllib.request.urlretrieve(url, tmp)

    actual = _sha256(tmp)
    if actual != expected_sha256:
        tmp.unlink(missing_ok=True)
        sys.exit(
            f"Checksum mismatch for {name}:\n"
            f"  expected {expected_sha256}\n"
            f"  got      {actual}\n"
            f"Not installing a file that doesn't match the pinned checksum."
        )
    tmp.rename(dest)
    print(f"  {name}: OK ({dest.stat().st_size / 1e6:.2f} MB, sha256 verified)")


def main() -> None:
    print("Fetching optional DNN face detector model into project/models/ ...")
    for f in FILES:
        _fetch(f["name"], f["url"], f["sha256"])
    print("\nDone. project/detector.py will pick these up automatically on next run.")


if __name__ == "__main__":
    main()
