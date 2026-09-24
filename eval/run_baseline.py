"""
run_baseline.py — score the CURRENT detectors in project/detector.py against
the seeded WIDER FACE subset built by prepare_wider_face.py, and write
eval/results/baseline.csv.

This does not change or improve any detector — it only measures what
project/detector.py already does today, so later phases have a "before"
number. Run `python eval/prepare_wider_face.py` first if
eval/data/wider_face_subset/ground_truth.json doesn't exist yet.

Face detectors evaluated (against real ground truth, precision/recall/FPS):
  - haar            : detector._detect_faces_haar only
  - dnn             : detector._detect_faces_dnn only (skipped, with a note in
                       the CSV, if project/models/*.caffemodel is absent)
  - haar+dnn (prod) : detector.detect_faces — the actual function app.py calls

Plate and screen detectors are also run and timed (FPS, raw detection count)
for reference, but WIDER FACE has no plate/screen ground truth, so their
precision/recall are reported as NaN with an explanatory note rather than a
fabricated number — a real plate/screen ground-truth set is Phase 2's job.

Usage:
    python eval/run_baseline.py
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
PROJECT_DIR = REPO_ROOT / "project"
SUBSET_DIR = EVAL_DIR / "data" / "wider_face_subset"
RESULTS_DIR = EVAL_DIR / "results"

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(EVAL_DIR))

import cv2  # noqa: E402

import detector  # noqa: E402  (project/detector.py)
from metrics import precision_recall, recall_by_size_bucket  # noqa: E402


def _cpu_model() -> str:
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=True,
            )
            return out.stdout.strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


def _load_subset() -> dict:
    gt_path = SUBSET_DIR / "ground_truth.json"
    if not gt_path.exists():
        sys.exit(
            f"No subset found at {gt_path}.\n"
            f"Run `python eval/prepare_wider_face.py` first."
        )
    return json.loads(gt_path.read_text())


def _load_images(manifest: dict) -> list[tuple[dict, "cv2.Mat"]]:
    loaded = []
    for rec in manifest["images"]:
        path = SUBSET_DIR / "images" / rec["file"]
        img = cv2.imread(str(path))
        if img is None:
            print(f"  WARNING: could not read {path}, skipping")
            continue
        loaded.append((rec, img))
    return loaded


def _eval_face_detector(name: str, detect_fn, records_and_images: list) -> dict:
    all_gt, all_pred = [], []
    n_images = len(records_and_images)
    t0 = time.perf_counter()
    for i, (rec, img) in enumerate(records_and_images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pred_boxes = detect_fn(img, gray)
        all_gt.append([tuple(b) for b in rec["boxes"]])
        all_pred.append([tuple(b) for b in pred_boxes])
        if (i + 1) % 50 == 0 or (i + 1) == n_images:
            print(f"    {name}: {i + 1}/{n_images} images ({time.perf_counter() - t0:.1f}s elapsed)", flush=True)
    elapsed = time.perf_counter() - t0

    pr = precision_recall(all_gt, all_pred)
    by_bucket = recall_by_size_bucket(all_gt, all_pred)
    n_images = len(records_and_images)
    return {
        "detector": name,
        "n_images": n_images,
        "n_gt_boxes": sum(len(g) for g in all_gt),
        "n_pred_boxes": sum(len(p) for p in all_pred),
        "tp": pr["tp"], "fp": pr["fp"], "fn": pr["fn"],
        "precision": pr["precision"], "recall": pr["recall"],
        "recall_small": by_bucket["small"]["recall"],
        "recall_medium": by_bucket["medium"]["recall"],
        "recall_large": by_bucket["large"]["recall"],
        "fps": n_images / elapsed if elapsed > 0 else float("nan"),
        "notes": "",
    }


def _time_only(name: str, detect_fn, records_and_images: list, note: str) -> dict:
    """For plate/screen detectors: no ground truth on this dataset, so just
    time it and count raw detections, and say so explicitly in `notes`."""
    n_pred = 0
    n_images = len(records_and_images)
    t0 = time.perf_counter()
    for i, (_, img) in enumerate(records_and_images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        n_pred += len(detect_fn(img, gray))
        if (i + 1) % 50 == 0 or (i + 1) == n_images:
            print(f"    {name}: {i + 1}/{n_images} images ({time.perf_counter() - t0:.1f}s elapsed)", flush=True)
    elapsed = time.perf_counter() - t0
    nan = float("nan")
    return {
        "detector": name,
        "n_images": n_images,
        "n_gt_boxes": nan, "n_pred_boxes": n_pred,
        "tp": nan, "fp": nan, "fn": nan,
        "precision": nan, "recall": nan,
        "recall_small": nan, "recall_medium": nan, "recall_large": nan,
        "fps": n_images / elapsed if elapsed > 0 else nan,
        "notes": note,
    }


def main() -> None:
    manifest = _load_subset()
    print(f"Loaded subset: seed={manifest['seed']}, {manifest['n_selected']} images, "
          f"{manifest['total_gt_boxes']} ground-truth face boxes")

    print("Reading images into memory...")
    records_and_images = _load_images(manifest)
    print(f"  {len(records_and_images)} images decoded successfully")

    models_dir = str(PROJECT_DIR / "models")
    dnn_available = detector._get_dnn(models_dir) is not None

    rows = []

    print("Evaluating: haar (face)")
    rows.append(_eval_face_detector(
        "faces_haar", lambda img, gray: detector._detect_faces_haar(gray), records_and_images
    ))

    if dnn_available:
        print("Evaluating: dnn (face)")
        rows.append(_eval_face_detector(
            "faces_dnn",
            lambda img, gray: detector._detect_faces_dnn(img, models_dir),
            records_and_images,
        ))
    else:
        print("Skipping: dnn (face) — project/models/*.caffemodel not present")
        nan = float("nan")
        rows.append({
            "detector": "faces_dnn", "n_images": len(records_and_images),
            "n_gt_boxes": nan, "n_pred_boxes": nan, "tp": nan, "fp": nan, "fn": nan,
            "precision": nan, "recall": nan,
            "recall_small": nan, "recall_medium": nan, "recall_large": nan,
            "fps": nan,
            "notes": "skipped: deploy.prototxt / res10_300x300_ssd_iter_140000.caffemodel not found in project/models/",
        })

    if dnn_available:
        print("Evaluating: haar+dnn (production detect_faces)")
        rows.append(_eval_face_detector(
            "faces_haar_dnn_production",
            lambda img, gray: detector.detect_faces(img, gray, models_dir),
            records_and_images,
        ))
    else:
        # detector.detect_faces() with no DNN net loaded degrades to exactly
        # the Haar-only path (see detector.py:149-150 and _get_dnn's None
        # return) — re-running the same expensive detectMultiScale pass a
        # second time would burn CPU for a byte-identical result.
        print("Skipping: haar+dnn (production) — DNN unavailable, identical to faces_haar")
        row = dict(rows[0])
        row["detector"] = "faces_haar_dnn_production"
        row["notes"] = "DNN unavailable at run time, so production detect_faces() == faces_haar (not re-run)"
        rows.append(row)

    print("Timing: plates (haar+contour, production) — no ground truth on this dataset")
    rows.append(_time_only(
        "plates_haar_contour_production",
        lambda img, gray: detector.detect_plates(gray),
        records_and_images,
        "no plate ground truth in WIDER FACE; count/FPS only, see PHASES.md Phase 2",
    ))

    print("Timing: screens (YOLOv8n, production) — no ground truth on this dataset")
    rows.append(_time_only(
        "screens_yolov8n_production",
        lambda img, gray: detector.detect_screens(img),
        records_and_images,
        "no screen ground truth in WIDER FACE; count/FPS only, see PHASES.md Phase 2",
    ))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "baseline.csv"
    fieldnames = [
        "detector", "n_images", "n_gt_boxes", "n_pred_boxes", "tp", "fp", "fn",
        "precision", "recall", "recall_small", "recall_medium", "recall_large",
        "fps", "notes",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {out_path}")
    print(f"Hardware: {_cpu_model()}, {platform.system()} {platform.release()}, no GPU used by these detectors")
    print()
    for row in rows:
        print(f"  {row['detector']:32s} precision={row['precision']!s:>8} "
              f"recall={row['recall']!s:>8} fps={row['fps']:.2f}")


if __name__ == "__main__":
    main()
