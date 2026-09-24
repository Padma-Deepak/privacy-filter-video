"""
prepare_wider_face.py — build a fixed, seeded subset of WIDER FACE validation
images with ground-truth face boxes, for eval/run_baseline.py to score against.

Why this doesn't use the `datasets` library: the Hugging Face mirror
(CUHK-CSE/wider_face) ships a legacy loading script that hardcodes
`hf://datasets/wider_face@main/...` URLs. Current huggingface_hub versions
reject that as an invalid repo ID ("must be 'namespace/name'"), so
`load_dataset("CUHK-CSE/wider_face", ...)` fails outright — this is a bug in
that dataset repo's script, not an environment issue (reproduced on a fresh
venv). It also requires `trust_remote_code=True` to run arbitrary Python from
the Hub. Instead, this script downloads the same two archives directly over
plain HTTPS from the Hub's resolve endpoint (public, no auth, no code
execution) and parses the official WIDER FACE annotation text format itself:

    https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/WIDER_val.zip
    https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/wider_face_split.zip

WIDER_val.zip is ~360MB and is cached under eval/data/.cache/ after the first
run — re-running this script does not re-download it. eval/data/ is
git-ignored; nothing here is committed. Everything downstream (the 300-image
subset and its ground truth) is regenerated deterministically from `--seed`.

Usage:
    python eval/prepare_wider_face.py                  # n=300, seed=42
    python eval/prepare_wider_face.py --n 50 --seed 7   # smaller/different sample
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

import certifi

# The python.org macOS build (unlike Homebrew/system Python) does not wire the
# system CA trust store into urllib's default SSL context, which makes
# urlretrieve fail with CERTIFICATE_VERIFY_FAILED even though e.g. curl works
# fine. Use certifi's bundle explicitly rather than disabling verification.
urllib.request.install_opener(
    urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=certifi.where()))
    )
)

WIDER_VAL_URL = "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/WIDER_val.zip"
SPLIT_URL = "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/wider_face_split.zip"
ANNOTATION_MEMBER = "wider_face_split/wider_face_val_bbx_gt.txt"
IMAGE_PREFIX = "WIDER_val/images/"

EVAL_DIR = Path(__file__).resolve().parent
CACHE_DIR = EVAL_DIR / "data" / ".cache"
SUBSET_DIR = EVAL_DIR / "data" / "wider_face_subset"


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  cached: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")

    def _report(block_num, block_size, total_size):
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        pct = 100 * done / total_size
        sys.stdout.write(f"\r  {pct:5.1f}%  ({done / 1e6:.1f} / {total_size / 1e6:.1f} MB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=_report)
    sys.stdout.write("\n")
    tmp.rename(dest)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_annotations(text: str) -> dict[str, list[tuple[int, int, int, int]]]:
    """
    Parse the official WIDER FACE bbox annotation format:

        <image relative path>
        <num boxes>
        x1 y1 w h blur expression illumination invalid occlusion pose
        ... (num boxes lines, or exactly one all-zero line if num boxes == 0)

    Boxes flagged `invalid` (10th field == 1) are dropped — the WIDER FACE
    devkit defines these as not usable as ground truth.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    boxes_by_image: dict[str, list[tuple[int, int, int, int]]] = {}
    i = 0
    while i < len(lines):
        image_rel = lines[i]
        i += 1
        n_boxes = int(lines[i])
        i += 1
        boxes = []
        n_lines_to_read = max(n_boxes, 1)  # a 0-face image still has one dummy line
        for j in range(n_lines_to_read):
            fields = lines[i + j].split()
            x, y, w, h = (int(fields[k]) for k in range(4))
            invalid = int(fields[7]) if len(fields) > 7 else 0
            if n_boxes > 0 and invalid == 0 and w > 0 and h > 0:
                boxes.append((x, y, w, h))
        i += n_lines_to_read
        boxes_by_image[image_rel] = boxes
    return boxes_by_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300, help="subset size (default 300)")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed (default 42)")
    parser.add_argument("--force", action="store_true", help="rebuild even if the subset already exists")
    args = parser.parse_args()

    if SUBSET_DIR.exists() and not args.force:
        gt_path = SUBSET_DIR / "ground_truth.json"
        if gt_path.exists():
            existing = json.loads(gt_path.read_text())
            if existing.get("seed") == args.seed and existing.get("n_requested") == args.n:
                print(f"Subset already built at {SUBSET_DIR} (seed={args.seed}, n={args.n}). Use --force to rebuild.")
                return

    print("Step 1/4: downloading WIDER FACE validation archives (cached after first run)")
    val_zip = CACHE_DIR / "WIDER_val.zip"
    split_zip = CACHE_DIR / "wider_face_split.zip"
    _download(WIDER_VAL_URL, val_zip)
    _download(SPLIT_URL, split_zip)

    print("Step 2/4: parsing official annotations")
    with zipfile.ZipFile(split_zip) as zf:
        annotation_text = zf.read(ANNOTATION_MEMBER).decode("utf-8")
    boxes_by_image = _parse_annotations(annotation_text)
    all_images = sorted(boxes_by_image.keys())
    print(f"  {len(all_images)} images with annotations in the full validation split")

    print(f"Step 3/4: sampling {args.n} images with seed={args.seed}")
    n = min(args.n, len(all_images))
    if n < args.n:
        print(f"  WARNING: requested {args.n} but only {len(all_images)} are available; using {n}")
    selected = sorted(random.Random(args.seed).sample(all_images, n))

    print("Step 4/4: extracting selected images and writing ground truth")
    if SUBSET_DIR.exists():
        shutil.rmtree(SUBSET_DIR)
    images_dir = SUBSET_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Local import so tests/other scripts don't need cv2 just to import this module's helpers.
    import cv2

    records = []
    total_boxes = 0
    with zipfile.ZipFile(val_zip) as zf:
        for rel_path in selected:
            member = IMAGE_PREFIX + rel_path
            flat_name = rel_path.replace("/", "__")
            out_path = images_dir / flat_name
            with zf.open(member) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

            img = cv2.imread(str(out_path))
            if img is None:
                print(f"  WARNING: could not decode {rel_path}, dropping from subset")
                out_path.unlink(missing_ok=True)
                continue
            h, w = img.shape[:2]
            boxes = boxes_by_image[rel_path]
            total_boxes += len(boxes)
            records.append({
                "file": flat_name,
                "source_path": rel_path,
                "width": w,
                "height": h,
                "boxes": [list(b) for b in boxes],
            })

    manifest = {
        "seed": args.seed,
        "n_requested": args.n,
        "n_selected": len(records),
        "total_gt_boxes": total_boxes,
        "source": "https://huggingface.co/datasets/CUHK-CSE/wider_face (validation split, official WIDER FACE annotations)",
        "wider_val_zip_sha256": _sha256(val_zip),
        "wider_face_split_zip_sha256": _sha256(split_zip),
        "box_format": "[x, y, w, h] in pixel coordinates, top-left origin",
        "images": records,
    }
    (SUBSET_DIR / "ground_truth.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nDone: {len(records)} images, {total_boxes} ground-truth face boxes")
    print(f"Subset:        {SUBSET_DIR}")
    print(f"Ground truth:  {SUBSET_DIR / 'ground_truth.json'}")


if __name__ == "__main__":
    main()
