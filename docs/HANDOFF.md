# Current handoff — Privacy Filter

Updated after the selective-review and YuNet work. Read the root `README.md`
for setup, measured results and user-facing behavior; read `CLAUDE.md` for
engineering rules and `PHASES.md` for roadmap status.

## Current product

The default `/review` route is a local, face-only anonymizer. A user uploads an
image or video, draws around people who may remain visible, previews every other
detected face redacted, and exports a full-resolution H.264 video or PNG. Creator
uses strong blur and keeps audio. Journalist uses solid masks, larger padding
and muted audio. Both remove source metadata and produce a report.

The `/legacy` route preserves the original automatic Haar/DNN face,
Haar/contour plate and YOLO screen pipeline for comparison. Do not describe that
route as the current default.

## Architecture

- `project/review_api.py`: session ownership, CSRF protection and review API.
- `project/core/jobs.py`: one-worker job queue, progress, cancellation, expiry
  and cleanup.
- `project/core/faces.py`: thread-safe YuNet adapter. Missing weights fail with
  a setup instruction; there is no silent Haar fallback.
- `project/core/review.py`: streamed analysis, stored track geometry, previews,
  full-resolution rendering, FFmpeg export and report generation.
- `project/core/tracking.py`: ByteTrack-assisted, velocity-aware IoU continuity,
  smoothing, padding and gap filling.
- `project/core/selection.py`: unambiguous draw-to-track matching and fail-closed
  keep/hide validation.
- `project/static/review.js` and `preview.js`: native video playback, canvas
  interaction, geometry prefetch and approximate live redaction.

The important privacy invariant is unchanged: every raw detection is emitted for
redaction on the frame where it occurs. Tracking may add coverage but never
removes a raw detection. Final regions use the padded union of raw and smoothed
boxes.

## Detector decision and results

The review pipeline uses `face_detection_yunet_2023mar.onnx` at confidence 0.8.
It is downloaded by `scripts/download_yunet.py` from a pinned OpenCV Zoo commit,
verified against SHA-256, ignored by Git and covered by the MIT licence stored at
`docs/YUNET_LICENSE.txt`.

On the identical 300-image, 3,022-box WIDER FACE subset:

| Detector | Precision | Recall | False positives | FPS |
|---|--:|--:|--:|--:|
| Haar+DNN legacy | 49.8% | 27.2% | 828 | 4.29 |
| YuNet 0.6 | 87.3% | 63.9% | 281 | 37.87 |
| YuNet 0.8, current | 98.0% | 47.9% | 29 | 39.18 |

These are detector-only local CPU measurements. Reproduce them with
`python eval/run_baseline.py` and `python eval/compare_yunet.py` after preparing
the dataset. YuNet 0.8 was selected to eliminate severe interactive false boxes;
its reduced recall, especially on small faces, remains an explicit limitation.

## Setup and verification

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r project/requirements.txt -r eval/requirements.txt
python scripts/download_yunet.py
python -m pytest -q
node --test tests/js/test_preview.cjs
python project/app.py
```

Expected current result: 103 Python tests and 4 JavaScript tests pass. FFmpeg
with `libx264` is required for video.

## Completed work

- Original audit and reproducible WIDER FACE baseline.
- Full-resolution streamed video processing, explicit rotation handling,
  H.264 output, audio keep/mute and cleanup.
- Motion tracking with stable per-job IDs, smoothing and gap filling.
- Session-owned background analysis/export jobs with progress and cancellation.
- Draw-to-keep face selection, advanced track choices and manual hide regions.
- Live canvas preview that hides unselected detected faces.
- Creator and Journalist YAML profiles.
- Metadata stripping and JSON/HTML reports.
- YuNet replacement for noisy Haar review detections.
- Security hardening for Flask configuration and detector failures.

## Open work

1. Annotate 5–10 short videos as described in `eval/video_gt/README.md`.
2. Implement and report frame-level and track-level leak-rate evaluation.
3. Reconsider YuNet confidence using annotated video evidence; 0.8 favors
   precision over the project's long-term high-recall goal.
4. Test more real rotation-tagged phone videos.
5. Improve re-identification through crossings and re-entry.
6. Replace and evaluate the legacy plate detector before enabling plates in the
   main review interface.
7. Add Docker, CI, repository licence, security documentation and any hosted
   demo only after its privacy notice and cleanup behavior are ready.

## Data and repository rules

Never commit personal footage, model weights, WIDER FACE data, job directories,
virtual environments or debug renders. The OpenCV SSD weights remain baseline
only because their upstream weights repository has no licence file. Document the
source and licence of every new model, dataset and demo clip.
