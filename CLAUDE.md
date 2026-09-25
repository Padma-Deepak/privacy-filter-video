# CLAUDE.md — Privacy Filter (Video)

This file is read at the start of every session. Follow it. If something here conflicts with a phase prompt in `PHASES.md`, ask before proceeding.

## What this project is

A local-first video and image anonymizer for **creators** (posting street or travel footage) and **journalists** (publishing footage with bystanders or sources in it).

A user uploads a clip, sees detected face tracks, chooses who may remain visible,
and exports a clean video **with audio intact** (or muted, in Journalist mode).
Two presets, Creator and Journalist, share one face-only review pipeline. Plate
and screen work remains available only in the legacy comparison route.

The goal is a **portfolio project for computer vision roles**. That means measured results, honest limitations and reproducible experiments matter more than feature count. Every claim in the README must be backed by a number produced by a script in this repo.

## Current implementation (verify against the code)

The default `/review` workflow is a face-only selective anonymizer:

- YuNet detects faces at the confidence configured by the active profile.
- `ClassTracker` assigns per-job IDs, smooths boxes and fills short detection gaps.
- Users draw around tracks to keep visible; every unselected track hides by default.
- Analysis records geometry once and export reuses it at full resolution.
- FFmpeg produces H.264 video, keeps or mutes audio and strips source metadata.
- Session-owned background jobs support progress, cancellation and expiry.
- Creator and Journalist YAML profiles control filtering, padding, audio and reports.

The original Haar/DNN face, Haar/contour plate and YOLO screen implementation is
retained at `/legacy` for historical comparison. It is not the default product path.

Read the actual code before changing anything. Where this file and the code disagree, the code is the truth and this file should be corrected.

## Key decisions (already made)

- **Detection stack:** the current review default is YuNet 2023mar at confidence
  0.8, chosen after a same-dataset comparison with Haar and SSD DNN. Continue to
  choose thresholds or replacement models from measured recall, precision, false
  positives and FPS rather than reputation. Plates and screens remain legacy-only.
- **Tracking:** use a detector-agnostic tracker (`supervision.ByteTrack` or equivalent) so the face detector can be swapped freely. Do not couple tracking to the Ultralytics `.track()` API.
- **Licensing:** Ultralytics is AGPL-3.0. Keep the repo public and add a `LICENSE` and a "Licenses" section in the README listing every model and dataset with its licence. Before using any third-party weights or dataset, check its licence and record it.
- **DNN face model licence gap:** the optional `res10_300x300_ssd_iter_140000.caffemodel` (fetched by `scripts/download_models.py`, see `project/README.md`) has no upstream licence file at all — confirmed via the GitHub API, and a known gap flagged by the OpenCV community itself. Do not bundle these weights in the Docker image or the hosted demo (Phases 5-6) unless this is resolved first (e.g. a clear licence is found, or the model is swapped for one with clear provenance, such as YuNet).
- **YuNet licensing:** `face_detection_yunet_2023mar.onnx` is fetched from a
  pinned OpenCV Zoo revision, checksum-verified and covered by the MIT licence in
  `docs/YUNET_LICENSE.txt`. The weights remain ignored by Git.
- **Local-first:** the tool must work fully offline. No cloud APIs, no telemetry.
- **Compute:** develop and benchmark locally. If fine-tuning is too slow on CPU, put a Colab/Kaggle notebook in `training/` and commit the resulting weights (or a download script) plus the exact training config.
- **Demo:** a hosted demo (Hugging Face Space, Docker) must work with bundled sample clips **and** accept any user upload. See the demo rules below.

## Non-negotiables

1. **Privacy safety over speed.** Redaction must be irreversible: strong blur (large kernel, applied after box padding) or solid fill. Never a light blur. Never keep an unredacted copy of a frame on disk.
2. **Recall is the headline metric.** A missed face is a failure. Prefer a lower confidence threshold plus a review step over a high threshold with silent misses.
3. **Never modify the user's original file.** Work on copies. Delete uploads and temp files after processing, including on errors and timeouts.
4. **No claims without numbers.** If a README sentence contains "accurate", "fast" or "robust", it needs a metric next to it or it gets removed.
5. **Be honest about limits.** Clothing, gait, voice, background and context can still identify a person. The README must say so.
6. **Reproducibility.** Pin dependencies. Every number in the README comes from a script in `eval/` that can be re-run with one command. Set random seeds.
7. **Small, reviewable changes.** One phase at a time, one logical change per commit.

## Repository layout

```
project/                 Flask app
  app.py
  review_api.py          selective-review API
  core/                  detection, tracking, jobs, redaction, video I/O
  profiles/              creator.yaml, journalist.yaml
eval/                    metrics scripts, dataset prep, results/*.csv
tests/                   pytest
docs/results/            before/after images, GIFs, metric tables
```

## Working rules for Claude Code

- **Start every phase by reading the relevant code and writing a short plan.** Show the plan and wait for approval before large refactors.
- Run the existing app and any tests before changing behaviour, so regressions are visible.
- Add or update tests with each change (pytest). Use tiny synthetic frames for unit tests, not large media files.
- Use type hints and docstrings on new functions. Keep functions small.
- Config over constants: thresholds, padding, blur strength and class lists live in profile YAML files, not hard-coded.
- Don't commit large binaries. Weights and sample videos are either small (<10 MB), tracked with Git LFS, or fetched by a script. Keep `.gitignore` current.
- Commit messages: imperative mood, one line summary, short body explaining why.
- After each phase, update `README.md` and the metrics table only with numbers you actually produced.
- When unsure whether something is in scope, check `PHASES.md`, then ask.

## Definitions (use these exact meanings)

- **Face recall:** matched ground-truth faces / all ground-truth faces (IoU ≥ 0.5), reported per dataset and per size bucket (small, medium, large).
- **Frame-level leak rate:** fraction of ground-truth boxes not sufficiently covered by a redaction region in a frame. "Covered" means at least 90% of the ground-truth box area lies inside the padded redaction region.
- **Track-level leak rate:** fraction of ground-truth tracks that leak in **at least one** frame. This is the privacy-critical metric.
- **FPS:** end-to-end frames per second including decode, detect, track, redact and encode, on a stated machine (CPU model, GPU or none), at a stated resolution.

## Demo rules (hosted, accepts uploads)

The local tool's selling point is that nothing leaves your machine. A hosted demo cannot claim that, so:

- Show a clear notice on the upload page: "This hosted demo sends your file to a server. For sensitive footage, run it locally."
- Limits: file size, clip duration (configurable by environment variable), allowed MIME types and extensions, request timeout, basic rate limiting.
- Use `secure_filename` or random names, never trust user file names, delete files immediately after the response, and never log file contents or file names.
- Ship 3 to 5 sample clips with permissive licences (Pexels/Pixabay or the team's own footage). Record the source and licence of each in `demo/SAMPLES.md`.
- The Space must build from the Dockerfile with no manual steps.

## Definition of done for any phase

- Code runs from a clean checkout with the documented commands.
- Tests pass.
- Any new metric is produced by an `eval/` script and saved under `eval/results/`.
- README is updated (what changed, how to run it, what the numbers are).
- Known limitations are listed, not hidden.
