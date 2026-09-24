# PHASES.md — build plan for Claude Code

Work through these in order. Give Claude Code **one phase prompt at a time**, review the result, commit, then move on. Each prompt below is written to be pasted as-is. Make sure `CLAUDE.md` is in the repo root first.

Why the order matters: tracking (Phase 1) is required by click-to-select (Phase 3), and the evaluation harness (Phase 0) is what lets you prove every later improvement.

| Phase | Outcome |
|---|---|
| 0 | Audit, baseline numbers and evaluation data |
| 1 | Streaming video pipeline with tracking, full-res output and audio |
| 2 | Better detectors chosen by data, plate detector fine-tuned |
| 3 | Click-to-select: keep or hide each tracked face or plate |
| 4 | Creator and Journalist profiles, metadata stripping, redaction report |
| 5 | Docker, tests, CI, README polish |
| 6 | Hosted demo with samples and uploads |
| 7 | Stretch features |

---

## Phase 0 — Audit and baseline

**Why:** you cannot show an improvement without a "before" number, and the notes so far are based only on the README, not the code.

**Prompt:**

> Read `CLAUDE.md`. Then do a full audit of this repo without changing any behaviour.
>
> 1. Read `project/app.py`, `project/detector.py`, `project/requirements.txt`, `project/README.md` and `instructions.txt`. Summarise in `docs/AUDIT.md`: what each file does, how detectors are called, how video is processed (frame loop, size caps, codec, temp files), what is hard-coded, and any bugs, security problems or fragile spots you find. Include file and line references.
> 2. Create a virtualenv, install the requirements and run the app on the sample images in `images_CV_AAT/`. Record what works and what fails.
> 3. Build the evaluation scaffold in `eval/`:
>    - `eval/prepare_wider_face.py`: downloads or documents how to obtain WIDER FACE validation, and creates a fixed 300-image subset (seeded) with ground-truth boxes in a simple JSON format.
>    - `eval/metrics.py`: precision, recall (IoU ≥ 0.5), and recall by face-size bucket.
>    - `eval/run_baseline.py`: runs the **current** face detector (Haar, and the DNN if its model files are present) and the **current** plate/screen detectors on the subset, and writes `eval/results/baseline.csv` with recall, precision and FPS.
> 4. Create `eval/video_gt/README.md` explaining how to annotate 5 to 10 short clips (5 to 15 s each) in CVAT with track IDs for faces and plates, and the export format expected. Do not create fake annotations.
> 5. Write a baseline section in the README that includes only numbers you actually measured, and state the hardware.
>
> Show me the audit and the plan before writing eval code. Ask if any dataset needs manual download.

**Done when:** `docs/AUDIT.md` exists, `eval/results/baseline.csv` has real numbers, and the annotation instructions are ready.

**Your manual task:** film or download 5 to 10 short clips (crowds, traffic, dashcam, indoor) and annotate them in CVAT. This is the slowest part of the project. Start early, and keep clips short.

---

## Phase 1 — Video foundation

**Why:** the current pipeline treats every frame as independent, which causes flicker and misses. Tracking gives each person a stable ID, and that unlocks selective redaction later.

**Prompt:**

> Read `CLAUDE.md` and `docs/AUDIT.md`. Refactor video handling into `project/core/video.py` and `project/core/tracking.py`:
>
> 1. **Streaming:** read frames with OpenCV (or PyAV) one at a time, never load the whole clip into memory. Remove the 12-second cap and the forced 960 px downscale from the default path. Run detection on a downscaled copy if needed for speed, but scale boxes back and apply redaction at the original resolution. Put any duration cap behind an environment variable (`MAX_CLIP_SECONDS`, default unlimited) for the hosted demo.
> 2. **Tracking:** add a detector-agnostic tracker (`supervision.ByteTrack` or equivalent) that assigns persistent IDs to faces and plates. Keep the track alive for N frames after the detector misses it (configurable, `track_buffer`) and keep applying redaction to the last known box during that gap. Expose IDs in the per-frame results.
> 3. **Frame skipping (fast mode):** option to run detection every K frames and propagate boxes between detections with the tracker. Default K=1.
> 4. **Audio and codec:** write the redacted video to a temp file, then use ffmpeg to mux the **original audio** back and encode H.264 (`libx264`, `yuv420p`, `+faststart`) so it plays in browsers. If the input has no audio, skip muxing. Check ffmpeg is installed and give a clear error if it is not.
> 5. **Box smoothing:** smooth box positions and sizes with an exponential moving average per track to reduce jitter, and pad boxes by a configurable percentage (default 15%).
> 6. **Cleanup:** temp files are deleted on success, error and timeout.
> 7. Keep the existing image path working. Add pytest tests for tracking persistence (synthetic moving boxes), the gap-fill behaviour and the audio mux (use a generated test tone and colour bars from ffmpeg).
> 8. Run `eval/run_baseline.py` again on the video clips if annotations exist and add a "with tracking" row to the results.
>
> Show a short plan before you start and commit in small steps.

**Done when:** a 60-second 1080p clip processes without hitting a size cap, the output keeps its audio, the boxes don't flicker on the sample clips, and tests pass.

---

## Phase 2 — Detectors and evaluation

**Why:** this is the core of the CV story. Choose models from measured data, and fine-tune one yourself.

**Prompt A — face detector comparison:**

> Read `CLAUDE.md`. Create a common detector interface in `project/core/detectors/base.py` (`detect(frame) -> list[Detection]` with box, score, class) and implement adapters for: Haar (existing), the SSD DNN (existing), YuNet (OpenCV), and at least two more of: SCRFD/InsightFace, YOLOv8-face weights, RetinaFace. Check and record each model's licence.
>
> Extend `eval/` to run every face detector on the WIDER FACE subset and on the annotated video clips, at several confidence thresholds. Produce `eval/results/faces.csv` and a precision-recall plot per detector. Report recall by face-size bucket (small, medium, large) and FPS on CPU (and GPU if available). Also report **track-level leak rate** on the video clips.
>
> Recommend a default detector and a "high-recall" detector, justified only by the numbers. Wire the choice into the profile config (not hard-coded). Don't delete the old detectors; keep them for the comparison table.

**Prompt B — plates and screens with YOLOv8:**

> Read `CLAUDE.md`. Fine-tune YOLOv8n (or s) to detect license plates:
>
> 1. Pick a public plate dataset (Roboflow Universe or another), check its licence, and document how to download it in `training/README.md`. Hold out a test split that is never used for training or tuning.
> 2. Write `training/train_plates.py` (or a notebook if GPU is needed) with a fixed seed and the exact hyperparameters saved to `training/config.yaml`. Train, then evaluate on the held-out split and save mAP@0.5 and mAP@0.5:0.95, precision and recall to `eval/results/plates.csv`.
> 3. Compare against the existing Haar + contour detector on the same held-out images and on the video clips. Report false positives per image. The contour detector's false positive on the crowd photo in the README is a known issue, so show that it is fixed or explain why not.
> 4. Replace the Haar/contour plate detector in the default pipeline with the fine-tuned model. Keep the old one behind a flag for the comparison.
> 5. Screens: keep the COCO classes (tv, laptop, cell phone) as the default, evaluate them on a small labelled set of your own images, and report the result honestly. Add "fine-tune a screen detector" to a TODO list in the README only if the numbers are weak.
> 6. Commit weights only if small, otherwise add `scripts/download_weights.py` that fetches them with a checksum.

**Done when:** `eval/results/` has faces, plates and video tables produced by scripts, and the default detectors are chosen by those numbers.

---

## Phase 3 — Selective redaction (click to select)

**Why:** this is the headline feature. It covers "blur strangers, keep my friends" (creators) and "hide this source only" (journalists) without building face recognition.

**Prompt:**

> Read `CLAUDE.md`. Build selective redaction on top of the tracks:
>
> 1. **Analyse step:** add an endpoint that runs detection and tracking over the whole clip and returns, for each track: ID, class, first and last frame, a representative thumbnail (best-scoring crop, small), and box positions per frame. Store results in a per-job temp directory that is deleted after export.
> 2. **Review UI:** a page that shows the clip with boxes labelled by track ID, plus a list or grid of track thumbnails. For each track the user can set **Hide** (default), **Keep visible**, or **Hide only from frame A to B**. Add "Hide all", "Keep all" and "Invert" buttons. Clicking a box in the video toggles that track.
> 3. **Export step:** re-render the video using the user's choices, at full resolution with audio (Phase 1). Redaction settings come from the selected profile.
> 4. **Background jobs:** processing runs in a worker (a thread pool or a simple queue is fine, and a task library is not needed yet). Show progress (frames done / total) and allow cancel. Handle multiple simultaneous jobs safely and clean up abandoned jobs after a timeout.
> 5. **Manual fixes:** allow drawing a box on a frame to add a missed face (it becomes a track propagated by a tracker, or applied for a chosen frame range). Missed detections happen and the tool must let the user fix them.
> 6. Tests for the choice logic (hide, keep, frame ranges, invert), and an end-to-end test on a small synthetic video.
>
> Show me a UI sketch and the API design before implementing.

**Done when:** you can upload a clip, click one person to keep them visible, export, and the result plays with sound.

---

## Phase 4 — Profiles and journalist features

**Prompt:**

> Read `CLAUDE.md`. Add profile-driven behaviour:
>
> 1. **Profiles:** `project/profiles/creator.yaml`, `journalist.yaml` and support for a user-defined custom YAML. Fields: classes to redact, per-class filter type (blur, pixelate, solid), strength, box padding, detector choice, confidence threshold, track buffer, detection stride, audio mode (`keep` / `mute`), strip metadata (bool), produce report (bool). Validate profiles with a schema and give clear errors.
>    - Creator: fast, natural look, audio kept, moderate padding.
>    - Journalist: high-recall detector, low confidence threshold, strong irreversible fill or heavy blur, larger padding, audio muted by default, metadata stripped, report on.
> 2. **Metadata stripping:** for video, remux with `-map_metadata -1 -map_chapters -1` and verify with `ffprobe` that GPS, creation time, device and encoder tags are gone (or reduced to a generic value). For images, confirm EXIF is not carried over, and test it. Add tests that check output metadata.
> 3. **Redaction report:** a JSON (and simple HTML) file per job: input hash, profile used, detector versions, number of tracks by class, which were hidden or kept, frame ranges, manual additions, and a statement of known limits. Do not include thumbnails or any pixels from the original.
> 4. **Irreversibility check:** add a test that confirms the redacted region contains no recoverable structure. For example, the blurred region should have low correlation with the original after deconvolution attempts, or the region should be fully solid in journalist mode. Document the reasoning.
> 5. Add the profile selector to the UI, and show what the profile will do in plain language before processing.

**Done when:** the same clip processed with both profiles gives visibly different outputs, and the report and metadata checks pass.

---

## Phase 5 — Polish, packaging and tests

**Prompt:**

> Read `CLAUDE.md`. Finish the engineering:
>
> 1. **Docker:** a `Dockerfile` (multi-stage if useful) that includes ffmpeg and the pinned requirements, runs as a non-root user and starts the app with gunicorn. Add a `docker-compose.yml` for local use.
> 2. **Tests and CI:** pytest suite, `ruff` for linting, a GitHub Actions workflow that runs lint and tests on every push and builds the Docker image. Keep CI fast by using tiny synthetic data.
> 3. **Repo hygiene:** move `Context-Aware-Privacy-Filtering-System (1).pdf` and `instructions.txt` into `docs/`, rename the PDF without spaces, update `.gitignore`, add `LICENSE`, and add a `CONTRIBUTING.md` with the dev setup.
> 4. **README rewrite:** structure it as: one-line pitch, demo GIF, who it's for, features, how it works (architecture diagram), results table (baseline vs final, from `eval/results/`), quick start (local and Docker), profiles, limitations, licences and datasets, and future work. Generate before/after GIFs from sample clips with a script in `scripts/`.
> 5. **Performance pass:** profile the pipeline, export the chosen models to ONNX if it gives a measured speed-up, and report before and after FPS.
> 6. **Security review:** check upload handling, path traversal, file type validation, size limits, temp file handling and dependency versions. Fix what you find and list it in `docs/SECURITY.md`.

**Done when:** a stranger can clone, run one command, and get the app working, and the README shows real numbers.

---

## Phase 6 — Hosted demo (samples and uploads)

**Why:** a working link is the single most useful thing on a CV, but it must not pretend to be private.

**Prompt:**

> Read `CLAUDE.md` (especially the "Demo rules"). Create a Hugging Face Space (Docker SDK) deployment in `demo/`:
>
> 1. `demo/Dockerfile` (or reuse the root one) that listens on port 7860, sets `MAX_CLIP_SECONDS=15`, a file-size limit and a request timeout through environment variables.
> 2. Bundle 3 to 5 sample clips in `demo/samples/` (permissive licences, small files, each under a few MB) and a "Try a sample" button for each one so a visitor can test with zero uploads. Document sources and licences in `demo/SAMPLES.md`.
> 3. Accept user uploads with the limits and validation from the demo rules, and show the privacy notice before upload.
> 4. Add rate limiting and make sure temp files are removed after each request, including when a job is cancelled or the connection drops.
> 5. Free CPU Spaces are slow. Use fast mode (detection stride) by default in the demo, show a progress bar, and document the expected processing time per second of video.
> 6. Write `demo/README.md` with the exact steps to create and push the Space, and add the Space link and badge to the main README.
>
> Do not add analytics or logging of user content.

**Done when:** the public link works end to end with a sample and with a fresh upload, and cleans up after itself.

---

## Phase 7 — Stretch (only after 0 to 6 are solid)

Do these one at a time, each behind a profile flag, each with an evaluation, and cut any that can't be measured.

1. **Reference-photo matching ("blur everyone except me"):** face embeddings (InsightFace/ArcFace) to auto-set Keep on tracks matching a reference photo. Report false accept and false reject rates. Treat as a convenience on top of click-to-select, never a safety guarantee.
2. **OCR text redaction:** detect text (PaddleOCR or EasyOCR) and redact emails, phone numbers, ID-like patterns and long digit strings in screens and documents. Report precision and recall on a small labelled set.
3. **Voice options for Journalist mode:** mute (already done) and pitch/formant shift, with an honest note that light voice changes can sometimes be reversed.
4. **Live webcam mode:** low-latency detect and blur for calls and streams, with a measured latency budget.
5. **Person-level anonymization:** segmentation masks (YOLOv8-seg) for full-body blur where faces are not the only identifier.
