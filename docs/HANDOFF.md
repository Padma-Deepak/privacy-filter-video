# Handoff — Privacy Filter (Video)

This document is self-contained. You should be able to pick up the project from here without having seen the session that produced it. Start with `CLAUDE.md` (project rules) and `PHASES.md` (roadmap) — this file is a snapshot of where things stand against that plan, not a replacement for either.

---

## 1. Goal and users

A **local-first video and image anonymizer** for two kinds of users:

- **Creators** posting street or travel footage, who want a fast, natural-looking blur and to keep their audio.
- **Journalists** publishing footage with bystanders or sources in it, who need high-recall, irreversible redaction and are fine muting audio or stripping metadata for safety.

Both share one core detect → track → redact pipeline; a profile (Creator/Journalist) changes the settings, not the code path.

**Why this project exists:** it's a portfolio piece for computer vision roles. That framing matters for every decision in here — the goal is measured results and honest limitations, not feature count. Every number in the README must come from a script in `eval/` that can be re-run. If you're tempted to add a feature without a way to measure whether it helps, that's a signal to stop and add the measurement first.

---

## 2. Key decisions already made (and why)

These are recorded in `CLAUDE.md` — this is a summary with the reasoning, not a replacement for reading it.

- **Detection stack:** YOLOv8 (Ultralytics) for plates and screens. For **faces**, the detector is chosen by *measurement* (recall/FPS on WIDER FACE), not reputation — Haar, DNN-SSD, and eventually YuNet/SCRFD/a YOLOv8-face model are meant to be compared head-to-head in Phase 2. Don't hardcode a "best" detector without a number behind it.
- **Tracking:** `supervision.ByteTrack`, wrapped so it's detector-agnostic — never coupled to Ultralytics' own `.track()` API, so the face detector chosen in Phase 2 can be swapped freely. See `project/core/tracking.py`.
- **Redaction is never gated by the tracker.** This is the most important design invariant in the codebase: every raw detection a frame produces gets redacted **that same frame**, unconditionally. The tracker (ByteTrack, or our own continuity matching) only *adds* two things on top — gap-fill (keep redacting a briefly-missed track's last known position) and smoothing (reduce jitter). It is never allowed to *delay* or *skip* a redaction. This was tightened twice during implementation after finding that `ByteTrack` has an internal, non-configurable confidence floor that can silently drop a detection on its first frame, and that its own frame-to-frame matching can lose continuity under motion in ways that would otherwise make a track's identity (and therefore whether it stays hidden) flicker. See the docstring at the top of `project/core/tracking.py` for the full reasoning — it's worth reading before touching that file.
- **Smoothing pads the union of the raw box and the smoothed box**, not the smoothed box alone. Padding only ever grows a region, so this guarantees the raw detection is always fully covered regardless of how much the smoothed estimate lags behind — smoothing can reduce jitter, it can never shrink the redacted area below what was actually detected.
- **Local-first:** no cloud APIs, no telemetry, ever. The hosted demo (Phase 6) is the one exception and must say so explicitly to users.
- **Licensing:**
  - Ultralytics (YOLOv8) is **AGPL-3.0**. A `LICENSE` file and a "Licenses" section in the README (listing every model/dataset and its licence) are still outstanding — see `PHASES.md` Phase 5.
  - The optional DNN face model (`res10_300x300_ssd_iter_140000.caffemodel`, fetched by `scripts/download_models.py`) **has no upstream licence at all** — confirmed via the GitHub API on `opencv/opencv_3rdparty`, and independently flagged by the OpenCV community as an unresolved gap. **Do not bundle these weights into a Docker image or the hosted demo** (Phases 5–6) unless this is resolved first — either a clear licence is found, or the model is swapped for one with real provenance (e.g. OpenCV's own YuNet, Apache-2.0). This restriction is recorded in `CLAUDE.md` under "Key decisions."

---

## 3. What is done

### Phase 0 — Audit, baseline, eval scaffold (complete)

- `docs/AUDIT.md`: a line-referenced read-through of the original `app.py`/`detector.py`, with every bug/fragile-spot found, ranked by severity.
- `eval/` scaffold: `metrics.py` (IoU/precision/recall/size-bucket recall), `prepare_wider_face.py` (seeded 300-image WIDER FACE subset — downloads directly over HTTPS, not via the `datasets` library, whose loader for this dataset is broken against current `huggingface_hub`), `run_baseline.py`.
- **Measured baseline** (`eval/results/baseline.csv`), on the machine that produced it — **an Apple M5, CPU only, no GPU** (see `README.md`'s Baseline section for the full hardware/software line; FPS numbers below will not transfer to your machine):

  | Detector | Precision | Recall | Recall (small) | FPS |
  |---|--:|--:|--:|--:|
  | Face — Haar (current default) | 47.1% | 25.1% | 6.5% | 2.31 |
  | Face — DNN (SSD) | 98.1% | 11.8% | 0.0% | 32.94 |
  | Face — Haar+DNN (production) | 49.8% | 27.2% | 6.5% | 2.18 |

  Plates/screens have no ground truth in WIDER FACE (it's a face-only dataset) — only raw counts/FPS are recorded, honestly, rather than a fabricated precision/recall.

Commits (oldest → newest):
```
3f8a3ad Ignore venv, downloaded model weights, and eval cache
70df166 Add CLAUDE.md and PHASES.md project planning docs
ac3996b Add Phase 0 repo audit and pin Python to 3.12
9f7dc76 Add Phase 0 evaluation scaffold with WIDER FACE baseline
03c1371 Add Phase 0 baseline results to README
```

### Hardening (part of Phase 0 close-out, done alongside Phase 1 setup)

- Flask no longer runs with `debug=True` on `0.0.0.0` by default (was a real remote-code-execution exposure via the Werkzeug debugger) — now env-controlled, defaults to off / `127.0.0.1`.
- Detector load failures are now loud: a genuine load failure (as opposed to the DNN model simply not being downloaded, which is an expected, silent fallback) is logged as an error and surfaced as a visible warning in the response.
- The YOLO weights path bug (relative path meant it only worked if the app was launched from exactly `project/`) is fixed — resolved relative to `detector.py`'s own location.
- `scripts/download_models.py` fetches the optional DNN face model with pinned SHA-256 checksums.

Commits (oldest → newest):
```
b366916 Harden Flask entrypoint: debug/host off by default, no exception leakage
d79d649 Make detector load failures loud, fix YOLO weights CWD dependency
069b7bb Add DNN face model download script, record its licence, re-run baseline
14ed210 Confirm venv Python is native arm64, record it in README
1c070be Document DNN face weights' licence gap as a Docker/demo blocker
```

### Phase 1 — Video foundation (in progress — see §4 for what's still open)

- `detect_all()` extracted from `detector.py`'s `process_frame()` as a pure refactor (detection separated from redaction, so tracking can sit in between) — behaviour-preserving, proven by regression tests.
- `project/core/tracking.py`: `ClassTracker` — one instance per object class per job, wraps `ByteTrack` for best-effort IDs, but **our own continuity matching is authoritative**, not ByteTrack's (see §2). Per-job ID renumbering (1, 2, 3... in order of first appearance) on top of ByteTrack's process-wide internal counter.
- `project/core/video.py`: streams frames one at a time (never the whole clip in memory), detects on a downscaled copy but redacts at full resolution, handles rotation via `ffprobe` + `cv2.rotate()` (not OpenCV's auto-orientation — see §4), normalizes variable-frame-rate sources to constant-rate at the *measured* average fps, mux­es real H.264 + original audio via `ffmpeg`, and cleans up temp files on every exit path (success, error, or interrupt).
- `scripts/debug_tracking.py`: visualizes raw detections (green) / tracker estimate (yellow) / final redaction region (red) with track IDs labelled — very useful for debugging anything tracking-related. Run it on any clip; output goes to `scripts/debug_output/` (git-ignored).
- A real bug found via real-clip testing and partially fixed: see §4.

Commits (oldest → newest):
```
6b3d398 Add supervision as a Phase 1 dependency for tracking
e1e3600 Extract detect_all() from process_frame() as a pure refactor
08862f5 Add tracking.py: gap-fill/smoothing tracker that never gates redaction
0ec33f0 Document the rotation-metadata testing gap in README
0599fb1 Fix track ID stability: own continuity beats ByteTrack's suggestion
602f3a4 Expose tracker's unpadded box on TrackedBox for debug visualization
cbf60d2 Add video.py: ffprobe-based probing, rotation and resolution helpers
a6a7aad Add process_video_streaming(): the main video pipeline
6db26ae Wire app.py to the new streaming video pipeline
405de28 Add scripts/debug_tracking.py for visual tracking debugging
0ad1029 Fix gap-fill ghost-box accumulation on noisy plate false positives
```

21 commits total, all local to the `upgrade` branch (see §7 — none of this has been pushed anywhere yet). 76 tests pass as of the last commit (`python -m pytest tests/`).

---

## 4. Current status and known issues — read this before trusting the pipeline

**Phase 1 is not done.** The `PHASES.md` "done when" bar for Phase 1 is "a 60-second 1080p clip processes without hitting a size cap, the output keeps its audio, the boxes don't flicker on the sample clips, and tests pass" — the last real-clip test did **not** clear that bar. Here's exactly what was found, tested with synthetic data, and what's still unverified on real footage.

**Real-clip test: a 15-second portrait phone video (a restaurant/indoor scene, one or more real faces, real audio).** Results:

1. **719 "plates masked" and a large black region across part of the frame.** Investigated and root-caused: the contour-based plate detector (already documented as noisy in `docs/AUDIT.md` — false positives on rectangular architecture) fired frequent, spatially-random false positives on doorframes/ceiling lines in the background. With the original 30-frame gap-fill buffer (same as faces), every one-off false trigger stayed "alive" and redacted for about a second, and on a busy background several accumulated into 13+ simultaneous overlapping boxes that visually merged into one large blocked-out region. No single detection was ever oversized — this was a volume/accumulation problem.
   - **Fix applied** (commit `0ad1029`): plates now default to a 5-frame gap-fill buffer instead of 30 (`DEFAULT_TRACK_BUFFER_BY_CLASS` in `project/core/video.py`), configurable via `track_buffer_by_class`. Verified with a seeded synthetic simulation reproducing the same noisy-detection pattern: max simultaneous ghost boxes dropped from 13+ to 5. Two permanent regression tests added (`tests/test_tracking.py`).
   - **Not yet confirmed:** whether this fix actually resolves the *visible* black-region problem on the real clip that produced it. The synthetic simulation is a reasonable proxy but is not the same as watching the real output. **Re-run the real clip and check visually before trusting this is fixed.**
2. **"Blocky patches over people"** — reported during testing, **not yet investigated at all.** No root cause identified. Possible directions to check first: pixelation (the screens filter) landing on people rather than actual screens (a YOLO misclassification?), or plate/contour false positives landing on clothing/skin rather than architecture. Start by running `scripts/debug_tracking.py` on the same clip and looking at which class (green box label) is drawing over people.
3. **The "Original" video panel not loading in the browser result page** — reported during testing, **not yet investigated at all.** The uploaded file was a `.mov`. One plausible direction: `project/app.py`'s `_to_data_url()` embeds the original file as a `data:video/quicktime;base64,...` URL, and Chrome/most browsers don't reliably play inline QuickTime-codec content via a `<video>` tag's `src` the way they do MP4/H.264 — worth checking whether this reproduces with a `.mp4` upload instead of `.mov`, which would confirm it's a MIME/codec-support issue rather than a data-encoding bug.

**Also open, not bugs so much as incomplete verification:**

- **Rotation handling is only verified with synthetic, mocked `ffprobe` values and a physically-transposed (not metadata-tagged) test clip** — see `tests/test_video_probe.py` and the "Testing" section of the root `README.md`. The machine this was built on has an ffmpeg build (9.0.2) that could not be made to write real rotation metadata into a test file despite trying five documented methods, so there is **no automated test with a real rotation-tagged file**. This needs manual verification with real portrait phone clips (which, per the point above, you now have reason to do anyway) before rotation handling can be trusted for real iPhone/Android footage.
- **Processing a real ~15s phone clip took roughly 8–9 minutes** on the machine this was tested on. This is expected given the current pipeline runs full-resolution Haar-cascade detection on every single frame with no frame-skipping — `PHASES.md`'s Phase 1 item 3 ("run detection every K frames, propagate boxes between detections with the tracker, default K=1") was never implemented. This isn't correctness-broken, but it makes iterating on real clips painfully slow — worth prioritizing before doing more real-clip QA.
- **Video-clip tracking evaluation is "pending annotations."** `eval/video_gt/README.md` documents the CVAT annotation process, but **no real annotations exist yet** — that's a manual task (see §6). Until they exist, there is no measured track-level or frame-level leak rate for video, only the image-based WIDER FACE numbers in §3.

None of this was hidden or glossed over — it's written here so you don't have to rediscover it.

---

## 5. What to do next, in order

### Setup

```bash
python3.12 -m venv .venv          # Python 3.12 — the system Python may be newer and unsupported by some CV deps
echo "3.12" > .python-version     # already committed, just confirming
source .venv/bin/activate
pip install -r project/requirements.txt -r eval/requirements.txt
```

Confirm `ffmpeg` is installed with `libx264` (`ffmpeg -version` should list it under `--enable-libx264`) — the video pipeline hard-requires it and will raise a clear `FFmpegNotFoundError` if it's missing, rather than a cryptic crash.

```bash
python scripts/download_models.py      # optional DNN face model, checksum-verified (see §2's licence note)
python eval/prepare_wider_face.py       # downloads ~365MB WIDER FACE validation subset, cached after first run
```

### Run the tests

```bash
python -m pytest tests/
```

Should show 76 passed as of the last commit here. If something fails, that's more informative than anything in this document — trust the tests over this file if they ever disagree.

### Re-run the baseline on your own machine

```bash
python eval/run_baseline.py
```

**The FPS numbers in `eval/results/baseline.csv` and the README will change** — they were measured on a different machine (Apple M5, CPU only; see §3). Precision/recall should be identical (same detectors, same seeded data, deterministic), but re-run it and update the README's hardware line to reflect your own machine before treating those numbers as current.

### Read before touching code

`docs/AUDIT.md` — the full original-codebase audit. Most of its findings are already fixed (see §3), but it's the fastest way to understand what the pipeline looked like before this work started, and a few findings are still open (check the file itself for what's marked resolved vs not).

### Then: finish Phase 1, then continue through PHASES.md

Phase 1 is not done (§4). At minimum, before calling it complete:
1. Confirm (or fix further) the plate ghost-box issue against a real clip, not just the synthetic simulation.
2. Investigate and fix the "blocky patches over people" and "Original panel not loading" issues.
3. Get real rotation-metadata test coverage, or explicitly accept the manual-verification-only gap and move on with it documented.
4. Consider implementing frame-skipping (`PHASES.md` Phase 1 item 3) — not strictly required by the "done when" bar, but real-clip iteration is currently very slow without it.
5. Once you're confident, re-run `eval/run_baseline.py` on the video clips (§6 — needs annotations first) and add the "with tracking" row PHASES.md Phase 0 already scoped.

After that, `PHASES.md` Phases 2–7 continue in order: **Phase 2** (detector comparison + a trained plate detector, which is the real fix for the contour-detector noise in §4), **Phase 3** (click-to-select UI), **Phase 4** (Creator/Journalist profiles, metadata stripping), **Phase 5** (Docker, CI, README/LICENSE polish), **Phase 6** (hosted demo), **Phase 7** (stretch features). Each phase's prompt is written out in full in `PHASES.md` — they're meant to be pasted to Claude Code as-is, one at a time.

### Using Claude Code for this

`CLAUDE.md` is read at the start of every session automatically — it has the project rules, non-negotiables, and working conventions (small commits, tests with every change, no claims without numbers). `PHASES.md` has one ready-to-paste prompt per phase. The workflow that produced everything in this document was: paste one phase's prompt, review the plan Claude Code proposes before it starts on anything large, let it work in small committed steps, review each step, and only move to the next phase once the current one's "done when" criteria are actually met (not just "the code exists"). Don't skip ahead to a later phase's prompt while an earlier one still has open issues — Phase 1's issues in §4 are a direct example of why: they were found by actually testing with real footage, not by assuming the tests passing meant it worked.

---

## 6. Manual tasks — only a human can do these

1. **Film and annotate 5–10 short clips (5–15s each) in CVAT** with track IDs for faces and plates. Full instructions, including the expected export format, are in `eval/video_gt/README.md`. This is explicitly called out in `PHASES.md` as "the slowest part of the project" — start early. Cover a crowd/street scene, traffic/car park, a dashcam-style clip, an indoor scene with screens/devices, and at least one clip where a face leaves and re-enters frame.
2. **Check the licence of every model, dataset, and sample clip before using it.** This project has already found one real gap (the DNN face model — §2) by actually checking rather than assuming; don't assume the next one is fine either. Record what you find, the same way `project/README.md`'s DNN section and `eval/prepare_wider_face.py`'s docstring do.
3. **Choose and licence-check the hosted-demo sample clips** (`PHASES.md` Phase 6) — 3–5 short clips, permissively licensed (Pexels/Pixabay or your own footage), documented in `demo/SAMPLES.md` (not created yet) with source and licence for each.

---

## 7. Git workflow

Everything described in this document lives on the **`upgrade`** branch. It has **not been pushed anywhere** — every commit in §3 is local to the checkout this session worked in. Two ways this can reach you:

- You get pushed a fork containing this branch, and fetch/merge or rebase from it, or
- A pull request gets opened against your repo from that fork.

Either way: **never commit directly to `main`**. Work happens on branches, gets reviewed, then merged.

Do not commit:
- Personal footage or any real people's faces/plates (uploaded test clips are deleted automatically by the app itself, and are never meant to be committed regardless).
- Model weights (`project/models/*`, fetched by `scripts/download_models.py`) — git-ignored already.
- The `.venv/` virtual environment — git-ignored already.
- The WIDER FACE dataset cache (`eval/data/`) — git-ignored already, regenerated by `eval/prepare_wider_face.py`.
- `scripts/debug_output/` (debug visualizations) — git-ignored already.

If you add a new kind of generated artifact, add it to `.gitignore` in the same commit, not after the fact.

---

## 8. Warnings

**Privacy rules (`CLAUDE.md` non-negotiables — read the full list there, this is not the complete set):**

- Redaction must be **irreversible**: a strong blur (large kernel, applied after box padding) or a solid fill. Never a light/reversible blur.
- **Never keep an unredacted copy of a frame on disk.** Uploads and all temp files are deleted after processing — including on errors and timeouts (`project/core/video.py`'s cleanup guarantees exist specifically for this; see §3).
- **A missed detection is a failure.** Recall is the headline metric, not speed or a polished-looking demo. Prefer a lower confidence threshold plus a review step over a high threshold with silent misses.
- Never modify the user's original uploaded file — always work on a copy.

**Metric definitions (`CLAUDE.md`) — use these exact meanings, they're easy to get subtly wrong:**

- **Face recall:** matched ground-truth faces ÷ all ground-truth faces, IoU ≥ 0.5, reported per dataset and per size bucket (small/medium/large).
- **Frame-level leak rate:** fraction of ground-truth boxes not sufficiently covered (≥90% of the ground-truth box area inside the padded redaction region) by a redaction region in a frame.
- **Track-level leak rate:** fraction of ground-truth *tracks* that leak in **at least one** frame — this is the privacy-critical metric, not frame-level (a track can be redacted correctly in 99% of its frames and still be a track-level leak if it's exposed in even one).
- **FPS:** end-to-end, including decode/detect/track/redact/encode, on a stated machine (CPU/GPU/none) at a stated resolution — never quote a detector-only number as if it were the full pipeline's FPS (see §3's baseline table, which is explicitly detector-only and says so).
