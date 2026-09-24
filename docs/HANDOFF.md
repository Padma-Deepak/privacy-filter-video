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
79101a8 Add detection-stride (fast mode): detect every Kth frame
28674c8 Fix Original preview not loading for .mov/.avi/.mkv uploads
```

23 commits total, all local to the `upgrade` branch (see §7 — none of this has been pushed anywhere yet). 89 tests pass as of the last commit (`python -m pytest tests/`).

### Investigation round 2 — real footage from an actual iPhone, findings below

Commit `0ad1029` was from a first pass using synthetic simulation only. `79101a8`/`28674c8` plus the analysis in §4 are from a second, deeper investigation using a real ~15s portrait iPhone clip, source-copied down to a 4-second/122-frame segment (`ffmpeg -t 4 -c copy`) and kept entirely outside the repo (a temp directory, not committed anywhere) for iteration speed. That segment is machine-local and ephemeral — regenerate your own 4s segment the same way (`ffmpeg -y -i <original> -t 4 -c copy segment.mov`, run outside the repo) if you need to re-run any of this investigation's scripts.

---

## 4. Current status and known issues — read this before trusting the pipeline

**Phase 1 is not done.** The `PHASES.md` "done when" bar for Phase 1 is "a 60-second 1080p clip processes without hitting a size cap, the output keeps its audio, the boxes don't flicker on the sample clips, and tests pass." Real-clip testing found real problems; some are fixed and confirmed on real footage below, one is fixed but not yet re-verified visually, and one is understood but deliberately not patched. Read all of it before assuming this is done.

**Test setup for round 2:** a real ~15s portrait iPhone `.mov` (1080×1920, h264+aac+a second Apple spatial-audio track, **no rotation metadata at all** — it's natively portrait, so the orientation issue reported was never a rotation bug), cut to a 4-second/122-frame segment with `ffmpeg -t 4 -c copy` and kept outside the repo. Two real people in frame, indoor restaurant/cafe setting with a hanging lamp and visible ceiling beams.

### 4.1 Plate ghost-box accumulation — fixed, confirmed on real footage, one separate issue remains

Original symptom: 719 "plates masked" and a large black region across part of the frame. Root cause (commit `0ad1029`): the noisy contour plate detector fired frequent false positives on architecture, and the original 30-frame gap-fill buffer let each one-off trigger stay redacted for ~1s, accumulating into 13+ simultaneous ghost boxes that visually merged.

**Confirmed on the real 4s segment after the fix** (`DEFAULT_TRACK_BUFFER_BY_CLASS["plates"] = 5`):

| | count |
|---|--:|
| Plates per frame | min 2, max 6, avg 4.75 |
| Total detected / gap-fill | 297 / 283 |

Down from the simulated 13+ simultaneous boxes — the accumulation fix works on real content, not just synthetic data.

**Still open — a separate issue, not fixed:** the single largest plate box observed was a genuinely large **raw detection**, not a tracking artifact: `(0, 0, 789, 325)`, 12.4% of the frame, on the lamp/ceiling area, `source="detected"`. The contour detector itself is producing oversized bounding boxes on architecture, independent of gap-fill. This is a detector-quality problem — `PHASES.md` Phase 2 (a trained plate detector replacing the contour heuristic) is the real fix, not something patchable in the tracker.

Debug visualization confirming this — real footage, green/yellow/red overlay with track IDs — is machine-local (see the note above about the test segment); regenerate with `python scripts/debug_tracking.py <your 4s segment>`.

### 4.2 "Blocky patches over people" — cause found, not fixed (deliberately)

**Not screens/pixelation.** Screens detected 0 boxes throughout the entire segment — YOLO is not misfiring on people.

**It's faces, via Gaussian blur, and the Haar cascade is flooding with false positives on this content:**

| | count |
|---|--:|
| Faces per frame | min 4, max **25**, avg **20.6** |
| Total detected / gap-fill | 573 / 1940 |
| Largest single face box | **882×882px, 37.5% of the frame**, `source="gap_fill"` |

There are only 2 real people in frame. Averaging 20.6 "face" boxes per frame — visually confirmed as scattered across clothing, torsos, and hair, not just the two real faces — means most of what's being blurred isn't a face at all. The 882×882 box is worth being precise about: **it did not grow from smoothing or velocity extrapolation** — gap-fill extrapolation only ever shifts a box's *position*, never its width/height (verified by reading the code, not assumed), so a box that large means Haar itself produced an oversized raw false-positive detection at some point, which then got a full 30-frame gap-fill life (faces' buffer was intentionally left at 30, unlike plates — see below).

**Why this wasn't fixed:** two candidate fixes exist, and neither is "small and clearly correct":
1. **Shorten faces' gap-fill buffer the same way plates' was.** Rejected without measurement: unlike plates (where there's no established recall value in tracking a heuristic that's mostly noise), faces are the one class this whole pipeline is built around, and `CLAUDE.md` non-negotiable #2 is explicit — "a missed face is a failure." A shorter buffer means a real face that Haar briefly loses (a known, documented flicker problem from the original `docs/AUDIT.md` audit) un-blurs sooner. That's a recall-vs-noise trade-off, and `CLAUDE.md` non-negotiable #6 requires numbers before making that call, not a guess under time pressure.
2. **Bound gap-fill's velocity extrapolation displacement** (cap how far a track can drift during the buffer window, regardless of how large/noisy the computed velocity is). This is a real, safe, recall-neutral idea — worth doing — but it would not have prevented the 882×882 box specifically (that was a *size* problem from the raw detection, not a *drift* problem), so implementing it here would have been solving a different, smaller problem while the actual investigation budget was needed for confirmed items. Left as a suggested small improvement for whoever picks up Phase 2, not implemented.

**The real fix is `PHASES.md` Phase 2** — comparing face detectors by measured recall/FPS and replacing Haar's default role. This flooding is a strong, concrete data point for that comparison: whatever replaces Haar needs to be measured against this exact failure mode, not just WIDER FACE recall.

### 4.3 "Original" panel not loading — fixed, confirmed

Root cause confirmed: `MIME_TYPES["mov"] = "video/quicktime"` in `project/app.py`, and Chrome (and most non-Safari browsers) does not reliably play `video/quicktime` inline via a `<video>` tag, especially from a `data:` URI. The processed panel never had this problem because the pipeline always outputs `.mp4`.

**Fix** (commit `28674c8`): `_preview_data_url()` remuxes non-web-playable containers (`.mov`/`.avi`/`.mkv`) to a clean single-video+audio MP4 for the preview only — a stream copy, not a re-encode (~0.1s on the test segment), so it's fast and lossless. The actual uploaded file and the processing pipeline are untouched. Falls back to the raw file's own MIME if the remux fails, so this can't turn into a 500. 5 tests, using a small synthetic clip (no real footage in the test suite).

### 4.4 Processing speed — profiled, fast mode implemented

Per-stage timing on the real 4s/122-frame segment, measured **sequentially** to isolate each stage (production runs faces/plates/screens in parallel via a `ThreadPoolExecutor`, so real per-frame wall time is closer to the slowest of the three plus the fixed costs, not their sum — treat the sequential numbers below as relative cost, not literal production timing):

| Stage | ms/frame | % of sequential total |
|---|--:|--:|
| Face detection (Haar+DNN) | 376.07 | 55% |
| Plate detection (Haar+contour) | 176.24 | 26% |
| Redaction (all filters) | 76.88 | 11% |
| Screen detection (YOLO) | 26.89 | 4% |
| Rotation + resize | 21.97 | 3% |
| Tracking | 0.96 | <1% |
| Decode | 1.00 | <1% |

Face and plate detection dominate — exactly the two Haar-cascade-based detectors, consistent with Phase 0's baseline already showing Haar at ~2 FPS standalone.

**Implemented** (commit `79101a8`): `detection_stride` (env var `DETECTION_STRIDE`, default 1 = every frame). Detecting every Kth frame and letting the tracker's existing gap-fill bridge the rest cuts detector calls ~K-fold for free — no new tracking logic needed, since a skipped frame just feeds zero raw boxes to the trackers, which already produces `source="gap_fill"` entries via the same mechanism as §4.1's fix.

**End-to-end FPS measured on the real segment** (includes encode/mux, not just detection):

| K | FPS | Faces detected | Speedup vs K=1 |
|--:|--:|--:|--:|
| 1 | 1.76 | 573 | — |
| 2 | 2.72 | 274 | 1.55× |
| 3 | 3.12 | 190 | 1.77× |

Diminishing returns are expected and observed: redaction, tracking, and the one-time encode/mux cost don't shrink with stride, only detector calls do, which caps how much speedup is available.

**The trade-off — this is not free, and it is documented in three places** (`project/core/video.py`'s docstring, `README.md`'s new "Performance" section, and here): with `K > 1`, a face/plate/screen that **first appears on a skipped frame is not redacted until the next detection frame runs — up to (K-1) frames of exposure.** This does not weaken the "every raw detection is redacted the frame it's found" guarantee (still exactly true on every frame detection actually runs), it just changes how often that check happens. Because of this, `PRIVACY_PROFILE=journalist` forces `DETECTION_STRIDE` back to 1 regardless of the env var — a stopgap since Phase 4's real profile system doesn't exist yet, but Journalist mode must never trade recall for speed. 6 tests added, including one that proves the guarantee still holds on actual detection frames by comparing real output pixels (not just asserting a count) between a run with detection enabled vs. disabled.

### Also still open

- **Rotation handling is only verified with synthetic, mocked `ffprobe` values and a physically-transposed (not metadata-tagged) test clip.** The real iPhone clip used for this investigation happened to have no rotation metadata at all (genuinely portrait-native), so it did not exercise this path either. **A real rotation-tagged clip still needs manual verification** — this remains completely unverified against real metadata-rotated footage.
- **Video-clip tracking evaluation is "pending annotations."** `eval/video_gt/README.md` documents the CVAT annotation process, but no real annotations exist yet (§6). Until they exist, there is no measured track-level or frame-level leak rate for video, only the image-based WIDER FACE numbers in §3.

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

Should show 89 passed as of the last commit here. If something fails, that's more informative than anything in this document — trust the tests over this file if they ever disagree.

### Re-run the baseline on your own machine

```bash
python eval/run_baseline.py
```

**The FPS numbers in `eval/results/baseline.csv` and the README will change** — they were measured on a different machine (Apple M5, CPU only; see §3). Precision/recall should be identical (same detectors, same seeded data, deterministic), but re-run it and update the README's hardware line to reflect your own machine before treating those numbers as current.

### Read before touching code

`docs/AUDIT.md` — the full original-codebase audit. Most of its findings are already fixed (see §3), but it's the fastest way to understand what the pipeline looked like before this work started, and a few findings are still open (check the file itself for what's marked resolved vs not).

### Then: finish Phase 1, then continue through PHASES.md

Phase 1 is not done (§4). At minimum, before calling it complete:
1. **The Haar face false-positive flooding (§4.2)** — this is the biggest remaining item. It needs Phase 2's detector comparison, not a Phase 1 patch; treat this real clip's ~20 boxes/frame for 2 people as a concrete test case any replacement detector must clear.
2. **The plate detector's oversized raw false-positives on architecture (§4.1)** — same story, same phase, same detector-quality root cause.
3. **Get real rotation-metadata test coverage** — the iPhone clip used for round-2 testing happened to have none (genuinely portrait-native), so this is still completely unverified against real metadata-rotated footage. Either find/film a clip that actually has the metadata, or explicitly accept the manual-verification-only gap and move on with it documented.
4. Once you're confident, re-run `eval/run_baseline.py` on the video clips (§6 — needs annotations first) and add the "with tracking" row `PHASES.md` Phase 0 already scoped.

Already done, from a second investigation pass: the plate ghost-box accumulation fix is confirmed on real footage (§4.1), the "Original panel" bug is fixed and tested (§4.3), and per-stage profiling plus a working `detection_stride` fast mode are in place (§4.4).

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
