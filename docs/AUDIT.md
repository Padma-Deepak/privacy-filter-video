# Phase 0 Audit — Privacy Filter (Video)

> Historical document: this is a point-in-time audit of the original coursework
> implementation before streaming, tracking, selective review, profiles, YuNet,
> metadata removal and security hardening were added. Line references and present-
> tense statements below describe that original revision. Use the root README and
> `docs/HANDOFF.md` for the current implementation.

Date: 2026-09-24
Scope: `project/app.py`, `project/detector.py`, `project/requirements.txt`, `project/README.md`, `instructions.txt`, plus `project/templates/*.html` (read for the security review since app.py renders user-influenced data into them).
No behaviour was changed while writing this document.

---

## 1. What each file does

### `project/app.py` (122 lines)

Flask app with two routes.

- `GET /` (`app.py:53-55`) — renders the upload form (`templates/index.html`).
- `POST /process` (`app.py:58-104`) — the only processing entry point:
  1. Validates a file was posted and has an allowed extension (`app.py:60-66`, extensions defined `app.py:19-21`).
  2. Saves it under a random UUID filename in `project/uploads/` (`app.py:70-71`) — the original filename is never used to build a path, only its extension, so there's no path-traversal via filename.
  3. Dispatches to `detector.process_video` or `detector.process_image` depending on extension (`app.py:75-78`).
  4. Base64-encodes both the original upload and the processed output into `data:` URLs and inlines them directly into `result.html` (`app.py:46-50`, `82-84`) — nothing is ever served from a `/uploads` or `/outputs` static route.
  5. Deletes both the upload and the output file in a `finally` block, even on error (`app.py:87-91`).
- Error handlers for 400/413/500 re-render `index.html` with a message (`app.py:107-117`).
- `app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024` caps uploads at 200 MB (`app.py:35`).
- Entry point: `app.run(debug=True, host="0.0.0.0", port=5000)` (`app.py:121`).

### `project/detector.py` (429 lines)

All detection, filtering and the two public pipeline entry points.

- **Model loading** (`detector.py:37-83`): Haar cascades loaded eagerly at import from `cv2.data.haarcascades` (bundled with `opencv-python`, no download). The DNN face net (`_get_dnn`, `:54-67`) and YOLOv8 (`_get_yolo`, `:72-80`) are lazy singletons, loaded once and cached in module globals. YOLO is also warmed up eagerly at import time (`:83`).
- **NMS** (`:90-110`): a hand-rolled greedy IoU suppression, used to merge Haar+DNN face boxes and Haar+contour plate boxes. Not applied to YOLO screen boxes (Ultralytics already runs NMS internally, so this is correct, not an oversight).
- **Detectors**:
  - `detect_faces` (`:149-150`) — Haar (`:117-122`) + DNN SSD (`:125-146`), merged with NMS.
  - `detect_plates` (`:182-183`) — Haar Russian-plate cascade (`:157-163`) + Canny/contour heuristic (`:166-179`), merged with NMS.
  - `detect_screens` (`:190-206`) — YOLOv8n, filtered to COCO classes `tv`(62)/`laptop`(63)/`cell phone`(67) (`:30-34`).
- **Filters** (`:213-250`): `apply_gaussian_blur` (kernel `(99,99)`, sigma `30`), `apply_pixelation` (15 blocks), `apply_black_mask` (zeroes the region). All clamp the box to image bounds first (`_clamp_box`, `:213-216`); none pad the box before filtering.
- **`process_frame`** (`:257-305`) — the shared per-frame pipeline for both images and video: runs all three detectors in parallel via `ThreadPoolExecutor` (reuses a passed-in executor if given, otherwise spins up a temporary 3-worker pool), then applies blur/mask/pixelation, then returns the annotated frame plus per-class counts.
- **`process_image`** (`:312-341`) — reads with `cv2.imread`, runs `process_frame` once, writes to `outputs/<uuid>.<original-ext>`.
- **`process_video`** (`:348-429`) — see §2 below.

### `project/requirements.txt`

```
flask>=3.0,<4
opencv-python>=4.9,<5
numpy>=1.26,<3
ultralytics>=8.2,<9
```

Four direct dependencies, loosely pinned (range, not exact). No `ffmpeg-python`, no audio library — consistent with the code never touching audio. Ultralytics pulls in `torch`/`torchvision` transitively (not listed directly), which is where the real weight of the install comes from (see §4).

### `project/README.md`

User-facing docs for the `project/` app specifically: install steps, optional DNN model download links, run command, project structure, a "Team"/course-assignment section, and a known-limitations list. Says video is capped at ~12s, downscaled to 960px wide, and has no audio — matches the code.

### `instructions.txt`

This is the **original coursework PRD**, not current documentation — treat it as historical, not a spec to satisfy going forward:
- It describes an **image-only** MVP ("Video processing... out of scope (though the architecture supports it as a future extension)") — video has since been built, so this file is stale.
- The file content itself is corrupted/duplicated: the same PRD + build-prompt text appears three times back-to-back (e.g. "Manual blurring in tools like Photoshop..." is truncated mid-word and immediately re-started at `instructions.txt:15`, and the whole document repeats again from `instructions.txt:183` to the end). This looks like an artifact of however the file was assembled, not intentional. Worth cleaning up in the Phase 5 repo-hygiene pass (`PHASES.md` already plans to move this file into `docs/`).
- It's still useful as a record of the original spec (kernel size 99×99, confidence 0.5, aspect ratio 2.0–5.5, min contour area 1500 — all of these match the current code exactly, e.g. `detector.py:219`, `:125`, `:177`, `:171`).

---

## 2. How video is processed

`detector.process_video(input_path, outputs_dir, models_dir, max_duration_sec=12.0, max_width=960)` (`detector.py:348-429`):

1. Opens the input with `cv2.VideoCapture` (`:375`). If it can't open, raises immediately.
2. Reads source FPS (`:379`, defaults to 24.0 if the container doesn't report one) and total frame count (`:380`).
3. **Duration cap:** `max_frames = int(max_duration_sec * fps)` (`:381`) — hardcoded default 12 seconds. Anything beyond this is never read from the file at all (the `while frame_idx < max_frames` loop, `:401`, simply stops calling `cap.read()`); the truncation is reported back as `truncated: bool` (`:427`).
4. **Downscale cap:** if source width > 960px, every frame is downscaled with `cv2.resize(..., INTER_AREA)` (`:405-406`) to `max_width`, *before* detection — so detection and output both happen at the reduced resolution. There is no "detect small, redact at full-res" path; the resolution reduction is destructive to the output.
5. **Frame loop** (`:399-414`): one shared `ThreadPoolExecutor(max_workers=3)` is created for the whole clip (`:400`) and passed into `process_frame` for every frame, so the pool is reused across frames rather than rebuilt each time — each frame still fully independently re-runs face/plate/screen detection (no tracking, no frame skipping, no reuse of previous-frame boxes).
6. **Encoding:** `cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))` (`:390-392`). This is **MPEG-4 Part 2**, not H.264, despite the `.mp4` extension and the README calling it "H.264" nowhere but implying browser-playable MP4. Confirmed by direct test — see §6.
7. **Audio:** none. OpenCV's `VideoCapture`/`VideoWriter` never touch audio streams, so the source audio is silently dropped; nothing mux es it back in. This matches what the README says, but is worth restating because it directly maps to Phase 1's job (`CLAUDE.md`/`PHASES.md` Phase 1 wants ffmpeg audio muxing + real H.264).
8. **Temp files:** the uploaded file and the produced output file are both deleted in `app.py`'s `finally` block (`app.py:87-91`) after the response is built — including on exceptions and on the 500 path, since `finally` always runs. One gap: if the process is killed (SIGKILL, OOM) between file creation and the `finally` block, the temp files are orphaned — there is no separate reaper/cron for `uploads/`/`outputs/`. Not a bug in the code as written, just a limitation to know about before the hosted demo (Phase 6 demo rules already call for this).
9. **Empty-video guard:** if zero frames were readable, the (empty) output is deleted and a `ValueError` is raised (`:416-419`).

---

## 3. Hard-coded values (candidates for profile YAML per `CLAUDE.md`)

| Value | Location | Current |
|---|---|---|
| Face blur kernel/sigma | `detector.py:219` | `(99,99)`, sigma 30 |
| Pixelation block count | `detector.py:228` | 15 |
| Box padding before redaction | — | **none** (0%) |
| DNN face confidence threshold | `detector.py:125` | 0.5 |
| YOLO screen confidence threshold | `detector.py:190` | 0.4 |
| Haar face params | `detector.py:118-121` | scaleFactor 1.05, minNeighbors 4, minSize 20×20 |
| Haar plate params | `detector.py:160-162` | scaleFactor 1.05, minNeighbors 3, minSize 30×10 |
| Contour plate area/aspect filter | `detector.py:171,177` | area ≥1500, aspect 2.0–5.5 |
| NMS overlap threshold | `detector.py:90` | 0.3 |
| Video duration cap | `detector.py:352` | 12.0 s |
| Video max width | `detector.py:353` | 960 px |
| Upload size cap | `app.py:35` | 200 MB |
| Screen classes | `detector.py:30-34` | COCO tv/laptop/cell phone |
| YOLO weights path | `detector.py:77` | literal string `"yolov8n.pt"`, relative to CWD |

None of this is currently in a config file; it's all Python literals. This is exactly what `CLAUDE.md`'s "config over constants" rule and Phase 4's profile YAML are meant to fix — flagging it here as the concrete list to move.

---

## 4. Bugs, security problems and fragile spots

Ordered roughly by severity.

1. **`app.run(debug=True, host="0.0.0.0", port=5000)` — `app.py:121`.** Flask's debug mode enables the Werkzeug interactive debugger. If an unhandled exception reaches it (bypassing the `abort(500)` handling — e.g. an error during template rendering itself, not inside the `try/except` in `/process`), Werkzeug serves a debugger console that can execute arbitrary Python in the request, protected only by a PIN that's often derivable. Combined with `host="0.0.0.0"` (binds every network interface, not just localhost), this exposes remote code execution to anything that can reach the machine on port 5000, and also auto-reloads on file changes. This is a real problem even for a "local-first" tool the moment it's run on a shared network (coffee shop wifi, a lab machine, a Docker container with a published port). Should not ship as-is; at minimum `debug` should be off by default and only enabled via an explicit dev env var, and `host` should default to `127.0.0.1`.
2. **500 errors leak exception text to the client** — `app.py:86` (`abort(500, description=f"Processing failed: {exc}")`) rendered verbatim in `templates/index.html:24` (`{{ error }}`, Jinja auto-escaped so not an XSS vector, but still an information-disclosure issue). A malformed upload can produce internal messages like file paths or library internals in the browser. Should log server-side and show a generic message to the user.
3. **Video output is not H.264** — `detector.py:391` uses fourcc `"mp4v"`. Verified with `ffprobe` (see §6): the produced stream is `codec_name=mpeg4`, not `h264`. Many browsers (notably Safari, and Chrome depending on build) either won't play this at all inline or will play it inconsistently. Since `result.html:47` embeds the processed video directly as a `<video src="data:video/mp4;base64,...">`, a user could get a "video won't play" result despite the pipeline succeeding. This is exactly the problem Phase 1 is scoped to fix (real `libx264` encode via ffmpeg).
4. **No box padding before redaction** — `apply_gaussian_blur`/`apply_black_mask`/`apply_pixelation` (`detector.py:219-250`) all operate on the raw detector box with zero padding. A tight bounding box (common with Haar) can leave a sliver of face/plate at the very edge unredacted. This directly touches `CLAUDE.md` non-negotiable #1 ("privacy safety over speed") and is explicitly called out as a fix in `PHASES.md` Phase 1 (default 15% padding) — flagging it now as a measured gap, not assuming it's fine.
5. **Fixed blur kernel regardless of face size** — a 99×99 kernel is a small fraction of a large close-up face (weaker relative blur, more risk of residual identifiable structure) and enormous relative to a 20×20 px detection (over-blurs harmlessly). No scaling of kernel size to box size. Worth measuring, not assuming — this is a natural `eval/metrics.py` question later (does the redacted region actually become unidentifiable at various face sizes) rather than something to fix blind right now.
6. **Silent detector degradation** — `_get_dnn` (`detector.py:63-66`) and `_get_yolo` (`:75-79`) both swallow every exception and return `None`, silently disabling that detector for the rest of the process lifetime. If `yolov8n.pt` fails to load (e.g. offline and not cached, or corrupted), screen detection just silently stops happening — `process_frame` still runs, still returns `screens: 0`, and nothing in the response tells the user detection was degraded. Given `CLAUDE.md`'s "a missed face is a failure" framing, a silent detector outage is worse than a crash: the user has no way to know their upload wasn't fully checked.
7. **`yolov8n.pt` loaded by relative path** — `detector.py:77`: `YOLO("yolov8n.pt")`. This resolves relative to the process's current working directory, not relative to `detector.py`'s own location. It works today because the documented run command is `cd project && python app.py`, and the 6.5 MB weights file is committed at `project/yolov8n.pt`. If the app is ever started from the repo root (e.g. a future `Dockerfile` with a different `WORKDIR`, or a test runner invoked from root), Ultralytics will look for `yolov8n.pt` in the wrong directory, not find it, and (per finding #6) silently fall back to no screen detection rather than erroring — or, if online, silently *download* a fresh copy to whatever the CWD happens to be, quietly breaking the "no cloud, no telemetry" local-first claim in `CLAUDE.md` for that one first run.
8. **`.avif` uploads are accepted by `cv2.imread` but rejected by the app** — `ALLOWED_IMAGE_EXTENSIONS` (`app.py:19`) does not include `avif`, so `_allowed()` (`app.py:38-39`) rejects the two `.avif` files that are actually present in `images_CV_AAT/` (`car-park_1203-3451.avif`, `tips-home-office-setup3.avif`) with a 400 before they ever reach `detector.py` — even though `cv2.imread` on this machine's OpenCV build (4.14.0) decodes them fine (confirmed in §5). Not a bug exactly (avif support in `cv2.imread` isn't guaranteed across OpenCV builds/platforms, so being conservative in the whitelist is defensible), but it means roughly a fifth of the provided sample images can't be exercised through the real upload path, only by calling `detector.py` directly, as this audit did.
9. **No content-type/magic-byte validation, only extension** — `_allowed()` (`app.py:38-39`) trusts the filename's extension entirely. A non-image/non-video file renamed to `.jpg` will fail inside `cv2.imread`/`cv2.VideoCapture` (returns `None`/`isOpened()==False`) and surface as a generic 500 (see #2), rather than a clean 400. Low risk given `MAX_CONTENT_LENGTH` and no execution of uploaded content, but worth a magic-byte sniff for a cleaner failure mode later.
10. **Committed model weights** — `project/yolov8n.pt` (6.5 MB) is committed directly in git (`git log` shows it landing in the initial commit). It's under `CLAUDE.md`'s 10 MB threshold so this is technically compliant, but it's Ultralytics AGPL-3.0-licensed weights and there is currently no `LICENSE` file or "Licenses" section anywhere in the repo listing this — `CLAUDE.md` explicitly requires that before/alongside using any third-party weights. Flagging as a compliance gap to close, not urgent for Phase 0 itself.
11. **`instructions.txt` is corrupted/duplicated** (see §1) and describes an outdated (image-only) scope. Purely a docs-hygiene issue, already scheduled for a move to `docs/` in Phase 5.
12. **Thread-safety of shared cascade/DNN/YOLO singletons is assumed, not verified.** `process_frame` (`detector.py:271-285`) calls the same module-level `_haar_face_clf`, `_haar_plate_clf`, `_dnn_net`, and the YOLO singleton from `detector.py:191` concurrently across a `ThreadPoolExecutor`. This works today because `app.run()` at `app.py:121` doesn't pass `threaded=True`, so the Werkzeug dev server itself only ever handles one `/process` request at a time — the *within-request* concurrency (3 workers detecting faces/plates/screens on the same frame simultaneously) is fine because it's 3 different detector objects, not the same object from two threads at once. But if a future change adds `threaded=True` or a production WSGI server (gunicorn with multiple threads per worker), two *concurrent requests* would call `detectMultiScale` on the same shared `_haar_face_clf` object from two threads simultaneously — OpenCV's Haar cascade classifier is generally reported thread-safe for read-only inference, but this has not been verified for this specific build/version and should be checked before Phase 5's gunicorn deployment.

---

## 5. Environment setup

Per your instructions: created the venv with Python 3.12, not the system 3.14.

```
python3.12 -m venv .venv
echo "3.12" > .python-version
source .venv/bin/activate
pip install -r project/requirements.txt
```

**Result: all four declared dependencies (plus their transitive deps — `ultralytics` pulls in `torch` 2.14.0, `torchvision`, `matplotlib`, etc.) installed cleanly with no errors, no version conflicts, and no build-from-source steps.** Full resolved set is in the pip output; nothing failed, so there's nothing to stop and report per your "tell me what failed" instruction — everything installed. `ffmpeg` was not exercised by the requirements install (the current code doesn't call it) but was confirmed separately as v9.0.2 with `libx264` available on this machine, which is what Phase 1 will need.

Versions actually installed: Python 3.12.5, Flask 3.1.3, opencv-python 4.14.0.94, numpy 2.5.3, ultralytics 8.4.161, torch 2.14.0.

## 6. Running against the sample images (`images_CV_AAT/`)

Ran `detector.process_image()` directly (not through the Flask app, to avoid the `.avif` extension rejection in `app.py`'s whitelist — see finding #8) against every file in `images_CV_AAT/`:

| File | Result | faces | plates | screens | time |
|---|---|--:|--:|--:|--:|
| Skoda-Superb-road-India.jpg | OK | 2 | 2 | 0 | 5.70s |
| Watermark_Inside_Title-Image_0708_v1.jpg | OK | 0 | 0 | 1 | 0.17s |
| a-happy-business-man-...jpg | OK | 2 | 0 | 1 | 2.20s |
| best-uk-and-international-street-photography-2020.webp | OK | 8 | 0 | 0 | 1.06s |
| car-park_1203-3451.avif | OK | 1 | 1 | 1 | 0.28s |
| gettyimages-2225625308-612x612.jpg | OK | 0 | 3 | 0 | 0.13s |
| images (1).jpg | OK | 1 | 0 | 0 | 0.04s |
| images (2).jpg | OK | 1 | 0 | 0 | 0.04s |
| india-skoda-license-plate.jpg | OK | 3 | 3 | 0 | 1.09s |
| tips-home-office-setup3.avif | OK | 14 | 4 | 2 | 4.15s |
| tips-home-office-setup3.jpg | OK | 12 | 4 | 2 | 3.23s |

**Everything ran without crashing.** No ground truth exists yet for these images (that's what Phase 0's eval scaffold is for), so these counts are not accuracy numbers — but two are worth flagging as likely false positives, matching the README's own documented limitation: `gettyimages-2225625308-612x612.jpg` (0 faces, 3 plates — plausible over-triggering of the contour heuristic) and `tips-home-office-setup3.*` (12-14 "faces" in what both filenames and content suggest is a home-office product shot, i.e. very likely mostly Haar false positives on furniture/patterns). Also note `tips-home-office-setup3.jpg` and its `.avif` twin give slightly different counts (12 vs 14 faces) on what should be the same image content — decode differences between JPEG and AVIF (chroma subsampling, compression artifacts) are enough to flip Haar detections, which is a useful illustration of why recall needs to be measured, not eyeballed.

I did not run anything through `app.py`/Flask's HTTP layer for this step — `process()` in `app.py` only adds file I/O, extension checking, and temp-file cleanup around the same `detector.py` calls, all of which were already read and reasoned about in §1; running the same images through curl/a browser would exercise the same code path minus the two `.avif` files it rejects.

## 7. Testing the video path (synthetic clip)

No sample video ships in the repo, so a synthetic 3s/640×480/15fps H.264 clip with a sine-wave audio track was generated locally with `ffmpeg -f lavfi ...` (not committed, scratch file only) to exercise `process_video()` end to end.

- `process_video()` completed successfully: 45/45 frames processed, no truncation, output file written.
- **Confirmed audio is dropped** — the output has exactly one stream, no audio track (expected, matches §2/README).
- **Confirmed the codec claim in finding #3**: `ffprobe` on the output reports `codec_name=mpeg4`, i.e. MPEG-4 Part 2, not H.264, despite the source being H.264 and the container being `.mp4`.

## 8. Summary — what works, what doesn't

**Works as documented:** image pipeline (all formats `cv2.imread` can decode, including `.avif`/`.webp` on this OpenCV build), video pipeline end-to-end (frame loop, duration cap, downscale, counts), parallel detector execution, temp file cleanup on success and on error, DNN/YOLO graceful fallback when model files are absent, no crashes anywhere in this audit.

**Doesn't work / doesn't match the "local-first, safe-by-default" bar in `CLAUDE.md`:** Flask debug mode + `0.0.0.0` binding (finding #1), non-H.264 video output that may not play in all browsers (finding #3), zero box padding before redaction (finding #4), silent detector degradation with no user-visible signal (finding #6), exception messages leaked to the client (finding #2). None of these are things Phase 0 is supposed to fix — they're recorded here so Phases 1–5 have a concrete, line-referenced punch list instead of re-discovering them.
