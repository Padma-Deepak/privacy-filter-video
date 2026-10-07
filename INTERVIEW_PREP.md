# Privacy Filter — One-Day Interview Prep

Everything here is derived from the current code and measured results in this
repository. Where an answer depends on what *you personally* built, debugged or
decided, it is marked **`>> YOU FILL THIS IN`**. Do not claim another contributor's
work as your own; explain the whole system, then state your contribution precisely.

**Legend:** 🟩 = implemented in this project · ⬜ = general concept you should know.

**Contents**

0. What changed and why
1. Complete architecture and data flow
2. Technologies and concepts: what, why, alternatives
3. Project-specific interview Q&A
4. Computer-vision, web and video fundamentals
5. End-to-end trace
6. Code-level questions
7. Why it was designed this way
8. Limitations and improvements
9. Contribution questions
10. HR and behavioral questions
11. Rapid-fire definitions
12. Final cheat sheet, likely questions and danger areas

---

# 0. What changed and why

This project began as a coursework demo that processed each frame independently
with Haar/DNN face detection, heuristic plate detection and YOLO screen detection.
It became a local selective-redaction system through several measurable engineering
changes.

## 0.1 The original detector produced excessive false boxes

The legacy Haar+DNN face path produced 828 false positives on the fixed 300-image
WIDER FACE subset. On real footage, Haar also produced many small false boxes over
hair, clothing and background objects.

The review path now uses OpenCV YuNet. At confidence 0.8 on the same subset it
produced:

| Metric | Haar+DNN legacy | YuNet 0.8 |
|---|---:|---:|
| Precision | 49.8% | 98.0% |
| Recall | 27.2% | 47.9% |
| False positives | 828 | 29 |
| Detector FPS | 4.29 | 39.18 |

The important nuance: YuNet 0.6 has higher recall, 63.9%, but more false positives,
281. Threshold 0.8 was chosen to make interactive review much less noisy. That is
an explicit precision/recall trade-off, not a claim that 0.8 is universally safest.

## 0.2 Frame-independent processing became tracking

Detection answers “where is a face in this frame?” It does not answer “is this the
same person as last frame?” `ClassTracker` adds persistent track IDs using
velocity-aware IoU matching, with ByteTrack as a secondary ID suggestion.

Two privacy-sensitive rules shape the tracker:

1. Every raw detection must be returned for redaction immediately. ByteTrack never
   gets to filter a detection out.
2. The final redaction region is the padded union of the raw box and the smoothed
   box. Smoothing can reduce jitter but cannot leave part of the raw detection open.

Gap filling keeps a recently seen track hidden during brief detector misses.

## 0.3 Fixed automatic filtering became selective redaction

The user can draw around one or more faces that should stay visible. The rectangle
is matched to a detected face track on that frame. The choice follows the track
through the clip; it is not a fixed transparent rectangle.

Every track defaults to hidden. Ambiguous and unmatched selections fail closed.
Manual hide regions take priority over keep-visible choices.

## 0.4 The video pipeline became production-oriented

The old path downscaled the output, lost audio and used OpenCV-only encoding. The
current review path:

- streams frames instead of loading the full clip;
- detects on a copy capped at 960 pixels wide;
- scales boxes back to full-resolution coordinates;
- applies redaction at the original resolution;
- handles rotation explicitly from `ffprobe` metadata;
- writes a silent intermediate and uses FFmpeg for H.264/AAC;
- keeps or mutes the first audio stream;
- removes source metadata and chapters; and
- cleans temporary files on success, error and cancellation.

## 0.5 A blocking request became a bounded job system

Analysis and export run in a one-worker `ThreadPoolExecutor`. Jobs move through
`queued → analysing → ready → exporting → complete`, with `error` and `cancelled`
paths. Jobs are session-owned, cancellable and expired after inactivity.

## 0.6 Security and privacy behavior was tightened

- Flask debug mode defaults off and the server binds to `127.0.0.1`.
- State-changing review requests require a CSRF token.
- A session can access only its own jobs.
- Responses are private/no-store and cannot be framed.
- Upload names never become filesystem paths; random job IDs are used.
- Detector load failures are surfaced instead of silently degrading.
- Sources are deleted after export, cancellation, failure or expiration.
- Redaction reports contain choices and hashes, not original pixels or filenames.

## 0.7 Proof

- 103 Python tests pass.
- 4 JavaScript preview tests pass.
- Detector numbers come from scripts in `eval/` and saved CSV output.
- Tests cover tracking invariants, audio, metadata removal, selection behavior,
  session isolation, cleanup and rendered preview masking.

---

# 1. Complete project architecture

## 1.1 What it is

Privacy Filter is a local-first image and video anonymizer. A creator or journalist
uploads footage, selects the people who may remain visible, and exports a video in
which every other detected face is hidden.

It does not perform face recognition. It tracks bounding boxes using motion and
overlap.

## 1.2 Users

| User | Need | Current profile behavior |
|---|---|---|
| Creator/vlogger | Hide strangers without frame-by-frame editing | Strong blur, 15% padding, audio kept |
| Journalist/reporter | Stronger visual redaction and fewer identifying channels | Solid masks, 25% padding, audio muted |

Both profiles remove source metadata and generate a report. Both currently use
YuNet confidence 0.8 and process faces only.

## 1.3 Main components

| Component | Responsibility |
|---|---|
| `project/app.py` | Flask entry point and legacy route |
| `project/review_api.py` | Review endpoints, sessions, CSRF and response headers |
| `project/core/jobs.py` | Job state, queue, cancellation, expiration and deletion |
| `project/core/faces.py` | Thread-safe YuNet loading and face detection |
| `project/core/tracking.py` | IDs, IoU matching, smoothing, padding and gap filling |
| `project/core/selection.py` | Draw-to-track matching and fail-closed choice validation |
| `project/core/review.py` | Analysis, previews, rendering, FFmpeg export and reports |
| `project/core/profiles.py` | YAML validation and filter application |
| `project/core/video.py` | Probe, rotation, coordinate scaling and legacy streaming utilities |
| `project/static/review.js` | Browser workflow, playback, drawing and API coordination |
| `project/static/preview.js` | Canvas masking renderer |
| `eval/` | Reproducible detector evaluation |

## 1.4 High-level flow

```text
Browser
  │ POST /api/jobs (video + profile + CSRF)
  ▼
Flask review API
  │ creates session-owned job
  ▼
JobManager (one background worker)
  │
  ├─ ffprobe metadata
  ├─ OpenCV frame decode
  ├─ YuNet face boxes
  ├─ ClassTracker track IDs and padded regions
  └─ analysis geometry stored in RAM
  │
  ▼
Browser review
  ├─ streams original source with range requests
  ├─ fetches per-frame geometry
  ├─ draws keep selections
  └─ previews masks on canvas
  │ POST /export (choices + manual masks + audio mode)
  ▼
Export worker
  ├─ decodes original frames again
  ├─ applies reviewed geometry at full resolution
  ├─ writes silent temporary MP4
  ├─ FFmpeg encodes H.264 and AAC/mute
  ├─ removes metadata and chapters
  ├─ builds report
  └─ deletes source and analysis pixels
```

## 1.5 Why analysis and export are separate

Analysis is performed once and stores geometry. The user reviews those exact
tracks. Export reuses the same geometry rather than rerunning the detector.

This gives three benefits:

1. Preview and export use the same decisions.
2. The detector cost is paid once.
3. A nondeterministic or upgraded detector cannot change results after review.

The source must remain temporarily available until export because export needs the
original full-resolution pixels and audio.

## 1.6 API endpoints

| Method and path | Purpose |
|---|---|
| `GET /review` | Review interface |
| `POST /api/jobs` | Create analysis job |
| `GET /api/jobs/<id>` | Poll state and progress |
| `GET /api/jobs/<id>/analysis` | Get metadata and track summaries |
| `GET /api/jobs/<id>/frames/<n>` | Get preview JPEG or geometry |
| `GET /api/jobs/<id>/source` | Stream session-owned source video |
| `POST /api/jobs/<id>/select-face` | Match drawn rectangle to a face track |
| `POST /api/jobs/<id>/export` | Start reviewed export |
| `DELETE /api/jobs/<id>` | Cancel and delete |
| `GET /api/jobs/<id>/download` | Download completed result |
| `GET /api/jobs/<id>/report` | Download JSON or view HTML report |

## 1.7 Job state machine

```text
queued → analysing → ready → exporting → complete
            │           │         │
            └───────────┴─────────┴→ error/cancelled
```

Analysis and export share one worker. This serializes model use and limits memory
pressure. It is appropriate for a local tool, not a horizontally scalable service.

## 1.8 Data kept during a job

On disk:

- original uploaded source while the job is active;
- silent temporary video only during export;
- completed redacted output.

In memory:

- track metadata;
- per-frame boxes;
- small base64 track thumbnails;
- job state and report.

After successful export, the source and analysis data are deleted. The report does
not contain thumbnails, filenames or original metadata.

## 1.9 Testing at a glance

The suite contains unit, integration and synthetic media tests:

- pure IoU, matching and precision/recall tests;
- tracker identity, smoothing, gap-fill and raw-coverage tests;
- video resolution, duration, VFR, audio and cleanup tests;
- Flask security/session tests;
- click-to-track and export pixel tests;
- JavaScript tests proving the preview actually paints masks.

There is not yet a real annotated-video leak-rate suite. That is the largest
remaining evidence gap.

---

# 2. Technologies and concepts — what, why, alternatives

## 2.1 🟩 Flask

**What:** a lightweight Python web framework.

**Why here:** the app needs a small set of pages and JSON endpoints, not a full ORM
or admin system. Flask keeps the request layer small while the CV pipeline lives in
plain Python modules.

**Alternatives:** FastAPI for typed async APIs and automatic OpenAPI; Django for a
larger data-heavy product; Gradio/Streamlit for a quicker but less custom demo.

**Interview angle:** Flask handles HTTP, not long-running video work. That is why
processing is placed behind a background job abstraction.

## 2.2 🟩 OpenCV

**What:** a computer-vision library for image arrays, decoding, resizing, drawing,
filters, model inference and video I/O.

**Why here:** it supplies `VideoCapture`, NumPy-compatible frames, YuNet's
`FaceDetectorYN`, Gaussian blur and pixel operations in one dependency.

**Alternatives:** PyAV for richer video timing, Pillow for still images, ONNX Runtime
for model inference, GStreamer for streaming media.

**Sharp edge:** OpenCV video writing does not preserve source audio. FFmpeg is used
for the final mux/encode step.

## 2.3 🟩 NumPy

Frames are arrays with shape `(height, width, channels)`. Cropping a face is array
slicing; masking changes pixel regions in place. Understand dtype `uint8`, BGR
channel order and why coordinates must be clamped.

## 2.4 🟩 YuNet

YuNet is the current face detector, loaded through OpenCV's `FaceDetectorYN`. The
model returns a box, landmarks and confidence; this project uses the box and filters
by a configurable confidence threshold.

Why it replaced Haar in review: on the same local evaluation set it had much higher
precision, recall and throughput. Its weights are fetched from a pinned OpenCV Zoo
revision, checksum-verified and covered by an MIT licence.

Alternatives: SCRFD, RetinaFace, YOLO face models, MediaPipe face detection. A model
should be chosen using the same dataset, threshold sweep, hardware and metrics.

## 2.5 🟩 ByteTrack plus custom IoU continuity

ByteTrack supplies candidate IDs for new detections, but it is not authoritative.
The project first matches raw boxes to its own predicted track positions using IoU.

Why: relying directly on ByteTrack could drop first-frame detections because of an
internal confidence floor, and its continuity threshold could churn IDs under motion.
Both behaviors are inconvenient in ordinary tracking and unsafe for redaction.

Alternatives: SORT, DeepSORT, BoT-SORT, OC-SORT or embedding-based re-identification.

## 2.6 🟩 FFmpeg and ffprobe

`ffprobe` reads width, height, duration, streams and rotation. FFmpeg creates the
final H.264/AAC MP4, keeps or removes audio, strips metadata and enables fast-start.

Know the distinction:

- MP4: container.
- H.264: video codec.
- AAC: audio codec.
- `yuv420p`: widely supported pixel format.
- `+faststart`: moves MP4 metadata so playback can begin before full download.

## 2.7 🟩 HTML video and Canvas

The `<video>` element handles native playback and range requests. A canvas overlays
the frame, draws masks and captures pointer rectangles. Browser coordinates are
converted to source-video coordinates before they are sent to the API.

The browser mask is a preview approximation. The export uses OpenCV on original
pixels.

## 2.8 🟩 YAML profiles

Creator and Journalist settings are data rather than branches spread through code.
The loader validates exact fields, ranges, types, class names and safe metadata
behavior.

Alternative: JSON or Python configuration. YAML is readable for users, but safe
loading and strict validation are mandatory.

## 2.9 🟩 pytest and Node's test runner

Pytest covers Python logic and media integration. Node's built-in runner exercises
the browser-independent preview functions. Synthetic frames and generated clips
make tests deterministic and keep private footage out of the repository.

## 2.10 ⬜ Threading and locks

The job queue uses one worker. Job fields are protected with `RLock`; cancellation
uses `threading.Event`; model access is serialized. Python threads are acceptable
because much OpenCV/FFmpeg work occurs in native code or child processes, but this
is not a high-throughput multi-user design.

---

# 3. Project-specific interview Q&A

## Q1. “Give me an overview of the project.”

**Strong answer:**

“Privacy Filter is a local-first video and image anonymizer. It detects faces with
YuNet, assigns persistent track IDs using motion and IoU, and lets the user draw
around people who may remain visible. Every other detected track is hidden by
default. Analysis is performed once, then export reuses the reviewed geometry on
the full-resolution source. OpenCV handles frame processing and FFmpeg produces an
H.264 video, keeps or mutes audio and removes metadata. The current detector was
chosen from a reproducible 300-image WIDER FACE comparison: at threshold 0.8 it
reduced false positives from 828 in the legacy path to 29, although its 47.9%
recall means missed faces remain an honest limitation. The project has 103 Python
tests and 4 JavaScript preview tests.”

## Q2. “Why local-first?”

Sensitive footage may contain sources, bystanders, homes, vehicle plates or GPS
metadata. Uploading it to a third party creates another privacy boundary. Running
locally keeps source pixels on the user's machine. The trade-off is that speed and
model capacity depend on local hardware.

## Q3. “Why YuNet instead of Haar?”

Because it performed better on the same measured dataset. Haar+DNN had 49.8%
precision, 27.2% recall and 828 false positives. YuNet 0.8 had 98.0% precision,
47.9% recall and 29 false positives, while also running faster in the local
detector-only benchmark. The decision is evidence-based, not reputation-based.

Follow-up: threshold 0.6 gives better recall, so the next decision should use
annotated video leak rates, not only image metrics or visual cleanliness.

## Q4. “What is the difference between detection and tracking?”

Detection finds face boxes independently in a frame. Tracking connects boxes over
time and assigns an ID. Detection says “face at these pixels”; tracking says “this
is probably the same moving face as the previous frame.” Tracking does not recover
a face the detector has never seen.

## Q5. “How does click-to-select work?”

The browser converts the drawn rectangle into source coordinates. The backend looks
only at raw face detections on that frame. A candidate must have its center inside
the selection and IoU of at least 0.2. If no candidate matches, or the top two are
too similar, selection fails. Otherwise the matching track ID receives `keep`.

## Q6. “Why do tracks hide by default?”

Fail-closed behavior. A missing client choice, stale UI or newly appearing track
must not become visible accidentally. Only an explicit validated `keep` choice
exempts a track.

## Q7. “Why not trust ByteTrack's output directly?”

ByteTrack can omit a detection on its first frame because of its internal confidence
behavior, and can assign a new ID when its own matching loses continuity. In a
privacy filter, neither event should remove the raw detection or silently change a
user's selection. The project therefore always emits every raw box and uses its own
velocity-aware IoU history as the primary continuity rule.

## Q8. “What is IoU?”

Intersection over Union is the overlapping area of two boxes divided by their total
union area. IoU 1 means identical boxes; 0 means no overlap. It is used both for
evaluation matching and tracker data association, with different thresholds.

## Q9. “How do you prevent smoothing from exposing a face?”

An exponential moving average can lag behind a fast-moving detection. The project
takes the union of the smoothed box and current raw box, then pads that union. This
keeps smoothing as a visual improvement without shrinking coverage below the actual
detection.

## Q10. “What is gap filling?”

When a known track is briefly not detected, the tracker keeps returning an
extrapolated/padded box for a bounded number of frames. This reduces flicker. A
longer buffer improves continuity but can leave ghost masks after false detections;
a shorter buffer may expose a real face during detector misses.

## Q11. “Why analyze once and export later?”

The user must review stable track IDs. If export reran detection, the boxes could
differ from the preview. Recording geometry once makes export consistent and avoids
paying inference cost twice. The trade-off is memory proportional to the number of
frame records, so the app enforces frame/record/track limits.

## Q12. “How is full resolution preserved?”

Detection may run on a resized copy. `detection_scale_factor` records the scale,
and boxes are divided by that factor before tracking/redaction. Export decodes the
original frame and applies the scaled boxes there.

## Q13. “Why both OpenCV and FFmpeg?”

OpenCV makes per-frame pixel processing convenient but does not preserve audio and
offers limited production encoding control. FFmpeg handles H.264, AAC, stream
mapping, metadata removal and browser-compatible MP4 packaging.

## Q14. “How is variable-frame-rate video handled?”

Analysis records decoder timestamps for browser frame mapping, but export uses a
constant average FPS. Duration is bounded to the probed source duration. This keeps
audio approximately synchronized, but exact local VFR timing is not preserved and
is documented as a limitation.

## Q15. “How do background jobs work?”

`JobManager` stores job objects in memory and submits analysis/export to a single
worker. Progress updates are lock-protected. Cancellation sets an event checked
between frames and while polling FFmpeg. Completed or inactive jobs expire and
their private directories are removed.

## Q16. “How do you isolate users?”

Each session receives a random owner token and CSRF token. Job lookup requires both
the random job ID and matching owner. State-changing calls require the CSRF header.
Cross-session tests verify another client cannot access a job.

## Q17. “What exactly do the profiles change?”

Creator uses blur, 15% padding and keeps audio. Journalist uses a solid mask, 25%
padding and mutes audio. Both remove metadata and produce a report. Currently both
use the same YuNet threshold, so Journalist is stronger redaction after detection,
not proven higher recall.

## Q18. “How did you test it?”

Pure algorithms have small unit tests. Video behavior uses generated clips and
`ffprobe` assertions. Selective review uses Flask clients, synthetic colored faces
and pixel assertions to prove kept and hidden regions. JavaScript tests use a fake
canvas context and verify masking operations. The biggest missing test is annotated
real-video leak-rate evaluation.

## Q19. “What would you improve next?”

First annotate short clips and measure frame-level and track-level leak rates. Then
tune detector confidence and tracker behavior from those results. After that, add
appearance-based re-identification for crossings/re-entry and evaluate a licensed
plate detector before returning plates to the main UI.

## Q20. “What was the hardest design decision?”

**`>> YOU FILL THIS IN`** Choose one real decision you understand personally:

- precision 0.8 versus recall 0.6;
- tracking continuity versus privacy coverage;
- keeping originals during review versus deleting immediately;
- native video playback plus canvas versus server-generated frame galleries.

Explain the alternatives, evidence, decision and remaining downside.

---

# 4. Fundamentals to know

## 4.1 Computer vision

- Pixel arrays, `uint8`, BGR versus RGB.
- Bounding boxes: `xywh` versus `xyxy`.
- Detection confidence and thresholding.
- True positive, false positive and false negative.
- Precision versus recall.
- IoU and one-to-one matching.
- Non-maximum suppression.
- Small/medium/large object performance.
- Exponential moving average smoothing.
- Data association and identity switches.
- Domain shift: WIDER FACE versus vlog or newsroom footage.
- Why detection recall is not the same as privacy safety.

## 4.2 Video

- Frame, FPS, duration and timestamp.
- Constant versus variable frame rate.
- Codec versus container.
- Decode, transform, encode and mux.
- Rotation stored in pixels versus metadata.
- Audio/video synchronization.
- HTTP range requests for seeking.

## 4.3 Web/backend

- GET, POST and DELETE semantics.
- `202 Accepted` for asynchronous work.
- Polling and job state machines.
- Session cookie and CSRF token.
- Authentication versus authorization.
- Locks, thread pools and cancellation.
- File-size limits and untrusted uploads.
- Cache-control and temporary-file cleanup.

## 4.4 Frontend

- DOM events and pointer capture.
- Canvas coordinate transforms.
- Native video events and `requestVideoFrameCallback`.
- Promises, `async`/`await` and polling.
- Why missing mask geometry must fail closed.

---

# 5. End-to-end trace

Use this answer for “walk me through what happens after I upload a video.”

1. The browser submits multipart form data to `POST /api/jobs`, including the file,
   chosen profile and CSRF token.
2. Flask validates the extension, loads and validates the YAML profile, creates a
   random private job directory and saves the source with restrictive permissions.
3. The job enters `queued`, then the single worker changes it to `analysing`.
4. `ffprobe` reads dimensions, duration, streams and rotation. OpenCV opens the
   source with automatic orientation disabled.
5. Each decoded frame is explicitly rotated. A reduced copy is created when the
   width exceeds 960 pixels.
6. YuNet detects faces. Boxes are scaled to full-resolution coordinates.
7. `ClassTracker` matches boxes to predicted prior positions, assigns stable per-job
   IDs, smooths them and returns padded regions. Every raw detection is present.
8. Analysis stores per-frame records, track summaries, timestamps and small
   in-memory thumbnails. The state becomes `ready`.
9. The browser fetches analysis metadata, streams the original video and prefetches
   frame geometry. The canvas paints the live privacy preview.
10. When the user draws a face rectangle, the API matches it to one unambiguous raw
    detection and returns that track's positions. The browser sets it to `keep`.
11. On export, the API validates every choice, manual region and audio mode. Missing
    choices still mean hide.
12. Export decodes the source again and applies the recorded geometry to the
    original-resolution frames. Manual hides run after track choices.
13. OpenCV writes a silent intermediate. FFmpeg encodes H.264, optionally maps the
    first source audio stream to AAC, removes metadata/chapters and writes the final
    MP4.
14. A report records the input SHA-256, profile, choices, counts, versions, warnings
    and timings without storing original pixels.
15. The source and analysis thumbnails are deleted. The job becomes `complete` and
    the session can download the output and report.

---

# 6. Code-level questions

## 6.1 `ClassTracker.update()`

Be able to explain this order:

1. Assign IDs to current raw boxes.
2. Update EMA-smoothed positions.
3. Union raw and smoothed boxes.
4. Pad and clamp to frame dimensions.
5. Return every current detection.
6. Age unmatched histories and emit bounded gap-fill boxes.

Why it matters: the order preserves raw coverage while still reducing jitter.

## 6.2 `_assign_ids()`

Priority order:

1. Match to existing predicted history by descending IoU.
2. For an unmatched box, accept an unused brand-new ByteTrack suggestion.
3. Otherwise allocate a negative internal fallback ID.
4. Map internal IDs to clean external IDs `1, 2, 3...` per tracker instance.

The external remapping avoids ByteTrack's process-global counter leaking odd IDs
between jobs.

## 6.3 `match_face_selection()`

Only face records with `source == "detected"` are candidates. Gap-fill boxes cannot
be selected because they are predictions, not current evidence. The raw-box center
must be inside the drawn box and IoU must be at least 0.2. Near-ties fail as
ambiguous.

## 6.4 `render_frame()`

For every stored record, `is_hidden()` decides whether to apply the profile filter.
Then manual regions are applied afterward. That order means a manual hide overrides
an accidental keep choice.

## 6.5 `JobManager`

Know why it uses:

- `ThreadPoolExecutor(max_workers=1)`: serialize heavy model work;
- `RLock`: allow nested job operations while protecting state;
- `Event`: cooperative cancellation;
- monotonic time: duration/expiry unaffected by system clock changes;
- random 48-hex IDs: unpredictable directories and API keys;
- a reaper thread: clear expired jobs and safe orphan directories.

## 6.6 `detect_faces()`

The model cache is keyed by model path. A lock protects mutable OpenCV model state
because input size and score threshold are set before every call. Missing weights
raise a clear setup error instead of silently invoking a weaker detector.

## 6.7 Evaluation matching

`match_boxes()` creates every ground-truth/prediction pair above IoU 0.5, sorts by
IoU and greedily assigns each GT and prediction at most once. Unmatched GT boxes are
false negatives; unmatched predictions are false positives.

---

# 7. “Why did you do it this way?”

| Decision | Reason | Trade-off |
|---|---|---|
| Local processing | Sensitive footage stays on the user's machine | Limited by local CPU and disk |
| YuNet 0.8 | Dramatically fewer false boxes in interactive review | Lower recall than 0.6 |
| Detect once, export later | Preview/export consistency and lower inference cost | Stores geometry; source retained until export |
| Hide by default | Fail closed for missing/new choices | User must explicitly keep people |
| Raw detection always emitted | Tracker cannot reduce privacy coverage | More duplicate/ghost coverage is possible |
| Union raw and smoothed boxes | Smoothing cannot lag behind raw evidence | Region can temporarily be larger |
| One job worker | Predictable local resource use and model safety | No parallel processing throughput |
| OpenCV plus FFmpeg | Convenient pixel operations plus reliable final media | Two tools and a temporary file |
| YAML profiles | Behavior is configurable and reviewable | Requires strict schema validation |
| Synthetic tests | Deterministic, small and privacy-safe | Less representative than real footage |
| Keep legacy route | Preserves baseline and coursework comparison | More code/documentation to maintain |

---

# 8. Limitations and improvements

## 8.1 Know these limitations cold

1. **Face recall is incomplete.** YuNet 0.8 recall is 47.9% on this subset and only
   24.5% for small faces.
2. **No video leak-rate number.** The project lacks annotated CVAT clips, so it
   cannot yet claim track-level privacy performance.
3. **Tracking is not identity recognition.** Crossings and re-entry may switch or
   split IDs.
4. **VFR is normalized.** Exact frame timing is not preserved.
5. **Only the first audio stream is kept.** Multi-track/spatial audio is simplified.
6. **Face filtering is not full anonymization.** Voice, body, clothing, location and
   context remain identifying.
7. **In-memory job state.** Restarting the process loses jobs; multiple WSGI workers
   would not share state.
8. **Local development server.** Packaging and production WSGI deployment remain.
9. **Legacy plate detector is noisy.** Plates/screens are excluded from the current
   main review experience.
10. **Model threshold is one operating point.** Creator and Journalist currently use
    the same detection threshold despite different risk goals.

## 8.2 Ranked next improvements

1. Annotate 5–10 representative videos and implement frame/track leak metrics.
2. Calibrate Creator and Journalist thresholds from those measurements.
3. Add appearance embeddings or stronger motion association for re-identification.
4. Test real rotation-tagged phone footage and more codecs.
5. Add a licensed, measured plate detector before enabling plates.
6. Move job state to a persistent queue only if multi-user deployment is required.
7. Add Docker, CI, a production server and security documentation.

## 8.3 How to discuss a weakness

Good answer pattern:

> “The current detector has 98% precision but only 47.9% recall on our subset, so
> I would not claim it catches every face. We selected that threshold to fix a
> specific false-positive usability problem. The next step is annotated-video leak
> evaluation, then separate thresholds or models for Creator and Journalist.”

Bad answer:

> “YuNet is accurate and the tracker makes sure no faces are missed.”

---

# 9. Contribution questions

## “What did you personally do?”

**`>> YOU FILL THIS IN`** Use this structure:

1. The code you personally wrote or changed.
2. The bug or limitation you personally observed.
3. The evidence you collected.
4. The design decision you influenced.
5. What another contributor did.

Example shape—not a claim to copy:

> “I focused on the review workflow and testing. I reproduced the false-box issue
> on a real clip, traced it to the legacy detector rather than the canvas, compared
> detector results, and helped validate that selecting one track left it visible
> while other tracks were actually masked. The earlier streaming/tracking foundation
> came from the contributor branch; I integrated and verified it.”

## “Tell me about a bug you found.”

Choose one:

- noisy Haar face boxes;
- selecting a person did not initially show live blur for others;
- ByteTrack could gate first-frame detections;
- non-browser-playable MOV preview;
- Flask debug/network exposure;
- detector load failures silently disabling protection.

Use: symptom → investigation → root cause → fix → test/measurement → limitation.

## “What would you do differently?”

A strong answer: start video annotation earlier. Image detector metrics helped, but
the product's real risk is temporal leakage. Annotated clips would have made
threshold and tracker choices evidence-based sooner.

---

# 10. HR and behavioral

## Project questions

- Why did you choose this project?
- Who is the user?
- What part are you most proud of?
- What was the hardest bug?
- Where did you disagree with an initial approach?
- What did measurement change your mind about?
- What would you build with two more weeks?
- How did you divide work with contributors?

## STAR story prompts

### False-positive investigation

- **Situation:** selected face remained clear, but small masks appeared on mouths,
  walls and clothing.
- **Task:** determine whether the problem was UI masking, tracking or detection.
- **Action:** inspect frame geometry, compare legacy and YuNet on the same data,
  choose a measured threshold and add regression tests.
- **Result:** false positives fell from 828 to 29 on the fixed subset and the live
  review became substantially cleaner.

### Privacy invariant

- **Situation:** an off-the-shelf tracker could omit detections or churn IDs.
- **Task:** use tracking without allowing it to weaken redaction.
- **Action:** emit every raw detection, make custom IoU continuity authoritative,
  and union raw/smoothed boxes.
- **Result:** tests prove first-frame and fast-moving raw boxes remain covered.

Only use a story as your own if it matches your real contribution.

---

# 11. Rapid-fire

- **Detection:** finding object locations independently in a frame.
- **Tracking:** associating detections across time.
- **Bounding box:** rectangle represented here mainly as `(x, y, w, h)`.
- **Confidence:** detector score indicating belief in a prediction.
- **IoU:** intersection area divided by union area.
- **Precision:** TP / (TP + FP).
- **Recall:** TP / (TP + FN).
- **False positive:** predicted face without matching ground truth.
- **False negative:** ground-truth face not detected.
- **NMS:** removes overlapping duplicate detections.
- **EMA:** weighted average that smooths current and previous positions.
- **Gap fill:** temporary predicted box during detector misses.
- **Identity switch:** tracker assigns a person's track ID to another person.
- **Track fragmentation:** one person receives multiple IDs over time.
- **Codec:** algorithm for encoding audio/video, e.g. H.264.
- **Container:** file format holding streams, e.g. MP4.
- **Muxing:** combining video and audio streams into a container.
- **VFR:** frames have non-uniform presentation intervals.
- **CSRF:** browser is tricked into sending an authenticated state-changing request.
- **Fail closed:** uncertainty results in hidden/denied, not visible/allowed.
- **Thread lock:** prevents unsafe concurrent access to shared mutable state.
- **Checksum:** digest used to verify a downloaded model exactly.
- **Domain shift:** evaluation and real input distributions differ.
- **Local-first:** core processing works on the user's machine without cloud APIs.

---

# 12. Final cheat sheet

## Project in one screen

```text
Problem: manually anonymizing people across video is slow and error-prone.

Input: image or video.
Detector: YuNet 2023mar, confidence 0.8.
Tracker: custom velocity-aware IoU continuity + ByteTrack suggestions.
Selection: draw around people to keep; all other tracks hide by default.
Preview: native video + canvas masks.
Export: full-resolution OpenCV rendering → FFmpeg H.264/AAC or mute.
Privacy: local processing, session-owned jobs, cleanup, metadata removal.
Profiles: Creator blur/keep audio; Journalist solid/mute audio.
Tests: 103 Python + 4 JavaScript.

Measured result:
Haar+DNN: 49.8% precision, 27.2% recall, 828 FP, 4.29 FPS.
YuNet 0.8: 98.0% precision, 47.9% recall, 29 FP, 39.18 FPS.

Biggest limitation: no annotated-video frame/track leak-rate result yet.
Next step: annotate clips, measure leaks, then tune detector/tracker.
```

## 20 most likely questions

1. Give me an overview of the project.
2. What problem does click-to-select solve?
3. Why YuNet instead of Haar?
4. Explain precision versus recall using your results.
5. What is IoU and where do you use it?
6. Detection versus tracking?
7. Why not trust ByteTrack directly?
8. How do you guarantee a raw detection remains covered?
9. How does drawing a face select a moving person?
10. Why hide missing choices by default?
11. Why analyze once and export later?
12. How do you preserve full resolution?
13. Why OpenCV plus FFmpeg?
14. How is audio handled?
15. How do jobs, cancellation and cleanup work?
16. How are sessions isolated?
17. What differs between Creator and Journalist?
18. How did you test the system?
19. What are the most serious limitations?
20. What did you personally contribute?

## Danger areas

- Do not say the system detects every face. It does not.
- Do not call tracking face recognition. There are no identity embeddings.
- Do not quote detector FPS as end-to-end video FPS.
- Do not say Journalist has proven higher recall; it currently shares threshold 0.8.
- Do not say metadata removal guarantees anonymity.
- Do not claim exact VFR timing is preserved.
- Do not imply the browser blur is the export implementation.
- Do not claim plate/screen support is part of the current main review UI.
- Do not call WIDER FACE a tracking evaluation.
- Do not claim personal credit for work you only integrated or studied.

## Last rehearsal

Practice three answers aloud:

1. A 60-second project overview.
2. A 3-minute end-to-end upload-to-export trace.
3. A 2-minute bug story with evidence and an honest remaining limitation.

If you can explain those without opening the code, then answer the 20 questions
above with concrete file names and numbers, you are ready to discuss the project.
