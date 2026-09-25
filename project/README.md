# Privacy Filter application

This directory contains the Flask application. The default `/review` workflow
is a local face-only anonymizer with YuNet detection, motion tracking,
click-to-keep selection, full-resolution export, audio handling and a redaction
report. The original coursework interface remains available at `/legacy`.

## Install and run

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r project/requirements.txt -r eval/requirements.txt
python scripts/download_yunet.py
python project/app.py
```

Open `http://127.0.0.1:5000/review`.

FFmpeg with `libx264` must be installed for video review and export. YuNet
weights are fetched into `project/models/`, verified by checksum and ignored by
Git. Processing does not require a network connection after setup.

## Current pipeline

1. `review_api.py` validates the upload and creates a session-owned job.
2. `core/jobs.py` runs analysis and export in a bounded single-worker queue.
3. `core/review.py` streams frames and calls `core/faces.py` for YuNet detection.
4. `core/tracking.py` assigns stable per-job IDs and adds smoothing/gap filling.
5. The browser lets the user choose face tracks to keep visible.
6. Unselected tracks are hidden by default; manual hide regions take priority.
7. Export reuses reviewed geometry on original-resolution frames and invokes
   FFmpeg for H.264, AAC audio handling and metadata removal.
8. The source upload and in-memory preview crops are deleted after completion.

## Profiles

`profiles/creator.yaml` and `profiles/journalist.yaml` are validated by
`core/profiles.py`. Both currently process faces only and use YuNet at confidence
0.8. Creator uses blur and keeps audio. Journalist uses a solid mask, larger
padding and muted audio. Both strip metadata and produce reports.

## Directory structure

```text
project/
├── app.py
├── review_api.py
├── detector.py              legacy detectors and filters
├── core/
│   ├── faces.py             YuNet adapter
│   ├── jobs.py              job lifecycle and cleanup
│   ├── profiles.py          profile validation and redaction
│   ├── review.py            analysis and export
│   ├── selection.py         keep/hide matching and validation
│   ├── tracking.py          motion tracking
│   └── video.py             video probe/rotation utilities
├── profiles/
├── static/
├── templates/
├── jobs/                    private temporary job data, ignored
└── models/                  downloaded weights, ignored
```

## Legacy route

`/legacy` retains the original automatic face/plate/screen pipeline for
comparison. Its optional SSD face weights can be downloaded with
`python scripts/download_models.py`, but the upstream trained weights have no
licence file and must not be bundled. Haar and contour plate detection remain
noisy. The legacy path is not the behavior described by the main product UI.

## Limits

- Detector misses and false positives are possible; review the entire export.
- Tracking is motion-based and can split or switch identities.
- The browser preview approximates the export filter.
- Only the first audio stream is kept.
- Variable frame timing is normalized to an average frame rate.
- Voices and contextual identifiers are not anonymized by face filtering.

Measured detector results, test counts, licences and the full architecture are
maintained in the repository root `README.md`.
