# Suggested next work

These prompts start from the current YuNet, face-only selective-review branch.
Use one at a time. Do not rerun the old Phase 1/3/4 prompts: streaming, tracking,
selection, profiles, metadata removal and reports already exist.

## 1. Video privacy evaluation

Human prerequisite: annotate 5–10 short clips following
`eval/video_gt/README.md` and keep unlicensed or personal video outside Git.

> Read README.md, CLAUDE.md, docs/HANDOFF.md and eval/video_gt/README.md. Inspect
> the current analysis and tracking record format. Add a tested CVAT for video
> 1.1 parser using the real annotation files, then compute frame-level leak rate
> and track-level leak rate using CLAUDE.md's exact definitions. Compare YuNet
> confidence 0.6 and 0.8, report results by face size and scenario, and save the
> generated table under eval/results/. Do not change the production threshold
> until the measurements and recommendation are reviewable.

## 2. Tracking failure analysis

> Use the annotated clips and existing leak-rate evaluator to categorize track
> fragmentation, identity switches, crossings, re-entry and gap-fill failures.
> Add visual debugging output outside Git. Propose the smallest measurable
> improvement, compare it against the current tracker on the same clips, and
> preserve the invariant that raw detections are never gated by tracking.

## 3. Plate detector work

> The main review UI is intentionally faces-only because the legacy Haar plus
> contour plate detector is noisy. Build a licence-documented plate dataset with
> both positive and indoor negative images, train or adapt a plate detector,
> report mAP, recall, false positives per negative image and CPU FPS, then make a
> recommendation. Do not enable plates in the review UI before the evaluation is
> committed and reviewed.

## 4. Packaging and CI

> Add a non-root Docker image with FFmpeg/libx264 and a production WSGI server.
> Pin dependencies, add ruff and a GitHub Actions workflow for Python and
> JavaScript tests, create docs/SECURITY.md, and verify a clean checkout can
> download YuNet and run. Do not bundle model weights, datasets or footage.

## Human-only tasks

- Annotate the video ground truth in CVAT.
- Verify model, dataset and sample licences.
- Review real exported clips frame by frame before making privacy claims.
- Select permissively licensed samples before creating a hosted demo.
