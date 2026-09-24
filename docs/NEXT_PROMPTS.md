# Prompts for the next contributor

Open Claude Code in the repo root. It reads CLAUDE.md automatically. Paste ONE prompt at a time, review the result, and commit locally in small steps. Work on a branch and open a pull request into main. Do not push to main directly. Never commit personal footage, model weights, .venv or datasets. Numbers in the README were measured on another machine, so re-measure on yours.

## Step 0: get the code (human, in a terminal)
Either merge the pull request on GitHub, or run:
git remote add suhani https://github.com/suhanii-23/privacy-filter-video
git fetch suhani
git checkout -b upgrade suhani/upgrade

## Prompt 0: set up and verify (first session, no code changes)
> Read CLAUDE.md, docs/HANDOFF.md, docs/AUDIT.md and PHASES.md. Do not change any code and do not commit. 1) Set up as described in HANDOFF section 5: Python 3.12 virtualenv (record the machine architecture), check ffmpeg has libx264, install requirements, run scripts/download_models.py, and prepare the WIDER FACE subset (tell me if a manual download is needed). 2) Run the test suite and report pass and fail counts. 3) Re-run eval/run_baseline.py and compare with the numbers in the README, noting hardware differences. 4) Tell me anything in HANDOFF.md that does not match what you find. Then stop and wait.

## Prompt 1: Phase 2A, face detector comparison
> Start Phase 2, Prompt A from PHASES.md, with these additions. On a real test clip the Haar detector produced about 20 detections per frame for 2 real people and took 376 ms per frame (55% of pipeline time). For each candidate detector report recall, precision, detections per image and false positives per image on the WIDER FACE subset, plus ms per frame on CPU on this machine. Include YuNet (OpenCV) first: check its licence and how its model file is obtained, and add it to scripts/download_models.py with a pinned checksum. Add at least two more candidates (for example SCRFD/InsightFace and a YOLOv8-face model) and record every licence. Keep the pipeline default unchanged until the numbers are in, then recommend a default and a high-recall detector, justified by the numbers, and wait for my approval before switching. Do not use my personal clips in any committed evaluation. Local commits, no push. Stop after the evaluation and recommendation.

## Prompt 2: Phase 2B, plates and screens
> Start Phase 2, Prompt B from PHASES.md, with these additions. The current Haar plus contour plate detector fires about 5 times per frame on a real clip that has no cars, and one raw false positive covered 12.4% of the frame. Build a small negative set of images with no vehicles (indoor scenes, signs, screens, shelves) and report false positives per image for each plate detector next to mAP on the held-out plate set. Keep a maximum-area sanity filter for plate boxes, configurable. Screens: keep COCO classes as default and evaluate them honestly. Local commits, no push. Stop after the evaluation and wait.

## Later phases
Phases 3 to 7 are in PHASES.md. Before Phase 3, check HANDOFF.md for any unfinished Phase 1 items. Remember that detection stride above 1 can leave a newly appearing face unredacted for up to K-1 frames, so the Journalist profile must force stride 1.

## Human-only tasks
- Film or find 5 to 10 short clips (5 to 15 s each) and annotate faces and plates with track IDs in CVAT, as described in eval/video_gt/README.md.
- Check and record the licence of every model, dataset and sample clip. Note that the OpenCV res10 DNN face weights have no upstream licence.
- Pick 3 to 5 sample clips with permissive licences for the hosted demo.
