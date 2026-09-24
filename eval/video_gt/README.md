# Video ground truth — annotation instructions

This directory will hold ground-truth track annotations for 5–10 short clips,
used to measure **track-level leak rate** and **frame-level leak rate**
(definitions in the root `CLAUDE.md`) once tracking exists (Phase 1 onward).

**Nothing is annotated yet.** No fake or placeholder annotation files exist in
this repo — do not create any. This README only documents the process so the
manual annotation work (filming/sourcing clips + labelling them in CVAT) can
start independently of the code work.

## What to film or source

5 to 10 clips, 5–15 seconds each, covering the scenarios the tool is meant for:

- A crowd or street scene (multiple faces, varied sizes/distances)
- Traffic or a car park (license plates, varied angles)
- A dashcam-style clip (motion blur, plates at speed)
- An indoor scene with screens/laptops/phones visible
- At least one clip with faces that leave and re-enter the frame (to exercise
  track re-identification / gap-filling once the tracker exists)

Keep clips short — this is explicitly the slowest part of the project
(`PHASES.md` Phase 0). Anything you film yourself avoids licensing questions;
if you source clips, use permissively licensed footage (Pexels/Pixabay or
similar) and record the source and licence for each clip here once you have
them, the same way `demo/SAMPLES.md` will for the hosted demo samples.

## Annotating in CVAT

1. Install CVAT locally (`docker compose up` from the [CVAT repo](https://github.com/cvat-ai/cvat)) or use the hosted app.cvat.ai for small personal projects.
2. Create one CVAT **task** per clip, uploading the clip as a video source.
3. Create two label classes: `face` and `plate`.
4. Use CVAT's **track** annotation mode (not per-frame shapes) so each face
   and each plate gets a single persistent track ID across frames — this is
   what "track ID" means in `CLAUDE.md`'s leak-rate definitions. Draw a
   bounding box on the object's first visible frame, then let CVAT interpolate
   between keyframes you place whenever the box needs correcting (movement,
   size change, occlusion). Mark a track "outside" for the frames where the
   object leaves frame or is fully occluded, and start a new track if the same
   physical face/plate reappears and you're not confident it's trackable
   through the occlusion — a new track ID for a re-appearance is fine and
   expected; don't force one ID across a gap you're not sure about.
5. Annotate every visible face and every visible plate, including small or
   partial ones at the edge of frame — recall is the metric that matters here,
   so under-annotating the ground truth itself would silently inflate measured
   recall later.

## Export format expected by `eval/`

Export each task as **CVAT for video 1.1** (XML) via CVAT's
`Task > Export task dataset`. Place the exported XML next to the clip:

```
eval/video_gt/
  clip01_crowd.mp4          (or a note here on where to obtain it, if not committed)
  clip01_crowd.xml          (CVAT for video 1.1 export)
  clip02_traffic.mp4
  clip02_traffic.xml
  ...
```

A CVAT-for-video XML `<track>` element carries the `id` and `label`
(`face`/`plate`) attributes directly, and each `<box>` child carries
`frame`, `xtl`, `ytl`, `xbr`, `ybr`, and `outside` — everything the leak-rate
scripts in a later phase need (`ground truth per frame per track`). A parser
for this format will be added in `eval/` once Phase 1's tracker exists and
there's something to compare it against — writing that parser now, before any
real annotation exists to test it on, would risk shipping an unverified format
assumption. This README is the contract; the parser comes with the first real
XML file.

## Licensing note

Don't commit raw video files if they're not small and permissively licensed —
follow the same `<10MB` / Git LFS / "fetched by a script" rule from
`CLAUDE.md` that applies to model weights. If a clip can't be committed
directly, document how to obtain it (URL + licence) instead, the same as
`demo/SAMPLES.md` will for the hosted demo.
