# GUIOdyssey Human Correspondence Annotation

GUIOdyssey human review creates a curated weak correspondence set for matcher
pretraining and NULL supervision. It is not the sole formal widget-mapping
benchmark because GUIOdyssey has no released accessibility tree or DOM candidate
set. Rules rank what a human should inspect; rules never create gold labels.

## Ready Pilot

The local pilot is:

```text
runtime/datasets/guiodyssey_review_pilot_20260721/
  manifest.json
  review_queue.jsonl
  review.html
  screenshots/
    download_manifest.json
    *.png
```

It contains 100 pairs selected from 628 cached episodes and 5,766 eligible
mirror-backed CLICK steps:

- 80 likely-correspondence candidates;
- 20 hard-NULL candidates;
- 175 unique pinned screenshots, about 260 MB;
- phone/tablet/fold transitions, including browser or WebView-like contexts;
- no rule-generated gold label.

Serve the directory instead of opening the HTML with a `file://` URL:

```bash
cd /Users/wuzewen/Projects/Omni/OmniTransfer
python -m http.server 8765 --directory runtime/datasets/guiodyssey_review_pilot_20260721
```

Open `http://127.0.0.1:8765/review.html`. Labels are saved continuously in
browser local storage. Use `导出 JSONL` before changing browser profile or
clearing site data.

## What To Label

Each item shows the source action target and a target-layout action target. The
question is:

> Does the source highlighted control have the same user-visible function as a
> control on the target screen?

Use exactly one label:

1. `correspondence`: the highlighted controls perform the same atomic function.
   Differences in text, style, position, size, density, or container are allowed.
2. `no_correspondence`: the source function is absent from the target screen.
   Do not use this merely because the red target SAM box is wrong.
3. `uncertain`: the screenshots do not provide enough evidence or the semantic
   equivalence is genuinely ambiguous.
4. `unusable_bbox`: a screenshot is corrupt, the source action is not visible,
   or a SAM box is too wrong to identify the intended source control.

If the corresponding target exists but the red target box is wrong, click the
correct target control and then choose `correspondence`. The click is stored as
`target_point_override_normalized` in the dataset-wide `0..1000` coordinate
system. The source side can be corrected in the same way. These points are
human labels for training/evaluation only; they are never replayed as runtime
coordinates.

Keyboard shortcuts `1` through `4` select the four labels. Add a short note for
ambiguous cases. A reviewed row preserves the candidate score and reasons, but
`selection.is_gold_label` remains false; only `annotation.label` is gold.

## Selection Policy

Eligible steps must have a CLICK action, a screenshot, a non-empty atomic
instruction, a usable SAM2 box, and a click sufficiently close to that box.
Launcher, home-screen, app-drawer, desktop icon-grid, and wallpaper screens are
excluded before pairing. Every reviewed source and target must be inside an app,
including Settings, a browser/WebView, or another application page.
Candidate priority combines:

- atomic-instruction agreement and same meta-task evidence;
- cross-device and cross-form-factor diversity;
- normalized position and scale shift;
- foldable and browser/WebView-like coverage;
- a small hard-NULL quota from same-app semantic conflicts.

Selection is greedily diversified by episode, atomic instruction, device
transition, and target form factor. Pair search uses bounded deterministic
neighbors within lexical/app blocks, so the full 8,334-episode release does not
require quadratic enumeration.

Normalized position changes are useful layout evidence, but extreme SAM box
scale disagreement is treated as annotation corruption rather than rewarded as
layout diversity. Candidate pairs whose normalized target-area ratio exceeds
`20x` are rejected before ranking.

## Quality Protocol

For paper data, use two annotators on at least all dev/test rows and 20% of train.
Adjudicate disagreements; do not resolve them with the selection rule. Exclude
`uncertain` and `unusable_bbox` from optimization. Use `correspondence` as
human-confirmed crop/relation supervision and `no_correspondence` for the same
learned NULL head.

Freeze splits by episode plus meta-task/app family before training. Never place
two steps from one episode in different splits. Report GUIOdyssey separately as
human-confirmed weak correspondence data, while AndroidControl and a dedicated
posture-paired collection remain the structured matcher benchmarks.

## Rebuild Commands

```bash
PYTHONPATH=src python scripts/build_gui_odyssey_review_queue.py \
  --annotations /path/to/GUIOdyssey/annotations \
  --output runtime/datasets/guiodyssey_review \
  --limit 100 \
  --likely-percent 80 \
  --require-individual-image-mirror

PYTHONPATH=src python scripts/download_gui_odyssey_review_images.py \
  --queue runtime/datasets/guiodyssey_review/review_queue.jsonl \
  --output runtime/datasets/guiodyssey_review/screenshots \
  --workers 16
```

The official screenshot release is a roughly 92.6 GB split ZIP. The downloader
instead fetches only selected PNGs from pinned Voxel51 individual-file mirrors:

- `Voxel51/gui-odyssey-train` at
  `920ab344102d68fc85c1a0fe37f38d9894b19e88`;
- `Voxel51/gui-odyssey-test` at
  `ccdb433074e1ae374621031fc011c906a0cfc1e9`.

The mirror is transport only. Every downloaded path, revision, byte count,
actual image size, orientation relation, and SHA-256 is stored in
`download_manifest.json`.
