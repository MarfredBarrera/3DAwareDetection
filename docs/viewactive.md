# ViewActive reimplementation: status and next steps

This is a from-scratch reimplementation of ViewActive (Wu et al., arXiv:2409.09997)
as the standalone package `viewactive/`. The paper's code link is a placeholder,
so everything comes from the paper text and figures. Phase 1 is to reproduce
the paper. Phase 2 is to plug it into the study as a viewpoint selector and
compare it with `identity`, `random`, our ambiguity selectors and `oracle`.

**Fidelity status:** This is not yet a one-to-one reproduction of the paper's
training or recognition experiment. Do not use its current classifier results
as a matched baseline. The authors' public repository contains only a README;
the exact Objaverse UIDs/split, Blender scene, classifier checkpoints and class
scoring rules, and several training settings are not public. The paper-scale
run below is a controlled reimplementation on a different cohort and renderer.

*Last updated: 2026-10-14.*

## Blender reproduction in progress

The independent run in `outputs/viewactive/blender_full/` uses the corrected
object split (1,120 train, 80 validation, 400 seen test, 200 unseen test) and
renders all 132 views per object in Blender 4.5.14 at the paper's 1920×1680
resolution. It keeps glTF textures and materials. The pinned portable Blender
binary lives under `/data/cache/blender` and can be installed inside the dev
container with `bash scripts/install_blender.sh`; it is not baked into the
SAM3D image. Its scene uses an explicit 50° vertical perspective FOV, distance
1.4 in the normalized object frame, three area lights, a white background,
and Eevee with 8 samples. These scene values are assumptions because the paper
does not publish them. One Blender process uses two CPU threads per shard;
the rendering stage does not reserve another training GPU.

The 14-object Blender smoke run completed every stage. All 1,848 camera poses
passed geometric verification. No view was empty or clipped; the median mask
bounding box occupied 59% of the larger frame dimension. This validates the
rendering and data pipeline, not the paper's accuracy claim. The full render
resumes at shard boundaries. The corrected PyTorch3D split completed its
60-epoch model and evaluation first; its Fig. 4 quality values are within 0.04
of the paper's step-9 plot readings, but all four recognition models remain
substantially below the paper.
The paired appearance sheet for eight pilot objects is
`outputs/viewactive/blender_pilot/eval/render_comparison.png`.

The old 150-object pilot's white-padded object crop improved ResNet50 all-view
accuracy by 0.031 on seen and 0.062 on unseen objects, and ViT by 0.054 and
0.122. The paired 95% interval excluded zero only for the unseen categories.
This supports framing as one contributor, but does not explain the whole
recognition gap. The Blender run is configured with closer framing and no
extra classifier crop. Its holdout remains untouched until the render and
model are ready.

The corrected PyTorch3D run's per-class ceiling is recorded in
`outputs/viewactive/fidelity_split/eval/class_audit.json`. Under its old
images, no classifier gets a single lion test view right (7 objects), and
the four-network average fraction of objects with *any* correct view is
0.11 for vending machines (20 unseen objects). This can reflect image domain,
category mapping, or annotation quality. It is a reason to measure the same
ceiling again on Blender images before changing the VQF predictor.

**Acceptance for the full run:** compare the 400 seen and 200 unseen objects
with the plotted Fig. 4 values using the fixed nine-step protocol. Each
step-9 quality channel should be within 0.10 absolute of the plot reading;
each classifier's step-9 accuracy should also be within 0.10, the mean
accuracy gain from step 0 should be at least 0.15, and ViewActive should beat
`random_walk` and `stay` with an object-bootstrap interval excluding zero.
The plot readings are approximate, so these are reproduction tolerances rather
than a claim that the unpublished experiment is exactly recoverable. Changes
to image framing, targets, or optimizer settings should be selected on the
validation objects and checked once on the held-out test objects.

## Implementation update

- The mesh loader now bakes each GLB scene part before concatenation. It keeps
  `COLOR_0`, broadcasts one material RGBA value across that part's vertices,
  and uses `baseColorFactor` when Trimesh cannot convert a material. A pilot
  bicycle that previously rendered nearly uniform now loads with five distinct
  part colors. Existing render caches still contain the old colors.
- Classification accepts `classification.crop_margin` (`null` by default).
  Set it to `0.15` for a square white-padded crop around each modal mask before
  the timm transform. This changes classifier evaluation only. Use a fresh
  output directory or regenerate classifier outputs when changing the setting.
- The full-frame versus cropped classifier experiment completed on the pilot
  (its code was removed in the 2026-10-22 cleanup; it is in git history).
- The paper-scale run finished on 2026-10-09. It includes 1,800 objects,
  237,600 renders, all four classifiers, 60 training epochs, and evaluation.
  Its evaluation was regenerated on CPU after the fix for undefined quality
  cells described below. The CPU suite passes 83 tests (18 skipped in the host
  environment).
- New cohorts now allocate the published 70/5/25 split globally. For 1,600
  seen objects this gives exactly 1,120 train, 80 validation and 400 test.
  The existing full run used 1,154/56/390 because of per-class rounding and
  must be retrained in a new output directory to use the corrected split.

## Why the current accuracy is below the paper

The largest known difference is the recognition setup, not just the VQF
predictor. In the current full run, even steering with the *ground-truth*
quality field reaches only 0.225 mean seen-class accuracy and 0.279 unseen,
compared with ViewActive's 0.190 and 0.235. This means that fixing VQF
prediction alone cannot close the observed recognition gap.

| Component | Published method | Current full run | Status |
|---|---|---|---|
| Seen-object split | 70/5/25 of 1,600 | 1,154/56/390 | Corrected for new cohorts; full run needs retraining |
| RGB renders | Blender, 1920×1680 | PyTorch3D, 320×320 | Different image domain and framing |
| Geometry sampling | 10 points per face | 20,000 points per mesh | Different VQF label estimator |
| Objaverse objects | Authors' 1,600 selected UIDs | Our own 1,600 selected UIDs | Exact IDs unavailable |
| VQF target scaling | Not specified precisely | Per-object min–max to [0.05, 1] | Unverified choice |
| Encoder/optimizer family | DINOv2 + CLIP, three MLP heads, AdamW 0.001, decay 0.9, stated loss weights | Matches these published components | Exact backbone/checkpoints, batch and schedule details unavailable |
| Model selection | Paper mentions cross-validation without fold details | One object-level validation split and best checkpoint over 60 epochs | Different or unverified procedure |
| Recognition models | Four networks; exact weights and preprocessing unspecified | Four selected timm checkpoints, full-frame transform, hand-built LVIS→ImageNet mapping | Unverified scoring protocol |

The median object in our test renders occupies 4.6% of a 320×320 frame
(median mask bbox 92×99 px). Whether the authors cropped or framed objects
tightly is unknown. Our upcoming crop experiment isolates this factor; a
large improvement would support it as a cause, but it would not establish a
one-to-one match to their preprocessing.

The official paper documents the 1,600/200 objects, image resolution,
10-points-per-face sampling, 70/5/25 split and optimizer settings in
[Section IV-A](https://arxiv.org/html/2409.09997). The
[public repository](https://github.com/jiayi-wu-umd/ViewActive) currently has
only a README. Exact numerical replication requires the authors' dataset and
implementation details, or explicit assumptions where those details are absent.

## The paper, briefly

- **Model:** from a single image, the model predicts a 3D viewpoint quality
  field (VQF) over 132 views on a 12-azimuth × 11-elevation grid.
- **Quality scores:** each view has three, the paper's Eqs. 1–3: visible
  ratio (1 − self-occlusion), occupancy-aware surface normal entropy, and
  visual entropy.
- **Architecture:** frozen DINOv2 and CLIP features are concatenated and fed
  to three small MLP decoders, one per score, about 0.5M parameters each.
- **Loss:** 0.3·L1 + 0.4·DSSIM + 0.3·SILog (λ = 0.85). AdamW, lr 1e-3,
  exponential decay of 0.9.
- **Algorithm 1:** move to the predicted best view if it can be reached in one
  step, otherwise to the reachable view nearest to it. Re-predict from the new
  image and repeat for 9 rounds.
- **Results:** the paper has no tables. Its only quantitative result is Fig. 4:
  the three scores and the top-1 accuracy of four ImageNet classifiers
  (ResNet50, EfficientNetV2, Swin, ViT) over steps 0–9, for seen and unseen
  classes. Values read off the plot are in the planning notes below.

## What has been built

**Package `viewactive/`.** The stages are
`select → download → canonicalize → cohort → render → label → features → classify → train → evaluate`.
Every stage resumes after an interruption.

| File | Responsibility |
|---|---|
| `grid.py` | 12×11 grid; relative-azimuth / absolute-elevation frame; reachability; SAM3D angle mapping |
| `cohort.py` | Choose seen and unseen classes; select objects; stratified 70/5/25 split by object |
| `download.py` | Symlink GLBs already on the NAS and download the rest from HF |
| `labels.py` | Per-view quality fields, reusing `study/ambiguity.py` |
| `features.py` | Cached DINOv2-B + CLIP-B/16 features (1280-d) |
| `model.py` | Decoders, network, and the public `ViewActive.load(...).predict(...)` API |
| `losses.py` | L1, DSSIM on the grid, SILog |
| `optimize.py` | Algorithm 1 and the baseline policies |
| `train.py` | Training loop |
| `classify.py` | The four timm ImageNet classifiers |
| `evaluate.py` | Field metrics, Fig. 4 rollouts, plots, FPS |

**Configs:**
- `configs/viewactive_blender.yaml` (full scale, the paper's 1,600 seen + 200 unseen objects)
- `configs/viewactive_blender_el0_45.yaml` (the same with Algorithm 1 targets limited to
  elevations 0-45; its model is the one the pipeline uses,
  `/data/checkpoints/viewactive/blender_full_el0_45.pt`)
- `configs/viewactive_blender_smoke.yaml` (5 classes, a few objects each)
- `configs/viewactive_imagenet_map.yaml` (94 LVIS categories mapped to ImageNet-1k
  indices, checked against `timm.data.ImageNetInfo`)

**Tests:** The host CPU suite passed 83 tests on 2026-10-14 (18 skipped because
the host lacks the container's GPU model dependencies). GPU-dependent tests
still need to run in the `sam3d` environment.

**Changes to existing files:**
- `study/config.py`: ViewActive defaults and validation, as CLAUDE.md requires.
- `harness/canonicalize.py`: an opt-in `--trust-file-up` flag. Default
  behaviour is unchanged.
- `README.md`: a ViewActive section.
- `compose.yaml` and `.env.example`: a new writable NAS mount,
  `VIEWACTIVE_OBJAVERSE_DIR`, which defaults to
  `/ICONAS/users/marfred/objaverse_viewactive` and appears in the container as
  `/data/objaverse_viewactive`. The 829 GLBs downloaded earlier were moved
  there. The dev container was recreated to pick up the mount.

Nothing is committed yet.

## Main design decisions (where the paper is silent)

- **Frame of the quality field.** Azimuth is relative to the current camera
  and elevation is absolute. Objaverse yaw is arbitrary, so a field in the
  object's frame can't be learned from one image.
- **Targets.** Each object's channels are min–max scaled to [0.05, 1]. The
  paper says relative values matter, and its Fig. 4 visible-ratio values are
  higher than raw values allow.
- **Splits.** The split is by object rather than by image, to avoid leaking
  labels between views of one object.
- **Rendering.** Renders reuse the study harness: PyTorch3D, distance 2.2,
  320 px, white background. The paper used Blender at 1920×1680. The benefit
  is that the study's ground-truth images are in-domain for phase 2.
- **Up axis.** The file's own up axis is kept (`objaverse.trust_file_up: true`).
  The automatic up-axis heuristic flagged 30–34% of objects as suspect and was
  unreliable for boxy and round classes.
- **Leakage.** The 110 upright-study objects are excluded from ViewActive
  training.
- **Reach.** Algorithm 1 may move ±1 grid cell per step (±30° azimuth,
  ±15° elevation).
- **Combined score.** The three channels are averaged with equal weights,
  matching the repo's earlier finding that equal weights are not worse.

## Results so far

### Smoke run (14 objects)

Every stage ran end to end and every camera check passed. It ran at 72.3 FPS at
batch 1 on an L40S, which matches the paper's 72 FPS. Its quality numbers mean
nothing: there were only 8 training objects.

### Pilot (150 objects: 72 train, 12 val, 36 seen-test, 30 unseen-test)

The pilot completed. Outputs are in `outputs/viewactive/pilot/eval/`.

**Field prediction vs. the per-elevation average baseline (`mean_prior`):**

| | ViewActive (seen) | Baseline (seen) | ViewActive (unseen) | Baseline (unseen) |
|---|---|---|---|---|
| Spearman ρ, combined field | 0.49 | 0.36 | 0.43 | 0.50 |
| True best view in top 5 | 16% | 8% | 10% | 5% |
| Regret (lower is better) | 0.16 | 0.24 | 0.18 | 0.19 |

- **Seen classes:** ViewActive beats the baseline on every quality channel by
  at least 0.05 ρ, which passes the pilot milestone.
- **Unseen classes:** it does not beat the baseline yet. This is expected with
  72 training objects.

**Fig. 4-style rollouts, seen classes (step 0 → step 9):**

| | ViewActive | Paper | Oracle (`gt_field`) | Random walk |
|---|---|---|---|---|
| Visible ratio | 0.63 → 0.74 | 0.57 → 0.72 | → 0.87 | → 0.64 |
| Normal entropy | 0.51 → 0.59 | 0.49 → 0.65 | → 0.87 | → 0.51 |
| Visual entropy | 0.57 → 0.67 | 0.58 → 0.68 | → 0.85 | → 0.59 |
| Classifier accuracy | ViT 0.38 → 0.44; others flat | ~0.14 → ~0.47 | ViT → 0.53; others +0–0.10 | flat |

- **Quality scores:** the gains are close to the paper's. ViewActive beats the
  random walk on combined quality at step 9, by +0.08 seen (95% CI +0.05 to
  +0.12) and +0.08 unseen. It is not significantly better than steering by the
  baseline field (seen +0.04, CI −0.01 to +0.08). It captures about a third of
  the oracle's gain.
- **Accuracy:** this is where we don't reproduce the paper. The oracle barely
  improves accuracy either, so the bottleneck is the renders or the evaluation
  setup, not the model. Unseen-class accuracy is 0.05–0.30 for every policy.
- **Speed:** 48 FPS here. The GPU was shared during this run.

### Paper-scale run (1,800 objects: 1,154 train, 56 val, 390 seen-test, 200 unseen-test)

The full run completed on 2026-10-09 in `outputs/viewactive/full/`. It used
the corrected part colours, full-frame classification, and 60 epochs. At batch
size 1, the measured speed was 75.1 FPS.

| Field metric | ViewActive seen | Mean prior seen | ViewActive unseen | Mean prior unseen |
|---|---:|---:|---:|---:|
| Combined Spearman ρ | 0.536 | 0.335 | 0.446 | 0.264 |
| True best view in top 5 | 18.7% | 4.9% | 14.7% | 3.0% |
| Regret (lower is better) | 0.147 | 0.218 | 0.180 | 0.245 |

Mean classifier accuracy over the four models rises from **0.151 to 0.190**
for seen classes and **0.191 to 0.235** for unseen classes over nine steps.
The random-walk endpoints are 0.159 and 0.191 respectively; the ViewActive
endpoint advantage has 95% object-bootstrap intervals of +0.016 to +0.045
(seen) and +0.020 to +0.068 (unseen). Thus the classifier gain is real on this
cohort, but remains well below the paper's roughly 0.45–0.49 endpoint accuracy.
Even the ground-truth quality-field oracle reaches only 0.225 seen and 0.279
unseen, pointing to the render/classifier evaluation as the main gap.

Combined ground-truth quality rises from **0.563 to 0.662** seen and **0.566 to
0.646** unseen. At step 9, ViewActive exceeds the random walk by +0.093 seen
(95% CI +0.083 to +0.104) and +0.076 unseen (+0.062 to +0.090). The corrected
seen normal-entropy endpoint is 0.624.

One seen-test object has five undefined normal-entropy labels. The original
rollout aggregation let those cells turn the step-9 seen normal-entropy and
combined-quality summaries into `NaN`. The code now ignores undefined cells
when averaging starts and objects. The full evaluation was regenerated from
cached features on CPU; `fig4.json` and `fields.json` now have no nonfinite
values. The classifier accuracy curves and field metrics were unaffected.

## Investigation in progress: why accuracy doesn't improve

Diagnostic outputs:
- `outputs/viewactive/pilot/eval/color_diag.json` (per-object colour baking)
- `outputs/viewactive/pilot/eval/textured_sheet.png` (renders of textured objects)

**1. Colour fallback ("gray renders"). Real but small.**
`ObjaverseStore._vertex_colors` (`harness/meshio.py:108`) falls back to the flat
`DEFAULT_ALBEDO` gray for 24 of 150 objects (16%). None of these has a texture
image.
- 21 are genuinely single-colour assets (one `baseColorFactor`). The bug
  replaces their real colour with gray; for example, an orange chair and a
  blue shoe become gray.
- The other 3 carry per-vertex colours (`COLOR_0`), which are dropped.

There are two causes:
- `to_color()` returns one RGBA value rather than one per vertex, so the
  length check fails.
- `to_color()` raises `TypeError` when the colour is only in the factor.

The fix is to broadcast the material colour and read `COLOR_0` before falling
back. Expect only a modest effect on accuracy.

**2. Texture detail lost by per-vertex baking. Moderate.**
40 of 91 textured objects have fewer than 5k vertices. Fine detail disappears
(for example, the mushroom's white spots), and some atlases come out noisy
(the race car). Most objects remain recognizable.

**3. Object scale in frame. Likely the biggest factor; not yet tested.**
Objects fill a small part of the 320 px frame, because of unit-diagonal
scaling, distance 2.2 and look-at height 0.35. For example, a bicycle's bbox
is 140×47 px. The classifiers receive the whole frame without cropping to the
object, and timm's transform then centre-crops it further. The paper's Blender
renders were probably framed tightly.

## Use in the pipeline

The `blender_full_el0_45` model is the `vqf` selector of `perception.Verifier`
and of the study's `pick` stage: it predicts the field from the input image and
mask, maps each SAM3D candidate with `grid.sam3d_to_viewactive`, and picks the
best interpolated combined quality within absolute elevations 0-45. Earlier
PyTorch3D runs (`full`, `pilot`, `smoke`, `fidelity_split`) and their configs were
removed in the 2026-10-22 cleanup; they are in git history.

## Operational notes

- An interrupted training run retrains unless `model/history.json` records
  every configured epoch.
- Host memory pressure earlier caused systemd-oomd to kill the dev container
  three times (exit 137). It was fine on the last run. Watch
  `/proc/pressure/memory` on long runs.
- The GPUs are shared. Check `nvidia-smi` before every GPU run.
- `.env` contains an HF token. Don't log it or copy it into configs.

## Planning notes: paper Fig. 4 values (read off the plot)

| Panel | Curve | step 0 | step 4 | step 9 |
|---|---|---|---|---|
| (a) seen, scores | Visible ratio / normal entropy / visual entropy | 0.57 / 0.485 / 0.58 | 0.65 / 0.588 / 0.64 | 0.715 / 0.645 / 0.683 |
| (b) seen, accuracy | ResNet50 / EffNetV2 / Swin / ViT | 0.155 / 0.15 / 0.145 / 0.11 | ~0.52 | 0.475 / 0.45 / 0.487 / 0.465 |
| (c) unseen, scores | Visible ratio / normal entropy / visual entropy | 0.605 / 0.528 / 0.592 | 0.645 / 0.572 / 0.611 | 0.692 / 0.615 / 0.628 |
| (d) unseen, accuracy | ResNet50 / EffNetV2 / Swin / ViT | 0.095 / 0.095 / 0.115 / 0.05 | 0.40–0.455 | 0.42 / 0.445 / 0.435 / 0.44 |
