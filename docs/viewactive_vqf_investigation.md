# ViewActive VQF investigation — 2026-10-17

The current exact VQF improves recognition, but optimizing it does not reliably
select the most recognizable view. There are three separate issues: the quality
metrics are imperfect proxies for recognition; our label estimator differs from
the published procedure; and our objects, rendering, and recognition evaluation
are not the authors' released experiment. The public repository still contains
only a README, so some fidelity questions cannot be resolved from public code.

Sources: [paper, version 5](https://arxiv.org/html/2409.09997v5),
[author repository](https://github.com/jiayi-wu-umd/ViewActive).

## What was measured

CPU diagnostics use the archived Blender run's locally cached `vqf.npz` and
all four classifiers' predictions. UIDs are explicitly joined. No predictor
training or render regeneration is involved. The audit includes all 400 seen
test objects and 200 unseen objects. For navigation, every one of the 132 start
views is evaluated, eliminating random-start sampling noise.

| Selection, seen test objects | ResNet50 | ViT |
|---|---:|---:|
| Uniform random view, exact expectation | 20.21% | 32.21% |
| Nine moves using the exact VQF | 28.76% | 45.56% |
| Direct selection of the global VQF maximum | 28.75% | 45.50% |
| Object has at least one correctly classified view | 67.75% | 80.75% |

The last row is a classifier-specific hindsight bound, not a deployable policy.
The exact-field policy reaches its chosen global maximum from 98.27% of starts
after nine moves, and 100% after 100 moves. Thus limited reach or local minima
do not explain the exact-field recognition gap. Unseen objects show the same
pattern: direct maximum accuracy is 30.0% / 39.5%, versus any-correct-view
bounds of 73.0% / 81.5%.

Within-object Spearman correlation between combined VQF and binary correctness
is 0.186 for ResNet50 and 0.223 for ViT on seen objects. These statistics exclude
objects whose correctness never changes across views; the JSON records the
number of eligible objects. A low correlation does not mean zero benefit:
exact VQF improves the average by 8.5 and 13.3 percentage points respectively.

The approximate paper endpoint quoted in the handoff is 47.5% / 46.5%. Our
exact VQF is already near that ViT endpoint, but starts from 32.2% instead of
the paper's approximately 11%. ResNet50 remains substantially below its paper
endpoint. Both the endpoint and gain must be compared, with classifier/domain
differences acknowledged.

## Why a perfect score need not identify a canonical view

“Perfect VQF” here means access to the exact **implemented label field**. It
does not establish that those labels reproduce the authors' estimator, or that
their maximum optimizes recognition.

Our visible-area measure rewards exposed surface area regardless of which
parts are distinctive. A roof, underside, pedestal, or irrelevant scene part
can contribute more area than a face or other identifying detail. Normal
entropy rewards diversity of visible normal directions. The grayscale
histogram ignores spatial arrangement: permuting all foreground pixels leaves
its entropy exactly unchanged, even if the image becomes unrecognizable.
None of these definitions encodes the semantic importance of a particular
part, an upright photographic convention, or the classifier's feature needs.

On seen objects, ViT accuracy averages 40.6% at elevation 0 and 41.7% at +15,
but 19.6% at -75 and 21.1% at +75. In contrast, mean normalized visible ratio
is lowest around zero; visual entropy peaks below the object. The combined
global maximum has absolute elevation at least 60 degrees for 25.5% of seen
objects and 20.5% of unseen objects. This measures a concrete disagreement
between score and recognizability, without proposing an elevation prior.

The paper reports aggregate improvements and qualitative examples; these do
not establish that every exact-field maximum is canonical. Nor is `gt_field`
a mathematical upper bound on a learned policy's recognition accuracy. A
smoothed or imperfect prediction could accidentally choose a more recognizable
view with lower geometric quality.

## Label fidelity: a real difference and a rejected hypothesis

**The occupancy multiplier is already implemented.**
`study/ambiguity.py:175` bins camera-frame normals over polar [0, pi/2] and
azimuth [-pi, pi]; lines 184–186 multiply entropy by occupied bins / total bins.
Those bins cover the camera-facing hemisphere. The handoff's suggestion that
the implementation used a full-sphere denominator or omitted occupancy is
incorrect. The subsequent division by `log2(162)` is constant and cancels
under per-object min–max normalization, so removing it alone cannot change
the selected normalized maximum (apart from numerical ties).

**Surface sampling and normal weighting differ.** The published implementation
description uses ten samples per face and normals of fully visible faces.
`study/ambiguity.py:102` instead draws 20,000 samples per mesh with probability
proportional to triangle area; line 162 feeds every visible sample's normal
into the histogram. Consequently, our histogram weights normals by visible
area, includes partially visible faces, and can omit small triangles entirely.
That is more than a change in the total number of rays.

The diagnostic relabeling uses 50 seen test objects sampled without selection
by outcome (seed 20261017), all 132 views, ten uniformly sampled points per
nondegenerate triangle. It retains two-sided visibility and the existing
camera/material assumptions to isolate sampling. Four variants are compared:

1. Dense sampling with area-weighted visible-sample normals, approximating the
   existing estimator with a different sampling allocation.
2. Fractional visible area plus one normal per fully visible face.
3. Fully visible face area plus one normal per fully visible face.
4. The original 20,000-point estimator resampled with seed 1 as a stability control.

Variants 2 and 3 bracket an ambiguity in handling partially occluded surface
area; neither is claimed to recover unpublished author code. Cached visual
entropy is held fixed. Object-bootstrap intervals accompany recognition
differences. These are diagnostic test-set comparisons, not tuning results.

The completed 50-object comparison gives:

| Estimator; direct combined-score maximum | ResNet50 | ViT | Best view changed |
|---|---:|---:|---:|
| Existing cached labels | 32% | 46% | — |
| Ten/face, dense area-weighted control | 28% | 48% | 44% |
| Ten/face, fully visible face normals, fractional area | 30% | 46% | 64% |
| Ten/face, fully visible face normals and area | 32% | 44% | 64% |
| Existing estimator, new sample seed | 30% | 44% | 38% |

For fully visible face normals with fractional area, paired 95% bootstrap
intervals for the accuracy change are [-8, +4] percentage points for ResNet50
and [-10, +10] for ViT. The normal-channel mean absolute difference is 0.120
on its globally scaled [0,1] range, versus 0.009 for a new sampling seed.
The dense area-weighted control changes normal entropy by 0.053 on average;
occupancy is sensitive to whether rare normal bins receive any samples.

This confirms that the estimator changes the labels substantially, but does
**not** demonstrate a recognition gain or explain the large paper gap. The
fully visible face variants reduce extreme-elevation selections from 40% to
32% on this subset without raising ViT accuracy. Even a new random sampling
seed changes 38% of exact maximizing cells while retaining a 0.982 combined
rank correlation, so the identity of a single maximizing view is less stable
than the field's overall ranking. Fifty objects leave wide uncertainty;
do not interpret the null result as proof that sampling never matters.

The diagnostic mesh transformation uses float64 before ray casting, whereas
the production loader transforms in float32. A seed-0 replay on a mask,
sunglasses, and a low-poly table measured mean absolute normal-channel
differences of 0.00002, 0.00134, and 0.00022 respectively; visible-area
differences were below 0.00006. Small numerical/bin-boundary effects therefore
remain in this audit and should not be attributed solely to sampling.

## Normalization and combination matter, but are not the entire gap

`viewactive/labels.py:51` independently stretches each object's three channels
to [0.05, 1]. `viewactive/grid.py:54` then averages them. Up to a constant
offset, this is a weighted sum of raw channels with weights proportional to
the inverse of each channel's within-object range. A nearly flat or noisy
channel can therefore receive as much decision weight as a strongly varying
one. Neither this min–max rule nor the scalar combination is specified in
the paper.

Using the cached globally scaled channels directly, with the same equal mean,
changes global-maximum accuracy to 30.0% / 47.75% seen and 31.0% / 44.0%
unseen (ResNet50 / ViT). These are modest, descriptive improvements; they do
not identify the authors' normalization or justify selecting new evaluation
settings on the test set. Individual channels perform worse than the combined
normalized field on seen ViT: visible area 38.75%, normal entropy 42.0%, visual
entropy 36.25%, combined 45.5%.

## Rendering, object preparation, and evaluation

**Camera centering differs from literal object centering.**
`harness/canonicalize.py:146` scales to a unit bounding-box diagonal and puts
the object's base at y=0. `viewactive/blender_render.py:115` always targets
(0, 0.35, 0). The bounding-box center is at half the object's height. Median
vertical target offset is 0.094 units for seen objects; 48% exceed 0.1 of the
unit diagonal. For the purple car in the existing visualization, the actual
center is y=0.139, giving an offset of 0.211. This affects framing and the
meaning of nominal elevation. Correcting it requires corresponding renders
and labels; relabeling against the existing images alone would introduce a
new mismatch.

**The geometric coordinate transforms pass a limited audit.** I imported the
12 previously visualized objects through the same Blender rest-pose and
canonical transform, evaluated the visible mesh geometry, and compared its
bounds with the label mesh's canonical extents. All agree within 5e-7 units.
The check excludes Blender's non-rendered armature helper meshes. It supports
coordinate consistency for those examples, not material or pose equivalence
for every object.

**Material visibility is not represented in geometric labels.** Open3D rays
treat every triangle as opaque; Blender can render alpha-masked or transparent
materials. The visualized wolf contains MASK and BLEND materials. Thus a
correct geometric visibility estimate may still disagree with actual RGB
visibility. The barely visible wolf/rabbit in the existing sheet should not
automatically be diagnosed as a small bounding box: their geometric extents
are comparable to the other normalized objects. Their appearance/materials
need inspection.

ViT gets none of the 132 views right for that wolf, that rabbit, the car UID
`4cc4f9f4c0254284a2d043d593265863`, or dresser UID
`1675f0d7c82f4ab59dc3b0f123e5737e`. Their failed rollout images therefore do
not isolate a viewpoint-selection failure. The existing input/output sheet
also depicts the **learned** policy, not the exact-field policy.

**Recognition and canonical appearance are separate outcomes.** Some objects
are not recognized from any of 132 views: 32.25% by ResNet50 and 19.25% by ViT
on the seen test set. For ViT, only 1/7 owl objects and 2/7 lion objects have
any correct view. This points to image domain, object/annotation quality,
and category mapping as additional constraints; it does not prove which
one is responsible. Our LVIS-to-ImageNet mapping, timm checkpoints, and
preprocessing are assumptions. The authors' UIDs and corresponding settings
are unavailable in their public repository.

## Next reproduction steps

1. Center the camera sphere on each actual object and audit asset preparation,
   material visibility, and object size after classifier preprocessing. Use a
   small paired rendering experiment to quantify the effect. Both labels and
   images must use the same cleaned geometry and camera.
2. Audit category mapping and the objects that no classifier view recognizes.
   Keep those diagnoses separate from the ability to choose a good view.
3. Establish labels from the documented face-sampling procedure, while
   recording ambiguous implementation choices. Validate with exact-field
   selection before spending time on model training. The completed ablation
   shows that this fidelity correction alone is not a demonstrated solution
   to recognition.
4. Resolve unpublished settings with author artifacts if they become
   available: object IDs, scene setup, normalization/combination, view grid,
   and classifier checkpoints/preprocessing. The present aggregate accuracy
   cannot identify which of these differed in the original experiment.

No added semantic score, fitted channel weights, or elevation prior is needed
to carry out these fidelity checks. Once the exact-field benchmark is
understood, revisit the predictor's overfitting and field-estimation error.

## Artifacts and reproduction

- `outputs/viewactive/vqf_audit/cached_audit.json`: all-cohort measurements.
- `outputs/viewactive/vqf_audit/vqf_diagnosis.png` and `.pdf`: comparison plot.
- `outputs/viewactive/vqf_audit/geometry_audit.json`: Blender bounds audit.
- `outputs/viewactive/vqf_audit/sampling/`: per-object diagnostic labels.
- `outputs/viewactive/vqf_audit/sampling_audit.json`: sampling comparison.

Run these with the existing `sam3d` environment in the dev container:

```sh
python -m scripts.diagnose_viewactive_vqf
python -m scripts.audit_viewactive_geometry
python -m scripts.audit_viewactive_sampling
python -m scripts.plot_viewactive_vqf_audit
```

The Blender portion uses `scripts/audit_viewactive_geometry.py` with
`blender --background --factory-startup --python`. Canonical and download
metadata are copied from the NAS into the audit directory. Production pipeline
code, original labels, model checkpoints, and archived renders are untouched.
