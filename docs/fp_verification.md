# False-positive verification with SAM3D views

The verifier starts with an existing open-vocabulary detection and its SAM3D
reconstruction. It renders the object at distinct azimuths, runs the same text
detector on those images, and vetoes claims that do not persist. It cannot add
detections. Rules use scores for the requested prompt and, optionally, other
prompts; no category templates or dataset-specific training are required.

All four rules use the same 24-view grid: eight azimuths at three elevations,
including the reconstructed source view. The first three fuse scores across
that grid. `peak_margin` searches it for one view with target score at least
0.5 and target-versus-rival margin at least 0.2. At least three views are
required; missing scores cause rejection. Nearby elevations are correlated,
so 24 rendered views are not 24 independent observations.

| Rule | Accept when | Extra inputs |
|---|---|---|
| `majority` | More than half of 24 views score query ≥0.5 | Query only |
| `lower_quartile` | The 25th percentile of all 24 query scores is ≥0.5 | Query only |
| `relative_margin` | More than half of 24 views score query ≥0.5 and above its strongest rival | Rival prompts |
| `peak_margin` | One of 24 views scores query ≥0.5 and has query minus strongest rival ≥0.2 | Rival prompts |

The margin rules are sensitive to the supplied comparison vocabulary. Include
plausible confusions for the application; the other rules can verify an arbitrary
single query. The reusable API is `study.fp_verification.verify_claim`.

## Cached-study result

Run `sam3d-env core python -m study.fp_verification --config configs/upright.yaml`.
The command saves `fp_verification.json` and, in `figures/fp_verification/`,
`summary_*.png`, `margins_*.png`, `detection_*.png`, `detection_any_fp_*.png`,
and `f1_*.png`. The `detection_any_fp_*.png` plots pair the fraction of inputs
with at least one wrong-category detection with TP recall; both bars use inputs
as their denominator. The comparison JSON retains every selector and all four
verification rules. With `configs/upright.yaml`, the saved charts show only
identity, random, ambiguity, detector entropy, the verification rules, and oracle.
The charts use the same input subsets and the same five study metrics: confidence
margin, top-1 accuracy, any false positive, FP rate, and TP recall, plus prompt-level
micro-F1. Rerun the
study's analyze stage first if its summary is missing or its settings changed.
For a fusion rule, a vetoed prompt's original score becomes zero; the standard
study outcome functions then calculate margin and top-1 from that score map,
and FP rate and TP from thresholded detections. Existing selector rows come
directly from `summary.json`, so their values match the earlier plots exactly.
For F1, each input contributes one true prompt and 11 wrong prompts. Across a
subset, micro-F1 is `2 TP / (2 TP + FP + FN)`, where a missing true-prompt
detection is an FN and a detected wrong prompt is an FP. The comparison output
also records prompt precision. This measures category verification over the
study vocabulary, not bounding-box detection F1 or mAP. It depends on which
prompts are included and their frequency in the cohort.
The TN and FP rates share the same denominator: all wrong-category prompts
across the subset. `TN rate = 1 - FP rate`. The JSON comparison also records
TN and FP counts (expected counts for the fractional random selector).
The bar charts display FP rate and TP recall; TN rate stays in the JSON.

The updated all-input comparison (6,600 reconstructed inputs) is:

| Method | FP rate | TP recall | Prompt micro-F1 |
|---|---:|---:|---:|
| Identity | 2.58% | 64.83% | 67.11% |
| Random | 1.98% | 45.39% | 54.29% |
| Detector entropy | 2.71% | 72.55% | 71.69% |
| Relative margin | 0.43% | 39.67% | 54.96% |
| Peak margin | 1.09% | 62.38% | 71.56% |
| Oracle | 0.82% | 76.20% | 82.29% |

On the available 110-object Objaverse cohort, 6,600 input views have complete
SAM3D reconstructions and detector scores. The table counts original prompt-level
detections above 0.5. True positives also require original-mask IoU at least 0.5;
wrong-category prompts are false positives. The metrics describe semantic label
verification, not localization of wrong-prompt masks.
The claim-level table below answers a different question from the side-by-side
study plots: it counts originally fired prompts retained or vetoed. Use the
plots for direct comparison against the previous viewpoint selectors.

| Rule | True positives kept | False positives kept | FP rejection | TP retention | Precision |
|---|---:|---:|---:|---:|---:|
| Original detector | 4,789 | 2,337 | 0.0% | 100.0% | 67.2% |
| Reconstructed identity view | 4,044 | 1,423 | 39.1% | 84.4% | 74.0% |
| Label-aware oracle view | 4,387 | 487 | 79.2% | 91.6% | 90.0% |
| `majority` | 2,861 | 677 | 71.0% | 59.7% | 80.9% |
| `lower_quartile` | 2,028 | 371 | 84.1% | 42.3% | 84.5% |
| `relative_margin` | 2,618 | 309 | 86.8% | 54.7% | 89.4% |
| `peak_margin` | 4,117 | 789 | 66.2% | 86.0% | 83.9% |

The oracle uses the known true category to pick the rendered view with the
largest true-versus-wrong score margin for each input. It is a label-aware view
selection reference, not a deployable filter. Relative to identity,
`peak_margin` improves FP rejection by 27.1 percentage points (95% object
bootstrap interval: +18.5 to +35.2 points). Its TP retention is 1.5 points
higher, with interval −2.4 to +5.2 points. The positive FP gain is clear on
this cohort; the small TP gain is uncertain.

The 0.2 peak margin was selected after inspecting this cohort, so its reported
performance is exploratory. This is an in-cohort result, not an estimate of
performance on other datasets.
The views are generated from one source image, so shared SAM3D errors can cause
consistent mistakes. The same SAM3 model is used before and after reconstruction,
so its errors can also correlate. Test on held-out real scenes and other object
vocabularies before choosing a deployment threshold. The current study uses only
synthetic Objaverse views, and its precision depends on the prompts and prevalence
of false positives in this cohort.

## Research basis

This follows the idea of aggregating recognition evidence from rendered 3D views,
as in [PointCLIP](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_PointCLIP_Point_Cloud_Understanding_by_CLIP_CVPR_2022_paper.html),
while using [SAM 3D Objects](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_SAM_3D_3Dfy_Anything_in_Images_CVPR_2026_paper.html)
to reconstruct the candidate from a single image. Multi-view semantic association
also appears in [OpenM3D](https://openaccess.thecvf.com/content/ICCV2025/html/Hsu_OpenM3D_Open_Vocabulary_Multi-view_Indoor_3D_Object_Detection_without_Human_ICCV_2025_paper.html).
These papers motivate the approach; the four rules and the numbers above are
specific to this repository.
