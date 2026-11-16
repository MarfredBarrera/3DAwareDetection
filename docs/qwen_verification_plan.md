# Three-way Qwen verification pilot

## Intended behavior

Use uncertainty to decide whether a detection needs verification. High uncertainty
means low absolute target margin, small mask area, or frequent SAM3 label flips
across SAM3D views. A frozen gate sends flagged requests to Qwen, which judges the
outlined camera image together with the reconstruction grid. Its verdict is
`correct`, `wrong`, or `insufficient`. Insufficient evidence requests a new real
observation; it never silently becomes accept or reject, including at STOP or
when a retry budget is exhausted.

## Implementation scope

Put reusable policy, prompt, image preparation, and Qwen inference in
`perception/qwen.py`. Keep the existing binary HTTP protocol unchanged: it cannot
safely represent insufficient information. Implement a reproducible experiment
in `scripts/qwen_verification_study.py`, with preparation, reconstruction,
inference, and analysis stages that resume into a new output directory. Preserve
the old binary experiment as a historical baseline. Unit tests cover gating,
missing features, ties, lazy Qwen invocation, three-way decisions, and bounded
navigation actions. Real GPU inference validates the prompt/model path.

## Frozen protocol before seeing new Qwen results

1. Use all 131 offline rows in
   `outputs/apexnav_offline/sufficiency_2026-10-28/requests.csv` only to calibrate
   empirical feature distributions and a top-third review threshold. Use
   midranks for ties. Fix a separate two-feature threshold for missing flip rate;
   missing flip is not zero. Calibration uses no correctness labels.
2. Remove online rows whose underlying episode number occurs in calibration,
   irrespective of strategy prefixes. Include every remaining online request
   and strategy (no subsampling); freeze and hash the manifest before inference.
   This yields more useful episode groups for uncertainty intervals than a
   40-episode draw. Episode grouping must also be used for bootstrap intervals;
   strategy-prefixed episodes are not independent. This is episode-disjoint,
   not necessarily scene-disjoint.
3. Primary gate uses the logged margin, mask area, and view flip. Freeze these
   features for replay. All online grids are missing on disk: reconstruct only
   flagged requests using the existing SAM3D worker, one idle GPU, one job at a
   time. Save tiled grids and timing, not large redundant assets. A regenerated
   grid may differ from the one that supplied the logged flip rate; explicitly
   measure/report that limitation if scores are rerun. Do not substitute a
   single selected view for the full grid.
4. Qwen3-VL-8B-Instruct is cached in the development container, with Python at
   `/data/envs/qwen/bin/python`. Use the same fixed three-choice prompt for all
   arms. A/B/C mean correct/wrong/insufficient. Validate single-token answer IDs,
   retain all three scores, distinguish protocol failures from semantic
   insufficiency, and do not interpret softmax scores as calibrated confidence.
5. Run input-only on all selected requests (a diagnostic control). On flagged
   requests also run input + matching grid, and input + a deterministic grid
   from a different episode (diagnostic mismatch control). Log actual Qwen
   time, token count, model/prompt versions, and errors. In the primary deployed
   policy Qwen is called only when the gate fires; extra calls are study controls.

## Evaluation

Join the existing eye-label files with review overrides, retaining `cant_tell`
as a third label. These labels were produced by agents and reviewed partly by
an agent; they are provisional, not independently validated human ground truth.
Do not relabel `cant_tell` using HM3D semantic masks. Report disagreement with
HM3D separately only if needed.

Compare SAM3 raw, gate -> defer, gate -> Qwen input, gate -> Qwen grid, and Qwen
input on all requests. Main outputs: three-way confusion counts; known false
accepts and false rejects; decided coverage; risk among decided clear cases;
cant_tell deferral recall; unsupported accepts/rejects on cant_tell; total review
and deferral rates; raw errors repaired and introduced; accept-side and
reject-side precision on clear cases. Report counts alongside rates, including
zero-denominator cases as undefined. Episode-bootstrap 95% intervals and paired
deltas use 2,000 draws with a fixed seed. Report the at-STOP subset separately.

For navigation relevance, report episodes with a known false accept, episodes
with only unresolved checks, and bounded per-object sequential replay where
logged later observations genuinely exist. Require same strategy/episode and
object association, increasing evidence step, and a different real image.
Report how often a deferral has a later recorded view and how it resolves.
Missing future views are censored; never fabricate recovery or success/SPL.
Observed future views are a diagnostic, not counterfactual policy trajectories.

Report Qwen latency separately from regeneration and from the cost of computing
the uncertainty gate. View flip requires SAM3D plus SAM3 on its views even for
requests that bypass Qwen; this pilot does not demonstrate reconstruction savings.

## Interpretation and deliverables

Produce a readable report, manifest, frozen gate, per-request decisions, summary
JSON, and reproducible commands in a new output directory. Retain evidence and
failure examples needed to judge whether the grid improves over input-only and
whether deferrals prevent errors without eliminating useful decisions.

This is a screening study for a later paired synchronous closed-loop experiment,
not a closed-loop success-rate claim. If the pilot is promising, the next test
must add explicit three-way planner handling with new-view acquisition, bounded
retries, STOP blocking for insufficient evidence, and cache rules, then compare
success/SPL/false stops/action counts on paired episodes. Do not install this
policy into the binary ApexNav gate merely by treating None as False.

The available online rows were also included in the earlier sufficiency study
that motivated the margin/area/view-flip feature choice. This is retrospective
development screening, not an untouched confirmatory test. The frozen
calibration and inference cohorts are episode-disjoint, but may overlap in
scene; do not claim strict held-out generalization.
