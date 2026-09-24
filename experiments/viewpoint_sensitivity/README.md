# SAM3 viewpoint sensitivity

This experiment examines how viewpoint affects SAM3 recognition for each object.
The viewing grid follows [Fan et al., *Active Open-Vocabulary Recognition*](https://arxiv.org/abs/2311.17938),
but object orientations are not aligned consistently enough to average scores
at a shared category azimuth. Each object therefore gets its own heatmap.

SAM3 is a prompted detector. For each rendered image, we query every configured
category and use its maximum candidate confidence as the category score. The
plotted value, `score_minus_true_category`, is the highest incorrect category
score minus the true category score. Positive values mean an incorrect category
outscored the true one; larger positive values indicate more misleading views.
Invisible targets appear as missing cells. A zero score is a tie.

## Run

From inside the project container:

```bash
# Small end-to-end check (4 x 3, one object per category)
sam3d-env core python -m experiments.viewpoint_sensitivity.run \
  --config experiments/viewpoint_sensitivity/config.yaml

# Paper-style 12 x 12 experiment (10 objects per category)
sam3d-env core python -m experiments.viewpoint_sensitivity.run \
  --config experiments/viewpoint_sensitivity/config.full.yaml
```

Stages can be resumed independently with `--stage render`, `--stage query`, or
`--stage analyze`. `--skip-render` reuses an existing render directory. The
query stage commits each view to `experiment.sqlite`, so an interrupted run
resumes from the last committed view. Pass `--restart-query` only when you intentionally want to discard that
progress, such as after changing the detection thresholds or vocabulary.

The query worker defaults `NUMPY_MADVISE_HUGEPAGE=0` before importing NumPy.
Repeated large mask allocations with huge-page advice caused severe memory
stalls on the shared host: a reproduction reached 72% cgroup memory pressure
with only about 1.5 GiB resident, then the entire container exited with 137.
The host's systemd-oomd monitors the parent user service at 50% pressure;
Docker reported `OOMKilled=false` and kernel OOM counters remained zero.
The daemon's kill log was not accessible, so attribution to systemd-oomd is
inferred from this evidence. Disabling NumPy huge-page advice eliminated the
reproduced pressure spike without changing scores, thresholds, or host policy.
See [NumPy's allocation setting](https://numpy.org/doc/2.0/reference/global_state.html).
An explicit environment value overrides the worker default.

The full configuration uses 12 azimuth bins at 30-degree spacing and 12 unique
latitude-bin centers at 15-degree spacing.  A literal 12 x 12 spherical grid at
30-degree spacing would repeat camera positions; the midpoint latitude grid
also avoids the singular look-at direction at +/-90 degrees.

## Outputs

Start with [the consolidated results index](../../outputs/RESULTS.md) for run status,
summary metrics, and links to the underlying data. Each run keeps the files needed
by the pipeline at its root: `settings.json`, `experiment.sqlite`, `render/`,
`plots/`, and `sam3d_inputs/`. Batch status, camera
verification, and stage logs live in `run/`. The render-only extreme run has no
recognition or analysis files.

### Upright verification and realistic views

The reviewed experiment uses `config.upright.yaml`. It keeps the original
dataset intact and writes to `outputs/viewpoint-sensitivity-upright/`.
`upright_reviews.json` records explicit up vectors, optional front vectors,
visual evidence, and exclusions for the original 120 example objects. Vectors
are expressed in the **original canonical frame**. Corrections are proper
3D rotations, followed by recentering and placing the base on y=0; RGB pixels
are never rotated after rendering. The original height-ratio heuristic can
choose a horizontal axis for beds/desks, which is why review is necessary.

The verification workflow is reproducible inside the container:

```bash
# Four original views per asset, then six candidate up-direction renders.
sam3d-env core python -m experiments.viewpoint_sensitivity.upright sheets \
  --render-dir outputs/viewpoint-sensitivity-full/render \
  --out outputs/viewpoint-sensitivity-upright/audit-before
CUDA_VISIBLE_DEVICES=3 sam3d-env sam3d python -m experiments.viewpoint_sensitivity.upright candidates \
  --source /data/objaverse_nas/canonical.json \
  --index outputs/viewpoint-sensitivity-full/render/index.jsonl \
  --out outputs/viewpoint-sensitivity-upright/candidates
# After visually reviewing the candidates, apply the saved review decisions.
sam3d-env sam3d python -m experiments.viewpoint_sensitivity.upright apply \
  --source /data/objaverse_nas/canonical.json \
  --index outputs/viewpoint-sensitivity-full/render/index.jsonl \
  --reviews experiments/viewpoint_sensitivity/upright_reviews.json \
  --out outputs/viewpoint-sensitivity-upright/canonical.json
sam3d-env core python -m experiments.viewpoint_sensitivity.batch \
  --config experiments/viewpoint_sensitivity/config.upright.yaml
```

From the host, use `docker compose exec -d dev sam3d-env core python -m
experiments.viewpoint_sensitivity.batch --config
experiments/viewpoint_sensitivity/config.upright.yaml` for a detached server
job. `run/batch-status.json` and `run/render.log`, `run/verify.log`,
`run/query.log`, `run/analyze.log` show progress. Re-run the same command to resume after a failed
stage; completed stages are skipped and query progress is resumed. A failed
render restarts rendering. Concurrent drivers are locked out, and changes to
the config or canonical file require a new output directory.

The realistic grid has 12 azimuths and elevations `[0, 15, 30, 45, 60]`.
The optional `config.upright-extreme.yaml` has elevations `[-60, -30, 75, 82.5]`
and a separate output directory. Use `batch --render-only` with that config
to produce stress-test RGBs without recognition scoring. Uncommon camera
directions and low recognition accuracy are different attributes.

This is **review-assisted verification**, not an automatic upright classifier.
New objects need their own review. Ambiguous/mislabeled/fragmented assets are
excluded explicitly. The retained cohort is consequently unequal by category;
`settings.json.category_object_counts` supplies the actual denominators.
The current review retains 110 of 120 objects, corrects 27 up directions,
and verifies semantic front for 33 objects. The realistic grid has 6,600 views;
the extreme grid has 5,280. `orientation-report.json` lists counts and exclusions.
For reviewed fronts, azimuth zero views the front (+Z); symmetric or uncertain
fronts retain a geometric azimuth and are marked `semantic_front_verified=false`.
Camera-matrix verification checks spherical positions and absence of roll,
but cannot establish semantic uprightness by itself. Contact sheets provide
the visual evidence.

- `experiment.sqlite`: indexed recognition trials and compressed analysis
  artifacts. The primary key is object ID plus azimuth and elevation, so a
  resumed query cannot silently duplicate a view. The `trials` table exposes
  category, viewpoint, correctness, and score-minus-true as queryable columns;
  each complete diagnostic row is stored as a compressed payload. The
  `artifacts` table holds `aggregate.json` and `aggregate.npz` content for
  export, without duplicate loose files. Run
  `python -m experiments.viewpoint_sensitivity.store export OUTPUT_DIR --target EXPORT_DIR`
  to recreate the previous formats for external tools.
- `sam3d_inputs/<category>/<object_id>/{high_accuracy,low_accuracy}/viewpoint_*/`:
  a human-readable hierarchy containing one untouched `rgb.png` and one
  self-contained `metadata.json` per selected viewpoint. The metadata groups
  the pose, object ID, true-category score, strongest incorrect category and
  score, top-1 result, aggregate score-minus-true selection value, top-1
  accuracy, and all category scores.
  `manifest.jsonl` provides a flat machine-readable index, while
  `selection.json` records the ranked viewpoints. Set
  `sam3d_collection.viewpoints_per_band` in the YAML (default 3). High and low
  selections are always disjoint.
- `sam3d_inputs/{high_accuracy,low_accuracy}_confusions.jsonl`: metadata kept
  separately from the untouched RGB files. For every selected input it records the
  highest-scoring non-ground-truth category and score, the true-category score,
  their signed difference, and whether that category actually won top-1.
- `plots/per_object/score_minus_true_category_*.png`: one heatmap per object,
  paired with that same object's angle-aligned RGB views. Every plot uses the
  same zero-centered color scale, so colors are comparable across objects.

## What TP, TN, FP, and FN mean

These labels do **not** define the per-object heatmap.  They evaluate SAM3 as a
binary prompted detector at the configured score and mask-IoU thresholds:

- TP: the true-category prompt has at least one candidate above both thresholds.
- FN: the object is visible, but the true prompt has no candidate satisfying
  both thresholds.  A confident mask in the wrong location is therefore an FN.
- FP: a non-true vocabulary prompt has a candidate above the score threshold.
- TN: a non-true vocabulary prompt has no candidate above the score threshold.
- SKIP: the rendered target has zero visible pixels; it contributes to neither
  recognition accuracy nor binary rates.

Each visible image contributes one TP-or-FN trial and `vocabulary_size - 1`
FP-or-TN trials.  Consequently, raw binary counts are prompt-level counts and
must not be read as a multiclass confusion matrix.

Azimuth is defined by each object's geometric/PCA canonical frame. The
per-object heatmaps show viewpoint changes within that frame. Their azimuths
are not semantic directions shared across objects.
