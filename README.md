# 3D-aware viewpoint study

This repository produces azimuth × elevation grids of object views and heatmaps of
SAM3 confidence margin. For each rendered ground-truth view, it also reconstructs
the object with SAM3D, renders a grid of candidate views, and scores their ambiguity
using visual entropy, self-occlusion, and surface-normal entropy. The plots show
whether choosing a less ambiguous reconstructed view improves recognition.

**Confidence margin** = SAM3 score for the true category minus its highest score
for a wrong category. Negative values mean a misleading view.

## Run

The GPU pipeline uses two model environments in the Docker image. Set
`OBJAVERSE_DIR` in `.env` (see `.env.example`), then:

```bash
cp .env.example .env
docker compose build
export HF_TOKEN=...                         # approved SAM3 and SAM3D model access
./scripts/download_models.sh
./scripts/check_dependencies.sh
```

For rootless Docker, add `-f compose.yaml -f compose.rootless.yaml` to Compose
commands.

Set `gpu_id` in the YAML to an idle GPU before running. Commands below run in
the container (`docker compose exec dev ...` or `docker compose run --rm dev ...`):

```bash
sam3d-env core python -m study --config configs/smoke.yaml
sam3d-env core python -m study --config configs/upright.yaml
```

The `smoke` config uses a small cohort and grid; `upright` is the reviewed study.
Both need a canonical Objaverse index. Build one with
`sam3d-env sam3d python -m harness.canonicalize --root /data/objaverse_nas --out /outputs/canonical.json`,
then set `dataset.canonical` in the YAML. The reviewed `upright` config uses
`harness/upright_reviews.json`; `python -m harness.upright --help` shows the
review and apply commands. Each stage resumes after interruption. To regenerate
plots from stored scores:

```bash
sam3d-env core python -m study --config configs/upright.yaml --stages analyze
```

Archived ApexNav replays and older output experiments are under
`/ICONAS/users/marfred/3DAwareVerification/`. Keep active `upright` and
`viewactive` outputs local while comparing methods.

For one input frame, put the image in `data/`. `main.py` runs SAM3 to select a
foreground mask, reconstructs with SAM3D, checks the claimed label across novel
views, saves `verification.json`, a turntable and PLY, and opens a viewer on port 7860:

```bash
sam3d-env core python main.py --image data/frame.png --query chair
sam3d-env core python main.py --image data/frame.png --query chair --no-viewer
```

Edit `configs/demo.yaml` for checkpoints, GPU, and viewer settings. Optional
aligned depth and intrinsics can also be supplied there.

`verification.json` contains an `accepted` verdict for each mask. The default
`majority` rule requires the query to score at least 0.5 in more than half of
24 reconstructed views. `lower_quartile` is stricter; `relative_margin` requires
both a score of at least 0.5 and a win over prompts listed in
`verification.rival_prompts` in more than half of the views. `peak_margin`
searches the full 24-view grid for a confident view with a 0.2 margin over its
rivals. See
[FP verification methods and results](docs/fp_verification.md).

To evaluate all four rules against cached study detections without new GPU work:

```bash
sam3d-env core python -m study.fp_verification --config configs/upright.yaml
```

This writes `outputs/upright/fp_verification.json` and side-by-side comparison
plots in `outputs/upright/figures/fp_verification/`. The plots use the same
margin, top-1 accuracy, false-positive, TP, and prompt-level micro-F1 metrics and input subsets as
the existing study analysis.

To inspect one object or one reconstruction's candidate grid:

```bash
sam3d-env core python -m study.figures --config configs/upright.yaml object <uid>
sam3d-env core python -m study.figures --config configs/upright.yaml reconstruction gt/<uid>/a30_e15
```

To compare one object's input viewpoints with an optimizer's chosen
viewpoints, run:

```bash
sam3d-env core python -m study.compare --config configs/upright.yaml --uid <uid> --selector detector_entropy
```

The PNG at `outputs/upright/figures/comparison/<selector>/<uid>.png` has two
columns: the input image grid and its confidence-margin heatmap on the left;
the selected-view image grid and its margin heatmap on the right. Both grids
are indexed by input camera, so corresponding cells align. Swap
`--selector` for `oracle` (requires labels),
`ambiguity:combined`, `detector_entropy`, or another stored selector. To compare an external method such as ViewActive on the
*same study inputs and candidate grid*, use `--selector viewactive --picks
path/to/picks.json`. The JSON format is
`{"viewactive": {"gt/<uid>/a0_e0": "sam3d/<candidate-key>"}}`; each candidate
key must belong to the stated input. The figure uses cached images and
scores, so it needs no GPU.

### ViewActive unseen split benchmark

`configs/unseen_blender.yaml` uses the 200 `test_unseen` objects from the
ViewActive `blender_full` manifest. Its render stage imports completed Blender
views by reference. Each input is reconstructed with SAM3D; every optimizer
selects from that input's same 24 reconstructed candidate views. Rerun stages
as the source render job finishes; the study database resumes completed work.

```bash
sam3d-env core python -m study --config configs/unseen_blender.yaml --stages render
sam3d-env sam3d python -m study --config configs/unseen_blender.yaml --stages reconstruct
sam3d-env sam3d python -m study --config configs/unseen_blender.yaml --stages score --workers 1
sam3d-env core python -m study --config configs/unseen_blender.yaml --stages detect
sam3d-env core python -m study --config configs/unseen_blender.yaml --stages analyze --workers 1 --no-object-figures
sam3d-env core python -m study.fp_verification --config configs/unseen_blender.yaml
sam3d-env core python -m study.compare --config configs/unseen_blender.yaml --all --selectors oracle detector_entropy ambiguity:visual_entropy
```

The final two commands write aggregate FP-rate and TP-recall graphs under
`outputs/unseen_blender/figures/fp_verification/` and per-object four-panel
grids under `outputs/unseen_blender/figures/comparison/<selector>/`. Use
`--selector viewactive --picks picks.json` to draw ViewActive predictions on
the identical reconstruction grid once the ViewActive model is available.
For a bounded GPU pilot, use `--limit 1` on `reconstruct` and `--limit 25` on
`detect`; the latter covers the first reconstructed input and its 24 views.
The full run is large (12,000 reconstructions and 288,000 candidate views).
Use a single free GPU and archive completed outputs to the NAS when finished.

## Outputs

Everything goes under the config's `output_dir`:

- `figures/before/`: confidence-margin, accuracy, and false-positive heatmaps by category.
- `figures/objects/<category>/before/`: per object, its margin heatmap, a bar chart
  of wrong categories whose scores exceed the true category, and its viewpoint
  grid. Bars show mean (true score − wrong score) over the views where that
  category wins. When reconstruction results exist, `after/` shows the identity
  and selected-view margins and chosen views, and `oracle/`, `detector_entropy/`
  and other selector folders show the input and chosen SAM3D views
  above the before and after margins.
- `figures/explore/`: on-demand reconstruction grids with SAM3 margins and entropy scores.
- `figures/after/`: before/after false-positive maps, accuracy and false-positive
  summary bars, and margin charts showing mean margin and change from input.
- `summary.md` and `summary.json`: aggregate before/after results.
- `study.sqlite`: source data for images, SAM3 scores, ambiguity scores, and view choices.
- `renders/gt/` and `renders/sam3d/`: ground-truth and reconstructed viewpoint images.

The stages are `render → reconstruct → score → detect → analyze`. `render` uses
Objaverse meshes and the camera grid in the YAML. `reconstruct` writes SAM3D
view grids and decoded meshes. `score` computes ambiguity per candidate;
`detect` evaluates every image against the full category vocabulary with SAM3.
`analyze` generates the figures. The candidate angles are relative to the input
view; the ground-truth angles use the canonical object frame.

## ViewActive reproduction

`viewactive/` reimplements ViewActive (Wu et al., arXiv:2409.09997; no public
code). The method predicts a 3D Viewpoint Quality Field from one image: the
paper's Eq. (1)-(3) metrics on a 12 azimuth x 11 elevation grid. It is the
learned, single-image counterpart of the `ambiguity` metrics. Run inside the
container, in the `sam3d` environment, after setting `gpu_id`:

```bash
sam3d-env sam3d python -m viewactive --config configs/viewactive_smoke.yaml   # 14 NAS objects
sam3d-env sam3d python -m viewactive --config configs/viewactive_pilot.yaml   # 15 classes x 10
sam3d-env sam3d python -m viewactive --config configs/viewactive.yaml         # paper scale
sam3d-env sam3d python -m viewactive --config configs/viewactive_blender_smoke.yaml
sam3d-env sam3d python -m viewactive --config configs/viewactive_blender.yaml # Blender, 1920x1680
```

The stages are `select → download → canonicalize → cohort → render → label → features → classify → train → evaluate`.
Seen and unseen classes are the ImageNet-mappable LVIS categories in
`configs/viewactive_imagenet_map.yaml`. GLBs not on the NAS are downloaded to
`objaverse.cache_root`. The viewpoint study's objects are excluded. The
PyTorch3D baseline uses the study harness; the Blender run preserves GLB
materials and textures at the paper's image resolution. Install the pinned
portable Blender inside the container with `bash scripts/install_blender.sh`.
Labels reuse
`study/ambiguity.py`, and targets use the relative-azimuth / absolute-elevation
frame from `viewactive/grid.py`. `evaluate` writes the following to
`<output_dir>/eval/`:

- `fields.json`: field accuracy against a mean-field prior.
- `fig4.json`, `fig4_viewactive.png` and `policies.png`: the paper's Fig. 4
  rollouts, with `gt_field`, `mean_prior`, `random_walk` and `stay` controls.
- `heatmaps/`

The predictor API is in `viewactive/model.py`:
`ViewActive.load(checkpoint).predict([image])`. Methods, deviations and results
are in [docs/viewactive.md](docs/viewactive.md).
The 1,800-object paper-scale reimplementation is complete in
`outputs/viewactive/full/`; it is not yet a one-to-one reproduction of the
paper's training or recognition protocol.
The corrected split is evaluated in `outputs/viewactive/fidelity_split/`, and
the independent Blender reproduction is running in
`outputs/viewactive/blender_full/`. Keep ViewActive out of downstream study
comparisons until its full Blender evaluation meets the criteria in
[docs/viewactive.md](docs/viewactive.md).
Field prediction beats the mean-field prior on seen and unseen classes, while
classifier accuracy gains remain below the paper's Fig. 4. The full evaluation
was regenerated on CPU after fixing undefined-label aggregation.
To compare full-frame classifier accuracy with a tight object crop using the
existing pilot renders, run
`sam3d-env sam3d python -m viewactive.crop_experiment --config configs/viewactive_pilot.yaml`.
For tightly framed classifier evaluation, set `classification.crop_margin: 0.15`
in a ViewActive config; the classifier crops each view around its modal mask.
The default `null` keeps full-frame evaluation. Use a fresh output directory
when changing this setting or regenerating renders after the mesh color fix.

## Code

- `study/`: stage driver, data store, confidence margins, view selection, and figures.
- `viewactive/`: the ViewActive reimplementation (data, model, Algorithm 1, evaluation).
- `study/ambiguity.py`: the three ambiguity metrics, including entropy scoring.
- `study/figures.py`: grid heatmaps and image montages.
- `harness/`: Objaverse canonicalization, upright review, and ground-truth rendering.
- `reconstruction/`: SAM3 detection and SAM3D viewpoint rendering.
- `docker/` and `tools/`: the two required model environments and shared camera convention.

Run CPU tests with `sam3d-env core python -m unittest discover -s tests -v`.
ViewActive label tests need the `sam3d` env: `sam3d-env sam3d python -m unittest discover -s tests -p test_viewactive.py`.
The mesh-metric tests also need the `sam3d` environment.
