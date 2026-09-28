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

For one input frame, put the image in `data/`. `main.py` runs SAM3 to select a
foreground mask, reconstructs with SAM3D, saves a turntable and PLY, and opens
a viewer on port 7860:

```bash
sam3d-env core python main.py --image data/frame.png --query chair
sam3d-env core python main.py --image data/frame.png --query chair --no-viewer
```

Edit `configs/demo.yaml` for checkpoints, GPU, and viewer settings. Optional
aligned depth and intrinsics can also be supplied there.

To inspect one object or one reconstruction's candidate grid:

```bash
sam3d-env core python -m study.figures --config configs/upright.yaml object <uid>
sam3d-env core python -m study.figures --config configs/upright.yaml reconstruction gt/<uid>/a30_e15
```

## Outputs

Everything goes under the config's `output_dir`:

- `figures/before/`: confidence-margin, accuracy, and false-positive heatmaps by category.
- `figures/objects/<category>/before/`: per object, its margin heatmap, a bar chart
  of wrong categories whose scores exceed the true category, and its viewpoint
  grid. Bars show mean (true score − wrong score) over the views where that
  category wins. When reconstruction results exist, `after/` shows the identity
  and selected-view margins and chosen views, and `oracle/`, `detector_entropy/`
  and `detector_gap/` show the input and chosen SAM3D views above the before and after margins.
- `figures/explore/`: on-demand reconstruction grids with SAM3 margins and entropy scores.
- `figures/after/`: before/after false-positive maps, accuracy and false-positive
  summary bars, and margin charts showing mean margin and change from input.
- `summary.md` and `summary.json`: aggregate before/after results.
- `study.sqlite`: source data for images, SAM3 scores, ambiguity scores, and view choices.
- `renders/gt/` and `renders/sam3d/`: ground-truth and reconstructed viewpoint images.

The stages are `render → reconstruct → score → detect → analyze`. `render` uses
Objaverse meshes and the camera grid in the YAML. `reconstruct` writes SAM3D
view grids and decoded meshes. `score` computes ambiguity per candidate.
`detect` evaluates every image against the full category vocabulary with SAM3.
`analyze` generates the figures. The candidate angles are relative to the input
view; the ground-truth angles use the canonical object frame.

## Code

- `study/`: stage driver, data store, confidence margins, view selection, and figures.
- `study/ambiguity.py`: the three ambiguity metrics, including entropy scoring.
- `study/figures.py`: grid heatmaps and image montages.
- `harness/`: Objaverse canonicalization, upright review, and ground-truth rendering.
- `reconstruction/`: SAM3 detection and SAM3D viewpoint rendering.
- `docker/` and `tools/`: the two required model environments and shared camera convention.

Run CPU tests with `sam3d-env core python -m unittest discover -s tests -v`.
The mesh-metric tests also need the `sam3d` environment.
