# 3D-aware detection verification

An object detector's false positives often come from a bad viewpoint. This
repository verifies a detection by looking at the object from a better one:

1. **SAM3** gives the object's mask (or the navigation planner supplies it).
2. **SAM3D** reconstructs the masked object and renders candidate views
   around it.
3. **Ensemble filter** (optional): stage 1 of SAM3D is rerun from several
   seeds; if the shapes disagree by more than a threshold, the reconstruction
   is not trusted.
4. **View selection**: `vqf` (learned ViewActive viewpoint quality) or
   `top2gap` (the view where SAM3's top category beats the runner-up most).
5. **SAM3** ranks a closed vocabulary on the chosen view. The detection is
   accepted when the top-1 category is the target.

The end goal is to measure what this does for off-the-shelf zero-shot
navigation (ObjectNav) planners such as ApexNav.

## Setup

Set `OBJAVERSE_DIR` in `.env` (see `.env.example`), then:

```bash
cp .env.example .env
docker compose build
export HF_TOKEN=...                         # approved SAM3 and SAM3D model access
./scripts/download_models.sh
./scripts/check_dependencies.sh
```

For rootless Docker, add `-f compose.yaml -f compose.rootless.yaml` to Compose
commands. SAM3 and SAM3D need incompatible PyTorch builds, so the image has two
conda envs, `core` and `sam3d`; `sam3d-env <env> python ...` runs in one.

The ViewActive model is trained here, not downloaded. The pipeline expects it at
`/data/checkpoints/viewactive/blender_full_el0_45.pt` (the `model/best.pt` of
`configs/viewactive_blender_el0_45.yaml`; see [docs/viewactive.md](docs/viewactive.md)).

## Use from a navigation planner

```python
from perception import Verifier

with Verifier(selector="vqf",              # or "top2gap"
              ensemble=True,               # the ensemble filter; False turns it off
              ensemble_threshold=0.15,
              on_uncertain="input",        # distrusted: classify the raw frame ("reject", "accept")
              vocabulary=["chair", "couch", "bed", "toilet", "tv", "table", "cabinet"],
              gpu=0) as verifier:
    verdict = verifier.verify(rgb, mask, target="chair", input_elevation=-pitch_degrees)
    if verdict.accepted:
        ...  # commit to the detection
```

- `rgb` and `mask` can be paths, PIL images, or numpy arrays (mask nonzero on
  the object). Without a mask, `verifier.segment(rgb, "chair")` returns SAM3
  instances `[(mask, score, box)]`, best first.
- `input_elevation` is the camera's elevation in degrees, positive when looking
  down at the object (Habitat: `-pitch`). It keeps the candidate views level.
- `verdict` has `accepted`, `label`, `scores`, `judged` (`"view"`, `"input"` or
  `"none"`), the chosen `view`, the ensemble `uncertainty`, `trusted`, and
  `timing`. The default vocabulary is `perception.HOUSEHOLD` (ApexNav's goals
  plus the mpcat40 object classes).
- Models load once, in two worker processes, and stay loaded between calls
  (about 11 s per verification with the filter, 7 s without, on this server).
  The `Verifier` can run inside the container, or on the host, where it starts
  its workers with `docker exec` into `$PERCEPTION_CONTAINER`
  (default `3dawareverification-dev-1`); pass `launcher=[...]` for anything else.
  The client needs only numpy and Pillow, so the repo can be put on the
  planner's `PYTHONPATH`.

One frame from the command line (writes `verdict.json` and the views to `--out`):

```bash
sam3d-env core python -m perception --image frame.png --target chair --elevation 20 --gpu 0
sam3d-env core python -m perception --image frame.png --mask mask.png --target chair \
    --selector top2gap --no-ensemble
```

### ApexNav false positives

`scripts/apexnav_fp.py` runs the `Verifier` on ApexNav's recorded
false-positive episodes (`data/apexnav/records/`) and compares every selector,
with and without the ensemble filter, against the manually verified labels:

```bash
sam3d-env core python -m scripts.apexnav_fp run --gpu 0
sam3d-env core python -m scripts.apexnav_fp figure   # results.json, apexnav_fp_resolved.png
sam3d-env core python -m scripts.apexnav_fp grids    # one page per episode
```

## Offline study (Objaverse ground truth)

The study measures both claims with controls on rendered Objaverse objects:
SAM3 is run on ground-truth views (the "before"), then on the SAM3D view each
selector picks (the "after"). **Margin** = SAM3 score of the true category
minus the strongest wrong one; negative means a misleading view. An after-result
only counts if it beats the `identity` (re-render the input view) and `random`
controls; `oracle` is the upper bound.

```bash
sam3d-env core python -m study --config configs/upright.yaml
sam3d-env core python -m study --config configs/upright.yaml --stages analyze
```

Stages are `render → reconstruct → score → pick → detect → analyze`; each
resumes. Set `gpu_id` to an idle GPU first. `study.sqlite` in the output
directory holds every image, angle, score and pick.

- `render`: ground-truth views with the PyTorch3D harness, or imported from a
  ViewActive Blender run (`dataset.source_run`).
- `reconstruct`: SAM3D candidate grid per input; `reconstruction.ensemble > 1`
  also writes the stage-1 ensemble (`stage1.npz`).
- `score`: the ambiguity metrics (`study/ambiguity.py`) on each candidate.
- `pick`: with a `picks` section, the learned `vqf` pick (plus `identity` and
  one `random` control) before detection; `detect_only` restricts SAM3 to them.
- `detect`: SAM3 scores the full vocabulary on every needed image.
- `analyze`: selections, `summary.md`/`summary.json`, and figures.
  `analysis.ensemble_threshold` turns the ensemble filter on: distrusted
  inputs keep the input view for every selector.

Selectors (`study/selectors.py`): `identity`, `random`, `ambiguity:<metric>`,
`ambiguity:combined`, `top2gap`, `vqf`, `oracle`. New selectors and metrics
need no GPU work.

Configs: `smoke.yaml` (tiny), `upright.yaml` (reviewed 110 objects x 60 views),
`lvis_uncertainty.yaml` (1,800 ViewActive objects, one view each; the
calibration set for the ensemble threshold).

Figures and inspection:

```bash
sam3d-env core python -m study.figures --config configs/upright.yaml object <uid>
sam3d-env core python -m study.figures --config configs/upright.yaml reconstruction gt/<uid>/a30_e15
sam3d-env core python -m study.compare --config configs/upright.yaml --uid <uid> --selector vqf
```

Outputs under `output_dir`: `figures/before/` (where ground-truth views mislead
SAM3), `figures/after/` (before/after by selector), `figures/objects/`
(per object), `summary.md`, `study.sqlite`, and `renders/`.

### Ensemble threshold

```bash
sam3d-env sam3d python -m study.ensemble gt_error --config configs/lvis_uncertainty.yaml
sam3d-env core python -m study.ensemble calibrate --config configs/lvis_uncertainty.yaml \
    --mesh-f 0.5 --max-brightness 0.9
```

A reconstruction is not useful when its mesh F-score@0.05 against the ground
truth is below 0.5. The threshold is chosen by Learn-then-Test on held-out
categories; 0.10-0.20 keeps ~99% of useful reconstructions.

## ViewActive

`viewactive/` reimplements ViewActive (Wu et al., arXiv:2409.09997): it predicts
a viewpoint quality field from one image. Train with

```bash
sam3d-env sam3d python -m viewactive --config configs/viewactive_blender_smoke.yaml
sam3d-env sam3d python -m viewactive --config configs/viewactive_blender_el0_45.yaml
```

Stages: `select → download → canonicalize → cohort → render → label → features
→ classify → train → evaluate`. Blender renders need `bash scripts/install_blender.sh`
in the container. Methods, deviations and results: [docs/viewactive.md](docs/viewactive.md).

## Code

- `perception/`: the pipeline (`Verifier`), its model workers and the selection rules.
- `reconstruction/`: SAM3 and SAM3D wrappers, the stage-1 ensemble, mesh error.
- `study/`: the offline benchmark (driver, database, selectors, analysis, figures).
- `viewactive/`: the ViewActive reimplementation.
- `harness/`: Objaverse canonicalization, upright review, ground-truth rendering.
- `tools/`: the shared camera convention and env helpers.

Tests: `sam3d-env core python -m unittest discover -s tests`, plus
`test_ambiguity.py`, `test_harness.py`, `test_viewactive.py` and
`test_gt_error.py` in the `sam3d` env
(`sam3d-env sam3d python -m unittest discover -s tests -p test_ambiguity.py`).
