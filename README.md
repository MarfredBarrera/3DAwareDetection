# Text-query 3D reconstruction

Give the pipeline an RGB image and a text query. It finds matching objects with
SAM3, selects the highest-confidence masks above a configured threshold, reconstructs them with Fast-SAM3D, and saves Gaussian PLYs, PNG views, and
turntable GIFs. An optional Gradio viewer displays the results.

## Setup and run

```bash
cp .env.example .env
# Check nvidia-smi and set gpu_id in config.yaml to an idle GPU.
docker compose build
# Requires approved Hugging Face access to facebook/sam3 and facebook/sam-3d-objects.
export HF_TOKEN=...
./scripts/download_models.sh
./scripts/check_dependencies.sh
docker compose run --rm --service-ports dev sam3d-env core python main.py
```

For rootless Docker, add `-f compose.yaml -f compose.rootless.yaml` to Compose
commands (or export `COMPOSE_FILE=compose.yaml:compose.rootless.yaml`).

Edit `config.yaml`: `image` is a path to an RGB image from any dataset, `query`
is the object description, and `output_dir` is the result directory. Relative
paths resolve against the YAML file. Use `--config path/to/config.yaml` for a
separate configuration. Set `preprocessing.threshold` to the minimum confidence
and `preprocessing.max_masks` to the maximum number of passing masks, ranked by
confidence. Compose exposes all GPUs; `gpu_id` selects the host GPU each worker
uses (as `cuda:0`). Old `source.*` selectors are replaced by `image`.

For a smoke test, check GPU usage first, then run:

```bash
docker compose run --rm dev sam3d-env core python main.py \
  --no-viewer --output-dir outputs/smoke
```

`--no-viewer` saves outputs and exits. Otherwise the viewer runs at
`http://localhost:7860` until interrupted (`WEB_PORT` changes the host port).
No duplicate debug configuration is needed.

Compose runs each worker under `cpulimit` at 200% (two CPU cores) and sets
numerical libraries to two threads. Set `CPU_LIMIT` to adjust the cap.
Check `nvidia-smi` before every GPU run on a shared server and set `gpu_id` in
the YAML accordingly.

## Depth and outputs

Optional `preprocessing.depth` is an aligned H×W NumPy depth array, paired with
`preprocessing.intrinsics: [fx, fy, cx, cy]`. Depth is camera-axis Z, not distance
along the ray; invalid pixels become zero points. Reconstruction uses the
input depth units (meters for metric images; canonical units for the harness).
Without depth, SAM3D estimates it.

Each run writes:

- `input.png`: one normalized RGB image shared by all selected objects.
- `masks/`: up to the configured number of highest-confidence binary masks and their confidence/box metadata.
- `run.json`: resolved settings and the current run's mask list.
- `reconstruction_timing.json`: GPU-synchronized SAM3D inference plus Gaussian
  preparation time for each object. It excludes model loading, PLY writing,
  turntable-frame rendering, PNG output, and GIF encoding.
- `object-NNN/`: `object.ply`, `views/*.png`, and `turntable.gif`.

Only the selected masks listed in `run.json` are reconstructed. No detections fails the run.
Use a new output directory to retain previous results or avoid stale artifacts.

## Code layout

For object-centered images from annotated Habitat scenes, see
[the Habitat dataset generator](harness/HABITAT.md). It samples camera rings,
off-center translations, and writes RGB/masks/depth with searchable metadata.

- `main.py`: configuration, sequential worker launch, and stage orchestration.
- `reconstruction/`: SAM3 segmentation, SAM3D inference, rendering/viewer.
- `harness/`: Objaverse canonicalization and controlled viewpoint/occlusion data.
  See [harness/README.md](harness/README.md).
- `experiments/viewpoint_sensitivity/`: balanced multi-view SAM3 recognition and
  paper-style top-1 accuracy heatmaps. See its
  [README](experiments/viewpoint_sensitivity/README.md).
- `docker/`: pinned model environments and upstream compatibility patches.

The two model environments are necessary: SAM3 and Fast-SAM3D use incompatible
PyTorch/CUDA versions. The SAM3 process exits before SAM3D loads. The image does
not build Habitat or CLIP; dataset images are ordinary files.

The smoke-test fixes are included in the build: pinned local DINOv2 source,
DINO weights loaded from SAM3D checkpoints, removal of the undefined upstream
solver registration, and Fast-SAM3D configs loaded from its source directory.
Rebuild the image to apply Docker dependency changes to new containers.

## Tests

```bash
docker compose run --rm dev sam3d-env core python -m unittest discover -s tests -p test_pipeline.py -v
docker compose run --rm dev sam3d-env sam3d python -m unittest discover -s tests -p test_harness.py -v
```

These tests use CPU computation; the full smoke test above exercises real models.
