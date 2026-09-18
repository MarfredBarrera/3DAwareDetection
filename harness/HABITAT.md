# Object-centered Habitat image dataset

## Plan and implementation

`python -m harness.habitat_dataset` implements these stages:

1. Load each Habitat scene and its semantic annotations. Match normalized category
   labels to queries, keeping all matching instances and their semantic IDs/bounds.
2. Sample horizontal camera rings around each instance's axis-aligned bounding-box
   center, crossing radii × elevations × azimuths × translation offsets.
3. Render aligned RGB and semantic images, optionally depth. Filter using the target
   instance's visible pixels; optionally reject border-touching masks or require a
   navigable floor point below the camera.
4. Save category/scene/instance folders and a JSONL index for sorting and filtering.
   Save object inventory and rejection counts to diagnose missing coverage.

## Environment and resources

Use an existing **Habitat-Sim** environment with `numpy`, `numpy-quaternion`, and
`Pillow`. The project's reconstruction Docker image does **not** include Habitat.
Follow the [official Habitat-Sim installation instructions](https://github.com/facebookresearch/habitat-sim#installation).
Supply a semantically annotated dataset, such as HM3D with its semantic assets,
using the corresponding `.scene_dataset_config.json`. Plain RGB scene geometry
without semantic instance annotations cannot identify beds. Scene handles are
resolved by Habitat; use absolute paths when passing scene assets/configs.

Before every GPU run, inspect `nvidia-smi` and select an idle device. Run generation
under `cpulimit` (install it in the Habitat environment/host if absent). The script
loads one scene at a time and writes each accepted frame immediately.

```bash
nvidia-smi
# Set this to the idle GPU selected from the output above.
GPU_ID=0
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
cpulimit --limit=200 --foreground -- python -m harness.habitat_dataset \
  --scene-dataset /datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json \
  --scenes /datasets/hm3d/val/SCENE/SCENE.basis.glb \
  --queries bed --out outputs/habitat-beds \
  --radii 1.5 2 3 --azimuth-step 30 --elevations 0 15 30 \
  --offset 0 0 0 --offset 0.3 0 0 --offset -0.3 0 0 --offset 0 0.2 0 \
  --save-depth --gpu-device-id "$GPU_ID"
```

Replace the example dataset paths with installed assets. Add scene paths after
`--scenes` and labels after `--queries` to build a multi-scene/category dataset.
The example proposes 3 × 12 × 3 × 4 = **432 views per matching instance**, before
filtering. Start with one scene, `--radii 2 --elevations 15 --azimuth-step 90`
and a single offset for a small smoke run. Use a new/empty output directory;
existing datasets are never overwritten. Failed runs retain partial outputs and
`summary.json` with `status: failed`.

## Sampling conventions

- Distances and translations are **meters** in the scene's world frame (+Y up).
- `--radii` are horizontal XZ distances from the bounding-box center, not distances
  from the object's surface. Height is `center_y + radius * tan(elevation)`.
- Azimuth starts at world +Z and progresses toward +X. Samples lie in `[0, 360)`;
  the endpoint is never duplicated. These are world directions, not semantic
  front/back labels. Non-divisors of 360 leave a shorter closing interval.
- The initial camera looks at the center with no roll. `--offset RIGHT UP FORWARD`
  translates along its local axes **without re-aiming**. Positive right moves the
  target left in the image; positive up moves it down. Repeat the option to list
  translations; include `--offset 0 0 0` explicitly if centered views are wanted
  alongside offsets. Without any offset options, only zero offset is used.
- `camera_to_world` is a row-major 4×4 matrix with +X right, +Y up, −Z forward.
  For computer-vision axes (+X right, +Y down, +Z forward), postmultiply by
  `diag(1, -1, -1, 1)`. Intrinsics use integer pixel centers, principal point
  `((width-1)/2, (height-1)/2)`, and square pixels.

## Outputs

- `config.json`: arguments, coordinate conventions, intrinsics.
- `objects.jsonl`: scene, category, matching queries, object ID, semantic ID, bounds.
- `index.jsonl`: one record per saved image, including object identity, requested
  sampling settings, actual center distance, camera pose, intrinsics, resolution,
  visible pixel count/fraction, centroid, bounding box, and relative file paths.
- `samples/<category-key>/<scene-key>/<instance-key>/*.rgb.png`, `*.mask.png`,
  and optional `*.depth.npy`. Keys include hashes to prevent naming collisions.
- `summary.json`: matched objects, saved views, available categories per scene,
  and rejection counts.

Filter `index.jsonl` by `category == "bed"`, then group by `scene` and `object_id`.
The mask is the **visible instance mask**, not a class mask or an amodal mask.
Visible fraction is fraction of image pixels, not an occlusion percentage.
Depth is Habitat's camera-axis depth in meters, saved as float32. The existing
reconstruction pipeline accepts `files.rgb`, optional `files.depth`, and
`intrinsics` (resolve paths relative to the dataset directory).

## Filters and limits

Exact case-insensitive category matching is default; underscores and repeated
whitespace are normalized. `--match substring` enables explicit substring matching.
Queries are semantic labels, not open-vocabulary descriptions: synonyms such as
`couch` and `sofa` must be supplied as separate queries when both are wanted.
No matches or no accepted views exits with an error and writes diagnostics.

Views inside the target's AABB are rejected. `--min-pixels` (default 100) and
`--min-fraction` (default 0.001) reject invisible/tiny targets, including occlusion
by scene geometry. `--reject-border` rejects masks touching an image edge.
Rooms often prevent a full ring; no fixed accepted view count is promised.

`--require-navigable` requires a loaded navmesh and checks a point
`--camera-height` below the translated camera with `--floor-tolerance` vertical
slack. It rejects poses instead of snapping them, preserving requested geometry.
It is a floor feasibility check, **not full camera-body collision detection**.
Unconstrained views can be inside other geometry or outside rooms; visibility
filtering alone does not guarantee physically reachable cameras. Elevated rings
and navmesh constraints may yield few views. Semantic annotation quality and
coverage determine which objects can be collected.

## Validation

CPU tests cover ring geometry, off-center translations, semantic IDs, mask
measurements, output writing/filtering, cleanup, and argument validation:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
cpulimit --limit=200 --foreground -- \
  python -m unittest discover -s tests -p test_habitat_dataset.py -v
```

A real Habitat smoke run remains necessary to verify the installed renderer and
scene annotations. Inspect RGB/mask alignment and the metadata before scaling up.
