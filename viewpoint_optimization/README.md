# FastSAM3D viewpoint grid and scores

Edit [config.yaml](config.yaml), then run inside the project container:

```bash
docker compose run --rm dev sam3d-env core python -m viewpoint_optimization
```

Use `--config /workspace/path/to/config.yaml` for another input. Paths in the
YAML resolve relative to the YAML file. Set `image` and exactly one of `query`
(a SAM3 text prompt) or `mask` (an aligned binary mask). The sample uses the
saved RGB and mask from `outputs/sam3d_correction`. Check GPU availability
before running jobs on a shared server.

`reconstruction.azimuths` are columns and `reconstruction.elevations` are rows.
The sample uses the same 12 azimuths and five elevations as
`experiments/viewpoint_sensitivity/config.upright.yaml`. FastSAM3D uses the
same spherical camera-angle formula, although its reconstructed object's local
orientation may differ from the earlier canonical mesh frame.

Each run writes `object.ply` (Gaussian splats), `mesh.ply` and `mesh.npz`
(aligned decoded triangles), rendered images and foreground masks in `views/`,
`candidates.json`, `scores_<metric>.json`, and `viewpoint_grid_<metric>.png`.
The latest score plot is also `viewpoint_grid.png`.

## Metrics

Select a metric under `scoring.metric` in the YAML:

- `visual_entropy`: [ViewActive Eq. (3)](https://arxiv.org/pdf/2409.09997) on the rendered foreground grayscale histogram. Higher entropy is better; ambiguity is `1 - entropy_bits / log2(bins)`.
- `self_occlusion`: approximate [Eq. (1)](https://arxiv.org/pdf/2409.09997), occluded mesh surface area divided by total surface area. Lower is better.
- `surface_normal_entropy`: approximate [Eq. (2)](https://arxiv.org/pdf/2409.09997), occupied normal bins divided by all hemisphere bins, multiplied by Shannon entropy of visible normal directions. Higher is better; ambiguity is `1 - quality / log2(total_bins)`.

Both mesh metrics sample triangles proportional to area and ray cast from each
camera. They reuse the same deterministic sample for all viewpoints. This is a
sampling approximation to the paper's polygon surface-area definitions; the
sample count and seed are metric options. The normal metric also accepts
`polar_bins` and `azimuth_bins`. Examples:

```yaml
scoring:
  metric: self_occlusion
  options:
    samples: 20000
    seed: 0
```

```yaml
scoring:
  metric: surface_normal_entropy
  options:
    samples: 20000
    seed: 0
    polar_bins: 9
    azimuth_bins: 18
```

After changing only `scoring`, reuse the saved reconstruction:

```bash
docker compose run --rm dev sam3d-env core python -m viewpoint_optimization --refresh-layout
```

New metrics register their scorer, defaults, validator, and preparation hook in
[metrics.py](metrics.py); the geometric implementation lives in
[mesh_metrics.py](mesh_metrics.py). Every scorer returns a 0–1 ambiguity value
where lower is better. These are visual or geometric proxies, not calibrated
semantic uncertainty scores. Empty views are stored as `null`.
