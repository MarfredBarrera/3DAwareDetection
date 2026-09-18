# Viewpoint × occlusion dataset

The harness canonicalizes Objaverse meshes, then renders RGB images, exact
visible/full-object masks, and optional aligned depth at controlled viewpoints
and occlusion levels. It shares the pipeline's `sam3d` environment.

## 1. Canonicalize (CPU)

Mount the dataset read-only at `/data/objaverse_nas` (Compose uses `OBJAVERSE_DIR`)
and keep generated metadata under `/outputs`:

```bash
sam3d-env sam3d python -m harness.canonicalize \
  --root /data/objaverse_nas --out /outputs/canonical.json
```

This reads `manifest.json`, `hf-objaverse-v1/object-paths.json.gz`, and GLBs.
It writes per-object transforms and quality flags; it never rewrites meshes.
The canonical frame is **+Y up, −Z forward, unit bounding-box diagonal, base at
y=0**. Up-axis selection uses category height statistics. Azimuth uses horizontal
PCA and upper-region asymmetry: it is repeatable geometry, not a semantic front.
Low-confidence, suspicious-up-axis, and non-watertight objects are flagged.

## 2. Render (GPU)

Check `nvidia-smi` and select an idle GPU before starting the container. Commands
run through `sam3d-env` inherit Compose's CPU limits.

```bash
sam3d-env sam3d python -m harness.build_dataset \
  --root /data/objaverse_nas --canonical /outputs/canonical.json \
  --out /outputs/viewocc --categories chair sofa --limit 20 \
  --azimuths 0 45 90 135 180 225 270 315 --elevations 0 20 40 \
  --occlusion-levels 0 0.15 0.3 0.45 0.6 --save-depth
```

Use `--help` for mesh filters, camera distance, occluder placement, and search
tolerance. A lateral sweep followed by bisection finds requested occlusion
levels. Unreachable levels are counted and skipped. An unoccluded image is
written only when level `0` is requested; it needs no occluder.

The full (amodal) mask rasterizes the target alone. The visible (modal) mask
uses target-face ownership when target and occluders are rasterized together.
Thus `occlusion_ratio = 1 − visible_area / full_area` is measured from pixels.
RGB and depth reuse the same scene fragments as the visible mask.

## Outputs and reconstruction

- `config.json`: run settings and coordinate convention.
- `index.jsonl`: sample paths, camera pose/intrinsics, measured occlusion, bounds,
  object identities, and truncation flags.
- `samples/<uid>/*.rgb.png`, `*.amodal.png`, `*.modal.png`, optional `*.depth.npy`.

Fractional view settings are retained in filenames. Each record includes
`intrinsics: [fx, fy, cx, cy]`; `files.depth` identifies depth when requested.
Set the main pipeline's `image` to the RGB path, `preprocessing.depth` to that
depth file, and `preprocessing.intrinsics` to the record's intrinsics. Depth is
camera-axis Z in **canonical units**, not meters; background depth is NaN.

## Limitations

Textures are sampled into vertex colors, losing high-frequency appearance.
The old `--texture-mode uv` option was removed because it did the same conversion.
Rendering uses a point light and no background/floor. Naive rasterization avoids
face-bin overflow on dense meshes but is slower. Geometric azimuth and
non-watertight meshes need care when interpreting semantic views or geometry
metrics.
