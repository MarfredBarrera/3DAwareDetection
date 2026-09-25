"""Plot angle-aligned FastSAM3D renders and ambiguity scores."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps


def render_layout(output_dir, scores_path=None):
    """Save a paired azimuth/elevation heatmap and RGB view grid as one PNG."""
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "candidates.json").read_text())
    payload = json.loads(Path(scores_path).read_text()) if scores_path else {}
    structured = "scores" in payload
    entries = payload["scores"] if structured else payload
    if not isinstance(entries, dict):
        raise ValueError("Scores must be a JSON object keyed by candidate filename stem")
    scores = {key: value["ambiguity"] if structured else value
              for key, value in entries.items()}
    candidates = manifest["candidates"]
    keys = {Path(item["image"]).stem for item in candidates}
    if set(scores) - keys:
        raise ValueError(f"Unknown score keys: {sorted(set(scores) - keys)}")
    for key, value in scores.items():
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value)):
            raise ValueError(f"Score for {key} must be a finite number or null")

    azimuths, elevations = manifest["azimuths"], manifest["elevations"]
    lookup = {(item["azimuth"], item["elevation"]): item for item in candidates}
    if len(lookup) != len(azimuths) * len(elevations):
        raise ValueError("Candidate grid is incomplete or contains duplicate angles")
    tile_size = 128
    montage = Image.new("RGB", (tile_size * len(azimuths), tile_size * len(elevations)), "white")
    grid = np.full((len(elevations), len(azimuths)), np.nan)
    for row, elevation in enumerate(elevations):
        for col, azimuth in enumerate(azimuths):
            item = lookup[(azimuth, elevation)]
            with Image.open(output_dir / item["image"]) as source:
                tile = ImageOps.fit(source.convert("RGB"), (tile_size, tile_size))
            montage.paste(tile, (col * tile_size, (len(elevations) - 1 - row) * tile_size))
            value = scores.get(Path(item["image"]).stem)
            grid[row, col] = value if value is not None else np.nan

    fig, (score_ax, rgb_ax) = plt.subplots(
        1, 2, figsize=(3 + 1.5 * len(azimuths), 2 + 1.2 * len(elevations)))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#e9ece9")
    values = np.asarray([value for value in scores.values() if value is not None], dtype=float)
    vmin, vmax = ((0.0, 1.0) if structured else
                  (float(values.min()), float(values.max())) if len(values) else (0.0, 1.0))
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5
    heat = score_ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    if len(values):
        fig.colorbar(heat, ax=score_ax, shrink=0.8, label="ambiguity score")
    for row in range(len(elevations)):
        for col in range(len(azimuths)):
            score_ax.text(col, row, f"{grid[row, col]:.2f}" if np.isfinite(grid[row, col]) else "—",
                          ha="center", va="center", color="white" if np.isfinite(grid[row, col]) else "#65716d")
    label = ((payload["metric"].replace("_", " ").capitalize() + " ambiguity (lower is better)")
             if structured else "Ambiguity score")
    score_ax.set_title(label if len(values) else "Ambiguity score (unscored)")
    rgb_ax.imshow(montage, extent=(-0.5, len(azimuths) - 0.5, -0.5, len(elevations) - 0.5),
                  origin="upper", aspect="auto")
    rgb_ax.set_title("FastSAM3D rendered views")
    for ax in (score_ax, rgb_ax):
        ax.set_xticks(range(len(azimuths)), [f"{azimuth:g}" for azimuth in azimuths])
        ax.set_yticks(range(len(elevations)), [f"{elevation:g}" for elevation in elevations])
        ax.set_xlabel("azimuth (deg)")
        ax.set_ylabel("elevation (deg)")
        ax.set_xticks(np.arange(-0.5, len(azimuths), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(elevations), 1), minor=True)
        ax.grid(which="minor", color="black", linewidth=0.5, alpha=0.25)
        ax.tick_params(which="minor", bottom=False, left=False)
    fig.suptitle(f"Viewpoint optimization: {manifest.get('query') or 'object'}")
    fig.tight_layout()
    path = output_dir / "viewpoint_grid.png"
    fig.savefig(path, dpi=150)
    if structured:
        fig.savefig(output_dir / f"viewpoint_grid_{payload['metric']}.png", dpi=150)
    plt.close(fig)
    return path
