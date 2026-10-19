"""Ground-truth VQF labels: paper Eqs. (1)-(3) at every grid view of one object.

The metrics are ``study.ambiguity``'s estimators, evaluated on the canonical
mesh with the harness camera (look-at ``HARNESS_LOOK_AT``) so labels and
renders share one geometry. Channels, each scaled to [0, 1] before per-object
normalization:

- ``visible_ratio``: 1 - occluded area / total area (Eq. 1).
- ``normal_entropy``: occupancy x visible-normal entropy / log2(bins) (Eq. 2).
- ``visual_entropy``: foreground grayscale entropy / log2(bins) (Eq. 3).
"""
import math
import warnings

import numpy as np
from PIL import Image

from study.ambiguity import build_context, self_occlusion, surface_normal_entropy, visual_entropy
from tools.cameras import HARNESS_LOOK_AT

CHANNELS = ("visible_ratio", "normal_entropy", "visual_entropy")


def raw_quality(vertices, faces, grid, views, *, distance, samples, seed, polar_bins,
                azimuth_bins, visual_bins, nthreads=0):
    """``[3, E, A]`` raw channel values; NaN where a view shows nothing.

    ``views[e][a]`` is ``(rgb_path, mask_path)`` for grid cell (e, a).
    """
    context = build_context(vertices, faces, samples=samples, seed=seed, target=HARNESS_LOOK_AT,
                            distance=distance, two_sided=True, nthreads=nthreads)
    out = np.full((len(CHANNELS), *grid.shape), np.nan, dtype=np.float32)
    for e, elevation in enumerate(grid.elevations):
        for a, azimuth in enumerate(grid.azimuths):
            camera = {"azimuth": azimuth, "elevation": elevation}
            visible = self_occlusion(context, camera)
            normals = surface_normal_entropy(context, camera, polar_bins=polar_bins,
                                             azimuth_bins=azimuth_bins)
            rgb, mask = views[e][a]
            with Image.open(rgb) as image, Image.open(mask) as foreground:
                visual = visual_entropy(image, foreground, bins=visual_bins)
            if visible["visible_samples"]:
                out[0, e, a] = visible["visible_ratio"]
            if normals["normal_entropy_bits"] is not None:
                out[1, e, a] = normals["normal_entropy_bits"] / math.log2(polar_bins * azimuth_bins)
            if visual["entropy_bits"] is not None:
                out[2, e, a] = visual["entropy_bits"] / math.log2(visual_bins)
    return out


def normalize(raw, floor=0.05):
    """Per-object, per-channel min-max to [floor, 1]; NaN stays NaN, constant -> midpoint.

    Relative values drive viewpoint choice (paper Sec. III-C), and the floor
    keeps the log in the SILog loss finite.
    """
    raw = np.asarray(raw, dtype=np.float32)
    flat = raw.reshape(*raw.shape[:-2], -1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN channels
        low = np.nanmin(flat, axis=-1)[..., None, None]
        high = np.nanmax(flat, axis=-1)[..., None, None]
        span = high - low
        unit = np.where(span > 1e-9, (raw - low) / np.where(span > 1e-9, span, 1), 0.5)
    return np.where(np.isfinite(raw), floor + (1 - floor) * unit, np.nan).astype(np.float32)


_STORE = None


def label_object(task):
    """Pool worker: ``task`` = (uid, store root, canonical, views, options) -> (uid, raw, error)."""
    global _STORE
    from harness.meshio import ObjaverseStore
    from viewactive.grid import Grid

    uid, root, canonical, views, options = task
    try:
        if _STORE is None or _STORE.root != root:
            _STORE = ObjaverseStore(root, canonical)
        mesh = _STORE.load(uid, device="cpu")
        if mesh is None:
            return uid, None, "unloadable mesh"
        grid = Grid(tuple(options["azimuths"]), tuple(options["elevations"]))
        raw = raw_quality(mesh.verts.numpy(), mesh.faces.numpy(), grid, views,
                          distance=options["distance"], samples=options["samples"],
                          seed=options["seed"], polar_bins=options["polar_bins"],
                          azimuth_bins=options["azimuth_bins"], visual_bins=options["visual_bins"],
                          nthreads=options.get("nthreads", 0))
        return uid, raw, None
    except Exception as error:
        return uid, None, f"{type(error).__name__}: {error}"
