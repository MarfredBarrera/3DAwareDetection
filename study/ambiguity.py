"""Registered per-view ambiguity metrics for one SAM3D reconstruction.

Every metric returns an ambiguity in [0, 1]: 0 = least ambiguous view, 1 = most.
``None`` means undefined (e.g. nothing visible). The three metrics approximate
ViewActive (Wu et al., arXiv:2409.09997) Eqs. (1)-(3):

- ``self_occlusion``: occluded mesh surface area / total area (Eq. 1).
- ``surface_normal_entropy``: occupancy x entropy of visible normals (Eq. 2), inverted.
- ``visual_entropy``: Shannon entropy of the foreground grayscale histogram (Eq. 3), inverted.

Candidates are dicts with ``azimuth``, ``elevation``, and absolute ``image`` and
``mask`` paths. Mesh metrics read ``<reconstruction dir>/mesh.npz``.
"""
from dataclasses import dataclass, field
import math
from pathlib import Path

import numpy as np
from PIL import Image

from tools.cameras import direction as camera_direction


def visual_entropy(image, mask, *, bins=32):
    """Return ambiguity, entropy in bits, and visible-pixel count for one view."""
    if type(bins) is not int or not 2 <= bins <= 256:
        raise ValueError("visual_entropy bins must be an integer in [2, 256]")
    grayscale = np.asarray(image.convert("L"))
    foreground = np.asarray(mask.convert("L")) > 0
    if grayscale.shape != foreground.shape:
        raise ValueError("Rendered image and mask must have the same size")
    count = int(foreground.sum())
    if count == 0:
        return {"ambiguity": None, "entropy_bits": None, "foreground_pixels": 0}
    histogram = np.histogram(grayscale[foreground], bins=bins, range=(0, 256))[0]
    probabilities = histogram[histogram > 0] / count
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    ambiguity = float(np.clip(1 - entropy / math.log2(bins), 0, 1))
    return {"ambiguity": ambiguity, "entropy_bits": entropy, "foreground_pixels": count}


@dataclass
class MeshContext:
    points: np.ndarray
    normals: np.ndarray
    scene: object
    epsilon: float
    target: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    distance: float = 1.0
    two_sided: bool = False
    nthreads: int = 0  # Open3D cast_rays threads; 0 = Open3D default (all cores).


def load_mesh_npz(path):
    """Read float64 vertices and int32 faces from a reconstruction mesh.npz."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing FastSAM3D mesh: {path}. Rerun reconstruction.")
    with np.load(path) as data:
        return (np.asarray(data["vertices"], dtype=np.float64),
                np.asarray(data["faces"], dtype=np.int32))


def build_context(vertices, faces, *, samples, seed, target=(0.0, 0.0, 0.0),
                  distance=1.0, two_sided=False, nthreads=0):
    """Validate a mesh and draw one deterministic area-weighted surface sample."""
    import open3d as o3d

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("mesh must contain vertices Nx3 and triangular faces Mx3")
    if (not np.isfinite(vertices).all() or not len(faces) or not np.issubdtype(faces.dtype, np.integer)
            or np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError("mesh contains invalid geometry")
    faces = faces.astype(np.int32)
    target = np.asarray(target, dtype=np.float64)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("target must be three finite values")
    if (not isinstance(distance, (int, float)) or isinstance(distance, bool)
            or not math.isfinite(distance) or distance <= 0):
        raise ValueError("distance must be a finite positive number")
    if type(nthreads) is not int or nthreads < 0:
        raise ValueError("nthreads must be a nonnegative integer")
    if type(samples) is not int or samples < 1:
        raise ValueError("samples must be a positive integer")
    triangles = vertices[faces]
    crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(crosses, axis=1)
    valid = lengths > 1e-12
    if not valid.any():
        raise ValueError("mesh has no positive-area faces")
    faces, triangles, crosses, lengths = faces[valid], triangles[valid], crosses[valid], lengths[valid]
    areas = lengths / 2
    normals = crosses / lengths[:, None]
    # Correct a globally reversed winding; visibility needs outward normals.
    centroid = triangles.mean(axis=1) - vertices.mean(axis=0)
    if np.sum(areas * np.einsum("ij,ij->i", normals, centroid)) < 0:
        normals = -normals
        faces = faces[:, [0, 2, 1]]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(faces), size=samples, p=areas / areas.sum())
    bary_u = np.sqrt(rng.random(len(chosen)))
    bary_v = rng.random(len(chosen))
    selected = triangles[chosen]
    points = ((1 - bary_u)[:, None] * selected[:, 0] +
              (bary_u * (1 - bary_v))[:, None] * selected[:, 1] +
              (bary_u * bary_v)[:, None] * selected[:, 2])
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices.astype(np.float32)), o3d.core.Tensor(faces))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    epsilon = max(1e-4, 1e-4 * np.ptp(vertices, axis=0).max())
    return MeshContext(points.astype(np.float32), normals[chosen].astype(np.float32), scene, epsilon,
                       target.astype(np.float32), float(distance), bool(two_sided), nthreads)


def visible_samples(context, candidate):
    """Return the visible-sample mask, camera direction, and camera-facing normals."""
    import open3d as o3d

    direction = camera_direction(candidate["azimuth"], candidate["elevation"])
    camera = (context.target + context.distance * direction).astype(np.float32)
    to_camera = camera - context.points
    distances = np.linalg.norm(to_camera, axis=1)
    directions = to_camera / np.maximum(distances[:, None], 1e-8)
    alignment = np.einsum("ij,ij->i", context.normals, directions)
    if context.two_sided:
        # Back faces are drawn too, so every sample is a candidate and its
        # normal is flipped toward the camera.
        candidates = np.ones(len(context.points), dtype=bool)
        view_normals = context.normals * np.where(alignment < 0, -1, 1).astype(np.float32)[:, None]
    else:
        candidates = alignment > 0
        view_normals = context.normals
    visible = np.zeros(len(context.points), dtype=bool)
    if candidates.any():
        # Ray direction points from the camera toward each sampled surface point.
        ray_directions = -directions[candidates]
        origins = np.broadcast_to(camera, ray_directions.shape)
        rays = np.column_stack((origins, ray_directions)).astype(np.float32)
        hit = context.scene.cast_rays(o3d.core.Tensor(rays), nthreads=context.nthreads)["t_hit"].numpy()
        visible[candidates] = np.isfinite(hit) & (hit >= distances[candidates] - context.epsilon)
    return visible, direction, view_normals


def self_occlusion(context, candidate, **options):
    """Estimate paper Eq. (1): occluded area divided by total mesh area."""
    visible, _, _ = visible_samples(context, candidate)
    ratio = 1 - float(visible.mean())
    return {"ambiguity": ratio, "occlusion_ratio": ratio,
            "visible_ratio": 1 - ratio, "visible_samples": int(visible.sum()),
            "total_samples": len(visible)}


def surface_normal_entropy(context, candidate, **options):
    """Estimate paper Eq. (2): occupancy ratio times visible-normal entropy."""
    visible, direction, view_normals = visible_samples(context, candidate)
    if not visible.any():
        return {"ambiguity": None, "normal_entropy_bits": None,
                "occupied_bins": 0, "visible_samples": 0}
    normals = view_normals[visible]
    forward = direction / np.linalg.norm(direction)
    # Eq. (2) bins the hemisphere facing the camera from the object center.
    normals = normals[(normals @ forward) >= 0]
    if not len(normals):
        return {"ambiguity": None, "normal_entropy_bits": None,
                "occupied_bins": 0, "visible_samples": int(visible.sum())}
    reference_up = np.array([0, 1, 0], dtype=np.float32)
    if abs(float(np.dot(forward, reference_up))) > 0.99:
        reference_up = np.array([0, 0, 1], dtype=np.float32)
    right = np.cross(reference_up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    polar = np.arccos(np.clip(normals @ forward, 0, 1))
    azimuth = np.arctan2(normals @ up, normals @ right)
    polar_bins, azimuth_bins = options["polar_bins"], options["azimuth_bins"]
    histogram = np.histogram2d(
        polar, azimuth, bins=(polar_bins, azimuth_bins),
        range=((0, np.pi / 2), (-np.pi, np.pi)))[0]
    occupied = histogram[histogram > 0]
    probabilities = occupied / occupied.sum()
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    total_bins = polar_bins * azimuth_bins
    occupancy_ratio = len(occupied) / total_bins
    quality = occupancy_ratio * entropy
    ambiguity = float(np.clip(1 - quality / math.log2(total_bins), 0, 1))
    return {"ambiguity": ambiguity, "normal_entropy_bits": quality,
            "histogram_entropy_bits": entropy, "occupancy_ratio": occupancy_ratio,
            "occupied_bins": len(occupied), "visible_samples": int(visible.sum())}


@dataclass(frozen=True)
class Metric:
    score: object      # (context, candidate, **options) -> {"ambiguity": float | None, ...}
    defaults: dict
    validate: object   # (options) -> None, raises ValueError
    prepare: object    # (source, options, distance, target) -> context shared by all candidates


def _visual_view(context, candidate, **options):
    with Image.open(candidate["image"]) as image, Image.open(candidate["mask"]) as mask:
        return visual_entropy(image, mask, **options)


def _entropy_options(options):
    if type(options["bins"]) is not int or not 2 <= options["bins"] <= 256:
        raise ValueError("visual_entropy bins must be an integer in [2, 256]")


def _mesh_options(options):
    if type(options["samples"]) is not int or options["samples"] < 100:
        raise ValueError("mesh metric samples must be an integer >= 100")
    if type(options["seed"]) is not int or options["seed"] < 0:
        raise ValueError("mesh metric seed must be a nonnegative integer")
    if type(options["two_sided"]) is not bool:
        raise ValueError("mesh metric two_sided must be a boolean")


def _normal_options(options):
    _mesh_options(options)
    for name in ("polar_bins", "azimuth_bins"):
        if type(options[name]) is not int or not 2 <= options[name] <= 180:
            raise ValueError(f"{name} must be an integer in [2, 180]")


def _prepare_mesh(source, options, distance, target=(0.0, 0.0, 0.0)):
    """``source`` is a reconstruction directory or a ``(vertices, faces)`` mesh."""
    mesh = source if isinstance(source, tuple) else load_mesh_npz(Path(source) / "mesh.npz")
    return build_context(*mesh, samples=options["samples"], seed=options["seed"],
                         two_sided=options["two_sided"], distance=distance, target=target)


def _occlusion(context, candidate, **options):
    return self_occlusion(context, candidate)


def _normal_entropy(context, candidate, **options):
    return surface_normal_entropy(context, candidate, polar_bins=options["polar_bins"],
                                  azimuth_bins=options["azimuth_bins"])


# Two-sided visibility is robust to the open, inconsistently wound meshes both
# Objaverse and SAM3D produce, and the Gaussian renderer has no back-face culling.
_MESH_DEFAULTS = {"samples": 20000, "seed": 0, "two_sided": True}
METRICS = {
    "visual_entropy": Metric(_visual_view, {"bins": 32}, _entropy_options,
                             lambda source, options, distance, target=None: None),
    "self_occlusion": Metric(_occlusion, _MESH_DEFAULTS, _mesh_options, _prepare_mesh),
    "surface_normal_entropy": Metric(
        _normal_entropy, _MESH_DEFAULTS | {"polar_bins": 9, "azimuth_bins": 18},
        _normal_options, _prepare_mesh),
}


def metric_options(name, supplied=None):
    """Resolve and validate options owned by a registered metric."""
    if not isinstance(name, str) or name not in METRICS:
        raise ValueError(f"Unknown metric {name!r}; choices: {sorted(METRICS)}")
    supplied = {} if supplied is None else supplied
    if not isinstance(supplied, dict) or supplied.keys() - METRICS[name].defaults.keys():
        raise ValueError(f"Invalid options for {name}; allowed: {sorted(METRICS[name].defaults)}")
    options = METRICS[name].defaults | supplied
    METRICS[name].validate(options)
    return options


def score_candidates(source, candidates, metrics, distance=1.0, target=(0.0, 0.0, 0.0)):
    """Score every candidate with every metric; ``metrics`` maps name -> options.

    ``source`` is a reconstruction directory (``mesh.npz``) or a ``(vertices,
    faces)`` mesh; cameras look at ``target`` from ``distance``.

    Returns one ``{metric: ambiguity or None}`` dict per candidate, in order.
    Mesh preparation is shared between metrics with identical options.
    """
    results = [{} for _ in candidates]
    contexts = {}
    for name, supplied in metrics.items():
        options = metric_options(name, supplied)
        metric = METRICS[name]
        key = (metric.prepare, tuple(sorted((k, options[k]) for k in _MESH_DEFAULTS if k in options)))
        if key not in contexts:
            contexts[key] = metric.prepare(source, options, distance, target)
        for result, candidate in zip(results, candidates):
            result[name] = metric.score(contexts[key], candidate, **options)["ambiguity"]
    return results
