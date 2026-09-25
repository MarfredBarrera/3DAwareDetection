"""Approximate mesh visibility and normal diversity for one camera grid.

Surface points are sampled proportional to triangle area. Open3D ray casts
identify the first surface seen from each FastSAM3D camera position. The same
samples are reused across views, so score differences reflect viewpoint changes.
"""
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


@dataclass
class MeshContext:
    points: np.ndarray
    normals: np.ndarray
    scene: object
    epsilon: float


def prepare_mesh(output_dir, manifest, options):
    """Load the mesh and one deterministic area-weighted surface sample."""
    import open3d as o3d

    path = Path(output_dir) / "mesh.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing FastSAM3D mesh: {path}. Rerun reconstruction.")
    with np.load(path) as data:
        vertices = np.asarray(data["vertices"], dtype=np.float64)
        faces = np.asarray(data["faces"], dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("mesh.npz must contain vertices Nx3 and triangular faces Mx3")
    if not np.isfinite(vertices).all() or not len(faces) or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("mesh.npz contains invalid geometry")
    triangles = vertices[faces]
    crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(crosses, axis=1)
    valid = lengths > 1e-12
    if not valid.any():
        raise ValueError("FastSAM3D mesh has no positive-area faces")
    faces, triangles, crosses, lengths = faces[valid], triangles[valid], crosses[valid], lengths[valid]
    areas = lengths / 2
    normals = crosses / lengths[:, None]
    # Correct a globally reversed winding; visibility needs outward normals.
    centroid = triangles.mean(axis=1) - vertices.mean(axis=0)
    if np.sum(areas * np.einsum("ij,ij->i", normals, centroid)) < 0:
        normals = -normals
        faces = faces[:, [0, 2, 1]]
    rng = np.random.default_rng(options["seed"])
    chosen = rng.choice(len(faces), size=options["samples"], p=areas / areas.sum())
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
    return MeshContext(points.astype(np.float32), normals[chosen].astype(np.float32), scene, epsilon)


def visible_samples(context, candidate):
    """Return a visible-sample mask and the camera's object-centered direction."""
    import open3d as o3d

    azimuth = math.radians(candidate["azimuth"])
    elevation = math.radians(candidate["elevation"])
    camera = np.array([math.sin(azimuth) * math.cos(elevation), math.sin(elevation),
                       math.cos(azimuth) * math.cos(elevation)], dtype=np.float32)
    to_camera = camera - context.points
    distances = np.linalg.norm(to_camera, axis=1)
    directions = to_camera / np.maximum(distances[:, None], 1e-8)
    facing = np.einsum("ij,ij->i", context.normals, directions) > 0
    visible = np.zeros(len(context.points), dtype=bool)
    if facing.any():
        # Ray direction points from the camera toward each sampled surface point.
        ray_directions = -directions[facing]
        origins = np.broadcast_to(camera, ray_directions.shape)
        rays = np.column_stack((origins, ray_directions)).astype(np.float32)
        hit = context.scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        visible[facing] = np.isfinite(hit) & (hit >= distances[facing] - context.epsilon)
    return visible, camera


def self_occlusion(context, candidate, **options):
    """Estimate paper Eq. (1): occluded area divided by total mesh area."""
    visible, _ = visible_samples(context, candidate)
    ratio = 1 - float(visible.mean())
    return {"ambiguity": ratio, "occlusion_ratio": ratio,
            "visible_ratio": 1 - ratio, "visible_samples": int(visible.sum()),
            "total_samples": len(visible)}


def surface_normal_entropy(context, candidate, **options):
    """Estimate paper Eq. (2): occupancy ratio times visible-normal entropy."""
    visible, camera = visible_samples(context, candidate)
    if not visible.any():
        return {"ambiguity": None, "normal_entropy_bits": None,
                "occupied_bins": 0, "visible_samples": 0}
    normals = context.normals[visible]
    forward = camera / np.linalg.norm(camera)
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
