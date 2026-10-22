"""Shape error of a SAM3D reconstruction against the ground-truth mesh.

The reconstruction (``mesh.npz``) lives in SAM3D's input-camera frame: the
input camera on -Z looking at +Z, +Y its up (``tools.cameras``). The ground-truth
mesh lives in the reviewed canonical frame, seen by the input camera at
``direction(azimuth, elevation)`` from ``HARNESS_LOOK_AT``. Rotating it into
the camera frame leaves only the single-image scale and depth ambiguity, so
both are centered on their bounding box and scaled to a largest extent of 1,
then a rigid ICP refines the pose (kept only if it lowers the Chamfer distance;
a free scale lets ICP shrink a shape into one part of the other). Extent
normalization alone leaves a size mismatch whenever one shape's largest extent
is wrong (a missing or extra part), so the ``*_scaled`` scores also search a
bounded isotropic scale (``SCALES``, too narrow to collapse) and keep the best
Chamfer.

Reported (normalized units, largest extent 1):
- ``chamfer``: mean of the two nearest-neighbour distance means, best pose.
- ``fscore_<tau>``: F-score of surface samples within ``tau``, best pose;
  ``1 - fscore`` is the error.
- ``*_camera``: the same before ICP (pose from the camera alone).
- ``icp_degrees``: rotation ICP added; large values mean the shapes did not
  match in the camera pose (wrong shape, or a symmetry flip).
- ``*_scaled``, ``scale``: the best pose over ``SCALES`` of the reconstruction.
"""
import math

import numpy as np

from tools.cameras import direction

TAUS = (0.02, 0.05)
SCALES = np.geomspace(0.7, 1 / 0.7, 9)


def camera_rotation(azimuth, elevation):
    """Rotation from the canonical frame into SAM3D's input-camera frame.

    Rows are the camera's (-right, up, forward) in the canonical frame: forward
    maps to +Z, up to +Y, and right (forward x up) to -X.
    """
    forward = -direction(azimuth, elevation).astype(np.float64)
    up = np.array([0.0, 1.0, 0.0]) - forward[1] * forward
    up /= np.linalg.norm(up)
    right = np.cross(forward, up)
    return np.stack([-right, up, forward])


def sample_surface(vertices, faces, count, generator):
    """``count`` area-weighted points on a triangle mesh."""
    triangles = np.asarray(vertices, np.float64)[np.asarray(faces)]
    areas = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                          triangles[:, 2] - triangles[:, 0]), axis=1)
    chosen = generator.choice(len(triangles), size=count, p=areas / areas.sum())
    u, v = generator.random((2, count))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = triangles[chosen]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


def normalize(points):
    """Center on the bounding box and scale to a largest extent of 1."""
    low, high = points.min(axis=0), points.max(axis=0)
    return (points - (low + high) / 2) / (high - low).max()


def rigid(source, target):
    """Kabsch: rotation, translation minimizing |R source + t - target|^2."""
    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    a, b = source - mu_s, target - mu_t
    u, _, vt = np.linalg.svd(b.T @ a)
    sign = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ sign @ vt
    return rotation, mu_t - rotation @ mu_s


def icp(source, target, iterations=40):
    """Rigid ICP of ``source`` onto ``target``; returns aligned source and total rotation."""
    from scipy.spatial import cKDTree

    tree = cKDTree(target)
    current, total = source, np.eye(3)
    for _ in range(iterations):
        _, nearest = tree.query(current)
        rotation, translation = rigid(current, target[nearest])
        current = current @ rotation.T + translation
        total = rotation @ total
    return current, total


def distances(a, b):
    from scipy.spatial import cKDTree

    return cKDTree(b).query(a)[0], cKDTree(a).query(b)[0]


def scores(predicted, truth, suffix=""):
    to_truth, to_predicted = distances(predicted, truth)
    result = {f"chamfer{suffix}": float((to_truth.mean() + to_predicted.mean()) / 2)}
    for tau in TAUS:
        precision, recall = (to_truth < tau).mean(), (to_predicted < tau).mean()
        result[f"fscore_{tau:g}{suffix}"] = float(
            0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return result


def shape_error(predicted_mesh, truth_mesh, azimuth, elevation, samples=20000, seed=0):
    """Errors of ``predicted_mesh`` (SAM3D frame) against ``truth_mesh`` (canonical frame).

    Meshes are (vertices, faces); see the module docstring for the outputs.
    """
    generator = np.random.default_rng(seed)
    predicted = normalize(sample_surface(*predicted_mesh, samples, generator))
    truth = sample_surface(*truth_mesh, samples, generator) @ camera_rotation(azimuth, elevation).T
    truth = normalize(truth)
    aligned, rotation = icp(predicted, truth)
    camera, refined = scores(predicted, truth, "_camera"), scores(aligned, truth)
    if refined["chamfer"] > camera["chamfer_camera"]:
        refined, rotation = scores(predicted, truth), np.eye(3)
    angle = math.degrees(math.acos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
    scaled, best = refined, 1.0
    for scale in SCALES:
        candidate = scores(icp(predicted * scale, truth)[0], truth)
        if candidate["chamfer"] < scaled["chamfer"]:
            scaled, best = candidate, float(scale)
    return {**refined, **camera, "icp_degrees": angle,
            **{f"{key}_scaled": value for key, value in scaled.items()}, "scale": best}
