"""Generate object-instance camera rings in semantically annotated Habitat scenes.

See harness/HABITAT.md. Habitat is imported only when opening a scene.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np


def normalize(label):
    return " ".join(label.casefold().replace("_", " ").split())


def path_key(value):
    """Readable, traversal-safe keys with a hash to avoid slug collisions."""
    return (re.sub(r"[^a-z0-9_-]+", "-", value.casefold()).strip("-")[:60] or "item") + "-" + hashlib.sha256(value.encode()).hexdigest()[:10]


def discover_objects(semantic_scene, queries, match):
    objects, categories = [], set()
    for obj in semantic_scene.objects:
        if obj is None or obj.category is None:
            continue
        category = normalize(obj.category.name())
        categories.add(category)
        matched = [q for q in queries if (q == category if match == "exact" else q in category)]
        if not matched:
            continue
        # Use the renderer's semantic ID, never the object's list index.
        if not hasattr(obj, "semantic_id"):
            raise RuntimeError("Habitat must expose SemanticObject.semantic_id; upgrade Habitat-Sim.")
        center = np.asarray(obj.aabb.center, dtype=float)
        size = np.asarray(obj.aabb.sizes, dtype=float)
        if not np.isfinite(center).all() or not np.isfinite(size).all() or np.any(size <= 0):
            raise RuntimeError(f"Invalid semantic bounds for {obj.id}")
        objects.append(dict(object_id=str(obj.id), semantic_id=int(obj.semantic_id),
                            category=category, queries=matched,
                            center=center.tolist(), size=size.tolist()))
    return sorted(objects, key=lambda x: (x["category"], x["object_id"])), sorted(categories)


def camera_pose(center, radius, azimuth, elevation, offset):
    """Radius is horizontal XZ distance; offsets are right/up/forward meters.

    Return camera-to-world with local axes +X right, +Y up, -Z forward.
    The unshifted camera looks at center; translation does not re-aim it.
    """
    a, e = np.deg2rad([azimuth, elevation])
    backward = np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
    right = np.array([np.cos(a), 0., -np.sin(a)])
    up = np.cross(backward, right)
    rotation = np.column_stack((right, up, backward))
    base = np.asarray(center) + np.array([radius * np.sin(a), radius * np.tan(e), radius * np.cos(a)])
    position = base + rotation @ np.array([offset[0], offset[1], -offset[2]])
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, position
    return transform


def candidates(obj, args):
    azimuths = np.arange(0., 360., args.azimuth_step)
    for radius, elevation, azimuth, offset in itertools.product(
            args.radii, args.elevations, azimuths, args.offset):
        pose = camera_pose(obj["center"], radius, azimuth, elevation, offset)
        yield pose, dict(radius_m=radius, azimuth_deg=float(azimuth),
                         elevation_deg=elevation, offset_right_up_forward_m=list(offset))


def mask_stats(mask):
    y, x = np.nonzero(mask)
    if not len(x):
        return dict(visible_pixels=0, visible_fraction=0., bbox_xyxy=None,
                    centroid_xy=None, touches_border=False)
    h, w = mask.shape
    return dict(visible_pixels=len(x), visible_fraction=len(x) / mask.size,
                bbox_xyxy=[int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1],
                centroid_xy=[float(x.mean()), float(y.mean())],
                touches_border=bool(x.min() == 0 or y.min() == 0 or x.max() == w-1 or y.max() == h-1))


def navigable_camera(pathfinder, position, args):
    """Check a floor point without snapping/moving the requested camera."""
    foot = np.asarray(position) - [0., args.camera_height, 0.]
    return bool(pathfinder.is_navigable(foot, args.floor_tolerance))


class HabitatRenderer:
    def __init__(self, scene, args):
        try:
            import habitat_sim
        except ImportError as exc:
            raise RuntimeError("Run in an environment with Habitat-Sim, numpy, and Pillow; see harness/HABITAT.md") from exc
        self.habitat = habitat_sim
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_id = scene
        cfg.gpu_device_id = args.gpu_device_id
        cfg.enable_physics = False
        if args.scene_dataset:
            cfg.scene_dataset_config_file = args.scene_dataset
        specs = []
        types = [("rgb", habitat_sim.SensorType.COLOR), ("semantic", habitat_sim.SensorType.SEMANTIC)]
        if args.save_depth:
            types.append(("depth", habitat_sim.SensorType.DEPTH))
        for name, kind in types:
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid, spec.sensor_type = name, kind
            spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            spec.resolution = [args.height, args.width]
            spec.position = [0., 0., 0.]
            spec.hfov, spec.near, spec.far = args.hfov, args.near, args.far
            specs.append(spec)
        agent = habitat_sim.agent.AgentConfiguration()
        agent.sensor_specifications = specs
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))

    def render(self, pose):
        import quaternion
        state = self.habitat.AgentState()
        state.position = pose[:3, 3]
        state.rotation = quaternion.from_rotation_matrix(pose[:3, :3])
        self.sim.get_agent(0).set_state(state, reset_sensors=True)
        return self.sim.get_sensor_observations()

    def close(self):
        self.sim.close()


def generate(args, renderer_factory=HabitatRenderer):
    from PIL import Image

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f"Output directory must be empty: {out}")
    focal = args.width / (2 * math.tan(math.radians(args.hfov) / 2))
    intrinsics = [focal, focal, (args.width-1)/2, (args.height-1)/2]
    config = dict(vars(args), schema_version=1,
                  convention="meters; world +Y up; camera +X right, +Y up, -Z forward; camera_to_world row-major",
                  intrinsics=intrinsics)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    summary = dict(status="running", matched_objects=0, saved=0, rejected={}, scenes=[])
    rejected = Counter()
    try:
        with (out / "index.jsonl").open("w") as index, (out / "objects.jsonl").open("w") as objects_file:
            for scene in args.scenes:
                renderer = renderer_factory(scene, args)
                try:
                    sim = renderer.sim
                    if args.require_navigable and not sim.pathfinder.is_loaded:
                        raise RuntimeError(f"No navmesh loaded for {scene}")
                    objects, categories = discover_objects(sim.semantic_scene, args.queries, args.match)
                    if not categories:
                        raise RuntimeError(f"No semantic objects in {scene}; provide semantic assets and the scene dataset config")
                    scene_summary = dict(scene=scene, categories=categories, matched_objects=len(objects), saved=0)
                    summary["scenes"].append(scene_summary)
                    for obj in objects:
                        summary["matched_objects"] += 1
                        object_record = dict(obj, scene=scene)
                        objects_file.write(json.dumps(object_record) + "\n")
                        folder = Path("samples") / path_key(obj["category"]) / path_key(scene) / path_key(obj["object_id"])
                        for number, (pose, sampling) in enumerate(candidates(obj, args)):
                            position = pose[:3, 3]
                            # Even unconstrained views must not be inside the target's AABB.
                            if np.all(np.abs(position - obj["center"]) <= np.asarray(obj["size"]) / 2):
                                rejected["inside_target_bounds"] += 1
                                continue
                            if args.require_navigable and not navigable_camera(sim.pathfinder, position, args):
                                rejected["not_navigable"] += 1
                                continue
                            observations = renderer.render(pose)
                            mask = np.asarray(observations["semantic"]) == obj["semantic_id"]
                            stats = mask_stats(mask)
                            if stats["visible_pixels"] < args.min_pixels or stats["visible_fraction"] < args.min_fraction:
                                rejected["target_not_visible_enough"] += 1
                                continue
                            if args.reject_border and stats["touches_border"]:
                                rejected["touches_border"] += 1
                                continue
                            (out / folder).mkdir(parents=True, exist_ok=True)
                            stem = folder / f"{number:07d}"
                            files = dict(rgb=f"{stem}.rgb.png", mask=f"{stem}.mask.png")
                            Image.fromarray(np.asarray(observations["rgb"], dtype=np.uint8)[..., :3]).save(out / files["rgb"])
                            Image.fromarray(mask.astype(np.uint8) * 255).save(out / files["mask"])
                            if args.save_depth:
                                files["depth"] = f"{stem}.depth.npy"
                                np.save(out / files["depth"], np.asarray(observations["depth"], dtype=np.float32))
                            record = dict(object_record, sampling=sampling, camera_to_world=pose.tolist(),
                                          actual_center_distance_m=float(np.linalg.norm(position - obj["center"])),
                                          intrinsics=intrinsics, resolution_hw=[args.height, args.width],
                                          files=files, **stats)
                            index.write(json.dumps(record) + "\n")
                            index.flush()
                            summary["saved"] += 1
                            scene_summary["saved"] += 1
                    print(f"{scene}: {len(objects)} matched objects, {scene_summary['saved']} images", flush=True)
                finally:
                    renderer.close()
        if not summary["saved"]:
            raise RuntimeError("No images saved. See summary.json for available categories and rejected views.")
        summary["status"] = "complete"
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        raise
    finally:
        summary["rejected"] = dict(rejected)
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--scenes", nargs="+", required=True, help="scene asset paths or dataset scene handles")
    p.add_argument("--scene-dataset", help="Habitat .scene_dataset_config.json with semantic assets")
    p.add_argument("--queries", nargs="+", required=True, help="semantic category labels, e.g. bed chair")
    p.add_argument("--match", choices=["exact", "substring"], default="exact")
    p.add_argument("--out", required=True, help="new or empty output directory")
    p.add_argument("--radii", type=float, nargs="+", default=[1.5, 2., 3.])
    p.add_argument("--azimuth-step", type=float, default=30.)
    p.add_argument("--elevations", type=float, nargs="+", default=[0., 15., 30.])
    p.add_argument("--offset", type=float, nargs=3, action="append", metavar=("RIGHT", "UP", "FORWARD"),
                   help="repeat for each camera translation in meters; default: 0 0 0")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--hfov", type=float, default=90.)
    p.add_argument("--near", type=float, default=0.05)
    p.add_argument("--far", type=float, default=100.)
    p.add_argument("--min-pixels", type=int, default=100)
    p.add_argument("--min-fraction", type=float, default=0.001)
    p.add_argument("--reject-border", action="store_true")
    p.add_argument("--save-depth", action="store_true")
    p.add_argument("--require-navigable", action="store_true")
    p.add_argument("--camera-height", type=float, default=1.5, help="floor probe distance below camera, meters")
    p.add_argument("--floor-tolerance", type=float, default=0.5)
    p.add_argument("--gpu-device-id", type=int, default=0)
    args = p.parse_args(argv)
    args.offset = args.offset or [[0., 0., 0.]]
    numeric = [*args.radii, *args.elevations, *itertools.chain.from_iterable(args.offset),
               args.azimuth_step, args.hfov, args.near, args.far, args.min_fraction,
               args.camera_height, args.floor_tolerance]
    if not all(math.isfinite(v) for v in numeric):
        p.error("all numeric settings must be finite")
    if any(r <= 0 for r in args.radii) or not 0 < args.azimuth_step <= 360:
        p.error("radii must be positive and azimuth-step must be in (0, 360]")
    if any(abs(e) >= 89 for e in args.elevations):
        p.error("elevations must be strictly between -89 and 89 degrees")
    if min(args.width, args.height) < 1 or not 0 < args.hfov < 180 or not 0 < args.near < args.far:
        p.error("invalid image dimensions, FOV, or clipping planes")
    if args.min_pixels < 1 or not 0 <= args.min_fraction <= 1:
        p.error("min-pixels must be >= 1 and min-fraction must be in [0, 1]")
    if args.camera_height <= 0 or args.floor_tolerance < 0 or args.gpu_device_id < 0:
        p.error("invalid camera height, floor tolerance, or GPU ID")
    args.queries = sorted(set(normalize(q) for q in args.queries))
    if not all(args.queries):
        p.error("queries cannot be empty")
    args.scenes = list(dict.fromkeys(args.scenes))
    return args


def main():
    args = parse_args()
    try:
        summary = generate(args)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Saved {summary['saved']} images to {args.out}")


if __name__ == "__main__":
    main()
