"""Orchestrate the SAM3 viewpoint-sensitivity experiment across two envs.

Three stages, mirroring main.py's two-interpreter subprocess pattern:

  1. render   (sam3d env): reuse harness.build_dataset to render an az x el grid
               of unoccluded views with pixel-exact ground-truth masks.
  2. query    (core  env): score every vocabulary class with SAM3 and make a
                           top-1 prediction for each rendered view.
  3. analyze  (CPU only) : average top-1 accuracy per (az, el) and emit the
                           paper-style heatmaps plus detection diagnostics.

Run inside the container, where both envs and the GPU live:
  sam3d-env core python -m experiments.viewpoint_sensitivity.run

Azimuth here is the harness's *geometric* (PCA) frame, not a semantic front.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]

# Every settings.json key the stages understand; anything else is a typo.
ALLOWED = {"output_dir", "gpu_id", "root", "canonical", "categories",
           "objects_per_category",
           "azimuths", "elevations", "distance", "image_size", "detection",
           "checkpoint", "sam3d_collection", "require_upright_verified", "use_reviewed_cohort"}


def load_config(path):
    """Validate and resolve the experiment YAML, mirroring main.py:load_config."""
    import yaml

    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    if config.keys() - ALLOWED:
        raise ValueError(f"Unknown config settings: {sorted(config.keys() - ALLOWED)}")

    def resolve(value):
        location = Path(value).expanduser()
        return str((path.parent / location).resolve())

    # output_dir/checkpoint/canonical resolve against the config directory;
    # root stays as given (it names a container mount, not a host path).
    if not isinstance(config.get("output_dir"), str) or not config["output_dir"].strip():
        raise ValueError("output_dir must be a nonempty string")
    config["output_dir"] = resolve(config["output_dir"])
    config["gpu_id"] = config.get("gpu_id", 0)
    if type(config["gpu_id"]) is not int or config["gpu_id"] < 0:
        raise ValueError("gpu_id must be a non-negative integer")

    config["root"] = config.get("root", "/data/objaverse_nas")
    if not isinstance(config["root"], str) or not config["root"].strip():
        raise ValueError("root must be a nonempty string")
    canonical = config.get("canonical") or os.path.join(config["root"], "canonical.json")
    # A relative canonical path resolves against the config dir; an absolute one
    # (the common case: it sits on the read-only slice) passes through unchanged.
    config["canonical"] = canonical if os.path.isabs(canonical) else resolve(canonical)
    config["checkpoint"] = resolve(config.get("checkpoint", "/data/checkpoints/sam3/sam3.pt"))

    categories = config.get("categories")
    if (not isinstance(categories, list) or len(categories) < 2
            or not all(isinstance(c, str) and c.strip() for c in categories)):
        raise ValueError("categories must contain at least two nonempty strings")
    if len(set(categories)) != len(categories):
        raise ValueError("categories must not contain duplicates")

    objects_per_category = config.get("objects_per_category", 10)
    if type(objects_per_category) is not int or objects_per_category < 1:
        raise ValueError("objects_per_category must be a positive integer")
    config["objects_per_category"] = objects_per_category

    for key in ("azimuths", "elevations"):
        grid = config.get(key)
        if (not isinstance(grid, list) or not grid
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in grid)):
            raise ValueError(f"{key} must be a nonempty list of numbers")
        config[key] = [float(v) for v in grid]
        if len(set(config[key])) != len(config[key]):
            raise ValueError(f"{key} must not contain duplicates")
    if len({a % 360.0 for a in config["azimuths"]}) != len(config["azimuths"]):
        raise ValueError("azimuths must be unique modulo 360 degrees")
    if any(not -90.0 < e < 90.0 for e in config["elevations"]):
        raise ValueError("elevations must be strictly between -90 and 90 degrees")

    distance = config.get("distance", 2.2)
    if not isinstance(distance, (int, float)) or isinstance(distance, bool) or distance <= 0:
        raise ValueError("distance must be a positive number")
    config["distance"] = float(distance)
    image_size = config.get("image_size", 512)
    if type(image_size) is not int or image_size < 16:
        raise ValueError("image_size must be an integer >= 16")
    config["image_size"] = image_size

    detection = config.get("detection", {})
    if not isinstance(detection, dict) or detection.keys() - {"score_threshold", "iou_threshold"}:
        raise ValueError("detection must have keys score_threshold, iou_threshold")
    detection = {"score_threshold": 0.5, "iou_threshold": 0.5} | detection
    for key in ("score_threshold", "iou_threshold"):
        value = detection[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
            raise ValueError(f"detection.{key} must be between 0 and 1")
    config["detection"] = detection

    collection = config.get("sam3d_collection", {})
    if (not isinstance(collection, dict)
            or collection.keys() - {"viewpoints_per_band"}):
        raise ValueError("sam3d_collection must have only viewpoints_per_band")
    viewpoints_per_band = collection.get("viewpoints_per_band", 3)
    if type(viewpoints_per_band) is not int or viewpoints_per_band < 1:
        raise ValueError("sam3d_collection.viewpoints_per_band must be a positive integer")
    if 2 * viewpoints_per_band > len(config["azimuths"]) * len(config["elevations"]):
        raise ValueError(
            "sam3d_collection needs at least twice viewpoints_per_band distinct views")
    config["sam3d_collection"] = {"viewpoints_per_band": viewpoints_per_band}
    if type(config.get("require_upright_verified", False)) is not bool:
        raise ValueError("require_upright_verified must be a boolean")
    if type(config.get("use_reviewed_cohort", False)) is not bool:
        raise ValueError("use_reviewed_cohort must be a boolean")
    if config.get("use_reviewed_cohort") and not config.get("require_upright_verified"):
        raise ValueError("use_reviewed_cohort requires require_upright_verified")

    return config


def _run_stage(argv, gpu_id):
    """Run one stage as a subprocess, restricting it to the configured GPU."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    print(f"[viewpoint] $ {' '.join(argv)}", flush=True)
    subprocess.run(argv, cwd=WORKSPACE, env=env, check=True)


def _canonicalize_command(config):
    """The exact command that regenerates a missing canonical.json (CPU only)."""
    return ["sam3d-env", "sam3d", "python", "-m", "harness.canonicalize",
            "--root", config["root"], "--out", config["canonical"]]


def run(config_path, *, output_dir=None, skip_render=False, stage=None,
        canonicalize=False, restart_query=False):
    """Resolve settings, then drive the requested stages in order."""
    config = load_config(config_path)
    if output_dir is not None:
        config["output_dir"] = str(Path(output_dir).resolve())
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    render_dir = output / "render"
    config["render_dir"] = str(render_dir)
    if config.get("use_reviewed_cohort"):
        objects = json.loads(Path(config["canonical"]).read_text())["objects"]
        counts = {cat: 0 for cat in config["categories"]}
        for uid, record in objects.items():
            if record["category"] in counts:
                if record.get("upright_verification", {}).get("status") != "verified":
                    raise ValueError(f"Unverified cohort object {uid}")
                if record["face_count"] > 300_000:
                    raise ValueError(f"Reviewed object {uid} exceeds renderer face limit")
                counts[record["category"]] += 1
        if not all(counts.values()):
            raise ValueError(f"Reviewed cohort has empty categories: {counts}")
        config["category_object_counts"] = counts
    # Every stage reads the same resolved settings, independent of cwd.
    settings = output / "settings.json"
    settings.write_text(json.dumps(config, indent=2) + "\n")

    want_render = stage in (None, "render") and not skip_render
    want_query = stage in (None, "query")
    want_analyze = stage in (None, "analyze")

    if want_render:
        canonical = Path(config["canonical"])
        if not canonical.is_file():
            if canonicalize:
                # CPU-only pre-stage; no GPU needed to derive canonical frames.
                _run_stage(_canonicalize_command(config), config["gpu_id"])
            else:
                command = " ".join(_canonicalize_command(config))
                raise FileNotFoundError(
                    f"Missing canonical.json: {canonical}. Re-run with --canonicalize, "
                    f"or generate it first:\n  {command}")
        _run_stage([
            "sam3d-env", "sam3d", "python", "-m", "harness.build_dataset",
            "--root", config["root"], "--canonical", config["canonical"],
            "--out", str(render_dir),
            "--categories", *config["categories"],
            *(["--limit", str(sum(config["category_object_counts"].values()))]
              if config.get("use_reviewed_cohort") else
              ["--objects-per-category", str(config["objects_per_category"])]),
            "--azimuths", *[str(a) for a in config["azimuths"]],
            "--elevations", *[str(e) for e in config["elevations"]],
            "--occlusion-levels", "0.0",
            "--distance", str(config["distance"]),
            "--image-size", str(config["image_size"]),
            *(["--require-upright-verified"] if config.get("require_upright_verified") else []),
        ], config["gpu_id"])

    if want_query:
        query_command = [
            "sam3d-env", "core", "python", "-m",
            "experiments.viewpoint_sensitivity.query_stage",
            "--config", str(settings),
        ]
        if restart_query:
            query_command.append("--restart")
        _run_stage(query_command, config["gpu_id"])

    if want_analyze:
        _run_stage([
            "sam3d-env", "core", "python", "-m",
            "experiments.viewpoint_sensitivity.analyze",
            "--config", str(settings),
        ], config["gpu_id"])

    print(f"[viewpoint] done -> {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parent / "config.yaml")
    parser.add_argument("--output-dir", type=Path, help="Override the output directory")
    parser.add_argument("--skip-render", action="store_true",
                        help="Reuse an existing render/index.jsonl")
    parser.add_argument("--stage", choices=("render", "query", "analyze"),
                        help="Run a single stage instead of all three")
    parser.add_argument("--canonicalize", action="store_true",
                        help="Generate canonical.json first if it is missing (CPU only)")
    parser.add_argument("--restart-query", action="store_true",
                        help="Discard existing query progress in experiment.sqlite")
    args = parser.parse_args()
    run(args.config, output_dir=args.output_dir, skip_render=args.skip_render,
        stage=args.stage, canonicalize=args.canonicalize,
        restart_query=args.restart_query)


if __name__ == "__main__":
    main()
