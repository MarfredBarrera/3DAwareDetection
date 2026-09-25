"""Reconstruct one RGB image and sample a FastSAM3D azimuth/elevation view grid."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess

from PIL import Image
import yaml

from .metrics import metric_options, score_views
from .visualize import render_layout


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def load_config(path):
    """Validate YAML and resolve paths relative to its location."""
    path = Path(path).expanduser().resolve()
    config = yaml.safe_load(path.read_text())
    allowed = {"image", "query", "mask", "output_dir", "gpu_id", "preprocessing",
               "reconstruction", "scoring"}
    if not isinstance(config, dict) or config.keys() - allowed:
        raise ValueError(f"Expected a YAML mapping with keys: {sorted(allowed)}")

    def resolve(value, name, required=True):
        if value is None and not required:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty path")
        return str((path.parent / Path(value).expanduser()).resolve())

    image = resolve(config.get("image"), "image")
    output_dir = resolve(config.get("output_dir"), "output_dir")
    mask = resolve(config.get("mask"), "mask", required=False)
    query = config.get("query")
    if query is not None and (not isinstance(query, str) or not query.strip()):
        raise ValueError("query must be a nonempty string")
    if bool(query) == bool(mask):
        raise ValueError("Supply exactly one of query or mask")
    for name, location in (("image", image), ("mask", mask)):
        if location and not Path(location).is_file():
            raise FileNotFoundError(f"{name} does not exist: {location}")

    prep = config.get("preprocessing", {})
    recon = config.get("reconstruction", {})
    if not isinstance(prep, dict) or prep.keys() - {"checkpoint", "threshold"}:
        raise ValueError("preprocessing accepts checkpoint and threshold")
    if not isinstance(recon, dict) or recon.keys() - {"checkpoint_dir", "resolution", "seed", "azimuths", "elevations"}:
        raise ValueError("reconstruction accepts checkpoint_dir, resolution, seed, azimuths, elevations")
    scoring = config.get("scoring", {})
    if not isinstance(scoring, dict) or scoring.keys() - {"metric", "options"}:
        raise ValueError("scoring accepts metric and options")
    metric = scoring.get("metric", "visual_entropy")
    options = metric_options(metric, scoring.get("options"))
    gpu_id = config.get("gpu_id", 0)
    threshold = prep.get("threshold", 0.3)
    resolution = recon.get("resolution", 512)
    seed = recon.get("seed", 42)
    if type(gpu_id) is not int or gpu_id < 0:
        raise ValueError("gpu_id must be a nonnegative integer")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("preprocessing.threshold must be between 0 and 1")
    if type(resolution) is not int or resolution < 64:
        raise ValueError("reconstruction.resolution must be an integer >= 64")
    if type(seed) is not int or seed < 0:
        raise ValueError("reconstruction.seed must be a nonnegative integer")

    def angles(name, default):
        values = recon.get(name, default)
        valid = (lambda v: 0 <= v < 360) if name == "azimuths" else (lambda v: -90 < v < 90)
        if (not isinstance(values, list) or not values or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                    not math.isfinite(v) or not valid(v) for v in values) or
                len(set(values)) != len(values)):
            raise ValueError(f"reconstruction.{name} must contain unique valid finite angles")
        return values

    return {
        "image": image, "mask": mask, "query": query, "output_dir": output_dir,
        "metric": metric, "metric_options": options, "gpu_id": gpu_id,
        "sam3_checkpoint": resolve(prep.get("checkpoint", "/data/checkpoints/sam3/sam3.pt"),
                                   "preprocessing.checkpoint"),
        "threshold": threshold,
        "sam3d_checkpoint": resolve(recon.get("checkpoint_dir", "/data/checkpoints/sam3d/hf"),
                                     "reconstruction.checkpoint_dir"),
        "resolution": resolution, "seed": seed,
        "azimuths": angles("azimuths", list(range(0, 360, 30))),
        "elevations": angles("elevations", [0, 15, 30, 45, 60]),
    }


def segment(settings_path):
    settings = json.loads(Path(settings_path).read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(settings["gpu_id"])
    from reconstruction.segmentation import SAM3Preprocessor

    with Image.open(settings["image"]) as source:
        image = source.convert("RGB")
    model = SAM3Preprocessor(checkpoint=settings["sam3_checkpoint"],
                             threshold=settings["threshold"], max_masks=1)
    masks = model.masks_for_query(image, settings["query"], Path(settings["output_dir"]) / "masks")
    if not masks:
        raise RuntimeError(f"SAM3 found no object matching {settings['query']!r}")
    settings["mask"] = str(masks[0])
    Path(settings_path).write_text(json.dumps(settings, indent=2) + "\n")


def reconstruct(settings_path):
    settings = json.loads(Path(settings_path).read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(settings["gpu_id"])
    from reconstruction.sam3d import SAM3D

    output = Path(settings["output_dir"])
    model = SAM3D(checkpoint_dir=settings["sam3d_checkpoint"])
    candidates = model.render_views(
        settings["image"], settings["mask"], output, resolution=settings["resolution"],
        seed=settings["seed"], view_grid=(settings["azimuths"], settings["elevations"]))
    manifest = {
        "source_image": "input.png", "query": settings["query"],
        "azimuths": settings["azimuths"], "elevations": settings["elevations"],
        "camera_frame": "FastSAM3D local azimuth/elevation degrees",
        "candidates": [{**item, "image": "views/" + item["image"],
                        "mask": "views/" + item["mask"]} for item in candidates],
    }
    (output / "candidates.json").write_text(json.dumps(manifest, indent=2) + "\n")


def score(settings_path):
    settings = (load_config(settings_path) if Path(settings_path).suffix in {".yaml", ".yml"}
                else json.loads(Path(settings_path).read_text()))
    score_views(settings["output_dir"], settings["metric"], options=settings["metric_options"])


def run(config_path, refresh_layout=False):
    settings = load_config(config_path)
    output = Path(settings["output_dir"])
    if refresh_layout:
        subprocess.run(["sam3d-env", "sam3d", "python", "-m", "viewpoint_optimization",
                        "--worker", "score", "--settings", str(Path(config_path).resolve())],
                       cwd=WORKSPACE, check=True)
        return render_layout(output, output / f"scores_{settings['metric']}.json")
    output.mkdir(parents=True, exist_ok=True)
    with Image.open(settings["image"]) as source:
        image = source.convert("RGB")
        image.save(output / "input.png")
    settings["image"] = str(output / "input.png")
    if settings["mask"]:
        with Image.open(settings["mask"]) as source:
            mask = source.convert("L")
            if mask.size != image.size or not mask.getbbox():
                raise ValueError("mask must match the input image and contain foreground pixels")
            mask.save(output / "mask.png")
        settings["mask"] = str(output / "mask.png")
    settings_path = output / "run.json"
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    if not settings["mask"]:
        subprocess.run(["sam3d-env", "core", "python", "-m", "viewpoint_optimization",
                        "--worker", "segment", "--settings", str(settings_path)],
                       cwd=WORKSPACE, check=True)
    subprocess.run(["sam3d-env", "sam3d", "python", "-m", "viewpoint_optimization",
                    "--worker", "reconstruct", "--settings", str(settings_path)],
                   cwd=WORKSPACE, check=True)
    subprocess.run(["sam3d-env", "sam3d", "python", "-m", "viewpoint_optimization",
                    "--worker", "score", "--settings", str(settings_path)],
                   cwd=WORKSPACE, check=True)
    return render_layout(output, output / f"scores_{settings['metric']}.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--refresh-layout", action="store_true",
                        help="Rescore existing renders and regenerate the plot")
    parser.add_argument("--worker", choices=("segment", "reconstruct", "score"), help=argparse.SUPPRESS)
    parser.add_argument("--settings", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.settings is None:
            parser.error("--settings is required for a worker")
        {"segment": segment, "reconstruct": reconstruct, "score": score}[args.worker](args.settings)
    else:
        print(f"Saved {run(args.config, args.refresh_layout)}")


if __name__ == "__main__":
    main()
