"""Run image retrieval, SAM3 preprocessing, SAM3D inference, and visualization."""
import argparse
import json
import os
from pathlib import Path
import subprocess


WORKSPACE = Path(__file__).resolve().parent


def load_config(path, *, image=None, query=None):
    import yaml

    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    if image is not None:
        config["image"] = str(Path(image).expanduser().resolve())
    if query is not None:
        config["query"] = query
    allowed = {"image", "query", "output_dir", "gpu_id", "preprocessing", "reconstruction", "visualization"}
    if config.keys() - allowed:
        raise ValueError(f"Unknown config settings: {sorted(config.keys() - allowed)}")
    for key in ("image", "query", "output_dir"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"{key} must be a nonempty string")

    def resolve(value):
        location = Path(value).expanduser()
        return str((path.parent / location).resolve())

    config["output_dir"] = resolve(config["output_dir"])
    config["image"] = resolve(config["image"])
    config["gpu_id"] = config.get("gpu_id", 0)
    if type(config["gpu_id"]) is not int or config["gpu_id"] < 0:
        raise ValueError("gpu_id must be a non-negative integer")
    if not Path(config["image"]).is_file():
        raise FileNotFoundError(f"Image does not exist: {config['image']}")
    defaults = {
        "preprocessing": {"checkpoint": "/data/checkpoints/sam3/sam3.pt", "threshold": 0.0,
                          "max_masks": 1,
                          "depth": None, "intrinsics": None},
        "reconstruction": {"checkpoint_dir": "/data/checkpoints/sam3d/hf", "views": 8,
                           "resolution": 512, "seed": 42},
        "visualization": {"enabled": True, "host": "0.0.0.0", "port": 7860},
    }
    for section, values in defaults.items():
        supplied = config.get(section, {})
        if not isinstance(supplied, dict) or supplied.keys() - values.keys():
            raise ValueError(f"Invalid {section} settings; expected keys: {list(values)}")
        config[section] = values | supplied
    prep = config["preprocessing"]
    recon = config["reconstruction"]
    viewer = config["visualization"]
    if not isinstance(prep["threshold"], (int, float)) or not 0 <= prep["threshold"] <= 1:
        raise ValueError("preprocessing.threshold must be between 0 and 1")
    if type(prep["max_masks"]) is not int or prep["max_masks"] < 1:
        raise ValueError("preprocessing.max_masks must be a positive integer")
    if (prep["depth"] is None) != (prep["intrinsics"] is None):
        raise ValueError("preprocessing.depth and intrinsics must be supplied together")
    if prep["intrinsics"] is not None:
        import math
        intrinsics = prep["intrinsics"]
        if (not isinstance(intrinsics, list) or len(intrinsics) != 4
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in intrinsics)
                or intrinsics[0] <= 0 or intrinsics[1] <= 0):
            raise ValueError("intrinsics must be [fx, fy, cx, cy] with positive focal lengths")
        prep["depth"] = resolve(prep["depth"])
    prep["checkpoint"] = resolve(prep["checkpoint"])
    recon["checkpoint_dir"] = resolve(recon["checkpoint_dir"])
    for key, minimum in (("views", 2), ("resolution", 64), ("seed", 0)):
        if type(recon[key]) is not int or recon[key] < minimum:
            raise ValueError(f"reconstruction.{key} must be an integer >= {minimum}")
    if type(viewer["enabled"]) is not bool:
        raise ValueError("visualization.enabled must be true or false")
    if type(viewer["port"]) is not int or not 1 <= viewer["port"] <= 65535:
        raise ValueError("visualization.port must be between 1 and 65535")
    if not isinstance(viewer["host"], str) or not viewer["host"].strip():
        raise ValueError("visualization.host must be a nonempty string")
    return config


def configure_gpu(config):
    """Restrict this worker to one Compose-exposed GPU before CUDA is imported."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_id"])
    print(f"Using configured GPU {config['gpu_id']} (visible as cuda:0)", flush=True)


def prepare(settings):
    """Retrieve and normalize the image, then generate masks and optional depth."""
    import numpy as np
    from PIL import Image

    config = json.loads(Path(settings).read_text())
    configure_gpu(config)
    output = Path(config["output_dir"])
    source = config["image"]
    print(f"Retrieving image: {source}", flush=True)
    image_path = output / "input.png"
    with Image.open(source) as source_image:
        image = source_image.convert("RGB")
    image.save(image_path)
    from reconstruction.segmentation import SAM3Preprocessor, depth_to_pointmap
    prep = config["preprocessing"]
    pointmap_path = None
    if prep["depth"] is not None:
        depth = np.load(prep["depth"])
        if depth.shape != (image.height, image.width):
            raise ValueError("Depth must match the RGB image dimensions")
        pointmap_path = output / "pointmap.npy"
        # Fast-SAM3D estimates intrinsics over the complete point map. Fill
        # zero/NaN background depth so its focal solver never divides by zero.
        np.save(pointmap_path, depth_to_pointmap(
            depth, *prep["intrinsics"], fill_invalid=True))
    print("Running SAM3 preprocessing", flush=True)
    segmenter = SAM3Preprocessor(checkpoint=prep["checkpoint"], threshold=prep["threshold"],
                                 max_masks=prep["max_masks"])
    masks = segmenter.masks_for_query(image, config["query"], output / "masks")
    # Record only this run's masks; old files must never become extra detections.
    config.update(prepared_image=str(image_path), masks=[str(mask) for mask in masks],
                  pointmap=str(pointmap_path) if pointmap_path else None)
    Path(settings).write_text(json.dumps(config, indent=2) + "\n")
    if not masks:
        raise RuntimeError(f"SAM3 found no objects matching {config['query']!r}")
    print(f"Saved {len(masks)} masks", flush=True)


def reconstruct(settings):
    """Reconstruct the selected objects, render views, then launch the viewer."""
    config = json.loads(Path(settings).read_text())
    configure_gpu(config)
    output = Path(config["output_dir"])
    recon = config["reconstruction"]
    from reconstruction.sam3d import SAM3D
    model = SAM3D(checkpoint_dir=recon["checkpoint_dir"])
    objects = []
    timings = []
    for index, mask in enumerate(config["masks"]):
        directory = output / f"object-{index:03d}"
        print(f"Reconstructing and rendering object {index + 1}/{len(config['masks'])}", flush=True)
        model.render_views(config["prepared_image"], mask, directory, views=recon["views"],
                           resolution=recon["resolution"], seed=recon["seed"],
                           pointmap=config["pointmap"])
        objects.append(directory / "object.ply")
        timing = getattr(model, "last_timing", None)
        if isinstance(timing, dict):
            record = {"object_index": index, **timing}
            timings.append(record)
            print(
                f"Object {index + 1} model reconstruction: {timing['model_seconds']:.2f}s "
                "(inference + Gaussian preparation; excludes PLY, views, and GIF)",
                flush=True,
            )
    if timings:
        (output / "reconstruction_timing.json").write_text(
            json.dumps({"model_load_included": False, "objects": timings}, indent=2) + "\n"
        )
    print(f"Saved reconstructions and visualizations to {output}", flush=True)
    if config["visualization"]["enabled"]:
        from reconstruction.sam3d import launch_viewer
        launch_viewer(objects, host=config["visualization"]["host"],
                      port=config["visualization"]["port"])


def run_pipeline(config_path, *, image=None, query=None, no_viewer=False, output_dir=None):
    config = load_config(config_path, image=image, query=query)
    if no_viewer:
        config["visualization"]["enabled"] = False
    if output_dir is not None:
        config["output_dir"] = str(Path(output_dir).resolve())
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    # Both interpreters consume the same resolved settings, independent of cwd.
    settings = output / "run.json"
    settings.write_text(json.dumps(config, indent=2) + "\n")
    for environment, function in (("core", "prepare"), ("sam3d", "reconstruct")):
        subprocess.run(
            ["sam3d-env", environment, "python", str(WORKSPACE / "main.py"),
             "--worker", function, "--config", str(settings)],
            cwd=WORKSPACE,
            check=True,
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=WORKSPACE / "configs" / "demo.yaml")
    parser.add_argument("--image", type=Path, help="Input frame; overrides the config")
    parser.add_argument("--query", help="SAM3 text prompt; overrides the config")
    parser.add_argument("--no-viewer", action="store_true", help="Save outputs and exit")
    parser.add_argument("--output-dir", type=Path, help="Override the output directory")
    parser.add_argument("--worker", choices=("prepare", "reconstruct"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        {"prepare": prepare, "reconstruct": reconstruct}[args.worker](args.config)
    else:
        run_pipeline(args.config, image=args.image, query=args.query,
                     no_viewer=args.no_viewer, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
