"""Run a bounded Blender shard and convert transparent renders to RGB/masks."""
import gzip
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from tools.cameras import HARNESS_LOOK_AT, direction


def _camera(azimuth, elevation, distance):
    # Blender's camera looks along local -Z. These columns are its actual
    # world-space local X/right, Y/up, and Z/back axes in the y-up dataset frame.
    back = direction(azimuth, elevation).astype(float)
    right = np.cross((0., 1., 0.), back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    rotation = np.column_stack((right, up, back))
    center = np.asarray(HARNESS_LOOK_AT) + distance * direction(azimuth, elevation)
    return rotation.tolist(), (-center @ rotation).tolist()


def _bbox(mask):
    box = mask.getbbox()
    return list(box) if box else []


def _convert(path, rgb_path, mask_path):
    if rgb_path.is_file() and mask_path.is_file():
        path.unlink(missing_ok=True)
        return
    rgb_pending = rgb_path.with_suffix(".tmp.png")
    mask_pending = mask_path.with_suffix(".tmp.png")
    with Image.open(path) as rendered:
        rgba = rendered.convert("RGBA")
        alpha = rgba.getchannel("A")
        mask = alpha.point(lambda value: 255 if value else 0)
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        background.convert("RGB").save(rgb_pending)
        mask.save(mask_pending)
    os.replace(rgb_pending, rgb_path)
    os.replace(mask_pending, mask_path)
    path.unlink()


def render_shard(run, number, uids, log):
    settings = run.config["render"]
    directory = run.shard_dir(number)
    directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(run.store / "hf-objaverse-v1" / "object-paths.json.gz", "rt") as source:
        paths = json.load(source)
    canonical = json.loads(run.canonical.read_text())["objects"]
    objects = run.objects()
    manifest = {"output": str(directory), "render": settings,
                "objects": [{"uid": uid, "path": paths[uid], "canonical": canonical[uid]}
                            for uid in uids]}
    manifest_path = directory / "blender-manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    binary = Path(settings["blender_binary"])
    if not binary.is_file():
        raise FileNotFoundError(f"Blender executable not found: {binary}")
    script = Path(__file__).with_name("blender_render.py")
    command = [str(binary), "-b", "-t", "2", "--python", str(script), "--",
               "--manifest", str(manifest_path)]
    with open(run.dir / "logs" / f"blender-{number:03d}.log", "a") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)

    focal = settings["height"] / (2 * math.tan(math.radians(settings["fov"]) / 2))
    index_path = directory / "index.jsonl"
    pending = index_path.with_suffix(".jsonl.tmp")
    with pending.open("w") as index:
        for uid in uids:
            for elevation in settings["elevations"]:
                for azimuth in settings["azimuths"]:
                    name = f"a{float(azimuth)}_e{float(elevation)}_o0.0"
                    folder = Path("samples") / uid
                    raw = directory / folder / f"{name}.rgba.png"
                    rgb = directory / folder / f"{name}.rgb.png"
                    modal = directory / folder / f"{name}.modal.png"
                    if raw.is_file():
                        _convert(raw, rgb, modal)
                    if not (rgb.is_file() and modal.is_file()):
                        raise FileNotFoundError(f"Incomplete Blender view: {uid} {name}")
                    with Image.open(modal) as mask:
                        area = int(np.count_nonzero(np.asarray(mask)))
                        box = _bbox(mask)
                        border = bool(box and (box[0] == 0 or box[1] == 0 or
                                               box[2] == settings["width"] or
                                               box[3] == settings["height"]))
                    rotation, translation = _camera(azimuth, elevation, settings["distance"])
                    row = {"uid": uid, "category": objects[uid]["category"],
                           "azimuth": float(azimuth), "elevation": float(elevation),
                           "distance": float(settings["distance"]), "fov": float(settings["fov"]),
                           "image_size": [settings["width"], settings["height"]],
                           "intrinsics": [focal, focal, (settings["width"] - 1) / 2,
                                          (settings["height"] - 1) / 2],
                           "amodal_area": area, "modal_area": area, "occlusion_ratio": 0.0,
                           "amodal_bbox": box, "modal_bbox": box, "occluder_uids": [],
                           "occluder_categories": [], "occluder_offset": 0.0,
                           "R": rotation, "T": translation, "truncated": border,
                           "sample_id": name,
                           "files": {"rgb": str(folder / rgb.name),
                                     "amodal": str(folder / modal.name),
                                     "modal": str(folder / modal.name)}}
                    index.write(json.dumps(row) + "\n")
    os.replace(pending, index_path)
    log(f"[render] Blender shard {number + 1}: {len(uids)} objects, "
        f"{len(uids) * len(settings['azimuths']) * len(settings['elevations'])} views")
