"""Blender side of the ViewActive RGB renderer.

Run by ``viewactive.blender``. This file deliberately uses only Blender's
bundled Python modules so the render worker needs no model environment.
"""
import argparse
import json
import math
import struct
import sys
import tempfile
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def complete_png(path):
    if not path.is_file() or path.stat().st_size < 20:
        return False
    with path.open("rb") as stream:
        if stream.read(8) != b"\x89PNG\r\n\x1a\n":
            return False
        stream.seek(-12, 2)
        return stream.read(8) == b"\x00\x00\x00\x00IEND"


def setup(width, height, fov, samples):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 2
    scene.render.image_settings.compression = 15
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = samples
    world = bpy.data.worlds.new("ViewActive world")
    world.use_nodes = True
    world.node_tree.nodes.get("Background").inputs[0].default_value = (1, 1, 1, 1)
    world.node_tree.nodes.get("Background").inputs[1].default_value = 0.8
    scene.world = world
    for name, location, power, size in (
        ("key", (3, -4, 5), 650, 5),
        ("fill", (-4, 2, 3), 450, 5),
        ("rim", (1, 4, 4), 300, 4),
    ):
        data = bpy.data.lights.new(name, "AREA")
        data.energy = power
        data.shape = "DISK"
        data.size = size
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (Vector((0, 0, 0)) - obj.location).to_track_quat("-Z", "Y").to_euler()
    camera_data = bpy.data.cameras.new("camera")
    camera_data.type = "PERSP"
    camera_data.sensor_fit = "VERTICAL"
    camera_data.angle = math.radians(fov)
    camera = bpy.data.objects.new("camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    return scene, camera


def canonical_transform(record):
    # Blender's glTF importer maps the file's +Y up frame to Blender's +Z up.
    # Conjugate the canonical y-up transform by that same axis change.
    axis = Matrix.Rotation(math.pi / 2, 4, "X")
    canonical = Matrix.Identity(4)
    for i, row in enumerate(record["rotation"]):
        for j, value in enumerate(row):
            canonical[i][j] = value * record["scale"]
        canonical[i][3] = record["translation"][i]
    return axis @ canonical @ axis.inverted(), axis


def rest_pose_glb(path, output):
    # Blender poses animated GLBs at their first keyframe, but the canonical
    # frame and labels come from trimesh's rest pose. Import a copy without
    # animations so both see the same geometry.
    data = Path(path).read_bytes()
    length, kind = struct.unpack_from("<II", data, 12)
    document = json.loads(data[20:20 + length])
    if kind != 0x4E4F534A or not document.pop("animations", None):
        return path
    chunk = json.dumps(document, separators=(",", ":")).encode()
    chunk += b" " * (-len(chunk) % 4)
    rest = data[20 + length:]
    header = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(chunk) + len(rest))
    output.write_bytes(header + struct.pack("<II", len(chunk), kind) + chunk + rest)
    return str(output)


def render_object(scene, camera, entry, config, output):
    before = set(bpy.data.objects)
    with tempfile.TemporaryDirectory() as scratch:
        bpy.ops.import_scene.gltf(filepath=rest_pose_glb(entry["path"], Path(scratch) / "rest.glb"))
    imported = set(bpy.data.objects) - before
    transform, axis = canonical_transform(entry["canonical"])
    for obj in imported:
        if obj.parent not in imported:
            obj.matrix_world = transform @ obj.matrix_world
    target = axis @ Vector((0.0, 0.35, 0.0, 1.0))
    for elevation in config["elevations"]:
        elev = math.radians(elevation)
        for azimuth in config["azimuths"]:
            name = f"a{float(azimuth)}_e{float(elevation)}_o0.0"
            relative = Path("samples") / entry["uid"] / f"{name}.rgba.png"
            path = output / relative
            rgb = path.with_name(f"{name}.rgb.png")
            mask = path.with_name(f"{name}.modal.png")
            if (complete_png(rgb) and complete_png(mask)) or complete_png(path):
                continue
            az = math.radians(azimuth)
            direction = Vector((math.sin(az) * math.cos(elev), math.sin(elev),
                                math.cos(az) * math.cos(elev), 0.0))
            camera.location = (target + config["distance"] * (axis @ direction)).to_3d()
            camera.rotation_euler = (target.to_3d() - camera.location).to_track_quat("-Z", "Y").to_euler()
            path.parent.mkdir(parents=True, exist_ok=True)
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in imported:
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for block in list(collection):
            if block.users == 0 and block.name != "Render Result":
                collection.remove(block)


def main():
    args = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parsed = parser.parse_args(args)
    manifest = json.loads(parsed.manifest.read_text())
    config = manifest["render"]
    scene, camera = setup(config["width"], config["height"], config["fov"], config["samples"])
    output = Path(manifest["output"])
    for index, entry in enumerate(manifest["objects"], 1):
        render_object(scene, camera, entry, config, output)
        print(f"[blender] {index}/{len(manifest['objects'])} {entry['uid']}", flush=True)


if __name__ == "__main__":
    main()
