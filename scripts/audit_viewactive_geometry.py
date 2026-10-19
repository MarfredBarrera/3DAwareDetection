"""Inspect metadata and evaluated Blender geometry without rendering or GPU use.

First run with ordinary Python to prepare the manifest, then with Blender:
blender --background --factory-startup --python scripts/audit_viewactive_geometry.py
"""
import json
from pathlib import Path
import struct
import sys
import tempfile

ROOT = Path("outputs/viewactive/vqf_audit")


def prepare():
    canonical = json.loads((ROOT / "canonical.json").read_text())["objects"]
    paths = json.loads((ROOT / "download.json").read_text())["glbs"]
    rows = json.loads(Path("outputs/viewactive/blender_full_viz/plan.json").read_text())
    manifest = []
    for row in rows:
        uid = row["uid"]
        path = Path(paths[uid])
        data = path.read_bytes()
        length = struct.unpack_from("<I", data, 12)[0]
        doc = json.loads(data[20:20 + length])
        entry = {"uid": uid, "category": row["category"], "path": str(path),
                 "canonical": canonical[uid], "skins": len(doc.get("skins", [])),
                 "animations": len(doc.get("animations", [])),
                 "materials": [{k: m.get(k) for k in ("name", "alphaMode", "doubleSided")}
                               for m in doc.get("materials", [])]}
        manifest.append(entry)
        print(uid, entry["category"], "skins", entry["skins"], "animations", entry["animations"], flush=True)
    (ROOT / "geometry_manifest.json").write_text(json.dumps({"objects": manifest}, indent=2))


def blender_audit():
    import bpy
    import numpy as np
    import runpy
    renderer = runpy.run_path("viewactive/blender_render.py", run_name="audit_renderer")
    canonical_transform, rest_pose_glb = renderer["canonical_transform"], renderer["rest_pose_glb"]

    result = []
    for entry in json.loads((ROOT / "geometry_manifest.json").read_text())["objects"]:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        with tempfile.TemporaryDirectory() as scratch:
            bpy.ops.import_scene.gltf(filepath=rest_pose_glb(entry["path"], Path(scratch) / "rest.glb"))
        imported = set(bpy.data.objects)
        transform, axis = canonical_transform(entry["canonical"])
        for obj in imported:
            if obj.parent not in imported:
                obj.matrix_world = transform @ obj.matrix_world
        bpy.context.view_layer.update()
        deps = bpy.context.evaluated_depsgraph_get()
        parts = [{"name": o.name, "vertices": len(o.data.vertices), "hide_render": o.hide_render,
                  "collections": [(c.name, c.hide_render) for c in o.users_collection]}
                 for o in imported if o.type == "MESH"]
        variants = {}
        for evaluated in (False, True):
            vertices = []
            for obj in imported:
                if (obj.type != "MESH" or obj.hide_render
                        or all(c.hide_render for c in obj.users_collection)):
                    continue
                source = obj.evaluated_get(deps) if evaluated else obj
                matrix = np.array(axis.inverted() @ source.matrix_world)
                v = np.empty(len(source.data.vertices)*3, dtype=np.float32)
                source.data.vertices.foreach_get("co", v)
                v = v.reshape(-1,3)
                vertices.append(v @ matrix[:3,:3].T + matrix[:3,3])
            v = np.concatenate(vertices)
            variants["evaluated" if evaluated else "base"] = {
                "min": v.min(0).tolist(), "max": v.max(0).tolist(),
                "extent": np.ptp(v, axis=0).tolist(),
                "center": ((v.max(0)+v.min(0))/2).tolist(),
            }
        record = {**entry, **variants, "parts": parts}
        result.append(record)
        print("AUDIT", entry["category"], entry["uid"], "expected", entry["canonical"]["extents_canonical"],
              "evaluated", variants["evaluated"], flush=True)
    (ROOT / "geometry_audit.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    if "bpy" in sys.modules:
        blender_audit()
    else:
        prepare()
