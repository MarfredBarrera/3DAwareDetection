"""Blender-only diagnostic for GLB to canonical-frame alignment."""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix


def main():
    arguments = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("uid")
    parser.add_argument("--expected-vertices", type=Path)
    args = parser.parse_args(arguments)
    manifest = json.loads(args.manifest.read_text())
    entry = next(row for row in manifest["objects"] if row["uid"] == args.uid)
    bpy.ops.import_scene.gltf(filepath=entry["path"])
    imported = set(bpy.data.objects)
    axis = Matrix.Rotation(math.pi / 2, 4, "X")
    rec = entry["canonical"]
    canonical = Matrix.Identity(4)
    for i, row in enumerate(rec["rotation"]):
        for j, value in enumerate(row):
            canonical[i][j] = value * rec["scale"]
        canonical[i][3] = rec["translation"][i]
    transform = axis @ canonical @ axis.inverted()
    for obj in imported:
        if obj.parent not in imported:
            obj.matrix_world = transform @ obj.matrix_world
    bpy.context.view_layer.update()
    low, high = [float("inf")] * 3, [float("-inf")] * 3
    for obj in imported:
        if obj.type != "MESH":
            continue
        if args.expected_vertices and len(obj.data.vertices) == rec["vertex_count"]:
            local = np.asarray([list(v.co) for v in obj.data.vertices])
            expected = np.load(args.expected_vertices)
            design = np.column_stack((local, np.ones(len(local))))
            affine = np.linalg.lstsq(design, expected, rcond=None)[0]
            error = np.linalg.norm(design @ affine - expected, axis=1)
            print("affine_local_to_trimesh", affine.tolist(), "median_error", np.median(error),
                  "max_error", np.max(error))
        node_low, node_high = [float("inf")] * 3, [float("-inf")] * 3
        for vertex in obj.data.vertices:
            point = axis.inverted() @ (obj.matrix_world @ vertex.co)
            for i in range(3):
                low[i] = min(low[i], point[i])
                high[i] = max(high[i], point[i])
                node_low[i] = min(node_low[i], point[i])
                node_high[i] = max(node_high[i], point[i])
        print("mesh", obj.name, "verts", len(obj.data.vertices), "faces", len(obj.data.polygons),
              "bounds", node_low, node_high,
              "local_bounds", [min(v.co[i] for v in obj.data.vertices) for i in range(3)],
              [max(v.co[i] for v in obj.data.vertices) for i in range(3)])
    print("uid", args.uid)
    print("expected_extents", rec["extents_canonical"])
    print("actual_min", low, "actual_max", high)
    print("actual_extents", [high[i] - low[i] for i in range(3)])
    print("nodes", [(obj.name, obj.type, obj.parent.name if obj.parent else None,
                     list(obj.matrix_world.translation)) for obj in imported])


if __name__ == "__main__":
    main()
