"""Small CPU geometry checks; run in the sam3d environment."""
import unittest
import importlib.util
import tempfile
from pathlib import Path

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("pytorch3d"), "requires sam3d environment")
class HarnessTests(unittest.TestCase):
    def test_balanced_object_selection(self):
        from harness.build_dataset import select_uids
        objects = {
            "c1": {"category": "chair"}, "c2": {"category": "chair"},
            "c3": {"category": "chair"}, "s1": {"category": "sofa"},
            "s2": {"category": "sofa"},
        }
        pool = ["c2", "s1", "c1", "s2", "c3"]
        self.assertEqual(
            select_uids(pool, objects, limit=1, objects_per_category=2,
                        categories=["chair", "sofa"]),
            ["c2", "c1", "s1", "s2"],
        )
        with self.assertRaisesRegex(ValueError, "sofa=2"):
            select_uids(pool, objects, limit=1, objects_per_category=3,
                        categories=["chair", "sofa"])

    def test_fractional_view_names_are_distinct(self):
        from harness.build_dataset import sample_id
        self.assertNotEqual(sample_id(0.1, 20, 0.15), sample_id(0.9, 20, 0.15))
        self.assertNotEqual(sample_id(0, 20, 0.151), sample_id(0, 20, 0.154))

    def test_axis_rotations_preserve_handedness(self):
        from harness.canonicalize import axis_swap
        for axis in range(3):
            rotation = axis_swap(axis)
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3))
            self.assertAlmostEqual(np.linalg.det(rotation), 1)
            np.testing.assert_allclose(rotation @ np.eye(3)[axis], [0, 1, 0])

    def test_constant_vertex_colors_are_preserved(self):
        import trimesh
        from harness.meshio import ObjaverseStore
        mesh = trimesh.creation.box()
        mesh.visual.vertex_colors = [255, 0, 0, 255]
        colors = ObjaverseStore._vertex_colors(mesh, len(mesh.vertices))
        np.testing.assert_allclose(colors, np.tile([1, 0, 0], (len(mesh.vertices), 1)))

    def test_multimaterial_glb_keeps_each_parts_color(self):
        import trimesh
        from harness.meshio import load_mesh

        red = trimesh.creation.box()
        red.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(baseColorFactor=[255, 0, 0, 255]))
        blue = trimesh.creation.box()
        blue.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(baseColorFactor=[0, 0, 255, 255]))
        blue.apply_translation([2, 0, 0])
        scene = trimesh.Scene()
        scene.add_geometry(red)
        scene.add_geometry(blue)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parts.glb"
            path.write_bytes(scene.export(file_type="glb"))
            mesh = load_mesh(str(path), with_colors=True)
        colors = np.asarray(mesh.visual.vertex_colors)[:, :3]
        self.assertEqual(len(colors), len(mesh.vertices))
        for is_red, color in ((True, [255, 0, 0]), (False, [0, 0, 255])):
            selected = colors[(mesh.vertices[:, 0] < 1) == is_red]
            np.testing.assert_allclose(selected, np.tile(color, (len(selected), 1)))

    def test_line_primitives_do_not_discard_mesh_surfaces(self):
        import trimesh
        from harness.meshio import load_mesh

        box = trimesh.creation.box()
        scene = trimesh.Scene()
        scene.add_geometry(box)
        scene.add_geometry(trimesh.load_path([[0, 0, 0], [1, 0, 0]]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.glb"
            path.write_bytes(scene.export(file_type="glb"))
            mesh = load_mesh(str(path), with_colors=True)
        self.assertEqual(len(mesh.vertices), len(box.vertices))
        self.assertEqual(len(mesh.faces), len(box.faces))
        self.assertEqual(len(mesh.visual.vertex_colors), len(box.vertices))

    def test_render_masks_and_depth_match_camera(self):
        import torch
        import trimesh
        from harness.meshio import CanonicalMesh
        from harness.render import Harness
        from reconstruction.segmentation import depth_to_pointmap
        mesh = trimesh.creation.box(extents=[0.5, 0.7, 0.4])
        mesh.apply_translation([0, 0.35, 0])
        target = CanonicalMesh("box", "box", torch.tensor(mesh.vertices, dtype=torch.float32),
                               torch.tensor(mesh.faces.copy(), dtype=torch.int64),
                               torch.ones((len(mesh.vertices), 3)))
        renderer = Harness(image_size=32, device="cpu")
        result = renderer.render(target, [], 2.2, 0, 0, want_depth=True)
        self.assertTrue(torch.equal(result["modal"], result["amodal"]))
        self.assertGreater(result["gt"]["amodal_area"], 0)
        self.assertEqual(result["gt"]["occlusion_ratio"], 0)
        self.assertEqual(renderer.sweep_occlusion(target, [], 2.2, 0, 0, []), ({}, []))
        points = depth_to_pointmap(result["depth"].numpy(), *result["gt"]["intrinsics"])
        # Front face is at a constant view-space depth of 2.0.
        np.testing.assert_allclose(points[..., 2][result["modal"].numpy()], 2.0, atol=1e-5)
        visible_points = torch.from_numpy(points[result["modal"].numpy()])
        projected = renderer.camera(2.2, 0, 0).get_projection_transform().transform_points(visible_points)
        pixels = (1 - projected[:, :2]) * 16 - 0.5
        expected = result["modal"].nonzero()[:, [1, 0]].float()
        torch.testing.assert_close(pixels, expected, atol=1e-4, rtol=0)
        self.assertEqual(tuple(result["rgb"].shape), (32, 32, 3))


if __name__ == "__main__":
    unittest.main()
