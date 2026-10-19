"""Ambiguity metrics and the SAM3D view-grid renderer (CPU; mesh tests need open3d)."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import unittest.mock

import numpy as np
from PIL import Image

from study.ambiguity import METRICS, Metric, metric_options, score_candidates, visual_entropy
from tools.cameras import direction
from reconstruction.sam3d import render_view_grid

HAS_OPEN3D = importlib.util.find_spec("open3d") is not None and importlib.util.find_spec("trimesh")


class VisualEntropyTests(unittest.TestCase):
    def test_scores_only_foreground_and_handles_empty_views(self):
        pixels = np.full((4, 4, 3), 255, dtype=np.uint8)
        pixels[:2, :2] = [0, 0, 0]
        image = Image.fromarray(pixels)
        mask = Image.fromarray(np.array([[255] * 4, [255] * 4, [0] * 4, [0] * 4], dtype=np.uint8))
        score = visual_entropy(image, mask, bins=2)
        self.assertAlmostEqual(score["entropy_bits"], 1.0)
        self.assertAlmostEqual(score["ambiguity"], 0.0)
        self.assertEqual(score["foreground_pixels"], 8)
        self.assertIsNone(visual_entropy(image, Image.new("L", (4, 4)), bins=2)["ambiguity"])

    def test_registry_validates_and_scores_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (4, 4), "gray").save(root / "a.png")
            Image.new("L", (4, 4), 255).save(root / "a.mask.png")
            candidate = {"azimuth": 0, "elevation": 0, "image": root / "a.png", "mask": root / "a.mask.png"}
            constant = Metric(lambda context, view: {"ambiguity": 0.25}, {}, lambda options: None,
                              lambda source, options, distance, target: None)
            with unittest.mock.patch.dict(METRICS, {"constant": constant}):
                scores = score_candidates(root, [candidate], {"constant": {}, "visual_entropy": {}})
            self.assertEqual(scores, [{"constant": 0.25, "visual_entropy": 1.0}])
        with self.assertRaisesRegex(ValueError, "bins"):
            metric_options("visual_entropy", {"bins": 1})
        with self.assertRaisesRegex(ValueError, "Unknown metric"):
            metric_options("missing")


class CameraTests(unittest.TestCase):
    def test_direction_is_y_up_with_azimuth_zero_on_plus_z(self):
        np.testing.assert_allclose(direction(0, 0), [0, 0, 1], atol=1e-7)
        np.testing.assert_allclose(direction(90, 0), [1, 0, 0], atol=1e-7)
        np.testing.assert_allclose(direction(0, 90), [0, 1, 0], atol=1e-7)


class RenderGridTests(unittest.TestCase):
    def test_renderer_uses_each_requested_angle_and_writes_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []

            def fake_render(_gaussian, **kwargs):
                calls.append((kwargs["yaw_start_deg"], kwargs["pitch_deg"]))
                color = np.full((1, 64, 64, 3), kwargs["bg_color"][0], dtype=np.float32)
                color[:, 16:48, 16:48] = 0.5
                return {"color": color}

            views = render_view_grid(None, fake_render, directory, [0, 180], [0, 30], 64)
            self.assertEqual(calls, [a for a in [(0, 0), (180, 0), (0, 30), (180, 30)] for _ in range(2)])
            self.assertEqual(len(views), 4)
            with Image.open(Path(directory) / views[0]["mask"]) as mask:
                self.assertEqual(int((np.asarray(mask) > 0).sum()), 32 * 32)


@unittest.skipUnless(HAS_OPEN3D, "requires open3d and trimesh (sam3d environment)")
class MeshMetricTests(unittest.TestCase):
    def cube(self):
        import trimesh
        return trimesh.creation.box(extents=(1, 1, 1))

    def test_cube_occlusion_and_normal_entropy(self):
        from study.ambiguity import build_context, self_occlusion, surface_normal_entropy

        cube = self.cube()
        context = build_context(cube.vertices, cube.faces, samples=10000, seed=0, nthreads=2)
        front, corner = {"azimuth": 0, "elevation": 0}, {"azimuth": 45, "elevation": 30}
        self.assertTrue(0.75 < self_occlusion(context, front)["occlusion_ratio"] < 0.92)
        options = {"polar_bins": 9, "azimuth_bins": 18}
        self.assertLess(surface_normal_entropy(context, corner, **options)["ambiguity"],
                        surface_normal_entropy(context, front, **options)["ambiguity"])

    def test_camera_target_and_distance(self):
        from study.ambiguity import build_context, visible_samples

        cube = self.cube()
        context = build_context(cube.vertices + [0, 0.35, 0], cube.faces, samples=10000, seed=0,
                                target=(0, 0.35, 0), distance=2.2, nthreads=2)
        visible, _, _ = visible_samples(context, {"azimuth": 90, "elevation": 0})
        np.testing.assert_allclose(context.points[visible, 0], 0.5, atol=1e-5)

    def test_two_sided_sees_back_of_open_surface(self):
        from study.ambiguity import build_context, self_occlusion

        quad = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])  # Normal +Z.
        back = {"azimuth": 180, "elevation": 0}
        one, two = (build_context(quad, faces, samples=2000, seed=0, distance=2.2, two_sided=sided,
                                  nthreads=2) for sided in (False, True))
        self.assertEqual(self_occlusion(one, back)["visible_ratio"], 0)
        self.assertGreater(self_occlusion(two, back)["visible_ratio"], 0.99)

    def test_mesh_metrics_read_reconstruction_mesh(self):
        cube = self.cube()
        with tempfile.TemporaryDirectory() as directory:
            np.savez(Path(directory) / "mesh.npz", vertices=cube.vertices, faces=cube.faces)
            views = [{"azimuth": 0, "elevation": 0}, {"azimuth": 45, "elevation": 30}]
            scores = score_candidates(directory, views, {"self_occlusion": {"samples": 5000},
                                                         "surface_normal_entropy": {"samples": 5000}})
        self.assertLess(scores[1]["surface_normal_entropy"], scores[0]["surface_normal_entropy"])


if __name__ == "__main__":
    unittest.main()
