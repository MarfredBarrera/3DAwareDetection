"""CPU tests with synthetic semantic observations; no Habitat or GPU required."""
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from harness.habitat_dataset import camera_pose, candidates, discover_objects, generate, mask_stats, navigable_camera, parse_args


class HabitatDatasetTests(unittest.TestCase):
    def args(self, out="unused", *extra):
        return parse_args(["--scenes", "scene", "--queries", "bed", "--out", out, *extra])

    def test_ring_geometry_and_no_duplicate_endpoint(self):
        args = self.args("unused", "--radii", "2", "--elevations", "30", "--azimuth-step", "90")
        center = np.array([1., 2., 3.])
        poses = list(candidates({"center": center}, args))
        self.assertEqual(len(poses), 4)
        for pose, _ in poses:
            r, t = pose[:3, :3], pose[:3, 3]
            np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(r), 1)
            self.assertAlmostEqual(np.linalg.norm((t-center)[[0, 2]]), 2)
            target_camera = r.T @ (center-t)
            np.testing.assert_allclose(target_camera[:2], 0, atol=1e-12)
            self.assertLess(target_camera[2], 0)

    def test_offset_moves_target_left_and_down_without_reaiming(self):
        base = camera_pose([0, 0, 0], 2, 45, 20, [0, 0, 0])
        shifted = camera_pose([0, 0, 0], 2, 45, 20, [.4, .2, .1])
        np.testing.assert_allclose(base[:3, :3], shifted[:3, :3])
        local = shifted[:3, :3].T @ -shifted[:3, 3]
        np.testing.assert_allclose(local[:2], [-.4, -.2], atol=1e-12)
        self.assertAlmostEqual(local[2], -2 / np.cos(np.deg2rad(20)) + .1)

    def test_discovery_uses_semantic_id_not_list_index(self):
        obj = SimpleNamespace(id="0_2_99", semantic_id=42,
                              category=SimpleNamespace(name=lambda: " Bed "),
                              aabb=SimpleNamespace(center=[0, 0, 0], sizes=[1, 1, 1]))
        scene = SimpleNamespace(objects=[None, obj])
        objects, categories = discover_objects(scene, ["bed"], "exact")
        self.assertEqual(objects[0]["semantic_id"], 42)
        self.assertEqual(categories, ["bed"])
        self.assertEqual(discover_objects(scene, ["be"], "exact")[0], [])
        self.assertEqual(len(discover_objects(scene, ["be"], "substring")[0]), 1)

    def test_visible_mask_measurements(self):
        mask = np.zeros((10, 20), dtype=bool)
        mask[2:5, 3:8] = True
        stats = mask_stats(mask)
        self.assertEqual(stats["visible_pixels"], 15)
        self.assertEqual(stats["bbox_xyxy"], [3, 2, 8, 5])
        self.assertEqual(stats["centroid_xy"], [5., 3.])
        self.assertFalse(stats["touches_border"])
        self.assertEqual(mask_stats(np.zeros_like(mask))["visible_pixels"], 0)

    def test_navigability_checks_translated_floor_without_moving_camera(self):
        position = np.array([2., 3., 4.])
        probes = []
        def check(point, tolerance):
            probes.append((point.copy(), tolerance))
            return False
        self.assertFalse(navigable_camera(SimpleNamespace(is_navigable=check), position, self.args()))
        np.testing.assert_allclose(probes[0][0], [2., 1.5, 4.])
        np.testing.assert_allclose(position, [2., 3., 4.])
        self.assertEqual(probes[0][1], .5)

    def test_export_filtering_and_cleanup(self):
        renderers = []

        class FakeRenderer:
            def __init__(self, scene, args):
                self.closed = False
                self.calls = 0
                obj = SimpleNamespace(id="instance_99", semantic_id=42,
                                      category=SimpleNamespace(name=lambda: "bed"),
                                      aabb=SimpleNamespace(center=[0, 0, 0], sizes=[1, 1, 1]))
                self.sim = SimpleNamespace(semantic_scene=SimpleNamespace(objects=[None, obj]))
                renderers.append(self)

            def render(self, pose):
                self.calls += 1
                semantic = np.full((8, 8), 7)
                if self.calls == 1:
                    semantic[2:6, 2:6] = 42
                return dict(rgb=np.full((8, 8, 4), 128, dtype=np.uint8),
                            semantic=semantic, depth=np.full((8, 8), 2.))

            def close(self):
                self.closed = True

        with TemporaryDirectory() as directory:
            args = self.args(directory, "--radii", "2", "--elevations", "0", "--azimuth-step", "180",
                             "--width", "8", "--height", "8", "--min-pixels", "1", "--save-depth")
            summary = generate(args, FakeRenderer)
            self.assertEqual(summary["saved"], 1)
            self.assertEqual(summary["rejected"], {"target_not_visible_enough": 1})
            record = json.loads((Path(directory) / "index.jsonl").read_text())
            self.assertEqual(record["semantic_id"], 42)
            self.assertEqual(record["visible_pixels"], 16)
            self.assertEqual(record["intrinsics"][2:], [3.5, 3.5])
            for file in record["files"].values():
                self.assertTrue((Path(directory) / file).is_file())
            self.assertTrue(renderers[0].closed)
            with self.assertRaises(ValueError):
                generate(args, FakeRenderer)
        with TemporaryDirectory() as directory:
            args.out = directory
            args.queries = ["sofa"]
            with self.assertRaisesRegex(RuntimeError, "No images saved"):
                generate(args, FakeRenderer)
            self.assertTrue(renderers[-1].closed)
            self.assertEqual(json.loads((Path(directory) / "summary.json").read_text())["status"], "failed")

    def test_bad_sampling_arguments_fail(self):
        for extra in [("--radii", "0"), ("--azimuth-step", "0"), ("--elevations", "90"),
                      ("--offset", "nan", "0", "0"), ("--min-pixels", "0")]:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args("unused", *extra)


if __name__ == "__main__":
    unittest.main()
