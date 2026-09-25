import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

import numpy as np
from PIL import Image

from reconstruction.visualization import render_view_grid
from viewpoint_optimization.visualize import render_layout
from viewpoint_optimization.metrics import METRICS, Metric, metric_options, score_views, visual_entropy
from viewpoint_optimization.__main__ import load_config, run


class ViewpointOptimizationTests(unittest.TestCase):
    def test_yaml_paths_and_grid_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 8)).save(root / "image.png")
            Image.new("L", (8, 8), 255).save(root / "mask.png")
            config = {"image": "image.png", "mask": "mask.png", "output_dir": "output",
                      "reconstruction": {"azimuths": [0, 180, 270], "elevations": [0, 20],
                                         "resolution": 128}}
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            settings = load_config(path)
            self.assertEqual(settings["image"], str(root / "image.png"))
            self.assertEqual(settings["output_dir"], str(root / "output"))
            self.assertEqual(settings["azimuths"], [0, 180, 270])
            with (patch("viewpoint_optimization.__main__.subprocess.run") as worker,
                  patch("viewpoint_optimization.__main__.score_views"),
                  patch("viewpoint_optimization.__main__.render_layout",
                        return_value=root / "output/viewpoint_grid.png")):
                self.assertEqual(run(path), root / "output/viewpoint_grid.png")
            self.assertEqual(worker.call_count, 2)  # Reconstruction, then scoring.
            self.assertEqual([call.args[0][5] for call in worker.call_args_list],
                             ["--worker", "--worker"])
            self.assertEqual([call.args[0][6] for call in worker.call_args_list],
                             ["reconstruct", "score"])
            self.assertEqual(json.loads((root / "output/run.json").read_text())["elevations"], [0, 20])
            config["reconstruction"]["elevations"] = [90]
            path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, "elevations"):
                load_config(path)

    def test_default_angles_match_reviewed_upright_experiment(self):
        project = Path(__file__).resolve().parents[1]
        previous = yaml.safe_load((project / "experiments/viewpoint_sensitivity/config.upright.yaml").read_text())
        current = load_config(project / "viewpoint_optimization/config.yaml")
        self.assertEqual(current["azimuths"], previous["azimuths"])
        self.assertEqual(current["elevations"], previous["elevations"])

    def test_renderer_uses_each_requested_angle_and_writes_distinct_images(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []

            def fake_render(_gaussian, **kwargs):
                calls.append((kwargs["yaw_start_deg"], kwargs["pitch_deg"]))
                # Center is opaque gray; outside follows the requested background.
                color = np.full((1, 64, 64, 3), kwargs["bg_color"][0], dtype=np.float32)
                color[:, 16:48, 16:48] = 0.5
                return {"color": color}

            views = render_view_grid(None, fake_render, directory, [0, 180, 270], [0, 30], 64)
            self.assertEqual(calls, [angle for angle in
                                     [(0, 0), (180, 0), (270, 0),
                                      (0, 30), (180, 30), (270, 30)]
                                     for _ in range(2)])
            self.assertEqual(len(views), 6)
            self.assertEqual(len(list(Path(directory).glob("*.png"))), 12)
            with Image.open(Path(directory) / views[0]["mask"]) as mask:
                pixels = np.asarray(mask)
                self.assertEqual(int((pixels > 0).sum()), 32 * 32)

    def test_visual_entropy_scores_only_foreground_and_handles_empty_views(self):
        pixels = np.full((4, 4, 3), 255, dtype=np.uint8)
        pixels[:2, :2] = [0, 0, 0]
        pixels[:2, 2:] = [255, 255, 255]
        image = Image.fromarray(pixels)
        mask = Image.fromarray(np.array([[255] * 4, [255] * 4, [0] * 4, [0] * 4], dtype=np.uint8))
        score = visual_entropy(image, mask, bins=2)
        self.assertAlmostEqual(score["entropy_bits"], 1.0)
        self.assertAlmostEqual(score["ambiguity"], 0.0)
        self.assertEqual(score["foreground_pixels"], 8)
        blank = visual_entropy(image, Image.new("L", (4, 4)), bins=2)
        self.assertIsNone(blank["ambiguity"])

    def test_registered_metric_scores_saved_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB", (4, 4), "gray")
            image.save(root / "frame.png")
            Image.new("L", (4, 4), 255).save(root / "frame.mask.png")
            (root / "candidates.json").write_text(json.dumps({
                "azimuths": [0], "elevations": [0],
                "candidates": [{"azimuth": 0, "elevation": 0,
                                "image": "frame.png", "mask": "frame.mask.png"}],
            }))
            with patch.dict(METRICS, {"constant": Metric(
                    lambda context, candidate: {"ambiguity": 0.25, "foreground_pixels": 16},
                    {}, lambda options: None,
                    lambda output_dir, manifest, options: None)}):
                path = score_views(root, "constant")
            result = json.loads(path.read_text())
            self.assertEqual(result["scores"]["frame"]["ambiguity"], 0.25)
            self.assertTrue(result["lower_is_better"])
            with self.assertRaisesRegex(ValueError, "Unknown metric"):
                score_views(root, "missing")
            with self.assertRaisesRegex(ValueError, "bins"):
                metric_options("visual_entropy", {"bins": 1})

    def test_mesh_metrics_on_cube(self):
        import trimesh

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cube = trimesh.creation.box(extents=(1, 1, 1))
            np.savez(root / "mesh.npz", vertices=cube.vertices, faces=cube.faces)
            candidates = [{"azimuth": azimuth, "elevation": elevation,
                           "image": f"azimuth_{azimuth}_elevation_{elevation}.png"}
                          for azimuth, elevation in ((0, 0), (45, 30))]
            (root / "candidates.json").write_text(json.dumps({"candidates": candidates}))
            occlusion = json.loads(score_views(root, "self_occlusion", options={
                "samples": 10000}).read_text())["scores"]
            front = occlusion["azimuth_0_elevation_0"]["occlusion_ratio"]
            self.assertGreater(front, 0.75)
            self.assertLess(front, 0.92)
            normals = json.loads(score_views(root, "surface_normal_entropy", options={
                "samples": 10000}).read_text())["scores"]
            self.assertLess(normals["azimuth_45_elevation_30"]["ambiguity"],
                            normals["azimuth_0_elevation_0"]["ambiguity"])

    def test_layout_aligns_images_and_partial_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "views").mkdir()
            Image.new("RGB", (4, 4), "white").save(root / "input.png")
            candidates = []
            for elevation in (0, 30):
                for azimuth in (0, 180, 270):
                    name = f"views/azimuth_{azimuth}_elevation_{elevation}.png"
                    Image.new("RGB", (4, 4), "red").save(root / name)
                    candidates.append({"azimuth": azimuth, "elevation": elevation, "image": name})
            (root / "candidates.json").write_text(json.dumps({
                "source_image": "input.png", "query": "desk",
                "azimuths": [0, 180, 270], "elevations": [0, 30], "candidates": candidates,
            }))
            render_layout(root)
            with Image.open(root / "viewpoint_grid.png") as plot:
                self.assertGreater(plot.width, plot.height)
            scores = root / "scores.json"
            scores.write_text(json.dumps({"azimuth_0_elevation_0": 0.2, "azimuth_180_elevation_30": 0.8}))
            render_layout(root, scores)
            with Image.open(root / "viewpoint_grid.png") as plot:
                self.assertGreater(plot.width, plot.height)
                self.assertEqual(plot.format, "PNG")
            scores.write_text(json.dumps({"wrong": 0.2}))
            with self.assertRaisesRegex(ValueError, "Unknown score"):
                render_layout(root, scores)
