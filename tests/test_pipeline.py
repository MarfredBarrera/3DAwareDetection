import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import yaml

import main


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "replays/episode/rgb/000003.jpg"
        self.image.parent.mkdir(parents=True)
        self.image.touch()
        self.config = {
            "image": "replays/episode/rgb/000003.jpg",
            "query": "chair", "output_dir": "output",
            "visualization": {"enabled": False},
        }
        self.path = self.root / "config.yaml"

    def save(self):
        self.path.write_text(yaml.safe_dump(self.config))
        return self.path

    def test_image_and_relative_paths(self):
        config = main.load_config(self.save())
        self.assertEqual(config["image"], str(self.image))
        self.assertEqual(config["output_dir"], str(self.root / "output"))
        self.assertEqual(config["gpu_id"], 0)

    def test_supplied_frame_and_query_override_demo_placeholder(self):
        config = main.load_config(main.WORKSPACE / "configs/demo.yaml",
                                  image=self.image, query="vase")
        self.assertEqual(config["image"], str(self.image))
        self.assertEqual(config["query"], "vase")

    def test_gpu_id_must_be_non_negative_integer(self):
        self.config["gpu_id"] = -1
        with self.assertRaisesRegex(ValueError, "gpu_id"):
            main.load_config(self.save())

    def test_missing_source_fails_before_workers(self):
        self.image.unlink()
        with patch("main.subprocess.run") as run:
            with self.assertRaises(FileNotFoundError):
                main.run_pipeline(self.save())
            run.assert_not_called()

    def test_depth_requires_intrinsics(self):
        self.config["preprocessing"] = {"depth": "depth.npy"}
        with self.assertRaisesRegex(ValueError, "supplied together"):
            main.load_config(self.save())

    def test_invalid_render_settings(self):
        self.config["reconstruction"] = {"views": 1}
        with self.assertRaisesRegex(ValueError, "views"):
            main.load_config(self.save())

    def test_verification_requires_rivals_for_relative_margin(self):
        self.config["verification"] = {"method": "relative_margin"}
        with self.assertRaisesRegex(ValueError, "rival_prompts"):
            main.load_config(self.save())

    def test_environment_order_and_resolved_handoff(self):
        with patch("main.subprocess.run") as run:
            output = main.run_pipeline(self.save())
        calls = run.call_args_list
        self.assertEqual([call.args[0][1] for call in calls], ["core", "sam3d", "core"])
        self.assertTrue(all(call.kwargs["check"] for call in calls))
        self.assertEqual(json.loads((output / "run.json").read_text())["image"], str(self.image))

    def test_preprocessing_failure_stops_pipeline(self):
        with patch("main.subprocess.run", side_effect=subprocess.CalledProcessError(1, "worker")) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                main.run_pipeline(self.save())
        self.assertEqual(run.call_count, 1)

    def test_batch_overrides_do_not_modify_yaml(self):
        self.config["visualization"]["enabled"] = True
        path = self.save()
        original = path.read_text()
        with patch("main.subprocess.run"):
            output = main.run_pipeline(path, no_viewer=True, output_dir=self.root / "batch")
        saved = json.loads((output / "run.json").read_text())
        self.assertFalse(saved["visualization"]["enabled"])
        self.assertEqual(output, self.root / "batch")
        self.assertEqual(path.read_text(), original)

    def test_no_matches_replace_stale_handoff(self):
        from PIL import Image
        Image.new("RGB", (4, 4)).save(self.image)
        config = main.load_config(self.save())
        output = Path(config["output_dir"])
        output.mkdir()
        config["masks"] = ["stale.png"]
        settings = output / "run.json"
        settings.write_text(json.dumps(config))
        with patch("reconstruction.segmentation.SAM3Preprocessor") as segmenter:
            segmenter.return_value.masks_for_query.return_value = []
            with self.assertRaisesRegex(RuntimeError, "no objects"):
                main.prepare(settings)
        self.assertEqual(json.loads(settings.read_text())["masks"], [])

    def test_depth_pointmap_axes_and_invalid_pixels(self):
        import numpy as np
        from reconstruction.segmentation import depth_to_pointmap
        points = depth_to_pointmap(np.array([[2, np.nan], [0, 2]]), 2, 2, 0, 0)
        np.testing.assert_allclose(points[0, 0], [0, 0, 2])
        np.testing.assert_allclose(points[1, 1], [-1, -1, 2])
        np.testing.assert_allclose(points[0, 1], [0, 0, 0])
        np.testing.assert_allclose(points[1, 0], [0, 0, 0])

    def test_highest_confidence_mask_selection(self):
        from reconstruction.segmentation import SAM3Preprocessor
        select = SAM3Preprocessor.highest_confidence_indices
        self.assertEqual(select([0.1, 0.9, 0.4], 3, 0.2, 2), [1, 2])
        self.assertEqual(select([0.9, 0.9], 2, 0.5, 1), [0])
        self.assertEqual(select([], 0, 0.2, 1), [])
        with self.assertRaisesRegex(ValueError, "scores and masks"):
            select([0.9], 2, 0.2, 1)

    def test_invalid_mask_selection_settings(self):
        self.config["preprocessing"] = {"threshold": 1.1}
        with self.assertRaisesRegex(ValueError, "threshold"):
            main.load_config(self.save())
        self.config["preprocessing"] = {"max_masks": 0}
        with self.assertRaisesRegex(ValueError, "max_masks"):
            main.load_config(self.save())

    def test_all_current_masks_are_reconstructed_and_visualized(self):
        self.config["visualization"]["enabled"] = True
        config = main.load_config(self.save())
        output = Path(config["output_dir"])
        output.mkdir()
        settings = output / "run.json"
        config.update(prepared_image=str(self.image),
                      masks=["mask-a.png", "mask-b.png"], pointmap=None)
        settings.write_text(json.dumps(config))
        (output / "stale-mask.png").touch()
        model = Mock()
        with patch("reconstruction.sam3d.SAM3D", return_value=model), \
                patch("reconstruction.sam3d.launch_viewer") as viewer:
            main.reconstruct(settings)
            main.view(settings)
        self.assertEqual(model.render_views.call_count, 2)
        self.assertEqual([call.args[1] for call in model.render_views.call_args_list],
                         ["mask-a.png", "mask-b.png"])
        self.assertEqual(viewer.call_args.args[0],
                         [output / "object-000/object.ply", output / "object-001/object.ply"])

    def test_verification_worker_writes_veto_decision(self):
        from PIL import Image
        config = main.load_config(self.save())
        output = Path(config["output_dir"])
        directory = output / "object-000/views"
        directory.mkdir(parents=True)
        for elevation in (0, 30, 60):
            for azimuth in (0, 45, 90, 135, 180, 225, 270, 315):
                Image.new("RGB", (8, 8)).save(
                    directory / f"azimuth_{azimuth}_elevation_{elevation}.png")
        masks = output / "masks"
        masks.mkdir()
        (masks / "predictions.json").write_text(json.dumps({"scores": [0.9]}))
        config.update(masks=["mask-a.png"])
        settings = output / "run.json"
        settings.write_text(json.dumps(config))
        with patch("reconstruction.segmentation.SAM3Preprocessor") as detector:
            detector.return_value.detect_many.return_value = {
                "chair": {"scores": __import__("numpy").array([0.2])}}
            main.verify(settings)
        decision = json.loads((output / "verification.json").read_text())
        self.assertFalse(decision["objects"][0]["accepted"])
        self.assertEqual(decision["objects"][0]["views"], 24)


if __name__ == "__main__":
    unittest.main()
