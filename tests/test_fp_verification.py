"""Claim-level verification and study accounting."""
import json
import tempfile
import unittest
from pathlib import Path

from study import db as store
from study.fp_verification import compare_with_existing, evaluate, save_plots, verify_claim


class VerificationTests(unittest.TestCase):
    def test_rules_reject_mislabeled_object_and_keep_supported_claim(self):
        views = [{"chair": 0.9, "table": 0.2},
                 {"chair": 0.1, "table": 0.9},
                 {"chair": 0.2, "table": 0.8},
                 {"chair": 0.3, "table": 0.8},
                 {"chair": 0.1, "table": 0.7}]
        for method in ("majority", "lower_quartile", "relative_margin"):
            self.assertFalse(verify_claim("chair", 0.95, views, method=method))
            self.assertTrue(verify_claim("table", 0.8, views, method=method))
        self.assertFalse(verify_claim("table", 0.2, views))
        self.assertFalse(verify_claim("chair", 0.9, views[:2]))
        self.assertFalse(verify_claim("chair", 0.9, [{"chair": 0.8}, {}, {}]))
        self.assertFalse(verify_claim("chair", 0.9,
                                      [{"chair": 0.1, "table": 0.0}] * 7,
                                      method="relative_margin"))
        tied = ([{"chair": 0.8, "table": 0.2}] * 12
                + [{"chair": 0.2, "table": 0.8}] * 12)
        self.assertFalse(verify_claim("chair", 0.9, tied, method="majority"))
        self.assertFalse(verify_claim("chair", 0.9, tied, method="relative_margin"))
        self.assertTrue(verify_claim("chair", 0.9, [
            {"chair": 0.8, "table": 0.3},
            {"chair": 0.2, "table": 0.7},
            {"chair": 0.2, "table": 0.6}], method="peak_margin"))
        self.assertFalse(verify_claim("chair", 0.9, [
            {"chair": 0.6, "table": 0.5},
            {"chair": 0.2, "table": 0.7},
            {"chair": 0.2, "table": 0.6}], method="peak_margin"))

    def test_evaluation_counts_prompt_false_positives(self):
        with tempfile.TemporaryDirectory() as directory:
            db = store.connect(directory)
            with db:
                parent = store.add_image(db, key="gt/1", source="gt", uid="one",
                                         category="table", azimuth=0, elevation=0,
                                         path="unused", visible_px=10)
            store.add_detections(db, parent, {
                "table": (0.8, 1, {"scores": [0.8], "ious": [0.9]}),
                "chair": (0.9, 1, None)})
            for elevation in (0, 30, 60):
                for azimuth in (0, 45, 90, 135, 180, 225, 270, 315):
                    with db:
                        child = store.add_image(db, key=f"sam3d/{azimuth}/{elevation}",
                                                source="sam3d", parent_id=parent, uid="one",
                                                category="table", azimuth=azimuth,
                                                elevation=elevation, path="unused", visible_px=10)
                    store.add_detections(db, child, {
                        "table": (0.2 if elevation == 0 else 0.8, 1, None),
                        "chair": (0.8 if elevation == 0 else 0.2, 1, None)})
            config = {"dataset": {"categories": ["chair", "table"]},
                      "detection": {"score_threshold": 0.5, "iou_threshold": 0.5}}
            result = evaluate(db, config)
            self.assertEqual(result["eligible_inputs"], 1)
            self.assertEqual(result["methods"]["input"]["fp"], 1)
            for method in ("majority", "relative_margin"):
                self.assertEqual(result["methods"][method]["fp"], 0)
                self.assertEqual(result["methods"][method]["tp"], 1)
            self.assertEqual(result["methods"]["lower_quartile"]["tp"], 0)
            self.assertEqual(result["methods"]["peak_margin"]["fp"], 1)
            metrics = result["study_metrics"]["subsets"]["all inputs"]["selectors"]
            self.assertEqual(metrics["input"]["n"], 1)
            self.assertEqual(metrics["majority"]["any_fp"], 0)
            self.assertEqual(metrics["relative_margin"]["tp"], 1)
            paths = save_plots(result, directory)
            self.assertEqual(len(paths), 15)
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertTrue((Path(directory) / "figures" / "fp_verification"
                             / "detection_any_fp_all_inputs.png").is_file())
            prior = {"inputs": result["eligible_inputs"], "objects": result["objects"],
                     "detection": result["detection"],
                     "selectors": ["input", "identity", "random", "oracle"],
                     "subsets": {name: {"selectors": {**{
                         key: data["selectors"][key] for key in ("input", "identity", "oracle")},
                         "random": data["selectors"]["input"]}}
                                 for name, data in result["study_metrics"]["subsets"].items()}}
            (Path(directory) / "summary.json").write_text(json.dumps(prior))
            comparison = compare_with_existing(result, directory)
            self.assertIn("random", comparison["selectors"])
            self.assertIn("relative_margin", comparison["selectors"])
            self.assertIn("peak_margin", comparison["selectors"])
            self.assertAlmostEqual(
                comparison["subsets"]["all inputs"]["selectors"]["input"]["f1"], 2 / 3)
            self.assertAlmostEqual(
                comparison["subsets"]["all inputs"]["selectors"]["input"]["tn_rate"], 0)
            db.close()


if __name__ == "__main__":
    unittest.main()
