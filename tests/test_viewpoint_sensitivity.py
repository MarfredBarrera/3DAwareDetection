"""CPU-only checks for the viewpoint-sensitivity metrics (numpy only)."""
import math
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from experiments.viewpoint_sensitivity import analyze, metrics


class MaskIoUTests(unittest.TestCase):
    def test_identical_masks_score_one(self):
        m = np.zeros((4, 4), dtype=bool)
        m[1:3, 1:3] = True
        self.assertEqual(metrics.mask_iou(m, m.copy()), 1.0)

    def test_disjoint_masks_score_zero(self):
        a = np.zeros((4, 4), dtype=bool); a[0, 0] = True
        b = np.zeros((4, 4), dtype=bool); b[3, 3] = True
        self.assertEqual(metrics.mask_iou(a, b), 0.0)

    def test_partial_overlap(self):
        a = np.zeros((1, 4), dtype=bool); a[0, 0:2] = True   # {0, 1}
        b = np.zeros((1, 4), dtype=bool); b[0, 1:3] = True   # {1, 2}
        self.assertAlmostEqual(metrics.mask_iou(a, b), 1 / 3)  # inter 1, union 3

    def test_empty_ground_truth_union_zero(self):
        empty = np.zeros((4, 4), dtype=bool)
        self.assertEqual(metrics.mask_iou(empty, empty), 0.0)


class ClassifyCorrectTests(unittest.TestCase):
    def test_confident_and_localized_is_tp(self):
        label = metrics.classify_correct([0.9, 0.2], [0.8, 0.1],
                                         tau_s=0.5, tau_iou=0.5, modal_area=100)
        self.assertEqual(label, "TP")

    def test_confident_but_mislocated_is_fn(self):
        # High score but low IoU: a confident detection in the wrong place.
        label = metrics.classify_correct([0.95], [0.2],
                                         tau_s=0.5, tau_iou=0.5, modal_area=100)
        self.assertEqual(label, "FN")

    def test_localized_but_below_threshold_is_fn(self):
        label = metrics.classify_correct([0.3], [0.9],
                                         tau_s=0.5, tau_iou=0.5, modal_area=100)
        self.assertEqual(label, "FN")

    def test_no_candidates_is_fn(self):
        label = metrics.classify_correct([], [], tau_s=0.5, tau_iou=0.5, modal_area=100)
        self.assertEqual(label, "FN")

    def test_invisible_object_is_skip(self):
        label = metrics.classify_correct([0.9], [0.9],
                                         tau_s=0.5, tau_iou=0.5, modal_area=0)
        self.assertEqual(label, "SKIP")


class ClassifyIncorrectTests(unittest.TestCase):
    def test_confident_detection_is_fp(self):
        self.assertEqual(metrics.classify_incorrect(0.7, tau_s=0.5), "FP")

    def test_no_confident_detection_is_tn(self):
        self.assertEqual(metrics.classify_incorrect(0.2, tau_s=0.5), "TN")

    def test_exact_threshold_is_fp(self):
        self.assertEqual(metrics.classify_incorrect(0.5, tau_s=0.5), "FP")


class Top1RecognitionTests(unittest.TestCase):
    def test_top1_uses_complete_vocabulary_and_stable_ties(self):
        prediction = metrics.top1_prediction(
            {"chair": 0.4, "sofa": 0.8, "table": 0.8},
            ["chair", "sofa", "table"],
        )
        self.assertEqual(prediction, ("sofa", 0.8, 0.8, 0.0))
        with self.assertRaisesRegex(ValueError, "exactly"):
            metrics.top1_prediction({"chair": 0.4}, ["chair", "sofa"])

    def test_view_aggregation_counts_one_accuracy_trial_per_image(self):
        rows = [
            {"category": "chair", "azimuth": 0, "elevation": 15,
             "top1_correct": True,
             "binary_confusion": {"TP": 1, "TN": 2, "FP": 1, "FN": 0}},
            {"category": "chair", "azimuth": 0, "elevation": 15,
             "top1_correct": False,
             "binary_confusion": {"TP": 0, "TN": 3, "FP": 0, "FN": 1}},
            {"category": "chair", "azimuth": 30, "elevation": 15,
             "top1_correct": None,
             "binary_confusion": {"TP": 0, "TN": 0, "FP": 0, "FN": 0}},
        ]
        cells = metrics.aggregate_viewpoints(rows)
        cell = cells[("chair", 0.0, 15.0)]
        self.assertEqual((cell["correct"], cell["incorrect"], cell["n"]), (1, 1, 2))
        self.assertEqual((cell["TP"], cell["FN"], cell["FP"], cell["TN"]),
                         (1, 1, 1, 5))
        self.assertEqual(cell["accuracy"], 0.5)
        self.assertEqual(cell["recall"], 0.5)
        self.assertAlmostEqual(cell["fp_rate"], 1 / 6)
        skipped = cells[("chair", 30.0, 15.0)]
        self.assertEqual(skipped["SKIP"], 1)
        self.assertTrue(math.isnan(skipped["accuracy"]))


class AnalysisRgbGridTests(unittest.TestCase):
    def test_one_consistent_representative_is_selected_per_category(self):
        records = [
            {"uid": "first", "category": "chair", "azimuth": az, "elevation": 0,
             "files": {"rgb": f"first-{az}.png"}}
            for az in (0, 90)
        ] + [
            {"uid": "second", "category": "chair", "azimuth": az, "elevation": 0,
             "files": {"rgb": f"second-{az}.png"}}
            for az in (0, 90)
        ]
        with TemporaryDirectory() as directory:
            render_dir = Path(directory)
            (render_dir / "index.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records))
            grids, uids = analyze._representative_rgb_records(
                render_dir, ["chair"], [0, 90], [0])

        self.assertEqual(uids, {"chair": "first"})
        self.assertEqual(grids["chair"][(90.0, 0.0)], render_dir / "first-90.png")


class Sam3dInputCollectionTests(unittest.TestCase):
    def test_collects_every_object_at_disjoint_best_and_worst_viewpoints(self):
        rows = []
        index = []
        accuracies = {0: 0.0, 90: 0.25, 180: 0.75, 270: 1.0}
        cells = {}
        with TemporaryDirectory() as directory:
            output = Path(directory)
            render = output / "render"
            (render / "rgb").mkdir(parents=True)
            for azimuth, accuracy in accuracies.items():
                cells[("chair", float(azimuth), 0.0)] = {
                    "accuracy": accuracy, "correct": int(accuracy * 4),
                    "incorrect": 4 - int(accuracy * 4), "n": 4,
                }
                for uid in ("object-a", "object-b"):
                    relative = f"rgb/{uid}-{azimuth}.png"
                    (render / relative).write_bytes(f"{uid}-{azimuth}".encode())
                    index.append({
                        "uid": uid, "category": "chair", "azimuth": azimuth,
                        "elevation": 0, "files": {"rgb": relative},
                    })
                    rows.append({
                        "uid": uid, "category": "chair", "azimuth": azimuth,
                        "elevation": 0, "top1_correct": accuracy >= 0.5,
                        "prediction": "chair" if accuracy >= 0.5 else "table",
                        "top1_score": max(0.8, {0: 0.5, 90: 0.6, 180: 0.9, 270: 0.95}[azimuth]),
                        "margin": 0.2,
                        "vocabulary": ["chair", "table", "sofa"],
                        "prompt_results": {
                            "chair": {"best_score": {0: 0.5, 90: 0.6, 180: 0.9, 270: 0.95}[azimuth]},
                            "table": {"best_score": 0.8},
                            "sofa": {"best_score": 0.6},
                        },
                    })
            (render / "index.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in index))

            manifest = analyze._collect_sam3d_inputs(
                rows, cells, render, output, ["chair"], viewpoints_per_band=2)

            self.assertEqual(len(manifest), 8)  # 2 bands x 2 views x 2 objects
            high = {record["azimuth"] for record in manifest
                    if record["quality_band"] == "high_accuracy"}
            low = {record["azimuth"] for record in manifest
                   if record["quality_band"] == "low_accuracy"}
            self.assertEqual(high, {180.0, 270.0})
            self.assertEqual(low, {0.0, 90.0})
            self.assertTrue(high.isdisjoint(low))
            collection = output / analyze.COLLECTION_DIRNAME
            rgb_files = list(collection.glob("chair/*/*/viewpoint_*/rgb.png"))
            metadata_files = list(collection.glob("chair/*/*/viewpoint_*/metadata.json"))
            self.assertEqual(len(rgb_files), 8)
            self.assertEqual(len(metadata_files), 8)
            self.assertEqual(len((collection / "manifest.jsonl").read_text().splitlines()), 8)
            confusions = [json.loads(line) for line in
                          (collection / "low_accuracy_confusions.jsonl").read_text().splitlines()]
            self.assertEqual(len(confusions), 4)
            self.assertEqual(
                {record["highest_scoring_incorrect_category"] for record in confusions},
                {"table"},
            )
            self.assertTrue(all(record["highest_scoring_incorrect_score"] == 0.8
                                for record in confusions))
            # Collection copies source bytes verbatim; confusion metadata is separate.
            copied = collection / confusions[0]["rgb"]
            source = Path(next(record["source_rgb"] for record in manifest
                               if record["rgb"] == confusions[0]["rgb"]))
            self.assertEqual(copied.read_bytes(), source.read_bytes())
            metadata = json.loads((collection / confusions[0]["metadata"]).read_text())
            self.assertEqual(metadata["object"]["id"], confusions[0]["uid"])
            self.assertEqual(metadata["object"]["true_category"], "chair")
            incorrect = metadata["highest_scoring_incorrect_object"]
            self.assertEqual(incorrect["category"], "table")
            self.assertEqual(incorrect["score"], 0.8)
            self.assertAlmostEqual(incorrect["score_minus_true_category"],
                                   confusions[0]["incorrect_minus_true_score"])
            self.assertEqual(metadata["viewpoint_selection"]["metric"],
                             "score_minus_true_category")
            self.assertEqual(metadata["viewpoint_accuracy"]["top1_accuracy"],
                             confusions[0]["viewpoint_accuracy"])
            self.assertIn("viewpoint_score_minus_true_category", confusions[0])
            self.assertEqual(metadata["rgb"]["file"], "rgb.png")
            high_confusions = [json.loads(line) for line in
                               (collection / "high_accuracy_confusions.jsonl")
                               .read_text().splitlines()]
            self.assertEqual(len(high_confusions), 4)
            self.assertTrue(all(record["quality_band"] == "high_accuracy"
                                for record in high_confusions))


if __name__ == "__main__":
    unittest.main()
