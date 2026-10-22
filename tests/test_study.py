"""Study config, database, labels, selectors, analysis, and plots (CPU, core env)."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import yaml

from study import analyze, config as study_config, db as store, labels, selectors
from study.figures import Heatmap, Images, margin_heatmap, save_bars, save_panels
from study.__main__ import cohort, import_renders, object_views

VOCABULARY = ["chair", "table", "desk"]
DETECTION = {"score_threshold": 0.5, "iou_threshold": 0.5}


class LabelTests(unittest.TestCase):
    def test_mask_iou(self):
        a = np.zeros((1, 4), bool); a[0, :2] = True
        b = np.zeros((1, 4), bool); b[0, 1:3] = True
        self.assertAlmostEqual(labels.mask_iou(a, b), 1 / 3)
        self.assertEqual(labels.mask_iou(np.zeros(3, bool), np.zeros(3, bool)), 0.0)

    def test_true_positive_needs_confident_and_localized_candidate(self):
        self.assertTrue(labels.true_positive({"scores": [0.9, 0.2], "ious": [0.8, 0.1]}, 0.5, 0.5))
        self.assertFalse(labels.true_positive({"scores": [0.95], "ious": [0.2]}, 0.5, 0.5))
        self.assertFalse(labels.true_positive({"scores": [0.3], "ious": [0.9]}, 0.5, 0.5))
        self.assertFalse(labels.true_positive(None, 0.5, 0.5))

    def test_prompt_text_cleans_lvis_names(self):
        self.assertEqual(labels.prompt_text("car_(automobile)"), "car")
        self.assertEqual(labels.prompt_text("monitor_(computer_equipment) computer_monitor"), "monitor")
        self.assertEqual(labels.prompt_text("traffic_light"), "traffic light")
        self.assertEqual(labels.prompt_text("chair"), "chair")

    def test_top1_breaks_ties_by_vocabulary_order(self):
        self.assertEqual(labels.top1({"chair": 0.4, "table": 0.8, "desk": 0.8}, VOCABULARY), "table")
        with self.assertRaisesRegex(ValueError, "exactly"):
            labels.top1({"chair": 0.4}, VOCABULARY)

    def test_outcome_margin_and_false_positives(self):
        image = {"category": "desk", "visible_px": 10}
        result = labels.outcome(image, {"chair": 0.1, "table": 0.7, "desk": 0.6}, None,
                                VOCABULARY, DETECTION)
        self.assertAlmostEqual(result["margin"], -0.1)
        self.assertEqual((result["prediction"], result["wrong_prompt"], result["fp"]), ("table", "table", 1))
        self.assertFalse(result["correct"])
        self.assertIsNone(labels.outcome({"category": "desk", "visible_px": 0}, {}, None,
                                         VOCABULARY, DETECTION))


class PickTests(unittest.TestCase):
    def test_gt_vqf_max_takes_the_candidate_nearest_the_constrained_maximum(self):
        from viewactive.grid import Grid
        grid = Grid((0.0, 90.0, 180.0, 270.0), (0.0, 45.0, 90.0))
        field = np.zeros(grid.shape)
        field[2, 0] = 1.0  # global maximum above the band
        field[0, 1] = 0.5  # best view within elevations 0-45: azimuth 90, elevation 0
        source = {"image_id": 7, "azimuth": 0.0, "elevation": 0.0}
        # With a level input, candidate azimuth 180 is the input view (absolute 0),
        # so absolute azimuth 90 is candidate azimuth 270.
        candidates = [{"image_id": i, "azimuth": a, "elevation": 0.0}
                      for i, a in enumerate((0.0, 90.0, 180.0, 270.0))]
        picked = selectors.picks(source, candidates, field, grid, [0, 45], seed=0)
        self.assertEqual(picked["gt_vqf_max"][0], 3)
        self.assertAlmostEqual(picked["gt_vqf_max"][1]["error_deg"], 0.0, places=4)
        self.assertEqual(picked["identity"][0], 2)
        self.assertEqual(picked["random"], selectors.picks(source, candidates, None, grid, None, 0)["random"])
        self.assertNotIn("gt_vqf_max", selectors.picks(source, candidates, None, grid, None, 0))


class SelectorTests(unittest.TestCase):
    scores = {1: {"a": 0.9, "b": 0.1}, 2: {"a": 0.1, "b": 0.5}, 3: {"a": 0.1, "b": None}}
    margins = {1: -0.5, 2: 0.3, 3: 0.1}
    angles = {1: (0.0, 0.0), 2: (180.0, 0.0), 3: (180.0, 30.0)}

    def pick(self, name):
        return selectors.weights(name, [1, 2, 3], self.scores, self.margins, ["a", "b"], self.angles)

    def test_single_picks_ties_and_expectation(self):
        self.assertEqual(self.pick("oracle"), {2: 1.0})
        self.assertEqual(self.pick("identity"), {2: 1.0})
        self.assertEqual(self.pick("ambiguity:a"), {2: 0.5, 3: 0.5})
        self.assertEqual(self.pick("ambiguity:b"), {1: 1.0})
        self.assertEqual(self.pick("random"), {1: 1 / 3, 2: 1 / 3, 3: 1 / 3})

    def test_combined_uses_equal_weight_ranks_and_ranks_missing_last(self):
        # Ranks a: [1, .25, .25]; b (None last): [0, .5, 1] -> means [.5, .375, .625].
        self.assertEqual(self.pick("ambiguity:combined"), {2: 1.0})
        only_b = selectors.weights("ambiguity:combined", [1, 2, 3], self.scores, self.margins,
                                   ["a", "b"], self.angles, metric_weights={"b": 1.0})
        self.assertEqual(only_b, {1: 1.0})
        with self.assertRaisesRegex(ValueError, "Unknown selector"):
            self.pick("ambiguity:c")
        self.assertEqual(selectors.selector_names(["a"]),
                         ["identity", "random", "ambiguity:a", "detector_entropy",
                          "detector_gap", "oracle"])

    def test_detector_entropy_prefers_a_confident_view(self):
        self.assertAlmostEqual(selectors.detector_entropy({"a": 0.5, "b": 0.5}), 1.0)
        self.assertAlmostEqual(selectors.detector_entropy({"a": 0.9, "b": 0.0}), 0.0)
        self.assertIsNone(selectors.detector_entropy({"a": 0.0, "b": 0.0}))
        prompts = {1: {"a": 0.5, "b": 0.5}, 2: {"a": 0.9, "b": 0.1}, 3: {"a": 0.0, "b": 0.0}}
        self.assertEqual(selectors.weights("detector_entropy", [1, 2, 3], self.scores, self.margins,
                                           ["a", "b"], self.angles, prompts), {2: 1.0})

    def test_detector_gap_prefers_the_largest_top_two_gap(self):
        self.assertAlmostEqual(selectors.detector_gap({"a": 0.8, "b": 0.3, "c": 0.5}), 0.3)
        self.assertIsNone(selectors.detector_gap({"a": 0.8}))
        prompts = {1: {"a": 0.9, "b": 0.8}, 2: {"a": 0.2, "b": 0.7}, 3: {"a": 0.1, "b": 0.1}}
        self.assertEqual(selectors.weights("detector_gap", [1, 2, 3], self.scores, self.margins,
                                           ["a", "b"], self.angles, prompts), {2: 1.0})

    def test_gt_vqf_selectors_follow_the_field_within_the_elevation_band(self):
        from viewactive.grid import Grid

        grid = Grid((0.0, 90.0, 180.0, 270.0), (0.0, 30.0, 60.0))
        views = {(e, a): 10 * e + a for e in range(3) for a in range(4)}
        # Visual-entropy ambiguity: lowest (best) at (60, 180), next best at (30, 90).
        ambiguity = {i: 0.9 for i in views.values()}
        ambiguity[22], ambiguity[11] = 0.0, 0.2
        scores = {i: {"visual_entropy": a} for i, a in ambiguity.items()}
        field = selectors.gt_vqf_field(views, scores, ["visual_entropy"], grid)
        self.assertAlmostEqual(field[2, 2], 1.0)
        margins = {i: 0.01 * i for i in views.values()}
        settings = {"reach_cells": [1, 1], "steps": 4, "elevation_range": None}
        pick = lambda name, settings: selectors.gt_view_weights(name, (0, 0), views, field, margins,
                                                               grid, settings)
        self.assertEqual(pick("gt_vqf", settings), {22: 1.0})
        band = settings | {"elevation_range": [0, 30]}
        self.assertEqual(pick("gt_vqf", band), {11: 1.0})
        self.assertEqual(pick("gt_oracle", band), {13: 1.0})
        self.assertEqual(sum(pick("gt_random", band).values()), 1.0)
        self.assertEqual(len(pick("gt_random", band)), 8)


class ConfigTests(unittest.TestCase):
    def test_paths_defaults_validation_and_fingerprints(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.yaml"
            raw = {"output_dir": "out", "dataset": {"canonical": "canonical.json",
                                                    "categories": VOCABULARY}}
            path.write_text(yaml.safe_dump(raw))
            config = study_config.load(path)
            self.assertEqual(config["output_dir"], str(Path(directory).resolve() / "out"))
            self.assertEqual(config["ambiguity"]["self_occlusion"]["samples"], 20000)
            fingerprint = study_config.fingerprint(config, "reconstruct")
            raw["reconstruction"] = {"objects_per_category": 3, "input_elevations": [0]}
            raw["detection"] = {"score_threshold": 0.4}
            path.write_text(yaml.safe_dump(raw))
            changed = study_config.load(path)
            self.assertEqual(study_config.fingerprint(changed, "reconstruct"), fingerprint)
            self.assertEqual(study_config.fingerprint(changed, "detect"),
                             study_config.fingerprint(config, "detect"))
            for bad in ({"typo": 1}, {"dataset": {"categories": ["chair"]}},
                        {"reconstruction": {"candidate_azimuths": [0, 90]}},
                        {"reconstruction": {"input_elevations": [7]}},
                        {"analysis": {"primary_selector": "ambiguity:nope"}},
                        {"analysis": {"combined_weights": {"nope": 1}}},
                        {"analysis": {"combined_weights": {"visual_entropy": 0}}}):
                path.write_text(yaml.safe_dump({**raw, **bad}))
                with self.assertRaises(ValueError):
                    study_config.load(path)

    def test_manifest_split_and_incremental_external_renders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            shard = source / "renders" / "shard-000"
            shard.mkdir(parents=True)
            canonical = {"objects": {
                "seen": {"category": "chair", "face_count": 100},
                "unseen": {"category": "desk", "face_count": 100}}}
            (root / "canonical.json").write_text(json.dumps(canonical))
            (root / "objects.json").write_text(json.dumps({
                "seen": {"category": "chair", "split": "train"},
                "unseen": {"category": "desk", "split": "test_unseen"}}))
            _image(shard / "rgb.png")
            _image(shard / "mask.png", 255)
            row = {"uid": "unseen", "category": "desk", "azimuth": 0.0,
                   "elevation": 0.0, "modal_area": 64,
                   "files": {"rgb": "rgb.png", "modal": "mask.png"}}
            index = shard / "index.jsonl"
            index.write_text(json.dumps(row) + "\n")
            path = root / "study.yaml"
            path.write_text(yaml.safe_dump({
                "output_dir": "out", "dataset": {"canonical": "canonical.json",
                "manifest": "objects.json", "split": "test_unseen", "source_run": "source",
                "categories": ["chair", "desk"], "objects_per_category": 1,
                "azimuths": [0], "elevations": [0]}}))
            config = study_config.load(path)
            # A benchmark can contain unseen classes only; its vocabulary can
            # include more classes than the split, so use the eligible group.
            config["dataset"]["categories"] = ["desk"]
            groups = cohort(config)
            self.assertEqual(groups, {"desk": ["unseen"]})
            self.assertEqual(import_renders(config, groups), 1)
            self.assertEqual(import_renders(config, groups), 1)
            with store.connect(root / "out") as db:
                imported = store.images(db, "source='gt'")
                self.assertEqual(len(imported), 1)
                self.assertEqual(imported[0]["uid"], "unseen")
                self.assertTrue((root / "out" / imported[0]["path"]).is_file())

    def test_views_per_object_imports_hash_chosen_views_from_split_list(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "source" / "renders" / "shard-000"
            (root / "canonical.json").write_text(json.dumps({"objects": {
                "a": {"category": "chair", "face_count": 100},
                "b": {"category": "desk", "face_count": 100}}}))
            (root / "objects.json").write_text(json.dumps({
                "a": {"category": "chair", "split": "train"},
                "b": {"category": "desk", "split": "test_unseen"}}))
            rows = []
            for uid, category in (("a", "chair"), ("b", "desk")):
                for azimuth in (0.0, 90.0):
                    for elevation in (0.0, 30.0):
                        name = f"{uid}_{azimuth:g}_{elevation:g}"
                        _image(shard / f"{name}.rgb.png")
                        _image(shard / f"{name}.mask.png", 255)
                        rows.append({"uid": uid, "category": category, "azimuth": azimuth,
                                     "elevation": elevation, "modal_area": 64,
                                     "files": {"rgb": f"{name}.rgb.png", "modal": f"{name}.mask.png"}})
            (shard / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            path = root / "study.yaml"
            path.write_text(yaml.safe_dump({
                "output_dir": "out", "dataset": {"canonical": "canonical.json",
                "manifest": "objects.json", "split": ["train", "test_unseen"],
                "source_run": "source", "views_per_object": 1,
                "categories": ["chair", "desk"], "objects_per_category": 1,
                "azimuths": [0, 90], "elevations": [0, 30]}}))
            config = study_config.load(path)
            groups = cohort(config)
            self.assertEqual(groups, {"chair": ["a"], "desk": ["b"]})
            self.assertEqual(import_renders(config, groups), 2)
            with store.connect(root / "out") as db:
                imported = store.images(db, "source='gt'")
            self.assertEqual(
                {(r["uid"], r["azimuth"], r["elevation"]) for r in imported},
                {(uid, *view) for uid in "ab"
                 for view in object_views(uid, [0.0, 90.0], [0.0, 30.0], 1)})


def _image(path, value=128, size=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L" if path.name.endswith("mask.png") else "RGB", (size, size),
              value if path.name.endswith("mask.png") else (value,) * 3).save(path)
    return path


class AnalysisTests(unittest.TestCase):
    """A two-object study with two input views and three candidates per input."""

    def build(self, root):
        db = store.connect(root)
        store.set_meta(db, "vocabulary", VOCABULARY)
        # (category, uid, input margin, candidate margins, candidate ambiguity)
        objects = [("desk", "d1", [-0.2, 0.1], [0.3, -0.4, 0.0], [0.1, 0.9, 0.5]),
                   ("chair", "c1", [0.4, -0.1], [0.2, 0.1, -0.3], [0.2, 0.3, 0.9])]
        with db:
            for category, uid, input_margins, candidate_margins, ambiguity in objects:
                wrong = "table"
                for index, (azimuth, margin) in enumerate(zip((0.0, 90.0), input_margins)):
                    image = f"{uid}/a{azimuth:g}.png"
                    _image(root / image); _image(root / f"{uid}/a{azimuth:g}.mask.png", 255)
                    input_id = store.add_image(db, key=f"gt/{uid}/{index}", source="gt", uid=uid,
                                               category=category, azimuth=azimuth, elevation=0.0,
                                               path=image, mask_path=f"{uid}/a{azimuth:g}.mask.png",
                                               visible_px=64)
                    self.add(db, input_id, category, wrong, margin)
                    for c, (m, a) in enumerate(zip(candidate_margins, ambiguity)):
                        view = f"{uid}/{index}/c{c}.png"
                        _image(root / view)
                        cid = store.add_image(db, key=f"sam3d/{input_id}/{c}", source="sam3d",
                                              parent_id=input_id, uid=uid, category=category,
                                              azimuth=90.0 * c, elevation=0.0, path=view, visible_px=64)
                        self.add(db, cid, category, wrong, m)
                        store.add_scores(db, [(cid, "visual_entropy", a)])
        return db

    @staticmethod
    def add(db, image_id, category, wrong, margin):
        true = 0.5 + margin / 2
        scores = {prompt: 0.0 for prompt in VOCABULARY}
        scores[wrong], scores[category] = true - margin, true
        store.add_detections(db, image_id, {
            p: (s, 1, {"scores": [s], "ious": [0.9]} if p == category else None)
            for p, s in scores.items()})

    def test_selections_summary_and_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = self.build(root)
            config = {"output_dir": str(root), "detection": DETECTION,
                      "dataset": {"categories": VOCABULARY, "azimuths": [0.0, 90.0], "elevations": [0.0]},
                      "reconstruction": {"candidate_azimuths": [0.0, 90.0, 180.0],
                                         "candidate_elevations": [0.0]},
                      "ambiguity": {"visual_entropy": {"bins": 32}},
                      "analysis": {"primary_selector": "ambiguity:visual_entropy"}}
            selections = analyze.select_all(db, config)
            self.assertEqual(len(selections), 4)
            desk = selections[1]
            self.assertAlmostEqual(desk["input"]["margin"], -0.2)
            self.assertAlmostEqual(desk["ambiguity:visual_entropy"]["margin"], 0.3)
            self.assertAlmostEqual(desk["identity"]["margin"], 0.0)  # candidate at (180, 0)
            self.assertAlmostEqual(desk["oracle"]["margin"], 0.3)
            self.assertAlmostEqual(desk["random"]["margin"], (0.3 - 0.4 + 0.0) / 3)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM selections").fetchone()[0], 4 * 7)
            from study.compare import build as build_comparison
            page = root / "figures" / "comparison" / "oracle" / "d1.png"
            from study import figures
            with patch("study.figures.save_panels", wraps=figures.save_panels) as save:
                self.assertEqual(build_comparison(config, "d1", "oracle", page), page)
            panels = save.call_args.args[0]
            self.assertEqual([type(p).__name__ for p in panels],
                             ["Images", "Images", "Heatmap", "Heatmap"])
            self.assertEqual(save.call_args.kwargs["ncols"], 2)
            self.assertAlmostEqual(panels[2].values[(0.0, 0.0)], -0.2)
            self.assertAlmostEqual(panels[3].values[(0.0, 0.0)], 0.3)
            self.assertTrue(page.is_file())
            external = root / "figures" / "comparison" / "viewactive" / "d1.png"
            self.assertEqual(build_comparison(
                config, "d1", "viewactive", external,
                picks={"viewactive": {"gt/d1/0": "sam3d/1/0"}}), external)
            with self.assertRaisesRegex(ValueError, "not a candidate"):
                build_comparison(config, "d1", "viewactive", external,
                                 picks={"viewactive": {"gt/d1/0": "sam3d/1/99"}})

            objects = {i: ("d1" if i < 8 else "c1") for i in selections}
            summary = analyze.summarize(selections, objects, "ambiguity:visual_entropy")
            misleading = summary["subsets"]["misleading before (margin < 0)"]
            self.assertEqual(misleading["selectors"]["input"]["n"], 2)
            self.assertEqual(misleading["ambiguity:visual_entropy vs"]["input"]["improved"], 1.0)
            before = {"views": 4, "objects": 2, "margin": 0.05, "correct": 0.5, "any_fp": 0.5,
                      "fp_rate": 0.25, "tp": 0.5}
            analyze.write_markdown(summary, before, "ambiguity:visual_entropy", root / "summary.md")
            self.assertIn("| oracle |", (root / "summary.md").read_text())

            gt = labels.outcomes(db, VOCABULARY, DETECTION, "i.source='gt'")
            with patch("study.figures.save_panels", wraps=figures.save_panels) as save:
                paths = figures.study_figures(db, config, gt, selections, summary)
            object_call = next(call for call in save.call_args_list
                               if "/objects/desk/before/d1.png" in str(call.args[3]))
            self.assertEqual(object_call.kwargs["ncols"], 1)
            self.assertEqual(len(object_call.args[0]), 3)
            self.assertAlmostEqual(object_call.args[0][1].margins["table"], -0.2)
            self.assertTrue(all(Path(p).is_file() for p in paths))
            self.assertTrue((root / "figures/after/margins_all_inputs.png").is_file())
            for kind in ("before", "after", *figures.SELECTOR_FIGURES):
                self.assertTrue((root / f"figures/objects/desk/{kind}/d1.png").is_file())
            self.assertTrue(figures.reconstruction_figure(db, config, "gt/d1/0", root / "r.png").is_file())
            self.assertEqual(figures._higher_scoring_categories(
                {"desk": 0.4, "chair": 0.7, "table": 0.7}, "desk"), ("chair", "table"))
            self.assertEqual(figures._higher_scoring_categories(
                {"desk": 0.4, "chair": 0.4, "table": 0.2}, "desk"), ())
            db.close()



class PlotTests(unittest.TestCase):
    def test_panels_and_bars(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = _image(root / "view.png")
            values = {(0.0, 0.0): 0.2, (90.0, 0.0): None}
            path = save_panels([margin_heatmap(values, "margin", marks={(0.0, 0.0)}),
                                Heatmap(values, "score", 0, 1, annotate=True),
                                Images({(0.0, 0.0): image}, "views")],
                               [0, 90], [0], root / "grid.png", title="t", note="n", ncols=2)
            self.assertTrue(path.is_file())
            self.assertTrue(save_bars({"a": {"x": 0.5, "y": 0.2}}, root / "bars.png",
                                      title="t", ylabel="y").is_file())


class WorkerEntryTests(unittest.TestCase):
    def test_all_stages_start_with_an_empty_study(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = root / "config.json"
            settings.write_text(json.dumps({
                "output_dir": str(root), "gpu_id": 0,
                "dataset": {"categories": VOCABULARY},
                "reconstruction": {"objects_per_category": 1, "input_elevations": None,
                                   "candidate_distance": 2.0},
                "ambiguity": {"visual_entropy": {"bins": 32}},
            }))
            for stage in ("reconstruct", "score", "detect"):
                result = subprocess.run(
                    [sys.executable, "-m", "study.workers", "--stage", stage,
                     "--settings", str(settings)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, f"{stage}: {result.stderr}")


if __name__ == "__main__":
    unittest.main()
