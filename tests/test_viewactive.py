"""ViewActive reimplementation (CPU): grid frames, Algorithm 1, losses, labels,
model, cohort selection, config, evaluation metrics. Label tests need open3d."""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from study.config import load_viewactive
from tools.cameras import direction, sam3d_gravity
from viewactive import cohort as cohorts
from viewactive.grid import (Grid, interpolate, nearest, reachable, sam3d_to_viewactive,
                             to_absolute, to_relative)
from viewactive.labels import normalize
from viewactive.optimize import field_policy, next_view, random_policy, rollout

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_OPEN3D = importlib.util.find_spec("open3d") is not None
HAS_TIMM = importlib.util.find_spec("timm") is not None
CONFIGS = Path(__file__).resolve().parents[1] / "configs"
GRID = Grid()


def peaked(row, col, grid=GRID):
    """Smooth field with its maximum at (row, col)."""
    directions = grid.directions()
    return directions @ directions[row, col]


class GridTests(unittest.TestCase):
    def test_shape_and_frames_round_trip(self):
        self.assertEqual(GRID.shape, (11, 12))
        self.assertEqual(GRID.shape[0] * GRID.shape[1], 132)
        field = np.random.default_rng(0).random((3, 11, 12))
        for k in range(12):
            np.testing.assert_array_equal(to_absolute(to_relative(field, k), k), field)
            np.testing.assert_array_equal(to_relative(field, k)[..., 0], field[..., k])

    def test_reachable_wraps_azimuth_and_clamps_elevation(self):
        mask = reachable(GRID, 10, (1, 1))
        self.assertEqual(mask.sum(), 2 * 3)
        self.assertTrue(mask[10, 11] and mask[10, 1] and mask[9, 0])
        self.assertFalse(mask[8, 0] or mask[10, 2])
        self.assertTrue(reachable(GRID, 0, None).all())

    def test_nearest_picks_smallest_angle(self):
        mask = reachable(GRID, 5, (1, 1))
        self.assertEqual(nearest(GRID, mask, 5, 4), (5, 1))
        self.assertEqual(nearest(GRID, mask, 10, 0), (6, 0))

    def test_interpolate_hits_nodes_and_wraps(self):
        field = np.arange(132, dtype=float).reshape(11, 12)
        self.assertAlmostEqual(interpolate(field, GRID, 60, 15), field[6, 2])
        self.assertAlmostEqual(interpolate(field, GRID, 345, -75), (field[0, 11] + field[0, 0]) / 2)
        self.assertAlmostEqual(interpolate(field, GRID, 0, 90), field[10, 0])

    def test_sam3d_mapping(self):
        for elevation in (0, 30, 60):
            azimuth, el = sam3d_to_viewactive(180, 0, elevation)
            self.assertAlmostEqual(min(azimuth, 360 - azimuth), 0, places=4)
            self.assertAlmostEqual(el, elevation, places=4)
        self.assertEqual(tuple(round(v, 4) for v in sam3d_to_viewactive(0, 0, 0)), (180.0, 0.0))
        self.assertEqual(tuple(round(v, 4) for v in sam3d_to_viewactive(180, 30, 0)), (0.0, 30.0))
        self.assertEqual(tuple(round(v, 4) for v in sam3d_to_viewactive(210, 0, 0)), (30.0, 0.0))
        for azimuth in range(0, 360, 45):
            self.assertTrue(-90 <= sam3d_to_viewactive(azimuth, 60, 60)[1] <= 90)

    def test_sam3d_gravity_matches_absolute_elevation(self):
        # Candidate cameras use this as their up vector; its angle to each camera
        # direction must agree with the absolute elevation used for the band.
        for input_elevation in (0, 30, 60):
            gravity = sam3d_gravity(input_elevation)
            for azimuth in range(0, 360, 45):
                for elevation in (0, 30, 60):
                    height = float(direction(azimuth, elevation) @ gravity)
                    self.assertAlmostEqual(math.degrees(math.asin(np.clip(height, -1, 1))),
                                           sam3d_to_viewactive(azimuth, elevation, input_elevation)[1], places=3)


class OptimizeTests(unittest.TestCase):
    def test_reachable_argmax_is_taken(self):
        self.assertEqual(next_view(peaked(6, 1), 5, GRID), (6, 1))

    def test_unreachable_argmax_goes_to_nearest_reachable(self):
        self.assertEqual(next_view(peaked(5, 3), 5, GRID), (5, 1))
        self.assertEqual(next_view(peaked(10, 0), 3, GRID), (4, 0))

    def test_converged_view_stays(self):
        self.assertEqual(next_view(peaked(5, 0), 5, GRID), (5, 0))

    def test_rollout_reaches_peak_without_violating_reach(self):
        peak = (9, 4)
        field = peaked(*peak)
        policy = field_policy(lambda e, a: to_relative(field, a), GRID, (1, 1))
        views = rollout(policy, (2, 1), 9)
        steps = next(i for i, view in enumerate(views) if view == peak)
        self.assertEqual(steps, max(math.ceil(7 / 1), math.ceil(3 / 1)))
        self.assertTrue(all(view == peak for view in views[steps:]))
        for (e0, a0), (e1, a1) in zip(views, views[1:]):
            self.assertLessEqual(abs(e1 - e0), 1)
            self.assertLessEqual(min((a1 - a0) % 12, (a0 - a1) % 12), 1)
        walk = rollout(random_policy(GRID, (1, 1)), (0, 0), 20, np.random.default_rng(1))
        self.assertTrue(all(a != b for a, b in zip(walk, walk[1:])))

    def test_elevation_range_limits_targets_and_climbs_from_outside(self):
        field = np.maximum(peaked(2, 0), 0.9 * peaked(7, 3))   # global max below, local above
        self.assertEqual(next_view(field, 2, GRID, None), (2, 0))
        self.assertEqual(next_view(field, 2, GRID, None, (0, 45)), (7, 3))
        policy = field_policy(lambda e, a: to_relative(field, a), GRID, (1, 1), (0, 45))
        views = rollout(policy, (0, 0), 9)
        self.assertEqual(views[-1], (7, 3))
        self.assertTrue(all(5 <= e <= 8 for e, _ in views[views.index((5, 3)):]))
        # Target on the far side of the bottom pole: still climbs one row per step.
        far = field_policy(lambda e, a: to_relative(peaked(7, 6), a), GRID, (1, 1), (0, 45))
        self.assertEqual([e for e, _ in rollout(far, (0, 0), 5)], [0, 1, 2, 3, 4, 5])
        walk = rollout(random_policy(GRID, (1, 1), (0, 45)), (0, 0), 20, np.random.default_rng(1))
        self.assertTrue(all(5 <= e <= 8 for e, _ in walk[5:]))


class CropExperimentTests(unittest.TestCase):
    def test_scores_pair_the_same_objects_and_best_quality_views(self):
        from viewactive.crop_experiment import score, summarize

        quality = np.zeros((2, 3, 2, 2))
        quality[:, :, 0, 1] = 1
        full = np.array([[[0, 1], [0, 0]], [[0, 0], [0, 1]]])
        cropped = np.array([[[0, 1], [0, 1]], [[0, 1], [0, 1]]])
        categories = np.array(["chair", "chair"])
        mapping = {"chair": [1]}
        before = score(full, categories, mapping, quality)
        after = score(cropped, categories, mapping, quality)
        report = summarize(before, after, np.array(["test", "test_unseen"]))
        self.assertEqual(report["seen"]["metrics"]["all_views"]["difference"], 0.25)
        self.assertEqual(report["unseen"]["metrics"]["best_quality_view"]["difference"], 1.0)
        self.assertEqual(report["seen"]["metrics"]["any_view"]["difference"], 0.0)


class LabelTests(unittest.TestCase):
    def test_normalize_range_nan_and_constant_channel(self):
        raw = np.random.default_rng(0).random((3, 11, 12)).astype(np.float32)
        raw[0, 0, 0] = np.nan
        raw[2] = 0.7
        norm = normalize(raw, floor=0.05)
        self.assertTrue(np.isnan(norm[0, 0, 0]))
        self.assertAlmostEqual(float(np.nanmin(norm[:2])), 0.05, places=6)
        self.assertAlmostEqual(float(np.nanmax(norm[:2])), 1.0, places=6)
        np.testing.assert_allclose(norm[2], 0.525)

    @unittest.skipUnless(HAS_OPEN3D, "open3d is not installed")
    def test_cube_quality(self):
        from viewactive.labels import raw_quality
        from tools.cameras import HARNESS_LOOK_AT

        corners = np.array([[x, y, z] for x in (-0.3, 0.3) for y in (-0.3, 0.3) for z in (-0.3, 0.3)])
        vertices = corners + np.asarray(HARNESS_LOOK_AT)
        faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                          [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
        grid = Grid(tuple(float(a) for a in range(0, 360, 45)), (0.0, 30.0))
        with tempfile.TemporaryDirectory() as directory:
            rgb, mask = Path(directory, "rgb.png"), Path(directory, "mask.png")
            Image.new("RGB", (8, 8), "gray").save(rgb)
            Image.new("L", (8, 8), 255).save(mask)
            views = [[(rgb, mask)] * 8 for _ in range(2)]
            raw = raw_quality(vertices, faces, grid, views, distance=2.2, samples=20000, seed=0,
                              polar_bins=9, azimuth_bins=18, visual_bins=32)
        self.assertAlmostEqual(raw[0, 0, 0], raw[0, 0, 2], delta=0.02)   # azimuth 0 vs 90
        self.assertAlmostEqual(raw[0, 0, 0], 1 / 6, delta=0.03)          # one face visible
        self.assertGreater(raw[1, 1, 1], raw[1, 0, 0])                    # corner > face-on
        np.testing.assert_allclose(raw[2], 0.0)                           # uniform gray


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class LossAndModelTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.target = torch.rand(4, 3, 11, 12) * 0.9 + 0.05
        self.mask = torch.ones_like(self.target, dtype=torch.bool)

    def test_losses_vanish_at_target_and_dssim_bounded(self):
        from viewactive import losses
        loss, parts = losses.total(self.target.clone(), self.target, self.mask)
        self.assertLess(float(loss), 1e-5)
        self.assertTrue(all(abs(v) < 1e-5 for v in parts.values()))
        value = float(losses.dssim(self.torch.rand(4, 3, 11, 12), self.target, self.mask))
        self.assertTrue(0 <= value <= 1)

    def test_silog_is_scale_invariant_at_lambda_one(self):
        from viewactive import losses
        self.assertLess(abs(float(losses.silog(self.target * 0.5, self.target, self.mask, lam=1.0))), 1e-5)
        self.assertGreater(float(losses.silog(self.target * 0.5, self.target, self.mask, lam=0.85)), 1e-3)

    def test_nan_targets_are_masked_with_finite_gradients(self):
        from viewactive import losses
        target = self.target.clone()
        target[:, :, 0, 0] = float("nan")
        pred = self.torch.zeros_like(target, requires_grad=True)
        loss, _ = losses.total(pred, target, self.torch.isfinite(target))
        loss.backward()
        self.assertTrue(self.torch.isfinite(loss) and self.torch.isfinite(pred.grad).all())

    def test_relative_targets_match_grid_roll(self):
        from viewactive.train import relative_targets
        fields = self.torch.rand(5, 3, 11, 12)
        az = self.torch.tensor([0, 3, 7, 11, 5])
        rolled = relative_targets(fields, az).numpy()
        for i, k in enumerate(az.tolist()):
            np.testing.assert_array_equal(rolled[i], to_relative(fields[i].numpy(), k))

    def test_decoder_size_and_predictor_output(self):
        from viewactive.model import Decoder, ViewActive, ViewActiveNet
        self.assertTrue(400_000 <= sum(p.numel() for p in Decoder(1280, 320, 132).parameters()) <= 600_000)
        net = ViewActiveNet(16, 8, (11, 12)).eval()
        torch = self.torch

        class Dummy:
            def encode(self, images):
                return torch.randn(len(images), 16)

        model = ViewActive(net, Dummy(), GRID)
        field = model.predict([Image.new("RGB", (8, 8))] * 2)
        self.assertEqual(field.shape, (2, 3, 11, 12))
        self.assertTrue(((field > 0) & (field < 1)).all())
        self.assertEqual(model.combined(field).shape, (2, 11, 12))
        self.assertEqual(model.score_views(field[0], [0, 30], [15, 15]).shape, (2,))
        masks = np.empty(2, dtype=object)
        masks[:] = [Image.new("L", (8, 8), 255)] * 2
        self.assertEqual(model.predict([Image.new("RGB", (8, 8))] * 2, masks).shape, field.shape)
        with self.assertRaises(ValueError):
            model.predict([Image.new("RGB", (8, 8))] * 2, [None])


class CohortTests(unittest.TestCase):
    def setUp(self):
        self.lvis = {c: [f"{c}{i:03d}" for i in range(n)]
                     for c, n in (("a", 40), ("b", 30), ("c", 20), ("d", 10), ("e", 5))}
        self.class_map = {c: [i] for i, c in enumerate("abcd")}
        self.cohort = {"seen_classes": 2, "unseen_classes": 1, "per_class": 10, "unseen_per_class": 5,
                       "total_seen": None, "oversample": 1.5, "split": [0.7, 0.1, 0.2], "seed": 0}

    def test_class_choice_ranks_and_keeps_unseen_disjoint(self):
        seen, unseen = cohorts.choose_classes(self.cohort, self.lvis, self.class_map)
        self.assertEqual((seen, unseen), (["a", "b"], ["c"]))
        explicit = self.cohort | {"unseen_classes": ["a"]}
        self.assertEqual(cohorts.choose_classes(explicit, self.lvis, self.class_map), (["b", "c"], ["a"]))
        with self.assertRaises(ValueError):
            cohorts.choose_classes(self.cohort | {"seen_classes": ["e"]}, self.lvis, self.class_map)

    def test_paper_scale_split_has_exact_global_ratios(self):
        counts = {f"class_{i:02d}": 29 if i < 32 else 28 for i in range(56)}
        test = cohorts._apportion(counts, 0.25, 400)
        remaining = {name: counts[name] - test[name] for name in counts}
        val = cohorts._apportion(counts, 0.05, 80, remaining)
        self.assertEqual(sum(test.values()), 400)
        self.assertEqual(sum(val.values()), 80)
        self.assertEqual(sum(counts.values()) - sum(test.values()) - sum(val.values()), 1120)
        self.assertTrue(all(test[name] + val[name] <= counts[name] for name in counts))

    def test_candidates_exclude_and_prefer_local(self):
        excluded = {"a000", "a001"}
        pairs = cohorts.candidates(self.cohort, self.lvis, ["a", "b"], ["c"], excluded, local={"b029"})
        uids = [u for u, _ in pairs]
        self.assertFalse(excluded & set(uids))
        self.assertEqual(len(uids), len(set(uids)))
        self.assertEqual([u for u, c in pairs if c == "b"][0], "b029")
        self.assertEqual(sum(c == "a" for _, c in pairs), 15)
        self.assertEqual(pairs, cohorts.candidates(self.cohort, self.lvis, ["a", "b"], ["c"], excluded,
                                                   local={"b029"}))

    def test_finalize_filters_splits_and_trims(self):
        pairs = cohorts.candidates(self.cohort, self.lvis, ["a", "b"], ["c"])
        canonical = {u: {"face_count": 10, "up_axis_suspect": False} for u, _ in pairs}
        heavy = [u for u, c in pairs if c == "a"][0]
        canonical[heavy]["face_count"] = 10 ** 6
        suspect = [u for u, c in pairs if c == "a"][1]
        canonical[suspect]["up_axis_suspect"] = True
        objects = cohorts.finalize(self.cohort, pairs, canonical, ["a", "b"], ["c"], max_faces=1000)
        self.assertNotIn(heavy, objects)
        self.assertIn(suspect, objects)
        self.assertNotIn(suspect, cohorts.finalize(self.cohort, pairs, canonical, ["a", "b"], ["c"], 1000,
                                                   drop_suspect=True))
        splits = {}
        for record in objects.values():
            splits.setdefault((record["category"], record["split"]), 0)
            splits[(record["category"], record["split"])] += 1
        self.assertEqual(splits[("a", "train")], 7)
        self.assertEqual(splits[("a", "test")], 2)
        self.assertEqual(splits[("b", "val")], 1)
        self.assertEqual(splits[("c", "test_unseen")], 5)
        self.assertEqual(objects, cohorts.finalize(self.cohort, pairs, canonical, ["a", "b"], ["c"], 1000))
        trimmed = cohorts.finalize(self.cohort | {"total_seen": 19}, pairs, canonical, ["a", "b"], ["c"], 1000)
        self.assertEqual(sum(r["category"] == "b" for r in trimmed.values()), 9)
        limited = {uid: record for uid, record in canonical.items()
                   if uid not in [u for u, category in pairs if category == "a"][8:]}
        capped = cohorts.finalize(self.cohort | {"total_seen": 16}, pairs, limited,
                                  ["a", "b"], ["c"], 1000)
        self.assertEqual(sum(r["split"] != "test_unseen" for r in capped.values()), 16)
        self.assertEqual(sum(r["category"] == "a" for r in capped.values()), 7)
        with self.assertRaises(ValueError):
            cohorts.finalize(self.cohort | {"total_seen": 18}, pairs, limited,
                             ["a", "b"], ["c"], 1000)
        with self.assertRaises(ValueError):
            cohorts.finalize(self.cohort | {"per_class": 15}, pairs, canonical, ["a", "b"], ["c"], 1000)

    def test_class_map_parses_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "map.yaml")
            path.write_text("dog: {151-153: dogs}\nmug: {504: coffee mug}\n")
            self.assertEqual(cohorts.load_class_map(path), {"dog": [151, 152, 153], "mug": [504]})

    @unittest.skipUnless(HAS_TIMM, "timm is not installed")
    def test_reviewed_class_map_matches_imagenet_descriptions(self):
        import yaml
        from timm.data import ImageNetInfo

        info = ImageNetInfo()
        raw = yaml.safe_load((CONFIGS / "viewactive_imagenet_map.yaml").read_text())
        for category, classes in raw.items():
            for index, name in classes.items():
                if isinstance(index, int):
                    self.assertTrue(info.index_to_description(index).startswith(name), (category, index, name))

    def test_correctness_uses_mapped_classes(self):
        from viewactive.classify import correct
        np.testing.assert_array_equal(correct(np.array([504, 505]), "mug", {"mug": [504]}), [True, False])
        self.assertFalse(correct(np.array([1]), "unmapped", {}).any())


class ConfigTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "config.yaml")
            path.write_text(text)
            return load_viewactive(path)

    def test_defaults_and_shipped_configs(self):
        config = self.load("output_dir: out\n")
        self.assertEqual(len(config["render"]["azimuths"]) * len(config["render"]["elevations"]), 132)
        for name in ("viewactive.yaml", "viewactive_pilot.yaml", "viewactive_smoke.yaml"):
            self.assertTrue(load_viewactive(CONFIGS / name)["cohort"]["class_map"].endswith(
                "viewactive_imagenet_map.yaml"))
        blender = load_viewactive(CONFIGS / "viewactive_blender.yaml")
        self.assertEqual(blender["render"]["backend"], "blender")
        self.assertEqual((blender["render"]["width"], blender["render"]["height"]), (1920, 1680))

    def test_rejects_invalid_settings(self):
        for text in ("output_dir: out\nbogus: 1\n",
                     "output_dir: out\nrender: {azimuths: [0, 30, 90]}\n",
                     "output_dir: out\nrender: {azimuths: [0, 90, 180]}\n",
                     "output_dir: out\ntrain: {loss_weights: [0, 0, 0]}\n",
                     "output_dir: out\ncohort: {split: [0.5, 0.5, 0.5]}\n",
                     "output_dir: out\ncohort: {seen_classes: [a], unseen_classes: [a]}\n",
                     "output_dir: out\noptimize: {reach_cells: [1]}\n",
                     "output_dir: out\noptimize: {elevation_range: [45, 0]}\n",
                     "output_dir: out\noptimize: {elevation_range: [80, 90]}\n",
                     "output_dir: out\nclassification: {crop_margin: -0.1}\n"):
            with self.assertRaises(ValueError, msg=text):
                self.load(text)


class EvaluationTests(unittest.TestCase):
    def test_classifier_crop_centers_object_and_pads_white(self):
        from viewactive.classify import crop_object

        image = Image.new("RGB", (40, 40), "white")
        mask = Image.new("L", (40, 40), 0)
        for x in range(2, 6):
            for y in range(16, 24):
                image.putpixel((x, y), (255, 0, 0))
                mask.putpixel((x, y), 255)
        cropped = crop_object(image, mask, margin=0.25)
        self.assertEqual(cropped.size, (12, 12))
        self.assertEqual(cropped.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(sum(pixel == (255, 0, 0) for pixel in cropped.getdata()), 32)
        with self.assertRaises(ValueError):
            crop_object(image, mask, margin=-0.1)

    def test_field_metrics_perfect_and_prior(self):
        from viewactive.evaluate import field_metrics, spearman
        truth = np.random.default_rng(0).random((6, 3, 11, 12))
        truth[0, 0, 0, 0] = np.nan
        perfect = field_metrics(truth.copy(), truth)
        self.assertAlmostEqual(perfect["combined"]["top1"], 1.0)
        self.assertAlmostEqual(perfect["combined"]["regret"], 0.0)
        self.assertAlmostEqual(perfect["visible_ratio"]["spearman"], 1.0)
        self.assertAlmostEqual(float(spearman(np.arange(5.0)[None], -np.arange(5.0)[None])[0]), -1.0)
        flat = field_metrics(np.full_like(truth, 0.5), truth)
        self.assertGreater(flat["combined"]["regret"], 0)

    def test_rollouts_follow_the_truth(self):
        from viewactive.evaluate import relative_truth, rollouts, summarize
        from viewactive.grid import combine
        fields = np.stack([np.stack([peaked(9, 4)] * 3), np.stack([peaked(2, 7)] * 3)])
        truth = combine(relative_truth(fields))
        policies = {"gt_field": lambda o: field_policy(lambda e, a: truth[o, a], GRID, (1, 1)),
                    "stay": lambda o: (lambda e, a, rng: (e, a))}
        starts = np.array([[[1, 0], [5, 6]], [[10, 0], [3, 3]]])
        correct = {"clf": np.zeros((2, 11, 12), dtype=bool)}
        correct["clf"][0, 9, 4] = correct["clf"][1, 2, 7] = True
        results = rollouts(policies, starts, 9, fields, fields, correct)
        self.assertEqual(results["gt_field"]["combined"].shape, (2, 10))
        np.testing.assert_allclose(results["gt_field"]["accuracy"]["clf"][:, -1], 1.0)
        np.testing.assert_allclose(results["stay"]["accuracy"]["clf"], 0.0)
        summary = summarize({"viewactive": results["gt_field"], "stay": results["stay"]})
        self.assertGreater(summary["viewactive_minus"]["stay"]["mean_accuracy"]["mean"], 0.9)

    def test_undefined_quality_cell_does_not_poison_rollout(self):
        from viewactive.evaluate import bootstrap, rollouts, summarize
        fields = np.full((2, 3, 11, 12), 0.6)
        fields[0, 1, 5, 0] = np.nan
        starts = np.array([[[5, 0], [5, 1]], [[5, 0], [5, 1]]])
        results = rollouts({"stay": lambda o: (lambda e, a, rng: (e, a))}, starts, 1,
                           fields, fields, {})
        self.assertAlmostEqual(results["stay"]["channels"][0, 0, 1], 0.6)
        self.assertAlmostEqual(results["stay"]["combined"][0, 0], 0.6)
        curve = bootstrap(np.array([[np.nan, 0.7], [0.5, 0.9]]), rounds=100)
        self.assertAlmostEqual(curve["mean"][0], 0.5)
        self.assertAlmostEqual(curve["mean"][1], 0.8)
        summary = summarize(results)
        self.assertTrue(np.isfinite(summary["stay"]["combined"]["mean"]).all())


class BlenderTests(unittest.TestCase):
    def test_camera_pose_and_alpha_conversion(self):
        from viewactive.blender import _camera, _convert
        from tools.cameras import HARNESS_LOOK_AT, direction

        rotation, translation = _camera(90, 30, 1.4)
        rotation, translation = np.asarray(rotation), np.asarray(translation)
        np.testing.assert_allclose(-rotation @ translation,
                                   np.asarray(HARNESS_LOOK_AT) + 1.4 * direction(90, 30), atol=1e-6)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        self.assertGreater(np.linalg.det(rotation), 0)
        np.testing.assert_allclose(_camera(0, 0, 1.4)[0], np.eye(3), atol=1e-6)
        with tempfile.TemporaryDirectory() as temporary:
            raw, rgb, mask = (Path(temporary) / name for name in ("raw.png", "rgb.png", "mask.png"))
            source = Image.new("RGBA", (2, 1), (200, 20, 10, 0))
            source.putpixel((1, 0), (20, 40, 60, 255))
            source.save(raw)
            _convert(raw, rgb, mask)
            self.assertEqual(Image.open(rgb).getpixel((0, 0)), (255, 255, 255))
            self.assertEqual(Image.open(rgb).getpixel((1, 0)), (20, 40, 60))
            self.assertEqual(list(Image.open(mask).getdata()), [0, 255])


if __name__ == "__main__":
    unittest.main()
