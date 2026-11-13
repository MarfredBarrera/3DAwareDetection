import unittest
from dataclasses import replace

from perception.qwen import Decision, UncertaintyGate, verify_if_needed
from scripts.qwen_verification_study import (
    episode_disjoint_cohort, episode_id, merge_label_sources,
    require_complete_inference, scalar_metrics_for_labels,
)

# sha1 prefix of prompt("chair", True) before the multi-view prompts were added (three-way-v1)
SINGLE_VIEW_PROMPT_SHA1 = "01d3bd2c6108"


class QwenGateTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"margin": m, "area": a, "view_flip": flip}
            for m, a, flip in [(-.9, .4, 0.0), (-.8, .3, .25), (-.7, .2, .5),
                              (-.6, .1, .75), (-.5, .05, 1.0)]
        ]
        self.gate = UncertaintyGate.fit(self.rows)

    def test_zero_flip_is_observed_and_scored_as_three_signals(self):
        self.assertEqual(len(self.gate.references), 3)
        self.assertIn(0.0, self.gate.references[2])
        got = self.gate.assess(-.9, .4, 0.0)
        self.assertEqual(got["reason"], "three_signals")
        self.assertIsNotNone(got["score"])

    def test_missing_flip_uses_two_feature_threshold(self):
        got = self.gate.assess(-.9, .4, None)
        self.assertEqual(got["reason"], "missing_flip_two_signal_gate")
        self.assertIn("2", self.gate.thresholds)

    def test_ties_have_equal_midranks(self):
        self.assertEqual(UncertaintyGate.percentile([0, 1, 1, 2], 1), .5)

    def test_unflagged_request_does_not_call_judge(self):
        calls=[]
        gate=UncertaintyGate(self.gate.references,{"2":1.1,"3":1.1})
        result, assessment=verify_if_needed(gate, -.9, .4, 0.0, True,
                                             lambda: calls.append(1))
        self.assertFalse(assessment["review"])
        self.assertEqual(calls, [])
        self.assertEqual((result.status,result.source),("correct","sam3"))

    def test_flagged_request_calls_judge_once_for_all_three_outcomes(self):
        gate=UncertaintyGate(self.gate.references,{"2":0.0,"3":0.0})
        for status in ("correct","wrong","insufficient"):
            calls=[]
            d,a=verify_if_needed(gate,-.65,.15,.5,True,
                lambda s=status:(calls.append(s) or {"status":s}))
            self.assertTrue(a["review"])
            self.assertEqual(calls,[status])
            self.assertEqual(d.status,status)

    def test_bad_features_fail_closed_to_review(self):
        self.assertTrue(self.gate.assess(float("nan"),.1,.1)["review"])
        self.assertTrue(self.gate.assess(-.1,-.1,.1)["review"])

    def test_insufficient_never_maps_to_accept_or_reject(self):
        d=Decision("insufficient", "qwen")
        self.assertIsNone(d.accepted)
        self.assertEqual(d.navigation_action(checks=1,max_checks=3), "acquire_new_view")
        self.assertEqual(d.navigation_action(checks=3,max_checks=3), "explore_unresolved")

    def test_underlying_episode_normalizes_strategy_prefix(self):
        self.assertEqual(episode_id({"set":"offline","episode":"offline_epi133","name":"x"}),"epi133")
        self.assertEqual(episode_id({"set":"online","episode":"vqf_epi133","name":"x"}),"epi133")

    def test_cohort_excludes_calibration_episode_across_strategy_names(self):
        rows=[{"set":"online","episode":"sam3_epi1","name":"sam3_epi1_x"},
              {"set":"online","episode":"vqf_epi1","name":"vqf_epi1_y"},
              {"set":"online","episode":"sam3_epi2","name":"sam3_epi2_x"}]
        self.assertEqual([r["name"] for r in episode_disjoint_cohort(rows,{"epi1"})],["sam3_epi2_x"])

    def test_review_label_overrides_both_blind_passes(self):
        got=merge_label_sources({"7":{"a":"goal","b":"not_goal","review":"cant_tell"},
                                 "8":{"a":"goal","b":"not_goal"}})
        self.assertEqual(got,{"7":"cant_tell","8":"not_goal"})

    def test_numeric_confusion_metrics_include_third_label_and_errors(self):
        rows=[
            {"name":"g","source_index":0,"raw_accept":"False","gate_review":"False"},
            {"name":"ng","source_index":1,"raw_accept":"True","gate_review":"True"},
            {"name":"ct","source_index":2,"raw_accept":"True","gate_review":"True"},
        ]
        labels={"0":"goal","1":"not_goal","2":"cant_tell"}
        pred={"g":"correct","ng":"correct","ct":"insufficient"}
        m=scalar_metrics_for_labels(rows,pred,labels)
        self.assertEqual(m["known_false_accepts"],1)
        self.assertEqual(m["raw_errors_repaired"],1)
        self.assertEqual(m["raw_correct_introduced_errors"],0)
        self.assertEqual(m["cant_tell_deferral_recall"],1.0)
        self.assertEqual(m["decided_coverage_clear"],1.0)

    def test_analysis_rejects_missing_diagnostic_arm(self):
        rows=[{"name":"flagged","gate_review":"True"}]
        decisions={("flagged","input"): {"status":"correct"},
                   ("flagged","matching_grid"): {"status":"wrong"}}
        with self.assertRaisesRegex(RuntimeError,"inference incomplete"):
            require_complete_inference(rows,decisions)
        decisions[("flagged","mismatch_grid")]={"status":"insufficient"}
        require_complete_inference(rows,decisions)

    def test_calibration_reference_does_not_change_with_evaluation_rows(self):
        before=self.gate.assess(-.75,.18,.5)
        # Gate only stores frozen references; an arbitrary eval row cannot mutate it.
        _=self.gate.assess(-.01,.99,1.0)
        self.assertEqual(before,self.gate.assess(-.75,.18,.5))


class MultiviewPromptTests(unittest.TestCase):
    """qwen.prompt_views / views_content (selector d1, close-range multi-view)."""

    def test_content_interleaves_captions_and_images(self):
        from perception.qwen import PROMPT_VERSION_VIEWS, ANSWER_ABC, views_content
        content, version = views_content(["i1", "i2", "i3"], "bed", distances=[1.04, 2.0, None])
        self.assertEqual(version, PROMPT_VERSION_VIEWS)
        self.assertEqual([c["type"] for c in content], ["text", "image"] * 3 + ["text"])
        self.assertEqual([c["text"] for c in content[:-1:2]],
                         ["Image 1 (about 1.0 m away):", "Image 2 (about 2.0 m away):", "Image 3:"])
        self.assertEqual([c["image"] for c in content[1::2]], ["i1", "i2", "i3"])
        text = content[-1]["text"]
        self.assertIn("Images 1 to 3 are robot camera observations of the SAME physical object", text)
        self.assertIn("'bed'", text)
        self.assertTrue(text.endswith(ANSWER_ABC))
        self.assertNotIn("reconstruction", text)

    def test_grid_and_crops(self):
        from perception.qwen import PROMPT_VERSION_VIEWS_GRID, views_content
        content, version = views_content(["i1", "i2"], "tv", grid="g", crops=[None, "c2"])
        self.assertEqual(version, PROMPT_VERSION_VIEWS_GRID)
        self.assertEqual([c["image"] for c in content if c["type"] == "image"], ["i1", "i2", "c2", "g"])
        text = content[-1]["text"]
        self.assertIn("NOT a camera observation", text)
        self.assertIn("camera images take precedence", text)
        self.assertIn("enlarged crop", text)
        self.assertIn("Images 1 to 2", text)
        with self.assertRaises(ValueError):
            views_content([], "tv")
        with self.assertRaises(ValueError):
            views_content(["i1"], "tv", crops=[None, None])

    def test_forced2_drops_insufficient(self):
        from perception.qwen import prompt_views
        text = prompt_views("bed", 4, forced2=True)
        self.assertNotIn("C: insufficient", text)
        self.assertTrue(text.endswith("Answer with only A or B."))
        self.assertIn("C: insufficient", prompt_views("bed", 4))

    def test_single_view_prompt_unchanged(self):
        from perception.qwen import prompt
        text = prompt("chair", True)
        self.assertTrue(text.startswith("Image 1 is a robot camera observation."))
        self.assertTrue(text.endswith("Lack of evidence for the target is not evidence that it is wrong. "
                                      "Answer with only A, B, or C."))
        self.assertEqual(__import__("hashlib").sha1(text.encode()).hexdigest()[:12], SINGLE_VIEW_PROMPT_SHA1)

    def test_crop_around(self):
        from PIL import Image
        from perception.qwen import crop_around
        image = Image.new("RGB", (640, 480))
        mask = Image.new("L", (640, 480))
        mask.paste(255, (600, 10, 620, 30))  # 20 px near the top-right corner
        crop = crop_around(image, mask)
        self.assertEqual(crop.size, (448, 448))  # 160 px minimum, enlarged to 448
        mask = Image.new("L", (640, 480))
        mask.paste(255, (100, 100, 400, 400))
        self.assertEqual(max(crop_around(image, mask).size), 448)
        with self.assertRaises(ValueError):
            crop_around(image, Image.new("L", (640, 480)))


class AggregateViewsTests(unittest.TestCase):
    def test_weights_and_vote(self):
        from perception.select import aggregate_views
        views = [{"scores": {"bed": 0.8, "couch": 0.1}, "area": 0.2, "clipped": False},   # w 1,   m 0.7
                 {"scores": {"bed": 0.2, "couch": 0.6}, "area": 0.05, "clipped": False},  # w 0.5, m -0.4
                 {"scores": {"bed": 0.5, "couch": 0.3}, "area": 0.2, "clipped": True}]    # w 0.7, m 0.2
        got = aggregate_views(views, "bed")
        self.assertAlmostEqual(got["margin"], (0.7 - 0.2 + 0.14) / 2.2)
        self.assertEqual((got["vote"], got["n_used"], got["label"]), (2 / 3, 3, "bed"))

    def test_unmatched_views_excluded_and_negative_label(self):
        from perception.select import aggregate_views
        views = [{"scores": {"bed": 0.0, "couch": 0.0}, "area": 0.3, "clipped": False},
                 {"scores": None, "area": 0.3, "clipped": False},
                 {"scores": {"bed": 0.1, "couch": 0.7}, "area": 0.01, "clipped": False}]
        got = aggregate_views(views, "bed")
        self.assertAlmostEqual(got["margin"], -0.6)
        self.assertEqual((got["vote"], got["n_used"], got["label"]), (0.0, 1, "couch"))
        self.assertEqual(aggregate_views([], "bed"), {"margin": None, "vote": None, "n_used": 0, "label": None})

    def test_fuse_views(self):
        """jev_sam3's multi-view rule: evidential opinions fused across views; the target must be the fused top."""
        from perception.select import fuse_views
        # a real couch: one clear view, two small views where chair barely leads (the d1mv vote vetoed this)
        a = [{"scores": {"couch": 0.60, "chair": 0.30}, "area": 0.12}, {"scores": {"couch": 0.20, "chair": 0.30}, "area": 0.01},
             {"scores": {"couch": 0.25, "chair": 0.30}, "area": 0.05}]
        got = fuse_views(a, "couch")
        self.assertEqual((got["accept"], got["label"], got["n_used"]), (True, "couch", 3))
        self.assertAlmostEqual(sum(got["belief"].values()) + got["uncertainty"], 1.0, places=5)
        # a look-alike that barely wins two views and loses the pooled evidence (the d1mv rule accepted this)
        b = [{"scores": {"couch": 0.45, "chair": 0.40}, "area": 0.12}, {"scores": {"couch": 0.42, "chair": 0.40}, "area": 0.12},
             {"scores": {"couch": 0.20, "chair": 0.50}, "area": 0.02}]
        self.assertEqual(fuse_views(b, "couch")["label"], "chair")
        self.assertEqual(fuse_views(b[::-1], "couch")["belief"], fuse_views(b, "couch")["belief"])   # order-free
        # views where SAM3 matched nothing are left out; none left -> no accept
        self.assertEqual(fuse_views([{"scores": {"couch": 0.0}, "area": 0.1}], "couch"),
                         {"accept": False, "label": None, "belief": None, "uncertainty": None, "n_used": 0})


if __name__ == "__main__": unittest.main()
