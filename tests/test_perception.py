"""perception.Verifier decisions with fake model workers, and the ensemble disagreement (CPU)."""
import unittest
from unittest.mock import patch

import numpy as np

from perception import client
from perception.images import encode

IMAGE = np.full((8, 8, 3), 200, dtype=np.uint8)
MASK = np.ones((8, 8), dtype=bool)
# SAM3 scores per rendered candidate: (180, 0) looks like a chair, (0, 0) is a
# confident table (largest top-2 gap), (90, 0) a weak table.
VIEW_SCORES = {(0.0, 0.0): {"chair": 0.1, "table": 0.9},
               (90.0, 0.0): {"chair": 0.4, "table": 0.5},
               (180.0, 0.0): {"chair": 0.6, "table": 0.3}}
QUALITY = {(0.0, 0.0): 0.2, (90.0, 0.0): 0.9, (180.0, 0.0): 0.5}


class FakeWorker:
    uncertainty = 0.05

    def __init__(self, env, args, launcher, log):
        self.role = args[1]
        self.rendered = []

    def wait_ready(self):
        pass

    def close(self):
        pass

    def call(self, op, **args):
        if op == "vqf":
            return [[QUALITY[(a, e)], e] for a, e in args["angles"]]
        if op == "reconstruct":
            angles = [(a, e) for e in args["elevations"] for a in args["azimuths"]]
            if args["max_uncertainty"] is not None and self.uncertainty > args["max_uncertainty"]:
                angles = []
            self.rendered.append(angles)
            # Each view image encodes its azimuth in the red channel.
            return {"uncertainty": self.uncertainty, "timing": {},
                    "views": [{"azimuth": a, "elevation": e,
                               "image": encode(np.full((4, 4, 3), int(a), dtype=np.uint8))} for a, e in angles]}
        if op == "classify_views":
            return [self._scores(image) for image in args["images"]]
        if op == "classify":
            return self._scores(args["image"]) if args.get("mask") is None else {"chair": 0.2, "table": 0.7}
        raise AssertionError(op)

    @staticmethod
    def _scores(image):
        from perception.images import decode
        return VIEW_SCORES[(float(np.asarray(decode(image))[0, 0, 0]), 0.0)]


class VerifierTests(unittest.TestCase):
    def verifier(self, uncertainty=0.05, **settings):
        FakeWorker.uncertainty = uncertainty
        with patch("perception.client._Worker", FakeWorker):
            return client.Verifier(vocabulary=["chair", "table"], azimuths=(0, 90, 180), elevations=(0,),
                                   **settings)

    def test_selectors(self):
        verdict = self.verifier(selector="top2gap").verify(IMAGE, MASK, "chair")
        self.assertEqual((verdict.view.azimuth, verdict.label, verdict.accepted), (0.0, "table", False))
        verifier = self.verifier(selector="vqf")
        verdict = verifier.verify(IMAGE, MASK, "table")
        self.assertEqual((verdict.view.azimuth, verdict.label, verdict.accepted), (90.0, "table", True))
        self.assertEqual(verifier.sam3d.rendered, [[(90.0, 0.0)]])  # only the VQF pick is rendered

    def test_ensemble_filter(self):
        trusted = self.verifier(selector="vqf").verify(IMAGE, MASK, "table")
        self.assertTrue(trusted.trusted)
        self.assertAlmostEqual(trusted.uncertainty, 0.05)
        verdict = self.verifier(0.4, on_uncertain="input").verify(IMAGE, MASK, "table")
        self.assertEqual((verdict.trusted, verdict.judged, verdict.label, verdict.accepted),
                         (False, "input", "table", True))
        self.assertFalse(self.verifier(0.4, on_uncertain="reject").verify(IMAGE, MASK, "table").accepted)
        self.assertTrue(self.verifier(0.4, on_uncertain="accept").verify(IMAGE, MASK, "chair").accepted)
        off = self.verifier(0.4, ensemble=False).verify(IMAGE, MASK, "table")
        self.assertEqual((off.trusted, off.uncertainty, off.judged), (True, None, "view"))

    def test_all_views_render_everything(self):
        verifier = self.verifier(0.4, all_views=True)
        verdict = verifier.verify(IMAGE, MASK, "table")
        self.assertEqual(verdict.judged, "input")
        self.assertTrue(all(v.scores is not None and v.quality is not None for v in verdict.views))

    def test_invalid_settings(self):
        for bad in ({"selector": "nope"}, {"on_uncertain": "nope"}, {"ensemble_seeds": 1}):
            with self.assertRaises(ValueError):
                self.verifier(**bad)


class DisagreementTests(unittest.TestCase):
    def test_one_shape_up_to_jitter_agrees(self):
        from reconstruction.stage1_uncertainty import disagreement

        shell = np.zeros((16, 16, 16), dtype=bool)
        shell[4:12, 4:12, 4] = True
        shifted = np.roll(shell, 1, axis=0)
        other = np.zeros_like(shell)
        other[4:12, 4, 4:12] = True
        self.assertAlmostEqual(disagreement([shell, shifted]), 0.0)
        self.assertGreater(disagreement([shell, other]), 0.5)
        self.assertIsNone(disagreement([shell, np.zeros_like(shell)]))


if __name__ == "__main__":
    unittest.main()
