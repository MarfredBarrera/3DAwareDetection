"""perception.Verifier decisions with fake model workers, its HTTP server, and the ensemble
disagreement (CPU)."""
import json
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

import numpy as np

from perception import client
from perception import select as perception_select
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

    def alive(self):
        return self.role != "dead"

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


def select_top1(scores):
    return perception_select.top1(scores)[0]


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

    def test_input_selector_skips_sam3d(self):
        verifier = self.verifier(selector="input")
        verdict = verifier.verify(IMAGE, MASK, "table")
        self.assertIsNone(verifier.sam3d)
        self.assertEqual((verdict.judged, verdict.label, verdict.accepted, verdict.views), ("input", "table", True, []))
        self.assertEqual(verifier.alive(), {"sam3": True})

    def test_flat_target_skips_sam3d(self):
        verifier = self.verifier(selector="top2gap")
        with patch.object(perception_select, "FLAT", ("table",)):
            verdict = verifier.verify(IMAGE, MASK, "table")
        self.assertEqual((verdict.judged, verdict.label, verdict.views), ("input", "table", []))
        self.assertEqual(verifier.sam3d.rendered, [])

    def test_judge_input(self):
        # The masked raw frame reads "table" (FakeWorker).
        verifier = self.verifier(selector="vqf", judge_input=True)
        verdict = verifier.verify(IMAGE, MASK, "chair")
        self.assertEqual((verdict.judged, verdict.label, verdict.accepted), ("input", "table", False))
        self.assertEqual(verifier.sam3d.rendered, [])  # rejected before SAM3D
        verdict = verifier.verify(IMAGE, MASK, "table")
        self.assertEqual((verdict.judged, verdict.label, verdict.accepted), ("view", "table", True))
        self.assertEqual(select_top1(verdict.input_scores), "table")
        verdict = self.verifier(selector="top2gap", judge_input=True).verify(IMAGE, MASK, "table")
        self.assertEqual((verdict.judged, verdict.label, verdict.accepted), ("view", "table", True))

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


class ServerTests(unittest.TestCase):
    """perception.server round trips against a Verifier with fake workers."""

    def setUp(self):
        from perception import server

        self.server_module = server
        FakeWorker.uncertainty = 0.05
        with patch("perception.client._Worker", FakeWorker):
            self.verifier = client.Verifier(selector="top2gap", vocabulary=["chair", "table"],
                                            azimuths=(0, 90, 180), elevations=(0,))
        self.server = server.VerifierServer(self.verifier, {"selector": "top2gap"}, port=0)
        self.server.log = lambda record: None
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def call(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        try:
            with urllib.request.urlopen(urllib.request.Request(self.url + path, data=data), timeout=10) as reply:
                return reply.status, json.loads(reply.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def request(self, **fields):
        return {"protocol": self.server_module.PROTOCOL, "request_id": "r1", "image": encode(IMAGE),
                "mask": encode(MASK, "L"), "target": "table", "input_elevation": 0.0, **fields}

    def test_health(self):
        status, reply = self.call("/health")
        self.assertEqual((status, reply["ok"], reply["protocol"], reply["settings"]),
                         (200, True, self.server_module.PROTOCOL, {"selector": "top2gap"}))

    def test_verify_round_trip(self):
        status, reply = self.call("/verify", self.request())
        self.assertEqual((status, reply["ok"]), (200, True))
        verdict = reply["verdict"]
        self.assertEqual((verdict["accepted"], verdict["label"], verdict["judged"]), (True, "table", "view"))
        self.assertEqual((verdict["view"]["azimuth"], len(verdict["views"])), (0.0, 3))
        self.assertIsNotNone(reply["view_image"])
        status, reply = self.call("/verify", self.request(target="chair"))
        self.assertEqual((status, reply["verdict"]["accepted"], reply["verdict"]["label"]), (200, False, "table"))
        # Looking down at 20 deg puts the rear candidate (0, 0) at -20 deg absolute, outside the
        # band (no views from below), so top2gap falls back to the chair-like input view (180, 0).
        status, reply = self.call("/verify", self.request(input_elevation=20.0))
        self.assertEqual((reply["verdict"]["view"]["azimuth"], reply["verdict"]["view"]["absolute_elevation"],
                          reply["verdict"]["label"]), (180.0, 20.0, "chair"))

    def test_dead_worker(self):
        self.verifier.sam3d.role = "dead"
        status, reply = self.call("/health")
        self.assertEqual((status, reply["ok"], reply["workers"]), (503, False, {"sam3": True, "sam3d": False}))
        self.server.watch_workers(period=0.01)
        self.server.stopped.wait(0.5)
        for _ in range(100):
            if self.server.worker_died:
                break
            self.server.stopped.wait(0.02)
        self.assertTrue(self.server.worker_died)

    def test_bad_requests(self):
        status, reply = self.call("/verify", self.request(mask=None))
        self.assertEqual((status, reply["ok"]), (400, False))
        self.assertIn("mask", reply["error"])
        status, reply = self.call("/verify", self.request(protocol=0))
        self.assertEqual((status, reply["ok"]), (400, False))
        self.assertIn("protocol", reply["error"])
        self.assertEqual(self.call("/nope")[0], 404)

    def test_worker_error(self):
        with patch.object(FakeWorker, "call", side_effect=RuntimeError("worker died")), \
                patch("traceback.print_exc"):
            status, reply = self.call("/verify", self.request())
        self.assertEqual((status, reply["ok"]), (500, False))
        self.assertIn("worker died", reply["error"])

    def test_summary_is_json_safe(self):
        verdict = self.verifier.verify(IMAGE, MASK, "table")
        summary = json.loads(json.dumps(verdict.summary()))
        self.assertEqual(summary["view"], {"azimuth": 0.0, "elevation": 0.0, "absolute_elevation": 0.0,
                                           "quality": None, "scores": {"chair": 0.1, "table": 0.9}})
        self.assertNotIn("image", summary["views"][0])


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
