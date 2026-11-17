"""perception.Verifier decisions with fake model workers, its HTTP server, and the ensemble
disagreement (CPU)."""
import json
import math
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

import numpy as np
from PIL import Image

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
    qwen_status = "wrong"

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
        if op == "judge":
            self.judged = args
            return {"status": self.qwen_status}
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

    def test_naive_selector(self):
        verifier = self.verifier(selector="naive")
        self.assertIsNone(verifier.sam3d)
        self.assertTrue(verifier.verify(IMAGE, MASK, "table").accepted)  # table 0.7 >= 0.5
        verdict = verifier.verify(IMAGE, MASK, "chair")  # chair 0.2 < 0.5, whatever the other scores
        self.assertEqual((verdict.judged, verdict.label, verdict.accepted), ("input", None, False))

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

    def test_view_when_unsure(self):
        # The masked raw frame reads table 0.7, chair 0.2: a gap of 0.5.
        verifier = self.verifier(selector="top2gap", judge_input=True, view_when="unsure")
        verdict = verifier.verify(IMAGE, MASK, "table")
        self.assertEqual((verdict.judged, verdict.label, verdict.accepted), ("input", "table", True))
        self.assertEqual(verifier.sam3d.rendered, [])  # a sure raw accept needs no view
        verifier = self.verifier(selector="top2gap", judge_input=True, view_when="unsure", unsure_margin=0.6)
        self.assertEqual(verifier.verify(IMAGE, MASK, "table").judged, "view")
        with patch.dict(perception_select.CONFUSABLE, {"table": ("chair",)}):
            verifier = self.verifier(selector="top2gap", judge_input=True, view_when="unsure")
            self.assertEqual(verifier.verify(IMAGE, MASK, "table").judged, "view")

    def test_unsure(self):
        self.assertFalse(perception_select.unsure({"table": 0.7, "chair": 0.2}, "table"))
        self.assertTrue(perception_select.unsure({"table": 0.5, "chair": 0.3}, "table"))
        self.assertTrue(perception_select.unsure({"couch": 0.9, "chair": 0.1}, "couch"))

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

    def qwen_grid(self, review, status="wrong", grid=True):
        from perception.qwen import UncertaintyGate
        # Threshold 0 flags everything, 2 nothing (scores are mean percentiles in [0, 1]).
        gate = UncertaintyGate([[-1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], {"2": 0 if review else 2, "3": 0})
        FakeWorker.qwen_status = status
        with patch("perception.qwen.UncertaintyGate.load", return_value=gate), \
                patch("perception.qwen.render_grid", return_value=Image.new("RGB", (8, 8))) as render:
            verifier = self.verifier(selector="qwen_grid", ensemble=False)
            if not grid:
                verifier.sam3d.call = lambda op, **args: (_ for _ in ()).throw(RuntimeError("no object"))
            return verifier, verifier.verify(IMAGE, MASK, "table"), render

    def test_qwen_grid(self):
        # Raw frame: table 0.7 vs chair 0.2, so a raw accept with margin 0.5 over a full mask.
        verifier, verdict, _ = self.qwen_grid(review=False)
        self.assertEqual((verdict.status, verdict.accepted, verdict.judged, verdict.qwen), ("correct", True, "input", None))
        self.assertEqual(verdict.gate["reason"], "missing_flip_two_signal_gate")  # 3 views < 6
        self.assertAlmostEqual(verdict.gate["margin"], 0.5)
        self.assertEqual(verdict.gate["area"], 1.0)
        self.assertFalse(hasattr(verifier.qwen, "judged"))  # Qwen is only called when the gate fires
        self.assertEqual(verifier.sam3d.rendered, [[(0.0, 0.0), (90.0, 0.0), (180.0, 0.0)]])
        verifier, verdict, render = self.qwen_grid(review=True, status="wrong")
        self.assertEqual((verdict.status, verdict.accepted, verdict.judged), ("wrong", False, "qwen"))
        self.assertEqual((verifier.qwen.judged["target"], render.call_count), ("table", 1))
        verdict = self.qwen_grid(review=True, status="insufficient")[1]
        self.assertEqual((verdict.status, verdict.accepted), ("insufficient", False))
        json.dumps(verdict.summary())

    def test_qwen_grid_without_reconstruction_is_insufficient(self):
        verifier, verdict, _ = self.qwen_grid(review=True, grid=False)
        self.assertEqual((verdict.status, verdict.grid, verdict.gate["reconstruct_error"]),
                         ("insufficient", None, "no object"))
        self.assertFalse(hasattr(verifier.qwen, "judged"))

    def test_view_flip(self):
        scores = {(a, 0.0): {"chair": 0.9, "table": 0.1} if a < 90 else {"chair": 0.1, "table": 0.9}
                  for a in range(0, 360, 45)}
        elevations = {key: 0.0 for key in scores}
        self.assertEqual(perception_select.view_flip(scores, elevations, "chair"), 0.75)
        elevations[(0, 0.0)] = elevations[(45, 0.0)] = -30.0  # below the floor plane: not counted
        self.assertEqual(perception_select.view_flip(scores, elevations, "chair"), 1.0)
        self.assertIsNone(perception_select.view_flip(dict(list(scores.items())[:5]), elevations, "chair"))

    def test_invalid_settings(self):
        for bad in ({"selector": "nope"}, {"on_uncertain": "nope"}, {"ensemble_seeds": 1}):
            with self.assertRaises(ValueError):
                self.verifier(**bad)


class D1Tests(unittest.TestCase):
    """Selector d1 with fake workers: lazy start, failures, reply schema."""

    def setUp(self):
        from perception.qwen import SufficiencyGate
        # References for margin 0.5 (table over chair, FakeWorker) and area 1.0 (full mask): the
        # score is (0.5 + 0.5) / 2 = 0.5, so threshold 0.5 is insufficient and 0.6 sufficient.
        self.gate = SufficiencyGate([-0.9, -0.1], [-1.0, -0.5], 0.6)
        FakeWorker.qwen_status, FakeWorker.uncertainty = "correct", 0.05
        self.started = []
        no_models = patch("perception.client.subprocess.Popen", side_effect=AssertionError("real worker started"))
        no_models.start()
        self.addCleanup(no_models.stop)

    def verifier(self, threshold, **settings):
        self.gate.threshold = threshold
        started = self.started
        workers = self.workers = []

        class Recording(FakeWorker):
            def __init__(self, env, args, launcher, log):
                super().__init__(env, args, launcher, log)
                self.args = args
                started.append(self.role)
                workers.append(self)

        # Patched for the whole test: d1 starts workers lazily, inside verify().
        for patcher in (patch("perception.client._Worker", Recording),
                        patch("perception.qwen.SufficiencyGate.load", return_value=self.gate)):
            patcher.start()
            self.addCleanup(patcher.stop)
        verifier = client.Verifier(selector="d1", vocabulary=["chair", "table"], **settings)
        self.addCleanup(verifier.close)
        return verifier

    def test_dead_worker_is_an_error(self):
        for role in ("sam3d", "qwen"):
            verifier = self.verifier(0.5)
            worker = getattr(verifier, role)
            worker.role = "dead"
            worker.call = lambda op, **args: (_ for _ in ()).throw(RuntimeError("perception worker exited (1)"))
            with self.assertRaisesRegex(RuntimeError, "exited"):
                verifier.verify(IMAGE, MASK, "table")

    def test_failed_start_closes_spawned_workers(self):
        closed, fail = [], {"qwen"}

        class Failing(FakeWorker):
            def __init__(self, env, args, launcher, log):
                if args[1] in fail:
                    raise OSError(f"cannot start {args[1]}")
                super().__init__(env, args, launcher, log)

            def close(self):
                closed.append(self.role)

        with patch("perception.client._Worker", Failing), \
                patch("perception.qwen.SufficiencyGate.load", return_value=self.gate):
            with self.assertRaisesRegex(OSError, "qwen"):
                client.Verifier(selector="d1", vocabulary=["chair", "table"])
            self.assertEqual(closed, ["sam3", "sam3d"])  # eager: nothing left running
            closed.clear()
            self.gate.threshold = 0.5
            verifier = client.Verifier(selector="d1", vocabulary=["chair", "table"], lazy_workers=True)
            with self.assertRaisesRegex(OSError, "qwen"):
                verifier.verify(IMAGE, MASK, "table")
            self.assertEqual((closed, verifier.sam3d, verifier.qwen), (["sam3d"], None, None))
            fail.clear()  # the next insufficient request starts both afresh
            self.assertEqual(verifier.verify(IMAGE, MASK, "table").status, "correct")
            self.assertEqual(set(verifier.alive()), {"sam3", "sam3d", "qwen"})

    def test_stagger_offloads_bf16_workers(self):
        self.verifier(0.6, stagger=True)
        args = {w.role: w.args for w in self.workers}
        self.assertIn("--offload", args["sam3d"])
        self.assertIn("--offload", args["qwen"])
        self.assertNotIn("--int8", args["qwen"])
        self.assertNotIn("--offload", args["sam3"])
        with self.assertRaisesRegex(ValueError, "bf16"):
            client.Verifier(selector="d1", vocabulary=["chair", "table"], stagger=True, qwen_int8=True)

    def test_eager_start(self):
        verifier = self.verifier(0.6)  # default: every d1 worker starts with the Verifier
        self.assertEqual(self.started, ["sam3", "sam3d", "qwen"])
        self.assertEqual(verifier.alive(), {"sam3": True, "sam3d": True, "qwen": True})
        self.assertEqual(verifier.verify(IMAGE, MASK, "table").source, "sam3")
        FakeWorker.qwen_status = "wrong"
        self.gate.threshold = 0.5
        self.assertEqual(verifier.verify(IMAGE, MASK, "table").status, "wrong")
        self.assertEqual(self.started, ["sam3", "sam3d", "qwen"])  # nothing restarted

    def test_sufficient_never_starts_sam3d_or_qwen(self):
        verifier = self.verifier(0.6, lazy_workers=True)
        for target, status in (("table", "correct"), ("chair", "wrong")):
            verdict = verifier.verify(IMAGE, MASK, target)
            self.assertEqual((verdict.status, verdict.source, verdict.accepted, verdict.qwen, verdict.grid),
                             (status, "sam3", status == "correct", None, None))
            self.assertEqual(verdict.gate, {"score": 0.5, "threshold": 0.6, "margin": verdict.gate["margin"],
                                            "area": 1.0, "insufficient": False})
        self.assertEqual(self.started, ["sam3"])
        self.assertEqual((verifier.sam3d, verifier.qwen, verifier.alive()), (None, None, {"sam3": True}))

    def test_insufficient_starts_workers_once(self):
        verifier = self.verifier(0.5, lazy_workers=True)
        self.assertEqual(self.started, ["sam3"])
        for status in ("correct", "wrong", "insufficient"):
            FakeWorker.qwen_status = status
            verdict = verifier.verify(IMAGE, MASK, "table", input_elevation=10.0)
            self.assertEqual((verdict.status, verdict.source, verdict.judged, verdict.accepted),
                             (status, "qwen", "qwen", status == "correct"))
            self.assertEqual((verdict.qwen["evidence"], verdict.qwen["error"]), ("grid", None))
            self.assertEqual(verdict.grid.size, (224 * 8, 224 * 3))
        self.assertEqual(self.started, ["sam3", "sam3d", "qwen"])
        self.assertIn("--int8", verifier.qwen.args)  # int8 by default for d1
        reconstruct = verifier.sam3d.rendered[-1]
        self.assertEqual(len(reconstruct), 24)
        self.assertIsNotNone(verifier.qwen.judged["grid"])

    def test_input_evidence_never_starts_sam3d(self):
        self.verifier(0.5, d1_evidence="input", lazy_workers=True).verify(IMAGE, MASK, "table")
        self.assertEqual(self.started, ["sam3", "qwen"])
        self.started.clear()
        verifier = self.verifier(0.5, d1_evidence="input", qwen_int8=False)
        self.assertEqual((self.started, verifier.alive()), (["sam3", "qwen"], {"sam3": True, "qwen": True}))
        verdict = verifier.verify(IMAGE, MASK, "table")
        self.assertEqual((verdict.status, verdict.source, verdict.qwen["evidence"], verdict.grid),
                         ("correct", "qwen", "input", None))
        self.assertEqual(self.started, ["sam3", "qwen"])
        self.assertIsNone(verifier.qwen.judged["grid"])
        self.assertNotIn("--int8", verifier.qwen.args)
        with self.assertRaises(ValueError):
            self.verifier(0.5, d1_evidence="nope")

    def test_accept_min_turns_unsure_correct_into_insufficient(self):
        verifier = self.verifier(0.5, d1_evidence="input", d1_accept_min=0.99)
        answer = {"status": "correct", "invalid_answer": False, "top_token": "A",
                  "probabilities": {"correct": 0.95, "wrong": 0.01, "insufficient": 0.04}}
        verifier.qwen.call = lambda op, **args: dict(answer)
        verdict = verifier.verify(IMAGE, MASK, "table")
        self.assertEqual((verdict.status, verdict.accepted, verdict.qwen["below_accept_min"]), ("insufficient", False, 0.99))
        answer["probabilities"] = {"correct": 0.995, "wrong": 0.001, "insufficient": 0.004}
        verdict = verifier.verify(IMAGE, MASK, "table")
        self.assertEqual((verdict.status, verdict.accepted), ("correct", True))
        answer.update(status="wrong", probabilities={"correct": 0.1, "wrong": 0.8, "insufficient": 0.1})
        self.assertEqual(verifier.verify(IMAGE, MASK, "table").status, "wrong")  # rejects are unchanged
        with self.assertRaises(ValueError):
            self.verifier(0.5, d1_accept_min=1.5)

    def test_failures_are_insufficient(self):
        verifier = self.verifier(0.5)
        verifier._start("sam3d", "qwen")
        verifier.sam3d.call = lambda op, **args: (_ for _ in ()).throw(RuntimeError("no object"))
        verdict = verifier.verify(IMAGE, MASK, "table")
        self.assertEqual((verdict.status, verdict.accepted, verdict.qwen["error"], verdict.grid),
                         ("insufficient", False, "reconstruct: no object", None))
        self.assertFalse(hasattr(verifier.qwen, "judged"))
        verifier.sam3d.call = lambda op, **args: {"timing": {}, "views": []}  # a partial grid
        verdict = verifier.verify(IMAGE, MASK, "table")
        self.assertEqual((verdict.status, verdict.qwen["error"]), ("insufficient", "reconstruct: 0 of 24 views rendered"))
        verifier = self.verifier(0.5)
        verifier._start("sam3d", "qwen")
        verifier.qwen.call = lambda op, **args: {"status": "insufficient", "invalid_answer": True, "top_token": "D",
                                                 "probabilities": {"correct": 0.9, "wrong": 0.05,
                                                                   "insufficient": 0.05}}
        verdict = verifier.verify(IMAGE, MASK, "table")
        self.assertEqual((verdict.status, verdict.accepted, verdict.qwen["invalid_answer"]),
                         ("insufficient", False, True))
        self.assertIn("invalid answer 'D'", verdict.qwen["error"])
        verifier.qwen.call = lambda op, **args: (_ for _ in ()).throw(RuntimeError("CUDA out of memory"))
        verdict = verifier.verify(IMAGE, MASK, "table")
        self.assertEqual((verdict.status, verdict.qwen["error"]), ("insufficient", "qwen: CUDA out of memory"))

    def test_reply_schema(self):
        from perception import server
        verifier = self.verifier(0.5)
        log = []
        http = server.VerifierServer(verifier, {"selector": "d1"}, port=0)
        http.log = log.append
        threading.Thread(target=http.serve_forever, daemon=True).start()
        self.addCleanup(http.server_close)
        self.addCleanup(http.shutdown)
        payload = {"protocol": server.PROTOCOL, "request_id": "r1", "image": encode(IMAGE),
                   "mask": encode(MASK, "L"), "target": "table", "input_elevation": 0.0}
        request = urllib.request.Request(f"http://127.0.0.1:{http.server_port}/verify",
                                         data=json.dumps(payload).encode())
        with urllib.request.urlopen(request, timeout=10) as reply:
            reply = json.loads(reply.read())
        verdict = reply["verdict"]
        self.assertEqual((verdict["status"], verdict["source"], verdict["accepted"]), ("correct", "qwen", True))
        self.assertEqual(set(verdict["gate"]), {"score", "threshold", "margin", "area", "insufficient"})
        self.assertLessEqual({"status", "probabilities", "invalid_answer", "error", "evidence"}, set(verdict["qwen"]))
        self.assertEqual(Image.open(__import__("io").BytesIO(__import__("base64").b64decode(reply["view_image"]))).size,
                         (224 * 8, 224 * 3))
        self.assertEqual((log[0]["status"], log[0]["verdict_status"], log[0]["source"], log[0]["gate"]["insufficient"],
                          log[0]["qwen"]["evidence"]), (200, "correct", "qwen", True, "grid"))


FRAME = np.full((480, 640, 3), 90, dtype=np.uint8)


def frame_mask(size=60, x=320, clipped=False):
    m = np.zeros((480, 640), dtype=bool)
    left = 0 if clipped else x - size // 2
    m[240 - size // 2:240 + size // 2, left:x + size // 2] = True
    return m


class D1MultiviewTests(unittest.TestCase):
    """Verifier.verify_views and /verify_views (selector d1, close range) with fake workers."""

    def setUp(self):
        from perception.qwen import SufficiencyGate
        self.gate = SufficiencyGate([-0.9, -0.1], [-1.0, -0.5], 0.6)
        self.answer = {"status": "correct", "invalid_answer": False, "top_token": "A",
                       "probabilities": {"correct": 0.9, "wrong": 0.05, "insufficient": 0.05}}
        self.calls, self.started = [], []
        self.view_scores = []  # SAM3 scores per classify call, in order
        test = self

        class Multiview(FakeWorker):
            def __init__(self, env, args, launcher, log):
                super().__init__(env, args, launcher, log)
                self.args = args
                test.started.append(self.role)

            def call(self, op, **args):
                test.calls.append((self.role, op))
                if op == "judge_views":
                    self.judged = args
                    if isinstance(test.answer, Exception):
                        raise test.answer
                    return dict(test.answer)
                if op == "classify" and args.get("mask") is not None:
                    return dict(test.view_scores.pop(0))
                return super().call(op, **args)

        for patcher in (patch("perception.client._Worker", Multiview),
                        patch("perception.qwen.SufficiencyGate.load", return_value=self.gate),
                        patch("perception.client.subprocess.Popen", side_effect=AssertionError("real worker"))):
            patcher.start()
            self.addCleanup(patcher.stop)

    def verifier(self, **settings):
        verifier = client.Verifier(selector="d1", vocabulary=["chair", "table"], **settings)
        self.addCleanup(verifier.close)
        return verifier

    def views(self, n=3):
        return [FRAME] * n, [frame_mask(60 + 20 * k) for k in range(n)], [1.0 + k for k in range(n)]

    def ops(self, op):
        return [c for c in self.calls if c[1] == op]

    def test_qwen_decisive_skips_sam3(self):
        verifier = self.verifier(d1_evidence="input")
        images, masks, distances = self.views()
        verdict = verifier.verify_views(images, masks, "table", distances=distances, keys=[(1, 0), (2, 0), (3, 0)])
        self.assertEqual((verdict.status, verdict.source, verdict.accepted), ("correct", "qwen", True))
        self.assertEqual((self.ops("classify"), verdict.aggregate), ([], None))
        self.assertEqual(self.started, ["sam3", "qwen"])  # SAM3D never started without grid
        judged = verifier.qwen.judged
        self.assertEqual((len(judged["images"]), judged["distances"], judged["grid"]), (3, distances, None))
        self.assertEqual([(v["step"], v["index"]) for v in verdict.per_view], [(1, 0), (2, 0), (3, 0)])
        self.answer.update(status="wrong")
        self.assertEqual(verifier.verify_views(images, masks, "table").status, "wrong")

    def test_insufficient_scores_each_view_and_aggregates(self):
        verifier = self.verifier(d1_evidence="input")
        self.answer.update(status="insufficient")
        images, masks, _ = self.views(3)
        masks[2] = frame_mask(60, clipped=True)
        self.view_scores = [{"chair": 0.2, "table": 0.7}, {"chair": 0.6, "table": 0.3}]
        cached = {(1, 0): {"chair": 0.1, "table": 0.9}}
        verdict = verifier.verify_views(images, masks, "table", keys=[(1, 0), (2, 0), (3, 0)], cached_scores=cached)
        self.assertEqual(len(self.ops("classify")), 2)  # the cached view is not scored again
        self.assertEqual([v["cached"] for v in verdict.per_view], [True, False, False])
        self.assertEqual([v["label"] for v in verdict.per_view], ["table", "table", "chair"])
        weights = [min(1.0, v["area"] / 0.10) * (0.7 if v["clipped"] else 1.0) for v in verdict.per_view]
        self.assertTrue(verdict.per_view[2]["clipped"])
        margins = [0.8, 0.5, -0.3]
        expected = sum(w * m for w, m in zip(weights, margins)) / sum(weights)
        self.assertAlmostEqual(verdict.aggregate["margin"], expected)
        self.assertEqual((verdict.aggregate["vote"], verdict.aggregate["n_used"], verdict.aggregate["label"]),
                         (2 / 3, 3, "table"))
        self.assertEqual((verdict.status, verdict.accepted), ("insufficient", False))

    def test_sam3_always_and_never(self):
        verifier = self.verifier(d1_evidence="input")
        images, masks, _ = self.views(2)
        self.view_scores = [{"chair": 0.2, "table": 0.7}] * 2
        verdict = verifier.verify_views(images, masks, "table", sam3="always")
        self.assertEqual((verdict.status, verdict.aggregate["n_used"]), ("correct", 2))
        self.answer.update(status="insufficient")
        verdict = verifier.verify_views(images, masks, "table", sam3="never")
        self.assertEqual((verdict.aggregate, len(self.ops("classify"))), (None, 2))

    def test_qwen_failures_are_insufficient(self):
        verifier = self.verifier(d1_evidence="input")
        images, masks, _ = self.views(2)
        self.view_scores = [{"chair": 0.2, "table": 0.7}] * 4
        self.answer = RuntimeError("CUDA out of memory")
        verdict = verifier.verify_views(images, masks, "table")
        self.assertEqual((verdict.status, verdict.qwen["error"]), ("insufficient", "qwen: CUDA out of memory"))
        self.assertEqual(verdict.aggregate["n_used"], 2)  # the SAM3 fallback still runs
        self.answer = {"status": "insufficient", "invalid_answer": True, "top_token": "D",
                       "probabilities": {"correct": 0.9, "wrong": 0.05, "insufficient": 0.05}}
        verdict = verifier.verify_views(images, masks, "table")
        self.assertIn("invalid answer 'D'", verdict.qwen["error"])
        dead = verifier.qwen
        dead.role = "dead"
        self.answer = RuntimeError("perception worker exited (1)")
        with self.assertRaisesRegex(RuntimeError, "exited"):
            verifier.verify_views(images, masks, "table")

    def test_grid_reconstructs_the_chosen_view(self):
        verifier = self.verifier(d1_evidence="input", multiview_grid=True)
        self.assertEqual(self.started, ["sam3", "sam3d", "qwen"])  # eager with multiview_grid
        images, masks, _ = self.views(3)
        verdict = verifier.verify_views(images, masks, "table", input_elevations=[5.0, 10.0, 20.0], grid=True,
                                        grid_view=1)
        self.assertEqual((verdict.qwen["grid"], verdict.qwen["grid_view"], verdict.grid.size),
                         (True, 1, (224 * 8, 224 * 3)))
        self.assertIsNotNone(verifier.qwen.judged["grid"])
        self.assertEqual(len(verifier.sam3d.rendered[-1]), 24)

    def test_grid_failure_falls_back_to_images(self):
        verifier = self.verifier(d1_evidence="input", lazy_workers=True)
        images, masks, _ = self.views(2)
        verifier._start("sam3d", "qwen")
        verifier.sam3d.call = lambda op, **args: (_ for _ in ()).throw(RuntimeError("OutOfMemoryError: CUDA"))
        verdict = verifier.verify_views(images, masks, "table", grid=True)
        self.assertEqual((verdict.status, verdict.qwen["grid"], verdict.qwen["reconstruct_error"], verdict.grid),
                         ("correct", False, "OutOfMemoryError: CUDA", None))
        self.assertIsNone(verifier.qwen.judged["grid"])
        verifier.sam3d.role = "dead"
        with self.assertRaises(RuntimeError):
            verifier.verify_views(images, masks, "table", grid=True)

    def test_grid_starts_sam3d_lazily_only_when_asked(self):
        verifier = self.verifier(d1_evidence="input", lazy_workers=True)
        images, masks, _ = self.views(2)
        verifier.verify_views(images, masks, "table")
        self.assertEqual(self.started, ["sam3", "qwen"])
        verifier.verify_views(images, masks, "table", grid=True)
        self.assertEqual(self.started, ["sam3", "qwen", "sam3d"])

    def test_sam3_shortcut(self):
        verifier = self.verifier(d1_evidence="input")
        images, masks, _ = self.views(2)
        # area 1.0 references make the score (margin pct + area pct) / 2; a big margin is sufficient
        self.gate.threshold = 0.9
        self.view_scores = [{"chair": 0.05, "table": 0.95}]
        verdict = verifier.verify_views(images, masks, "table", sam3_shortcut=True)
        self.assertEqual((verdict.status, verdict.source), ("correct", "sam3"))
        self.assertEqual(self.ops("judge_views"), [])
        self.gate.threshold = 0.0  # everything insufficient: Qwen decides
        self.view_scores = [{"chair": 0.05, "table": 0.95}]
        self.assertEqual(verifier.verify_views(images, masks, "table", sam3_shortcut=True).source, "qwen")

    def test_limits(self):
        verifier = self.verifier(d1_evidence="input")
        with self.assertRaises(ValueError):
            verifier.verify_views([FRAME] * 9, [frame_mask()] * 9, "table")
        with self.assertRaises(ValueError):
            verifier.verify_views([], [], "table")
        with self.assertRaises(ValueError):
            verifier.verify_views([FRAME], [np.zeros((480, 640), bool)], "table")
        with self.assertRaises(ValueError):
            verifier.verify_views([FRAME], [frame_mask()], "table", sam3="sometimes")
        with patch("perception.client._Worker", FakeWorker):
            other = client.Verifier(selector="input", vocabulary=["chair", "table"])
        with self.assertRaises(ValueError):
            other.verify_views([FRAME], [frame_mask()], "table")

    def test_worker_candidates_score_the_same_segment(self):
        from perception import worker
        h, w = FRAME.shape[:2]
        bed = np.zeros((h, w), bool); bed[100:300, 100:400] = True
        chair = np.zeros((h, w), bool); chair[50:150, 500:600] = True
        found = {"bed": {"scores": np.array([0.86, 0.1]), "boxes": np.array([[100, 100, 400, 300], [500, 50, 600, 150]]),
                         "masks": np.stack([bed, chair])},
                 "couch": {"scores": np.array([0.17, 0.2]), "boxes": np.array([[500, 50, 600, 150], [100, 100, 400, 300]]),
                           "masks": np.stack([chair, bed])},
                 "chair": {"scores": np.array([0.9]), "boxes": np.array([[500, 50, 600, 150]]), "masks": chair[None]}}

        class Model:
            def detect_many(self, image, prompts):
                return {p: found[p] for p in prompts}

        server = worker.SAM3Server.__new__(worker.SAM3Server)
        server.model = Model()
        out = server.candidates(encode(FRAME), ["bed", "couch", "chair"], ["couch", "bed"], min_score=0.3)
        self.assertEqual([c["word"] for c in out], ["bed"])          # couch 0.17 / 0.2 < 0.3; chair not a candidate word
        self.assertEqual(out[0]["scores"], {"bed": 0.86, "couch": 0.2, "chair": 0.0})   # same segment only

    def test_worker_outlines_and_crops(self):
        from perception import worker
        seen = {}

        class Model:
            def judge_views(self, frames, target, distances, grid, crops, forced2=False, forced2_min=0.5):
                seen.update(frames=frames, crops=crops, grid=grid, forced2=forced2, distances=distances)
                return {"status": "correct"}

        server = worker.QwenServer.__new__(worker.QwenServer)
        server.model, server.offload = Model(), False
        big, small = frame_mask(200), frame_mask(40)  # 0.13 and 0.005 of the frame
        self.assertEqual(server.judge_views([encode(FRAME)] * 2, [encode(big, "L"), encode(small, "L")], "tv",
                                            distances=[1.0, 3.0], crop_small=True, forced2=True),
                         {"status": "correct"})
        self.assertEqual([c is None for c in seen["crops"]], [True, False])
        self.assertEqual(seen["crops"][1].size, (448, 448))
        self.assertEqual(np.asarray(seen["frames"][1])[218, 310].tolist(), [255, 0, 0])  # the red outline
        self.assertEqual((seen["grid"], seen["forced2"]), (None, True))
        server.judge_views([encode(FRAME)], [encode(small, "L")], "tv")
        self.assertEqual(seen["crops"], [None])

    def test_http(self):
        from perception import server
        verifier = self.verifier(d1_evidence="input")
        log = []
        http = server.VerifierServer(verifier, {"selector": "d1", "multiview": True}, port=0)
        http.log = log.append
        threading.Thread(target=http.serve_forever, daemon=True).start()
        self.addCleanup(http.server_close)
        self.addCleanup(http.shutdown)
        url = f"http://127.0.0.1:{http.server_port}"

        def call(path, payload=None):
            data = None if payload is None else json.dumps(payload).encode()
            try:
                with urllib.request.urlopen(urllib.request.Request(url + path, data=data), timeout=10) as reply:
                    return reply.status, json.loads(reply.read())
            except urllib.error.HTTPError as error:
                return error.code, json.loads(error.read())

        def view(step, size=60, scores=None, image=FRAME, mask=None):
            return {"image": encode(image), "mask": encode(frame_mask(size) if mask is None else mask, "L"),
                    "input_elevation": 0.0, "step": step, "index": 0, "range_m": 1.0 + step,
                    **({"scores": scores} if scores else {})}

        def request(views, **fields):
            return {"protocol": server.PROTOCOL, "request_id": "r1", "target": "table", "sam3": "fallback",
                    "views": views, **fields}

        status, reply = call("/health")
        self.assertTrue(reply["settings"]["multiview"])
        self.answer.update(status="insufficient")
        self.view_scores = [{"chair": 0.6, "table": 0.3}]
        status, reply = call("/verify_views", request([view(1, scores={"chair": 0.1, "table": 0.9}), view(2, 80)]))
        self.assertEqual((status, reply["ok"], reply["grid_image"]), (200, True, None))
        verdict = reply["verdict"]
        self.assertEqual((verdict["status"], verdict["n_views"], verdict["aggregate"]["n_used"]),
                         ("insufficient", 2, 2))
        self.assertEqual([(v["step"], v["cached"]) for v in verdict["per_view"]], [(1, True), (2, False)])
        self.assertEqual((log[-1]["path"], log[-1]["n_views"], log[-1]["verdict_status"], log[-1]["status"]),
                         ("/verify_views", 2, "insufficient", 200))
        bad = [request([]), request([view(k) for k in range(9)]),
               request([view(1, image=np.zeros((48, 64, 3), np.uint8), mask=np.ones((48, 64), bool))]),
               request([view(1, mask=np.zeros((480, 640), bool))]), request([view(1), view(1)]),
               request([view(1)], sam3="sometimes"), request([view(1)], grid_view=3), request([view(1)], protocol=0)]
        for payload in bad:
            status, reply = call("/verify_views", payload)
            self.assertEqual((status, reply["ok"]), (400, False), reply)
        verifier.selector = "input"  # a non-d1 server
        status, reply = call("/verify_views", request([view(1)]))
        self.assertEqual(status, 400)
        self.assertIn("selector d1", reply["error"])


OFFLINE = client.WORKSPACE / "outputs/apexnav_offline"
PILOT_CSV = OFFLINE / "qwen_verification_pilot/per_request.csv"
D1_DIR = OFFLINE / "qwen_all_requests_2026-11-03"


@unittest.skipUnless(PILOT_CSV.exists() and (D1_DIR / "d1_gate.json").exists(), "offline D1 data not on this machine")
class D1ReplayTests(unittest.TestCase):
    """The frozen D1 gate and selector reproduce _compare.py on the 295 pilot requests."""

    @classmethod
    def setUpClass(cls):
        import csv
        cls.rows = list(csv.DictReader(open(PILOT_CSV)))
        cls.expected = {r["name"]: r for r in csv.DictReader(open(D1_DIR / "per_request.tsv"), delimiter="\t")}
        cls.answers = {r["name"]: r for r in map(json.loads, open(D1_DIR / "decisions.jsonl"))}

    def test_gate_export_matches_compare(self):
        import importlib.util
        from perception.qwen import SufficiencyGate
        spec = importlib.util.spec_from_file_location("_compare", D1_DIR / "_compare.py")
        compare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compare)
        _, _, s_t, _, score = compare.fit_thresholds(compare.offline_reference())
        frozen = SufficiencyGate.load(D1_DIR / "d1_gate.json")
        refit, _ = SufficiencyGate.fit(compare.offline_reference())
        self.assertEqual((frozen.threshold, refit.threshold), (s_t, s_t))
        self.assertEqual((refit.neg_abs_margin, refit.neg_area), (frozen.neg_abs_margin, frozen.neg_area))
        self.assertEqual(len(self.rows), 295)
        for r in self.rows:
            margin, area = float(r["margin"]), float(r["area"])
            got = frozen.assess(margin, area)
            self.assertEqual(got["score"], score(margin, area), r["name"])
            self.assertEqual(got["insufficient"], self.expected[r["name"]]["insuf_rank"] == "True", r["name"])

    def test_d1_decisions_match_compare(self):
        from perception.images import to_pil
        current = {}

        class Replay(FakeWorker):
            def call(self, op, **args):
                r = current["row"]
                if op == "classify":  # the stored raw SAM3 scores, for exactly the prompts sent
                    stored = current["scores"]
                    if set(args["prompts"]) != set(stored):
                        raise AssertionError(f"{r['name']}: prompts {set(args['prompts']) ^ set(stored)} differ")
                    return {p: stored[p] for p in args["prompts"]}
                if op == "judge":
                    self.judged = args
                    return dict(self.answers[r["name"]])
                return super().call(op, **args)

        Replay.answers = self.answers
        for patcher in (patch("perception.client._Worker", Replay),
                        patch("perception.client.subprocess.Popen", side_effect=AssertionError("real worker"))):
            patcher.start()
            self.addCleanup(patcher.stop)
        verifier = client.Verifier(selector="d1")  # the default vocabulary, client.HOUSEHOLD
        self.addCleanup(verifier.close)
        to_action = {"correct": "accept", "wrong": "reject", "insufficient": "approach"}
        for r in self.rows:
            decision = json.loads((client.WORKSPACE / r["folder"] / "decision.json").read_text())
            self.assertEqual(decision["server"]["vocabulary"], list(client.HOUSEHOLD))
            stored = decision["verdict"]
            current["row"], current["scores"] = r, stored["scores" if stored["judged"] == "input" else "input_scores"]
            mask = to_pil(client.WORKSPACE / r["folder"] / "mask.png", "L")
            verdict = verifier.verify(np.zeros((mask.height, mask.width, 3), np.uint8), mask, r["target"])
            self.assertEqual(verdict.gate["margin"], float(r["margin"]), r["name"])
            self.assertEqual(verdict.gate["area"], float(r["area"]), r["name"])
            self.assertEqual(verdict.label == r["target"], r["raw_accept"] == "True", r["name"])
            self.assertEqual(to_action[verdict.status], self.expected[r["name"]]["D1 sufficiency-first (area+margin)"],
                             r["name"])
            self.assertEqual(verdict.accepted, verdict.status == "correct")


class SelectRulesTests(unittest.TestCase):
    def test_hm3d_rules_unchanged(self):
        """The open-vocabulary rules leave ApexNav's six HM3D-v2 goals as before."""
        for target in ("chair", "couch", "bed", "toilet", "tv", "potted plant"):
            parts_only = [c for c in dict.fromkeys([*client.HOUSEHOLD, target])
                          if c not in perception_select.PARTS.get(target, ())]
            self.assertEqual(perception_select.competitors(client.HOUSEHOLD, target), parts_only, target)
            self.assertEqual(perception_select.flat(target), target == "tv", target)

    def test_open_vocabulary_competitors(self):
        def lacks(target, name):
            return name not in perception_select.competitors(client.HOUSEHOLD, target)
        self.assertTrue(lacks("pillow", "cushion"))
        self.assertTrue(lacks("refrigerator", "appliance"))
        self.assertTrue(lacks("plant", "potted plant"))
        self.assertTrue(lacks("footstool", "stool"))
        self.assertTrue(lacks("nightstand", "table") and lacks("nightstand", "cabinet"))
        self.assertFalse(lacks("vase", "potted plant"))
        self.assertFalse(lacks("tv stand", "tv"))
        self.assertFalse(perception_select.same_object("bathtub", "tub"))  # whole words only
        self.assertIn("pillow", perception_select.competitors(client.HOUSEHOLD, "pillow"))
        for target in ("photo", "window glass", "rug", "photo mount", "calendar"):
            self.assertTrue(perception_select.flat(target), target)
        for target in ("sideboard", "tv stand", "pillow"):
            self.assertFalse(perception_select.flat(target), target)


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

    def test_views_request_cached_jev(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        mask = np.zeros((480, 640), np.uint8)
        mask[100:200, 100:200] = 255
        view = {"image": encode(frame), "mask": encode(mask, "L"), "step": 3, "index": 0,
                "jev": {"couch": 0.6, "something else": 0.4}}
        body = {"protocol": self.server_module.PROTOCOL, "target": "couch", "views": [view]}
        parsed = self.server_module.parse_views_request(json.dumps(body))
        self.assertEqual(parsed["cached_jev"], {(3, 0): {"couch": 0.6, "something else": 0.4}})
        with self.assertRaises(self.server_module.BadRequest):
            self.server_module.parse_views_request(json.dumps({**body, "views": [{**view, "jev": {"couch": "x"}}]}))

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


# ---------------------------------------------------------------------------
# Selector jev (JEV-27B-VL naming + frozen entropy/contention gate), CPU, no weights
# ---------------------------------------------------------------------------

def jev_dist(target, **named):
    """A raw JEV naming distribution over ``jev.name_opts(target)``: the named options (underscores
    for spaces) get their probability, the rest share what is left equally."""
    from perception import jev
    options = jev.name_opts(target)
    named = {k.replace("_", " "): v for k, v in named.items()}
    rest = (1.0 - sum(named.values())) / (len(options) - len(named))
    return {o: named.get(o, rest) for o in options}


def jev_test_gate():
    """References chosen so that percentiles are easy: entropy ranks 0.1..0.8, contention 0..0.35."""
    from perception.jev import JevGate
    return JevGate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35], 0.5)


class JevMathTests(unittest.TestCase):
    """perception.jev decision and uncertainty math on fixed fake distributions."""

    def test_options(self):
        from perception import jev
        self.assertEqual(jev.name_opts("chair"), ["armchair", "bed", "bench", "cabinet", "chair", "couch", "lamp",
                                                  "mirror", "picture", "potted plant", "stool", "table", "toilet",
                                                  "tv", "window", "something else"])
        for target in jev.CONFUSERS:
            options = jev.name_opts(target)
            self.assertEqual((len(options), options[-1], options[:-1]), (16, "something else", sorted(options[:-1])))
            self.assertLessEqual({target, *jev.CONFUSERS[target]}, set(options))
        self.assertEqual(jev.contention_confusers("chair"), ["stool", "bench", "couch", "table"])
        self.assertEqual(jev.contention_confusers("couch"), jev.CONFUSERS["couch"])  # armchair competes with couch

    def test_approach_options(self):
        """Approach-phase naming: 16 options over SAM3's HM3D vocabulary (client.HOUSEHOLD) + "something else"."""
        from perception import client, jev
        self.assertEqual(jev.approach_opts("couch"), ["bed", "cabinet", "chair", "couch", "cushion", "door", "lamp",
                                                      "mirror", "picture", "potted plant", "stool", "table", "toilet",
                                                      "tv", "window", "something else"])
        for target in jev.CONFUSERS:
            options = jev.approach_opts(target)
            self.assertEqual((len(options), options[-1], options[:-1]), (16, "something else", sorted(options[:-1])))
            self.assertLessEqual({target, *jev.APPROACH_LOOKALIKES[target]}, set(options))
            self.assertLessEqual(set(options[:-1]), set(client.HOUSEHOLD), target)
        self.assertEqual(jev.naming_text("tv", "approach")[1], jev.approach_opts("tv"))
        self.assertEqual(jev.naming_text("tv")[1], jev.name_opts("tv"))   # the gate keeps its words
        with self.assertRaises(ValueError):
            jev.naming_text("tv", "hm3d")

    def test_template(self):
        from perception import jev
        text, options = jev.naming_text("tv")
        self.assertTrue(text.startswith("[kind] choice\n[state] Robot camera image of a room; one object is outlined "
                                        "in red: <|vision_start|><|image_pad|><|vision_end|>\n[question] What is the "
                                        "object outlined in red?\n[options]\nA) bed\nB) cabinet\n"))
        self.assertTrue(text.endswith("\nP) something else\n[decision]:"))
        self.assertEqual(len(options), 16)

    def test_unknown_target_falls_back_to_fillers(self):
        from perception import jev
        options = jev.name_opts("fireplace")
        self.assertEqual(options, sorted(["fireplace"] + jev.FILLERS[:14]) + ["something else"])
        self.assertEqual(jev.name_opts("lamp"), sorted(jev.FILLERS[:15]) + ["something else"])  # a filler itself
        gate = jev_test_gate()
        status, a = gate.decide(jev_dist("fireplace", fireplace=0.9), "fireplace")
        self.assertEqual((a["contention"], a["known_target"], a["pct_contention"]), (0.0, False, 1 / 16))
        self.assertEqual(status, "correct")

    def test_armchair_folds_into_chair(self):
        from perception import jev
        raw = jev_dist("chair", chair=0.35, armchair=0.4, stool=0.2)
        self.assertEqual(jev.top1(raw), "armchair")
        folded = jev.fold(raw, "chair")
        self.assertNotIn("armchair", folded)
        self.assertAlmostEqual(folded["chair"], 0.75)
        self.assertAlmostEqual(sum(folded.values()), 1.0)
        self.assertEqual(jev.fold(raw, "couch"), raw)  # only chair targets fold
        status, a = jev_test_gate().decide(raw, "chair", forced=True)
        self.assertEqual((status, a["top1"]), ("correct", "chair"))
        self.assertAlmostEqual(a["contention"], 0.2)  # min(p_chair 0.75, max(stool 0.2, ...))
        self.assertEqual(jev.decision_text("choice", [jev.STATE, None], jev.QUESTION, jev.name_opts("couch"))
                         .count("armchair"), 1)

    def test_entropy_contention_score_and_decision(self):
        from perception import jev
        gate = jev_test_gate()
        raw = jev_dist("couch", couch=0.6, armchair=0.3)        # 14 others share 0.1
        h = -(0.6 * math.log(0.6) + 0.3 * math.log(0.3) + 14 * (0.1 / 14) * math.log(0.1 / 14)) / math.log(16)
        a = gate.assess(raw, "couch")
        self.assertAlmostEqual(a["entropy"], h)
        self.assertAlmostEqual(a["contention"], 0.3)
        # entropy 0.419 lies between references 0.4 and 0.5 (4 of 8 below); contention 0.3 ties one reference
        self.assertAlmostEqual(h, 0.4190, places=3)
        self.assertEqual((a["pct_entropy"], a["pct_contention"]), (4 / 8, (6 + 7) / 16))
        self.assertAlmostEqual(a["score"], (4 / 8 + 13 / 16) / 2)
        self.assertTrue(a["insufficient"])
        self.assertEqual(gate.decide(raw, "couch")[0], "insufficient")
        self.assertEqual(gate.decide(raw, "couch", forced=True)[0], "correct")
        sure = jev_dist("couch", couch=0.97)
        self.assertEqual(gate.decide(sure, "couch")[0], "correct")
        self.assertEqual(gate.decide(jev_dist("couch", bed=0.97), "couch")[0], "wrong")
        self.assertEqual(jev.entropy({"a": 1.0, "b": 0.0}), 0.0)
        self.assertEqual(jev.average([{"a": 0.2, "b": 0.8}, {"a": 0.6, "b": 0.4}]), {"a": 0.4, "b": 0.6000000000000001})

    @unittest.skipUnless((client.WORKSPACE / "outputs/apexnav_offline/reliability_2026-11-05/jev_gate.json").exists(),
                         "frozen JEV gate not on this machine")
    def test_frozen_gate(self):
        from perception import jev
        original = client.WORKSPACE / jev.RELIABILITY / "jev_gate.json"   # the study's gate (per-target words)
        frozen = json.loads(original.read_text())
        gate = jev.JevGate.load(original)
        self.assertEqual((frozen["n_reference"], frozen["n_study"], frozen["approach_rate"]), (131, 444, 0.45))
        self.assertEqual(len(gate.entropy_reference), 131)
        self.assertEqual(gate.entropy_reference, sorted(gate.entropy_reference))
        self.assertAlmostEqual(gate.threshold, 0.5076335877862596)
        self.assertAlmostEqual(frozen["flagged_fraction_study"], 202 / 444)
        self.assertEqual(frozen["options"], {t: jev.name_opts(t) for t in jev.CONFUSERS})
        self.assertEqual(gate.decide(jev_dist("bed", bed=0.999), "bed")[0], "correct")
        self.assertEqual(gate.decide(jev_dist("bed", bed=0.45, ottoman_or_pouf=0.45), "bed")[0], "insufficient")
        self.assertEqual(gate.decide(jev_dist("bed", bed=0.45, ottoman_or_pouf=0.46), "bed", forced=True)[0], "wrong")

    @unittest.skipUnless((client.WORKSPACE / "outputs/apexnav_offline/reliability_2026-11-05/_jev_run.py").exists(),
                         "_jev_run.py not on this machine")
    def test_template_matches_jev_run(self):
        """The text _jev_run.decide itself builds (its source, run up to the processor call)."""
        from perception import jev
        image = Image.new("RGB", (64, 48))
        for target in jev.CONFUSERS:
            text, images = jev.reference_text(target, image)
            self.assertEqual(text, jev.naming_text(target)[0], target)
            self.assertEqual(images, [image])


class FakeJev(FakeWorker):
    """The JEV worker (role jev): op ``name`` answers ``FakeJev.answers`` in order (a dict, or an
    exception to raise)."""
    answers = []
    calls = []

    def call(self, op, **args):
        if op != "name":
            raise AssertionError(op)
        FakeJev.calls.append(args)
        answer = FakeJev.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return {"probabilities": dict(answer), "options": list(answer), "latency_s": 0.4, "n_tokens": 900}


class JevSam3Tests(unittest.TestCase):
    """Selector jev_sam3: JEV judges the uncertainty, SAM3 decides (fake JEV and SAM3 workers)."""

    def setUp(self):
        FakeJev.answers, FakeJev.calls = [], []
        self.started, self.sam3_scores, self.classified = [], [], []
        test = self

        class Worker(FakeJev):
            def __init__(self, env, args, launcher, log):
                super().__init__(env, args, launcher, log)
                test.started.append((env, args))

            def call(self, op, **args):
                if self.role == "jev":
                    return FakeJev.call(self, op, **args)
                if op != "classify" or args.get("mask") is None:
                    raise AssertionError(op)
                test.classified.append(args["prompts"])
                return dict(test.sam3_scores.pop(0))

        for patcher in (patch("perception.client._Worker", Worker),
                        patch("perception.jev.JevGate.load", return_value=jev_test_gate()),
                        patch("perception.client.subprocess.Popen", side_effect=AssertionError("real worker"))):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.verifier = client.Verifier(selector="jev_sam3", vocabulary=["couch", "bed", "chair"])
        self.addCleanup(self.verifier.close)

    def test_workers(self):
        self.assertEqual(sorted((env, args[1]) for env, args in self.started), [("core", "sam3"), ("qwen", "jev")])
        with self.assertRaises(ValueError):
            client.Verifier(selector="jev_sam3", all_views=True)

    def test_first_request(self):
        mask = frame_mask(60)
        # JEV unsure: approach, SAM3 not asked
        FakeJev.answers = [jev_dist("couch", couch=0.6, armchair=0.3)]
        verdict = self.verifier.verify(FRAME, mask, "couch")
        self.assertEqual((verdict.status, verdict.source, self.classified), ("insufficient", "jev", []))
        # JEV sure (even of another name): SAM3's raw label decides
        FakeJev.answers = [jev_dist("couch", bed=0.97), jev_dist("couch", couch=0.97)]
        self.sam3_scores = [{"couch": 0.8, "bed": 0.1, "chair": 0.05}, {"couch": 0.2, "bed": 0.7, "chair": 0.1}]
        verdict = self.verifier.verify(FRAME, mask, "couch")
        self.assertEqual((verdict.status, verdict.accepted, verdict.source, verdict.label), ("correct", True, "sam3", "couch"))
        self.assertEqual(verdict.gate["top1"], "bed")           # JEV's assessment is kept for the log
        self.assertEqual(verdict.scores["couch"], 0.8)          # SAM3's scores
        verdict = self.verifier.verify(FRAME, mask, "couch")
        self.assertEqual((verdict.status, verdict.accepted, verdict.label), ("wrong", False, "bed"))
        self.assertEqual(self.classified[0], ["couch", "bed", "chair"])
        json.dumps(verdict.summary())

    def test_verify_views(self):
        images, masks, keys = [FRAME] * 2, [frame_mask(120), frame_mask(100)], [(10, 0), (8, 1)]
        unsure = [jev_dist("couch", couch=0.6, armchair=0.3)] * 2
        # on the way and still unsure: keep approaching, no SAM3
        FakeJev.answers = list(unsure)
        verdict = self.verifier.verify_views(images, masks, "couch", keys=keys, forced=False)
        self.assertEqual((verdict.status, verdict.source, verdict.aggregate, self.classified), ("insufficient", "jev", None, []))
        # forced (arrival): SAM3 over both views, fused belief decides (couch 1.0 vs bed 0.6 summed)
        FakeJev.answers = list(unsure)
        self.sam3_scores = [{"couch": 0.7, "bed": 0.2, "chair": 0.0}, {"couch": 0.3, "bed": 0.4, "chair": 0.0}]
        verdict = self.verifier.verify_views(images, masks, "couch", keys=keys)
        self.assertEqual((verdict.status, verdict.accepted, verdict.source), ("correct", True, "sam3"))
        self.assertEqual((verdict.fused["label"], verdict.fused["n_used"]), ("couch", 2))
        w = [float((m > 0).mean()) / perception_select.FULL_VIEW_AREA for m in masks]  # logged only: area-weighted
        self.assertAlmostEqual(verdict.aggregate["margin"], (w[0] * 0.5 - w[1] * 0.1) / (w[0] + w[1]))
        self.assertEqual((verdict.aggregate["vote"], verdict.jev["forced"]), (0.5, True))
        self.assertEqual([v["label"] for v in verdict.per_view], ["couch", "bed"])
        self.assertEqual([v["jev_label"] for v in verdict.per_view], ["couch", "couch"])
        json.dumps(verdict.summary())
        # a look-alike that barely wins one view loses the pooled evidence (couch .65 vs bed .9); a cached view
        # is not re-scored
        FakeJev.answers = list(unsure)
        self.sam3_scores = [{"couch": 0.2, "bed": 0.5, "chair": 0.0}]
        verdict = self.verifier.verify_views(images, masks, "couch", keys=keys,
                                             cached_scores={(10, 0): {"couch": 0.45, "bed": 0.40, "chair": 0.0}})
        self.assertEqual((verdict.status, verdict.fused["label"], verdict.per_view[0]["cached"], len(self.classified)),
                         ("wrong", "bed", True, 3))
        with self.assertRaises(TypeError):   # the margin-and-vote knobs are gone
            self.verifier.verify_views(images, masks, "couch", keys=keys, sam3_margin=0.3)
        # on the way and JEV sure: SAM3 decides already
        FakeJev.answers = [jev_dist("couch", couch=0.97)] * 2
        self.sam3_scores = [{"couch": 0.1, "bed": 0.8, "chair": 0.0}] * 2
        verdict = self.verifier.verify_views(images, masks, "couch", keys=keys, forced=False)
        self.assertEqual((verdict.status, verdict.source, verdict.jev["forced"]), ("wrong", "sam3", False))
        with self.assertRaises(ValueError):
            self.verifier.verify_views(images, masks, "couch", keys=keys, grid=True)

    def test_verify_views_cached_jev(self):
        """CLOSER: each view is judged once; cached JEV probabilities (and SAM3 scores) skip the models."""
        images, masks, keys = [FRAME] * 2, [frame_mask(120), frame_mask(100)], [(10, 0), (8, 1)]
        unsure = jev_dist("couch", couch=0.6, armchair=0.3)
        FakeJev.answers = [unsure]                       # only view (8, 1) reaches JEV
        verdict = self.verifier.verify_views(images, masks, "couch", keys=keys, forced=False,
                                             cached_jev={(10, 0): unsure})
        self.assertEqual((verdict.status, len(FakeJev.calls)), ("insufficient", 1))
        self.assertEqual([v.get("jev_cached", False) for v in verdict.per_view], [True, False])
        self.assertEqual(verdict.per_view[0]["probabilities"], unsure)
        # both cached (JEV and SAM3): a forced decision without any model call
        verdict = self.verifier.verify_views(images, masks, "couch", keys=keys, cached_jev={k: unsure for k in keys},
                                             cached_scores={k: {"couch": 0.7, "bed": 0.2, "chair": 0.0} for k in keys})
        self.assertEqual((verdict.status, len(FakeJev.calls), self.classified), ("correct", 1, []))

    def test_jev_gate_on_a_frame(self):
        """JEV's part (``_jev``): the gate's status and JEV's top-1 (armchair folded into chair)."""
        FakeJev.answers = [jev_dist("couch", couch=0.97), jev_dist("couch", bed=0.97),
                           jev_dist("couch", couch=0.6, armchair=0.3), jev_dist("chair", chair=0.35, armchair=0.4),
                           jev_dist("fireplace", fireplace=0.8)]
        mask = frame_mask(60)
        for target, status in [("couch", "correct"), ("couch", "wrong"), ("couch", "insufficient"),
                               ("chair", "correct"), ("fireplace", "correct")]:
            verdict = self.verifier._jev(FRAME, mask, target)
            self.assertEqual((verdict.status, verdict.source, verdict.label), (status, "jev", verdict.gate["top1"]), target)
        self.assertEqual((verdict.gate["known_target"], verdict.gate["contention"]), (False, 0.0))
        self.assertEqual(FakeJev.calls[0]["target"], "couch")
        from perception.images import decode
        np.testing.assert_array_equal(np.asarray(decode(FakeJev.calls[0]["mask"], "L")) > 0, mask)

    def test_jev_failures(self):
        FakeJev.answers = [RuntimeError("ValueError: empty object mask")]
        verdict = self.verifier.verify(FRAME, frame_mask(60), "bed")
        self.assertEqual((verdict.status, verdict.accepted, self.classified), ("insufficient", False, []))
        self.assertIn("empty object mask", verdict.jev["error"])
        FakeJev.answers = [RuntimeError("worker exited")]
        with patch.object(self.verifier.jev, "alive", return_value=False), self.assertRaises(RuntimeError):
            self.verifier.verify(FRAME, frame_mask(60), "bed")
        # a failed view is left out; all failed: the error propagates
        FakeJev.answers = [RuntimeError("bad view"), jev_dist("couch", couch=0.6, armchair=0.3)]
        verdict = self.verifier.verify_views([FRAME] * 2, [frame_mask(60)] * 2, "couch", keys=[(1, 0), (2, 0)],
                                             forced=False)
        self.assertEqual((verdict.status, verdict.jev["n_used"]), ("insufficient", 1))
        self.assertIn("bad view", verdict.per_view[0]["error"])
        FakeJev.answers = [RuntimeError("x")] * 2
        with self.assertRaises(RuntimeError):
            self.verifier.verify_views([FRAME] * 2, [frame_mask(60)] * 2, "couch", keys=[(1, 0), (2, 0)])

    def test_http(self):
        from perception import server
        log = []
        http = server.VerifierServer(self.verifier, {"selector": "jev_sam3", "multiview": True, "multiview_grid": False},
                                     port=0)
        http.log = log.append
        threading.Thread(target=http.serve_forever, daemon=True).start()
        self.addCleanup(http.server_close)
        self.addCleanup(http.shutdown)
        url = f"http://127.0.0.1:{http.server_port}"

        def call(path, payload=None):
            data = None if payload is None else json.dumps(payload).encode()
            try:
                with urllib.request.urlopen(urllib.request.Request(url + path, data=data), timeout=10) as reply:
                    return reply.status, json.loads(reply.read())
            except urllib.error.HTTPError as error:
                return error.code, json.loads(error.read())

        status, reply = call("/health")
        self.assertEqual((status, reply["settings"]["selector"], reply["workers"]),
                         (200, "jev_sam3", {"sam3": True, "jev": True}))
        FakeJev.answers = [jev_dist("couch", couch=0.6, armchair=0.3)]
        status, reply = call("/verify", {"protocol": server.PROTOCOL, "request_id": "r1", "image": encode(FRAME),
                                         "mask": encode(frame_mask(60), "L"), "target": "couch"})
        self.assertEqual((status, reply["verdict"]["status"], reply["verdict"]["source"]), (200, "insufficient", "jev"))
        self.assertEqual((log[-1]["gate"]["insufficient"], log[-1]["jev"]["probabilities"]["armchair"]), (True, 0.3))
        views = [{"image": encode(FRAME), "mask": encode(frame_mask(120), "L"), "step": s, "index": 0,
                  "range_m": 1.0 + s} for s in (1, 2)]
        FakeJev.answers = [jev_dist("couch", couch=0.6, armchair=0.3)] * 2
        self.sam3_scores = [{"couch": 0.7, "bed": 0.1, "chair": 0.1}] * 2
        status, reply = call("/verify_views", {"protocol": server.PROTOCOL, "target": "couch", "views": views})
        self.assertEqual((status, reply["verdict"]["status"], reply["verdict"]["source"],
                          reply["verdict"]["fused"]["label"]), (200, "correct", "sam3", "couch"))
        self.assertEqual((log[-1]["path"], log[-1]["jev"]["n_used"], log[-1]["per_view"][0]["label"]),
                         ("/verify_views", 2, "couch"))
        FakeJev.answers = [jev_dist("couch", couch=0.6, armchair=0.3)] * 2
        status, reply = call("/verify_views", {"protocol": server.PROTOCOL, "target": "couch", "views": views,
                                               "forced": False})
        self.assertEqual(reply["verdict"]["status"], "insufficient")
        # /name: approach words by default (no gate signals), the gate's words on request
        body = {"protocol": server.PROTOCOL, "request_id": "n1", "image": encode(FRAME),
                "mask": encode(frame_mask(60), "L"), "target": "couch"}
        FakeJev.answers = [jev_dist("couch", couch=0.9)] * 2
        status, reply = call("/name", body)
        self.assertEqual((status, FakeJev.calls[-1]["words"], reply["gate"]), (200, "approach", None))
        status, reply = call("/name", {**body, "words": "gate"})
        self.assertEqual((status, FakeJev.calls[-1]["words"], reply["gate"]["top1"]), (200, "gate", "couch"))
        status, reply = call("/name", {**body, "words": "hm3d"})
        self.assertEqual(status, 400)
        for bad in ({"grid": True}, {"sam3_margin": 0.15}, {"sam3_vote": 0.5}):
            status, reply = call("/verify_views", {"protocol": server.PROTOCOL, "target": "couch", "views": views, **bad})
            self.assertEqual((status, reply["ok"]), (400, False))

    def test_server_settings(self):
        """serve() wires --selector jev_sam3 to the Verifier and /health (no workers started)."""
        from perception import server
        seen = {}

        class Stop(Exception):
            pass

        class Server:
            def __init__(self, verifier, settings, *args):
                seen.update(settings=settings, selector=verifier.selector)
                raise Stop

        with patch.object(server, "VerifierServer", Server), self.assertRaises(Stop):
            server.serve(["--selector", "jev_sam3", "--log-dir", "/tmp/jev_server_test"])
        self.assertEqual((seen["selector"], seen["settings"]["selector"], seen["settings"]["multiview"],
                          seen["settings"]["multiview_grid"], seen["settings"]["jev_threshold"]),
                         ("jev_sam3", "jev_sam3", True, False, 0.5))


RELIABILITY = OFFLINE / "reliability_2026-11-05"


@unittest.skipUnless((RELIABILITY / "jevgpu.jsonl").exists() and (RELIABILITY / "jev_gate.json").exists(),
                     "offline JEV reliability data not on this machine")
class JevReplayTests(unittest.TestCase):
    """The frozen JEV gate and selector reproduce the reliability study (_analyze / _vis_conf /
    _wrong_only) on its 444 eye-labelled requests."""

    @classmethod
    def setUpClass(cls):
        import os
        import sys
        from perception import jev
        cls.offline, cls.online = jev.study_rows()
        here = os.getcwd()
        os.chdir(client.WORKSPACE)
        sys.path.insert(0, str(RELIABILITY))
        try:
            import _analyze
            import _vis_conf
            import _wrong_only
        finally:
            os.chdir(here)
            sys.path.remove(str(RELIABILITY))
        cls.A, cls.V, cls.W = _analyze, _vis_conf, _wrong_only
        cls.gate = jev.JevGate.load(client.WORKSPACE / jev.RELIABILITY / "jev_gate.json")   # the study's gate

    def test_export_reproduces_the_frozen_gate(self):
        import tempfile
        from pathlib import Path
        from perception import jev
        with tempfile.TemporaryDirectory() as folder:
            jev.export_jev_gate(f"{folder}/gate.json", "gate")
            again = json.loads(Path(folder, "gate.json").read_text())
        frozen = json.loads((client.WORKSPACE / jev.RELIABILITY / "jev_gate.json").read_text())
        self.assertEqual(again.pop("words"), "gate")   # the original gate predates the field (default "gate")
        self.assertEqual(again, frozen)
        self.assertEqual((len(self.offline), len(self.online)), (131, 313))

    def test_export_reproduces_the_hm3d_gate(self):
        """The deployed gate (HM3D words, 2026-11-09) is export_jev_gate over jev_hm3d_names.jsonl."""
        import tempfile
        from pathlib import Path
        from perception import jev
        with tempfile.TemporaryDirectory() as folder:
            jev.export_jev_gate(f"{folder}/gate.json", "approach")
            again = json.loads(Path(folder, "gate.json").read_text())
        frozen = json.loads((client.WORKSPACE / jev.JEV_GATE).read_text())
        self.assertEqual(again, frozen)
        self.assertEqual(frozen["words"], "approach")
        self.assertEqual(jev.JevGate.load(client.WORKSPACE / jev.JEV_GATE).words, "approach")

    def test_scores_and_decisions_match_the_study(self):
        rows = self.offline + self.online
        ref = self.A.refs(self.offline)
        expected = np.concatenate([self.V.u_of("entropy+conf mean", self.offline, ref),
                                   self.V.u_of("entropy+conf mean", self.online, ref)])
        got = np.array([self.gate.assess(r["jev"]["name_full"], r["target"])["score"] for r in rows])
        np.testing.assert_array_equal(got, expected)
        self.assertEqual([self.gate.decide(r["jev"]["name_full"], r["target"], forced=True)[0] == "correct"
                          for r in rows], [self.A.DECIDERS["JEV naming"](r) == 1 for r in rows])
        self.assertEqual(sorted(self.gate.entropy_reference), sorted(ref["JEV name entropy H/logK"]))
        self.assertEqual(sorted(self.gate.contention_reference), sorted(ref["JEV min(p_t, max confuser)"]))
        # _wrong_only at 45% approaches (threshold = the online quantile): 1.2% wrong among kept
        u_on = got[len(self.offline):]
        bad, _, _ = self.W.labels(self.online, "JEV naming")
        rate, kept = self.W.kept(u_on, bad, np.nanquantile(u_on, 1 - 0.45))
        self.assertAlmostEqual(rate, 0.0116, places=4)
        # the frozen threshold (45% of all 444 at or above it): 43.5% online approaches, 2.3% wrong among kept
        rate, kept = self.W.kept(u_on, bad, self.gate.threshold)
        self.assertEqual((int((got >= self.gate.threshold).sum()), int((u_on >= self.gate.threshold).sum()),
                          int(kept.sum()), int(np.nansum(bad[kept]))), (202, 136, 177, 4))
        self.assertAlmostEqual(rate, 4 / 177)

    def test_verifier_replays_the_study(self):
        """The raw jevgpu.jsonl distributions through the jev_sam3 Verifier's JEV gate (fake worker)."""
        lines = (RELIABILITY / "jevgpu.jsonl").read_text().splitlines()
        raw = {d["index"]: d for d in (json.loads(line) for line in lines if line.strip())}
        rows = self.offline + self.online
        FakeJev.answers = [raw[r["index"]]["name_full"] for r in rows]
        with patch("perception.client._Worker", FakeJev), \
                patch("perception.client.subprocess.Popen", side_effect=AssertionError("real worker")):
            verifier = client.Verifier(selector="jev_sam3", jev_gate="outputs/apexnav_offline/reliability_2026-11-05/jev_gate.json")
        self.addCleanup(verifier.close)
        mask = frame_mask(60)
        statuses = [verifier._jev(FRAME, mask, r["target"]).status for r in rows]
        for r, status in zip(rows, statuses):
            score = self.gate.assess(r["jev"]["name_full"], r["target"])["score"]
            top = self.A.top1(r["jev"]["name_full"])
            self.assertEqual(status, "insufficient" if score >= self.gate.threshold
                             else "correct" if top == r["target"] else "wrong", r["name"])
        self.assertEqual(statuses.count("insufficient"), 202)


if __name__ == "__main__":
    unittest.main()
