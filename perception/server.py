"""Serve ``perception.Verifier`` over HTTP, for a planner in another container (ApexNav).

    sam3d-env core python -m perception.server --host 0.0.0.0 --port 8770 --gpu G --selector vqf \\
        --threshold 0.15 --on-uncertain input --log-dir outputs/apexnav_gate/server
    sam3d-env core python -m perception.server --host 0.0.0.0 --port 8770 --gpu G --selector d1 \\
        --log-dir outputs/apexnav_online/<run>/server      # D1: int8 Qwen + SAM3D grid, d1_gate.json
    sam3d-env core python -m perception.server --host 0.0.0.0 --port 8770 --gpu G --selector jev_sam3 \
        --log-dir outputs/apexnav_online/<run>/server      # JEV gate (uncertainty) + SAM3 (decision)
    sam3d-env core python -m perception.server ping http://localhost:8770
    sam3d-env core python -m perception.server verify http://localhost:8770 \\
        --image frame.jpg --mask mask.png --target tv --elevation 30

One ``Verifier`` with fixed settings per run; requests are served one at a time.

- ``GET /health`` -> ``{ok, protocol, settings, workers}``; ``ok`` is false (status 503)
  once a model worker has died, and the server then exits with status 1.
- ``POST /verify`` with ``{protocol, request_id, image, mask, target, input_elevation,
  vocabulary?}`` (images base64 PNG/JPEG, mask nonzero on the object) ->
  ``{ok, verdict, view_image}``: ``Verdict.summary()`` and the judged SAM3D view as
  base64 PNG (null unless ``judged == "view"``; with selector ``qwen_grid``, the
  grid Qwen saw when ``judged == "qwen"``). ``verdict.status`` is the three-way
  qwen_grid / d1 verdict (correct/wrong/insufficient; null for the binary selectors), with
  ``verdict.source`` (sam3/qwen/jev), ``verdict.gate`` and ``verdict.qwen`` (null unless Qwen was asked);
  selector jev_sam3: ``verdict.gate`` holds JEV's folded naming distribution, entropy, contention, their
  percentiles, score and threshold, ``verdict.jev`` the raw distribution (``perception.jev``).
  A malformed request is a 400, a verifier failure a 500, both ``{ok: false, error}``.
- ``POST /verify_views`` (selector d1 or jev_sam3; ``settings.multiview`` is true) with ``{protocol,
  request_id, target, vocabulary?, sam3: fallback|always|never, grid?, grid_view?, crop_small?,
  forced2?, forced2_min?, sam3_shortcut?, grid_scores?, forced?, views: [{image, mask, input_elevation,
  step, index, range_m, scores?, jev?}]}``: one decision over up to 8 views of one object, nearest
  first (``Verifier.verify_views``; ``scores`` are SAM3 scores and ``jev`` raw JEV probabilities
  (``per_view[].probabilities``) cached by the caller, so each view is judged once) ->
  ``{ok, verdict: ViewsVerdict.summary(), grid_image}``. Every image must be exactly 640x480
  (never resized, so offline replays and live requests match) with a non-empty mask.
  Selector jev_sam3: JEV names each view and its frozen gate judges the mean distribution; unforced and
  unsure -> insufficient; otherwise (``forced``, default true, or JEV sure) SAM3 over the views decides:
  correct iff the target has the highest ``fused`` belief (``select.fuse_views``; ``aggregate``, the
  older margin-and-vote pooling, is logged only); ``grid`` is a 400. ``verdict.jev`` holds JEV's assessment, ``per_view`` each view's distribution and
  SAM3 scores.
- ``POST /name`` (selector jev_sam3; the approach belief's per-view naming) with the /verify body and
  ``words?`` -> ``{ok, jev, gate, sam3_scores}``: JEV's naming distribution of the masked object over
  ``words`` "approach" (default, HM3D words, ``jev.approach_opts``) or "gate" (``jev.name_opts``, then also the
  frozen gate's signals ``JevGate.assess``; ``gate`` is null for "approach"); with ``"sam3": true`` also the SAM3 scores /verify_views would compute for this view
  (over the body's ``vocabulary`` if given); ``"jev": false`` skips JEV (``jev`` and ``gate`` null).
- ``POST /ask`` (selector jev_sam3) with the /verify body plus ``question`` and ``options`` (2-16 strings) ->
  ``{ok, jev: {probabilities, options, latency_s, n_tokens}}``: JEV's choice about the outlined object
  (``JevJudge.ask``; the active-approach pilot asks for its next move).
- ``POST /embed`` (any selector) with ``{protocol, request_id?, images: [base64 image]}`` -> ``{ok,
  embeddings}``: L2-normalised CLIP ViT-L/14 features of the whole frames (``Verifier.embed``), for
  CLOSER's view-diversity gate (ApexNav ``perception_gate.diversity``). At most 16 images.

``--log-dir`` holds the worker logs and ``requests.jsonl`` (one line per request).
"""
import argparse
import base64
import json
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from perception import select
from perception.client import D1_EVIDENCE, HOUSEHOLD, MULTIVIEW_MAX, MULTIVIEW_SAM3, QWEN_GATE, Verifier
from perception.jev import JEV_DIR, JEV_GATE
from perception.qwen import D1_GATE
from perception.images import decode, encode

PROTOCOL = 1
EMBED_MAX = 16          # images per /embed request
VIEW_SIZE = (640, 480)  # /verify_views images: ApexNav's camera frames, never resized


class BadRequest(ValueError):
    pass


def parse_request(body):
    """The /verify arguments from a JSON body; raises BadRequest."""
    try:
        request = json.loads(body)
    except ValueError as error:
        raise BadRequest(f"body is not JSON: {error}")
    if not isinstance(request, dict):
        raise BadRequest("body must be a JSON object")
    if request.get("protocol") != PROTOCOL:
        raise BadRequest(f"protocol {request.get('protocol')!r} != server protocol {PROTOCOL}")
    missing = [key for key in ("image", "mask", "target") if not request.get(key)]
    if missing:
        raise BadRequest(f"missing field(s): {', '.join(missing)}")
    try:
        image, mask = decode(request["image"]), decode(request["mask"], "L")
        elevation = float(request.get("input_elevation", 0.0))
    except Exception as error:
        raise BadRequest(f"bad image, mask or input_elevation: {type(error).__name__}: {error}")
    vocabulary = request.get("vocabulary")
    if vocabulary is not None and not (isinstance(vocabulary, list) and all(isinstance(v, str) for v in vocabulary)):
        raise BadRequest("vocabulary must be a list of strings")
    return {"request_id": request.get("request_id"), "image": image, "mask": mask,
            "target": str(request["target"]), "input_elevation": elevation, "vocabulary": vocabulary}


def parse_views_request(body):
    """The /verify_views arguments (``Verifier.verify_views``) from a JSON body; raises BadRequest."""
    try:
        request = json.loads(body)
    except ValueError as error:
        raise BadRequest(f"body is not JSON: {error}")
    if not isinstance(request, dict):
        raise BadRequest("body must be a JSON object")
    if request.get("protocol") != PROTOCOL:
        raise BadRequest(f"protocol {request.get('protocol')!r} != server protocol {PROTOCOL}")
    if not request.get("target"):
        raise BadRequest("missing field(s): target")
    views = request.get("views")
    if not isinstance(views, list) or not views:
        raise BadRequest("views must be a non-empty list")
    if len(views) > MULTIVIEW_MAX:
        raise BadRequest(f"{len(views)} views > {MULTIVIEW_MAX}")
    sam3 = request.get("sam3", "fallback")
    if sam3 not in MULTIVIEW_SAM3:
        raise BadRequest(f"sam3 must be one of {MULTIVIEW_SAM3}")
    vocabulary = request.get("vocabulary")
    if vocabulary is not None and not (isinstance(vocabulary, list) and all(isinstance(v, str) for v in vocabulary)):
        raise BadRequest("vocabulary must be a list of strings")
    images, masks, distances, elevations, keys, cached, cached_jev = [], [], [], [], [], {}, {}
    for n, view in enumerate(views):
        if not isinstance(view, dict) or not view.get("image") or not view.get("mask"):
            raise BadRequest(f"view {n}: image and mask are required")
        try:
            image, mask = decode(view["image"]), decode(view["mask"], "L")
            elevation = float(view.get("input_elevation", 0.0))
            distance = None if view.get("range_m") is None else float(view["range_m"])
            key = (int(view["step"]), int(view["index"])) if "step" in view else (None, n)
        except Exception as error:
            raise BadRequest(f"view {n}: bad image, mask, input_elevation, range_m, step or index: "
                             f"{type(error).__name__}: {error}")
        if image.size != VIEW_SIZE or mask.size != VIEW_SIZE:
            raise BadRequest(f"view {n}: image {image.size} and mask {mask.size} must be {VIEW_SIZE}")
        if mask.getbbox() is None:
            raise BadRequest(f"view {n}: empty mask")
        if key in keys:
            raise BadRequest(f"view {n}: duplicate step/index {key}")
        scores = view.get("scores")
        if scores is not None:
            if not (isinstance(scores, dict) and all(isinstance(v, (int, float)) for v in scores.values())):
                raise BadRequest(f"view {n}: scores must map categories to numbers")
            cached[key] = {str(c): float(v) for c, v in scores.items()}
        jev = view.get("jev")
        if jev is not None:
            if not (isinstance(jev, dict) and jev and all(isinstance(v, (int, float)) for v in jev.values())):
                raise BadRequest(f"view {n}: jev must map JEV options to probabilities")
            cached_jev[key] = {str(c): float(v) for c, v in jev.items()}
        images.append(image), masks.append(mask), distances.append(distance), elevations.append(elevation)
        keys.append(key)
    try:
        grid_view = int(request.get("grid_view", 0))
        forced2_min = float(request.get("forced2_min", 0.5))
        forced = bool(request.get("forced", True))
    except (TypeError, ValueError) as error:
        raise BadRequest(f"bad grid_view, forced2_min or forced: {error}")
    for gone in ("sam3_margin", "sam3_vote"):   # jev_sam3 decides with select.fuse_views since 2026-11-07
        if gone in request:
            raise BadRequest(f"{gone} was removed: selector jev_sam3 decides with the fused SAM3 belief")
    if not 0 <= grid_view < len(views):
        raise BadRequest(f"grid_view {grid_view} is not one of the {len(views)} views")
    return {"request_id": request.get("request_id"), "target": str(request["target"]), "images": images,
            "masks": masks, "distances": distances, "input_elevations": elevations, "keys": keys,
            "cached_scores": cached, "cached_jev": cached_jev, "vocabulary": vocabulary, "sam3": sam3, "grid": bool(request.get("grid")),
            "grid_view": grid_view, "crop_small": bool(request.get("crop_small")),
            "forced2": bool(request.get("forced2")), "forced2_min": forced2_min,
            "sam3_shortcut": bool(request.get("sam3_shortcut")), "grid_scores": bool(request.get("grid_scores")),
            "forced": forced}


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/health":
            return self._reply(404, {"ok": False, "error": f"unknown path {self.path}"})
        workers = self.server.verifier.alive()
        ok = all(workers.values())
        self._reply(200 if ok else 503, {"ok": ok, "protocol": PROTOCOL, "settings": self.server.settings,
                                         "workers": workers})

    def do_POST(self):
        if self.path == "/verify_views":
            return self._verify_views()
        if self.path == "/name":
            return self._name()
        if self.path == "/detect":
            return self._detect()
        if self.path == "/embed":
            return self._embed()
        if self.path == "/ask":
            return self._ask()
        if self.path != "/verify":
            return self._reply(404, {"ok": False, "error": f"unknown path {self.path}"})
        start, record = time.time(), {"request_id": None}
        try:
            request = parse_request(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            record.update(request_id=request["request_id"], target=request["target"],
                          input_elevation=request["input_elevation"])
            verdict = self.server.verifier.verify(request["image"], request["mask"], request["target"],
                                                  request["input_elevation"], request["vocabulary"])
        except BadRequest as error:
            status, reply = 400, {"ok": False, "error": str(error)}
        except Exception as error:  # A failed verification is reported; the server keeps running.
            traceback.print_exc()
            status, reply = 500, {"ok": False, "error": f"{type(error).__name__}: {error}"}
        else:
            view = verdict.view.image if verdict.judged == "view" and verdict.view else verdict.grid
            status, reply = 200, {"ok": True, "verdict": verdict.summary(),
                                  "view_image": None if view is None else encode(view)}
            record.update(accepted=verdict.accepted, label=verdict.label, judged=verdict.judged,
                          uncertainty=verdict.uncertainty, verdict_status=verdict.status, source=verdict.source,
                          gate=verdict.gate, qwen=verdict.qwen, jev=verdict.jev)
        record.update(status=status, error=reply.get("error"), latency_s=round(time.time() - start, 3))
        self.server.log(record)
        self._reply(status, reply)

    def _verify_views(self):
        start, record = time.time(), {"request_id": None, "path": "/verify_views"}
        try:
            request = parse_views_request(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            selector = self.server.verifier.selector
            if selector not in ("d1", "jev_sam3"):
                raise BadRequest(f"/verify_views needs selector d1 or jev_sam3, not {selector!r}")
            if selector == "jev_sam3" and request["grid"]:
                raise BadRequest(f"selector {selector} has no SAM3D grid")
            record.update(request_id=request.pop("request_id"), target=request["target"],
                          n_views=len(request["images"]), keys=request["keys"], sam3=request["sam3"],
                          grid=request["grid"], crop_small=request["crop_small"], forced2=request["forced2"],
                          forced=request["forced"])
            verdict = self.server.verifier.verify_views(**request)
        except BadRequest as error:
            status, reply = 400, {"ok": False, "error": str(error)}
        except Exception as error:  # A failed verification is reported; the server keeps running.
            traceback.print_exc()
            status, reply = 500, {"ok": False, "error": f"{type(error).__name__}: {error}"}
        else:
            status, reply = 200, {"ok": True, "verdict": verdict.summary(),
                                  "grid_image": None if verdict.grid is None else encode(verdict.grid)}
            record.update(verdict_status=verdict.status, source=verdict.source, accepted=verdict.accepted,
                          aggregate=verdict.aggregate, fused=verdict.fused, qwen=verdict.qwen, grid_scores=verdict.grid_scores,
                          jev=verdict.jev,
                          per_view=[{k: v[k] for k in ("step", "index", "label", "margin", "cached", "jev_cached", "probabilities",
                                                       "entropy", "contention", "score", "error") if k in v}
                                    for v in verdict.per_view])
        record.update(status=status, error=reply.get("error"), latency_s=round(time.time() - start, 3))
        self.server.log(record)
        self._reply(status, reply)

    def _ask(self):
        """POST /ask (selector jev_sam3): the /verify body plus ``question`` and ``options`` -> {ok, jev:
        {probabilities, options, latency_s, n_tokens}}: JEV's choice among ``options`` about the outlined object."""
        start, record = time.time(), {"request_id": None, "path": "/ask"}
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            request = parse_request(body)
            extra = json.loads(body)
            question, options = extra.get("question"), extra.get("options")
            if not isinstance(question, str) or not question:
                raise BadRequest("question must be a non-empty string")
            if not (isinstance(options, list) and 2 <= len(options) <= 16 and all(isinstance(o, str) for o in options)):
                raise BadRequest("options must be a list of 2 to 16 strings")
            record.update(request_id=request["request_id"], target=request["target"])
            answer = self.server.verifier.ask_jev(request["image"], request["mask"], question, options)
        except BadRequest as error:
            status, reply = 400, {"ok": False, "error": str(error)}
        except ValueError as error:
            status, reply = 400, {"ok": False, "error": str(error)}
        except Exception as error:
            traceback.print_exc()
            status, reply = 500, {"ok": False, "error": f"{type(error).__name__}: {error}"}
        else:
            status, reply = 200, {"ok": True, "jev": answer}
            top = max(answer["probabilities"], key=answer["probabilities"].get)
            record.update(top1=top, probabilities=answer["probabilities"])
        record.update(status=status, error=reply.get("error"), latency_s=round(time.time() - start, 3))
        self.server.log(record)
        self._reply(status, reply)

    def _embed(self):
        """POST /embed: CLIP features of whole frames for the view-diversity gate."""
        start, record = time.time(), {"request_id": None, "path": "/embed"}
        try:
            try:
                request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            except ValueError as error:
                raise BadRequest(f"body is not JSON: {error}")
            if not isinstance(request, dict) or request.get("protocol") != PROTOCOL:
                raise BadRequest(f"protocol must be {PROTOCOL}")
            images = request.get("images")
            if not isinstance(images, list) or not 1 <= len(images) <= EMBED_MAX:
                raise BadRequest(f"images must be a list of 1 to {EMBED_MAX} base64 images")
            try:
                images = [decode(image) for image in images]
            except Exception as error:
                raise BadRequest(f"bad image: {type(error).__name__}: {error}")
            record.update(request_id=request.get("request_id"), n_images=len(images))
            embeddings = self.server.verifier.embed(images)
        except BadRequest as error:
            status, reply = 400, {"ok": False, "error": str(error)}
        except Exception as error:
            traceback.print_exc()
            status, reply = 500, {"ok": False, "error": f"{type(error).__name__}: {error}"}
        else:
            status, reply = 200, {"ok": True, "embeddings": embeddings}
        record.update(status=status, error=reply.get("error"), latency_s=round(time.time() - start, 3))
        self.server.log(record)
        self._reply(status, reply)

    def _name(self):
        """POST /name (selector jev_sam3; logging only, decides nothing): JEV's naming distribution of the
        masked object and the frozen gate's signals: {ok, jev: {probabilities, options, latency_s, n_tokens},
        gate: JevGate.assess}. Same body as /verify (input_elevation and vocabulary ignored)."""
        start, record = time.time(), {"request_id": None, "path": "/name"}
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            request = parse_request(body)
            verifier = self.server.verifier
            if verifier.jev is None:
                raise BadRequest(f"/name needs selector jev_sam3, not {verifier.selector!r}")
            record.update(request_id=request["request_id"], target=request["target"])
            raw = json.loads(body)
            answer = gate = None
            words = raw.get("words", "approach")
            if words not in ("approach", "gate"):
                raise BadRequest(f"words must be approach or gate, not {words!r}")
            if raw.get("jev", True):  # "jev": false -> SAM3 scores only (offline re-scoring)
                answer = verifier._jev_name(request["image"], request["mask"], request["target"], words)
                if words == verifier.jev_gate.words:  # the gate's signals are defined on its own words only
                    gate = verifier.jev_gate.assess(answer["probabilities"], request["target"])
            scores = None
            if raw.get("sam3"):  # the SAM3 scores /verify_views would compute for this view
                vocabulary = select.competitors(request["vocabulary"] or verifier.vocabulary, request["target"])
                scores = verifier.classify(request["image"], request["mask"], vocabulary)
        except BadRequest as error:
            status, reply = 400, {"ok": False, "error": str(error)}
        except Exception as error:
            traceback.print_exc()
            status, reply = 500, {"ok": False, "error": f"{type(error).__name__}: {error}"}
        else:
            status, reply = 200, {"ok": True, "jev": answer, "gate": gate, "sam3_scores": scores}
            if gate is not None:
                record.update(score=gate["score"], top1=gate["top1"])
        record.update(status=status, error=reply.get("error"), latency_s=round(time.time() - start, 3))
        self.server.log(record)
        self._reply(status, reply)

    def _detect(self):
        """POST /detect {image, target, min_score?}: SAM3 as the approach detector. Instances of the target and its
        approach look-alikes (jev.APPROACH_LOOKALIKES) scoring >= min_score (0.3), each {word, score, box, mask
        (PNG, base64), scores (SAM3 score of every vocabulary word on that segment)}: {ok, candidates}."""
        start, record = time.time(), {"request_id": None, "path": "/detect"}
        try:
            raw = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            from perception.jev import APPROACH_LOOKALIKES
            target = raw["target"]
            record.update(request_id=raw.get("request_id"), target=target)
            verifier = self.server.verifier
            vocabulary = select.competitors(verifier.vocabulary, target)
            words = [target] + [w for w in APPROACH_LOOKALIKES.get(target, []) if w in vocabulary]
            found = verifier.candidates(decode(raw["image"]), words, vocabulary, float(raw.get("min_score", 0.3)))
        except (KeyError, ValueError) as error:
            status, reply = 400, {"ok": False, "error": str(error)}
        except Exception as error:
            traceback.print_exc()
            status, reply = 500, {"ok": False, "error": f"{type(error).__name__}: {error}"}
        else:
            status, reply = 200, {"ok": True, "candidates": found}
            record.update(n=len(found))
        record.update(status=status, error=reply.get("error"), latency_s=round(time.time() - start, 3))
        self.server.log(record)
        self._reply(status, reply)

    def log_message(self, *args):  # VerifierServer.log records every /verify instead
        pass

    def _reply(self, status, reply):
        body = json.dumps(reply).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class VerifierServer(HTTPServer):
    """Single-threaded HTTP server around one ``Verifier``; ``port=0`` picks a free port."""

    def __init__(self, verifier, settings, host="127.0.0.1", port=8770, log_dir=None):
        super().__init__((host, port), _Handler)
        self.verifier, self.settings = verifier, settings
        self.requests = Path(log_dir) / "requests.jsonl" if log_dir else None

    def watch_workers(self, period=5.0):
        """Stop serving when a model worker dies (a dead worker fails every request)."""
        def watch():
            while not self.stopped.wait(period):
                workers = self.verifier.alive()
                if not all(workers.values()):
                    print(f"perception worker died: {workers}; stopping", flush=True)
                    self.worker_died = True
                    self.shutdown()
                    return

        self.stopped, self.worker_died = threading.Event(), False
        threading.Thread(target=watch, daemon=True).start()

    def log(self, record):
        print(json.dumps(record), flush=True)
        if self.requests:
            with self.requests.open("a") as stream:
                stream.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}) + "\n")


def _call(url, path, payload=None, timeout=30.0):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url.rstrip("/") + path, data=data,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:  # 400/500 still carry {ok: false, error}
        return json.loads(error.read())


def serve(argv):
    parser = argparse.ArgumentParser(prog="python -m perception.server", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--selector", choices=select.SELECTORS, default="vqf")
    parser.add_argument("--no-ensemble", action="store_true", help="turn the ensemble filter off")
    parser.add_argument("--threshold", type=float, default=0.15, help="ensemble disagreement threshold")
    parser.add_argument("--judge-input", action="store_true",
                        help="the raw frame must also show the target (rejects before SAM3D)")
    parser.add_argument("--view-when", choices=("always", "unsure"), default="always",
                        help="with --judge-input: which raw accepts get a SAM3D view")
    parser.add_argument("--unsure-margin", type=float, default=select.UNSURE_MARGIN)
    parser.add_argument("--on-uncertain", choices=select.ON_UNCERTAIN, default="input")
    parser.add_argument("--qwen-gate", help=f"frozen gate (default: qwen_grid {QWEN_GATE}, d1 {D1_GATE})")
    parser.add_argument("--qwen-int8", action=argparse.BooleanOptionalAction,
                        help="8-bit Qwen weights (default: on for d1, off for qwen_grid)")
    parser.add_argument("--d1-evidence", choices=D1_EVIDENCE, default="grid",
                        help="selector d1: Qwen sees the outlined frame plus the SAM3D grid, or the frame alone "
                             "(input: SAM3D never starts)")
    parser.add_argument("--d1-accept-min", type=float, default=None,
                        help="selector d1: Qwen 'correct' needs p(correct) >= this, else insufficient (default: top answer)")
    parser.add_argument("--lazy-workers", action="store_true",
                        help="selector d1: start SAM3D/Qwen on the first insufficient request (default: at launch). "
                             "That request then also waits for the models to load, so the planner's gate "
                             "timeout must be well above the load time; runs use the eager default")
    parser.add_argument("--stagger", action="store_true",
                        help="selector d1: SAM3D and Qwen (bf16) take turns on the GPU, each kept on CPU "
                             "between calls (fits next to another job; a few seconds per insufficient request)")
    parser.add_argument("--multiview-grid", action="store_true",
                        help="selector d1: start SAM3D for /verify_views with grid (close-range multi-view "
                             "decision with the reconstruction), also with --d1-evidence input")
    parser.add_argument("--jev-gate", help=f"selector jev_sam3: frozen gate (default {JEV_GATE})")
    parser.add_argument("--jev-model", help=f"selector jev_sam3: pre-quantized JEV folder (default {JEV_DIR})")
    parser.add_argument("--vocabulary", nargs="+", default=list(HOUSEHOLD),
                        help="default vocabulary (a request may send its own)")
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/perception_server"))
    args = parser.parse_args(argv)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    raw = args.selector in ("input", "naive", "qwen_grid", "d1", "jev_sam3")  # no ensemble filter, no judge_input
    ensemble = not args.no_ensemble and not raw
    settings = {"selector": args.selector, "ensemble": ensemble, "threshold": args.threshold if ensemble else None,
                "on_uncertain": args.on_uncertain, "judge_input": args.judge_input and not raw,
                "view_when": args.view_when, "unsure_margin": args.unsure_margin,
                "vocabulary": args.vocabulary, "gpu": args.gpu}
    if args.selector in ("qwen_grid", "d1"):
        settings.update(qwen_gate=args.qwen_gate or (D1_GATE if args.selector == "d1" else QWEN_GATE),
                        qwen_int8=args.selector == "d1" and not args.stagger if args.qwen_int8 is None
                        else args.qwen_int8)
    if args.selector == "d1":
        settings.update(d1_evidence=args.d1_evidence, d1_accept_min=args.d1_accept_min, lazy_workers=args.lazy_workers, stagger=args.stagger,
                        multiview=True, multiview_grid=args.multiview_grid)
    if args.selector == "jev_sam3":  # no SAM3D grid
        from perception.client import WORKSPACE
        from perception.jev import JevGate
        gate = args.jev_gate or JEV_GATE
        settings.update(jev_gate=gate, jev_model=args.jev_model or JEV_DIR,
                        jev_threshold=JevGate.load(WORKSPACE / gate).threshold, multiview=True, multiview_grid=False)
    with Verifier(selector=args.selector, ensemble=ensemble, ensemble_threshold=args.threshold,
                  on_uncertain=args.on_uncertain, vocabulary=args.vocabulary, gpu=args.gpu,
                  judge_input=settings["judge_input"], view_when=args.view_when,
                  unsure_margin=args.unsure_margin, qwen_gate=args.qwen_gate, qwen_int8=args.qwen_int8,
                  d1_evidence=args.d1_evidence, d1_accept_min=args.d1_accept_min, lazy_workers=args.lazy_workers,
                  stagger=args.stagger, multiview_grid=args.multiview_grid and args.selector == "d1",
                  jev_gate=args.jev_gate, jev_model=args.jev_model, log_dir=args.log_dir) as verifier:
        server = VerifierServer(verifier, settings, args.host, args.port, args.log_dir)
        print(f"perception server on http://{args.host}:{server.server_port} {json.dumps(settings)}", flush=True)
        server.watch_workers()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.stopped.set()
            server.server_close()
    if server.worker_died:
        sys.exit(1)


def ping(argv):
    parser = argparse.ArgumentParser(prog="python -m perception.server ping")
    parser.add_argument("url")
    print(json.dumps(_call(parser.parse_args(argv).url, "/health"), indent=1))


def verify(argv):
    parser = argparse.ArgumentParser(prog="python -m perception.server verify")
    parser.add_argument("url")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--elevation", type=float, default=0.0,
                        help="camera elevation in degrees, positive looking down")
    parser.add_argument("--view-out", type=Path, help="save the judged SAM3D view here")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)
    payload = {"protocol": PROTOCOL, "request_id": f"cli-{int(time.time())}",
               "image": base64.b64encode(args.image.read_bytes()).decode("ascii"),
               "mask": encode(args.mask, "L"), "target": args.target, "input_elevation": args.elevation}
    start = time.time()
    reply = _call(args.url, "/verify", payload, timeout=args.timeout)
    if reply.get("view_image") and args.view_out:
        decode(reply["view_image"]).save(args.view_out)
    reply["view_image"] = bool(reply.get("view_image"))
    reply["latency_s"] = round(time.time() - start, 2)
    print(json.dumps(reply, indent=1))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    commands = {"ping": ping, "verify": verify}
    if argv and argv[0] in commands:
        return commands[argv[0]](argv[1:])
    serve(argv)


if __name__ == "__main__":
    main()
