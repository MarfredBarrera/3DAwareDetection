"""Serve ``perception.Verifier`` over HTTP, for a planner in another container (ApexNav).

    sam3d-env core python -m perception.server --host 0.0.0.0 --port 8770 --gpu G --selector vqf \\
        --threshold 0.15 --on-uncertain input --log-dir outputs/apexnav_gate/server
    sam3d-env core python -m perception.server ping http://localhost:8770
    sam3d-env core python -m perception.server verify http://localhost:8770 \\
        --image frame.jpg --mask mask.png --target tv --elevation 30

One ``Verifier`` with fixed settings per run; requests are served one at a time.

- ``GET /health`` -> ``{ok, protocol, settings}``.
- ``POST /verify`` with ``{protocol, request_id, image, mask, target, input_elevation,
  vocabulary?}`` (images base64 PNG/JPEG, mask nonzero on the object) ->
  ``{ok, verdict, view_image}``: ``Verdict.summary()`` and the judged SAM3D view as
  base64 PNG (null unless ``judged == "view"``). A malformed request is a 400, a
  verifier failure a 500, both ``{ok: false, error}``.

``--log-dir`` holds the worker logs and ``requests.jsonl`` (one line per request).
"""
import argparse
import base64
import json
import sys
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from perception import select
from perception.client import HOUSEHOLD, Verifier
from perception.images import decode, encode

PROTOCOL = 1


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


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/health":
            return self._reply(404, {"ok": False, "error": f"unknown path {self.path}"})
        self._reply(200, {"ok": True, "protocol": PROTOCOL, "settings": self.server.settings})

    def do_POST(self):
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
            view = verdict.view.image if verdict.judged == "view" and verdict.view else None
            status, reply = 200, {"ok": True, "verdict": verdict.summary(),
                                  "view_image": None if view is None else encode(view)}
            record.update(accepted=verdict.accepted, label=verdict.label, judged=verdict.judged,
                          uncertainty=verdict.uncertainty)
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
    parser.add_argument("--on-uncertain", choices=select.ON_UNCERTAIN, default="input")
    parser.add_argument("--vocabulary", nargs="+", default=list(HOUSEHOLD),
                        help="default vocabulary (a request may send its own)")
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/perception_server"))
    args = parser.parse_args(argv)

    from tools.envs import no_huge_pages
    no_huge_pages()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    settings = {"selector": args.selector, "ensemble": not args.no_ensemble,
                "threshold": None if args.no_ensemble else args.threshold,
                "on_uncertain": args.on_uncertain, "vocabulary": args.vocabulary, "gpu": args.gpu}
    with Verifier(selector=args.selector, ensemble=not args.no_ensemble, ensemble_threshold=args.threshold,
                  on_uncertain=args.on_uncertain, vocabulary=args.vocabulary, gpu=args.gpu,
                  log_dir=args.log_dir) as verifier:
        server = VerifierServer(verifier, settings, args.host, args.port, args.log_dir)
        print(f"perception server on http://{args.host}:{server.server_port} {json.dumps(settings)}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


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
