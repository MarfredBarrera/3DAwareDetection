"""Persistent model server for ``perception.Verifier``: one JSON request per stdin line.

SAM3 (``core`` env) and SAM3D + ViewActive (``sam3d`` env) need incompatible
PyTorch builds, so the client runs one server per role and keeps the models
loaded between calls. Replies go to the original stdout as one JSON line each;
everything else the models print is redirected to stderr.

    request: {"op": "<method>", "args": {...}}
    reply:   {"ok": true, "result": ...} | {"ok": false, "error": "..."}
"""
import argparse
import base64
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

# Before numpy is imported: huge-page advice stalls the shared host until
# systemd-oomd kills the container (see tools.envs.no_huge_pages).
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

from perception.images import decode, encode  # noqa: E402
from tools.envs import no_huge_pages, use_gpu  # noqa: E402


def _iou(a, b):
    import numpy as np

    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


class SAM3Server:
    def __init__(self, args):
        from reconstruction.segmentation import SAM3Preprocessor

        self.model = SAM3Preprocessor(checkpoint=args.sam3, threshold=0.0)

    def classify(self, image, prompts, mask=None, iou=0.3):
        """{prompt: best SAM3 score}. With ``mask`` (the object in a full frame), only
        detections overlapping it by ``iou`` count; 0 when none do."""
        import numpy as np

        frame = decode(image)
        found = self.model.detect_many(frame, prompts, spatial_queries=None if mask else set())
        if mask is not None:
            mask = np.asarray(decode(mask, "L")) > 127
        scores = {}
        for prompt, detection in found.items():
            values = [float(s) for n, s in enumerate(detection["scores"]) if np.isfinite(s)
                      and (mask is None or _iou(detection["masks"][n], mask) >= iou)]
            scores[prompt] = max(values, default=0.0)
        return scores

    def classify_views(self, images, prompts):
        return [self.classify(image, prompts) for image in images]

    def segment(self, image, query, threshold=0.0, max_masks=1):
        """Up to ``max_masks`` SAM3 instances of ``query``, best first: [{mask, score, box}]."""
        frame = decode(image)
        found = self.model.detect(frame, query)
        keep = self.model.highest_confidence_indices(found["scores"], len(found["masks"]),
                                                     threshold, max_masks)
        return [{"mask": encode(found["masks"][i], "L"), "score": float(found["scores"][i]),
                 "box": [float(v) for v in found["boxes"][i]]} for i in keep]


class SAM3DServer:
    def __init__(self, args):
        from reconstruction.sam3d import SAM3D

        self.model = SAM3D(checkpoint_dir=args.sam3d)
        self.viewactive = None
        if args.viewactive:
            from viewactive.model import ViewActive
            self.viewactive = ViewActive.load(args.viewactive)

    def vqf(self, image, mask, input_elevation, angles):
        """Learned ViewActive quality of each SAM3D candidate angle:
        [[quality, absolute elevation]] in the order of ``angles``."""
        from viewactive.grid import sam3d_to_viewactive

        if self.viewactive is None:
            raise RuntimeError("worker started without --viewactive")
        field = self.viewactive.predict([decode(image)], [decode(mask, "L")])[0]
        converted = [sam3d_to_viewactive(a, e, input_elevation) for a, e in angles]
        quality = self.viewactive.score_views(field, *zip(*converted))
        return [[float(q), float(e)] for q, (_, e) in zip(quality, converted)]

    def reconstruct(self, image, mask, input_elevation, azimuths, elevations, distance=2.0,
                    resolution=320, seed=42, ensemble=1, max_uncertainty=None):
        """Reconstruct, then render the azimuths x elevations grid unless the ensemble
        disagreement exceeds ``max_uncertainty``.
        Returns {uncertainty, views: [{azimuth, elevation, image}], timing}."""
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            decode(image).save(folder / "input.png")
            decode(mask, "L").save(folder / "mask.png")
            rendered = self.model.render_views(
                folder / "input.png", folder / "mask.png", folder, azimuths, elevations,
                input_elevation=input_elevation, distance=distance, resolution=resolution,
                seed=seed, ensemble=ensemble, max_uncertainty=max_uncertainty, save_mesh=False)
            uncertainty = (self.model.last_ensemble or {}).get("disagreement")
            return {"uncertainty": uncertainty, "timing": self.model.last_timing,
                    "views": [{"azimuth": c["azimuth"], "elevation": c["elevation"],
                               "image": base64.b64encode((folder / "views" / c["image"]).read_bytes())
                               .decode("ascii")} for c in rendered]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=("sam3", "sam3d"), required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sam3", default="/data/checkpoints/sam3/sam3.pt")
    parser.add_argument("--sam3d", default="/data/checkpoints/sam3d/hf")
    parser.add_argument("--viewactive", help="ViewActive checkpoint (sam3d role; enables the vqf op)")
    args = parser.parse_args()
    use_gpu(args.gpu)
    no_huge_pages()
    replies = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    server = (SAM3Server if args.role == "sam3" else SAM3DServer)(args)
    replies.write(json.dumps({"ok": True, "result": "ready"}) + "\n")
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            reply = {"ok": True, "result": getattr(server, request["op"])(**request.get("args", {}))}
        except Exception as error:  # Reported to the client; the server keeps running.
            traceback.print_exc()
            reply = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        replies.write(json.dumps(reply) + "\n")


if __name__ == "__main__":
    main()
