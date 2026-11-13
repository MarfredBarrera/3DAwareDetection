"""Persistent model server for ``perception.Verifier``: one JSON request per stdin line.

SAM3 (``core`` env) and SAM3D + ViewActive (``sam3d`` env) need incompatible
PyTorch builds, so the client runs one server per role and keeps the models
loaded between calls. Qwen3-VL (role ``qwen``, selector ``qwen_grid``) and JEV-27B-VL (role
``jev``, selector ``jev``) run in the qwen venv (``client.QWEN_ENV``). Replies go to the
original stdout as one JSON line each; everything else the models print is redirected to stderr.

    request: {"op": "<method>", "args": {...}}
    reply:   {"ok": true, "result": ...} | {"ok": false, "error": "..."}
"""
import argparse
import base64
import contextlib
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

from perception.images import decode, encode
from tools.envs import use_gpu


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

    def candidates(self, image, prompts, words, min_score=0.3, same=0.5):
        """SAM3 as the approach detector: every instance of one of ``words`` (the target and its look-alikes)
        scoring >= ``min_score`` in the full frame, each with its mask, box and the SAM3 score of every prompt on
        that same segment (the best instance of the prompt overlapping it by mask IoU >= ``same``):
        [{word, score, box, mask, scores}]. The caller decides which candidate is the approached object."""
        import numpy as np

        found = self.model.detect_many(decode(image), prompts)
        out = []
        for word in words:
            d = found.get(word)
            if d is None:
                continue
            for n, s in enumerate(d["scores"]):
                if not np.isfinite(s) or s < min_score:
                    continue
                mask, box = d["masks"][n], d["boxes"][n]
                scores = {}
                for prompt, other in found.items():
                    best = 0.0
                    for k in np.argsort(-np.nan_to_num(other["scores"], nan=-1.0)):
                        t = float(other["scores"][k])
                        if not np.isfinite(t) or t <= best:
                            break   # best first: nothing after this can raise the score
                        b = other["boxes"][k]
                        if b[0] > box[2] or b[2] < box[0] or b[1] > box[3] or b[3] < box[1]:
                            continue   # boxes do not touch: not the same segment
                        if _iou(other["masks"][k], mask) >= same:
                            best = t
                            break
                    scores[prompt] = best
                out.append({"word": word, "score": float(s), "box": [float(v) for v in d["boxes"][n]],
                            "mask": encode(mask, "L"), "scores": scores})
        return out

    def classify_views(self, images, prompts):
        return [self.classify(image, prompts) for image in images]

    def embed(self, images):
        """L2-normalised CLIP ViT-L/14 features of whole frames (CLOSER's diversity gate): one list of
        floats per image. The model loads on first use (fp16, ~0.9 GB)."""
        import torch

        if getattr(self, "clip", None) is None:
            import open_clip
            from torchvision import transforms

            from perception.client import CLIP_DIR
            model = open_clip.create_model("ViT-L-14-quickgelu", pretrained="openai", cache_dir=str(CLIP_DIR))
            self.clip = model.half().cuda().eval()
            # the whole frame, squashed to 224 x 224 (no centre crop: truncation at the borders stays visible)
            self.clip_preprocess = transforms.Compose([
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        batch = torch.stack([self.clip_preprocess(decode(image)) for image in images]).half().cuda()
        with torch.no_grad():
            features = self.clip.encode_image(batch).float()
        return torch.nn.functional.normalize(features, dim=-1).cpu().tolist()

    def segment(self, image, query, threshold=0.0, max_masks=1):
        """Up to ``max_masks`` SAM3 instances of ``query``, best first: [{mask, score, box}]."""
        frame = decode(image)
        found = self.model.detect(frame, query)
        keep = self.model.highest_confidence_indices(found["scores"], len(found["masks"]),
                                                     threshold, max_masks)
        return [{"mask": encode(found["masks"][i], "L"), "score": float(found["scores"][i]),
                 "box": [float(v) for v in found["boxes"][i]]} for i in keep]


@contextlib.contextmanager
def _on_gpu(model, offload):
    """With ``offload`` (selector d1 --stagger), the model is on the GPU only for this call, so
    SAM3D and Qwen take turns with the same memory."""
    if offload:
        model.onload()
    try:
        yield
    finally:
        if offload:
            model.offload()


class SAM3DServer:
    def __init__(self, args):
        from reconstruction.sam3d import SAM3D

        self.model = SAM3D(checkpoint_dir=args.sam3d, gaussian_only=args.gaussian_only)
        self.offload = args.offload
        if self.offload:
            self.model.offload()
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
        with _on_gpu(self.model, self.offload), tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            decode(image).save(folder / "input.png")
            decode(mask, "L").save(folder / "mask.png")
            rendered = self.model.render_views(
                folder / "input.png", folder / "mask.png", folder, azimuths, elevations,
                input_elevation=input_elevation, distance=distance, resolution=resolution,
                seed=seed, ensemble=ensemble, max_uncertainty=max_uncertainty, save_mesh=False)
            uncertainty = (self.model.last_ensemble or {}).get("disagreement")
            if self.model.gaussian_only:  # selector d1 shares one GPU: hand cached memory back
                import torch
                torch.cuda.empty_cache()
            return {"uncertainty": uncertainty, "timing": self.model.last_timing,
                    "views": [{"azimuth": c["azimuth"], "elevation": c["elevation"],
                               "image": base64.b64encode((folder / "views" / c["image"]).read_bytes())
                               .decode("ascii")} for c in rendered]}


class QwenServer:
    def __init__(self, args):
        from perception.qwen import QwenJudge

        self.model = QwenJudge(args.qwen, int8=args.int8)
        self.offload = args.offload
        if self.offload:
            self.model.offload()

    def judge(self, image, mask, target, grid=None):
        """Three-way verdict on the masked object (outlined in red), with the SAM3D grid if
        given: ``perception.qwen.QwenJudge.judge``."""
        from perception.qwen import outlined

        with _on_gpu(self.model, self.offload):
            return self.model.judge(outlined(decode(image), decode(mask, "L")), target,
                                    None if grid is None else decode(grid))

    def judge_views(self, images, masks, target, distances=None, grid=None, crop_small=False, forced2=False,
                    forced2_min=0.5):
        """Three-way verdict on one object from several views (nearest first), each outlined in red,
        with the SAM3D grid if given: ``perception.qwen.QwenJudge.judge_views``. ``crop_small``: views
        whose mask covers < ``qwen.CROP_SMALL_AREA`` of the frame also get an enlarged crop."""
        import numpy as np

        from perception.qwen import CROP_SMALL_AREA, crop_around, outlined

        frames, crops = [], []
        for image, mask in zip(images, masks):
            image, mask = decode(image), decode(mask, "L")
            frame = outlined(image, mask)
            frames.append(frame)
            area = float((np.asarray(mask) > 0).mean())
            crops.append(crop_around(frame, mask) if crop_small and area < CROP_SMALL_AREA else None)
        with _on_gpu(self.model, self.offload):
            return self.model.judge_views(frames, target, distances, None if grid is None else decode(grid),
                                          crops, forced2=forced2, forced2_min=forced2_min)


class JevServer:
    def __init__(self, args):
        from perception.jev import JEV_DIR, JevJudge

        self.model = JevJudge(args.jev or JEV_DIR)

    def name(self, image, mask, target, words="gate"):
        """JEV's raw naming distribution for the masked object (outlined in red):
        ``perception.jev.JevJudge.choice`` (``words``: "gate" or "approach")."""
        from perception.qwen import outlined

        return self.model.choice(outlined(decode(image), decode(mask, "L")), target, words)

    def ask(self, image, mask, question, options):
        """JEV's distribution over ``options`` for ``question`` about the masked object (outlined in red):
        ``perception.jev.JevJudge.ask``."""
        from perception.qwen import outlined

        return self.model.ask(outlined(decode(image), decode(mask, "L")), question, options)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=("sam3", "sam3d", "qwen", "jev"), required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sam3", default="/data/checkpoints/sam3/sam3.pt")
    parser.add_argument("--sam3d", default="/data/checkpoints/sam3d/hf")
    parser.add_argument("--viewactive", help="ViewActive checkpoint (sam3d role; enables the vqf op)")
    parser.add_argument("--gaussian-only", action="store_true",
                        help="sam3d role: decode Gaussians only, no mesh/GLB (less GPU memory; selector d1)")
    parser.add_argument("--qwen", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--int8", action="store_true", help="qwen role: 8-bit weights")
    parser.add_argument("--jev", default=None, help="jev role: model folder (default perception.jev.JEV_DIR)")
    parser.add_argument("--offload", action="store_true",
                        help="sam3d/qwen roles: keep the model on CPU between calls (selector d1 --stagger)")
    args = parser.parse_args()
    if args.role == "jev":  # JevJudge._slim frees ~4.7 GiB inside loading-time segments: let the allocator return it
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    use_gpu(args.gpu)
    replies = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    server = {"sam3": SAM3Server, "sam3d": SAM3DServer, "qwen": QwenServer, "jev": JevServer}[args.role](args)
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
