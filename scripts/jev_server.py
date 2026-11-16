"""JEV-27B-VL "can't tell" judge over HTTP, for the active-approach harness (ApexNav active_approach, --judge jev).

The user's chosen signal (2026-11-05, outputs/apexnav_offline/reliability_2026-11-05 README "JEV decides; only wrong
decisions count"): JEV naming question on the red-outlined frame (same template, options, decision head and
temperature as reliability_2026-11-05/_jev_run.py; p(armchair) folded into p(chair) for chair targets), then
    u = mean(pct(entropy H/log16), pct(min(p_target, max confuser p)))   (midrank pct vs the offline reference)
uncertain iff u >= threshold (45% approach rate), frozen in
outputs/apexnav_online/active_approach_2026-11-05/jev_gate.json (_jev_gate.py). Decision: JEV naming top-1 == target.

POST /judge {"image": b64 PNG, "mask": b64 PNG (nonzero = object), "target": str}  (one request at a time)
  -> {"status": correct|wrong|insufficient, "u", "entropy", "con", "top1", "name": {option: p}}
GET /health -> {"ok": true}

Run in a fresh container with the qwen env, the repo and the NAS models (int8 on one GPU, ~29 GiB):
    docker run -d --rm --name jev-judge --gpus device=$GPU --memory 60g --network 3dawareverification_default \
      -v ~/codes/3DAwareVerification:/workspace -v ~/codes/3DAwareVerification/data:/data \
      -v /ICONAS/users/marfred/models:/models:ro -w /workspace 3d-aware:latest \
      bash -c "OMP_NUM_THREADS=4 /data/envs/qwen/bin/python scripts/jev_server.py --port 8771"
"""
import argparse
import base64
import io
import json
import math
import sys
from bisect import bisect_left, bisect_right
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

sys.path.insert(0, "/workspace")
from perception.qwen import outlined  # noqa: E402

GATE = "outputs/apexnav_online/active_approach_2026-11-05/jev_gate.json"
# identical to reliability_2026-11-05/_jev_run.py (naming options per target)
CONFUSERS = {"couch": ["armchair", "chair", "bench", "bed", "cushion"],
             "chair": ["armchair", "stool", "bench", "couch", "table"],
             "bed": ["couch", "bench", "ottoman or pouf", "table", "clothes"],
             "tv": ["picture", "mirror", "window", "cabinet", "monitor"],
             "potted plant": ["vase", "lamp", "flowers", "sculpture", "curtain"],
             "toilet": ["sink", "bathtub", "toilet paper holder", "trash can"]}
FILLERS = ["chair", "couch", "bed", "toilet", "tv", "potted plant", "table", "lamp", "cabinet", "picture", "mirror",
           "window", "door", "shelf", "pillow", "rug", "box", "sink"]
IMG = "<|vision_start|><|image_pad|><|vision_end|>"


def name_opts(t):
    o = [t] + CONFUSERS[t]
    o += [f for f in FILLERS if f not in o][:15 - len(o)]
    return sorted(o) + ["something else"]


class JEV:
    def __init__(self, path):
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
        path = Path(path)
        self.proc = AutoProcessor.from_pretrained(path)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            path, dtype=torch.bfloat16, device_map="cuda",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True)).eval()
        cfg = json.load(open(path / "adapter_vllm/adapter_config.json"))
        scale = cfg["lora_alpha"] / cfg["r"]
        sd = load_file(path / "adapter_vllm/adapter_model.safetensors")
        mods = dict(self.model.named_modules())
        for k in sd:  # LoRA as unmerged forward hooks, as _jev_run.py INT8=1
            if ".lora_A." not in k:
                continue
            name = k.split("base_model.model.", 1)[-1].split(".lora_A.")[0]
            m = next(mods[c] for c in (name, name.replace("model.", "model.language_model.", 1), "model." + name)
                     if c in mods)
            a = sd[k].float().to("cuda", torch.bfloat16)
            b = (scale * sd[k.replace(".lora_A.", ".lora_B.")].float()).to("cuda", torch.bfloat16)
            m.register_forward_hook(lambda mod, inp, out, a=a, b=b: out + ((inp[0].to(a.dtype) @ a.T) @ b.T).to(out.dtype))
        self.head = json.load(open(path / "adapter_vllm/decision_head.json"))
        self.temp = json.load(open(path / "calibration.json"))["per_kind"]

    @torch.inference_mode()
    def choice(self, state, question, options):
        lines = [f"{'ABCDEFGHIJKLMNOP'[i]}) {o}" for i, o in enumerate(options)]
        images = [p for p in state if not isinstance(p, str)]
        text = "[kind] choice\n[state] " + "".join(p if isinstance(p, str) else IMG for p in state)
        text += f"\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"
        x = self.proc(text=[text], images=images, return_tensors="pt", add_special_tokens=False).to("cuda")
        logits = self.model(**x, logits_to_keep=1).logits[0, -1].float().log_softmax(-1)
        s = self.head["slots"]["ranges"]["choice"][0]
        ids = self.head["verbalizer_ids"][s: s + len(options)]
        z = [(logits[t].item() + self.head["bias"][s + i]) / self.temp["choice"] for i, t in enumerate(ids)]
        e = [math.exp(v - max(z)) for v in z]
        return {o: round(v / sum(e), 5) for o, v in zip(options, e)}

    def name(self, image, mask, target):
        state = ["Robot camera image of a room; one object is outlined in red: ", outlined(image, mask)]
        d = self.choice(state, "What is the object outlined in red?", name_opts(target))
        if target == "chair":  # the project counts an armchair as a chair (_analyze.merge_armchair)
            d["chair"] = d["chair"] + d.pop("armchair", 0.0)
        return d


def midrank(ref, v):
    return (bisect_left(ref, v) + bisect_right(ref, v)) / (2 * len(ref))


def assess(d, target, gate):
    p = [x for x in d.values() if x > 0]
    entropy = float(-sum(x * math.log(x) for x in p) / math.log(16)) if len(p) > 1 else 0.0
    con = min(d[target], max(d.get(c, 0.0) for c in gate["confusers"][target]))
    u = (midrank(gate["ref_entropy"], entropy) + midrank(gate["ref_con"], con)) / 2
    top1 = max(d, key=d.get)
    status = "insufficient" if u >= gate["threshold"] else ("correct" if top1 == target else "wrong")
    return {"status": status, "u": round(u, 4), "entropy": round(entropy, 4), "con": round(con, 5), "top1": top1,
            "name": d}


def decode(text, mode):
    return Image.open(io.BytesIO(base64.b64decode(text))).convert(mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--model", default="/models/JEV-27B-VL")
    parser.add_argument("--gate", default=GATE)
    args = parser.parse_args()
    gate = json.load(open(args.gate))
    jev = JEV(args.model)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._reply(200, {"ok": True, "threshold": gate["threshold"]})

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if req["target"] not in CONFUSERS:
                    raise ValueError(f"no JEV option set for target {req['target']!r}")
                image = decode(req["image"], "RGB")
                self._reply(200, assess(jev.name(image, decode(req["mask"], "L"), req["target"]), req["target"], gate))
            except Exception as error:
                self._reply(500, {"error": f"{type(error).__name__}: {error}"})

        def _reply(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    print(f"JEV judge on :{args.port}, threshold {gate['threshold']:.4f}", flush=True)
    HTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
