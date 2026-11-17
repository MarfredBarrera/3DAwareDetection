"""JEV-27B-VL naming and its frozen uncertainty gate (selector ``jev_sam3``: JEV judges the uncertainty).

One JEV ``choice`` call per frame on the outlined frame (``qwen.outlined``): "What is the
object outlined in red?" over ``name_opts(target)`` (the target, its look-alikes
``CONFUSERS``, general ``FILLERS`` up to 15, sorted, then "something else"). During an approach the
belief names over ``approach_opts(target)`` instead (the same recipe over SAM3's HM3D words). For chair
targets p(armchair) is folded into p(chair) (``fold``). The top-1 option decides.

Uncertainty (``JevGate``): score = mean of the midrank percentiles, in frozen references (the
131 offline eye-labelled rows of reliability_2026-11-05/jevgpu.jsonl), of the normalized naming
entropy H/log 16 and the look-alike contention min(p_target, max confuser p). A first request
with score >= threshold is "insufficient" (the planner approaches); the threshold puts 45% of
the 444 study requests at or above it (``export_jev_gate``). Close range (``Verifier.verify_views``)
judges the mean of the views' distributions the same way; SAM3 makes the decisions.

Reproduces outputs/apexnav_offline/reliability_2026-11-05/_jev_run.py (model, "bare-v1"
template, verbalizer ids + bias / temperature) and _analyze.py / _vis_conf.py ("entropy+conf
mean"). Targets without a confuser list (OVON) get the fillers only and contention 0.

Weights: ``JEV_DIR`` (/data/models/JEV-27B-VL-int8, local disk) is the release quantized once to
bitsandbytes 8-bit by ``quantize`` (from ``JEV_SOURCE``, the bf16 release on the NAS), so a load reads
~28 GB of ready int8 weights instead of quantizing 53 GB from the NAS.

    /data/envs/qwen/bin/python -m perception.jev dry-run <request folder> <target>   # processor only
    /data/envs/qwen/bin/python -m perception.jev quantize [src dst]                   # once: JEV_SOURCE -> JEV_DIR
    sam3d-env core python -m perception.jev export                                   # freeze JEV_GATE
"""
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import math
import os
import time

JEV_SOURCE = "/models/JEV-27B-VL"  # the bf16 release (NAS: /ICONAS/users/marfred/models), read only by quantize
JEV_DIR = os.environ.get("JEV_DIR", "/data/models/JEV-27B-VL-int8")  # the pre-quantized copy JevJudge loads
RELIABILITY = "outputs/apexnav_offline/reliability_2026-11-05"
JEV_GATE = f"{RELIABILITY}/jev_gate_hm3d.json"   # HM3D words (2026-11-09); the original: jev_gate.json
TEMPLATE_VERSION = "bare-v1"
APPROACH_RATE = 0.45  # fraction of the 444 study requests the frozen threshold sends to an approach
# Per-target look-alikes and general fillers, verbatim from _jev_run.py (fixed up front, not tuned).
CONFUSERS = {"couch": ["armchair", "chair", "bench", "bed", "cushion"],
             "chair": ["armchair", "stool", "bench", "couch", "table"],
             "bed": ["couch", "bench", "ottoman or pouf", "table", "clothes"],
             "tv": ["picture", "mirror", "window", "cabinet", "monitor"],
             "potted plant": ["vase", "lamp", "flowers", "sculpture", "curtain"],
             "toilet": ["sink", "bathtub", "toilet paper holder", "trash can"]}
FILLERS = ["chair", "couch", "bed", "toilet", "tv", "potted plant", "table", "lamp", "cabinet", "picture", "mirror",
           "window", "door", "shelf", "pillow", "rug", "box", "sink"]
OTHER = "something else"
STATE = "Robot camera image of a room; one object is outlined in red: "
QUESTION = "What is the object outlined in red?"
IMG = "<|vision_start|><|image_pad|><|vision_end|>"
LETTERS = "ABCDEFGHIJKLMNOP"


def known(target):
    """True when ``target`` has a confuser list (the six HM3D goals)."""
    return target in CONFUSERS


def name_opts(target):
    """The 16 naming options: target + confusers + fillers to 15, sorted, then "something else".
    An unknown target gets the fillers only (and itself)."""
    options = [target] + CONFUSERS.get(target, [])
    options += [f for f in FILLERS if f not in options][:15 - len(options)]
    return sorted(options) + [OTHER]


def contention_confusers(target):
    """The look-alikes contention compares with: ``CONFUSERS`` without armchair for chair (folded)."""
    return [c for c in CONFUSERS.get(target, []) if not (target == "chair" and c == "armchair")]


def decision_text(kind, state, question, options):
    """The "bare-v1" decision template of _jev_run.decide (``state``: strings, and None per image)."""
    lines = options if kind != "choice" else [f"{LETTERS[i]}) {o}" for i, o in enumerate(options)]
    text = f"[kind] {kind}\n[state] " + "".join(p if isinstance(p, str) else IMG for p in state)
    return text + f"\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"


# Approach-phase naming (2026-11-07, docs/selfcontained_view_planners.md 6.6): during an approach JEV names over
# SAM3's HM3D vocabulary (client.HOUSEHOLD) only. Same recipe as name_opts: the target, its HM3D look-alikes
# (CONFUSERS mapped to HM3D classes: armchair -> chair, bench / ottoman -> stool; vase, flowers, sculpture,
# monitor, toilet paper holder, trash can have no HM3D class), HM3D fillers to 15, sorted, then "something else".
# The start-frame gate (JevGate) keeps name_opts: its references and threshold were fit on those words.
APPROACH_LOOKALIKES = {"couch": ["chair", "stool", "bed", "cushion"],
                       "chair": ["stool", "couch", "table"],
                       "bed": ["couch", "stool", "table", "clothes"],
                       "tv": ["picture", "mirror", "window", "cabinet"],
                       "potted plant": ["lamp", "curtain"],
                       "toilet": ["sink", "bathtub", "shower"]}
APPROACH_FILLERS = ["chair", "couch", "bed", "toilet", "tv", "potted plant", "table", "lamp", "cabinet", "picture",
                    "mirror", "window", "door", "shelf", "sink", "stool", "cushion", "counter", "curtain", "towel"]


def approach_opts(target):
    """The 16 approach-phase naming options over HM3D words (see APPROACH_LOOKALIKES)."""
    options = [target] + APPROACH_LOOKALIKES.get(target, [])
    options += [f for f in APPROACH_FILLERS if f not in options][:15 - len(options)]
    return sorted(options) + [OTHER]


def naming_text(target, words="gate"):
    """(text, options) of the naming question for ``target`` on one outlined frame. ``words``: "gate"
    (``name_opts``, the start-frame gate's words) or "approach" (``approach_opts``, HM3D words)."""
    if words not in ("gate", "approach"):
        raise ValueError(f"words must be gate or approach, not {words!r}")
    options = name_opts(target) if words == "gate" else approach_opts(target)
    return decision_text("choice", [STATE, None], QUESTION, options), options


def fold(probabilities, target):
    """For chair targets, p(armchair) added to p(chair) (an armchair is a chair); a copy."""
    folded = dict(probabilities)
    if target == "chair" and "armchair" in folded:
        folded["chair"] = folded.get("chair", 0.0) + folded.pop("armchair")
    return folded


def entropy(probabilities, k=16):
    """Normalized entropy H / log k over the positive probabilities, bit for bit as _analyze.ent."""
    import numpy as np
    p = np.array([x for x in probabilities.values() if x > 0])
    return float(-(p * np.log(p)).sum() / math.log(k)) if len(p) > 1 else 0.0


def contention(probabilities, target, words="gate"):
    """min(p_target, max look-alike p) on a folded distribution; 0 without a confuser list. ``words``: the
    look-alikes of that word list ("gate": ``CONFUSERS``, "approach": ``APPROACH_LOOKALIKES``)."""
    confusers = contention_confusers(target) if words == "gate" else APPROACH_LOOKALIKES.get(target, [])
    if not confusers:
        return 0.0
    return min(probabilities.get(target, 0.0), max(probabilities.get(c, 0.0) for c in confusers))


def top1(probabilities):
    """The most probable option; ties go to the first in option order (_analyze.top1)."""
    return max(probabilities, key=probabilities.get)


def average(distributions):
    """The mean of several distributions over the same options."""
    if not distributions:
        raise ValueError("no distributions")
    return {o: sum(d[o] for d in distributions) / len(distributions) for o in distributions[0]}


@dataclass
class JevGate:
    """Frozen references (sorted) of the two signals and the threshold on their mean percentile, for JEV's naming
    over ``words`` ("gate": ``name_opts``, "approach": ``approach_opts``, SAM3's HM3D classes)."""
    entropy_reference: list
    contention_reference: list
    threshold: float
    words: str = "gate"

    @staticmethod
    def percentile(reference, value):
        # midranks, as _analyze.pct
        return (bisect_left(reference, value) + bisect_right(reference, value)) / (2 * len(reference))

    def score(self, h, c):
        return (self.percentile(self.entropy_reference, h) + self.percentile(self.contention_reference, c)) / 2

    def assess(self, probabilities, target):
        """Signals of one raw JEV naming distribution: {probabilities (folded), top1, entropy,
        contention, pct_entropy, pct_contention, score, threshold, insufficient, known_target}."""
        folded = fold(probabilities, target)
        h, c = entropy(folded), contention(folded, target, self.words)
        pe, pc = self.percentile(self.entropy_reference, h), self.percentile(self.contention_reference, c)
        score = (pe + pc) / 2
        return {"probabilities": folded, "top1": top1(folded), "entropy": h, "contention": c, "pct_entropy": pe,
                "pct_contention": pc, "score": score, "threshold": self.threshold,
                "insufficient": score >= self.threshold, "known_target": known(target)}

    def decide(self, probabilities, target, forced=False):
        """(status, assessment): "insufficient" when unsure (never when ``forced``), else
        "correct" iff the top-1 option is the target."""
        assessment = self.assess(probabilities, target)
        if assessment["insufficient"] and not forced:
            return "insufficient", assessment
        return ("correct" if assessment["top1"] == target else "wrong"), assessment

    @classmethod
    def load(cls, path):
        import json
        with open(path) as stream:
            frozen = json.load(stream)
        return cls(frozen["entropy_reference"], frozen["contention_reference"], frozen["threshold"],
                   frozen.get("words", "gate"))


def study_rows(root=None):
    """The 444 eye-labelled study requests with a JEV row, loaded as _wrong_only.main does
    (_analyze.load: label fixes applied, armchair folded): (offline rows, online rows)."""
    import importlib.util
    import sys
    from pathlib import Path
    root = Path(root or Path(__file__).resolve().parents[1])
    here, folder = os.getcwd(), str(root / RELIABILITY)
    os.chdir(root)  # _analyze reads paths relative to the repo root
    sys.path.insert(0, folder)
    try:
        spec = importlib.util.spec_from_file_location("_analyze", f"{folder}/_analyze.py")
        analyze = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(analyze)
        rows = [r for r in analyze.load() if r["set"] != "d1" and r["eye"] in ("goal", "not_goal", "cant_tell")
                and r["jev"]]
    finally:
        os.chdir(here)
        sys.path.remove(folder)
    return [r for r in rows if r["set"] == "offline"], [r for r in rows if r["set"] == "online"]


def gate_names(words, root=None):
    """{study row index: JEV naming distribution} over ``words``: "gate" from jevgpu.jsonl (via ``study_rows``),
    "approach" from jev_hm3d_names.jsonl (the same frames named over ``approach_opts`` through POST /name)."""
    import json
    from pathlib import Path
    root = Path(root or Path(__file__).resolve().parents[1])
    with open(root / RELIABILITY / "jev_hm3d_names.jsonl") as stream:
        return {d["index"]: d[words] for d in map(json.loads, stream)}


def export_jev_gate(path=None, words="approach"):
    """Freeze ``JEV_GATE``: references from the 131 offline rows and the threshold at the (1 - APPROACH_RATE)
    quantile (numpy, linear) of the score on all 444 study rows, for JEV's naming over ``words`` ("approach":
    HM3D words, the default since 2026-11-09; "gate": the original per-target words of jevgpu.jsonl).
    ``sam3d-env core python -m perception.jev export [path] [words]``."""
    import json
    from pathlib import Path

    import numpy as np
    root = Path(__file__).resolve().parents[1]
    path = Path(path or root / JEV_GATE)
    offline, online = study_rows(root)
    names = gate_names(words, root) if words == "approach" else {r["index"]: r["jev"]["name_full"]
                                                                 for r in offline + online}
    folded = [(fold(names[r["index"]], r["target"]), r["target"]) for r in offline + online]
    signals = [(entropy(d), contention(d, t, words)) for d, t in folded]
    n = len(offline)
    gate = JevGate(sorted(h for h, _ in signals[:n]), sorted(c for _, c in signals[:n]), None, words)
    scores = np.array([gate.score(h, c) for h, c in signals])
    gate.threshold = float(np.quantile(scores, 1 - APPROACH_RATE))
    with open(path, "w") as out:
        src = "jevgpu.jsonl" if words == "gate" else "jev_hm3d_names.jsonl (approach_opts via POST /name)"
        json.dump({"source": f"{RELIABILITY}/{src} through _analyze.load: {n} offline rows (references), "
                             f"{len(signals)} eye-labelled study rows (threshold)",
                   "rule": "insufficient iff mean(pct(H/log16), pct(min(p_t, max confuser p))) >= threshold; "
                           "pct = midrank in the offline references; p(armchair) folded into p(chair) for chair",
                   "template": TEMPLATE_VERSION, "state": STATE, "question": QUESTION,
                   "approach_rate": APPROACH_RATE, "flagged_fraction_study": float((scores >= gate.threshold).mean()),
                   "flagged_fraction_online": float((scores[n:] >= gate.threshold).mean()),
                   "n_reference": n, "n_study": len(signals), "threshold": gate.threshold,
                   "words": words,
                   "options": {t: (name_opts(t) if words == "gate" else approach_opts(t)) for t in CONFUSERS},
                   "confusers": CONFUSERS if words == "gate" else APPROACH_LOOKALIKES,
                   "fillers": FILLERS if words == "gate" else APPROACH_FILLERS,
                   "entropy_reference": gate.entropy_reference, "contention_reference": gate.contention_reference},
                  out, indent=1)
    return gate


class _CpuEmbedding:
    """The token embedding table kept on CPU: rows are looked up there and copied to the ids' device
    (an exact copy of the GPU lookup). A plain callable, so ``model.to`` never moves it back."""

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, ids):
        import torch
        return torch.nn.functional.embedding(ids.cpu(), self.weight).to(ids.device)


class JevJudge:
    """JEV-27B-VL from the pre-quantized ``JEV_DIR``: bitsandbytes 8-bit weights on the GPU, the LoRA
    unmerged as forward hooks (as _jev_run.py with INT8=1), then slimmed: ~4.7 GiB less GPU memory, same
    probabilities. The output head keeps only the decision head's verbalizer rows, with the head's LoRA
    restricted to the same rows (``choice`` reads only those logits; the log-softmax constant cancels in the
    softmax over the options), and the 248k x 5120 embedding table stays on CPU."""

    def __init__(self, path=JEV_DIR):
        import json
        from pathlib import Path

        import torch
        from safetensors.torch import load_file
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        path = Path(path)
        if not (path / "config.json").exists() or "quantization_config" not in json.load(open(path / "config.json")):
            raise FileNotFoundError(f"{path} is not a pre-quantized JEV copy; run: python -m perception.jev quantize")
        self.processor = AutoProcessor.from_pretrained(path)
        self.device = "cuda"
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16,
                                                                     device_map="cuda").eval()
        cfg = json.load(open(path / "adapter_vllm/adapter_config.json"))
        scale = cfg["lora_alpha"] / cfg["r"]
        weights = load_file(path / "adapter_vllm/adapter_model.safetensors")
        modules = dict(self.model.named_modules())
        self.lora_pairs, self.head_lora = 0, None
        for key in weights:
            if ".lora_A." not in key:
                continue
            name = key.split("base_model.model.", 1)[-1].split(".lora_A.")[0]
            candidates = [name, name.replace("model.", "model.language_model.", 1), "model." + name]
            module = next((modules[c] for c in candidates if c in modules), None)
            if module is None:
                raise KeyError(f"no module for {key}")
            a, b = weights[key].float(), weights[key.replace(".lora_A.", ".lora_B.")].float()
            a, b = a.to(self.device, torch.bfloat16), (scale * b).to(self.device, torch.bfloat16)
            if module is self.model.lm_head:
                self.head_lora = (a, b)  # _slim keeps the verbalizer rows of b
            module.register_forward_hook(
                lambda mod, inp, out, a=a, b=b: out + ((inp[0].to(a.dtype) @ a.T) @ b.T).to(out.dtype))
            self.lora_pairs += 1
        del weights, modules, module  # these still reference lm_head and embed_tokens, which _slim frees
        self.head = json.load(open(path / "adapter_vllm/decision_head.json"))
        self.temperature = json.load(open(path / "calibration.json"))["per_kind"]
        self._slim()
        print(f"JEV {path}: int8, LoRA hooks {self.lora_pairs} pairs, slim head {len(self.rows)} rows, "
              f"GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    def _slim(self):
        import torch
        head = self.model.lm_head
        if len(head._forward_hooks) != (self.head_lora is not None) or head._forward_pre_hooks:
            raise RuntimeError("lm_head has hooks other than its LoRA; cannot slim it")
        ids = sorted(set(self.head["verbalizer_ids"]))
        self.rows = {t: k for k, t in enumerate(ids)}
        self.head_weight = head.weight.detach()[ids].clone()
        if self.head_lora is not None:
            a, b = self.head_lora
            self.head_lora = (a, b[ids].clone())
        self.model.lm_head = torch.nn.Identity()  # the model now returns the final hidden state
        language = self.model.model.language_model
        weight = language.embed_tokens.weight.detach().cpu()
        del language.embed_tokens  # a registered submodule: remove it, then set the CPU lookup as a plain attribute
        language.embed_tokens = _CpuEmbedding(weight)
        del head, weight
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def choice(self, image, target, words="gate"):
        """JEV's naming distribution for the outlined frame ``image``: {probabilities (raw, per
        option, rounded to 5 decimals as offline), options, latency_s, n_tokens}; ``words`` as naming_text."""
        text, options = naming_text(target, words)
        return self._choose(image, text, options)

    def _choose(self, image, text, options):
        import torch
        start = time.perf_counter()
        x = self.processor(text=[text], images=[image], return_tensors="pt", add_special_tokens=False).to(self.device)
        with torch.inference_mode():
            last = self.model(**x, logits_to_keep=1).logits[0, -1]  # final hidden state (lm_head slimmed)
            kept = last @ self.head_weight.T  # raw logits of the verbalizer tokens (the log-softmax constant cancels)
            if self.head_lora is not None:  # the lm_head LoRA hook, on the kept rows only
                a, b = self.head_lora
                kept = kept + ((last.to(a.dtype) @ a.T) @ b.T).to(kept.dtype)
            kept = kept.float()
            value = lambda t: kept[self.rows[t]].item()
        s = self.head["slots"]["ranges"]["choice"][0]
        ids = self.head["verbalizer_ids"][s: s + len(options)]
        z = [(value(t) + self.head["bias"][s + i]) / self.temperature["choice"] for i, t in enumerate(ids)]
        e = [math.exp(v - max(z)) for v in z]
        return {"probabilities": {o: round(v / sum(e), 5) for o, v in zip(options, e)}, "options": options,
                "latency_s": time.perf_counter() - start, "n_tokens": int(x["input_ids"].shape[1])}


def reference_text(target, image, source=None):
    """(text, images) that _jev_run.decide builds for the naming question: its own ``decide`` and
    ``name_opts`` source (decorator dropped) run up to the processor call."""
    import ast
    from pathlib import Path
    source = Path(source or Path(__file__).resolve().parents[1] / RELIABILITY / "_jev_run.py")
    keep = []
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in ("CONFUSERS", "FILLERS", "IMG"):
            keep.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in ("decide", "name_opts"):
            node.decorator_list = []
            keep.append(node)

    class Captured(Exception):
        pass

    def proc(text, images, return_tensors, add_special_tokens):
        raise Captured(text[0], images)

    namespace = {"math": math, "proc": proc}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(source), "exec"), namespace)
    try:
        namespace["decide"]("choice", ["Robot camera image of a room; one object is outlined in red: ", image],
                            "What is the object outlined in red?", namespace["name_opts"](target))
    except Captured as captured:
        return captured.args
    raise RuntimeError("_jev_run.decide did not call the processor")


def dry_run(folder, target, path=JEV_DIR):
    """CPU, processor only (no weights): the input tensors of one real request, checked against
    the text and tensors _jev_run.py builds (``reference_text``)."""
    import json
    from pathlib import Path

    from PIL import Image
    from transformers import AutoProcessor

    from perception.qwen import outlined
    folder, path = Path(folder), Path(path)
    processor = AutoProcessor.from_pretrained(path)
    image = outlined(Image.open(folder / "input.jpg"), Image.open(folder / "mask.png"))
    text, options = naming_text(target)
    reference, reference_images = reference_text(target, image)
    x = processor(text=[text], images=[image], return_tensors="pt", add_special_tokens=False)
    y = processor(text=[reference], images=reference_images, return_tensors="pt", add_special_tokens=False)
    head = json.load(open(path / "adapter_vllm/decision_head.json"))
    s = head["slots"]["ranges"]["choice"][0]
    ids = head["verbalizer_ids"][s: s + len(options)]
    same = {k: x[k].shape == y[k].shape and bool((x[k] == y[k]).all()) for k in x}
    return {"folder": str(folder), "target": target, "options": options, "text_matches_reference": text == reference,
            "tensors_match_reference": same, "shapes": {k: list(v.shape) for k, v in x.items()},
            "n_tokens": int(x["input_ids"].shape[1]), "image_tokens": int((x["input_ids"] == processor.tokenizer
                                                                           .convert_tokens_to_ids("<|image_pad|>")).sum()),
            "verbalizer": [processor.tokenizer.decode([i]) for i in ids], "text": text}


def quantize(src=JEV_SOURCE, dst=JEV_DIR):
    """Once: load the bf16 release with bitsandbytes 8-bit (as JevJudge did from the NAS) and save the quantized
    model with the files JevJudge reads (processor, adapter_vllm/, calibration.json) to ``dst``."""
    import shutil
    from pathlib import Path

    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
    src, dst = Path(src), Path(dst)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        src, dtype=torch.bfloat16, device_map="cuda", quantization_config=BitsAndBytesConfig(load_in_8bit=True))
    model.save_pretrained(dst)
    AutoProcessor.from_pretrained(src).save_pretrained(dst)
    shutil.copytree(src / "adapter_vllm", dst / "adapter_vllm", dirs_exist_ok=True)
    for name in ("calibration.json", "LICENSE"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    print(f"quantized {src} -> {dst}", flush=True)


if __name__ == "__main__":
    import json
    import sys
    if sys.argv[1:2] == ["quantize"] and len(sys.argv) in (2, 4):
        quantize(*sys.argv[2:4])
    elif sys.argv[1:2] == ["export"] and len(sys.argv) <= 4:
        frozen = export_jev_gate(*sys.argv[2:4])
        print(f"froze {sys.argv[2] if len(sys.argv) > 2 else JEV_GATE}: words {frozen.words}, "
              f"threshold {frozen.threshold}")
    elif sys.argv[1:2] == ["dry-run"] and len(sys.argv) == 4:
        print(json.dumps(dry_run(*sys.argv[2:4]), indent=1))
    else:
        sys.exit(__doc__)
