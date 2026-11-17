"""Three-way visual verification and a frozen, per-request uncertainty gate.

Qwen sees only images and the target, never detector scores or evaluation labels.
Reconstruction views are hypotheses from one observation, not independent evidence.
This module does not overload the legacy binary Verdict/HTTP protocol.
"""
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import math
import time


STATUSES = ("correct", "wrong", "insufficient")
PROMPT_VERSION = "three-way-v1"


def features(margin, area, view_flip=None):
    values = [-abs(float(margin)), -float(area)]
    if view_flip is not None:
        values.append(float(view_flip))
    if not all(math.isfinite(x) for x in values):
        raise ValueError("uncertainty features must be finite")
    if not 0 <= float(area) <= 1 or (view_flip is not None and not 0 <= float(view_flip) <= 1):
        raise ValueError("area and flip rate must be fractions")
    return values


@dataclass
class UncertaintyGate:
    references: list
    thresholds: dict

    @staticmethod
    def percentile(reference, value):
        # Midranks give identical scores to ties, independent of request order.
        return (bisect_left(reference, value) + bisect_right(reference, value)) / (2 * len(reference))

    @classmethod
    def fit(cls, rows, review_fraction=1 / 3):
        if not 0 < review_fraction <= 1:
            raise ValueError("review_fraction must be in (0, 1]")
        vals = [features(r["margin"], r["area"],
                         None if r.get("view_flip") in (None, "") else r["view_flip"])
                for r in rows]
        refs = [sorted(v[j] for v in vals if len(v) > j) for j in range(3)]
        if not refs[0] or not refs[2]:
            raise ValueError("calibration requires margin, area, and some observed flip rates")
        gate = cls(refs, {})
        for n in (2, 3):
            scores = sorted(gate.score_values(v[:n]) for v in vals if len(v) >= n)
            gate.thresholds[str(n)] = scores[min(len(scores) - 1, int((1 - review_fraction) * len(scores)))]
        return gate

    @classmethod
    def load(cls, path):
        """The gate frozen by the Qwen verification pilot (frozen_gate.json)."""
        import json
        frozen = json.loads(open(path).read())
        return cls(frozen["references"], frozen["thresholds"])

    def score_values(self, values):
        return sum(self.percentile(ref, val) for ref, val in zip(self.references, values)) / len(values)

    def assess(self, margin, area, view_flip=None):
        try:
            values = features(margin, area, view_flip)
        except (TypeError, ValueError):
            return {"review": True, "score": None, "reason": "invalid_features"}
        score = self.score_values(values)
        return {"review": score >= self.thresholds[str(len(values))], "score": score,
                "reason": "three_signals" if len(values) == 3 else "missing_flip_two_signal_gate"}


D1_GATE = "outputs/apexnav_offline/qwen_all_requests_2026-11-03/d1_gate.json"


@dataclass
class SufficiencyGate:
    """Step 1 of the D1 design (selector ``d1``): is the raw frame enough for SAM3 to decide?

    score = mean of the percentiles of -|margin| and -area in frozen reference
    distributions (the 131 offline eye-labelled requests); insufficient iff
    score >= threshold. Reproduces ``fit_thresholds``/``score`` of
    outputs/apexnav_offline/qwen_all_requests_2026-11-03/_compare.py exactly
    (percentile = fraction of references <= value, unlike ``UncertaintyGate``'s midranks).
    """
    neg_abs_margin: list  # sorted references
    neg_area: list
    threshold: float

    @staticmethod
    def percentile(reference, value):
        return bisect_right(reference, value) / len(reference)

    def score(self, margin, area):
        return (self.percentile(self.neg_abs_margin, -abs(float(margin)))
                + self.percentile(self.neg_area, -float(area))) / 2

    def assess(self, margin, area):
        """{score, threshold, margin, area, insufficient}; non-finite features are insufficient."""
        try:
            score = self.score(margin, area) if math.isfinite(margin) and math.isfinite(area) else None
        except (TypeError, ValueError):
            score = None
        return {"score": score, "threshold": self.threshold, "margin": margin, "area": area,
                "insufficient": score is None or score >= self.threshold}

    @classmethod
    def fit(cls, rows):
        """Max Youden J threshold for rows [{margin, area, insufficient}] (``_compare.fit_thresholds``)."""
        import numpy as np
        gate = cls(sorted(-abs(float(r["margin"])) for r in rows), sorted(-float(r["area"]) for r in rows), None)
        y = np.array([bool(r["insufficient"]) for r in rows])
        s = np.array([gate.score(r["margin"], r["area"]) for r in rows])
        cand = np.unique(s)
        js = [((s >= t) & y).sum() / y.sum() - ((s >= t) & ~y).sum() / (~y).sum() for t in cand]
        gate.threshold = float(cand[int(np.argmax(js))])
        return gate, float(max(js))

    @classmethod
    def load(cls, path):
        import json
        with open(path) as stream:
            frozen = json.load(stream)
        return cls(frozen["neg_abs_margin"], frozen["neg_area"], frozen["threshold"])

    def save(self, path, **info):
        import json
        with open(path, "w") as out:
            json.dump({**info, "threshold": self.threshold, "neg_abs_margin": self.neg_abs_margin,
                       "neg_area": self.neg_area}, out, indent=1)


def export_d1_gate(path=None):
    """Fit the D1 gate on the 131 offline eye-labelled requests (``_compare.offline_reference``)
    and freeze it to ``D1_GATE``: ``sam3d-env core python -m perception.qwen``."""
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    path = Path(path or root / D1_GATE)
    spec = importlib.util.spec_from_file_location("_compare", path.parent / "_compare.py")
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    rows = compare.offline_reference()
    gate, j = SufficiencyGate.fit(rows)
    gate.save(path, source="_compare.offline_reference (131 offline eye-labelled requests)", n=len(rows),
              youden_j=j, rule="insufficient iff mean(pct(-|margin|), pct(-area)) >= threshold; "
              "pct = fraction of references <= value")
    return gate


@dataclass(frozen=True)
class Decision:
    status: str
    source: str

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"invalid verdict: {self.status}")

    @property
    def accepted(self):
        return {"correct": True, "wrong": False, "insufficient": None}[self.status]

    def navigation_action(self, checks=1, max_checks=3):
        """Budget exhaustion abandons this attempt without calling the object wrong.

        The caller must acquire a new real observation before another check. The
        same insufficient view must not be retried, accepted at STOP, or blacklisted.
        """
        if self.status == "correct":
            return "allow_target"
        if self.status == "wrong":
            return "reject_target"
        return "explore_unresolved" if checks >= max_checks else "acquire_new_view"


def verify_if_needed(gate, margin, area, view_flip, raw_accept, judge):
    """Lazy callback: no Qwen invocation for requests outside the frozen gate."""
    if not isinstance(raw_accept, bool):
        raise ValueError("raw_accept must be a boolean")
    assessment = gate.assess(margin, area, view_flip)
    if not assessment["review"]:
        return Decision("correct" if raw_accept else "wrong", "sam3"), assessment
    result = judge()
    return Decision(result["status"], "qwen"), assessment


def prompt(target, with_grid):
    text = (f"Image 1 is a robot camera observation. Judge ONLY the object outlined in red. "
            f"The requested category is {target!r}. Determine whether the visible evidence is enough "
            "to accept or reject that category for this object. Ignore other objects in the room. ")
    if with_grid:
        text += ("Image 2 shows a reconstruction of the outlined object from 24 viewpoints, "
                 "rows elevation 0/30/60 degrees, columns azimuth 0..315 degrees. ALL renders "
                 "come from Image 1 alone. Hidden parts are guessed and can be wrong. Use the grid "
                 "to interpret visible shape, but invented details cannot establish identity. "
                 "Agreement among renders is not independent confirmation. A bad reconstruction "
                 "does not by itself make the real object wrong. ")
    return text + ANSWER_ABC


# The couch/armchair sentence and the three-way answer block, shared by ``prompt`` and
# ``prompt_views`` (unchanged since three-way-v1).
ANSWER_ABC = ("A couch is upholstered seating for two or more people; an armchair is a chair. "
              "Choose exactly one answer:\n"
              "A: correct — visible evidence is sufficient that the outlined object is the target.\n"
              "B: wrong — visible evidence is sufficient that it is not the target.\n"
              "C: insufficient — the object is too small, occluded, cropped, ambiguous, or the "
              "evidence does not distinguish the target from plausible alternatives.\n"
              "Lack of evidence for the target is not evidence that it is wrong. "
              "Answer with only A, B, or C.")
# ``forced2``: a forced choice between A and B (no "insufficient"), for forced decisions.
ANSWER_AB = ("A couch is upholstered seating for two or more people; an armchair is a chair. "
             "Choose exactly one answer:\n"
             "A: correct — the outlined object is the target.\n"
             "B: wrong — the outlined object is not the target.\n"
             "If the evidence is weak, choose the more likely answer. "
             "Answer with only A or B.")
PROMPT_VERSION_VIEWS = "multiview-v1"
PROMPT_VERSION_VIEWS_GRID = "multiview-grid-v1"
# crop_small: views whose mask covers less than this fraction of the frame get an extra crop
CROP_SMALL_AREA = 0.02


def prompt_views(target, n_views, with_grid=False, grid_source=1, crops=False, forced2=False):
    """The instruction after ``n_views`` interleaved camera images (Image 1 the closest), optionally
    followed by enlarged crops (``crops``) and the SAM3D grid of camera image ``grid_source``
    (``with_grid``, the last image). Versions: ``PROMPT_VERSION_VIEWS`` / ``_GRID``."""
    text = (f"Images 1 to {n_views} are robot camera observations of the SAME physical object, outlined in "
            "red in every image, taken from different distances and directions while the robot approached "
            "it (Image 1 is the closest). The outline comes from an automatic detector: it may cover only "
            "part of the object or spill onto neighbours; judge the single physical object the outlines "
            f"mark consistently. The requested category is {target!r}. Combine the evidence from all "
            "images: a clear, whole, close view outweighs small, distant or cut-off views. Ignore other "
            "objects in the room. ")
    if crops:
        text += ("A small, distant camera image is followed by an enlarged crop of it around the outline; "
                 "the crop shows the same pixels, not a new observation. ")
    if with_grid:
        text += ("The final image is NOT a camera observation: it is an automatic 3D reconstruction of the "
                 f"outlined object, made from camera Image {grid_source} alone and rendered from 24 viewpoints "
                 "(rows elevation 0/30/60 degrees, columns azimuth 0..315 degrees). It may be inaccurate or "
                 "misleading from some angles: hidden sides such as the back are invented, and the shape can "
                 "be distorted and the texture wrong. Use it only as supporting evidence for whether the "
                 "outlined object is the target; where it conflicts with the camera images, the camera "
                 "images take precedence. ")
    return text + (ANSWER_AB if forced2 else ANSWER_ABC)


def views_content(images, target, distances=None, grid=None, crops=None, forced2=False):
    """The interleaved chat content of a multi-view request: ``"Image k (about X.X m away):"``
    then camera image k (nearest first), each followed by its crop if ``crops[k]`` is not None,
    then the grid (if any) and ``prompt_views``. Returns (content, prompt version)."""
    if not images:
        raise ValueError("no views")
    crops = list(crops) if crops is not None else [None] * len(images)
    distances = list(distances) if distances is not None else [None] * len(images)
    if len(crops) != len(images) or len(distances) != len(images):
        raise ValueError("crops and distances must match images")
    content = []
    for k, (image, distance, crop) in enumerate(zip(images, distances, crops), start=1):
        away = "" if distance is None else f" (about {float(distance):.1f} m away)"
        content += [{"type": "text", "text": f"Image {k}{away}:"}, {"type": "image", "image": image}]
        if crop is not None:
            content += [{"type": "text", "text": f"Enlarged crop of Image {k}:"}, {"type": "image", "image": crop}]
    if grid is not None:
        content += [{"type": "text", "text": "3D reconstruction (not a camera image):"},
                    {"type": "image", "image": grid}]
    content.append({"type": "text", "text": prompt_views(
        target, len(images), with_grid=grid is not None, crops=any(c is not None for c in crops),
        forced2=forced2)})
    return content, PROMPT_VERSION_VIEWS_GRID if grid is not None else PROMPT_VERSION_VIEWS


def crop_around(image, mask, pad=1.0, min_px=160, max_px=448):
    """A square crop of ``image`` around the mask's bounding box, padded by ``pad`` times the
    object's size on each side, at least ``min_px`` wide, resized down to at most ``max_px``."""
    from PIL import Image
    mask = mask.convert("L").resize(image.size, Image.Resampling.NEAREST).point(lambda p: 255 if p > 0 else 0)
    box = mask.getbbox()
    if box is None:
        raise ValueError("empty object mask")
    x0, y0, x1, y1 = box
    side = max(min_px, int(round(max(x1 - x0, y1 - y0) * (1 + 2 * pad))))
    side = min(side, max(image.size))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = int(round(min(max(cx - side / 2, 0), max(image.width - side, 0))))
    top = int(round(min(max(cy - side / 2, 0), max(image.height - side, 0))))
    crop = image.crop((left, top, min(left + side, image.width), min(top + side, image.height)))
    if max(crop.size) > max_px:
        crop.thumbnail((max_px, max_px))
    elif max(crop.size) < max_px:  # enlarge small crops, so the object is bigger than in the frame
        scale = max_px / max(crop.size)
        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.BICUBIC)
    return crop


def outlined(image, mask):
    import numpy as np
    from PIL import Image, ImageFilter
    image = image.convert("RGB")
    mask = mask.convert("L").resize(image.size, Image.Resampling.NEAREST).point(lambda p: 255 if p > 0 else 0)
    if mask.getbbox() is None:
        raise ValueError("empty object mask")
    edge = (np.asarray(mask.filter(ImageFilter.MaxFilter(7))) > 0) & (np.asarray(mask) == 0)
    pixels = np.asarray(image).copy()
    pixels[edge] = (255, 0, 0)
    return Image.fromarray(pixels)


def render_grid(views, tile=224):
    """views is {(azimuth, elevation): PIL image}; require the complete grid."""
    from PIL import Image, ImageDraw
    grid = Image.new("RGB", (tile * 8, tile * 3), "white")
    draw = ImageDraw.Draw(grid)
    for row, el in enumerate((0, 30, 60)):
        for col, az in enumerate(range(0, 360, 45)):
            view = views[(az, el)].convert("RGB").copy()
            view.thumbnail((tile, tile))
            grid.paste(view, (col * tile + (tile - view.width) // 2, row * tile + (tile - view.height) // 2))
            draw.text((col * tile + 4, row * tile + 2), f"az {az} el {el}", fill=(90, 90, 90))
    return grid


class QwenJudge:
    """``int8``: bitsandbytes 8-bit weights (~half the memory), so Qwen fits on the GPU
    next to SAM3, SAM3D and ApexNav in the closed loop."""

    def __init__(self, model="Qwen/Qwen3-VL-8B-Instruct", int8=False):
        import torch
        self.int8 = int8
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model, local_files_only=True)
        quantization = BitsAndBytesConfig(load_in_8bit=True) if int8 else None
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model, dtype=torch.bfloat16, device_map="cuda", local_files_only=True,
            quantization_config=quantization).eval()
        self.ids = [self.processor.tokenizer.encode(x, add_special_tokens=False) for x in "ABC"]
        if any(len(x) != 1 for x in self.ids) or len({x[0] for x in self.ids}) != 3:
            raise ValueError("answer letters must be distinct single tokens")
        self.ids = [x[0] for x in self.ids]

    def offload(self):
        """Weights to CPU and the GPU cache freed (selector d1 --stagger); ``onload`` moves them
        back (~7 s / ~2 s for bf16, answers unchanged). bitsandbytes 8-bit weights stay on the
        GPU when moved, so staggering needs bf16."""
        import torch
        if self.int8:
            raise RuntimeError("8-bit Qwen cannot be offloaded; stagger with bf16 weights")
        self.model.to("cpu")
        torch.cuda.empty_cache()

    def onload(self):
        self.model.to("cuda")

    def judge(self, image, target, grid=None):
        images = [image] + ([] if grid is None else [grid])
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": prompt(target, grid is not None)})
        return self._answer(content)

    def judge_views(self, images, target, distances=None, grid=None, crops=None, forced2=False,
                    forced2_min=0.5):
        """Three-way verdict on one object from several camera images (nearest first; outlined by
        the caller), optionally with enlarged crops and the SAM3D grid (``views_content``). The
        same dict as ``judge`` plus n_images and prompt_version. ``forced2``: A/B only, "correct"
        iff p(A) / (p(A) + p(B)) >= ``forced2_min``."""
        content, version = views_content(images, target, distances, grid, crops, forced2)
        answer = self._answer(content, forced2=forced2, forced2_min=forced2_min)
        answer.update(n_images=sum(c["type"] == "image" for c in content), prompt_version=version,
                      forced2=bool(forced2))
        return answer

    def _answer(self, content, forced2=False, forced2_min=0.5):
        """The next-token A/B/C logits for one user turn: {status, probabilities, invalid_answer,
        top_token, latency_s, n_tokens}."""
        import torch
        torch.cuda.synchronize()
        start = time.perf_counter()
        inputs = self.processor.apply_chat_template([{"role": "user", "content": content}],
                    tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            logits = self.model(**inputs, logits_to_keep=1).logits[0, -1].float()
        probabilities = logits[self.ids].softmax(0).tolist()
        top = int(logits.argmax())
        # A nonconforming unconstrained answer is a protocol failure, never acceptance.
        invalid = top not in self.ids
        status = "insufficient" if invalid else STATUSES[self.ids.index(top)]
        extra = {}
        if forced2:  # A versus B only; C is not offered and not a valid answer
            pair = logits[self.ids[:2]].softmax(0).tolist()
            invalid = top not in self.ids[:2]
            status = "insufficient" if invalid else ("correct" if pair[0] >= forced2_min else "wrong")
            extra = {"forced2_p_correct": pair[0], "forced2_min": forced2_min}
        torch.cuda.synchronize()
        return {"status": status, "probabilities": dict(zip(STATUSES, probabilities)),
                "invalid_answer": invalid, "top_token": self.processor.tokenizer.decode(top),
                "latency_s": time.perf_counter() - start, "n_tokens": inputs["input_ids"].shape[1], **extra}

if __name__ == "__main__":
    frozen = export_d1_gate()
    print(f"froze {D1_GATE}: threshold {frozen.threshold}")
