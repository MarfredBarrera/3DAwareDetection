"""``Verifier``: check an object detection by classifying a better SAM3D view of it.

    from perception import Verifier

    with Verifier(vocabulary=["chair", "couch", "bed", "toilet", "tv", "table"]) as verifier:
        verdict = verifier.verify(rgb, mask, target="chair", input_elevation=20.0)
        if verdict.accepted:
            ...  # commit to the detection

Per call: SAM3D reconstructs the masked object; the ensemble filter (optional)
measures stage-1 seed disagreement; a selector (``vqf`` or ``top2gap``) picks
one candidate view; SAM3 scores the vocabulary on it; the detection is
accepted when the top-1 category is the target. Selector ``input`` skips SAM3D
and judges the raw frame (no SAM3D worker is started); so does every flat target
(``select.flat``, e.g. tv), whatever the selector. The models live in two
persistent worker processes (see ``perception.worker``), loaded once.

Selector ``qwen_grid`` returns a three-way ``status``: SAM3 judges the raw frame,
SAM3D renders the full candidate grid and SAM3 classifies every view; a frozen
uncertainty gate (``qwen_gate``: raw margin, mask area, view flip rate) passes
confident raw decisions through and sends the rest to Qwen3-VL (a third worker),
which answers correct, wrong or insufficient from the outlined frame and the grid.

Selector ``d1`` (sufficiency first): SAM3 judges the raw frame; a frozen
``qwen.SufficiencyGate`` (raw margin and mask area only) decides whether that is
enough. Sufficient: SAM3's raw top-1 decides (correct / wrong; source "sam3").
Insufficient: Qwen3-VL answers correct, wrong or insufficient (source "qwen") from the
outlined frame plus the SAM3D grid (``d1_evidence="grid"``) or the frame alone
(``"input"``, SAM3D never starts). Its workers start with the Verifier; with
``lazy_workers``, SAM3D and Qwen start on the first insufficient request instead.

Close-range multi-view decision (selector ``d1``, ``Verifier.verify_views``; the ApexNav gate's
view bank): Qwen3-VL answers correct, wrong or insufficient from up to ``MULTIVIEW_MAX`` outlined
camera views of one object (nearest first), with the SAM3D grid of one of them (``grid``; on a
reconstruct failure, images only). SAM3 scores each view when Qwen is not decisive (``sam3``
"fallback"), always, or never; ``select.aggregate_views`` pools them for the caller's fallback.

Selector ``jev_sam3`` (JEV judges the uncertainty, SAM3 decides; a JEV-27B-VL int8 worker in the qwen
venv + the SAM3 worker): JEV names the object outlined by the request's mask over
``jev.name_opts(target)``; the frozen ``jev.JevGate`` (naming entropy and look-alike contention) says
"insufficient" (approach; source "jev") when unsure, else SAM3's raw label decides (correct iff it is
the target; source "sam3"). ``verify_views``: JEV's gate over the views' mean distribution; unforced
(views on the way) and unsure -> "insufficient" (keep approaching); otherwise (forced: arrival or a
failed approach, or JEV sure) SAM3 scores every view and ``select.fuse_views`` decides: correct iff the
target has the highest fused (evidential, Dempster-Shafer) belief over the views (2026-11-07; it replaced
the d1mv margin-and-vote rule, which ``aggregate`` still reports). ``/name`` with words "approach" names over
HM3D words (``jev.approach_opts``) for the approach belief; the gate keeps ``jev.name_opts``.

Workers start with ``launcher``: by default ``sam3d-env <env>`` inside the
container, or ``docker exec`` into ``$PERCEPTION_CONTAINER`` from the host.
"""
import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

from perception import select
from perception.images import decode, encode
from viewactive.grid import sam3d_to_viewactive

WORKSPACE = Path(__file__).resolve().parents[1]
CONTAINER = os.environ.get("PERCEPTION_CONTAINER", "zsonverification-dev-1")
VIEWACTIVE = "/data/checkpoints/viewactive/blender_full_el0_45.pt"
QWEN_ENV = "/data/envs/qwen"  # a venv in the container, not a sam3d-env env
QWEN_GATE = "outputs/apexnav_offline/qwen_verification_pilot/frozen_gate.json"
D1_EVIDENCE = ("grid", "input")
MULTIVIEW_MAX = 8                 # views per verify_views request
MULTIVIEW_SAM3 = ("fallback", "always", "never")
# ApexNav's six HM3D goals plus the object classes of mpcat40 (structure and
# catch-all classes dropped; "plant" duplicates "potted plant").
HOUSEHOLD = ("chair", "couch", "bed", "toilet", "tv", "potted plant", "door", "table", "picture",
             "cabinet", "cushion", "window", "curtain", "chest of drawers", "sink", "stool", "towel",
             "mirror", "shower", "bathtub", "counter", "fireplace", "lamp", "shelf", "blinds",
             "exercise equipment", "appliance", "clothes")


@dataclass
class View:
    """One SAM3D candidate, angles relative to the input camera ((180, 0) = input view)."""
    azimuth: float
    elevation: float
    absolute_elevation: float = None  # elevation in the gravity frame (selectors drop views from below, < 0)
    quality: float = None        # ViewActive VQF quality (vqf selector)
    scores: dict = None          # SAM3 score per category, when this view was classified
    image: object = None         # PIL image, when rendered


@dataclass
class Verdict:
    accepted: bool               # True: keep the detection (top-1 is the target, or unverified)
    label: str                   # top-1 category on the judged view; None if nothing was judged
    scores: dict                 # SAM3 score per category on the judged view
    judged: str                  # "view" (SAM3D candidate), "input" (raw frame) or "none" (abstained)
    view: View = None            # the chosen candidate, when judged == "view"
    uncertainty: float = None    # ensemble disagreement in [0, 1]; None without the filter
    trusted: bool = True         # passed the ensemble filter
    views: list = field(default_factory=list)  # every candidate considered
    timing: dict = None
    input_scores: dict = None    # SAM3 scores on the raw frame, when ``judge_input`` also judged it
    status: str = None           # qwen_grid, d1: "correct", "wrong" or "insufficient" (accepted iff correct)
    source: str = None           # qwen_grid, d1: who decided the status, "sam3" or "qwen"
    gate: dict = None            # qwen_grid, d1: uncertainty features and the gate's assessment
    qwen: dict = None            # qwen_grid, d1: Qwen's answer, when the gate sent the detection to it
    grid: object = None          # qwen_grid, d1: the PIL grid image Qwen saw
    jev: dict = None             # jev: JEV's raw naming distribution, options, latency (gate: its signals)

    def summary(self):
        """JSON-safe dict of the verdict; views without their images."""
        def view(v):
            return {"azimuth": v.azimuth, "elevation": v.elevation, "absolute_elevation": v.absolute_elevation,
                    "quality": v.quality, "scores": v.scores}

        return {"accepted": self.accepted, "label": self.label, "scores": dict(self.scores),
                "judged": self.judged, "view": None if self.view is None else view(self.view),
                "uncertainty": self.uncertainty, "trusted": self.trusted,
                "views": [view(v) for v in self.views], "timing": self.timing, "input_scores": self.input_scores,
                "status": self.status, "source": self.source, "gate": self.gate, "qwen": self.qwen,
                "jev": self.jev}


@dataclass
class ViewsVerdict:
    """``Verifier.verify_views``: one three-way verdict over several views of one object."""
    status: str                  # "correct", "wrong" or "insufficient" (accepted iff correct)
    source: str                  # "qwen", or "sam3" (sam3_shortcut on a sufficient view)
    qwen: dict = None            # Qwen's answer; error/reconstruct_error when a stage failed
    per_view: list = field(default_factory=list)  # [{step, index, area, clipped, scores, margin, label, cached}]
    aggregate: dict = None       # select.aggregate_views over the scored views (None: none scored)
    fused: dict = None           # jev_sam3: select.fuse_views over the scored views (the decision)
    accepted: bool = False
    grid: object = None          # the PIL grid image Qwen saw, if any
    timing: dict = None          # SAM3D reconstruct timing
    grid_scores: dict = None     # optional diagnostic: SAM3 top-1 per grid render, not a decision input
    jev: dict = None             # jev: the gate's assessment of the views' mean distribution, forced, n_used

    def summary(self):
        return {"status": self.status, "source": self.source, "accepted": self.accepted, "qwen": self.qwen,
                "per_view": self.per_view, "aggregate": self.aggregate, "fused": self.fused, "timing": self.timing,
                "grid_scores": self.grid_scores, "n_views": len(self.per_view), "jev": self.jev}


def _grid_image(views):
    """The 24-view grid (``qwen.render_grid``) as Qwen saw it offline: through JPEG quality 90."""
    import io

    from PIL import Image

    from perception.qwen import render_grid

    buffer = io.BytesIO()
    render_grid({(int(v.azimuth), int(v.elevation)): v.image for v in views}).save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def default_launcher():
    if shutil.which("sam3d-env"):
        return ["sam3d-env", "{env}"]
    return ["docker", "exec", "-i", "-w", "/workspace", CONTAINER, "sam3d-env", "{env}"]


class _Worker:
    def __init__(self, env, args, launcher, log):
        if env == "qwen":  # its own venv: replace "sam3d-env {env}" by its PATH
            at = launcher.index("sam3d-env")
            launcher = launcher[:at] + ["env", "PYTHONPATH=/workspace", f"PATH={QWEN_ENV}/bin:/usr/bin:/bin"] \
                + launcher[at + 2:]
        command = [part.format(env=env) for part in launcher] + ["python", "-m", "perception.worker", *args]
        self.log = open(log, "a") if log else None
        self.process = subprocess.Popen(command, cwd=WORKSPACE, text=True, bufsize=1,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log)
        self.lock = threading.Lock()
        self.name = " ".join(command)

    def wait_ready(self):
        self._read()

    def _read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"perception worker exited ({self.process.poll()}): {self.name}")
        reply = json.loads(line)
        if not reply["ok"]:
            raise RuntimeError(reply["error"])
        return reply["result"]

    def call(self, op, **args):
        with self.lock:
            self.process.stdin.write(json.dumps({"op": op, "args": args}) + "\n")
            return self._read()

    def alive(self):
        return self.process.poll() is None

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.log:
            self.log.close()


class Verifier:
    """Verify detections through SAM3D views; see the module docstring.

    selector: "vqf" (learned ViewActive viewpoint quality; one rendered view),
        "top2gap" (SAM3's largest top-1 minus top-2 gap over all candidates) or
        "input" (no SAM3D: SAM3 on the masked detection in the raw frame; ensemble ignored) or
        "naive" (no SAM3D: the target prompt alone, accepted iff its score >= select.NAIVE_THRESHOLD) or
        "qwen_grid", "d1" or "jev_sam3" (three-way; see the module docstring; ensemble and
        judge_input ignored).
    jev_gate: the frozen JEV gate (``jev.JEV_GATE``); jev_model: the JEV folder inside the worker's
        container (``jev.JEV_DIR``: $JEV_DIR or /models/JEV-27B-VL).
    qwen_gate: the frozen gate, ``qwen_grid``: frozen_gate.json (``QWEN_GATE``), ``d1``:
        d1_gate.json (``qwen.D1_GATE``); qwen_int8: load Qwen with 8-bit weights (~10 GB
        instead of ~17.5 GB; default on for ``d1`` unless ``stagger``). d1_evidence: "grid" or "input" (``d1``). d1_accept_min: Qwen "correct" counts only with p(correct) >= this, else
        "insufficient" (``d1``; None: Qwen's top answer).
    stagger: ``d1`` keeps SAM3D and Qwen (bf16) on CPU between calls, so only one of them holds
        GPU memory at a time (one shared GPU; a few seconds per insufficient request).
    lazy_workers: ``d1`` starts SAM3D/Qwen on the first insufficient request, not at construction.
    multiview_grid: ``d1`` also starts SAM3D for ``verify_views(grid=True)`` (with ``d1_evidence``
        "input" it would otherwise start on the first such request, inside its timeout).
    ensemble: run the ensemble filter (``ensemble_seeds`` stage-1 seeds); a
        reconstruction is trusted when its disagreement <= ``ensemble_threshold``
        (0.10-0.20 keeps ~99% of useful reconstructions on 1,800 LVIS objects).
    on_uncertain: when not trusted, "input" classifies the raw frame, "reject"
        rejects the detection, "accept" keeps it unverified.
    vocabulary: the closed set of categories SAM3 ranks; the target is added and its
        parts (``select.PARTS``, e.g. "cushion" for "couch") and other names of it
        (``select.same_object``, e.g. "cushion" for "pillow") are left out.
    judge_input: the raw frame must also show the target: SAM3 judges it first and a
        rejection there ends the call (no SAM3D); otherwise the view decides.
    view_when: with ``judge_input``, "always" sends every raw accept to a SAM3D view;
        "unsure" only raw accepts that ``select.unsure`` flags (margin < ``unsure_margin``
        or a confusable runner-up); a sure raw accept is returned as judged "input".
    all_views: render and classify every candidate, also of distrusted reconstructions
        (for evaluation and figures; slower).
    """

    def __init__(self, selector="vqf", ensemble=True, ensemble_threshold=0.15, ensemble_seeds=4,
                 on_uncertain="input", vocabulary=HOUSEHOLD, gpu=0, all_views=False, judge_input=False,
                 view_when="always", unsure_margin=select.UNSURE_MARGIN,
                 azimuths=(0, 45, 90, 135, 180, 225, 270, 315), elevations=(0, 30, 60),
                 distance=2.0, resolution=320, seed=42, elevation_range=select.VIEW_ELEVATIONS,
                 input_iou=0.3, sam3="/data/checkpoints/sam3/sam3.pt",
                 sam3d="/data/checkpoints/sam3d/hf", viewactive=VIEWACTIVE, qwen_gate=None,
                 qwen_model="Qwen/Qwen3-VL-8B-Instruct", qwen_int8=None, d1_evidence="grid", d1_accept_min=None, lazy_workers=False, stagger=False,
                 multiview_grid=False, jev_gate=None, jev_model=None, launcher=None, log_dir=None):
        if selector not in select.SELECTORS:
            raise ValueError(f"selector must be one of {select.SELECTORS}")
        if on_uncertain not in select.ON_UNCERTAIN:
            raise ValueError(f"on_uncertain must be one of {select.ON_UNCERTAIN}")
        if view_when not in ("always", "unsure"):
            raise ValueError('view_when must be "always" or "unsure"')
        if d1_evidence not in D1_EVIDENCE:
            raise ValueError(f"d1_evidence must be one of {D1_EVIDENCE}")
        if d1_accept_min is not None and not 0.0 <= d1_accept_min <= 1.0:
            raise ValueError("d1_accept_min must be in [0, 1] or None")
        if selector == "jev_sam3" and all_views:
            raise ValueError("selector jev_sam3 has no SAM3D views (all_views)")
        if ensemble and ensemble_seeds < 2:
            raise ValueError("the ensemble filter needs ensemble_seeds >= 2")
        self.selector, self.on_uncertain, self.all_views = selector, on_uncertain, all_views
        self.judge_input, self.view_when, self.unsure_margin = judge_input, view_when, unsure_margin
        self.threshold = ensemble_threshold if ensemble else None
        self.seeds = ensemble_seeds if ensemble else 1
        self.vocabulary = list(dict.fromkeys(vocabulary))
        self.grid = {"azimuths": [float(a) for a in azimuths], "elevations": [float(e) for e in elevations],
                     "distance": distance, "resolution": resolution, "seed": seed}
        self.elevation_range, self.input_iou = tuple(elevation_range), input_iou
        launcher = launcher or default_launcher()
        log = (lambda name: Path(log_dir) / f"{name}.log") if log_dir else (lambda name: None)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        common = ["--gpu", str(gpu)]
        uses_vqf = selector == "vqf" or all_views
        self.sam3 = self.sam3d = self.qwen = self.qwen_gate = self.jev = self.jev_gate = None
        self.d1_evidence, self.d1_accept_min = d1_evidence, d1_accept_min
        qwen_int8 = selector == "d1" and not stagger if qwen_int8 is None else qwen_int8
        if stagger and (selector != "d1" or qwen_int8):
            raise ValueError("stagger needs selector d1 and bf16 Qwen (8-bit weights cannot be offloaded)")
        offload = ["--offload"] if stagger else []
        # Worker factories; selector d1 starts sam3d and qwen lazily (``_start``), with the
        # same worker class as sam3 (bound now, not looked up at the first insufficient request).
        make_worker = _Worker
        self._spawn = {
            "sam3d": lambda: make_worker("sam3d", ["--role", "sam3d", "--sam3d", sam3d, *common]
                                         + (["--viewactive", str(viewactive)] if uses_vqf else [])
                                         + (["--gaussian-only"] if selector == "d1" else []) + offload, launcher,
                                         log("sam3d")),
            "qwen": lambda: make_worker("qwen", ["--role", "qwen", "--qwen", qwen_model, *common]
                                        + (["--int8"] if qwen_int8 else []) + offload, launcher, log("qwen"))}
        self._spawn["sam3"] = lambda: make_worker("core", ["--role", "sam3", "--sam3", sam3, *common], launcher,
                                                  log("sam3"))
        self._spawn["jev"] = lambda: make_worker("qwen", ["--role", "jev", *common]
                                                 + (["--jev", str(jev_model)] if jev_model else []), launcher,
                                                 log("jev"))
        self.d1_workers = ("sam3d", "qwen") if d1_evidence == "grid" or multiview_grid else ("qwen",)
        names = ["sam3"]
        if selector == "jev_sam3":  # no SAM3D or Qwen: the GPU also holds ApexNav's detectors
            from perception.jev import JEV_GATE, JevGate
            self.jev_gate = JevGate.load(WORKSPACE / (jev_gate or JEV_GATE))
            names = ["jev", "sam3"]
        elif selector == "d1" and not all_views:
            from perception.qwen import D1_GATE, SufficiencyGate
            self.qwen_gate = SufficiencyGate.load(WORKSPACE / (qwen_gate or D1_GATE))
            names += [] if lazy_workers else self.d1_workers
        elif selector not in ("input", "naive") or all_views:
            names.append("sam3d")
        if selector == "qwen_grid":
            from perception.qwen import UncertaintyGate
            self.qwen_gate = UncertaintyGate.load(WORKSPACE / (qwen_gate or QWEN_GATE))
            names.append("qwen")
        self._start(*names)

    def segment(self, image, query, threshold=0.0, max_masks=1):
        """SAM3 instances of ``query`` in a frame: [(mask PIL "L" image, score, xyxy box)], best first."""
        found = self.sam3.call("segment", image=encode(image), query=query, threshold=threshold,
                               max_masks=max_masks)
        return [(decode(f["mask"], "L"), f["score"], f["box"]) for f in found]

    def classify(self, image, mask=None, vocabulary=None):
        """SAM3 score per category for a frame; with ``mask``, only detections on that object."""
        return self.sam3.call("classify", image=encode(image), prompts=vocabulary or self.vocabulary,
                              mask=None if mask is None else encode(mask, "L"), iou=self.input_iou)

    def candidates(self, image, words, vocabulary=None, min_score=0.3):
        """SAM3 as the approach detector (worker op ``candidates``): instances of ``words`` in the full frame,
        each with its mask and SAM3's score of every vocabulary word on that segment."""
        return self.sam3.call("candidates", image=encode(image), prompts=list(vocabulary or self.vocabulary),
                              words=list(words), min_score=min_score)

    def verify(self, image, mask, target, input_elevation=0.0, vocabulary=None):
        """Verify that the object under ``mask`` in ``image`` is a ``target``.

        ``image``/``mask``: path, PIL image or array (mask nonzero on the object).
        ``input_elevation``: camera elevation in degrees, positive looking down at
        the object (Habitat: -pitch). It levels the candidate views.
        """
        if self.selector == "naive" and not self.all_views:
            scores = self.classify(image, mask, [target])
            accepted = scores.get(target, 0.0) >= select.NAIVE_THRESHOLD
            return Verdict(accepted=accepted, label=target if accepted else None, scores=scores, judged="input",
                           trusted=True)
        vocabulary = select.competitors(vocabulary or self.vocabulary, target)
        if self.selector == "jev_sam3":
            return self._jev_sam3(image, mask, target, vocabulary)
        if self.selector == "qwen_grid" and not self.all_views:
            return self._qwen_grid(image, mask, target, input_elevation, vocabulary)
        if self.selector == "d1" and not self.all_views:
            return self._d1(image, mask, target, input_elevation, vocabulary)
        if (self.selector == "input" or select.flat(target)) and not self.all_views:
            scores = self.classify(image, mask, vocabulary)
            label = select.top1(scores)[0]
            return Verdict(accepted=label == target, label=label, scores=scores, judged="input", trusted=True)
        input_scores = None
        if self.judge_input and not self.all_views:
            input_scores = self.classify(image, mask, vocabulary)
            label = select.top1(input_scores)[0]
            if label != target:
                return Verdict(accepted=False, label=label, scores=input_scores, judged="input",
                               input_scores=input_scores)
            if self.view_when == "unsure" and not select.unsure(input_scores, target, self.unsure_margin):
                return Verdict(accepted=True, label=label, scores=input_scores, judged="input",
                               input_scores=input_scores)
        rgb, silhouette = encode(image), encode(mask, "L")
        angles = [(a, e) for e in self.grid["elevations"] for a in self.grid["azimuths"]]
        views = {angle: View(*angle, absolute_elevation=sam3d_to_viewactive(*angle, input_elevation)[1])
                 for angle in angles}
        if self.selector == "vqf" or self.all_views:
            quality = self.sam3d.call("vqf", image=rgb, mask=silhouette, input_elevation=input_elevation,
                                      angles=angles)
            for angle, (value, _) in zip(angles, quality):
                views[angle].quality = value
        render = dict(self.grid)
        if self.selector == "vqf" and not self.all_views:
            pick = select.vqf_pick({a: (v.quality, v.absolute_elevation) for a, v in views.items()},
                                   self.elevation_range)
            render.update(azimuths=[pick[0]], elevations=[pick[1]])
        # Distrusted reconstructions are not rendered, unless every view was asked for.
        result = self.sam3d.call("reconstruct", image=rgb, mask=silhouette, input_elevation=input_elevation,
                                 ensemble=self.seeds, max_uncertainty=None if self.all_views else self.threshold,
                                 **render)
        uncertainty = result["uncertainty"] if self.threshold is not None else None
        trusted = select.trusted(uncertainty, self.threshold)
        rendered = [views[(v["azimuth"], v["elevation"])] for v in result["views"]]
        for view, data in zip(rendered, result["views"]):
            view.image = decode(data["image"])
        if rendered and (self.all_views or (self.selector == "top2gap" and trusted)):
            scores = self.sam3.call("classify_views", images=[v["image"] for v in result["views"]],
                                    prompts=vocabulary)
            for view, value in zip(rendered, scores):
                view.scores = value
        verdict = Verdict(accepted=False, label=None, scores={}, judged="none", uncertainty=uncertainty,
                          trusted=trusted, views=list(views.values()), timing=result["timing"],
                          input_scores=input_scores)
        if not trusted:
            if self.on_uncertain == "input":
                verdict.scores, verdict.judged = self.classify(image, mask, vocabulary), "input"
            else:
                verdict.accepted = self.on_uncertain == "accept"
                return verdict
        else:
            if self.selector == "top2gap":
                key = select.top2gap_pick({(v.azimuth, v.elevation): v.scores for v in rendered},
                                          {(v.azimuth, v.elevation): v.absolute_elevation for v in rendered},
                                          self.elevation_range)
            else:
                key = select.vqf_pick({(v.azimuth, v.elevation): (v.quality, v.absolute_elevation)
                                       for v in rendered}, self.elevation_range)
            verdict.view, verdict.judged = views[key], "view"
            if verdict.view.scores is None:
                verdict.view.scores = self.sam3.call("classify", image=encode(verdict.view.image),
                                                     prompts=vocabulary)
            verdict.scores = verdict.view.scores
        verdict.label = select.top1(verdict.scores)[0]
        verdict.accepted = verdict.label == target
        return verdict

    def _qwen_grid(self, image, mask, target, input_elevation, vocabulary):
        """Selector ``qwen_grid``: a three-way verdict (see the module docstring). Flat
        targets are reconstructed too (the gate was frozen with their view flip rates)."""
        input_scores, raw_label, margin, area, rgb, encoded_mask = self._raw(image, mask, target, vocabulary)
        angles = [(a, e) for e in self.grid["elevations"] for a in self.grid["azimuths"]]
        views = {angle: View(*angle, absolute_elevation=sam3d_to_viewactive(*angle, input_elevation)[1])
                 for angle in angles}
        timing, error, flip, grid = None, None, None, None
        try:
            result = self.sam3d.call("reconstruct", image=rgb, mask=encoded_mask, input_elevation=input_elevation,
                                     ensemble=1, max_uncertainty=None, **self.grid)
        except RuntimeError as failure:  # e.g. SAM3D found no object; the worker keeps running
            error = str(failure)
        else:
            timing = result["timing"]
            rendered = [views[(v["azimuth"], v["elevation"])] for v in result["views"]]
            scores = self.sam3.call("classify_views", images=[v["image"] for v in result["views"]],
                                    prompts=vocabulary)
            for view, data, value in zip(rendered, result["views"], scores):
                view.image, view.scores = decode(data["image"]), value
            flip = select.view_flip({(v.azimuth, v.elevation): v.scores for v in rendered},
                                    {(v.azimuth, v.elevation): v.absolute_elevation for v in rendered}, raw_label)
            if len(rendered) == len(angles):
                grid = _grid_image(rendered)
        assessment = self.qwen_gate.assess(margin, area, flip)
        gate = {"margin": margin, "area": area, "view_flip": flip, "reconstruct_error": error, **assessment}
        qwen = None
        if not assessment["review"]:
            status = "correct" if raw_label == target else "wrong"
        elif grid is None:  # never substitute a partial grid or the bare frame
            status = "insufficient"
        else:
            qwen = self.qwen.call("judge", image=rgb, mask=encoded_mask, target=target, grid=encode(grid))
            status = qwen["status"]
        return Verdict(accepted=status == "correct", label=raw_label, scores=input_scores,
                       judged="qwen" if assessment["review"] else "input", trusted=True,
                       views=list(views.values()), timing=timing, input_scores=input_scores,
                       status=status, source="qwen" if assessment["review"] else "sam3", gate=gate, qwen=qwen,
                       grid=grid)

    def _raw(self, image, mask, target, vocabulary):
        """SAM3 on the masked raw frame: (scores, top-1, target margin, mask area fraction,
        encoded frame, encoded mask)."""
        from perception.images import to_pil

        scores = self.classify(image, mask, vocabulary)
        margin = scores[target] - max(s for c, s in scores.items() if c != target)
        silhouette = to_pil(mask, "L")
        area = sum(silhouette.point(lambda p: 1 if p > 0 else 0).getdata()) / (silhouette.width * silhouette.height)
        return scores, select.top1(scores)[0], margin, area, encode(image), encode(silhouette, "L")

    def _start(self, *names):
        """Start the named workers (``_spawn``) that are not running yet; all load concurrently."""
        new = [name for name in names if getattr(self, name) is None]
        try:
            for name in new:
                setattr(self, name, self._spawn[name]())
            for name in new:
                getattr(self, name).wait_ready()
        except Exception:  # close every worker this call spawned, so none is left half-started
            for name in new:
                if getattr(self, name) is not None:
                    getattr(self, name).close()
                    setattr(self, name, None)
            raise

    def _d1(self, image, mask, target, input_elevation, vocabulary):
        """Selector ``d1`` (see the module docstring). Qwen's answer is the status as is;
        a failed reconstruction or Qwen call is "insufficient" with ``qwen["error"]``."""
        input_scores, raw_label, margin, area, rgb, encoded_mask = self._raw(image, mask, target, vocabulary)
        gate = self.qwen_gate.assess(margin, area)
        verdict = Verdict(accepted=False, label=raw_label, scores=input_scores, judged="input", trusted=True,
                          input_scores=input_scores, gate=gate)
        if not gate["insufficient"]:
            verdict.status, verdict.source = ("correct" if raw_label == target else "wrong"), "sam3"
            verdict.accepted = verdict.status == "correct"
            return verdict
        # Worker start failures propagate (an HTTP 500): they are not a property of the request.
        self._start(*self.d1_workers)  # no-op unless lazy_workers
        qwen = {"status": "insufficient", "probabilities": None, "invalid_answer": False, "error": None,
                "evidence": self.d1_evidence}
        verdict.judged, verdict.source, verdict.qwen = "qwen", "qwen", qwen
        try:
            if self.d1_evidence == "grid":
                result = self.sam3d.call("reconstruct", image=rgb, mask=encoded_mask,
                                         input_elevation=input_elevation, ensemble=1, max_uncertainty=None,
                                         **self.grid)
                verdict.timing = result["timing"]
                rendered = [View(v["azimuth"], v["elevation"], image=decode(v["image"])) for v in result["views"]]
                expected = len(self.grid["azimuths"]) * len(self.grid["elevations"])
                if len(rendered) != expected:  # never a partial grid
                    raise RuntimeError(f"{len(rendered)} of {expected} views rendered")
                verdict.grid = _grid_image(rendered)
            answer = self.qwen.call("judge", image=rgb, mask=encoded_mask, target=target,
                                    grid=None if verdict.grid is None else encode(verdict.grid))
        except RuntimeError as failure:
            stage = "qwen" if verdict.grid is not None or self.d1_evidence == "input" else "reconstruct"
            if not getattr(self, stage if stage == "qwen" else "sam3d").alive():
                raise  # a dead worker is a server failure (HTTP 500), not a property of the request
            qwen["error"] = f"{stage}: {failure}"  # the worker reported it and keeps running
        else:
            qwen.update(answer)
            if qwen["status"] not in ("correct", "wrong", "insufficient") or qwen["invalid_answer"]:
                qwen["error"] = f"invalid answer {qwen.get('top_token')!r} (status {qwen['status']!r})"
                qwen["status"] = "insufficient"
            elif (qwen["status"] == "correct" and self.d1_accept_min is not None
                  and qwen["probabilities"]["correct"] < self.d1_accept_min):
                qwen["status"], qwen["below_accept_min"] = "insufficient", self.d1_accept_min  # not sure enough
        verdict.status = qwen["status"]
        verdict.accepted = verdict.status == "correct"
        return verdict

    def _jev_name(self, image, mask, target, words="gate"):
        """JEV's raw naming distribution of the masked object (worker op ``name``): ``words`` "gate" (the gate's
        words, ``jev.name_opts``) or "approach" (HM3D words, ``jev.approach_opts``; the approach belief)."""
        from perception.jev import known
        if not known(target):  # OVON category without a confuser list: fillers only, contention 0
            print(f"jev: target {target!r} has no confuser list; fillers only, contention 0", flush=True)
        return self.jev.call("name", image=encode(image), mask=encode(mask, "L"), target=target, words=words)

    def _jev(self, image, mask, target):
        """JEV's gate on one frame: "insufficient" when unsure, else correct/wrong by JEV's top-1 (source "jev";
        ``_jev_sam3`` replaces the decision by SAM3's). A failed JEV call is "insufficient" with
        ``jev["error"]`` (as ``_d1``); a dead worker raises."""
        verdict = Verdict(accepted=False, label=None, scores={}, judged="jev", trusted=True, source="jev",
                          status="insufficient")
        try:
            answer = self._jev_name(image, mask, target, self.jev_gate.words)
        except RuntimeError as failure:
            if not self.jev.alive():
                raise  # a dead worker is a server failure (HTTP 500), not a property of the request
            verdict.jev = {"error": f"jev: {failure}"}
            return verdict
        verdict.status, verdict.gate = self.jev_gate.decide(answer["probabilities"], target)
        verdict.jev = answer
        verdict.label, verdict.scores = verdict.gate["top1"], dict(verdict.gate["probabilities"])
        verdict.accepted = verdict.status == "correct"
        return verdict

    def _jev_views(self, images, masks, target, keys, cached_jev=None):
        """JEV names every view (nearest first); the gate judges the mean of their raw distributions
        (insufficient, or correct/wrong by its top-1). ``cached_jev`` {key: raw probabilities} the
        caller kept from an earlier reply (JEV is deterministic) skips JEV for that view ("jev_cached").
        Views whose call fails are left out; when every call fails, the error propagates (HTTP 500)."""
        import numpy as np

        from perception.images import to_pil
        from perception.jev import average
        per_view, answers, failure = [], [], None
        for key, image, mask in zip(keys, images, masks):
            binary = np.asarray(to_pil(mask, "L")) > 0
            if not binary.any():
                raise ValueError(f"empty mask for view {key}")
            step, index = key if isinstance(key, tuple) and len(key) == 2 else (None, key)
            view = {"step": step, "index": index, "area": float(binary.mean()),
                    "clipped": bool(binary[0].any() or binary[-1].any() or binary[:, 0].any() or binary[:, -1].any()),
                    "scores": None, "margin": None, "label": None, "cached": False, "error": None}
            per_view.append(view)
            try:
                if key in (cached_jev or {}):
                    answer = {"probabilities": dict(cached_jev[key]), "latency_s": 0.0}
                    view["jev_cached"] = True
                else:
                    answer = self._jev_name(image, mask, target, self.jev_gate.words)
            except RuntimeError as error:
                if not self.jev.alive():
                    raise
                view["error"], failure = f"jev: {error}", error
                continue
            answers.append(answer["probabilities"])
            single = self.jev_gate.assess(answer["probabilities"], target)
            view.update(probabilities=answer["probabilities"], label=single["top1"], entropy=single["entropy"],
                        contention=single["contention"], score=single["score"], latency_s=answer["latency_s"])
        if not answers:
            raise RuntimeError(f"jev failed on all {len(per_view)} views: {failure}")
        status, assessment = self.jev_gate.decide(average(answers), target)
        return ViewsVerdict(status=status, source="jev", per_view=per_view, accepted=status == "correct",
                            jev={**assessment, "n_used": len(answers)})

    def _jev_sam3(self, image, mask, target, vocabulary):
        """Selector ``jev_sam3`` (see the module docstring): JEV's gate on the frame; when sure, SAM3's
        raw label decides. ``gate``/``jev`` keep JEV's assessment, ``scores`` are SAM3's."""
        verdict = self._jev(image, mask, target)
        if verdict.status == "insufficient":  # unsure (or a failed JEV call): approach
            return verdict
        input_scores, raw_label = self._raw(image, mask, target, vocabulary)[:2]
        verdict.status = "correct" if raw_label == target else "wrong"
        verdict.accepted, verdict.source, verdict.judged = verdict.status == "correct", "sam3", "input"
        verdict.label, verdict.scores, verdict.input_scores = raw_label, input_scores, input_scores
        return verdict

    def _jev_sam3_views(self, images, masks, target, keys, forced, vocabulary, cached, cached_jev=None):
        """``verify_views`` for selector ``jev_sam3``: JEV's unforced gate over the views' mean
        distribution; unforced and unsure -> "insufficient"; else SAM3 over every view
        (``cached`` scores reused) and ``select.fuse_views``: correct iff the target has the highest fused
        belief. ``fused`` is the decision, ``aggregate`` (margin, vote) is logged only. ``jev`` keeps JEV's
        assessment (``forced`` as asked)."""
        verdict = self._jev_views(images, masks, target, keys, cached_jev)
        verdict.jev["forced"] = bool(forced)
        if verdict.status == "insufficient" and not forced:
            return verdict
        for key, image, mask, view in zip(keys, images, masks, verdict.per_view):
            if key in cached:
                view["scores"], view["cached"] = dict(cached[key]), True
            else:
                view["scores"] = self.classify(image, mask, vocabulary)
            others = [v for c, v in view["scores"].items() if c != target]
            view["margin"] = view["scores"].get(target, 0.0) - (max(others) if others else 0.0)
            view["jev_label"], view["label"] = view.get("label"), select.top1(view["scores"])[0]
        verdict.aggregate = select.aggregate_views(verdict.per_view, target)
        verdict.fused = select.fuse_views(verdict.per_view, target)
        accepted = verdict.fused["accept"]
        verdict.status, verdict.source, verdict.accepted = ("correct" if accepted else "wrong"), "sam3", accepted
        return verdict

    def verify_views(self, images, masks, target, distances=None, input_elevations=None, keys=None,
                     vocabulary=None, sam3="fallback", cached_scores=None, grid=False, grid_view=0,
                     crop_small=False, forced2=False, forced2_min=0.5, sam3_shortcut=False, grid_scores=False,
                     forced=True, cached_jev=None):
        """Selector ``d1``, close range: is the object seen in ``images`` (nearest first; ``masks``
        nonzero on it, one per image) a ``target``? Returns a ``ViewsVerdict``.

        Qwen judges all views at once (``qwen.views_content``). ``grid``: SAM3D reconstructs
        view ``grid_view`` (at ``input_elevations[grid_view]``) and Qwen also sees the 24-view
        grid; a failed reconstruction falls back to the images alone (``qwen["reconstruct_error"]``),
        never to "insufficient". A failed or invalid Qwen answer is "insufficient" with
        ``qwen["error"]`` (as ``_d1``); ``d1_accept_min`` applies to "correct".
        ``sam3``: SAM3 scores every view whose key (``keys[k]``, e.g. (step, index)) has no
        ``cached_scores`` entry "always", only when the status is insufficient ("fallback"), or
        "never"; ``select.aggregate_views`` pools them. ``sam3_shortcut`` (ablation): SAM3 on view
        ``grid_view`` first, and a frame the sufficiency gate calls sufficient decides as in
        ``_d1`` (no Qwen, no SAM3D). ``grid_scores`` (diagnostic only): SAM3 top-1 of each render.
        ``crop_small``/``forced2``/``forced2_min``: ``QwenServer.judge_views`` variants.
        Selector ``jev_sam3`` (``_jev_sam3_views``): JEV judges the uncertainty, SAM3 decides with
        ``select.fuse_views``; ``sam3`` and the variants are ignored, ``grid`` is an error;
        ``cached_jev`` {key: raw JEV probabilities} skips JEV for those views (``_jev_views``).
        """
        import numpy as np

        from perception.images import to_pil

        if self.selector not in ("d1", "jev_sam3") or self.all_views:
            raise ValueError("verify_views needs selector d1 or jev_sam3")
        n = len(images)
        if not 1 <= n <= MULTIVIEW_MAX or len(masks) != n:
            raise ValueError(f"verify_views takes 1 to {MULTIVIEW_MAX} images, each with a mask")
        if sam3 not in MULTIVIEW_SAM3:
            raise ValueError(f"sam3 must be one of {MULTIVIEW_SAM3}")
        if not 0 <= grid_view < n:
            raise ValueError("grid_view must index one of the images")
        distances = list(distances) if distances is not None else [None] * n
        elevations = list(input_elevations) if input_elevations is not None else [0.0] * n
        keys = [tuple(k) if isinstance(k, list) else k for k in keys] if keys is not None else list(range(n))
        if len(distances) != n or len(elevations) != n or len(keys) != n:
            raise ValueError("distances, input_elevations and keys must match images")
        if self.selector == "jev_sam3" and grid:
            raise ValueError("selector jev_sam3 has no SAM3D grid")
        cached = {tuple(k) if isinstance(k, list) else k: v for k, v in (cached_scores or {}).items()}
        vocabulary = select.competitors(vocabulary or self.vocabulary, target)
        if self.selector == "jev_sam3":
            cached_jev = {tuple(k) if isinstance(k, list) else k: v for k, v in (cached_jev or {}).items()}
            return self._jev_sam3_views(images, masks, target, keys, forced, vocabulary, cached, cached_jev)
        rgbs, silhouettes, per_view = [], [], []
        for key, image, mask in zip(keys, images, masks):
            silhouette = to_pil(mask, "L")
            binary = np.asarray(silhouette) > 0
            if not binary.any():
                raise ValueError(f"empty mask for view {key}")
            rgbs.append(encode(image))
            silhouettes.append(encode(silhouette, "L"))
            step, index = key if isinstance(key, tuple) and len(key) == 2 else (None, key)
            per_view.append({"step": step, "index": index, "area": float(binary.mean()),
                             "clipped": bool(binary[0].any() or binary[-1].any() or binary[:, 0].any()
                                             or binary[:, -1].any()),
                             "scores": None, "margin": None, "label": None, "cached": False})

        def score(k):
            view = per_view[k]
            if view["scores"] is not None:
                return
            if keys[k] in cached:
                view["scores"], view["cached"] = dict(cached[keys[k]]), True
            else:
                view["scores"] = self.sam3.call("classify", image=rgbs[k], prompts=vocabulary, mask=silhouettes[k],
                                                iou=self.input_iou)
            scores = view["scores"]
            others = [v for c, v in scores.items() if c != target]
            view["margin"] = scores.get(target, 0.0) - (max(others) if others else 0.0)
            view["label"] = select.top1(scores)[0] if scores else None

        verdict = ViewsVerdict(status="insufficient", source="qwen", per_view=per_view)
        if sam3_shortcut:
            score(grid_view)
            view = per_view[grid_view]
            gate = self.qwen_gate.assess(view["margin"], view["area"])
            if not gate["insufficient"]:
                verdict.status, verdict.source = ("correct" if view["label"] == target else "wrong"), "sam3"
                verdict.qwen = {"shortcut_gate": gate}
                if sam3 == "always":
                    for k in range(n):
                        score(k)
                verdict.aggregate = select.aggregate_views([v for v in per_view if v["scores"] is not None], target)
                verdict.accepted = verdict.status == "correct"
                return verdict
        self._start("qwen", *(("sam3d",) if grid else ()))  # no-op unless lazy (or grid with evidence "input")
        qwen = {"status": "insufficient", "probabilities": None, "invalid_answer": False, "error": None,
                "grid": False, "grid_view": grid_view if grid else None, "reconstruct_error": None}
        verdict.qwen = qwen
        encoded_grid = None
        if grid:
            try:
                result = self.sam3d.call("reconstruct", image=rgbs[grid_view], mask=silhouettes[grid_view],
                                         input_elevation=float(elevations[grid_view]), ensemble=1,
                                         max_uncertainty=None, **self.grid)
                rendered = [View(v["azimuth"], v["elevation"], image=decode(v["image"])) for v in result["views"]]
                expected = len(self.grid["azimuths"]) * len(self.grid["elevations"])
                if len(rendered) != expected:  # never a partial grid
                    raise RuntimeError(f"{len(rendered)} of {expected} views rendered")
            except RuntimeError as failure:  # e.g. CUDA out of memory: the images alone still decide
                if not self.sam3d.alive():
                    raise  # a dead worker is a server failure (HTTP 500), not a property of the request
                qwen["reconstruct_error"] = str(failure)
            else:
                verdict.timing, verdict.grid = result["timing"], _grid_image(rendered)
                encoded_grid, qwen["grid"] = encode(verdict.grid), True
                if grid_scores:
                    labels = [select.top1(s)[0] for s in self.sam3.call(
                        "classify_views", images=[v["image"] for v in result["views"]], prompts=vocabulary)]
                    verdict.grid_scores = {"labels": {f"{v.azimuth:g},{v.elevation:g}": label
                                                      for v, label in zip(rendered, labels)},
                                           "target_fraction": sum(label == target for label in labels) / len(labels)}
        try:
            answer = self.qwen.call("judge_views", images=rgbs, masks=silhouettes, target=target,
                                    distances=distances, grid=encoded_grid, crop_small=crop_small,
                                    forced2=forced2, forced2_min=forced2_min)
        except RuntimeError as failure:
            if not self.qwen.alive():
                raise
            qwen["error"] = f"qwen: {failure}"
        else:
            qwen.update(answer)
            if qwen["status"] not in ("correct", "wrong", "insufficient") or qwen["invalid_answer"]:
                qwen["error"] = f"invalid answer {qwen.get('top_token')!r} (status {qwen['status']!r})"
                qwen["status"] = "insufficient"
            elif (qwen["status"] == "correct" and self.d1_accept_min is not None and not forced2
                  and qwen["probabilities"]["correct"] < self.d1_accept_min):
                qwen["status"], qwen["below_accept_min"] = "insufficient", self.d1_accept_min
        verdict.status = qwen["status"]
        if sam3 == "always" or (sam3 == "fallback" and verdict.status == "insufficient"):
            for k in range(n):
                score(k)
        scored = [v for v in per_view if v["scores"] is not None]
        verdict.aggregate = select.aggregate_views(scored, target) if scored else None
        verdict.accepted = verdict.status == "correct"
        return verdict

    def alive(self):
        """{worker: True while its process runs}."""
        return {name: worker.alive() for name, worker in
                (("sam3", self.sam3), ("sam3d", self.sam3d), ("qwen", self.qwen), ("jev", self.jev)) if worker}

    def close(self):
        for worker in (getattr(self, "sam3", None), getattr(self, "sam3d", None), getattr(self, "qwen", None),
                       getattr(self, "jev", None)):
            if worker is not None:
                worker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
