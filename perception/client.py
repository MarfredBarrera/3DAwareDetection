"""``Verifier``: check an object detection by classifying a better SAM3D view of it.

    from perception import Verifier

    with Verifier(vocabulary=["chair", "couch", "bed", "toilet", "tv", "table"]) as verifier:
        verdict = verifier.verify(rgb, mask, target="chair", input_elevation=20.0)
        if verdict.accepted:
            ...  # commit to the detection

Per call: SAM3D reconstructs the masked object; the ensemble filter (optional)
measures stage-1 seed disagreement; a selector (``vqf`` or ``top2gap``) picks
one candidate view; SAM3 scores the vocabulary on it; the detection is
accepted when the top-1 category is the target. The models live in two
persistent worker processes (see ``perception.worker``), loaded once.

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

WORKSPACE = Path(__file__).resolve().parents[1]
CONTAINER = os.environ.get("PERCEPTION_CONTAINER", "3dawareverification-dev-1")
VIEWACTIVE = "/data/checkpoints/viewactive/blender_full_el0_45.pt"
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
    quality: float = None        # ViewActive VQF quality (vqf selector)
    vqf_elevation: float = None  # absolute elevation (gravity frame) the quality refers to
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


def default_launcher():
    if shutil.which("sam3d-env"):
        return ["sam3d-env", "{env}"]
    return ["docker", "exec", "-i", "-w", "/workspace", CONTAINER, "sam3d-env", "{env}"]


class _Worker:
    def __init__(self, env, args, launcher, log):
        command = [part.format(env=env) for part in launcher] + ["python", "-m", "perception.worker", *args]
        self.log = open(log, "a") if log else None
        environment = {**os.environ, "NUMPY_MADVISE_HUGEPAGE": "0"}
        self.process = subprocess.Popen(command, cwd=WORKSPACE, env=environment, text=True, bufsize=1,
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

    selector: "vqf" (learned ViewActive viewpoint quality; one rendered view) or
        "top2gap" (SAM3's largest top-1 minus top-2 gap over all candidates).
    ensemble: run the ensemble filter (``ensemble_seeds`` stage-1 seeds); a
        reconstruction is trusted when its disagreement <= ``ensemble_threshold``
        (0.10-0.20 keeps ~99% of useful reconstructions on 1,800 LVIS objects).
    on_uncertain: when not trusted, "input" classifies the raw frame, "reject"
        rejects the detection, "accept" keeps it unverified.
    vocabulary: the closed set of categories SAM3 ranks; the target is added.
    all_views: render and classify every candidate, also of distrusted reconstructions
        (for evaluation and figures; slower).
    """

    def __init__(self, selector="vqf", ensemble=True, ensemble_threshold=0.15, ensemble_seeds=4,
                 on_uncertain="input", vocabulary=HOUSEHOLD, gpu=0, all_views=False,
                 azimuths=(0, 45, 90, 135, 180, 225, 270, 315), elevations=(0, 30, 60),
                 distance=2.0, resolution=320, seed=42, vqf_elevations=select.VQF_ELEVATIONS,
                 input_iou=0.3, sam3="/data/checkpoints/sam3/sam3.pt",
                 sam3d="/data/checkpoints/sam3d/hf", viewactive=VIEWACTIVE, launcher=None, log_dir=None):
        if selector not in select.SELECTORS:
            raise ValueError(f"selector must be one of {select.SELECTORS}")
        if on_uncertain not in select.ON_UNCERTAIN:
            raise ValueError(f"on_uncertain must be one of {select.ON_UNCERTAIN}")
        if ensemble and ensemble_seeds < 2:
            raise ValueError("the ensemble filter needs ensemble_seeds >= 2")
        self.selector, self.on_uncertain, self.all_views = selector, on_uncertain, all_views
        self.threshold = ensemble_threshold if ensemble else None
        self.seeds = ensemble_seeds if ensemble else 1
        self.vocabulary = list(dict.fromkeys(vocabulary))
        self.grid = {"azimuths": [float(a) for a in azimuths], "elevations": [float(e) for e in elevations],
                     "distance": distance, "resolution": resolution, "seed": seed}
        self.vqf_elevations, self.input_iou = tuple(vqf_elevations), input_iou
        launcher = launcher or default_launcher()
        log = (lambda name: Path(log_dir) / f"{name}.log") if log_dir else (lambda name: None)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        common = ["--gpu", str(gpu)]
        self.sam3 = _Worker("core", ["--role", "sam3", "--sam3", sam3, *common], launcher, log("sam3"))
        uses_vqf = selector == "vqf" or all_views
        self.sam3d = _Worker("sam3d", ["--role", "sam3d", "--sam3d", sam3d, *common]
                             + (["--viewactive", str(viewactive)] if uses_vqf else []), launcher, log("sam3d"))
        try:
            self.sam3.wait_ready()  # Both load concurrently.
            self.sam3d.wait_ready()
        except Exception:
            self.close()
            raise

    def segment(self, image, query, threshold=0.0, max_masks=1):
        """SAM3 instances of ``query`` in a frame: [(mask PIL "L" image, score, xyxy box)], best first."""
        found = self.sam3.call("segment", image=encode(image), query=query, threshold=threshold,
                               max_masks=max_masks)
        return [(decode(f["mask"], "L"), f["score"], f["box"]) for f in found]

    def classify(self, image, mask=None, vocabulary=None):
        """SAM3 score per category for a frame; with ``mask``, only detections on that object."""
        return self.sam3.call("classify", image=encode(image), prompts=vocabulary or self.vocabulary,
                              mask=None if mask is None else encode(mask, "L"), iou=self.input_iou)

    def verify(self, image, mask, target, input_elevation=0.0, vocabulary=None):
        """Verify that the object under ``mask`` in ``image`` is a ``target``.

        ``image``/``mask``: path, PIL image or array (mask nonzero on the object).
        ``input_elevation``: camera elevation in degrees, positive looking down at
        the object (Habitat: -pitch). It levels the candidate views.
        """
        vocabulary = list(dict.fromkeys([*(vocabulary or self.vocabulary), target]))
        rgb, silhouette = encode(image), encode(mask, "L")
        angles = [(a, e) for e in self.grid["elevations"] for a in self.grid["azimuths"]]
        views = {angle: View(*angle) for angle in angles}
        if self.selector == "vqf" or self.all_views:
            quality = self.sam3d.call("vqf", image=rgb, mask=silhouette, input_elevation=input_elevation,
                                      angles=angles)
            for angle, (value, elevation) in zip(angles, quality):
                views[angle].quality, views[angle].vqf_elevation = value, elevation
        render = dict(self.grid)
        if self.selector == "vqf" and not self.all_views:
            pick = select.vqf_pick({a: (v.quality, v.vqf_elevation) for a, v in views.items()},
                                   self.vqf_elevations)
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
                          trusted=trusted, views=list(views.values()), timing=result["timing"])
        if not trusted:
            if self.on_uncertain == "input":
                verdict.scores, verdict.judged = self.classify(image, mask, vocabulary), "input"
            else:
                verdict.accepted = self.on_uncertain == "accept"
                return verdict
        else:
            if self.selector == "top2gap":
                key = select.top2gap_pick({(v.azimuth, v.elevation): v.scores for v in rendered})
            else:
                key = select.vqf_pick({(v.azimuth, v.elevation): (v.quality, v.vqf_elevation)
                                       for v in rendered}, self.vqf_elevations)
            verdict.view, verdict.judged = views[key], "view"
            if verdict.view.scores is None:
                verdict.view.scores = self.sam3.call("classify", image=encode(verdict.view.image),
                                                     prompts=vocabulary)
            verdict.scores = verdict.view.scores
        verdict.label = select.top1(verdict.scores)[0]
        verdict.accepted = verdict.label == target
        return verdict

    def close(self):
        for worker in (getattr(self, "sam3", None), getattr(self, "sam3d", None)):
            if worker is not None:
                worker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
