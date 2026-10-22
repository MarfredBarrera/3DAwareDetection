"""Load and validate a study YAML; relative paths resolve against the YAML file."""
import hashlib
import json
import math
from pathlib import Path

from study.ambiguity import metric_options
from study.labels import prompt_text as _prompt

DEFAULTS = {
    "gpu_id": 0,
    "checkpoints": {"sam3": "/data/checkpoints/sam3/sam3.pt",
                    "sam3d": "/data/checkpoints/sam3d/hf"},
    "dataset": {"root": "/data/objaverse_nas", "canonical": "/data/objaverse_nas/canonical.json",
                "manifest": None, "split": None, "source_run": None,
                # Import only N views per object from source_run (hash-chosen per uid).
                "views_per_object": None,
                "require_upright_verified": False, "categories": None,
                "objects_per_category": 10,
                "azimuths": [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
                "elevations": [0, 15, 30, 45, 60], "distance": 2.2, "image_size": 320},
    "detection": {"score_threshold": 0.5, "iou_threshold": 0.5},
    "reconstruction": {"objects_per_category": 10, "input_elevations": None,
                       "candidate_azimuths": [0, 45, 90, 135, 180, 225, 270, 315],
                       "candidate_elevations": [0, 30, 60], "candidate_distance": 2.0,
                       "resolution": 320, "seed": 42,
                       # Stage-1 seed ensemble size written to stage1.npz while
                       # reconstructing (1 = none). Not fingerprinted: it only adds a file.
                       "ensemble": 1},
    "ambiguity": {"visual_entropy": {}, "self_occlusion": {}, "surface_normal_entropy": {}},
    # Pre-detect view picks: the learned ViewActive ``vqf`` pick plus one seeded
    # ``random`` and the ``identity`` candidate; null disables the pick stage and
    # the vqf selector. detect_only runs SAM3 on the inputs and picks only.
    "picks": None,
    # ensemble_threshold: the ensemble filter (null = off). Inputs whose stage-1
    # disagreement exceeds it keep the input view for every selector; needs the
    # stage1.npz that reconstruction.ensemble > 1 writes.
    "analysis": {"primary_selector": "ambiguity:combined", "combined_weights": None,
                 "figure_selectors": None, "ensemble_threshold": None},
}
# Settings each stage's stored outputs depend on. Thresholds, input selection,
# and analysis settings are applied later and may change freely.
FINGERPRINTS = {
    # views_per_object is left out when unset to keep earlier studies' fingerprints.
    "render": lambda c: {k: v for k, v in c["dataset"].items()
                         if k != "views_per_object" or v is not None},
    "detect": lambda c: [c["dataset"]["categories"], c["checkpoints"]["sam3"]] + (
        # Prompts differ from the names only for LVIS-style categories.
        ["lvis-prompts"] if any(_prompt(n) != n for n in c["dataset"]["categories"]) else []),
    "reconstruct": lambda c: [c["checkpoints"]["sam3d"]] + [
        c["reconstruction"][k] for k in ("candidate_azimuths", "candidate_elevations",
                                         "candidate_distance", "resolution", "seed")] + [
        "gravity-up"],  # Candidate cameras keep gravity up (earlier studies were rolled).
    "score": lambda c: c["ambiguity"],
    "pick": lambda c: c["picks"],
}
PICK_DEFAULTS = {"viewactive": "/data/checkpoints/viewactive/blender_full_el0_45.pt",
                 "elevation_range": [0, 45], "seed": 0, "detect_only": True}


def _number(value, name, low=-math.inf, high=math.inf):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


def _integer(value, name, low):
    if type(value) is not int or value < low:
        raise ValueError(f"{name} must be an integer >= {low}")
    return value


def _angles(values, name, elevation=False):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a nonempty list")
    values = [float(_number(v, name, -89.999, 89.999) if elevation else _number(v, name, 0, 359.999))
              for v in values]
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _merge(defaults, supplied, name):
    if supplied is None:
        return dict(defaults)
    if not isinstance(supplied, dict) or supplied.keys() - defaults.keys():
        raise ValueError(f"{name} accepts only {sorted(defaults)}")
    return defaults | supplied


def load(path):
    """Return the resolved study settings as a plain JSON-serializable dict."""
    import yaml

    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Study config must be a YAML mapping")
    allowed = {"output_dir", *DEFAULTS}
    if raw.keys() - allowed:
        raise ValueError(f"Unknown study settings: {sorted(raw.keys() - allowed)}")

    def resolve(value, name):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty path")
        return str((path.parent / Path(value).expanduser()).resolve())

    config = {"output_dir": resolve(raw.get("output_dir"), "output_dir"),
              "gpu_id": _integer(raw.get("gpu_id", 0), "gpu_id", 0)}
    checkpoints = _merge(DEFAULTS["checkpoints"], raw.get("checkpoints"), "checkpoints")
    config["checkpoints"] = {k: resolve(v, f"checkpoints.{k}") for k, v in checkpoints.items()}

    dataset = _merge(DEFAULTS["dataset"], raw.get("dataset"), "dataset")
    dataset["canonical"] = resolve(dataset["canonical"], "dataset.canonical")
    for key in ("manifest", "source_run"):
        if dataset[key] is not None:
            dataset[key] = resolve(dataset[key], f"dataset.{key}")
    if dataset["manifest"] is None and dataset["split"] is not None:
        raise ValueError("dataset.split requires dataset.manifest")
    splits = dataset["split"] if isinstance(dataset["split"], list) else [dataset["split"]]
    if dataset["manifest"] is not None and not (
            splits and all(isinstance(s, str) and s.strip() for s in splits)):
        raise ValueError("dataset.manifest requires a nonempty dataset.split (name or list)")
    if dataset["source_run"] is not None and dataset["manifest"] is None:
        raise ValueError("dataset.source_run requires dataset.manifest")
    if dataset["views_per_object"] is not None:
        _integer(dataset["views_per_object"], "dataset.views_per_object", 1)
        if dataset["source_run"] is None:
            raise ValueError("dataset.views_per_object requires dataset.source_run")
    categories = dataset["categories"]
    if (not isinstance(categories, list) or len(categories) < 2
            or not all(isinstance(c, str) and c.strip() for c in categories)
            or len(set(categories)) != len(categories)):
        raise ValueError("dataset.categories must list at least two distinct category names")
    _integer(dataset["objects_per_category"], "dataset.objects_per_category", 1)
    dataset["azimuths"] = _angles(dataset["azimuths"], "dataset.azimuths")
    dataset["elevations"] = _angles(dataset["elevations"], "dataset.elevations", elevation=True)
    _number(dataset["distance"], "dataset.distance", 1e-6)
    _integer(dataset["image_size"], "dataset.image_size", 16)
    if type(dataset["require_upright_verified"]) is not bool:
        raise ValueError("dataset.require_upright_verified must be a boolean")
    config["dataset"] = dataset

    detection = _merge(DEFAULTS["detection"], raw.get("detection"), "detection")
    for key, value in detection.items():
        _number(value, f"detection.{key}", 0, 1)
    config["detection"] = detection

    recon = _merge(DEFAULTS["reconstruction"], raw.get("reconstruction"), "reconstruction")
    _integer(recon["objects_per_category"], "reconstruction.objects_per_category", 0)
    if recon["input_elevations"] is not None:
        recon["input_elevations"] = _angles(recon["input_elevations"],
                                            "reconstruction.input_elevations", elevation=True)
        if set(recon["input_elevations"]) - set(dataset["elevations"]):
            raise ValueError("reconstruction.input_elevations must be dataset elevations")
    recon["candidate_azimuths"] = _angles(recon["candidate_azimuths"], "reconstruction.candidate_azimuths")
    recon["candidate_elevations"] = _angles(recon["candidate_elevations"],
                                            "reconstruction.candidate_elevations", elevation=True)
    from tools.cameras import SAM3D_INPUT_VIEW
    if (SAM3D_INPUT_VIEW[0] not in recon["candidate_azimuths"]
            or SAM3D_INPUT_VIEW[1] not in recon["candidate_elevations"]):
        raise ValueError(f"The candidate grid must include the input view {SAM3D_INPUT_VIEW} "
                         "(azimuth 180, elevation 0) for the identity control")
    _number(recon["candidate_distance"], "reconstruction.candidate_distance", 0.5, 10)
    _integer(recon["resolution"], "reconstruction.resolution", 64)
    _integer(recon["seed"], "reconstruction.seed", 0)
    _integer(recon["ensemble"], "reconstruction.ensemble", 1)
    config["reconstruction"] = recon

    metrics = raw.get("ambiguity", DEFAULTS["ambiguity"])
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("ambiguity must map at least one metric name to its options")
    config["ambiguity"] = {name: metric_options(name, options or {}) for name, options in metrics.items()}

    config["picks"] = None
    if raw.get("picks") is not None:
        picks = config["picks"] = _merge(PICK_DEFAULTS, raw["picks"], "picks")
        picks["viewactive"] = resolve(picks["viewactive"], "picks.viewactive")
        limits = picks["elevation_range"]
        if limits is not None and (not isinstance(limits, list) or len(limits) != 2
                                   or _number(limits[0], "picks.elevation_range", -90, 90)
                                   > _number(limits[1], "picks.elevation_range", -90, 90)):
            raise ValueError("picks.elevation_range must be [low, high] degrees or null")
        _integer(picks["seed"], "picks.seed", 0)
        if type(picks["detect_only"]) is not bool:
            raise ValueError("picks.detect_only must be a boolean")

    analysis = _merge(DEFAULTS["analysis"], raw.get("analysis"), "analysis")
    from study.selectors import selector_names
    names = selector_names(config["ambiguity"], vqf=config["picks"] is not None)
    if analysis["primary_selector"] not in names:
        raise ValueError(f"analysis.primary_selector must be one of {names}")
    figure_selectors = analysis["figure_selectors"]
    if figure_selectors is not None and (
            not isinstance(figure_selectors, list) or not figure_selectors
            or any(name not in names for name in figure_selectors)
            or len(set(figure_selectors)) != len(figure_selectors)):
        raise ValueError(f"analysis.figure_selectors must list distinct names from {names}")
    if analysis["ensemble_threshold"] is not None:
        _number(analysis["ensemble_threshold"], "analysis.ensemble_threshold", 0, 1)
    combined = analysis["combined_weights"]
    if combined is not None:
        if not isinstance(combined, dict) or combined.keys() - config["ambiguity"].keys():
            raise ValueError(f"analysis.combined_weights must map metrics in {sorted(config['ambiguity'])} "
                             "to weights (missing metrics get 0)")
        for name, value in combined.items():
            _number(value, f"analysis.combined_weights.{name}", 0)
        if sum(combined.values()) <= 0:
            raise ValueError("analysis.combined_weights must have a positive weight")
    config["analysis"] = analysis
    return config


def fingerprint(config, stage, table=FINGERPRINTS):
    """Hash of the settings a stage's stored outputs depend on."""
    relevant = table[stage](config)
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]


# ViewActive (arXiv:2409.09997) reimplementation; see viewactive/__main__.py.
VIEWACTIVE_DEFAULTS = {
    "gpu_id": 0,
    "objaverse": {"nas_root": "/data/objaverse_nas", "cache_root": "/data/objaverse_viewactive",
                  "max_mb": 64, "max_faces": 300_000, "download_workers": 8,
                  # Keep each file's glTF +Y up (the paper only scales and centers);
                  # false applies harness.canonicalize's up-axis heuristic and drops
                  # objects it flags as suspect.
                  "trust_file_up": True},
    # Class counts or explicit LVIS category lists; counts take the mappable
    # categories with the most LVIS objects, seen first, then unseen.
    "cohort": {"class_map": "viewactive_imagenet_map.yaml", "seen_classes": 56,
               "unseen_classes": 10, "per_class": 29, "total_seen": 1600,
               "unseen_per_class": 20, "oversample": 2.0, "split": [0.70, 0.05, 0.25],
               "exclude_uids_from": [], "seed": 0},
    "render": {"azimuths": list(range(0, 360, 30)), "elevations": list(range(-75, 76, 15)),
               "distance": 2.2, "image_size": 320, "shard_size": 50,
               "backend": "pytorch3d", "width": 1920, "height": 1680,
               "fov": 50.0, "samples": 16,
               "blender_binary": "/data/cache/blender/blender-4.5.14-linux-x64/blender"},
    "labels": {"samples": 20000, "seed": 0, "polar_bins": 9, "azimuth_bins": 18,
               "visual_bins": 32, "floor": 0.05, "workers": 8},
    "encoders": {"dino": "facebook/dinov2-base", "clip": "openai/clip-vit-base-patch16", "batch": 128},
    "model": {"hidden": 320},
    "train": {"batch": 256, "epochs": 60, "lr": 1.0e-3, "weight_decay": 0.01, "gamma": 0.9,
              "loss_weights": [0.3, 0.4, 0.3], "silog_lambda": 0.85, "seed": 0},
    # elevation_range [low, high] degrees limits optimizer targets (not in the paper).
    "optimize": {"steps": 9, "reach_cells": [1, 1], "starts_per_object": 10,
                 "combine_weights": None, "elevation_range": None, "seed": 0},
    "classification": {"crop_margin": None},
    "classifiers": {"ResNet50": "resnet50.tv_in1k", "EfficientNetV2": "tf_efficientnetv2_s.in1k",
                    "Swin": "swin_tiny_patch4_window7_224.ms_in1k",
                    "ViT": "vit_base_patch16_224.augreg2_in21k_ft_in1k"},
}
_VIEWACTIVE_COHORT_VERSION = "global-stratified-split-v2"
_VIEWACTIVE_DATA = lambda c: [{k: v for k, v in c["objaverse"].items() if k != "download_workers"},
                              c["cohort"], _VIEWACTIVE_COHORT_VERSION]
_VIEWACTIVE_RENDER_VERSION = "partwise-colors-v2"
def _viewactive_render_settings(c):
    render = c["render"]
    if render["backend"] == "pytorch3d":
        # Preserve fingerprints of the existing PyTorch3D runs.
        render = {key: render[key] for key in ("azimuths", "elevations", "distance",
                                               "image_size", "shard_size")}
        return render, _VIEWACTIVE_RENDER_VERSION
    return render, "blender-glb-v1"


VIEWACTIVE_FINGERPRINTS = {
    "cohort": _VIEWACTIVE_DATA,
    "render": lambda c: [_VIEWACTIVE_DATA(c), _viewactive_render_settings(c)[0],
                         _viewactive_render_settings(c)[1]],
    "label": lambda c: [_VIEWACTIVE_DATA(c), _viewactive_render_settings(c)[0], c["labels"],
                        _viewactive_render_settings(c)[1]],
    "features": lambda c: [_VIEWACTIVE_DATA(c), _viewactive_render_settings(c)[0], c["encoders"]["dino"],
                           c["encoders"]["clip"], _viewactive_render_settings(c)[1]],
    "classify": lambda c: [_VIEWACTIVE_DATA(c), _viewactive_render_settings(c)[0], c["classifiers"],
                           _viewactive_render_settings(c)[1]]
                + ([c["classification"]] if c["classification"]["crop_margin"] is not None else []),
    "train": lambda c: [c["labels"], c["encoders"]["dino"], c["encoders"]["clip"], c["model"], c["train"]],
}


def _classes(value, name):
    if type(value) is int and value >= 1:
        return value
    if (isinstance(value, list) and value and all(isinstance(v, str) and v.strip() for v in value)
            and len(set(value)) == len(value)):
        return value
    raise ValueError(f"{name} must be a positive count or a list of distinct LVIS categories")


def _uniform(values, name, period=None):
    """Evenly spaced ascending angles; with ``period`` they must tile the full circle."""
    steps = {round(b - a, 6) for a, b in zip(values, values[1:])}
    if len(values) < 2 or values != sorted(values) or len(steps) != 1:
        raise ValueError(f"{name} must list at least two evenly spaced ascending angles")
    if period is not None and abs(len(values) * steps.pop() - period) > 1e-6:
        raise ValueError(f"{name} must cover the full circle")
    return values


def load_viewactive(path):
    """Return resolved ViewActive settings; relative paths resolve against the YAML file."""
    import yaml

    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("ViewActive config must be a YAML mapping")
    unknown = raw.keys() - {"output_dir", *VIEWACTIVE_DEFAULTS}
    if unknown:
        raise ValueError(f"Unknown ViewActive settings: {sorted(unknown)}")

    def resolve(value, name):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty path")
        return str((path.parent / Path(value).expanduser()).resolve())

    config = {"output_dir": resolve(raw.get("output_dir"), "output_dir"),
              "gpu_id": _integer(raw.get("gpu_id", 0), "gpu_id", 0)}
    section = {name: _merge(VIEWACTIVE_DEFAULTS[name], raw.get(name), name)
               for name in VIEWACTIVE_DEFAULTS if name not in ("gpu_id", "classifiers")}

    objaverse = section["objaverse"]
    for key in ("nas_root", "cache_root"):
        objaverse[key] = resolve(objaverse[key], f"objaverse.{key}")
    _number(objaverse["max_mb"], "objaverse.max_mb", 1e-3)
    _integer(objaverse["max_faces"], "objaverse.max_faces", 1)
    _integer(objaverse["download_workers"], "objaverse.download_workers", 1)
    if type(objaverse["trust_file_up"]) is not bool:
        raise ValueError("objaverse.trust_file_up must be a boolean")

    cohort = section["cohort"]
    cohort["class_map"] = resolve(cohort["class_map"], "cohort.class_map")
    cohort["seen_classes"] = _classes(cohort["seen_classes"], "cohort.seen_classes")
    cohort["unseen_classes"] = _classes(cohort["unseen_classes"], "cohort.unseen_classes")
    if (isinstance(cohort["seen_classes"], list) and isinstance(cohort["unseen_classes"], list)
            and set(cohort["seen_classes"]) & set(cohort["unseen_classes"])):
        raise ValueError("cohort.seen_classes and cohort.unseen_classes must be disjoint")
    _integer(cohort["per_class"], "cohort.per_class", 1)
    _integer(cohort["unseen_per_class"], "cohort.unseen_per_class", 1)
    if cohort["total_seen"] is not None:
        _integer(cohort["total_seen"], "cohort.total_seen", 1)
    _number(cohort["oversample"], "cohort.oversample", 1, 10)
    split = cohort["split"]
    if not isinstance(split, list) or len(split) != 3:
        raise ValueError("cohort.split must be [train, val, test] fractions")
    for value in split:
        _number(value, "cohort.split", 0, 1)
    if split[0] <= 0 or split[2] <= 0 or abs(sum(split) - 1) > 1e-6:
        raise ValueError("cohort.split must be [train, val, test] fractions summing to 1 "
                         "with positive train and test")
    if not isinstance(cohort["exclude_uids_from"], list):
        raise ValueError("cohort.exclude_uids_from must list canonical.json paths")
    cohort["exclude_uids_from"] = [resolve(p, "cohort.exclude_uids_from")
                                   for p in cohort["exclude_uids_from"]]
    _integer(cohort["seed"], "cohort.seed", 0)

    render = section["render"]
    render["azimuths"] = _uniform(_angles(render["azimuths"], "render.azimuths"),
                                  "render.azimuths", period=360)
    render["elevations"] = _uniform(_angles(render["elevations"], "render.elevations", elevation=True),
                                    "render.elevations")
    _number(render["distance"], "render.distance", 1e-6)
    _integer(render["image_size"], "render.image_size", 16)
    _integer(render["shard_size"], "render.shard_size", 1)
    if render["backend"] not in ("pytorch3d", "blender"):
        raise ValueError("render.backend must be pytorch3d or blender")
    for key in ("width", "height"):
        _integer(render[key], f"render.{key}", 16)
    _number(render["fov"], "render.fov", 1, 179)
    _integer(render["samples"], "render.samples", 1)
    render["blender_binary"] = resolve(render["blender_binary"], "render.blender_binary")

    labels = section["labels"]
    metric_options("self_occlusion", {"samples": labels["samples"], "seed": labels["seed"]})
    metric_options("surface_normal_entropy", {"polar_bins": labels["polar_bins"],
                                              "azimuth_bins": labels["azimuth_bins"]})
    metric_options("visual_entropy", {"bins": labels["visual_bins"]})
    _number(labels["floor"], "labels.floor", 1e-3, 0.5)
    _integer(labels["workers"], "labels.workers", 1)

    encoders = section["encoders"]
    for key in ("dino", "clip"):
        if not isinstance(encoders[key], str) or not encoders[key].strip():
            raise ValueError(f"encoders.{key} must name a Hugging Face model")
    _integer(encoders["batch"], "encoders.batch", 1)
    _integer(section["model"]["hidden"], "model.hidden", 1)

    train = section["train"]
    _integer(train["batch"], "train.batch", 1)
    _integer(train["epochs"], "train.epochs", 1)
    _number(train["lr"], "train.lr", 1e-8, 1)
    _number(train["weight_decay"], "train.weight_decay", 0, 1)
    _number(train["gamma"], "train.gamma", 1e-3, 1)
    _number(train["silog_lambda"], "train.silog_lambda", 0, 1)
    _integer(train["seed"], "train.seed", 0)
    weights = train["loss_weights"]
    if not isinstance(weights, list) or len(weights) != 3 or not all(
            _number(w, "train.loss_weights", 0) >= 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("train.loss_weights must be three nonnegative [L1, DSSIM, SILog] "
                         "weights with a positive sum")

    optimize = section["optimize"]
    _integer(optimize["steps"], "optimize.steps", 1)
    _integer(optimize["starts_per_object"], "optimize.starts_per_object", 1)
    _integer(optimize["seed"], "optimize.seed", 0)
    reach = optimize["reach_cells"]
    if reach is not None and (not isinstance(reach, list) or len(reach) != 2
                              or any(type(r) is not int or r < 0 for r in reach)):
        raise ValueError("optimize.reach_cells must be [azimuth, elevation] cell counts or null (unlimited)")
    combine = optimize["combine_weights"]
    if combine is not None and (not isinstance(combine, list) or len(combine) != 3 or not all(
            _number(w, "optimize.combine_weights", 0) >= 0 for w in combine) or sum(combine) <= 0):
        raise ValueError("optimize.combine_weights must be three nonnegative weights with a positive sum")
    limits = optimize["elevation_range"]
    if limits is not None:
        if not isinstance(limits, list) or len(limits) != 2:
            raise ValueError("optimize.elevation_range must be [low, high] degrees or null")
        low, high = (_number(v, "optimize.elevation_range", -90, 90) for v in limits)
        if low > high or not any(low <= e <= high for e in section["render"]["elevations"]):
            raise ValueError("optimize.elevation_range must contain at least one render elevation")

    margin = section["classification"]["crop_margin"]
    if margin is not None:
        _number(margin, "classification.crop_margin", 0, 2)

    classifiers = raw.get("classifiers") or VIEWACTIVE_DEFAULTS["classifiers"]
    if (not isinstance(classifiers, dict) or not classifiers
            or not all(isinstance(v, str) and v for v in classifiers.values())):
        raise ValueError("classifiers must map display names to timm model names")
    section["classifiers"] = classifiers
    return config | section


def read(settings_path):
    """Read the resolved settings a driver wrote for its workers."""
    return json.loads(Path(settings_path).read_text())
