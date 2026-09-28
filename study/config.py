"""Load and validate a study YAML; relative paths resolve against the YAML file."""
import hashlib
import json
import math
from pathlib import Path

from study.ambiguity import metric_options

DEFAULTS = {
    "gpu_id": 0,
    "checkpoints": {"sam3": "/data/checkpoints/sam3/sam3.pt",
                    "sam3d": "/data/checkpoints/sam3d/hf"},
    "dataset": {"root": "/data/objaverse_nas", "canonical": "/data/objaverse_nas/canonical.json",
                "require_upright_verified": False, "categories": None,
                "objects_per_category": 10,
                "azimuths": [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
                "elevations": [0, 15, 30, 45, 60], "distance": 2.2, "image_size": 320},
    "detection": {"score_threshold": 0.5, "iou_threshold": 0.5},
    "reconstruction": {"objects_per_category": 10, "input_elevations": None,
                       "candidate_azimuths": [0, 45, 90, 135, 180, 225, 270, 315],
                       "candidate_elevations": [0, 30, 60], "candidate_distance": 2.0,
                       "resolution": 320, "seed": 42},
    "ambiguity": {"visual_entropy": {}, "self_occlusion": {}, "surface_normal_entropy": {}},
    "analysis": {"primary_selector": "ambiguity:combined", "combined_weights": None},
}
# Settings each stage's stored outputs depend on. Thresholds, input selection,
# and analysis settings are applied later and may change freely.
FINGERPRINTS = {
    "render": lambda c: c["dataset"],
    "detect": lambda c: [c["dataset"]["categories"], c["checkpoints"]["sam3"]],
    "reconstruct": lambda c: [c["checkpoints"]["sam3d"]] + [
        c["reconstruction"][k] for k in ("candidate_azimuths", "candidate_elevations",
                                         "candidate_distance", "resolution", "seed")],
    "score": lambda c: c["ambiguity"],
}


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
    config["reconstruction"] = recon

    metrics = raw.get("ambiguity", DEFAULTS["ambiguity"])
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("ambiguity must map at least one metric name to its options")
    config["ambiguity"] = {name: metric_options(name, options or {}) for name, options in metrics.items()}

    analysis = _merge(DEFAULTS["analysis"], raw.get("analysis"), "analysis")
    from study.selectors import selector_names
    if analysis["primary_selector"] not in selector_names(config["ambiguity"]):
        raise ValueError(f"analysis.primary_selector must be one of {selector_names(config['ambiguity'])}")
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


def fingerprint(config, stage):
    """Hash of the settings a stage's stored outputs depend on."""
    relevant = FINGERPRINTS[stage](config)
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]


def read(settings_path):
    """Read the resolved settings a driver wrote for its workers."""
    return json.loads(Path(settings_path).read_text())
