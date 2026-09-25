"""Interchangeable per-frame viewpoint metrics.

Visual entropy follows Eq. (3) of ViewActive (Wu et al., arXiv:2409.09997):
Shannon entropy of the object's grayscale histogram. The paper treats larger
entropy as better viewpoint quality; we invert its bin-normalized value so
all saved ambiguity scores use 0 = least ambiguous, 1 = most ambiguous.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def visual_entropy(image, mask, *, bins=32):
    """Return ambiguity, entropy in bits, and visible-pixel count for one view."""
    if type(bins) is not int or not 2 <= bins <= 256:
        raise ValueError("visual_entropy bins must be an integer in [2, 256]")
    grayscale = np.asarray(image.convert("L"))
    foreground = np.asarray(mask.convert("L")) > 0
    if grayscale.shape != foreground.shape:
        raise ValueError("Rendered image and mask must have the same size")
    count = int(foreground.sum())
    if count == 0:
        return {"ambiguity": None, "entropy_bits": None, "foreground_pixels": 0}
    histogram = np.histogram(grayscale[foreground], bins=bins, range=(0, 256))[0]
    probabilities = histogram[histogram > 0] / count
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    ambiguity = float(np.clip(1 - entropy / math.log2(bins), 0, 1))
    return {"ambiguity": ambiguity, "entropy_bits": entropy, "foreground_pixels": count}


@dataclass(frozen=True)
class Metric:
    score: object
    defaults: dict
    validate: object
    prepare: object


def _visual_view(output_dir, candidate, **options):
    if "mask" not in candidate:
        raise ValueError("Candidate masks are missing; rerender this reconstruction")
    with Image.open(output_dir / candidate["image"]) as image, Image.open(output_dir / candidate["mask"]) as mask:
        return visual_entropy(image, mask, **options)


def _entropy_options(options):
    bins = options["bins"]
    if type(bins) is not int or not 2 <= bins <= 256:
        raise ValueError("visual_entropy bins must be an integer in [2, 256]")


def _mesh_options(options):
    if type(options["samples"]) is not int or options["samples"] < 100:
        raise ValueError("mesh metric samples must be an integer >= 100")
    if type(options["seed"]) is not int or options["seed"] < 0:
        raise ValueError("mesh metric seed must be a nonnegative integer")


def _normal_options(options):
    _mesh_options(options)
    for name in ("polar_bins", "azimuth_bins"):
        if type(options[name]) is not int or not 2 <= options[name] <= 180:
            raise ValueError(f"{name} must be an integer in [2, 180]")


def _prepare_mesh(output_dir, manifest, options):
    from .mesh_metrics import prepare_mesh
    return prepare_mesh(output_dir, manifest, options)


def _occlusion(context, candidate, **options):
    from .mesh_metrics import self_occlusion
    return self_occlusion(context, candidate, **options)


def _normal_entropy(context, candidate, **options):
    from .mesh_metrics import surface_normal_entropy
    return surface_normal_entropy(context, candidate, **options)


METRICS = {
    "visual_entropy": Metric(_visual_view, {"bins": 32}, _entropy_options,
                             lambda output_dir, manifest, options: output_dir),
    "self_occlusion": Metric(_occlusion, {"samples": 20000, "seed": 0},
                             _mesh_options, _prepare_mesh),
    "surface_normal_entropy": Metric(
        _normal_entropy, {"samples": 20000, "seed": 0, "polar_bins": 9, "azimuth_bins": 18},
        _normal_options, _prepare_mesh),
}


def metric_options(name, supplied=None):
    """Resolve and validate options owned by a registered metric."""
    if not isinstance(name, str) or name not in METRICS:
        raise ValueError(f"Unknown metric {name!r}; choices: {sorted(METRICS)}")
    supplied = {} if supplied is None else supplied
    if not isinstance(supplied, dict) or supplied.keys() - METRICS[name].defaults.keys():
        raise ValueError(f"Invalid options for {name}; allowed: {sorted(METRICS[name].defaults)}")
    options = METRICS[name].defaults | supplied
    METRICS[name].validate(options)
    return options


def score_views(output_dir, metric="visual_entropy", *, options=None):
    """Score every candidate using the selected registered metric; save JSON."""
    options = metric_options(metric, options)
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "candidates.json").read_text())
    scorer = METRICS[metric]
    context = scorer.prepare(output_dir, manifest, options)
    scores = {}
    for candidate in manifest["candidates"]:
        scores[Path(candidate["image"]).stem] = scorer.score(context, candidate, **options)
    result = {
        "metric": metric, "options": options, "score_name": "ambiguity",
        "lower_is_better": True, "scores": scores,
    }
    path = output_dir / f"scores_{metric}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path
