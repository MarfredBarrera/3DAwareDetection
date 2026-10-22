"""View selectors: pick one SAM3D candidate per input image.

A selector returns weights over one input's candidates (summing to 1): a single
pick gets weight 1, ties share it, and ``random`` is the exact expectation over
a uniform pick. Selection reads stored scores only, so new selectors need no
GPU work.

- ``identity``: the candidate at ``SAM3D_INPUT_VIEW``, i.e. the reconstruction
  re-rendered from the input view. It separates the effect of re-rendering
  from the effect of choosing a new viewpoint.
- ``random``: a uniformly random candidate.
- ``ambiguity:<metric>``: the least ambiguous candidate under one metric.
- ``ambiguity:combined``: weighted mean of within-input metric ranks; equal
  weights unless ``analysis.combined_weights`` is set. (Weights fitted on
  ground-truth views were ~1/3 each and not significantly better than equal.)
- ``detector_entropy``: the candidate whose SAM3 category scores, normalized to
  a distribution over the vocabulary, have the lowest entropy. Label-free.
- ``detector_gap``: the candidate with the largest gap between SAM3's top two
  category scores. Label-free; equals the oracle margin when top-1 is correct.
- ``oracle``: the candidate with the best SAM3 margin (an upper bound; uses labels).

Ground-truth-view selectors pick another *ground-truth* view of the input's
object instead of a SAM3D candidate (``analysis.gt_vqf``; needs scored
ground-truth views):

- ``gt_vqf``: ViewActive Algorithm 1 from the input view on the object's
  ground-truth VQF (per metric, quality = 1 - ambiguity min-max normalized over
  the object's views; equal mean), within the configured reach, steps and
  elevation range.
- ``gt_random``: a uniformly random ground-truth view in the elevation range.
- ``gt_oracle``: the ground-truth view in the elevation range with the best
  SAM3 margin (uses labels).

Pick selectors run before detection (``picks`` config, ``pick`` stage), so SAM3
can be run on the chosen candidates only:

- ``gt_vqf_max``: the SAM3D candidate nearest in angle to the maximum of the
  object's ground-truth VQF (equal-mean channels) within ``elevation_range``.
  Candidate angles are mapped to the object frame with the input's known pose
  (``viewactive.grid.sam3d_absolute``).
- ``identity`` and ``random`` (one seeded candidate) as controls.
"""
import numpy as np

from tools.cameras import SAM3D_INPUT_VIEW

GT_SELECTORS = ("gt_vqf", "gt_random", "gt_oracle")
PICK_SELECTORS = ("gt_vqf_max", "identity", "random")


def picks(source, candidates, field, grid, limits, seed):
    """{selector: (candidate image_id, meta)} for one input before detection.

    ``source`` and ``candidates`` are image rows; ``field`` is the object's
    combined ground-truth VQF ``[E, A]`` on ``grid`` (None skips ``gt_vqf_max``).
    """
    from tools.cameras import direction
    from viewactive.grid import elevation_rows, sam3d_absolute

    if not candidates:
        return {}
    chosen = {}
    rng = np.random.default_rng([seed, source["image_id"]])
    pick = candidates[int(rng.integers(len(candidates)))]
    chosen["random"] = (pick["image_id"], {})
    identity = [c for c in candidates if (c["azimuth"], c["elevation"]) == SAM3D_INPUT_VIEW]
    if identity:
        chosen["identity"] = (identity[0]["image_id"], {})
    if field is not None:
        allowed = elevation_rows(grid, limits)[:, None] & np.isfinite(field)
        if allowed.any():
            e, a = map(int, np.unravel_index(np.argmax(np.where(allowed, field, -np.inf)), field.shape))
            target = (grid.azimuths[a], grid.elevations[e])
            goal = direction(*target)
            angles = [sam3d_absolute(c["azimuth"], c["elevation"], source["azimuth"], source["elevation"])
                      for c in candidates]
            errors = [float(np.degrees(np.arccos(np.clip(direction(*angle) @ goal, -1, 1))))
                      for angle in angles]
            best = int(np.argmin(errors))
            chosen["gt_vqf_max"] = (candidates[best]["image_id"], {
                "target": list(target), "target_vqf": float(field[e, a]),
                "absolute": [float(v) for v in angles[best]], "error_deg": errors[best]})
    return chosen


def selector_names(metrics):
    names = ["identity", "random", *(f"ambiguity:{name}" for name in metrics)]
    if len(metrics) > 1:
        names.append("ambiguity:combined")
    return names + ["detector_entropy", "detector_gap", "oracle"]


def detector_entropy(prompt_scores):
    """Entropy of SAM3 scores normalized over the vocabulary, in [0, 1]; None if all zero."""
    values = np.clip(np.asarray(list(prompt_scores.values()), dtype=float), 0, None)
    if len(values) < 2 or values.sum() <= 0:
        return None
    p = values / values.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(values)))


def _pick(ids, values, best=min):
    """Weights for the best finite value; ties split evenly. None if nothing is defined."""
    finite = {i: v for i, v in zip(ids, values) if v is not None and np.isfinite(v)}
    if not finite:
        return None
    target = best(finite.values())
    chosen = [i for i, v in finite.items() if abs(v - target) <= 1e-12]
    return {i: 1 / len(chosen) for i in chosen}


def _ranks(values):
    """Ranks in [0, 1] (average ties); undefined values rank as most ambiguous."""
    values = np.array([np.inf if v is None else v for v in values], dtype=float)
    if len(values) == 1:
        return np.zeros(1)
    order = values.argsort(kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    for value in np.unique(values):
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks / (len(values) - 1)


def detector_gap(prompt_scores):
    """SAM3's top category score minus its second; None with fewer than two scores."""
    if len(prompt_scores) < 2:
        return None
    first, second = sorted(prompt_scores.values(), reverse=True)[:2]
    return float(first - second)


def weights(name, candidate_ids, scores, margins, metrics, angles, prompt_scores=None,
            metric_weights=None):
    """Weights over ``candidate_ids``.

    ``scores`` = {id: {metric: value}}, ``margins`` = {id: margin},
    ``angles`` = {id: (azimuth, elevation)} in the reconstruction frame,
    ``prompt_scores`` = {id: {category: SAM3 score}} (for the ``detector_*`` selectors),
    ``metric_weights`` = {metric: weight} for ``ambiguity:combined`` (None = equal).
    """
    ids = list(candidate_ids)
    if not ids:
        return None
    if name == "random":
        return {i: 1 / len(ids) for i in ids}
    if name == "oracle":
        return _pick(ids, [margins[i] for i in ids], best=max)
    if name == "detector_entropy":
        return _pick(ids, [detector_entropy(prompt_scores[i]) for i in ids])
    if name == "detector_gap":
        return _pick(ids, [detector_gap(prompt_scores[i]) for i in ids], best=max)
    if name == "identity":
        return _pick(ids, [1.0 if angles[i] == SAM3D_INPUT_VIEW else None for i in ids], best=max)
    if name == "ambiguity:combined":
        mix = [1.0 if metric_weights is None else metric_weights.get(m, 0.0) for m in metrics]
        combined = np.average([_ranks([scores.get(i, {}).get(metric) for i in ids])
                               for metric in metrics], axis=0, weights=mix)
        return _pick(ids, combined.tolist())
    if name.startswith("ambiguity:") and name.removeprefix("ambiguity:") in metrics:
        return _pick(ids, [scores.get(i, {}).get(name.removeprefix("ambiguity:")) for i in ids])
    raise ValueError(f"Unknown selector {name!r}; choices: {selector_names(metrics)}")


def gt_vqf_field(views, scores, metrics, grid):
    """Ground-truth VQF ``[E, A]`` of one object; ``views`` = {(e, a): image_id}."""
    from viewactive.grid import combine
    from viewactive.labels import normalize

    raw = np.full((len(metrics), *grid.shape), np.nan)
    for (e, a), image_id in views.items():
        for m, metric in enumerate(metrics):
            value = scores.get(image_id, {}).get(metric)
            if value is not None:
                raw[m, e, a] = 1 - value
    return combine(normalize(raw))


def gt_view_weights(name, start, views, field, margins, grid, settings):
    """Weights over one object's ground-truth views for the input at grid cell ``start``.

    ``views`` = {(e, a): image_id}, ``field`` from ``gt_vqf_field``, ``margins`` =
    {image_id: margin}, ``settings`` = ``analysis.gt_vqf``.
    """
    from viewactive.grid import elevation_rows, to_relative
    from viewactive.optimize import field_policy, rollout

    limits = settings["elevation_range"]
    allowed = elevation_rows(grid, limits)
    ids = [i for (e, a), i in views.items() if allowed[e] and i in margins]
    if not ids:
        return None
    if name == "gt_random":
        return {i: 1 / len(ids) for i in ids}
    if name == "gt_oracle":
        return _pick(ids, [margins[i] for i in ids], best=max)
    if name == "gt_vqf":
        reach = settings["reach_cells"]
        policy = field_policy(lambda e, a: to_relative(field, a), grid,
                              None if reach is None else tuple(reach), limits)
        final = rollout(policy, start, settings["steps"])[-1]
        return {views[final]: 1.0} if views.get(final) in margins else None
    raise ValueError(f"Unknown ground-truth selector {name!r}; choices: {GT_SELECTORS}")
