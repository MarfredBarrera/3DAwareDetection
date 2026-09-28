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
"""
import numpy as np

from tools.cameras import SAM3D_INPUT_VIEW


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
