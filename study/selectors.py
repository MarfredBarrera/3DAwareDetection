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
- ``top2gap``: the candidate with the largest gap between SAM3's top two
  category scores (``perception.select``). Label-free.
- ``vqf``: the candidate with the best learned ViewActive quality, stored by
  the ``pick`` stage (``picks`` config) before detection.
- ``oracle``: the candidate with the best SAM3 margin (an upper bound; uses labels).
"""
import numpy as np

from perception.select import top1
from tools.cameras import SAM3D_INPUT_VIEW


def selector_names(metrics, vqf=False):
    names = ["identity", "random", *(f"ambiguity:{name}" for name in metrics)]
    if len(metrics) > 1:
        names.append("ambiguity:combined")
    return names + ["top2gap"] + (["vqf"] if vqf else []) + ["oracle"]


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


def weights(name, candidate_ids, scores, margins, metrics, angles, prompt_scores=None,
            metric_weights=None, vqf_pick=None):
    """Weights over ``candidate_ids``.

    ``scores`` = {id: {metric: value}}, ``margins`` = {id: margin},
    ``angles`` = {id: (azimuth, elevation)} in the reconstruction frame,
    ``prompt_scores`` = {id: {category: SAM3 score}} (for ``top2gap``),
    ``metric_weights`` = {metric: weight} for ``ambiguity:combined`` (None = equal),
    ``vqf_pick`` = the stored ``vqf`` candidate id.
    """
    ids = list(candidate_ids)
    if not ids:
        return None
    if name == "random":
        return {i: 1 / len(ids) for i in ids}
    if name == "oracle":
        return _pick(ids, [margins[i] for i in ids], best=max)
    if name == "top2gap":
        return _pick(ids, [top1(prompt_scores[i])[1] if len(prompt_scores[i]) > 1 else None
                           for i in ids], best=max)
    if name == "vqf":
        return {vqf_pick: 1.0} if vqf_pick in ids else None
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
