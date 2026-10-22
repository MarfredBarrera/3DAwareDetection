"""View selection and the accept/reject decision, from stored scores only (no models).

Keys of ``views`` are SAM3D candidate angles (azimuth, elevation) relative to the
input camera; (180, 0) re-renders the input view.

- ``top2gap``: the candidate whose SAM3 top-1 category score beats its second by
  the most. Label-free; needs SAM3 on every candidate.
- ``vqf``: the candidate with the highest learned ViewActive viewpoint quality,
  restricted to absolute elevations in ``elevation_range`` (the trained policy's
  range). Needs SAM3 on the chosen candidate only.
"""

SELECTORS = ("vqf", "top2gap")
# What to do when the ensemble filter distrusts a reconstruction:
# "input" classifies the raw frame instead, "reject" abstains (rejects the
# detection), "accept" keeps the detection unverified.
ON_UNCERTAIN = ("input", "reject", "accept")
VQF_ELEVATIONS = (0.0, 45.0)


def ranked(scores):
    """[(category, score)] from highest to lowest; ties keep vocabulary order."""
    return sorted(scores.items(), key=lambda item: -item[1])


def top1(scores):
    """(top-1 category, its score minus the second score); gap is 0 for one category."""
    order = ranked(scores)
    return order[0][0], order[0][1] - (order[1][1] if len(order) > 1 else 0.0)


def top2gap_pick(view_scores):
    """The candidate key with the largest top-1 minus top-2 gap; first wins ties."""
    gaps = {key: top1(scores)[1] for key, scores in view_scores.items()}
    return max(gaps, key=gaps.get)


def vqf_pick(quality, elevation_range=VQF_ELEVATIONS):
    """The candidate key with the best quality; ``quality`` = {key: (quality, absolute
    elevation)}. Only candidates inside ``elevation_range`` compete, if any are."""
    low, high = elevation_range
    allowed = [key for key, (_, elevation) in quality.items() if low <= elevation <= high]
    return max(allowed or quality, key=lambda key: quality[key][0])


def trusted(uncertainty, threshold):
    """The ensemble filter: True when the reconstruction may be used. ``threshold``
    None disables the filter; an undefined uncertainty (empty shapes) is not trusted."""
    return threshold is None or (uncertainty is not None and uncertainty <= threshold)
