"""View selection and the accept/reject decision, from stored scores only (no models).

Keys of ``views`` are SAM3D candidate angles (azimuth, elevation) relative to the
input camera; (180, 0) re-renders the input view.

- ``top2gap``: the candidate whose SAM3 top-1 category score beats its second by
  the most. Label-free; needs SAM3 on every candidate.
- ``vqf``: the candidate with the highest learned ViewActive viewpoint quality.
  Needs SAM3 on the chosen candidate only.
- ``input``: no SAM3D view; SAM3 judges the masked detection on the raw frame
  (the 2D control).

``unsure``: whether a raw-frame accept is close enough to call to need a SAM3D view.

Both pick only among candidates whose absolute elevation (gravity frame) is in
``elevation_range`` (default ``VIEW_ELEVATIONS``: at or above the object, never from
below the floor plane), if any are. ViewActive was trained on 0-45, so VQF
qualities above 45 are extrapolated.
"""

SELECTORS = ("vqf", "top2gap", "input", "naive")
# "naive": the target prompt alone on the raw frame, accepted iff its score >= NAIVE_THRESHOLD
# (SAM3's default detection threshold); the baseline for the closed-vocabulary "input".
NAIVE_THRESHOLD = 0.5
# What to do when the ensemble filter distrusts a reconstruction:
# "input" classifies the raw frame instead, "reject" abstains (rejects the
# detection), "accept" keeps the detection unverified.
ON_UNCERTAIN = ("input", "reject", "accept")
VIEW_ELEVATIONS = (0.0, 90.0)
# Vocabulary entries that are parts or accessories of a target: a view of the target
# may legitimately look most like one (SAM3 calls isolated couch renders "cushion"),
# so they do not compete with it. Chosen from category semantics, not from results.
PARTS = {"couch": ("cushion",), "chair": ("cushion",), "bed": ("cushion",)}
# Targets that are essentially flat: their category is in surface appearance and
# context, not 3D shape, so a SAM3D view cannot help (a framed picture and a TV are the
# same thin rectangle). They are judged on the raw frame only.
FLAT = ("tv", "picture", "painting", "poster", "mirror", "window", "whiteboard")
# Categories a target is easily mistaken for (and vice versa). A raw-frame accept whose
# runner-up is one of these is unsure. Chosen from category semantics, not from results.
CONFUSABLE = {"chair": ("stool", "couch", "exercise equipment"), "couch": ("chair", "bed"),
              "bed": ("couch",), "toilet": ("sink", "bathtub")}
# A raw-frame accept with a smaller top-1 minus top-2 gap is unsure
# (outputs/apexnav_offline/defer_replay_2026-10-26: 0.2-0.5 behave alike).
UNSURE_MARGIN = 0.3


def competitors(vocabulary, target):
    """``vocabulary`` (with ``target``) without the target's parts (``PARTS``)."""
    return [c for c in dict.fromkeys([*vocabulary, target]) if c not in PARTS.get(target, ())]


def ranked(scores):
    """[(category, score)] from highest to lowest; ties keep vocabulary order."""
    return sorted(scores.items(), key=lambda item: -item[1])


def top1(scores):
    """(top-1 category, its score minus the second score); gap is 0 for one category."""
    order = ranked(scores)
    return order[0][0], order[0][1] - (order[1][1] if len(order) > 1 else 0.0)


def unsure(scores, target, margin=UNSURE_MARGIN):
    """True when the target is top-1 in ``scores`` but by less than ``margin``, or the
    runner-up is one of its confusables (``CONFUSABLE``)."""
    order = ranked(scores)
    if len(order) < 2:
        return False
    return order[0][1] - order[1][1] < margin or order[1][0] in CONFUSABLE.get(target, ())


def in_band(elevations, elevation_range=VIEW_ELEVATIONS):
    """Keys of ``elevations`` ({key: absolute elevation}) inside ``elevation_range``;
    every key when none is (a candidate is always picked)."""
    low, high = elevation_range
    return [key for key, elevation in elevations.items() if low <= elevation <= high] or list(elevations)


def top2gap_pick(view_scores, elevations, elevation_range=VIEW_ELEVATIONS):
    """The candidate key with the largest top-1 minus top-2 gap among those in the
    elevation band; ``elevations`` = {key: absolute elevation}. First wins ties."""
    allowed = in_band({key: elevations[key] for key in view_scores}, elevation_range)
    gaps = {key: top1(view_scores[key])[1] for key in allowed}
    return max(gaps, key=gaps.get)


def vqf_pick(quality, elevation_range=VIEW_ELEVATIONS):
    """The candidate key with the best quality in the elevation band; ``quality`` =
    {key: (quality, absolute elevation)}."""
    allowed = in_band({key: elevation for key, (_, elevation) in quality.items()}, elevation_range)
    return max(allowed, key=lambda key: quality[key][0])


def trusted(uncertainty, threshold):
    """The ensemble filter: True when the reconstruction may be used. ``threshold``
    None disables the filter; an undefined uncertainty (empty shapes) is not trusted."""
    return threshold is None or (uncertainty is not None and uncertainty <= threshold)
