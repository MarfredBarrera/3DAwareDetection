"""View selection and the accept/reject decision, from stored scores only (no models).

Keys of ``views`` are SAM3D candidate angles (azimuth, elevation) relative to the
input camera; (180, 0) re-renders the input view.

- ``top2gap``: the candidate whose SAM3 top-1 category score beats its second by
  the most. Label-free; needs SAM3 on every candidate.
- ``vqf``: the candidate with the highest learned ViewActive viewpoint quality.
  Needs SAM3 on the chosen candidate only.
- ``input``: no SAM3D view; SAM3 judges the masked detection on the raw frame
  (the 2D control).
- ``qwen_grid``: no view is picked; a frozen uncertainty gate (``perception.qwen``) sends
  unsure raw-frame decisions to Qwen3-VL with the full SAM3D grid (three-way verdict).
- ``d1``: sufficiency first; a frozen 2-signal gate (raw margin, mask area) lets SAM3's raw
  top-1 decide sufficient frames and sends the rest to Qwen3-VL (grid or frame alone).
- ``jev_sam3``: JEV-27B-VL names the outlined object and a frozen 2-signal gate on its naming
  distribution (``perception.jev``) decides only the uncertainty (unsure -> approach); SAM3 makes every
  decision: its raw label on a sure frame, ``fuse_views`` over the close views (``aggregate_views``: d1).

``unsure``: whether a raw-frame accept is close enough to call to need a SAM3D view.

Both pick only among candidates whose absolute elevation (gravity frame) is in
``elevation_range`` (default ``VIEW_ELEVATIONS``: at or above the object, never from
below the floor plane), if any are. ViewActive was trained on 0-45, so VQF
qualities above 45 are extrapolated.
"""

SELECTORS = ("vqf", "top2gap", "input", "naive", "qwen_grid", "d1", "jev_sam3")
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
FLAT = ("tv", "picture", "painting", "poster", "mirror", "window", "whiteboard", "rug", "carpet", "calendar",
        "blinds")
# Open-vocabulary names (HM3D-OVON) that denote the same object as a vocabulary entry: a
# target never competes with its own synonyms or broader names (an OVON "pillow" against
# the vocabulary's "cushion"). Symmetric; chosen from category semantics, not from results.
ALIASES = {
    "cushion": ("pillow",),
    "picture": ("photo", "photo mount"),
    "potted plant": ("flowerpot", "decorative plant"),
    "chest of drawers": ("dresser", "nightstand"),
    "cabinet": ("nightstand",),
    "table": ("nightstand",),
    "counter": ("island",),
    "window": ("window glass",),
    "shower": ("shower glass",),
    "stool": ("ottoman", "footrest", "footstool"),
    "bed": ("bedframe",),
    "exercise equipment": ("exercise bike",),
    "appliance": ("refrigerator", "freezer", "microwave", "dishwasher", "coffee machine", "boiler"),
}
# Categories a target is easily mistaken for (and vice versa). A raw-frame accept whose
# runner-up is one of these is unsure. Chosen from category semantics, not from results.
CONFUSABLE = {"chair": ("stool", "couch", "exercise equipment"), "couch": ("chair", "bed"),
              "bed": ("couch",), "toilet": ("sink", "bathtub")}
# A raw-frame accept with a smaller top-1 minus top-2 gap is unsure
# (outputs/apexnav_offline/defer_replay_2026-10-26: 0.2-0.5 behave alike).
UNSURE_MARGIN = 0.3


def same_object(a, b):
    """True when names ``a`` and ``b`` denote the same object: equal, one is the other's
    head noun as whole trailing words ("potted plant" and "plant", not "tv stand" and
    "tv"), or an alias (``ALIASES``). Case-insensitive."""
    a, b = a.lower().strip(), b.lower().strip()
    return (a == b or a.endswith(" " + b) or b.endswith(" " + a)
            or b in ALIASES.get(a, ()) or a in ALIASES.get(b, ()))


def competitors(vocabulary, target):
    """``vocabulary`` (with ``target``) without the target's parts (``PARTS``) and
    other names of the same object (``same_object``)."""
    return [c for c in dict.fromkeys([*vocabulary, target])
            if c == target or (c not in PARTS.get(target, ()) and not same_object(c, target))]


def flat(target):
    """True for flat targets (``FLAT``, or another name of one), judged on the raw frame only."""
    return any(same_object(f, target) for f in FLAT)


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


def view_flip(view_scores, elevations, raw_label, elevation_range=VIEW_ELEVATIONS, min_views=6):
    """Fraction of in-band candidates whose SAM3 top-1 differs from the raw frame's
    (``raw_label``); None with fewer than ``min_views`` classified candidates."""
    if len(view_scores) < min_views:
        return None
    tops = [top1(view_scores[key])[0] for key in in_band({k: elevations[k] for k in view_scores}, elevation_range)]
    return sum(t != raw_label for t in tops) / len(tops)


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


# aggregate_views: a view whose mask covers this fraction of the frame counts fully; smaller views
# count proportionally less (gate evidence.FULL_VIEW_AREA_FRACTION), and a mask running off the
# frame counts CLIPPED_WEIGHT as much.
FULL_VIEW_AREA = 0.10
CLIPPED_WEIGHT = 0.7


# fuse_views: SAM3 score -> evidence scale (offline 2026-11-07: flat for 1-10; docs/selfcontained_view_planners.md 6.4)
FUSE_SCALE = 2.0


def fuse_views(per_view, target, scale=FUSE_SCALE):
    """SAM3 over several views of one object, pooled as evidential opinions (Fan et al., "Evidential Active
    Recognition", CVPR 2024): selector jev_sam3's multi-view decision. ``per_view`` as ``aggregate_views``.

    Per view, evidence e_k = ``scale`` x score_k over its K categories, S = sum e + K, belief b_k = e_k / S and
    uncertainty u = K / S (a view where SAM3 scores everything low is mostly uncertainty and counts little).
    Views are combined pairwise (approximate Dempster-Shafer, order-free):
        b = (b1 b2 + b1 u2 + b2 u1) / (1 - c),  u = u1 u2 / (1 - c),  c = sum_{i != j} b1_i b2_j.
    At this scale it is close to the per-category score summed over the views.
    Views where SAM3 matched nothing are left out (as ``aggregate_views``).

    Returns {accept, label, belief, uncertainty, n_used}: accept iff the target has the highest fused belief;
    label = the fused top category; belief = {category: b}. Without a usable view: accept False, the rest None
    and n_used 0."""
    used = [v for v in per_view if v.get("scores") and any(s > 0 for s in v["scores"].values())]
    if not used:
        return {"accept": False, "label": None, "belief": None, "uncertainty": None, "n_used": 0}
    categories = sorted({c for v in used for c in v["scores"]})
    K = len(categories)
    b = u = None
    for view in used:
        e = [scale * max(float(view["scores"].get(c, 0.0)), 0.0) for c in categories]
        S = sum(e) + K
        b2, u2 = [x / S for x in e], K / S
        if b is None:
            b, u = b2, u2
            continue
        conflict = sum(b) * sum(b2) - sum(x * y for x, y in zip(b, b2))
        z = max(1.0 - conflict, 1e-12)
        b, u = [(x * y + x * u2 + y * u) / z for x, y in zip(b, b2)], u * u2 / z
    belief = dict(zip(categories, b))
    label = max(categories, key=lambda c: (belief[c], c == target))   # a tie goes to the target
    return {"accept": label == target, "label": label, "belief": {c: round(v, 6) for c, v in belief.items()},
            "uncertainty": round(u, 6), "n_used": len(used)}


def aggregate_views(per_view, target):
    """SAM3 over several views of one object (selector d1, multi-view): ``per_view`` is
    [{scores, area, clipped}] (``scores``: SAM3 score per category on the masked view).
    Views where SAM3 matched nothing (no scores, or all 0: the IoU gate missed) are left out.

    Returns {margin, vote, n_used, label}: margin = sum_k w_k m_k / sum_k w_k with
    m_k = s_k[target] - max over the other categories and w_k = min(1, area_k / FULL_VIEW_AREA)
    (times CLIPPED_WEIGHT if clipped); vote = fraction of used views whose top-1 is the target;
    label = target if margin > 0, else the category with the highest summed score.
    Without a usable view: margin, vote and label None, n_used 0."""
    used = [v for v in per_view if v.get("scores") and any(s > 0 for s in v["scores"].values())]
    if not used:
        return {"margin": None, "vote": None, "n_used": 0, "label": None}
    total = weighted = 0.0
    votes, summed = 0, {}
    for view in used:
        scores = view["scores"]
        others = [s for c, s in scores.items() if c != target]
        margin = scores.get(target, 0.0) - (max(others) if others else 0.0)
        weight = min(1.0, float(view["area"]) / FULL_VIEW_AREA) * (CLIPPED_WEIGHT if view.get("clipped") else 1.0)
        total, weighted = total + weight, weighted + weight * margin
        votes += top1(scores)[0] == target
        for category, score in scores.items():
            summed[category] = summed.get(category, 0.0) + score
    margin = weighted / total if total > 0 else None
    label = target if margin is not None and margin > 0 else ranked(summed)[0][0]
    return {"margin": margin, "vote": votes / len(used), "n_used": len(used), "label": label}
