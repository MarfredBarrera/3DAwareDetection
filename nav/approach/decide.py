"""Decision flow of the approach (formal flow 2026-11-07, size opinion 2026-11-10), one frame at a time.

One tracked candidate. Each frame facing the object is named by JEV over the approach words (POST /name; SAM3's
scores on the segment when it has >= 400 px). A frame is usable when it is the start frame or judgeable and carries
SAM3 scores; it counts as evidence only if it is a new viewpoint (pose diversity gate: DIST_M closer than, or
ANGLE_DEG around, every counted frame). Each counted frame updates a Dirichlet belief over JEV's naming options
(``Belief``: uniform prior of mass 2, evidence = the folded naming distribution). Once N_MIN frames are counted and
P(consistent) >= P_STAR, SAM3 verifies the counted views (``verify``): the fused evidential belief over the 6
nearest views without tiny ones (REL_AREA) decides, multiplied by the metric-size opinion (size_opinion; a veto
only). The belief only says WHEN to verify; SAM3 decides accept / reject.
"""
import math

import numpy as np

from nav.approach import size_opinion as SZ
from perception import jev as J, select as SEL

MAX_VIEWS = 6
PRIOR_MASS = 2.0
DEFAULTS = {"DIST_M": 0.5, "ANGLE_DEG": 20.0, "N_MIN": 2, "P_STAR": 0.8, "REL_AREA": 0.1,
            # the metric-size opinion multiplied into the fused SAM3 belief when the object's width is complete and the
            # target is in size_opinion.SIZE_TARGETS; a veto only (it can lower the target, never raise it)
            "SIZE": True}


class Belief:
    """Dirichlet belief over JEV's naming options: alpha = uniform prior (mass PRIOR_MASS) + the counted frames'
    folded naming distributions."""

    def __init__(self, target):
        self.target, self.options, self.alpha0, self.evidence = target, None, None, []

    def add(self, frame):
        folded = J.fold(frame["jev"]["probabilities"], self.target)
        if self.options is None:
            self.options = list(folded)
            self.alpha0 = np.full(len(self.options), PRIOR_MASS / len(self.options))
        self.evidence.append(np.array([folded[o] for o in self.options]))

    @property
    def alpha(self):
        return self.alpha0 + sum(self.evidence)


def usable(frame):
    """The start frame, or a judgeable frame whose segment SAM3 scored."""
    if frame["step"] == 0:
        return True
    return frame["pose"]["judgeable"] and bool((frame["jev"].get("sam3") or {}).get("scores"))


def same_view(frame, other, P):
    """The pose gate's duplicate test: not DIST_M closer and within ANGLE_DEG of ``other``."""
    db = abs((frame["pose"]["bearing"] - other["pose"]["bearing"] + math.pi) % (2 * math.pi) - math.pi)
    return not (frame["pose"]["distance"] <= other["pose"]["distance"] - P["DIST_M"]
                or math.degrees(db) >= P["ANGLE_DEG"])


def admit(frame, counted, P):
    """The diversity gate: (new counted list, changed). Only new viewpoints count."""
    if any(same_view(frame, g, P) for g in counted):
        return counted, False
    return counted + [frame], True


def build_belief(counted, target):
    belief = Belief(target)
    for g in counted:
        belief.add(g)
    return belief


def chosen_views(counted, P):
    """The (<= MAX_VIEWS) nearest counted views, without those whose mask area is below REL_AREA x the largest."""
    top = max(f["mask"]["area"] for f in counted)
    counted = [f for f in counted if f["mask"]["area"] >= P["REL_AREA"] * top]
    return sorted(counted, key=lambda f: (f["pose"]["distance"], f["step"]))[:MAX_VIEWS]


def fused_accept(per_view, target, P, size=None, info=None):
    """The fused SAM3 verdict (select.fuse_views) with the size opinion: belief' = belief x
    size_opinion.likelihood(W, complete, target, classes), renormalised; accept iff the target is still the top
    class. The opinion is a veto: it can turn an accept into a reject, never the reverse. ``size``: {"W": width (m),
    "complete": bool} of the tracked object, or None (no geometry: the plain fused verdict, as with P["SIZE"] False).
    ``info`` (a dict, optional) receives the belief before / after, the likelihoods and both verdicts (logging)."""
    fused = SEL.fuse_views(per_view, target)
    accept, before = fused["accept"], fused["belief"]
    lik = after = None
    if P.get("SIZE") and size is not None and before:
        lik = SZ.likelihood(size.get("W"), size.get("complete"), target, list(before))
        after = SZ.update(before, lik)
        accept = SZ.top(after, target) == target
    if info is not None:
        def r6(d):
            return None if d is None else {c: round(v, 6) for c, v in d.items()}
        info.update({"applied": bool(lik) and any(v != 1.0 for v in lik.values()),
                     "accept_before": fused["accept"], "accept_after": accept,
                     "belief_before": before, "likelihood": r6(lik), "belief_after": r6(after)})
    return accept


def verify(counted, target, P, size=None, info=None):
    """SAM3's verdict ("correct" / "wrong") over the chosen views (fused, with the size opinion); "wrong" when no view
    carries SAM3 scores (the object never got big enough)."""
    per_view = [{"scores": (f["jev"].get("sam3") or {}).get("scores"), "area": f["mask"]["area"],
                 "clipped": any(f["mask"][k] for k in ("top", "bottom", "left", "right"))}
                for f in chosen_views(counted, P)]
    if SEL.aggregate_views(per_view, target)["margin"] is None:
        return "wrong"
    return "correct" if fused_accept(per_view, target, P, size, info) else "wrong"


def p_consistent(belief):
    """P(JEV's top name outweighs its runner-up) under the Dirichlet belief, whatever the names (target-free)."""
    from scipy.special import betainc
    a = np.sort(np.asarray(belief.alpha, float))[::-1]
    return 1 - betainc(a[0], a[1], 0.5)


def sure(belief, P):
    """The trigger: JEV names the object the same way across the counted views."""
    return p_consistent(belief) >= P["P_STAR"]
