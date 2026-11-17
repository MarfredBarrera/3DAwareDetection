"""The naive approach (closer): face the object and drive the straight line to the distance at which the whole object fits
(r = clip(half extent + d_fit on the phantom extents, 0.9, 2.2) m), then stop facing it. It is the warm start and far-out guide of
mppi.MppiV15 and runs on the Observation of any ObjectNav agent: own RGB-D frame, odometry, the object mask and the fused
object cloud, and the self-built local map through ``obs.pathfinder`` (local_nav.LocalPathfinder).
"""
import math
from dataclasses import dataclass, field

import numpy as np

from nav.approach import geom as G

R_MIN, R_MAX = 0.9, 2.2
PRIORS = {"chair": (0.9, 0.55), "couch": (0.85, 2.0), "bed": (0.6, 2.0), "toilet": (0.8, 0.45), "tv": (0.7, 1.1),
          "potted plant": (1.0, 0.5)}


@dataclass
class Viewpoint:
    position: np.ndarray    # x, y, z (sim frame)
    yaw: float              # rad, 0 = -z
    info: dict = field(default_factory=dict)


@dataclass
class Observation:
    step: int
    position: np.ndarray
    yaw: float
    rgb: np.ndarray
    depth: np.ndarray
    mask: np.ndarray
    cloud: np.ndarray
    pathfinder: object
    hmap: object = None
    yaw0: float = 0.0
    floor_y: float = 0.0
    judged: list = field(default_factory=list)   # frames judged so far: camera, yaw, clip flags (phantom_model)
    remaining: int = 40
    arrived: bool = False
    meta: dict = field(default_factory=dict)   # target, p_consistent


def prior_hw(meta):
    """(H, W) size prior of the target category."""
    return PRIORS.get(meta.get("target"), (0.8, 0.8))


def phantom_model(obs):
    """(ObjectModel with the unknown extent, phantom points, info)."""
    model = G.ObjectModel(obs.cloud, obs.floor_y)
    info = {"top": False, "bottom": False, "left": False, "right": False}
    if not obs.judged or not len(obs.cloud):
        return model, np.zeros((0, 3)), info
    last = obs.judged[-1]
    clip = last.get("clip") or {}
    h_prior, w_prior = prior_hw(obs.meta)
    extra = []
    cam = np.asarray(last["camera"], float)
    yaw = last.get("yaw", obs.yaw)
    f, right = G.forward(yaw), np.array([math.cos(yaw), -math.sin(yaw)])
    z_lo, z_hi = model.z_lo, model.z_hi
    if clip.get("top"):
        depth = float((model.center - cam) @ f)
        z_frustum = G.H_CAM + max(depth, 0.1) * math.tan(G.BETA_V)
        z_hi = max(z_hi + 0.2, z_frustum + 0.1, z_lo + h_prior)
        info["top"] = True
    if clip.get("bottom"):
        z_lo = max(0.0, min(z_lo - 0.2, z_hi - h_prior))
        info["bottom"] = True
    xz = obs.cloud[:, [0, 2]]
    proj = (xz - model.center) @ right
    w_obs = float(proj.max() - proj.min()) if len(proj) else 0.0
    ext = max(0.2, w_prior - w_obs)
    for side, sign, edge in (("left", -1.0, proj.min()), ("right", 1.0, proj.max())):
        if not clip.get(side):
            continue
        info[side] = round(ext, 2)
        rim = obs.cloud[np.abs(proj - edge) < 0.08]
        for t in np.linspace(0.05, ext, 6):
            shifted = rim.copy()
            shifted[:, 0] += sign * t * right[0]
            shifted[:, 2] += sign * t * right[1]
            extra.append(shifted[::max(1, len(shifted) // 30)])
    if info["top"] or info["bottom"]:
        foot = obs.cloud[::max(1, len(obs.cloud) // 60)][:, [0, 2]]
        for zz in ((z_hi,) if info["top"] else ()) + ((z_lo,) if info["bottom"] else ()):
            extra.append(np.stack([foot[:, 0], np.full(len(foot), obs.floor_y + zz), foot[:, 1]], 1))
    extra = np.concatenate(extra) if extra else np.zeros((0, 3))
    if len(extra):
        model = G.ObjectModel(np.concatenate([obs.cloud, extra]), obs.floor_y)
    model.z_lo, model.z_hi = z_lo, z_hi
    return model, extra, info



def _model(obs):
    model, extra, info = phantom_model(obs)
    return model, info


def fit_radius(obs, model, info, u):
    """Distance from the object centre along unit ``u`` (centre -> camera) at which the whole object fits."""
    yaw = G.bearing_yaw(model.center + u, model.center)
    side = info.get("left") or info.get("right")
    fit = G.d_fit(model, yaw, top_clipped=bool(info.get("top")), side_clipped=bool(side))
    return float(np.clip(model.half_extent(u) + fit, R_MIN, R_MAX))


def unit(v):
    return v / max(np.linalg.norm(v), 1e-6)


def vp3(obs, xz, center, info):
    pos = np.array([xz[0], obs.floor_y, xz[1]])
    return Viewpoint(pos, G.bearing_yaw(xz, center), info)


class Closer:
    """N1. One stop on the straight line at the fit distance; the goal is re-estimated every 4 steps."""
    name, enroute, driver, reason = "closer", False, None, None
    EVERY = 4

    def reset(self, ctx):
        self.goal, self.age, self.stops = None, 0, 0

    def done_after(self):
        return 1

    def line_goal(self, obs):
        model, info = _model(obs)
        here = np.asarray(obs.position)[[0, 2]]
        u = unit(here - model.center)
        r = fit_radius(obs, model, info, u)
        if np.linalg.norm(here - model.center) <= r + 0.15:   # already close enough: just face it
            return vp3(obs, here, model.center, {"r": round(r, 2), "stop": "here"})
        return vp3(obs, model.center + r * u, model.center, {"r": round(r, 2)})

    def first_stop(self, obs):
        if self.goal is None or self.age >= self.EVERY:
            self.goal, self.age = self.line_goal(obs), 0
        self.age += 1
        return self.goal

    def after_first(self, obs):
        return None

    def next_viewpoint(self, obs):
        if not len(obs.cloud):
            return None
        if obs.arrived:
            self.stops += 1
            self.goal = None
        if self.stops == 0:
            return self.first_stop(obs)
        return self.after_first(obs)
