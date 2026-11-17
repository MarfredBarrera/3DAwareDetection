"""View-aware MPPI approach planner (mppi_v15; Stage B 99/104, second subset 103/120; self-contained: RGB-D, odometry
and the object mask only, no navmesh). Receding-horizon sampling over the robot's own action set (move_forward 0.25 m,
turn 30 deg, stay; no strafe, no tilt), rolled out on the self-built occupancy grid (local_nav) with the object's view
cost from the fused depth cloud (plus geometric phantoms where a frame cuts the object) in a 3D voxel map of the own
depth hits. The naive approach (closer.Closer driven by local_nav.LocalController) is the warm start and, far out, the
guide. ``MppiV15().reset(ctx); action = MppiV15().act(Observation)`` -> "move_forward" / "turn_left" / "turn_right"
or None (stop: no rollout beats staying here).
"""
import math

import numpy as np

from nav.approach import geom as G
from nav.approach import local_nav as L
from nav.approach import scene as S

ACTIONS = ("move_forward", "turn_left", "turn_right")
IMG_AREA = float(S.WIDTH * S.HEIGHT)


class VoxelMap:
    """3D occupancy of the robot's own depth hits (10 cm voxels, 0-2.4 m above the floor), object pixels excluded.
    A voxel is occupied after >= 2 hits. Unlike the 2.5D max-height map it keeps overhangs (ceilings, lintels,
    table tops) from hiding what is under them."""
    CELL, NZ = 0.10, 24

    def __init__(self, center_xz, floor_y, half=8.0):
        self.origin = np.asarray(center_xz, float) - half
        self.n = int(2 * half / self.CELL)
        self.floor_y = floor_y
        self.hits = np.zeros((self.n, self.n, self.NZ), np.uint8)

    def update(self, depth, mask, position, yaw, stride=4):
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        cam = np.array([position[0], self.floor_y + G.H_CAM, position[2]])
        valid = (depth > 0.05) & (depth < G.DEPTH_MAX) & ~mask
        pts = S.backproject(depth, valid, cam, rot, stride=stride)
        if not len(pts):
            return
        idx = self.index(pts[:, [0, 2]], pts[:, 1] - self.floor_y)
        ok = idx[3]
        np.add.at(self.hits, (idx[0][ok], idx[1][ok], idx[2][ok]), 1)
        np.minimum(self.hits, 50, out=self.hits)

    def index(self, xz, h):
        ij = np.floor((np.asarray(xz) - self.origin) / self.CELL).astype(int)
        k = np.floor(np.asarray(h) / self.CELL).astype(int)
        ok = (ij >= 0).all(-1) & (ij < self.n).all(-1) & (k >= 0) & (k < self.NZ)
        return ij[..., 0], ij[..., 1], k, ok

    def occupied(self):
        return self.hits >= 2


def _tukey(x, a):
    """1 on [a, 1 - a], cosine taper to 0 at 0 and 1, 0 outside (torch)."""
    import torch
    lo = 0.5 * (1 - torch.cos(math.pi * torch.clamp(x, 0, a) / a))
    hi = 0.5 * (1 - torch.cos(math.pi * torch.clamp(1 - x, 0, a) / a))
    return torch.where((x < 0) | (x > 1), torch.zeros_like(x), torch.minimum(lo, hi))


class MppiV15:
    """View-aware MPPI approach planner (2026-11-09; Stage B 99/104, second subset 103/120). One cost, minimized over
    action rollouts; no phases and no triggers.

    Terms
    - rollout: an action sequence of horizon H on the lattice (F 0.25 m, L / R 30 deg, S stay; S is absorbing),
      simulated on the agent's own occupancy grid to poses p_0 .. p_H (p_0 the current pose).
    - warm-start trajectory: the rollout of the naive approach behaviour (closer.Closer + LocalController: drive
      to a fit distance and face the object), recomputed every step; its goal pose g is also a proposal.
    - view quality of a pose, J(p) = sat(size) * completeness^KAPPA * centeredness, with view cost c(p) = -log J(p):
      * object model: the fused cloud voxel-thinned (one point per 5 cm voxel, area a each) plus phantom points over
        extents a frame cut and no frame has shown since (top / bottom / sides, sized by the category priors
        closer.PRIORS, mass proportional to the missing extent);
      * completeness: sum a * o * w / sum a, the fraction of the object's area projected inside a Tukey window
        (w = 1 in the central 1 - 2 WIN of each image axis) and unoccluded (o: ray through the own voxel map);
      * size: in-window projected area / image area, saturated as size / (size + S0);
      * centeredness: 1 within 15 deg of the object's bearing, Gaussian falloff (10 deg) beyond.
      Moving in raises size and lowers completeness; the minimum is the largest view that keeps the whole object.

    Cost of a rollout (to minimize). For stopping at step t,
        cost_t = sum_{s<t} q_s R_s c(p_s) + R_t c(p_t) + C_ACT t,
      where q_s = P_COMMIT for a usable frame (facing within FACE_DEG and visible) and 0 otherwise (q_0 = 0, the
      current frame is already seen) and R_s = prod_{r<s} (1 - q_r) is the probability that no decision was made
      before s: the expected view cost of the frame the decision flow commits on (P_COMMIT 0.3, measured: the flow
      committed at 33 % of first admitted frames, 22 % of all). Then
        cost = min_t cost_t + C_BUMP bumps + w W_REF track,
      track = mean_s (|p_s - p^ws_s|^2 / SIG_P^2 + wrap(yaw_s - yaw^ws_s)^2 / SIG_Y^2) against the warm-start
      trajectory, w = clip(|g - p_0| / D_BLEND, 0, 1) and non-increasing over the episode (it guides the far-out
      approach; a deliberate back-off for the view never hands control back to it). t* = argmin_t cost_t.

    Optimizer (Biased-MPPI, Trevisan & Alonso-Mora 2024): K rollouts per iteration, ITERS iterations, half perturbed
    around warm starts (keep each action with 1 - EPS), half from the softmax-updated per-step action distribution
    (weights exp(-(cost - min cost) / LAMBDA)). Warm starts: path-following rollouts to ring proposals (bearings every
    15 deg, RADII beyond the near face, reachable and faceable within H; both turn directions; the 2 N_PROP with the
    lowest cost of 3 N_PROP), the warm-start trajectory and its goal g, the previous rollout shifted (preferred unless
    beaten by DELTA), stay, and turn-to-face.
    Execution: the first action of the lowest-cost rollout; stop when staying costs less than DELTA_eff more than it
    (turning to face first if needed). Uncertainty-aware stop (2026-11-09): an extra action for a better view is worth
    more while the verdict is open (value of information ~ 1 - p), so DELTA_eff = DELTA (1 - P_STAR) / (1 - p),
    clipped to [0.2, 3] DELTA, with p = the decision belief's P(consistent) (obs.meta["p_consistent"], given by
    nav.gate.approach.LiveApproach; 0 before any counted frame); DELTA at p = P_STAR. Without the key (the
    offline harness run.py) DELTA_eff = DELTA. The previous-rollout hysteresis in plan() keeps DELTA."""
    name = "mppi_v15"
    H, K, ITERS, EPS = 16, 256, 3, 0.15
    KAPPA, WIN, TOP_PAD, SIDE_PAD, EDGE, SEEN_PAD = 2.0, 0.1, 0.2, 0.2, 4, 0.15
    LOOK, S0 = 0.4, 0.05
    N_POINTS, N_PROP, RADII = 400, 6, (0.5, 0.8, 1.1, 1.5, 2.0, 2.6)
    P_COMMIT, DELTA, C_ACT = 0.3, 0.1, 0.02
    P_STAR = 0.8                     # nav.approach.decide.DEFAULTS["P_STAR"], the decision flow's commit threshold: DELTA_eff = DELTA there
    C_MAX = -math.log(1.5e-3)        # a pose whose view cost exceeds this is not a usable frame
    LAMBDA, C_BUMP = 0.15, 1.0
    PRIOR = np.array([0.5, 0.25, 0.25])
    FACE_DEG = 20.0
    STAY = 3
    W_REF, SIG_P, SIG_Y, D_BLEND = 1.0, 0.25, math.radians(30.0), 1.5   # the warm-start trajectory's tracking cost
    # Clearance term: + CLEAR_W * mean_t max(0, CLEAR_D - d_t) / CLEAR_D over the rollout's poses, d_t = distance from pose t
    # to the nearest cell of the INFLATED blocked region (the same quantity as ApexNav's ESDF, measured from the
    # 0.18 m-inflated occupancy; free poses therefore have d >= ~0.1 m). Scale: the view cost c = -log J spans ~0..6.5 and
    # differences between rollouts are O(0.1-1) (LAMBDA 0.15), a bump costs C_BUMP = 1; a rollout that hugs the inflated
    # edge for its whole length (d ~ 0.1 -> penalty 0.67) therefore costs about CLEAR_W x 0.67.
    CLEAR_W, CLEAR_D = 1.0, 0.3

    def reset(self, ctx):
        from nav.approach.closer import Closer
        self.prev = None          # (position, yaw, action) of the last call: bump detection
        self.last = None
        self.rng = np.random.default_rng(0)
        self.done = False
        self.vox = None
        self.top_seen, self.bot_seen, self.top_cut, self.bot_cut = False, False, False, False
        self.side_ph, self.prev_rollout, self._w_cap = None, None, 1.0
        import torch
        self.dev = torch.device("cpu")
        self.closer = Closer()    # the warm start: closer's drive, steered by local_nav.LocalController
        self.closer.reset(ctx)
        self.sim, self.ctrl = ctx.get("sim"), None
        self.c_arrived, self.c_idle, self.c_skips, self.c_done = False, 0, 0, False


    def cut_flags(self, obs):
        """Sticky vertical-extent flags from facing frames (>= 400 mask px, <= 35 deg off): an extent is cut when the
        mask touches that image border (EDGE px band), and seen once a frame's border at the object's depth clears the
        cloud's extreme by SEEN_PAD (the tracked mask grows into new object pixels, so a taller object would have
        shown up there)."""
        mask = np.asarray(obs.mask, bool)
        cloud = np.asarray(obs.cloud)
        if mask.sum() < 400 or len(cloud) < 10:
            return
        here = np.asarray(obs.position)[[0, 2]]
        center = G.ObjectModel(cloud, obs.floor_y).center
        if math.degrees(abs(G.wrap(G.bearing_yaw(here, center) - obs.yaw))) > 35.0:
            return
        h = cloud[:, 1] - obs.floor_y
        depth = max(float(np.median((cloud[:, [0, 2]] - here) @ G.forward(obs.yaw))), 0.3)
        span = depth * math.tan(G.BETA_V)
        if mask[: self.EDGE].any():
            self.top_cut = True
        if G.H_CAM + span > float(np.percentile(h, 98)) + self.SEEN_PAD:
            self.top_seen = True
        if mask[-self.EDGE:].any():
            self.bot_cut = True
        if G.H_CAM - span < float(np.percentile(h, 2)) - self.SEEN_PAD or float(np.percentile(h, 2)) < 0.1:
            self.bot_seen = True

    def act(self, obs):
        """Recompute the warm-start trajectory and its weight, then plan."""
        self._w, self._ws_seq, self._ws_poses, self._ws_goal = 0.0, None, None, None
        self.cut_flags(obs)
        cloud = np.asarray(obs.cloud)
        if len(cloud) and hasattr(obs.pathfinder, "lmap") and not self.done:
            here = np.asarray(obs.position)[[0, 2]].astype(float)
            center = G.ObjectModel(cloud, obs.floor_y).center
            H = max(1, min(self.H, obs.remaining))
            try:
                seq, g = self.reference(obs, here, center, H)   # naive approach: next action + descent to g
            except Exception:
                seq, g = None, None
            if seq is not None and not (seq == self.STAY).all() and g is not None and np.isfinite(g).all():
                self._ws_seq, self._ws_goal = seq, g
                self._w = float(np.clip(np.hypot(*(g - here)) / self.D_BLEND, 0.0, 1.0))
                self._ws_poses = self.rollout(seq[None], here, obs.yaw, obs.pathfinder)[0][0]
        self._w = self._w_cap = min(self._w, self._w_cap)
        return self.view_act(obs)

    def model_points(self, obs):
        from nav.approach.closer import prior_hw
        cloud = np.asarray(obs.cloud, float)
        mask = np.asarray(obs.mask, bool)
        _, idx = np.unique(np.floor(cloud / 0.05).astype(np.int64), axis=0, return_index=True)
        pts = cloud[idx]
        n_vox = len(pts)
        if n_vox > self.N_POINTS:
            pts = pts[self.rng.choice(n_vox, self.N_POINTS, replace=False)]
        area = 0.0025 * n_vox / len(pts)
        prior_h, prior_w = prior_hw(obs.meta or {})
        h = pts[:, 1] - obs.floor_y
        z_lo, z_hi = float(h.min()), float(h.max())
        hv = max(z_hi - z_lo, 0.1)
        here = np.asarray(obs.position)[[0, 2]]
        extra, info = [], {}
        n = len(pts)
        if self.top_cut and not self.top_seen:
            top = max(z_hi + self.TOP_PAD, z_lo + prior_h)
            m = int(min(2 * n, n * (top - z_hi) / hv))
            rim = pts[h >= z_hi - 0.15 * hv]
            if m and len(rim):
                foot = rim[self.rng.integers(0, len(rim), m)]
                extra.append(np.stack([foot[:, 0], obs.floor_y + self.rng.uniform(z_hi, top, m), foot[:, 2]], 1))
                info["top"] = round(top, 2)
        if self.bot_cut and not self.bot_seen and z_lo > 0.1:
            bot = max(0.02, min(z_lo - self.TOP_PAD, z_hi - prior_h))
            m = int(min(2 * n, n * (z_lo - bot) / hv))
            rim = pts[h <= z_lo + 0.15 * hv]
            if m > 0 and len(rim):
                foot = rim[self.rng.integers(0, len(rim), m)]
                extra.append(np.stack([foot[:, 0], obs.floor_y + self.rng.uniform(bot, z_lo, m), foot[:, 2]], 1))
                info["bot"] = round(bot, 2)
        if mask.sum() >= 200:   # side cuts of this frame (kept while the object is out of view)
            right = np.array([math.cos(obs.yaw), -math.sin(obs.yaw)])
            proj = (pts[:, [0, 2]] - here) @ right
            wv = max(float(proj.max() - proj.min()), 0.1)
            miss = max(self.SIDE_PAD, prior_w - wv)
            side = []
            for sign, edge, cols, side_name in ((-1, proj.min(), slice(0, self.EDGE), "left"),
                                                (1, proj.max(), slice(-self.EDGE, None), "right")):
                if mask[:, cols].any():
                    rim = pts[np.abs(proj - edge) < 0.08]
                    m = int(min(n, n * miss / wv))
                    if m and len(rim):
                        q = rim[self.rng.integers(0, len(rim), m)].copy()
                        t = sign * self.rng.uniform(0, miss, m)
                        q[:, 0] += t * right[0]
                        q[:, 2] += t * right[1]
                        side.append(q)
            self.side_ph = np.concatenate(side) if side else None
        if self.side_ph is not None:
            extra.append(self.side_ph)
            info["side"] = len(self.side_ph)
        allp = np.concatenate([pts] + extra) if extra else pts
        return allp, np.full(len(allp), area), info

    def _prepare(self, obs, pts, wts, center, obj):
        import torch
        occ = self.vox.occupied() & ~obj[..., None]
        T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=self.dev)
        self.t_occ = torch.as_tensor(occ, device=self.dev).reshape(-1)
        self.t_pts, self.t_w, self.t_center = T(pts), T(wts), T(center)
        self.vcache, self.cache = {}, {}

    def _vis(self, cams):
        """(Q, P) float: the ray camera -> point crosses no occupied voxel."""
        import torch
        vox, P = self.vox, self.t_pts
        out = []
        h_pt = P[:, 1] - vox.floor_y
        org = torch.as_tensor(vox.origin, dtype=torch.float32, device=self.dev)
        for a in range(0, len(cams), 64):
            c = cams[a:a + 64]
            d = P[None, :, [0, 2]] - c[:, None]
            L = d.norm(dim=-1)
            s = torch.arange(0.25, max(float(L.max()) - 0.15, 0.3), 0.05, device=self.dev)
            frac = s[None, None] / L.clamp(min=1e-3)[..., None]
            valid = s[None, None] < (L[..., None] - 0.15)
            xy = c[:, None, None] + frac[..., None] * d[:, :, None]
            rh = G.H_CAM + frac * (h_pt[None, :, None] - G.H_CAM)
            ij = torch.floor((xy - org) / vox.CELL).long()
            k = torch.floor(rh / vox.CELL).long()
            ok = valid & (ij >= 0).all(-1) & (ij < vox.n).all(-1) & (k >= 0) & (k < vox.NZ)
            flat = (ij[..., 0].clamp(0, vox.n - 1) * vox.n + ij[..., 1].clamp(0, vox.n - 1)) * vox.NZ + k.clamp(0, vox.NZ - 1)
            blocked = (self.t_occ[flat] & ok).any(-1)
            out.append((~blocked).float())
        return torch.cat(out)

    def view_cost(self, poses):
        """View cost c = -log J for poses (M, 3) (x, z, yaw), numpy in and out (torch-batched)."""
        import torch
        if not len(poses):
            return np.zeros(0)
        pk = [(round(x / 0.02), round(z / 0.02)) for x, z in poses[:, :2]]
        new = sorted({k for k in pk if k not in self.vcache})
        if new:
            cams = torch.as_tensor(np.array(new, float) * 0.02, dtype=torch.float32, device=self.dev)
            vis = self._vis(cams)
            for k, v in zip(new, vis):
                self.vcache[k] = v
        vis = torch.stack([self.vcache[k] for k in pk])                        # (M, P)
        ps = torch.as_tensor(poses, dtype=torch.float32, device=self.dev)
        cam, yaw = ps[:, :2], ps[:, 2]
        f = torch.stack([-torch.sin(yaw), -torch.cos(yaw)], 1)
        r = torch.stack([torch.cos(yaw), -torch.sin(yaw)], 1)
        d = self.t_pts[None, :, [0, 2]] - cam[:, None]
        zc = (d * f[:, None]).sum(-1)
        xc = (d * r[:, None]).sum(-1)
        yc = (self.t_pts[:, 1] - (self.vox.floor_y + G.H_CAM))[None]
        front = zc > 0.1
        z = zc.clamp(min=0.1)
        u = (S.WIDTH / 2 + S.FOCAL * xc / z) / S.WIDTH
        v = (S.HEIGHT / 2 - S.FOCAL * yc / z) / S.HEIGHT
        win = _tukey(u, self.WIN) * _tukey(v, self.WIN) * front.float() * vis
        a = self.t_w[None]
        completeness = (a * win).sum(1) / a.sum()
        size = ((a * win * S.FOCAL ** 2 / z ** 2).sum(1) / IMG_AREA).clamp(max=1.0)
        dc = self.t_center[None] - cam
        bearing = torch.atan2(-dc[:, 0], -dc[:, 1])
        off = torch.rad2deg(torch.abs(torch.remainder(bearing - yaw + math.pi, 2 * math.pi) - math.pi))
        centeredness = torch.exp(-((off - 15.0).clamp(min=0) / 10.0) ** 2)
        J = size / (size + self.S0) * completeness ** self.KAPPA * centeredness
        return (-torch.log(J + 1e-4)).cpu().numpy().astype(float)

    def costs(self, rollouts, here, yaw0, pf, center):
        """(cost (K,), t* (K,), view cost per step (K, H+1), poses (K, H+1, 3)) of action rollouts (K, H)."""
        poses, bumps = self.rollout(rollouts, here, yaw0, pf)
        K, T, _ = poses.shape
        key = np.stack([np.round(poses[..., 0] / 0.02), np.round(poses[..., 1] / 0.02),
                        np.round((poses[..., 2] - yaw0) / G.LATTICE) % 12], -1).reshape(-1, 3).astype(np.int64)
        uniq, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        todo = [i for i in range(len(uniq)) if tuple(uniq[i]) not in self.cache]
        if todo:
            for i, c in zip(todo, self.view_cost(poses.reshape(-1, 3)[first[todo]])):
                self.cache[tuple(uniq[i])] = c
        c = np.array([self.cache[tuple(q)] for q in uniq])[np.asarray(inv).ravel()].reshape(K, T)
        d = center[None, None] - poses[..., :2]
        bearing = np.arctan2(-d[..., 0], -d[..., 1])
        off = np.abs((bearing - poses[..., 2] + np.pi) % (2 * np.pi) - np.pi)
        usable = (off <= math.radians(self.FACE_DEG)) & (c < self.C_MAX)
        q = usable.astype(float) * self.P_COMMIT
        q[:, 0] = 0.0                                                       # the current frame is already seen
        R = np.cumprod(np.c_[np.ones((K, 1)), 1.0 - q[:, :-1]], axis=1)       # P(no decision before t)
        A = np.c_[np.zeros((K, 1)), np.cumsum((q * R * c)[:, :-1], axis=1)]   # expected cost of earlier decisions
        cost_t = A + R * c + self.C_ACT * np.arange(T)[None]
        tstar = cost_t.argmin(1)
        cost = cost_t.min(1) + self.C_BUMP * bumps
        cost = cost + self.clearance_penalty(poses, pf)
        if self._w > 0 and self._ws_poses is not None:   # tracking the warm-start trajectory, weighted far out
            wp = self._ws_poses
            n = min(T, len(wp))
            dp = ((poses[:, 1:n, :2] - wp[None, 1:n, :2]) ** 2).sum(-1) / self.SIG_P ** 2
            dy = ((poses[:, 1:n, 2] - wp[None, 1:n, 2] + np.pi) % (2 * np.pi) - np.pi) ** 2 / self.SIG_Y ** 2
            cost = cost + self._w * self.W_REF * (dp + dy).mean(1)
        return cost, tstar, c, poses

    def clearance_penalty(self, poses, pf):
        """CLEAR_W x (K,) mean over steps of max(0, CLEAR_D - d)/CLEAR_D at the poses (K, T, 3); d from the inflated blocked cells."""
        from scipy.ndimage import distance_transform_edt
        lm = pf.lmap
        blocked = ~np.isfinite(pf.cost)
        if not blocked.any():
            return np.zeros(len(poses))
        dist = distance_transform_edt(~blocked) * L.CELL
        ij = np.clip(np.floor((poses[..., :2] - lm.origin) / L.CELL).astype(int), 0, lm.n - 1)
        d = dist[ij[..., 0], ij[..., 1]]
        return self.CLEAR_W * np.maximum(0.0, self.CLEAR_D - d).mean(1) / self.CLEAR_D

    def proposals(self, obs, here, center, pts):
        """Ring poses around the object, facing it, reachable and faceable within H; the 3 N_PROP lowest by view cost
        plus C_ACT x estimated actions, at least 0.4 m apart."""
        pf, lm = obs.pathfinder, obs.pathfinder.lmap
        r_face = float(np.percentile(np.hypot(*(pts[:, [0, 2]] - center).T), 90))
        b = np.radians(np.arange(0, 360, 15))
        cand = np.array([center + (r_face + rr) * np.array([math.sin(t), math.cos(t)]) for rr in self.RADII for t in b])
        ij = np.floor((cand - lm.origin) / L.CELL).astype(int)
        ok = (ij >= 0).all(1) & (ij < lm.n).all(1)
        cand, ij = cand[ok], ij[ok]
        ok = np.isfinite(pf.cost[ij[:, 0], ij[:, 1]])
        cand = cand[ok]
        if not len(cand):
            return []
        yaw0 = obs.yaw
        k = np.round(np.array([G.wrap(G.bearing_yaw(c, center) - yaw0) for c in cand]) / G.LATTICE)
        poses = np.c_[cand, yaw0 + k * G.LATTICE]
        c = self.view_cost(poses)
        dist_t, _ = pf._tree(lm.cell(here))
        cij = np.floor((cand - lm.origin) / L.CELL).astype(int)
        togo = dist_t[cij[:, 0] * lm.n + cij[:, 1]] * L.CELL / S.FORWARD_M
        turns = np.abs(k) + 2 * (togo > 0.5)
        # a warm-start rollout can only evaluate a proposal it reaches (and faces from) within the horizon
        head = np.array([G.bearing_yaw(here, c) for c in cand])
        k_out = np.round(np.abs([G.wrap(h - yaw0) for h in head]) / G.LATTICE)
        k_in = np.round(np.abs([G.wrap(G.bearing_yaw(c, center) - h) for c, h in zip(cand, head)]) / G.LATTICE)
        steps = np.where(togo > 0.5, togo + k_out + k_in, np.abs(k))
        togo = np.where(steps <= min(self.H, getattr(obs, "remaining", self.H)), togo, np.inf)
        rank = c + self.C_ACT * (np.where(np.isfinite(togo), togo, 1e3) + turns)
        order = np.argsort(rank)
        keep = []
        for i in order:
            if not np.isfinite(togo[i]):
                break
            if all(np.hypot(*(cand[i] - cand[j])) > 0.4 for j in keep):
                keep.append(i)
            if len(keep) >= 3 * self.N_PROP:
                break
        self.prop_info = [(round(float(cand[i][0]), 2), round(float(cand[i][1]), 2), round(float(c[i]), 2)) for i in keep]
        return [cand[i] for i in keep]

    def path_rollout(self, obs, here, yaw0, goal, center, H, long_way=False):
        """Warm-start rollout: lattice actions (0 F, 1 L, 2 R, 3 stay) that follow the local-map path here -> goal (pure pursuit on a
        LOOK m look-ahead point, turning the short way when off by > 15 deg), then turn to face the object and stay."""
        pf = obs.pathfinder
        cells = pf.path_cells(here, goal)
        if not cells:
            return None
        path = np.array([pf.lmap.center(c) for c in cells] + [np.asarray(goal, float)])
        seq, pos, k = [], here.astype(float).copy(), 0
        arrived = False
        if long_way:   # the initial rotation the other way round (does not sweep the view past the object)
            diff = G.wrap(G.bearing_yaw(pos, path[min(len(path) - 1, 2)]) - yaw0)
            n = int(round(abs(diff) / G.LATTICE))
            if n == 0:
                return None
            seq = [2 if diff > 0 else 1] * (12 - n)
            k = -(12 - n) if diff > 0 else (12 - n)
        for _ in range(H - len(seq)):
            yaw = yaw0 + k * G.LATTICE
            arrived = arrived or np.hypot(*(goal - pos)) <= 0.2
            if arrived:
                diff = G.wrap(G.bearing_yaw(pos, center) - yaw)
                a = self.STAY if abs(math.degrees(diff)) <= 15.0 else (1 if diff > 0 else 2)
            else:
                dd = np.hypot(*(path - pos).T)
                i0 = int(dd.argmin())
                ahead = next((q for q in path[i0:] if np.hypot(*(q - pos)) >= self.LOOK), path[-1])
                diff = G.wrap(G.bearing_yaw(pos, ahead) - yaw)
                a = 0 if abs(math.degrees(diff)) <= 15.0 else (1 if diff > 0 else 2)
            seq.append(a)
            if a == 0:
                pos = pos + S.FORWARD_M * G.forward(yaw)
            elif a in (1, 2):
                k += 1 if a == 1 else -1
        return np.array(seq[:H], int)

    def rollout(self, seqs, x0, yaw0, pf):
        """Poses (K, H+1, 3) and bump counts (K,) of action sequences (K, H) on the inflated local grid (stay: neither
        turn nor step)."""
        seqs = np.where(seqs == self.STAY, -1, seqs)
        K, H = seqs.shape
        lm = pf.lmap
        pos = np.tile(x0, (K, 1)).astype(float)
        k = np.zeros(K, int)
        poses = np.zeros((K, H + 1, 3))
        poses[:, 0, :2], poses[:, 0, 2] = pos, yaw0
        bumps = np.zeros(K)
        for t in range(H):
            a = seqs[:, t]
            k = k + (a == 1) - (a == 2)
            yaw = yaw0 + k * G.LATTICE
            fw = a == 0
            if fw.any():
                f = np.stack([-np.sin(yaw[fw]), -np.cos(yaw[fw])], 1)
                ok = np.ones(fw.sum(), bool)
                for frac in (0.5, 1.0):
                    q = pos[fw] + frac * S.FORWARD_M * f
                    ij = np.clip(np.floor((q - lm.origin) / L.CELL).astype(int), 0, lm.n - 1)
                    ok &= np.isfinite(pf.cost[ij[:, 0], ij[:, 1]])
                moved = pos[fw] + S.FORWARD_M * f
                newp = pos[fw]
                newp[ok] = moved[ok]
                pos[fw] = newp
                b = bumps[fw]
                b[~ok] += 1
                bumps[fw] = b
            poses[:, t + 1, :2], poses[:, t + 1, 2] = pos, yaw
        return poses, bumps

    def nominal(self, obs, here, yaw0, goal, center, H):
        """closer's next H actions on the lattice (0 F, 1 L, 2 R, 3 stay): the cost-to-go descent to ``goal`` (each step: the
        heading k x 30 deg minimising |k| turns + forward steps until blocked + the cost-to-go there), facing the object
        at the goal."""
        pf = obs.pathfinder
        lm = pf.lmap
        if goal is None:   # no goal: face the object
            seq, k = [], 0
            for _ in range(H):
                diff = G.wrap(G.bearing_yaw(here, center) - (yaw0 + k * G.LATTICE))
                a = self.STAY if abs(math.degrees(diff)) <= 15.0 else (1 if diff > 0 else 2)
                seq.append(a)
                k += 1 if a == 1 else -1 if a == 2 else 0
            return np.array(seq)
        dist, _ = pf._tree(lm.cell(goal))
        togo = lambda xz: dist[lm.cell(xz)[0] * lm.n + lm.cell(xz)[1]] * L.CELL / S.FORWARD_M

        def free(p, h):
            f = G.forward(h)
            return all(np.isfinite(pf.cost[lm.cell(p + frac * S.FORWARD_M * f)]) for frac in (0.5, 1.0))

        seq, pos, k = [], here.astype(float).copy(), 0
        for _ in range(H):
            yaw = yaw0 + k * G.LATTICE
            if np.hypot(*(goal - pos)) <= 0.25 or not np.isfinite(togo(pos)):
                diff = G.wrap(G.bearing_yaw(pos, center) - yaw)
                a = self.STAY if abs(math.degrees(diff)) <= 15.0 else (1 if diff > 0 else 2)
            else:
                best = (math.inf, 0)
                for dk in range(-5, 7):
                    h, p, c = yaw + dk * G.LATTICE, pos, math.inf
                    for j in range(1, 5):
                        if not free(p, h):
                            break
                        p = p + S.FORWARD_M * G.forward(h)
                        c = min(c, abs(dk) + j + togo(p))
                    if c < best[0] - 1e-6:
                        best = (c, dk)
                dk = best[1]
                a = 0 if dk == 0 and np.isfinite(best[0]) else (1 if dk > 0 else 2)
                if not np.isfinite(best[0]):   # boxed in: face the object
                    diff = G.wrap(G.bearing_yaw(pos, center) - yaw)
                    a = self.STAY if abs(math.degrees(diff)) <= 15.0 else (1 if diff > 0 else 2)
            seq.append(a)
            if a == 0:
                pos = pos + S.FORWARD_M * G.forward(yaw)
            elif a in (1, 2):
                k += 1 if a == 1 else -1
        return np.array(seq)

    def reference(self, obs, here, center, H):
        first = self.closer_action(obs)
        goal = self.closer.goal
        g = np.asarray(goal.position)[[0, 2]] if goal is not None else None
        if first == self.STAY:
            return np.full(H, self.STAY), None
        nxt = self.rollout(np.array([[first]]), here, obs.yaw, obs.pathfinder)[0][0, 1]
        rest = self.nominal(obs, nxt[:2], nxt[2], g, center, H - 1) if H > 1 else np.zeros(0, int)
        return np.r_[[first], rest][:H].astype(int), g

    def closer_action(self, obs):
        """closer's next action as the harness would drive it (Closer.next_viewpoint + local_nav.LocalController),
        as an index (0 F, 1 L, 2 R; STAY when closer is done)."""
        import dataclasses
        if self.c_done:
            return self.STAY
        if self.ctrl is None:
            self.ctrl = L.LocalController(self.sim, obs.pathfinder.lmap, obs.pathfinder)
        if self.prev is not None:   # the action actually executed (MPPI may have deviated): bump detection
            self.ctrl.last = (self.prev[0], self.prev[1], self.prev[2])
        o = dataclasses.replace(obs, arrived=self.c_arrived)
        while True:
            vp = self.closer.next_viewpoint(o)
            if vp is None:
                self.c_done = True
                return self.STAY
            try:
                act = self.ctrl.action(vp, o)
            except Exception:
                self.c_skips += 1
                if self.c_skips > 3:
                    self.c_done = True
                    return self.STAY
                o = dataclasses.replace(o, arrived=True)
                continue
            if act is None:
                self.c_idle += 1
                if self.c_idle > 4:
                    self.c_done = True
                    return self.STAY
                o = dataclasses.replace(o, arrived=True)
                continue
            self.c_arrived, self.c_idle = False, 0
            return ACTIONS.index(act)

    def object_cells(self, vox, cloud):
        from scipy.ndimage import binary_dilation
        grid = np.zeros((vox.n, vox.n), bool)
        i, j, _, ok = vox.index(cloud[:, [0, 2]], np.zeros(len(cloud)))
        grid[i[ok], j[ok]] = True
        return binary_dilation(grid, iterations=1)

    def view_act(self, obs):
        pf = obs.pathfinder
        mask = np.asarray(obs.mask, bool)
        if self.vox is None:
            self.vox = VoxelMap(np.asarray(obs.position)[[0, 2]], obs.floor_y)
        self.vox.update(np.asarray(obs.depth), mask, obs.position, obs.yaw)
        if self.prev is not None and self.prev[2] == "move_forward":
            if pf.lmap.forward_feedback(self.prev[0], self.prev[1], obs.position):
                pf.refresh(obs.position)
        if self.done or not len(obs.cloud) or not hasattr(pf, "lmap"):
            self.last = {"stop": "no_object" if not len(obs.cloud) else "done"}
            return None
        return self.execute(obs, self.plan(obs))

    def plan(self, obs):
        """One MPPI optimization from the current pose: the lowest-cost rollout and the costs the stop rule needs."""
        pf = obs.pathfinder
        here = np.asarray(obs.position)[[0, 2]].astype(float)
        cloud = np.asarray(obs.cloud)
        center = G.ObjectModel(cloud, obs.floor_y).center
        H = max(1, min(self.H, obs.remaining))
        pts, wts, info = self.model_points(obs)
        self._prepare(obs, pts, wts, center, self.object_cells(self.vox, cloud))
        goals = list(self.proposals(obs, here, center, pts)) + ([self._ws_goal] if self._ws_goal is not None else [])
        warm = []
        for g in goals:
            for lw in (False, True):
                r = self.path_rollout(obs, here, obs.yaw, np.asarray(g, float), center, H, long_way=lw)
                if r is not None:
                    warm.append(r)
        if len(warm) > 2 * self.N_PROP:   # keep the warm starts with the lowest cost
            cst = self.costs(np.array(warm), here, obs.yaw, pf, center)[0]
            warm = [warm[i] for i in np.argsort(cst)[: 2 * self.N_PROP]]
        j_prev = j_ws = None
        if self.prev_rollout is not None and len(self.prev_rollout) > 1:   # the previous rollout, shifted one step
            j_prev = len(warm)
            warm.append(np.r_[self.prev_rollout[1:], np.full(H, self.STAY)][:H].astype(int))
        if self._ws_seq is not None:
            j_ws = len(warm)
            warm.append(np.r_[self._ws_seq, np.full(H, self.STAY)][:H].astype(int))
        j_stay = len(warm)
        warm.append(np.full(H, self.STAY))
        diff = G.wrap(G.bearing_yaw(here, center) - obs.yaw)
        warm.append(np.r_[[1 if diff > 0 else 2], np.full(H - 1, self.STAY)][:H])   # turn to face, then stay
        warm = np.array(warm)
        pi = np.tile(self.PRIOR, (H, 1))
        best, best_cost, best_t, ws_cost = None, np.inf, 0, None
        for it in range(self.ITERS):
            n_warm = self.K // 2
            centres = warm if best is None else np.r_[warm, best[None]]
            base = centres[self.rng.integers(0, len(centres), n_warm)]
            keep = self.rng.random((n_warm, H)) >= self.EPS
            pert = np.where(keep, base, self.rng.integers(0, 3, (n_warm, H)))
            cum = pi.cumsum(1)
            free = (self.rng.random((self.K - n_warm, H, 1)) > cum[None]).sum(-1).clip(0, 2)
            rollouts = np.r_[centres, pert, free][: self.K + len(centres)]
            stopped = np.maximum.accumulate(rollouts == self.STAY, axis=1)
            rollouts = np.where(stopped, self.STAY, rollouts)
            cost, tstar, c, poses = self.costs(rollouts, here, obs.yaw, pf, center)
            if it == 0:
                self.warm_info = [("".join("FLRS"[a] for a in warm[j]), round(float(cost[j]), 2), int(tstar[j]),
                                   round(float(c[j].min()), 2), [round(float(x), 2) for x in poses[j, -1]])
                                  for j in range(len(warm))]
                stay_cost = float(cost[j_stay])
                c0 = float(c[0, 0])
                if j_ws is not None:
                    ws_cost = float(cost[j_ws])
                if j_prev is not None:
                    cost[j_prev] -= self.DELTA   # a new rollout must beat the previous one by the stop margin
            i = int(cost.argmin())
            if cost[i] < best_cost:
                best, best_cost, best_t = rollouts[i].copy(), float(cost[i]), int(tstar[i])
            wt = np.exp(-(cost - cost.min()) / self.LAMBDA)
            wt /= wt.sum()
            pi = np.einsum("k,kha->ha", wt, np.eye(4)[rollouts])[:, :3] + 1e-3
            pi /= pi.sum(1, keepdims=True)
        return {"rollout": best, "cost": best_cost, "t*": best_t, "c0": c0, "ws_cost": ws_cost,
                "stay_cost": stay_cost, "diff": diff, "info": info}

    def stop_delta(self, p):
        """The stop tolerance DELTA_eff for the belief's P(consistent) ``p`` (None: DELTA)."""
        if p is None:
            return self.DELTA
        return float(np.clip(self.DELTA * (1 - self.P_STAR) / max(1 - float(p), 1e-3), 0.2 * self.DELTA,
                             3 * self.DELTA))

    def execute(self, obs, r):
        best, diff = r["rollout"], r["diff"]
        off = math.degrees(abs(diff))
        self.last = {"phase": "mppi_v15", "J0": round(float(np.exp(-r["c0"])), 4), "cost": round(r["cost"], 3),
                     "t*": r["t*"], "rollout": "".join("FLRS"[a] for a in best), "phantom": r["info"],
                     "props": getattr(self, "prop_info", []), "warm": self.warm_info,
                     "cut": [self.top_cut, self.top_seen, self.bot_cut, self.bot_seen]}
        if self._ws_seq is not None:   # the warm-start trajectory and its tracking weight
            self.last["ws"] = "".join("FLRS"[a] for a in self._ws_seq)
            self.last["w"] = round(self._w, 2)
            self.last["ws_cost"] = round(r["ws_cost"], 3)
        p = (obs.meta or {}).get("p_consistent")
        delta = self.stop_delta(p)
        if p is not None:
            self.last.update(p=round(float(p), 4), delta=round(delta, 4))
        stop = r["stay_cost"] - r["cost"] < delta or best[0] == self.STAY or (self._w == 0.0 and r["t*"] == 0)
        if stop:
            if off > self.FACE_DEG:
                action = "turn_left" if diff > 0 else "turn_right"
                self.prev_rollout = None
            else:
                self.done = True
                self.last["stop"] = "best_view"
                return None
        else:
            action = ACTIONS[int(best[0])]
            self.prev_rollout = best
        self.prev = (np.asarray(obs.position).copy(), obs.yaw, action)
        self.last["action"] = action
        return action
