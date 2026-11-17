"""Medial last mile (MEDIAL_NAV=1): one map, one planner in the agent's own action space, one stop rule.

map       our episode collision map (nav.lastmile.collision_map), via BodyMap.planning_grid, DEPTH points only: per-cell
          clearance = distance to the nearest measured obstacle point minus the body radius (m). Bump contacts are left out
          (MAP_AUDIT.md: depth clearance reads +0.05 m over the navmesh's; a third of the contact cells lie on open floor)
plan      A* over the agent's state lattice: position and one of 12 headings; actions FORWARD 0.25 m and LEFT / RIGHT 30 deg,
          exactly the agent's moves (no sliding). A FORWARD is valid when every sample of its step keeps clearance
          >= C_MIN, the map clearance where the body touches the navmesh edge (a step cut short there leaves the agent at
          edge distance 0, where nearly every heading is dead: lastmile/_tmp/wedge_probe), or, from a pose already below
          C_MIN, does not lower it; a FORWARD that already failed from a pose and heading is not planned again from that pose (wherever it
          lies on the plan: a wedged pose is otherwise re-entered from a neighbour, newctrl 773).
          Cost: 1 per action + W * shortfall of the step's clearance below C_PREF (relative: keeps to the middle of free
          space) + UNKNOWN_W on unseen cells. The plan ends at the state minimising cost + GOAL_K * (object distance beyond
          stop_m). Heuristic: that objective's clearance-weighted 2D cost-to-go (Dijkstra from the object side).
execute   the plan's actions in order while the plan stays valid: replan when the agent is not where the plan put it (a FORWARD
          fell short) or a remaining FORWARD fails the step rule on the current map, or the object cloud changed (per-step replanning on a re-centred grid
          flips marginal steps and oscillates between two poses: newctrl 770, 773)
stop      within stop_m of the object. When the plan has no FORWARD (nothing closer at C_MIN), the same planner continues
          with the limit C_FINAL (contact only matters for later motion and none follows the stop); STOP when that
          plan has no FORWARD either
No learned policy, fallbacks, shields, verification or hand-backs.
"""
import heapq
import math

import numpy as np

C_MIN = 0.08       # map clearance where the body touches the navmesh edge (the map reads +0.05 m, p5 +0.02)
C_PREF = 0.40
W = 4.0
UNKNOWN_W = 1.0
RES = 0.05
STEP_M = 0.25
HEAD_BINS = 12
STEP_S = np.linspace(0.05, STEP_M, 5)
GOAL_K = 40.0      # actions per metre of object distance beyond stop_m (0.1 m closer is worth 4 actions = 1 m of route)
MAX_POP = 40000
C_FINAL = -0.10     # the finish: when nothing is closer at C_MIN, plan again allowing steps the map reads as touching (the
                   # navmesh is up to ~0.08 m looser than our map near furniture: 957/468/69); a STOP follows, so contact is harmless
STOP, FORWARD, LEFT, RIGHT = 0, 1, 2, 3


def fwd(yaw):
    return np.array([-math.sin(yaw), -math.cos(yaw)])


class Lattice:
    """One plan on a clearance grid (cell (i, j) centre = origin + ((i, j) + .5) * res; i -> x, j -> z)."""

    def __init__(self, clear, unknown, origin, res, cloud_xz, stop_m, c_min=None):
        from scipy.spatial import cKDTree
        self.c = np.where(np.isfinite(clear), clear, -1.0)
        self.unknown = np.asarray(unknown, bool)
        self.origin, self.res, self.n = np.asarray(origin, float), res, clear.shape[0]
        centers = self.origin + (np.indices((self.n, self.n)).transpose(1, 2, 0) + 0.5) * res
        self.d_obj = cKDTree(np.asarray(cloud_xz, float)).query(centers.reshape(-1, 2))[0].reshape(self.n, self.n)
        self.term = GOAL_K * np.clip(self.d_obj - stop_m, 0.0, None)
        self.c_min = C_MIN if c_min is None else c_min

    def cell(self, p):
        ij = np.floor((np.asarray(p, float) - self.origin) / self.res).astype(int)
        if (ij < 0).any() or (ij >= self.n).any():
            return None
        return int(ij[0]), int(ij[1])

    def clear_along(self, pts):
        ij = np.floor((pts - self.origin) / self.res).astype(int)
        if (ij < 0).any() or (ij >= self.n).any():
            return -1.0, True
        return float(self.c[ij[:, 0], ij[:, 1]].min()), bool(self.unknown[ij[-1, 0], ij[-1, 1]])

    def heuristic(self, c_start):
        """Clearance-weighted 2D cost-to-go (actions) of the end objective, from every cell."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        n, res = self.n, self.res
        ok = self.c >= min(self.c_min, c_start - 1e-6)
        cost = (1.0 + W * np.clip((C_PREF - self.c) / C_PREF, 0.0, None) + UNKNOWN_W * self.unknown) / STEP_M
        idx = np.arange(n * n).reshape(n, n)
        rows, cols, wts = [], [], []
        for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
            s0 = (slice(0, n - di), slice(max(0, -dj), n - max(0, dj)))
            s1 = (slice(di, n), slice(max(0, dj), n + min(0, dj)))
            both = ok[s0] & ok[s1]
            w = math.hypot(di, dj) * res * 0.5 * (cost[s0] + cost[s1])
            rows.append(idx[s0][both]); cols.append(idx[s1][both]); wts.append(w[both])
        end = np.flatnonzero(ok.reshape(-1))           # virtual source: every passable cell, at its end cost
        r, cc, w = (np.concatenate(v) for v in (rows, cols, wts))
        src = n * n
        g = coo_matrix((np.r_[w, w, self.term.reshape(-1)[end]],
                        (np.r_[r, cc, np.full(len(end), src)], np.r_[cc, r, end])), shape=(n * n + 1, n * n + 1))
        self.h = dijkstra(g.tocsr(), indices=src)[:n * n].reshape(n, n)

    def plan(self, here, yaw, blocked):
        """A* from (here, yaw). Returns ((actions, end position), pops) or None (agent off grid)."""
        c0 = self.cell(here)
        if c0 is None:
            return None
        self.heuristic(float(self.c[c0]))
        key0 = (c0[0], c0[1], 0)
        pos, best, parent = {key0: np.asarray(here, float)}, {key0: 0.0}, {key0: None}
        cnt, pops = 0, 0
        h0 = self.h[c0] if np.isfinite(self.h[c0]) else self.term[c0]
        heap = [(h0, 0.0, cnt, key0, False)]
        fallback = (self.term[c0], key0)
        while heap and pops < MAX_POP:
            _, gk, _, key, is_end = heapq.heappop(heap)
            if is_end:
                return self._path(key, parent, pos), pops
            if gk > best.get(key, math.inf) + 1e-9:
                continue
            pops += 1
            p, hb = pos[key], key[2]
            ci = self.cell(p)
            t_end = gk + self.term[ci]
            if t_end < fallback[0]:
                fallback = (t_end, key)
            cnt += 1
            heapq.heappush(heap, (t_end, t_end, cnt, key, True))
            h_yaw = yaw + hb * 2 * math.pi / HEAD_BINS
            succ = [((key[0], key[1], (hb + 1) % HEAD_BINS), p, 1.0, LEFT),
                    ((key[0], key[1], (hb - 1) % HEAD_BINS), p, 1.0, RIGHT)]
            if not blocked(p, hb):                                 # a FORWARD that already failed from this pose
                f = fwd(h_yaw)
                cmin, unk = self.clear_along(p[None] + STEP_S[:, None] * f[None])
                if cmin >= self.c_min or cmin >= float(self.c[ci]) - 1e-6:
                    q = p + STEP_M * f
                    cq = self.cell(q)
                    if cq is not None:
                        succ.append(((cq[0], cq[1], hb), q,
                                     1.0 + W * max(0.0, (C_PREF - cmin) / C_PREF) + UNKNOWN_W * unk, FORWARD))
            for k2, p2, w, a in succ:
                g2 = gk + w
                if g2 < best.get(k2, math.inf) - 1e-9:
                    c2 = self.cell(p2)
                    h2 = self.h[c2] if np.isfinite(self.h[c2]) else self.term[c2]
                    best[k2], pos[k2], parent[k2] = g2, p2, (key, a)
                    cnt += 1
                    heapq.heappush(heap, (g2 + h2, g2, cnt, k2, False))
        return self._path(fallback[1], parent, pos), pops

    def valid(self, here, yaw, acts, blocked):
        """Whether ``acts`` from (here, yaw) still obey the FORWARD rule on this map."""
        p, hb = np.asarray(here, float), 0
        for k, a in enumerate(acts):
            if a != FORWARD:
                hb += 1 if a == LEFT else -1
                continue
            ci = self.cell(p)
            if ci is None or blocked(p, hb % HEAD_BINS):
                return False
            f = fwd(yaw + hb * 2 * math.pi / HEAD_BINS)
            cmin, _ = self.clear_along(p[None] + STEP_S[:, None] * f[None])
            if not (cmin >= self.c_min or cmin >= float(self.c[ci]) - 1e-6):
                return False
            p = p + STEP_M * f
        return True

    @staticmethod
    def _path(key, parent, pos):
        acts, end = [], pos[key]
        while parent[key] is not None:
            key, a = parent[key]
            acts.append(a)
        return acts[::-1], end


class MedialNav:
    """One last mile. ``act(position, yaw, body_map, cloud_xz)`` -> ("move_forward"|"turn_left"|"turn_right"|"stop", info)."""

    POSE_M, BLOCKED_M = 0.05, STEP_M - 0.05   # a FORWARD that fell short (contact) failed: live 773 looped on a 0.17 m step

    def __init__(self, sim=None, stop_m=0.65, blocked_m=BLOCKED_M):
        self.stop_m = float(stop_m)
        self.blocked_m = float(blocked_m)     # a FORWARD that moved less than this failed
        self.blocked = set()                  # (x, z, heading bin) keys of FORWARDs that did not move
        self.last = None                      # (action, position xz, yaw) of the previous act
        self.final = None                     # True once nothing is closer at C_MIN: finish at C_FINAL
        self.plan = None                      # (remaining actions, expected position xz, expected yaw, end, cloud) of the plan
        self.stats = dict(plans=0, blocked=0, pops=0, stay=0, kept=0)

    def _key(self, xz, yaw):
        return (int(round(xz[0] / self.POSE_M)), int(round(xz[1] / self.POSE_M)),
                int(round(yaw / (2 * math.pi / HEAD_BINS))) % HEAD_BINS)

    def act(self, position, yaw, body_map, cloud_xz):
        here = np.asarray(position, float)[[0, 2]]
        if self.last is not None and self.last[0] == FORWARD and np.linalg.norm(here - self.last[1]) < self.blocked_m:
            self.blocked.add(self._key(self.last[1], self.last[2]))
            self.stats["blocked"] = len(self.blocked)
        self.last = None
        cloud_xz = np.asarray(cloud_xz, float)
        d_obj = float(np.min(np.linalg.norm(cloud_xz - here, axis=1)))
        info = {"d": round(d_obj, 3)}
        if d_obj <= self.stop_m:
            return "stop", {**info, "why": "reached"}
        half = min(8.0, max(3.0, d_obj + 1.5))
        g = body_map.planning_grid(here, half=half, res=RES)
        final = self.final is not None
        lat = Lattice(g.clearance, g.unknown, g.origin, g.res, cloud_xz, self.stop_m, c_min=C_FINAL if final else C_MIN)
        blocked = lambda p, hb: self._key(p, yaw + hb * 2 * math.pi / HEAD_BINS) in self.blocked
        kept = bool(self.plan is not None and len(self.plan[0]) > 0 and self.plan[4].shape == cloud_xz.shape \
            and np.array_equal(self.plan[4], cloud_xz) and np.linalg.norm(here - self.plan[1]) < self.POSE_M \
            and abs(math.remainder(yaw - self.plan[2], 2 * math.pi)) < 0.1 and lat.valid(here, yaw, self.plan[0], blocked))
        if kept:
            acts, end, pops = self.plan[0], self.plan[3], 0
            self.stats["kept"] += 1
        else:
            res = lat.plan(here, yaw, blocked)
            self.stats["plans"] += 1
            if res is None:
                return "stop", {**info, "why": "off_grid"}
            (acts, end), pops = res
            self.stats["pops"] += pops
        c0, ce = lat.cell(here), lat.cell(end)
        info.update(kept=kept, n_plan=len(acts), n_fwd=sum(a == FORWARD for a in acts), pops=pops, phase="final" if final else "plan",
                    end_d=round(float(lat.d_obj[ce]), 3) if ce else None, c_here=round(float(lat.c[c0]), 3))
        if FORWARD not in acts:
            self.plan = None
            if final:                                       # nothing closer even allowing contact: done
                return "stop", {**info, "why": "final"}
            self.stats["stay"] += 1                         # the object is not closer at C_MIN: finish at C_FINAL
            self.final = True
            return self.act(position, yaw, body_map, cloud_xz)
        a = acts[0]
        self.last = (a, here, yaw)
        step = STEP_M * fwd(yaw) if a == FORWARD else 0.0
        yaw2 = yaw + {FORWARD: 0.0, LEFT: math.pi / 6, RIGHT: -math.pi / 6}[a]
        self.plan = (acts[1:], here + step, yaw2, end, cloud_xz.copy())
        return {FORWARD: "move_forward", LEFT: "turn_left", RIGHT: "turn_right"}[a], info
