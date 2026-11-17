"""Split navigation: the host planner chooses targets and decides STOP; the perception gate verifies targets. While the
planner heads for a gate-ACCEPTED object (or, scope "object", for an object the gate has not decided), each of its
motion actions is replaced by MedialNav's action toward that object (nav.lastmile.medial: A* over the agent's state
lattice on our own episode collision map). The planner's own STOP, exploration and other actions are untouched.

``SplitNav.act(native_action, target, sim, pose_pf, gate, body_map)`` -> (override action or None, info).
``target``: what the planner is heading for, in the planner frame: {"cluster_id", "x", "y" (the target point),
"accepted" (the gate accepted that object), "heading_to_object" (the planner navigates to an unaccepted object),
"log" (extra fields for ``info``)}. ``pose_pf``: the agent's (x, y, yaw) in the planner frame at the step the target
was published. ``body_map``: nav.lastmile.body_map.BodyMap (EpisodeCollisionMap.lmap).
"""
import math

import numpy as np

from nav.gate import gate as G
from nav.approach import scene
from nav.lastmile.medial import MedialNav, STEP_M

MOTIONS = ("move_forward", "turn_left", "turn_right")
MATCH_M = 1.0      # an accepted object whose points come this near the planner's target (horizontal) is the goal
FREEZE_M = 0.30    # a point goal is kept until the planner's target moves farther than this (plan commitment needs equal arrays)
STOP_M = 0.45
BLOCKED_M = STEP_M - 0.02      # any forward cut short (e.g. 0.22 m) failed
SPECK_LINK_M, SPECK_PTS = 0.12, 10   # an accepted object's points drop connected parts (SPECK_LINK_M linkage) under
                                     # SPECK_PTS points: depth-edge specks 1-5 points, 2 cm across, ended stops 2-3 m short
                                     # of the object; real parts have >= 36


def solid(pts):
    """``pts`` without connected parts (SPECK_LINK_M linkage) of fewer than SPECK_PTS points (all of ``pts`` if none is
    that large)."""
    pts = np.asarray(pts)
    if len(pts) < SPECK_PTS:
        return pts
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    pairs = cKDTree(pts).query_pairs(SPECK_LINK_M, output_type="ndarray")
    n = len(pts)
    lab = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)),
                               directed=False)[1]
    keep = np.bincount(lab)[lab] >= SPECK_PTS
    return pts[keep] if keep.any() else pts


def pf_to_world_xz(point_pf, position, yaw, pose_pf):
    """A planner-frame (x, y) point in the sim's (x, z), through the agent's pose in both frames: its offset ahead /
    to the left of the agent is frame-independent (both yaws turn left positive)."""
    d = np.asarray(point_pf, float)[:2] - np.asarray(pose_pf[:2], float)
    psi = float(pose_pf[2])
    ahead = d[0] * math.cos(psi) + d[1] * math.sin(psi)
    left = -d[0] * math.sin(psi) + d[1] * math.cos(psi)
    f = np.array([-math.sin(yaw), -math.cos(yaw)])
    l = np.array([-math.sin(yaw + math.pi / 2), -math.cos(yaw + math.pi / 2)])
    return np.asarray(position, float)[[0, 2]] + ahead * f + left * l


def scope(target):
    """"accepted" (the planner heads for a gate-accepted target), "object" (it navigates to an unaccepted object) or None."""
    if target["accepted"]:
        return "accepted"
    return "object" if target["heading_to_object"] else None


class SplitNav:
    def __init__(self, stop_m=STOP_M):
        self.stop_m = stop_m
        self.ident = None      # identity of the current goal: accepted-object index or ("point", cluster_id)
        self.nav = None
        self.point_ident = None
        self.point = None      # frozen single-point goal (world xz)

    def goal(self, target_xz, accepted_objects, cluster_id):
        """(identity, goal points (N, 2), source) for the planner's target point. ``accepted_objects``: [(index, points (M, 3))]."""
        best, best_d = None, None
        for index, pts in accepted_objects:
            d = float(np.min(np.linalg.norm(np.asarray(pts, float)[:, [0, 2]] - target_xz, axis=1)))
            if best_d is None or d < best_d:
                best, best_d = (index, pts), d
        if best is not None and best_d <= MATCH_M:
            pts = np.asarray(solid(best[1]))
            return best[0], pts[:, [0, 2]], "accepted_object"
        ident = ("point", int(cluster_id))
        if self.point_ident != ident or self.point is None or np.linalg.norm(target_xz - self.point) > FREEZE_M:
            self.point, self.point_ident = np.asarray(target_xz, float).copy(), ident
        return ident, self.point[None, :], "point"

    def act(self, native_name, target, sim, pose_pf, gate, body_map):
        sc = scope(target)
        info = {"target_cluster_id": int(target["cluster_id"]), "target_accepted": bool(target["accepted"]),
                "scope": sc, **target.get("log", {})}
        if native_name not in MOTIONS or sc is None:
            if sc is None:
                self.ident = self.nav = None          # the planner explores again: the next object gets a fresh navigator
            return None, info
        position, yaw, _ = scene.agent_pose(sim)
        target_xz = pf_to_world_xz((target["x"], target["y"]), position, yaw, pose_pf)
        accepted = [(i, p) for i, (p, s) in enumerate(gate.decided_objects) if s == G.ACCEPTED] if sc == "accepted" else []
        ident, goal_xz, source = self.goal(target_xz, accepted, target["cluster_id"])
        if ident != self.ident or self.nav is None:
            self.ident, self.nav = ident, MedialNav(sim, stop_m=self.stop_m, blocked_m=BLOCKED_M)
        action, ninfo = self.nav.act(position, yaw, body_map, goal_xz)
        info.update(source=source, n_goal=int(len(goal_xz)), **ninfo)
        return (None if action == "stop" else action), info
