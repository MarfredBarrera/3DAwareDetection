"""python -m unittest nav.approach.test_mppi_v15 (no Habitat needed). Closed-loop MppiV15 (with its closer warm start) on
a synthetic box, kinematic agent: the action strings were produced by the planner before it was flattened out of
the mppi.py class chain (V2..V10), so they pin its behaviour."""
import math
import unittest
from unittest import mock

import numpy as np

from nav.approach import geom as G, local_nav as L, scene as S
from nav.approach.closer import Observation
from nav.approach.mppi import MppiV15

LETTER = {"move_forward": "F", "turn_left": "L", "turn_right": "R", None: "S"}


def run(start=(0.0, 3.5), yaw0=0.3, wall=False, steps=40):
    state = {"pos": np.array([start[0], 0.0, start[1]]), "yaw": yaw0}
    pts = np.array([[0.4 + dx, y, -1.0 + dz] for dx in np.linspace(-0.4, 0.4, 9) for dz in np.linspace(-0.3, 0.3, 7)
                    for y in np.linspace(0.0, 0.9, 6)])
    hm = G.HeightMap(np.zeros(2), 0.0)
    if wall:
        hm.seen[:] = True
        ij, ok = hm.cells(np.stack([np.full(41, 0.2), np.linspace(-0.5, 2.5, 41)], 1))
        for di in (0, 1):
            hm.top[ij[ok, 0] + di, ij[ok, 1]] = 1.0
            hm.low[ij[ok, 0] + di, ij[ok, 1]] = 1.0
    lm = L.LocalMap(hm)
    pf = L.LocalPathfinder(lm, 0.0)
    policy = MppiV15()
    policy.reset({"sim": object()})
    depth = np.full((480, 640), 4.0, np.float32)
    mask = np.zeros((480, 640), bool)
    mask[200:300, 280:360] = True
    out = ""
    with mock.patch.object(S, "agent_pose", lambda sim: (state["pos"].copy(), state["yaw"], 0.0)):
        for step in range(steps):
            position, yaw = state["pos"].copy(), state["yaw"]
            pf.refresh(position)
            rot = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
            hm.update(depth, mask, np.array([position[0], 0.88, position[2]]), rot)
            lm.set_object(pts)
            obs = Observation(step, position, yaw, None, depth, mask, pts, pf, hm, 0.3, 0.0, [], steps - step, False,
                              {"target": "chair", "p_consistent": 0.5})
            action = policy.act(obs)
            out += LETTER[action]
            if action is None:
                break
            f = np.array([-math.sin(yaw), -math.cos(yaw)])
            if action == "move_forward":
                state["pos"] = position + np.array([0.25 * f[0], 0, 0.25 * f[1]])
            else:
                state["yaw"] += math.radians(30) * (1 if action == "turn_left" else -1)
    return out


class MppiV15Test(unittest.TestCase):
    def test_golden_trajectories(self):
        self.assertEqual(run(), "FFFFRFFFFFFFFS")
        self.assertEqual(run(start=(2.0, 2.0), yaw0=1.5), "RFRFFFFFFS")
        self.assertEqual(run(start=(-1.5, 1.0), yaw0=-2.0), "FLLFFFFLS")
        self.assertEqual(run(start=(1.0, 2.5), yaw0=3.0, wall=True), "FFFFFFFFFFFFFFLFLLLLLFS")
        self.assertEqual(run(start=(0.4, 0.6), yaw0=0.0), "S")


if __name__ == "__main__":
    unittest.main()
