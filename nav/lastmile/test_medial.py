"""nav.lastmile.medial lattice planner on synthetic clearance grids (no simulator)."""
import math
import unittest

import numpy as np
from scipy.spatial import cKDTree

from nav.lastmile import medial as M

RES, HALF = 0.05, 3.0


def line(a, b, step=0.02):
    a, b = np.asarray(a, float), np.asarray(b, float)
    k = int(np.linalg.norm(b - a) / step) + 1
    return a + np.linspace(0, 1, k)[:, None] * (b - a)


class Grid:
    def __init__(self, walls):
        n = int(2 * HALF / RES)
        self.origin, self.res, self.walls = np.array([-HALF, -HALF]), RES, walls
        cen = self.origin + (np.indices((n, n)).transpose(1, 2, 0) + 0.5) * RES
        self.clearance = cKDTree(walls).query(cen.reshape(-1, 2))[0].reshape(n, n) - 0.19
        self.unknown = np.zeros((n, n), bool)

    def planning_grid(self, here, half, res):
        return self


def drive(walls, start, yaw, obj, steps=80):
    """Closed loop with a kinematic agent: a FORWARD whose step would touch (true clearance < 0) does not move."""
    g, nav, p = Grid(walls), M.MedialNav(stop_m=0.65), np.array(start, float)
    touches, why = 0, None
    for _ in range(steps):
        a, info = nav.act([p[0], 0, p[1]], yaw, g, np.array(obj, float))
        if a == "stop":
            why = info["why"]
            break
        if a == "turn_left":
            yaw += math.pi / 6
        elif a == "turn_right":
            yaw -= math.pi / 6
        else:
            q = p + 0.25 * M.fwd(yaw)
            if cKDTree(walls).query(line(p, q, 0.01))[0].min() - 0.19 >= 0:
                p = q
            else:
                touches += 1
    return why, float(np.min(np.linalg.norm(np.array(obj) - p, axis=1))), touches


class LatticeTest(unittest.TestCase):
    def test_through_a_doorway_without_touching(self):
        for half_door in (0.4, 0.325):   # 0.8 m and 0.65 m doors in a wall at x = 0, centre z = 0.8
            walls = np.vstack([line((0, -3), (0, 0.8 - half_door)), line((0, 0.8 + half_door), (0, 3))])
            why, d, touches = drive(walls, (-1.2, -0.6), 0.3, [[1.5, 1.6]])
            self.assertEqual(why, "reached")
            self.assertLessEqual(d, 0.65 + 1e-6)
            self.assertEqual(touches, 0)

    def test_corridor(self):
        walls = np.vstack([line((-3, -0.5), (3, -0.5)), line((-3, 0.5), (3, 0.5))])
        why, d, touches = drive(walls, (-2.5, 0.2), -1.4, [[2.5, 0.0]])
        self.assertEqual((why, touches), ("reached", 0))

    def test_unreachable_object_finishes_at_the_wall(self):
        why, d, touches = drive(line((0, -3), (0, 3)), (-1.5, 0.0), -math.pi / 2, [[1.0, 0.0]])
        self.assertEqual(why, "final")
        self.assertLess(d, 1.35)

    def test_failed_forward_is_not_replanned_from_any_heading(self):
        # open floor, object ahead; the agent's FORWARD has failed here: the plan must not turn away and come back to it
        g = Grid(line((-3, 2.9), (3, 2.9)))
        nav = M.MedialNav(stop_m=0.65)
        nav.blocked.add(nav._key(np.zeros(2), 0.0))
        lat = M.Lattice(g.clearance, g.unknown, g.origin, RES, np.array([[0.0, -2.0]]), 0.65)
        (acts, _), _ = lat.plan(np.zeros(2), 0.0, lambda p, hb: nav._key(p, hb * math.pi / 6) in nav.blocked)
        self.assertNotEqual(acts[0], M.FORWARD)
        i = acts.index(M.FORWARD)
        heading = sum(1 if a == M.LEFT else -1 for a in acts[:i]) % 12
        self.assertNotEqual(heading, 0)


if __name__ == "__main__":
    unittest.main()
