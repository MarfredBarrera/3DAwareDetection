import unittest
import numpy as np
from nav.lastmile.split import SplitNav, scope, solid, BLOCKED_M
from nav.lastmile.medial import MedialNav, STEP_M


def obj(x, z):
    return np.array([[x + dx, 0.5, z + dz] for dx in (0, .1, .2) for dz in (0, .1, .2)], np.float32)


class T(unittest.TestCase):
    def test_nearest_accepted(self):
        s = SplitNav()
        ident, g, src = s.goal(np.array([5.0, 5.0]), [(0, obj(0, 0)), (3, obj(5.3, 5.2))], 7)
        self.assertEqual((ident, src, g.shape[1]), (3, "accepted_object", 2))

    def test_far_falls_back_to_frozen_point(self):
        s = SplitNav()
        i1, g1, src = s.goal(np.array([5.0, 5.0]), [(0, obj(0, 0))], 7)
        self.assertEqual((i1, src), (("point", 7), "point"))
        i2, g2, _ = s.goal(np.array([5.2, 5.0]), [], 7)       # within 0.30: frozen, same array
        self.assertTrue(np.array_equal(g1, g2))
        i3, g3, _ = s.goal(np.array([5.5, 5.0]), [], 7)       # 0.5 from the frozen point: moves
        self.assertFalse(np.array_equal(g1, g3))
        i4, g4, _ = s.goal(np.array([5.5, 5.0]), [], 8)       # new cluster: new point
        self.assertEqual(i4, ("point", 8))

    def test_non_motion_passes(self):
        target = {"cluster_id": 1, "accepted": True, "heading_to_object": False, "x": 0, "y": 0}
        self.assertIsNone(SplitNav().act("stop", target, None, None, None, None)[0])
        target["accepted"] = False
        self.assertIsNone(SplitNav().act("move_forward", target, None, None, None, None)[0])

    def test_scope_rule(self):
        target = {"accepted": False, "heading_to_object": False}
        self.assertIsNone(scope(target))
        self.assertEqual(scope({**target, "accepted": True}), "accepted")
        self.assertEqual(scope({**target, "heading_to_object": True}), "object")
        self.assertEqual(scope({"accepted": True, "heading_to_object": True}), "accepted")

    def test_info_and_passthrough(self):
        target = {"cluster_id": -1, "accepted": False, "heading_to_object": False, "x": 0, "y": 0,
                  "log": {"final_state": 0}}
        out, info = SplitNav().act("move_forward", target, None, None, None, None)
        self.assertIsNone(out)
        self.assertIsNone(info["scope"])
        self.assertEqual(info["final_state"], 0)

    def test_solid_drops_specks(self):
        main = np.array([[x * 0.02, 0.5, 0.0] for x in range(40)])
        speck = np.array([[5.0, 0.5, 0.0], [5.01, 0.5, 0.0]])
        kept = solid(np.vstack([main, speck]))
        self.assertEqual(len(kept), len(main))

    def test_blocked_m(self):
        self.assertAlmostEqual(MedialNav().blocked_m, STEP_M - 0.05)
        self.assertAlmostEqual(MedialNav(blocked_m=BLOCKED_M).blocked_m, STEP_M - 0.02)
        self.assertGreater(BLOCKED_M, 0.22)        # a 0.22 m forward counts as failed


if __name__ == "__main__":
    unittest.main()
