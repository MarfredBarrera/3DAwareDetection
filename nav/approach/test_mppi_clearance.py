"""python -m unittest nav.approach.test_mppi_clearance (no Habitat needed)."""
import unittest

import numpy as np

from nav.approach.mppi import MppiV15
from nav.approach.test_local_nav import wall_map


class ClearanceTest(unittest.TestCase):
    def test_penalty_decreases_with_distance(self):
        lm, pf = wall_map()
        blocked = ~np.isfinite(pf.cost)
        self.assertTrue(blocked.any())
        i0, j0 = lm.cell([0.0, 0.0])
        first = i0 + int(np.argmax(blocked[i0:, j0]))      # first blocked cell along +x from the origin
        poses = np.zeros((3, 2, 3))
        for k, off in enumerate((1, 2)):                    # 1 and 2 cells (0.1, 0.2 m) from the inflated region
            poses[k, :, :2] = lm.center([first - off, j0])
        poses[2, :, :2] = lm.center([i0 - 20, j0])           # far away
        planner = MppiV15()
        pen = planner.clearance_penalty(poses, pf)
        self.assertGreater(pen[0], pen[1])
        self.assertGreater(pen[1], pen[2])
        self.assertAlmostEqual(pen[2], 0.0)
        self.assertLessEqual(pen.max(), planner.CLEAR_W)
        planner.CLEAR_W = 2.0
        self.assertAlmostEqual(planner.clearance_penalty(poses, pf)[0], 2 * pen[0])


if __name__ == "__main__":
    unittest.main()
