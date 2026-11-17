import unittest
from unittest.mock import patch
import numpy as np

from nav.approach import geom as G, local_nav as L, scene as S_for_test
from nav.lastmile.collision_map import EpisodeCollisionMap
from nav.lastmile.body_map import BodyMap


class CollisionMapTest(unittest.TestCase):
    def test_body_height_keeps_overhead_out_of_collision_layer(self):
        points = np.array([[1., y, z] for y in (.5, 1.2) for z in np.linspace(-1., 1., 41)])
        hm = G.HeightMap(np.zeros(2), 0., body_top=.88)
        with patch.object(G.S, "backproject", return_value=points):
            hm.update(np.ones((4, 4)), np.zeros((4, 4), bool), np.array([0., .88, 0.]), np.eye(3))
        ij, _ = hm.cells(np.array([1., 0.]))
        self.assertAlmostEqual(hm.top[tuple(ij)], 1.2, places=5)
        self.assertAlmostEqual(hm.low[tuple(ij)], .5, places=5)
        hm2 = G.HeightMap(np.zeros(2), 0., body_top=.88)
        with patch.object(G.S, "backproject", return_value=points[points[:, 1] > 1.]):
            hm2.update(np.ones((4, 4)), np.zeros((4, 4), bool), np.array([0., .88, 0.]), np.eye(3))
        lm = L.LocalMap(hm2, obstacle_low=.06)
        self.assertFalse(lm.grids()[0][lm.cell([1., 0.])])

    def test_extent_growth_retains_observations_and_collision_marks(self):
        mapping = EpisodeCollisionMap.__new__(EpisodeCollisionMap)
        mapping.hmap = G.HeightMap(np.zeros(2), 0., half=10., body_top=.88)
        mapping.lmap = BodyMap(mapping.hmap, 0.)
        hm, lm = mapping.hmap, mapping.lmap
        point = np.array([1., 2.])
        ij, _ = hm.cells(point)
        hm.low[tuple(ij)] = .5
        hm.seen[tuple(ij)] = True
        lm.bumps[lm.cell(point)] = True
        mapping._grow([9., 0.])
        ij, ok = hm.cells(point)
        self.assertTrue(ok)
        self.assertEqual(hm.low[tuple(ij)], .5)
        self.assertTrue(hm.seen[tuple(ij)])
        self.assertTrue(lm.bumps[lm.cell(point)])
        self.assertEqual(lm.n, hm.n//lm.k)


class BodyMapTest(unittest.TestCase):
    def mapping(self):
        return BodyMap(G.HeightMap(np.zeros(2), 0., half=2.), 0.)

    def update_points(self, m, points, base):
        with patch.object(S_for_test, "backproject", return_value=np.asarray(points, float).reshape(-1, 3)):
            m.update(np.ones((4, 4)), np.array([0., base+.88, 0.]), np.eye(3), base)

    def test_walkable_floor_elevation_change_does_not_remain_a_wall(self):
        m = self.mapping()
        points = [[.51, .14, .02], [.52, .14, .02], [1.01, .6, .02], [1.02, .6, .02]]
        self.update_points(m, points, 0.)
        self.assertTrue(m.grids()[0][m.cell([.51, .02])])
        self.update_points(m, points, .14)
        self.assertFalse(m.grids()[0][m.cell([.51, .02])])
        self.assertTrue(m.grids()[0][m.cell([1.01, .02])])

    def test_old_storey_contact_does_not_block_new_storey(self):
        m = self.mapping()
        m.bump(np.zeros(3), 0.)
        self.assertTrue(m.grids()[0].any())
        self.update_points(m, [], 3.)
        occupied, free, unknown = m.grids()
        self.assertFalse(occupied.any())
        self.assertFalse(free.any())
        self.assertTrue(unknown.all())

    def test_dense_hits_saturate_without_uint8_wraparound(self):
        m = self.mapping()
        self.update_points(m, [[.51, .5, .02]]*512, 0.)
        self.assertEqual(m.hits.max(), 50)
        self.assertTrue(m.grids()[0][m.cell([.51, .02])])

    def test_sensor_far_clip_is_not_an_obstacle_surface(self):
        m = self.mapping()
        m.update(np.full((4, 4), 5.), np.array([0., .88, 0.]), np.eye(3), 0.)
        self.assertFalse(m.hits.any())

    def test_agent_climb_band_ignores_small_steps_and_retains_furniture(self):
        m = BodyMap(G.HeightMap(np.zeros(2), 0., half=2.), 0., obstacle_low=.20)
        points = [[.51, .14, .02], [.52, .14, .02], [1.01, .5, .02], [1.02, .5, .02]]
        self.update_points(m, points, 0.)
        self.assertFalse(m.grids()[0][m.cell([.51, .02])])
        self.assertTrue(m.grids()[0][m.cell([1.01, .02])])

    def test_unoccluded_depth_ray_can_clear_stale_obstacle_evidence(self):
        m = BodyMap(G.HeightMap(np.zeros(2), 0., half=4.), 0.)
        self.update_points(m, [[.05, .55, -1.05]]*2, 0.)
        self.assertTrue(m.grids()[0][m.cell([.05, -1.05])])
        # A later view observes a farther surface on that ray. The earlier hit
        # cannot remain an immutable wall after measured free-space evidence.
        self.update_points(m, [[.10, .22, -2.10]]*2, 0.)
        self.assertFalse(m.grids()[0][m.cell([.05, -1.05])])
        self.assertTrue(m.grids()[0][m.cell([.10, -2.10])])

    def test_surface_positions_avoid_coarse_cell_inflation_error(self):
        m = self.mapping()
        self.update_points(m, [[.515, .5, .02]]*3, 0.)
        fine = m.planning_grid([0., 0.], half=1., res=.025)
        # The real surface is 21.6 cm away: the 18 cm body plus 1 cm
        # measurement allowance fits, although a 10 cm cell's inflated proxy does not.
        self.assertGreater(float(fine.clearance_at([.30, 0.])), .02)

    def test_surface_centroids_survive_extent_growth(self):
        m = self.mapping()
        self.update_points(m, [[.515, .5, .02]]*3, 0.)
        before = m.obstacle_points().copy()
        m.pad_xy(10)
        m.origin -= 10*L.CELL
        m.n += 20
        np.testing.assert_allclose(m.obstacle_points(), before)


if __name__ == "__main__":
    unittest.main()
