"""Ground-truth shape error: camera-frame alignment and scores."""
import math
import unittest

import numpy as np

from reconstruction.gt_error import camera_rotation, icp, shape_error
from tools.cameras import direction


def _box():
    """An asymmetric box (vertices, faces), so a wrong rotation shows."""
    corners = np.array([[x, y, z] for x in (0, 0.6) for y in (0, 1.0) for z in (0, 0.3)], float)
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return corners, faces


class CameraRotationTests(unittest.TestCase):
    def test_input_view_is_the_identity(self):
        # (180, 0) puts the camera on -Z looking at +Z with +Y up: SAM3D's own frame.
        self.assertTrue(np.allclose(camera_rotation(180, 0), np.eye(3)))

    def test_camera_maps_onto_minus_z_with_gravity_in_the_yz_plane(self):
        for azimuth, elevation in [(0, 0), (90, 30), (240, 60)]:
            rotation = camera_rotation(azimuth, elevation)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.0)
            self.assertTrue(np.allclose(rotation @ direction(azimuth, elevation), [0, 0, -1], atol=1e-6))
            up = rotation @ np.array([0.0, 1.0, 0.0])
            self.assertAlmostEqual(up[0], 0.0)
            self.assertTrue(np.allclose(up, [0, math.cos(math.radians(elevation)),
                                             -math.sin(math.radians(elevation))], atol=1e-6))


class ShapeErrorTests(unittest.TestCase):
    def test_reconstruction_in_the_camera_frame_scores_perfectly(self):
        vertices, faces = _box()
        predicted = (2.0 * vertices @ camera_rotation(240, 30).T + 0.3, faces)  # any scale and offset
        error = shape_error(predicted, (vertices, faces), 240, 30)
        self.assertGreater(error["fscore_0.02_camera"], 0.99)
        self.assertLess(error["chamfer"], 0.01)

    def test_scaled_scores_undo_a_wrong_largest_extent(self):
        vertices, faces = _box()
        # An extra thin part stretches the reconstruction's largest extent by 25%.
        extra = np.array([[0.25, 1.0, 0.1], [0.35, 1.0, 0.1], [0.3, 1.25, 0.2]])
        predicted = (np.vstack([vertices, extra]), np.vstack([faces, [[8, 9, 10]]]))
        error = shape_error(predicted, (vertices, faces), 180, 0)
        self.assertGreater(error["fscore_0.02_scaled"], error["fscore_0.02"] + 0.1)
        self.assertGreater(error["scale"], 1.0)

    def test_icp_recovers_a_small_rotation(self):
        points = np.random.default_rng(0).random((2000, 3)) * [1.0, 0.6, 0.3]
        angle = math.radians(8)
        turn = np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0],
                         [-math.sin(angle), 0, math.cos(angle)]])
        aligned, _ = icp(points @ turn.T, points)
        self.assertLess(np.abs(aligned - points).max(), 1e-3)


if __name__ == "__main__":
    unittest.main()
