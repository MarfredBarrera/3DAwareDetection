"""Frame corrections must preserve handedness, mesh geometry, and camera up."""
import unittest

import numpy as np

from harness.upright import corrected_record, frame_rotation, UP_DIRECTIONS


class UprightFrameTests(unittest.TestCase):
    def test_composed_raw_transform_matches_rigid_correction_and_rebases_mesh(self):
        raw = np.array([[0., 0., 0.], [3., 0., 0.], [0., 2., 0.], [0., 0., 1.]])
        old_rotation = frame_rotation([0, 0, -1], [0, 1, 0])
        old = {"rotation": old_rotation.tolist(), "translation": [2., 3., -1.], "scale": 0.25}
        old_vertices = (raw @ old_rotation.T) * old["scale"] + old["translation"]
        review = {"up": [1, 0, 0], "front": [0, -1, 0], "status": "verified"}
        rec = corrected_record(old, old_vertices, review)
        actual = raw @ np.array(rec["rotation"]).T * rec["scale"] + rec["translation"]
        expected = old_vertices @ frame_rotation(review["up"], review["front"]).T
        lo, hi = expected.min(0), expected.max(0)
        expected -= [(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2]
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        self.assertAlmostEqual(actual[:, 1].min(), 0.)
        self.assertEqual(rec["scale"], old["scale"])
        np.testing.assert_allclose(
            np.linalg.norm(actual[:, None] - actual[None, :], axis=-1),
            np.linalg.norm(old_vertices[:, None] - old_vertices[None, :], axis=-1))

    def test_all_six_up_directions_are_proper_rotations(self):
        for up in UP_DIRECTIONS.values():
            with self.subTest(up=up):
                rotation = frame_rotation(up)
                np.testing.assert_allclose(rotation @ up, [0, 1, 0], atol=1e-7)
                np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-7)
                self.assertAlmostEqual(np.linalg.det(rotation), 1.0)

    def test_tilted_up_and_front_are_aligned_without_reflection(self):
        up = np.array([1., 1., 0.]) / np.sqrt(2)
        front = np.array([0., 0., -1.])
        rotation = frame_rotation(up, front)
        np.testing.assert_allclose(rotation @ up, [0, 1, 0], atol=1e-7)
        np.testing.assert_allclose(rotation @ front, [0, 0, 1], atol=1e-7)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0)

    def test_degenerate_frames_are_rejected(self):
        for up, front in [([0, 0, 0], None), ([0, 1, 0], [0, -1, 0]),
                          ([0, float("nan"), 1], None)]:
            with self.subTest(up=up, front=front), self.assertRaises(ValueError):
                frame_rotation(up, front)


if __name__ == "__main__":
    unittest.main()
