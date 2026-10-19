"""Laplace uncertainty bookkeeping (CPU torch; rendering needs a GPU and gsplat)."""
import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "needs torch (sam3d env)")
class RemainingVarianceTests(unittest.TestCase):
    def test_unobserved_parameters_keep_their_prior(self):
        import torch
        from reconstruction.uncertainty import per_gaussian, remaining_variance

        fisher = {"means": torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
                  "sh": torch.zeros(2, 1, 3)}
        ratios = remaining_variance(fisher, prior=0.5)
        # lambda = 0.5 * mean(F) = 0.5: unobserved -> 1, F = 2 -> 0.5 / 2.5.
        self.assertTrue(torch.allclose(ratios["means"][0], torch.ones(3)))
        self.assertTrue(torch.allclose(ratios["means"][1], torch.full((3,), 0.2)))
        self.assertTrue(torch.equal(ratios["sh"], torch.ones(2, 1, 3)))  # never observed
        self.assertTrue(torch.allclose(per_gaussian(ratios), torch.tensor([1.0, 0.6])))


if __name__ == "__main__":
    unittest.main()
