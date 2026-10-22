"""Last-layer Laplace bookkeeping (CPU torch, sam3d env)."""
import importlib.util
import unittest
import unittest.mock

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "needs torch (sam3d env)")
class LastLayerLaplaceTests(unittest.TestCase):
    def test_head_gradient_matches_autograd(self):
        import torch
        from reconstruction.stage1_uncertainty import head_gradient

        torch.manual_seed(0)
        head = torch.nn.Linear(16, 8)
        features, target = torch.randn(3, 5, 16), torch.randn(3, 5, 8)
        weight, bias = head_gradient(features, head(features).detach() - target)
        for b in range(3):
            head.zero_grad()
            ((head(features[b]) - target[b]) ** 2).sum().div(2).backward()
            self.assertTrue(torch.allclose(weight[b], head.weight.grad, atol=1e-6))
            self.assertTrue(torch.allclose(bias[b], head.bias.grad, atol=1e-6))

    def test_perturbed_restores_the_map_weights(self):
        import torch
        from reconstruction.stage1_uncertainty import perturbed

        head = torch.nn.Linear(4, 2)
        saved = head.weight.detach().clone()
        std = {"weight": torch.full((2, 4), 0.5), "bias": torch.full((2,), 0.5)}
        with perturbed(head, std, torch.Generator().manual_seed(0)):
            self.assertFalse(torch.equal(head.weight, saved))
        self.assertTrue(torch.equal(head.weight, saved))

    def test_posterior_std_and_entropy(self):
        import torch
        from reconstruction.stage1_uncertainty import latent_entropy, posterior_std

        std = posterior_std({"weight": torch.tensor([0.0, 6.0])}, count=2, data_size=1.0, prior=1.0)
        self.assertTrue(torch.allclose(std["weight"], torch.tensor([1.0, 0.5])))  # 1 / sqrt(1), 1 / sqrt(3 + 1)
        same = latent_entropy(np.zeros((4, 10)), noise=1e-3)
        self.assertAlmostEqual(same, 0.5 * np.log(2 * np.pi * np.e * 1e-3))
        self.assertGreater(latent_entropy(np.random.default_rng(0).normal(size=(4, 10))), same)


@unittest.skipUnless(HAS_TORCH, "needs torch (sam3d env)")
class InformationTests(unittest.TestCase):
    @staticmethod
    def _model(effect):
        """Fake recorded stage 1 whose velocity moves by ``effect`` when conditioned."""
        import torch
        from types import SimpleNamespace

        class Backbone:
            def __call__(self, x, t, *args, d=None, cfg=False, **kwargs):
                return {"shape": x["shape"] * 0.5 + (0.0 if cfg else effect)}

        def add_flag(args, kwargs):  # In place, as SAM3D's classifier_free_guidance.add_flag.
            kwargs["cfg"] = True
            return args, kwargs

        reverse_fn = SimpleNamespace(backbone=Backbone(), _make_unconditional_args=add_flag)
        recorder = SimpleNamespace(call=((None, None, "image"), {}), output={"shape": torch.zeros(1, 6, 8)},
                                   generator=SimpleNamespace(time_scale=1000.0))
        return recorder, reverse_fn

    def test_an_ignored_condition_carries_no_information(self):
        import torch
        from reconstruction.stage1_uncertainty import information

        with unittest.mock.patch("torch.autocast"):
            result = information(*self._model(0.0), samples=8, dtype=torch.float32)
        self.assertEqual(result["information"], 0.0)

    def test_a_stronger_condition_carries_more(self):
        import torch
        from reconstruction.stage1_uncertainty import information

        with unittest.mock.patch("torch.autocast"):
            weak = information(*self._model(0.1), samples=16, dtype=torch.float32)
            strong = information(*self._model(1.0), samples=16, dtype=torch.float32)
        self.assertGreater(strong["information"], weak["information"])
        self.assertEqual(weak["per_token"].shape, (6,))


if __name__ == "__main__":
    unittest.main()
