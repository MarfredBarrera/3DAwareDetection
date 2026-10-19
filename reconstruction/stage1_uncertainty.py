"""Uncertainty over SAM3D's stage-1 reconstructions (sparse structure + layout).

Two estimates of how much the coarse shape and pose could differ given one image:

- ``tweedie``: closed form from one sampling trajectory. For the interpolant
  ``x_t = t x1 + (1 - t) eps`` (t = 1 is data), Tweedie's formula gives
  ``E[x1 | x_t] = x_t + (1 - t) v`` and
  ``Cov[x1 | x_t] = ((1 - t)^2 / t) (I + (1 - t) J_v)``, ``J_v = dv/dx_t``.
  diag(J_v) comes from Rademacher probes ``z * (J_v^T z)`` (vector-Jacobian
  products; diag J = diag J^T). Shape latents drawn from N(mean, diag var) are
  decoded to voxel occupancy. It describes the spread *given* x_t, so it depends
  on t*; with a diagonal it misses coherent alternatives (sofa vs chair).
- ``ensemble``: the same stage 1 rerun from other noise seeds; the spread of
  the decoded occupancies is the direct sample of p(shape | image).

Both report occupancy probability on SAM3D's 64^3 grid (object frame, before
the layout pose) and ``disagreement = 1 - mean pairwise F-score`` of the
sampled occupancies at a ``tolerance`` of 1 voxel, in [0, 1] (lower = one
consistent shape). Plain IoU is not used: the decoded occupancy is a thin
shell, and two seeds of the same shape overlap only ~0.5 IoU but ~0.95 within
one voxel. Layout variances are in SAM3D's pose-token space.
"""
import itertools
import math
from contextlib import contextmanager

import numpy as np

SHAPE = "shape"


class Recorder:
    """Records stage-1 generator calls and the x_t seen at each solver step."""

    def __init__(self, generator):
        self.generator = generator
        self.call = None        # (args, kwargs) of the first generator forward
        self.output = None      # its x1
        self.steps = []         # [(t, x_t)], t in [0, 1] with 1 = data
        self.enabled = True

    @contextmanager
    def attached(self):
        generator = self.generator
        forward, dynamics = generator.forward, generator._generate_dynamics

        def record_forward(*args, **kwargs):
            first = self.enabled and self.call is None
            result = forward(*args, **kwargs)
            if first:
                self.call, self.output = (args, kwargs), _detach(result)
            return result

        def record_dynamics(x_t, t, *args, **kwargs):
            if self.enabled:
                self.steps.append((float(t), _detach(x_t)))
            return dynamics(x_t, t, *args, **kwargs)

        generator.forward, generator._generate_dynamics = record_forward, record_dynamics
        try:
            yield self
        finally:
            del generator.forward, generator._generate_dynamics

    def resample(self, seed, dtype):
        """Rerun the recorded stage-1 call from another noise seed; returns x1."""
        import torch

        self.enabled = False
        try:
            torch.manual_seed(seed)
            args, kwargs = self.call
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
                return _detach(self.generator(*args, **kwargs))
        finally:
            self.enabled = True


def _detach(tree):
    import optree
    return optree.tree_map(lambda x: x.detach().clone(), tree)


def _shape_grid(latent):
    """(B, 4096, 8) shape tokens -> (B, 8, 16, 16, 16) decoder input."""
    return latent.permute(0, 2, 1).contiguous().view(latent.shape[0], 8, 16, 16, 16)


def occupancy(decoder, latents, dtype):
    """Boolean (B, 64, 64, 64) occupancy, as the pipeline thresholds it (> 0)."""
    import torch

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        decoded = decoder(_shape_grid(latents))
    return (decoded[:, 0] > 0).cpu().numpy()


def disagreement(samples, tolerance=1):
    """1 - mean pairwise F-score of boolean occupancies within ``tolerance`` voxels.

    Each direction's precision is the fraction of one sample's voxels within
    ``tolerance`` of the other's; in [0, 1], None with fewer than two non-empty samples.
    """
    from scipy.ndimage import binary_dilation

    samples = [sample for sample in samples if sample.any()]
    grown = [binary_dilation(sample, iterations=tolerance) if tolerance else sample
             for sample in samples]
    scores = []
    for a, b in itertools.combinations(range(len(samples)), 2):
        precision = (samples[a] & grown[b]).sum() / samples[a].sum()
        recall = (samples[b] & grown[a]).sum() / samples[b].sum()
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return None if not scores else float(1 - np.mean(scores))


def voxel_disagreement(samples, tolerance=1):
    """Per voxel, the share of other samples with nothing within ``tolerance`` of it.

    Averaged over the samples that occupy the voxel; NaN where none does. The
    per-voxel form of ``disagreement``: 0 everywhere for one shape up to jitter.
    """
    from scipy.ndimage import binary_dilation

    samples = np.asarray(samples, dtype=bool)
    grown = np.stack([binary_dilation(sample, iterations=tolerance) if tolerance else sample
                      for sample in samples])
    unsupported = np.zeros(samples.shape[1:], dtype=np.float32)
    for i, sample in enumerate(samples):
        missing = (len(samples) - 1) - (np.delete(grown, i, axis=0).sum(axis=0))
        unsupported += sample * missing / max(1, len(samples) - 1)
    count = samples.sum(axis=0)
    return np.where(count > 0, unsupported / np.maximum(count, 1), np.nan)


def velocity_diagonal(reverse_fn, x_t, t, time_scale, args, kwargs, probes, dtype, generator):
    """Velocity at (x_t, t) and Hutchinson diag(J_v) per latent key."""
    import optree
    import torch

    t_in = torch.tensor([t * time_scale], device=x_t[SHAPE].device, dtype=torch.float32)
    d_in = torch.zeros_like(t_in)  # no shortcut: the plain flow velocity
    total = None
    for _ in range(probes):
        x = optree.tree_map(lambda leaf: leaf.detach().float().requires_grad_(True), x_t)
        with torch.enable_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            v = reverse_fn(x, t_in, *args, d=d_in, **kwargs)
        keys = sorted(v)
        z = {key: torch.randint(0, 2, v[key].shape, generator=generator,
                                device=v[key].device).float() * 2 - 1 for key in keys}
        dot = sum((v[key].float() * z[key]).sum() for key in keys)
        grads = torch.autograd.grad(dot, [x[key] for key in keys])
        estimate = {key: z[key] * grad.float() for key, grad in zip(keys, grads)}
        total = estimate if total is None else {k: total[k] + estimate[k] for k in keys}
        velocity = {key: v[key].detach().float() for key in keys}
    return velocity, {key: value / probes for key, value in total.items()}


def tweedie(recorder, reverse_fn, decoder, t_star, probes, samples, dtype, seed=0,
            guided=False):
    """Posterior mean/variance of x1 at the recorded step nearest ``t_star``.

    ``guided=False`` drops classifier-free guidance for the Jacobian (Tweedie
    assumes v is a conditional expectation; the guided velocity is not).
    Returns ``(t, mean, variance, occupancy samples)`` with dict trees.
    """
    import torch

    t, x_t = min(recorder.steps, key=lambda step: abs(step[0] - t_star))
    if not 0 < t < 1:
        raise ValueError(f"t* must be inside (0, 1); nearest step is {t}")
    args, kwargs = recorder.call
    args, kwargs = args[2:], dict(kwargs)  # drop (x_shape, device)
    saved = (reverse_fn.strength, getattr(reverse_fn, "strength_pm", None))
    if not guided:
        reverse_fn.strength = 0.0
        if saved[1] is not None:
            reverse_fn.strength_pm = 0.0
    try:
        generator = torch.Generator(device=x_t[SHAPE].device).manual_seed(seed)
        velocity, diagonal = velocity_diagonal(reverse_fn, x_t, t, recorder.generator.time_scale,
                                               args, kwargs, probes, dtype, generator)
    finally:
        reverse_fn.strength = saved[0]
        if saved[1] is not None:
            reverse_fn.strength_pm = saved[1]
    mean = {key: x_t[key].float() + (1 - t) * velocity[key] for key in velocity}
    variance = {key: ((1 - t) ** 2 / t * (1 + (1 - t) * diagonal[key])).clamp_min(0)
                for key in velocity}
    noise = torch.Generator(device=mean[SHAPE].device).manual_seed(seed + 1)
    draws = mean[SHAPE] + variance[SHAPE].sqrt() * torch.randn(
        (samples,) + tuple(mean[SHAPE].shape[1:]), generator=noise, device=mean[SHAPE].device)
    return t, mean, variance, occupancy(decoder, draws, dtype)


def summarize(occupancies):
    """Occupancy probability (uint8 0-255 on 64^3), bit-packed samples, and disagreement."""
    probability = np.mean(occupancies, axis=0)
    return {"probability": np.round(probability * 255).astype(np.uint8),
            "samples": np.packbits(np.asarray(occupancies), axis=-1),
            "disagreement": disagreement(occupancies),
            "occupied": float(np.mean([o.sum() for o in occupancies]))}


def pose_variance(tree):
    """Mean variance per non-shape latent key (layout tokens)."""
    return {key: float(value.float().mean()) for key, value in tree.items() if key != SHAPE}


def is_data_at_one(generator):
    """True if the generator's time runs noise (0) -> data (1), as Tweedie assumes here."""
    return not generator.reversed_timestamp and math.isclose(getattr(generator, "sigma_min", 0.0), 0.0)
