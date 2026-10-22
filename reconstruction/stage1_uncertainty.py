"""Uncertainty over SAM3D's stage-1 reconstructions (sparse structure + layout).

``ensemble``: stage 1 rerun from other noise seeds; the spread of the decoded
occupancies is a direct sample of p(shape | image). It reports occupancy
probability on SAM3D's 64^3 grid (object frame, before the layout pose) and
``disagreement = 1 - mean pairwise F-score`` of the sampled occupancies at a
``tolerance`` of 1 voxel, in [0, 1] (lower = one consistent shape). Plain IoU is
not used: the decoded occupancy is a thin shell, and two seeds of the same shape
overlap only ~0.5 IoU but ~0.95 within one voxel. Layout variances are in
SAM3D's pose-token space.
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


def ensemble(recorder, decoder, dtype, seed, size):
    """Seed-ensemble summary: the recorded run (``seed``) plus ``size - 1`` reruns of its
    stage-1 call from seeds ``seed + 1000 k``, which reuse its image conditioning."""
    import torch

    draws = [recorder.output] + [recorder.resample(seed + 1000 * k, dtype) for k in range(1, size)]
    shapes = torch.cat([draw[SHAPE] for draw in draws])
    layout = {key: torch.cat([draw[key] for draw in draws]).float().var(dim=0)
              for key in draws[0] if key != SHAPE}
    return {**summarize(occupancy(decoder, shapes, dtype)), "layout_variance": pose_variance(layout)}


def save_ensemble(path, summary, size):
    """Write ``stage1.npz`` in the layout ``scripts.reconstruction_uncertainty`` reads."""
    import json

    scalars = {"ensemble": {key: value for key, value in summary.items()
                            if key not in ("probability", "samples")}, "seeds": size}
    np.savez_compressed(path, scalars=json.dumps(scalars), ensemble_probability=summary["probability"],
                        ensemble_samples=summary["samples"])


def is_data_at_one(generator):
    """True if the generator's time runs noise (0) -> data (1), as the uncertainty code assumes."""
    return not generator.reversed_timestamp and math.isclose(getattr(generator, "sigma_min", 0.0), 0.0)


# ---------------------------------------------------------------- last-layer Laplace
# Generative uncertainty (Jazbec et al., UAI 2025, arXiv:2502.20946): a Laplace
# posterior over the last layer, diagonal Fisher of the training loss; for a
# fixed noise seed, rerun stage 1 with M posterior weight draws and measure the
# spread of the outputs. Here the last layer is the shape head (Linear 1024 -> 8)
# and the training data is replaced by SAM3D's own samples (sampled Fisher; the
# training set is not public).

def shape_head(generator):
    """Stage 1's last layer for the shape latent, ``Linear(1024, 8)``."""
    return generator.reverse_fn.backbone.latent_mapping[SHAPE].out_layer


@contextmanager
def captured_input(layer):
    """Collects the input of every ``layer`` forward in a list."""
    inputs = []
    handle = layer.register_forward_hook(lambda module, args, output: inputs.append(args[0].detach()))
    try:
        yield inputs
    finally:
        handle.remove()


def head_gradient(features, residual):
    """Per-example gradient of ``sum((head(features) - target)^2) / 2`` w.r.t. (weight, bias).

    The Gaussian negative log-likelihood with unit noise; divide by sigma^2 for
    other noise. ``features`` (B, T, 1024) is the head's input, ``residual``
    (B, T, 8) its output minus the target. Returns (B, 8, 1024) and (B, 8); no
    backward pass through the transformer is needed for the last layer.
    """
    import torch

    return (torch.einsum("btc,btf->bcf", residual.float(), features.float()),
            residual.float().sum(dim=1))


def fisher_terms(recorder, reverse_fn, head, times, dtype, seed=0):
    """Squared head gradients of the flow-matching loss at ``times``, own sample as data.

    Uses the recorded stage-1 call: x1 = its output, x0 ~ N(0, I),
    x_t = t x1 + (1 - t) x0, target velocity x1 - x0, conditional (no CFG),
    as in training. Returns summed (weight, bias) squared gradients of the
    unit-noise Gaussian NLL, the summed squared residual, and its element count.
    """
    import torch

    args, kwargs = recorder.call
    args, kwargs = args[2:], dict(kwargs)  # drop (x_shape, device)
    x1 = recorder.output
    saved = (reverse_fn.strength, getattr(reverse_fn, "strength_pm", None))
    reverse_fn.strength = 0.0
    if saved[1] is not None:
        reverse_fn.strength_pm = 0.0
    generator = torch.Generator(device=x1[SHAPE].device).manual_seed(seed)
    weight = bias = squared = 0.0
    try:
        for t in times:
            x0 = {key: torch.randn(value.shape, generator=generator, device=value.device)
                  for key, value in x1.items()}
            x_t = {key: t * x1[key].float() + (1 - t) * x0[key] for key in x1}
            t_in = torch.tensor([t * recorder.generator.time_scale], device=x1[SHAPE].device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype), \
                    captured_input(head) as features:
                v = reverse_fn(x_t, t_in, *args, d=torch.zeros_like(t_in), **kwargs)
            residual = v[SHAPE].float() - (x1[SHAPE].float() - x0[SHAPE])
            g_weight, g_bias = head_gradient(features[-1], residual)
            weight = weight + (g_weight ** 2).sum(dim=0)
            bias = bias + (g_bias ** 2).sum(dim=0)
            squared = squared + float((residual ** 2).sum())
    finally:
        reverse_fn.strength = saved[0]
        if saved[1] is not None:
            reverse_fn.strength_pm = saved[1]
    return weight, bias, squared, len(times) * residual.numel()


def evidence_prior(fisher, count, data_size, parameters):
    """Prior precision maximizing the Laplace evidence (MacKay; laplace-torch's default).

    For an isotropic prior tau on the MAP ``parameters`` and diagonal Hessian
    ``data_size * mean Fisher``, the tau-dependent log evidence is
    ``-tau/2 |theta|^2 + P/2 log tau - 1/2 sum log(H + tau)``; grid search over tau.
    """
    theta = np.concatenate([np.ravel(value) for value in parameters.values()]).astype(np.float64)
    hessian = np.concatenate([np.ravel(data_size * fisher[key] / count) for key in parameters])
    grid = np.logspace(-4, 8, 241)
    evidence = [-0.5 * tau * theta @ theta + 0.5 * theta.size * np.log(tau)
                - 0.5 * np.log(hessian + tau).sum() for tau in grid]
    return float(grid[int(np.argmax(evidence))])


def posterior_std(fisher, count, data_size, prior):
    """Diagonal Laplace std: precision = data_size * mean Fisher + prior.

    ``fisher`` holds summed squared NLL gradients over ``count`` examples, with
    the likelihood's noise already applied (unit-noise sums / sigma^4).
    """
    return {key: (data_size * value / count + prior) ** -0.5 for key, value in fisher.items()}


@contextmanager
def perturbed(layer, std, generator):
    """``layer`` with weight and bias drawn from N(MAP, diag std^2); restored on exit."""
    import torch

    saved = {name: getattr(layer, name).data.clone() for name in ("weight", "bias")}
    try:
        for name, value in saved.items():
            noise = torch.randn(value.shape, generator=generator, device=value.device)
            getattr(layer, name).data = value + (std[name].to(value.device) * noise).to(value.dtype)
        yield layer
    finally:
        for name, value in saved.items():
            getattr(layer, name).data = value


def latent_entropy(latents, noise=1e-3):
    """Mean per-dimension entropy of N(mean, var + noise) fitted to (M, ...) latents.

    The paper's moment-matched posterior predictive (Eq. 8) with the stage-1
    shape latent as the feature space; ``noise`` is its sigma^2. Nats per dimension.
    """
    variance = np.var(np.asarray(latents, dtype=np.float64), axis=0)
    return float(np.mean(0.5 * np.log(2 * np.pi * np.e * (variance + noise))))


# ---------------------------------------------------------------- image-shape information
# Pointwise mutual information between the condition c (image, mask, point map)
# and the reconstructed shape x: the orthogonal estimator of Kong et al.,
# "Interpretable Diffusion via Information Decomposition" (arXiv:2310.07972, Eq. 5),
# built on the I-MMSE relations of "Information-Theoretic Diffusion" (ICLR 2023):
# i(x; c) = 1/2 int E_eps |eps_hat(x_a, c) - eps_hat(x_a)|^2 da over log-SNR a,
# non-negative, with a per-dimension split (Eq. 6) used for the per-token map.
# With x_t = t x1 + (1 - t) eps: a = 2 logit(t), eps_hat = x_t - t v,
# so the integrand is t^2 |v_cond - v_uncond|^2. Low = the image barely shaped
# the reconstruction; the unconditional prior would have produced about the same.

def information(recorder, reverse_fn, samples, dtype, seed=0, loc=-2.0, scale=1.5):
    """Pointwise MI (nats) between the condition and the recorded stage-1 shape.

    ``samples`` log-SNR draws from Logistic(``loc``, ``scale``), importance
    weighted; the default puts t (= sigmoid(a / 2)) where SAM3D trains
    (logit t ~ N(-1, 1)). Returns ``information``, its ``stderr`` over draws,
    and ``per_token`` (4096,) nats, the share of each 16^3 shape token.
    """
    import torch

    args, kwargs = recorder.call
    args, kwargs = args[2:], dict(kwargs)  # drop (x_shape, device)
    # SAM3D's add_flag sets kwargs["cfg"] in place: give it a copy.
    uncond_args, uncond_kwargs = reverse_fn._make_unconditional_args(args, dict(kwargs))
    x1 = recorder.output
    device = x1[SHAPE].device
    generator = torch.Generator(device=device).manual_seed(seed)
    draws, per_token = [], 0.0
    for log_snr in np.random.default_rng(seed).logistic(loc, scale, size=samples):
        t = float(1 / (1 + np.exp(-log_snr / 2)))
        weight = 1.0 / (np.exp(-(log_snr - loc) / scale) / (scale * (1 + np.exp(-(log_snr - loc) / scale)) ** 2))
        x0 = {key: torch.randn(value.shape, generator=generator, device=device) for key, value in x1.items()}
        x_t = {key: t * x1[key].float() + (1 - t) * x0[key] for key in x1}
        t_in = torch.tensor([t * recorder.generator.time_scale], device=device)
        d_in = torch.zeros_like(t_in)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            v_cond = reverse_fn.backbone(x_t, t_in, *args, d=d_in, **kwargs)[SHAPE].float()
            v_uncond = reverse_fn.backbone(x_t, t_in, *uncond_args, d=d_in, **uncond_kwargs)[SHAPE].float()
        tokens = 0.5 * weight * t ** 2 * ((v_cond - v_uncond) ** 2).sum(dim=-1)[0]
        per_token = per_token + tokens / samples
        draws.append(float(tokens.sum()))
    return {"information": float(np.mean(draws)), "stderr": float(np.std(draws) / np.sqrt(samples)),
            "per_token": per_token.cpu().numpy().astype(np.float32)}


# ---------------------------------------------------------------- one-step mean (mean-gap)
# For any distribution E|x - x'|^2 = 2 E|x - mean|^2, so one sample and the mean
# measure the ensemble's spread. At t = 0 the input is pure noise, which carries
# nothing about the shape, so the flow model's prediction x0 + v(x0, 0) is
# E[shape | image]: one network pass instead of more seeds. With the image removed
# (SAM3D's unconditional branch) the same pass gives the prior mean E[shape]; the
# ratio |x - E[x|image]|^2 / |x - E[x]|^2 is the share of the shape's spread the
# image left unexplained, comparable across objects.

def mean_prediction(recorder, reverse_fn, dtype, mode="guided"):
    """One-step mean shape latent from the recorded noise x0 (t = 0, no shortcut).

    ``mode``: "guided" (the sampler's classifier-free guidance), "unguided" (the
    conditional branch alone, E[shape | image]) or "prior" (image removed, E[shape]).
    """
    import torch

    t0, x0 = recorder.steps[0]
    if t0 != 0.0:
        raise RuntimeError(f"first recorded step is t = {t0}, not 0")
    args, kwargs = recorder.call
    args, kwargs = args[2:], dict(kwargs)  # drop (x_shape, device)
    if mode == "prior":
        # SAM3D's add_flag sets kwargs["cfg"] in place: give it a copy.
        args, kwargs = reverse_fn._make_unconditional_args(args, dict(kwargs))
    saved = (reverse_fn.strength, getattr(reverse_fn, "strength_pm", None))
    if mode != "guided":
        reverse_fn.strength = 0.0
        if saved[1] is not None:
            reverse_fn.strength_pm = 0.0
    try:
        t_in = torch.zeros(1, device=x0[SHAPE].device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            network = reverse_fn.backbone if mode == "prior" else reverse_fn
            v = network(x0, t_in, *args, d=torch.zeros_like(t_in), **kwargs)
    finally:
        reverse_fn.strength = saved[0]
        if saved[1] is not None:
            reverse_fn.strength_pm = saved[1]
    return {key: x0[key].float() + v[key].float() for key in v}
