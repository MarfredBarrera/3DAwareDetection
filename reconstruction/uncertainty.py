"""Closed-form (linearized Laplace) uncertainty of a SAM3D Gaussian reconstruction.

The input image is the only observation of the reconstruction. Treating SAM3D's
Gaussians theta as the estimate it supports, the Gauss-Newton information from
the input view is ``F = J_in^T J_in`` with ``J_in = d render(input camera) / d theta``
(RGB and alpha). Its diagonal is estimated with Rademacher probes:
``E[(J^T v)^2] = diag(J^T J)``. With an isotropic prior per parameter group
(``lambda_g = prior * mean F_g``), each parameter's posterior variance keeps the
fraction ``r = lambda_g / (F + lambda_g)`` of its prior variance: 0 means the
input pins it down, 1 means only SAM3D's prior decides it.

Per candidate view:
- ``unseen``: the per-Gaussian mean of ``r`` alpha-composited into that view and
  averaged over its foreground, in [0, 1] (lower = the view shows what the
  input constrains). The composited image is the uncertainty map.
- ``trace``: ``tr(J_c Sigma J_c^T)`` in units of the prior, the predicted
  variance of the candidate render summed over pixels (probe estimate).

Runs in the ``sam3d`` env (gsplat, utils3d) on a GPU.
"""
import math

import numpy as np

from tools.cameras import SAM3D_FOV, SAM3D_INPUT_VIEW, sam3d_gravity

GROUPS = ("means", "log_scales", "quats", "opacity_logits", "sh")


def load(path, device="cuda"):
    """Unconstrained leaf tensors of a ``gaussians.npz`` from ``save_gaussian_npz``."""
    import torch

    data = np.load(path)
    opacity = np.clip(data["opacities"], 1e-6, 1 - 1e-6)
    raw = {"means": data["means"], "log_scales": np.log(np.maximum(data["scales"], 1e-12)),
           "quats": data["quats"], "opacity_logits": np.log(opacity / (1 - opacity)),
           "sh": data["sh"]}
    params = {name: torch.tensor(value, dtype=torch.float32, device=device).requires_grad_(True)
              for name, value in raw.items()}
    return params, int(data["sh_degree"])


def camera(azimuth, elevation, distance, resolution, input_elevation=None, device="cuda"):
    """(viewmat, K) of a candidate camera, as ``reconstruction.sam3d.render_view`` builds it."""
    import torch
    import utils3d

    yaw, pitch = math.radians(azimuth), math.radians(elevation)
    origin = distance * np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch),
                                  math.cos(yaw) * math.cos(pitch)])
    up = np.array([0.0, 1.0, 0.0]) if input_elevation is None else sam3d_gravity(input_elevation)
    if np.linalg.norm(np.cross(origin / distance, up)) < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
    viewmat = utils3d.torch.extrinsics_look_at(
        torch.tensor(origin, dtype=torch.float32, device=device), torch.zeros(3, device=device),
        torch.tensor(up, dtype=torch.float32, device=device))
    fov = torch.deg2rad(torch.tensor(SAM3D_FOV, device=device))
    K = utils3d.torch.intrinsics_from_fov_xy(fov, fov).clone()
    K[:2] *= resolution
    return viewmat, K


def render(params, sh_degree, view, resolution, colors=None):
    """(H, W, C) colour and (H, W) alpha on a black background; differentiable."""
    import torch.nn.functional as functional
    from gsplat import rasterization

    viewmat, K = view
    colors_in = params["sh"] if colors is None else colors
    image, alpha, _ = rasterization(
        means=params["means"], quats=functional.normalize(params["quats"], dim=-1),
        scales=params["log_scales"].exp(), opacities=params["opacity_logits"].sigmoid(),
        colors=colors_in, sh_degree=sh_degree if colors is None else None,
        viewmats=viewmat[None], Ks=K[None], width=resolution, height=resolution)
    return image[0], alpha[0, ..., 0]


def _probe_gradients(params, sh_degree, view, resolution, probes, generator):
    """Yield ``J^T v`` per group for Rademacher probes v over the render's RGB+alpha."""
    import torch

    leaves = [params[name] for name in GROUPS]
    for _ in range(probes):
        image, alpha = render(params, sh_degree, view, resolution)
        output = torch.cat([image, alpha[..., None]], dim=-1)
        signs = torch.randint(0, 2, output.shape, generator=generator, device=output.device)
        grads = torch.autograd.grad((output * (2.0 * signs - 1)).sum(), leaves, allow_unused=True)
        yield {name: torch.zeros_like(leaf) if grad is None else grad
               for name, leaf, grad in zip(GROUPS, leaves, grads)}


def information(params, sh_degree, view, resolution, probes=16, seed=0):
    """Probe estimate of diag(J^T J) for one camera, per parameter group."""
    import torch

    generator = torch.Generator(device=params["means"].device).manual_seed(seed)
    total = {name: torch.zeros_like(params[name]) for name in GROUPS}
    for grads in _probe_gradients(params, sh_degree, view, resolution, probes, generator):
        for name in GROUPS:
            total[name] += grads[name].detach() ** 2
    return {name: value / probes for name, value in total.items()}


def remaining_variance(fisher, prior=0.01):
    """Per-parameter posterior/prior variance ``lambda_g / (F + lambda_g)`` in (0, 1]."""
    ratios = {}
    for name, value in fisher.items():
        scale = prior * float(value.mean())
        ratios[name] = (scale / (value + scale) if scale > 0
                        else value.new_ones(value.shape))
    return ratios


def per_gaussian(ratios):
    """Mean remaining-variance fraction over each Gaussian's parameters, (N,)."""
    import torch

    columns = [value.reshape(value.shape[0], -1) for value in ratios.values()]
    return torch.cat(columns, dim=1).mean(dim=1)


def score_reconstruction(path, candidates, distance, resolution, input_elevation,
                         probes=16, prior=0.01, seed=0):
    """Uncertainty of one reconstruction from its input view.

    ``candidates`` are (azimuth, elevation) pairs in the SAM3D frame. Returns
    ``(gaussian_u, rows)``: per-Gaussian remaining-variance fraction (N,) and,
    per candidate, ``{"azimuth", "elevation", "unseen", "trace", "map", "alpha"}``
    with the uncertainty map and alpha as (H, W) float arrays.
    """
    import torch

    params, sh_degree = load(path)
    views = {pair: camera(*pair, distance, resolution, input_elevation) for pair in candidates}
    source = views.get(SAM3D_INPUT_VIEW) or camera(*SAM3D_INPUT_VIEW, distance, resolution,
                                                   input_elevation)
    ratios = remaining_variance(information(params, sh_degree, source, resolution, probes, seed),
                                prior)
    gaussian_u = per_gaussian(ratios)
    generator = torch.Generator(device=params["means"].device).manual_seed(seed + 1)
    rows = []
    for (azimuth, elevation), view in views.items():
        with torch.no_grad():
            weighted, alpha = render(params, sh_degree, view, resolution,
                                     colors=gaussian_u[:, None].expand(-1, 3).contiguous())
        weighted, alpha = weighted[..., 0], alpha
        unseen = float(weighted.sum() / alpha.sum()) if float(alpha.sum()) > 0 else None
        trace = 0.0
        for grads in _probe_gradients(params, sh_degree, view, resolution, probes, generator):
            # ratios = posterior variance in units of each group's prior variance.
            trace += sum(float((grads[name] ** 2 * ratios[name]).sum()) for name in GROUPS)
        rows.append({"azimuth": azimuth, "elevation": elevation, "unseen": unseen,
                     "trace": trace / probes,
                     "map": (weighted / alpha.clamp_min(1e-6)).cpu().numpy() * (alpha > 0.5).cpu().numpy(),
                     "alpha": alpha.cpu().numpy()})
    return gaussian_u.detach().cpu().numpy(), rows
