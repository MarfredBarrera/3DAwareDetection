"""Paper Eq. (4): L = 0.3 L1 + 0.4 L_DSSIM + 0.3 L_SILog over the VQF grid.

Fields are ``[B, C, E, A]``; ``mask`` marks defined targets. SSIM treats each
channel as an image with circular azimuth and replicated elevation borders.
"""
import torch
import torch.nn.functional as F


def l1(pred, target, mask):
    return (pred - target).abs()[mask].mean()


def _pad(x):
    return F.pad(F.pad(x, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1), mode="replicate")


def dssim(pred, target, mask, c1=0.01 ** 2, c2=0.03 ** 2):
    """(1 - SSIM) / 2 with a 3x3 uniform window; undefined targets copy the
    (detached) prediction, so they add no gradient."""
    target = torch.where(mask, target, pred.detach())
    mu_p, mu_t = F.avg_pool2d(_pad(pred), 3, 1), F.avg_pool2d(_pad(target), 3, 1)
    var_p = F.avg_pool2d(_pad(pred * pred), 3, 1) - mu_p ** 2
    var_t = F.avg_pool2d(_pad(target * target), 3, 1) - mu_t ** 2
    cov = F.avg_pool2d(_pad(pred * target), 3, 1) - mu_p * mu_t
    ssim = ((2 * mu_p * mu_t + c1) * (2 * cov + c2)) / ((mu_p ** 2 + mu_t ** 2 + c1) * (var_p + var_t + c2))
    return ((1 - ssim) / 2).clamp(0, 1).mean()


def silog(pred, target, mask, lam=0.85, eps=1e-3):
    """Scale-invariant log loss per (sample, channel) over its N valid views."""
    d = torch.where(mask, torch.log(pred.clamp_min(eps)) - torch.log(target.clamp_min(eps)),
                    torch.zeros_like(pred))
    n = mask.flatten(2).sum(-1).clamp_min(1)
    first = (d * d).flatten(2).sum(-1) / n
    second = d.flatten(2).sum(-1) ** 2 / n ** 2
    return (first - lam * second).mean()


def total(pred, target, mask, weights=(0.3, 0.4, 0.3), lam=0.85):
    """(weighted loss, {"l1", "dssim", "silog"} as floats)."""
    target = torch.where(mask, target, torch.ones_like(target))  # NaN-free for autograd
    parts = {"l1": l1(pred, target, mask), "dssim": dssim(pred, target, mask),
             "silog": silog(pred, target, mask, lam)}
    loss = sum(w * v for w, v in zip(weights, parts.values()))
    return loss, {k: float(v.detach()) for k, v in parts.items()}
