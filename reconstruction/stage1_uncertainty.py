"""Seed-ensemble uncertainty of SAM3D's stage-1 reconstruction (sparse structure + layout).

Stage 1 is rerun from other noise seeds; the spread of the decoded
occupancies is a direct sample of p(shape | image). It reports occupancy
probability on SAM3D's 64^3 grid (object frame, before the layout pose) and
``disagreement = 1 - mean pairwise F-score`` of the sampled occupancies at a
``tolerance`` of 1 voxel, in [0, 1] (lower = one consistent shape). Plain IoU is
not used: the decoded occupancy is a thin shell, and two seeds of the same shape
overlap only ~0.5 IoU but ~0.95 within one voxel. Layout variances are in
SAM3D's pose-token space.
"""
import itertools
import json
from contextlib import contextmanager

import numpy as np

SHAPE = "shape"


class Recorder:
    """Records the first stage-1 generator call so it can be rerun from other seeds."""

    def __init__(self, generator):
        self.generator = generator
        self.call = None        # (args, kwargs) of the first generator forward
        self.output = None      # its x1
        self.enabled = True

    @contextmanager
    def attached(self):
        generator = self.generator
        forward = generator.forward

        def record_forward(*args, **kwargs):
            first = self.enabled and self.call is None
            result = forward(*args, **kwargs)
            if first:
                self.call, self.output = (args, kwargs), _detach(result)
            return result

        generator.forward = record_forward
        try:
            yield self
        finally:
            del generator.forward

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
    """Write ``stage1.npz`` (read back with ``load_ensemble``)."""
    scalars = {"ensemble": {key: value for key, value in summary.items()
                            if key not in ("probability", "samples")}, "seeds": size}
    np.savez_compressed(path, scalars=json.dumps(scalars), ensemble_probability=summary["probability"],
                        ensemble_samples=summary["samples"])


def load_ensemble(path):
    """The scalars ``save_ensemble`` wrote: {"ensemble": {"disagreement", ...}, "seeds"}."""
    with np.load(path) as data:
        return json.loads(str(data["scalars"]))
