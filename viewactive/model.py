"""ViewActive network and the public predictor.

Concatenated frozen-encoder features feed three independent 3-layer MLP
decoders (~0.5M parameters each at hidden 320), one per VQF channel; their
outputs form the field ``[3, E, A]`` in (0, 1), relative azimuth x absolute
elevation (see ``viewactive.grid``)::

    model = ViewActive.load("outputs/viewactive/pilot/model/best.pt")
    field = model.predict([image])           # [1, 3, 11, 12]
    best = model.combined(field)             # [1, 11, 12]
"""
import numpy as np
import torch
from torch import nn

from viewactive.grid import Grid, combine, interpolate
from viewactive.labels import CHANNELS


class Decoder(nn.Module):
    def __init__(self, dim, hidden, cells):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                    nn.ReLU(), nn.Linear(hidden, cells))

    def forward(self, x):
        return self.layers(x)


class ViewActiveNet(nn.Module):
    def __init__(self, dim, hidden, shape, channels=len(CHANNELS)):
        super().__init__()
        self.shape = tuple(shape)
        self.register_buffer("feature_mean", torch.zeros(dim))
        self.register_buffer("feature_std", torch.ones(dim))
        self.decoders = nn.ModuleList(Decoder(dim, hidden, shape[0] * shape[1]) for _ in range(channels))

    def forward(self, features):
        x = (features - self.feature_mean) / self.feature_std
        return torch.stack([torch.sigmoid(d(x)).view(-1, *self.shape) for d in self.decoders], dim=1)


class ViewActive:
    """Single-image viewpoint-quality predictor (encoders + trained decoders)."""

    def __init__(self, net, encoders, grid, channels=CHANNELS):
        self.net, self.encoders, self.grid, self.channels = net, encoders, grid, channels

    @classmethod
    def load(cls, checkpoint, device="cuda"):
        from viewactive.features import Encoders

        state = torch.load(checkpoint, map_location=device, weights_only=False)
        grid = Grid(tuple(state["grid"]["azimuths"]), tuple(state["grid"]["elevations"]))
        net = ViewActiveNet(state["dim"], state["hidden"], grid.shape).to(device)
        net.load_state_dict(state["state_dict"])
        encoders = Encoders(*state["encoders"], device=device)
        return cls(net.eval(), encoders, grid, tuple(state["channels"]))

    def predict(self, images, masks=None):
        """``[B, 3, E, A]`` numpy field for PIL images (background outside ``masks`` whitened)."""
        from viewactive.features import prepare

        if masks is None:
            masks = [None] * len(images)
        elif len(masks) != len(images):
            raise ValueError("masks must have one entry per image")
        features = self.encoders.encode([prepare(i, m) for i, m in zip(images, masks)])
        with torch.inference_mode():
            return self.net(features).cpu().numpy()

    def combined(self, field, weights=None):
        return combine(field, weights)

    def score_views(self, field, rel_azimuths, elevations, weights=None):
        """Interpolated combined quality of one field ``[3, E, A]`` at each (rel azimuth, elevation)."""
        combined = combine(field, weights)
        return np.array([interpolate(combined, self.grid, a, e) for a, e in zip(rel_azimuths, elevations)])
