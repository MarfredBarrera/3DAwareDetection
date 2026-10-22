"""The 3D Viewpoint Quality Field (VQF) grid and its two frames.

Fields are arrays ``[..., E, A]``: elevation rows (ascending, absolute: gravity
is known) by azimuth columns. In the *absolute* frame column ``j`` is azimuth
``azimuths[j]`` in the object's canonical frame. Objaverse yaw is arbitrary, so
the model predicts the *relative* frame instead: column ``j`` is the camera
rotated by ``j`` azimuth steps from the current view (column 0 = here). Angles
follow ``tools.cameras.direction`` (y-up, azimuth 0 on +Z).
"""
from dataclasses import dataclass
import math

import numpy as np

from tools.cameras import SAM3D_INPUT_VIEW, direction


@dataclass(frozen=True)
class Grid:
    azimuths: tuple = tuple(float(a) for a in range(0, 360, 30))
    elevations: tuple = tuple(float(e) for e in range(-75, 76, 15))

    @classmethod
    def from_config(cls, render):
        return cls(tuple(map(float, render["azimuths"])), tuple(map(float, render["elevations"])))

    @property
    def shape(self):
        return len(self.elevations), len(self.azimuths)

    @property
    def azimuth_step(self):
        return 360.0 / len(self.azimuths)

    @property
    def elevation_step(self):
        return self.elevations[1] - self.elevations[0]

    def directions(self):
        """Unit camera directions ``[E, A, 3]``; relative columns use offsets from azimuth 0."""
        return np.array([[direction(a - self.azimuths[0], e) for a in self.azimuths]
                         for e in self.elevations])


def to_relative(field, az_index):
    """Absolute field -> relative to a camera at absolute column ``az_index``."""
    return np.roll(field, -int(az_index), axis=-1)


def to_absolute(field, az_index):
    return np.roll(field, int(az_index), axis=-1)


def combine(field, weights=None):
    """Mean quality across channels: ``[..., 3, E, A] -> [..., E, A]``."""
    weights = np.ones(field.shape[-3]) if weights is None else np.asarray(weights, dtype=float)
    return np.tensordot(weights / weights.sum(), np.moveaxis(field, -3, 0), axes=1)


def reachable(grid, el_index, reach=(1, 1)):
    """Boolean ``[E, A]`` mask (relative frame) of cells within ``reach`` =
    (azimuth cells, elevation cells) of the current view; ``None`` = unlimited."""
    rows, cols = grid.shape
    if reach is None:
        return np.ones((rows, cols), dtype=bool)
    offsets = np.minimum(np.arange(cols), cols - np.arange(cols))
    return ((np.abs(np.arange(rows) - el_index) <= reach[1])[:, None]
            & (offsets <= reach[0])[None, :])


def elevation_rows(grid, limits=None):
    """Boolean ``[E]`` of rows with elevation in ``limits`` = (low, high) degrees; ``None`` = all."""
    elevations = np.asarray(grid.elevations)
    if limits is None:
        return np.ones(len(elevations), dtype=bool)
    return (elevations >= limits[0]) & (elevations <= limits[1])


def toward(grid, mask, limits=None):
    """Keep the cells of ``mask`` whose rows are closest to the ``limits`` elevation band."""
    rows = np.flatnonzero(elevation_rows(grid, limits))
    gap = np.abs(np.arange(grid.shape[0])[:, None] - rows).min(1)[:, None]
    gap = np.broadcast_to(gap, mask.shape)
    return mask & (gap <= gap[mask].min())


def nearest(grid, mask, el_index, az_index):
    """The masked cell with the smallest angle to cell ``(el_index, az_index)``."""
    directions = grid.directions()
    cosine = directions @ directions[el_index, az_index]
    cosine = np.where(mask, cosine, -np.inf)
    return tuple(int(i) for i in np.unravel_index(np.argmax(cosine), mask.shape))


def interpolate(field, grid, rel_azimuth, elevation):
    """Bilinear lookup of ``field [..., E, A]`` at an angle; circular in azimuth,
    clamped in elevation. ``rel_azimuth`` is in the field's azimuth frame."""
    columns = len(grid.azimuths)
    x = ((rel_azimuth - grid.azimuths[0]) % 360) / grid.azimuth_step
    y = np.clip((elevation - grid.elevations[0]) / grid.elevation_step, 0, len(grid.elevations) - 1)
    x0, y0 = int(math.floor(x)) % columns, int(math.floor(y))
    x1, y1 = (x0 + 1) % columns, min(y0 + 1, len(grid.elevations) - 1)
    fx, fy = x - math.floor(x), y - y0
    top = (1 - fx) * field[..., y0, x0] + fx * field[..., y0, x1]
    bottom = (1 - fx) * field[..., y1, x0] + fx * field[..., y1, x1]
    return (1 - fy) * top + fy * bottom


def sam3d_absolute(azimuth, elevation, input_azimuth, input_elevation):
    """SAM3D candidate angle -> (azimuth, elevation) in the ground-truth object frame."""
    relative, absolute_elevation = sam3d_to_viewactive(azimuth, elevation, input_elevation)
    return (input_azimuth + relative) % 360.0, absolute_elevation


def sam3d_to_viewactive(azimuth, elevation, input_elevation=0.0):
    """SAM3D candidate angle -> (relative azimuth, absolute elevation).

    SAM3D candidates live in the input camera's frame: ``SAM3D_INPUT_VIEW``
    re-renders the input and the frame is tilted by the input's elevation.
    Rotate the input view onto azimuth 0, then untilt about x by the input
    elevation. The relative azimuth adds to the input's absolute azimuth (see
    ``sam3d_absolute``); verified 2026-10-19 by silhouette IoU against
    ground-truth renders (64 of 80 candidates on 6 asymmetric objects, 9 mirrored).
    """
    d = direction(azimuth - SAM3D_INPUT_VIEW[0], elevation - SAM3D_INPUT_VIEW[1]).astype(np.float64)
    tilt = math.radians(input_elevation)
    y = d[1] * math.cos(tilt) + d[2] * math.sin(tilt)
    z = -d[1] * math.sin(tilt) + d[2] * math.cos(tilt)
    return (math.degrees(math.atan2(d[0], z)) % 360.0,
            math.degrees(math.asin(float(np.clip(y, -1, 1)))))

