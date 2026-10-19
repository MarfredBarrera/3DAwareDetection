"""Paper Algorithm 1: reachable-aware progressive viewpoint optimization.

From the current view, predict the VQF, move to its best cell if reachable,
else to the reachable cell nearest it, and repeat from the new image. An
optional elevation range (not in the paper) limits targets to those rows; a
start outside it first moves straight toward the band (the nearest-cell rule
alone can stay at a pole when the target is on the far side). Views are grid cells ``(elevation index, absolute azimuth index)``; fields are in the
relative frame of ``viewactive.grid``.
"""
import numpy as np

from viewactive.grid import elevation_rows, nearest, reachable, toward


def next_view(combined, el_index, grid, reach=(1, 1), elevations=None):
    """One Algorithm 1 step on a relative combined field ``[E, A]`` -> (new elevation index,
    azimuth offset in cells). Undefined (NaN) cells and rows outside ``elevations``
    (low, high) degrees are never targets."""
    allowed = elevation_rows(grid, elevations)[:, None]
    scores = np.where(np.isfinite(combined) & allowed, combined, -np.inf)
    target = np.unravel_index(int(np.argmax(scores)), scores.shape)
    mask = toward(grid, reachable(grid, el_index, reach), elevations)
    row, col = target if mask[target] else nearest(grid, mask, *target)
    return int(row), int(col)


def field_policy(relative_field, grid, reach=(1, 1), elevations=None):
    """Policy that follows ``relative_field(e, a) -> [E, A]`` combined quality from view (e, a)."""
    def step(e, a, rng):
        row, col = next_view(relative_field(e, a), e, grid, reach, elevations)
        return row, (a + col) % grid.shape[1]
    return step


def random_policy(grid, reach=(1, 1), elevations=None):
    """Uniformly random move to another reachable view (a random walk control).
    Moves stay in the ``elevations`` rows, or get as close to them as possible."""
    def step(e, a, rng):
        mask = reachable(grid, e, reach)
        mask[e, 0] = False
        mask = toward(grid, mask, elevations)
        rows, cols = np.nonzero(mask)
        i = rng.integers(len(rows))
        return int(rows[i]), (a + int(cols[i])) % grid.shape[1]
    return step


def stay(e, a, rng):
    return e, a


def rollout(policy, start, steps, rng=None):
    """Views visited from ``start`` over ``steps`` moves (``steps + 1`` views)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    views = [tuple(start)]
    for _ in range(steps):
        views.append(policy(*views[-1], rng))
    return views
