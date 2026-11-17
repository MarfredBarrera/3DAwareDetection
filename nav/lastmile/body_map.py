"""World-height occupancy projected through the current agent's body band.

A persistent map cannot classify height relative to the episode's starting floor:
walkable floor elevation changes then become walls. Keep vertical evidence and
select the collision band using the current odometry height instead.
"""
import math

import numpy as np

from nav.approach import geom as G, local_nav as L, scene as S


class BodyMap(L.LocalMap):
    DZ = .10

    def __init__(self, hmap, base_y, obstacle_low=.06):
        super().__init__(hmap, obstacle_low=obstacle_low)
        self.base_y = float(base_y)
        self.z_origin = math.floor((base_y-5.)/self.DZ)*self.DZ
        self.hits = np.zeros((self.n, self.n, 120), np.uint8)
        self.observed = np.zeros(self.hits.shape, bool)
        self.contacts = np.zeros(self.hits.shape, bool)
        self._projection = None
        self._index_offset = np.zeros(3, int)
        self._centroids = {}

    def pad_xy(self, pad):
        for attr in ("hits", "observed", "contacts"):
            setattr(self, attr, np.pad(getattr(self, attr), ((pad, pad), (pad, pad), (0, 0))))
        self._projection = None
        self._index_offset[:2] -= pad

    def _grow_z(self, lo, hi):
        n = self.hits.shape[2]
        low = max(0, int(math.ceil((self.z_origin-lo)/self.DZ/40))*40)
        high = max(0, int(math.ceil((hi-self.z_origin-n*self.DZ)/self.DZ/40))*40)
        if low or high:
            for attr in ("hits", "observed", "contacts"):
                setattr(self, attr, np.pad(getattr(self, attr), ((0, 0), (0, 0), (low, high))))
            self.z_origin -= low*self.DZ
            self._index_offset[2] -= low

    def _indices(self, points):
        p = np.asarray(points)
        ij = np.floor((p[:, [0, 2]]-self.origin)/L.CELL).astype(int)
        z = np.floor((p[:, 1]-self.z_origin)/self.DZ).astype(int)
        valid = (ij >= 0).all(1) & (ij < self.n).all(1) & (z >= 0) & (z < self.hits.shape[2])
        return ij, z, valid

    def update(self, depth, cam_pos, cam_rot, base_y, max_depth=5., stride=6):
        self.base_y = float(base_y)
        d = np.asarray(depth, np.float32)
        valid = np.isfinite(d) & (d > .05) & (d < max_depth-.01)
        points = S.backproject(d, valid, cam_pos, cam_rot, stride=stride)
        self._projection = None
        if not len(points):
            return
        self._grow_z(float(points[:, 1].min())-.1, float(points[:, 1].max())+.1)
        ij, z, ok = self._indices(points)
        flat = np.ravel_multi_index((ij[ok, 0], ij[ok, 1], z[ok]), self.hits.shape)
        ids, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
        old = self.hits.reshape(-1)
        sums = np.stack([np.bincount(inverse, weights=points[ok, axis]) for axis in range(3)], 1)
        absolute = np.array(np.unravel_index(ids, self.hits.shape)).T+self._index_offset
        for key, idx, total, count in zip(absolute, ids, sums, counts):
            key = tuple(key)
            weight = int(old[idx])
            previous = self._centroids.get(key)
            self._centroids[key] = total/count if previous is None or weight == 0 else (previous*weight+total)/(weight+count)
        # Observed free rays retain height, preventing a floor observed on one
        # storey from certifying unseen space on a different storey.
        targets = points[::4]
        lengths = np.linalg.norm(targets-cam_pos, axis=1)
        samples = np.linspace(0., 1., max(2, int(np.ceil(lengths.max()/L.CELL))+1))
        # Stop short of the measured surface. Current-frame hits are protected
        # even if another ray clips the same voxel at a depth discontinuity.
        fractions = samples[None, :]*np.maximum(1-.15/np.maximum(lengths[:, None], 1e-6), 0.)
        rays = cam_pos+fractions[:, :, None]*(targets[:, None]-cam_pos)
        ij, z, ok = self._indices(rays.reshape(-1, 3))
        self.observed[ij[ok, 0], ij[ok, 1], z[ok]] = True
        free_ids = np.unique(np.ravel_multi_index((ij[ok, 0], ij[ok, 1], z[ok]), self.hits.shape))
        misses = np.setdiff1d(free_ids, ids, assume_unique=True)
        old[misses] = np.maximum(old[misses].astype(np.int16)-2, 0).astype(np.uint8)
        old[ids] = np.minimum(50, old[ids].astype(np.int32)+counts).astype(np.uint8)

    def grids(self):
        if self._projection is None:
            z = self.z_origin+(np.arange(self.hits.shape[2])+.5)*self.DZ
            band = (z > self.base_y+self.obstacle_low) & (z < self.base_y+G.H_CAM)
            occupied = ((self.hits[:, :, band] >= 2) | self.contacts[:, :, band]).any(2)
            known = self.observed[:, :, band].any(2)
            self._projection = occupied, known & ~occupied, ~known & ~occupied
        return tuple(a.copy() for a in self._projection)

    def bump(self, position, yaw):
        # Keep collision feedback on the level where it was observed.
        super().bump(position, yaw)
        f = G.forward(yaw)
        side = np.array([-f[1], f[0]])
        xz = np.array([np.asarray(position)[[0, 2]]+(S.AGENT_RADIUS+.12)*f+t*side for t in (-.1, 0., .1)])
        points = np.c_[xz[:, 0], np.full(len(xz), position[1]+.3), xz[:, 1]]
        self._grow_z(float(points[:, 1].min())-.1, float(points[:, 1].max())+.1)
        ij, z, ok = self._indices(points)
        self.contacts[ij[ok, 0], ij[ok, 1], z[ok]] = True
        self._projection = None

    def obstacle_points(self):
        """Measured surface centroids in the body band (x, z), retaining sub-cell position precision. Depth only: bump
        contact cells are left out (a third of them lie on open floor and they close most open routes)."""
        return self._depth_points3()[:, [0, 2]]

    def _depth_points3(self):
        if not self._centroids:
            return np.zeros((0, 3))
        keys = np.array(list(self._centroids), int)
        points = np.array(list(self._centroids.values()))
        local = keys-self._index_offset
        hits = self.hits[local[:, 0], local[:, 1], local[:, 2]]
        band = (points[:, 1] > self.base_y+self.obstacle_low) & (points[:, 1] < self.base_y+G.H_CAM)
        return points[(hits >= 2) & band]

    def planning_grid(self, position, half=8., res=.025):
        return SurfaceGrid(self, position, half, res)


class SurfaceGrid:
    """Local planning window; continuous swept checks use the same measured points."""
    def __init__(self, body, position, half, res):
        from scipy.spatial import cKDTree
        self.res = res
        self.n = 2*int(math.ceil(half/res))+1
        self.origin = np.asarray(position, float)-(self.n/2)*res
        self.body_radius = S.AGENT_RADIUS+.01  # measurement/sampling allowance, separate from desired corridor margin
        points = body.obstacle_points()
        keep = (np.abs(points-np.asarray(position)) <= half+1.).all(1)
        self.tree = cKDTree(points[keep]) if keep.any() else None
        xy = self.origin+(np.indices((self.n, self.n)).transpose(1, 2, 0)+.5)*res
        self.clearance = self.clearance_at(xy)
        ij = np.floor((xy-body.origin)/L.CELL).astype(int)
        inside = (ij >= 0).all(-1) & (ij < body.n).all(-1)
        safe = np.clip(ij, 0, body.n-1)
        unknown = body.grids()[2]
        self.unknown = ~inside | unknown[safe[..., 0], safe[..., 1]]

    def cell(self, point):
        return tuple(np.clip(np.floor((np.asarray(point)-self.origin)/self.res).astype(int), 0, self.n-1))

    def center(self, cell):
        return self.origin+(np.asarray(cell)+.5)*self.res

    def clearance_at(self, points):
        p = np.asarray(points, float)
        ij = np.floor((p-self.origin)/self.res).astype(int)
        inside = (ij >= 0).all(-1) & (ij < self.n).all(-1)
        distance = self.tree.query(p.reshape(-1, 2))[0].reshape(p.shape[:-1]) if self.tree is not None else np.full(p.shape[:-1], 99.)
        return np.where(inside, distance-self.body_radius, -np.inf)
