"""Run-time geometry for the planners. Inputs are what any ObjectNav agent has: the fused object cloud (world points
under the object mask), depth frames (a local 2.5D height map) and the agent pose. World frame: Habitat (y up); 2D
quantities use (x, z).

    ObjectModel(cloud, floor_y)   z_lo/z_hi (5/95 % height above the floor), PCA footprint, centre
    HeightMap                     5 cm cells: max height of non-object depth points, the same below BODY_TOP, seen flag
    d_fit(...)                    the fit distance (pitch 0, margin 3 deg, kappa of the height may be cut)
"""
import math

import numpy as np

from nav.approach import scene as S

H_CAM = S.CAMERA_HEIGHT
BETA_V = math.atan((S.HEIGHT / 2) / S.FOCAL)          # 31.7 deg
BETA_H = math.radians(S.HFOV / 2)                     # 39.5 deg
MARGIN = math.radians(3.0)
CELL = 0.05
DEPTH_MAX = 4.99                                       # ApexNav's depth range (map_ros depth_filter_maxdist)
LATTICE = math.radians(S.TURN_DEG)
BODY_TOP = 1.6                                         # m above the floor: higher structure does not block the base


def forward(yaw):
    """Horizontal (x, z) unit vector the camera looks along (yaw 0 = -z, scene.facing convention)."""
    return np.array([-math.sin(yaw), -math.cos(yaw)])


def bearing_yaw(position_xz, target_xz):
    d = np.asarray(target_xz, float) - np.asarray(position_xz, float)
    return math.atan2(-d[0], -d[1])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class ObjectModel:
    def __init__(self, cloud, floor_y):
        xz = cloud[:, [0, 2]]
        h = cloud[:, 1] - floor_y
        self.z_lo, self.z_hi = (float(np.percentile(h, 5)), float(np.percentile(h, 95))) if len(h) else (0.0, 1.0)
        self.xz = xz
        mean = xz.mean(0)
        cov = np.cov((xz - mean).T) if len(xz) > 2 else np.eye(2)
        w, v = np.linalg.eigh(cov)
        self.axes = v[:, ::-1].T                      # long axis first
        local = (xz - mean) @ self.axes.T
        lo, hi = local.min(0), local.max(0)
        self.extent = hi - lo                         # (L, S)
        self.center = mean + ((lo + hi) / 2) @ self.axes
        self.bbox = np.r_[xz.min(0), xz.max(0)]

    def robust_extent(self, q=(2.0, 98.0)):
        """Footprint (L, S) along the PCA axes between the ``q`` percentiles (2-98 %): ``extent`` without the
        stray points of a merge (size_opinion's width W). (0, 0) for fewer than 3 points."""
        if len(self.xz) < 3:
            return np.zeros(2)
        local = (self.xz - self.xz.mean(0)) @ self.axes.T
        lo, hi = np.percentile(local, q, axis=0)
        return hi - lo

    @property
    def height(self):
        return max(self.z_hi - self.z_lo, 0.05)

    def half_extent(self, direction):
        """Half width of the footprint along a horizontal unit ``direction``."""
        proj = (self.xz - self.center) @ direction
        return float(max(proj.max(), -proj.min())) if len(proj) else 0.0


def d_fit(model, yaw, kappa=0.25, top_clipped=False, side_clipped=False):
    """Distance from the object's near face at which, looking along ``yaw`` at pitch 0, the top fits,
    at most ``kappa`` of the height is cut at the bottom, and the width fits (5 deg margin)."""
    z_hi = model.z_hi
    if top_clipped:  # the top ran off a judged frame: z_hi is only a lower bound
        z_hi = z_hi + 0.2
    t = math.tan(BETA_V - MARGIN)
    need = [0.3]
    if z_hi > H_CAM:
        need.append((z_hi - H_CAM) / t)
    bottom = model.z_lo + kappa * model.height
    if bottom < H_CAM:
        need.append((H_CAM - bottom) / t)
    side = np.array([math.cos(yaw), -math.sin(yaw)])  # perpendicular to forward(yaw)
    half = model.half_extent(side) + (0.3 if side_clipped else 0.0)  # a side ran off a judged frame: W is a lower bound
    need.append(half / math.tan(BETA_H - math.radians(5)))
    return max(need)


class HeightMap:
    """Local 2.5D map around the object: per 5 cm cell the max height (above the floor) of non-object
    depth points (``top``, for occlusion), the max height of those below BODY_TOP (``low``, for collision: a wall
    or a table seen up to the ceiling has ``top`` above BODY_TOP but ``low`` in the base's height band), and
    whether the cell was seen (a hit or crossed by a depth ray)."""

    def __init__(self, center_xz, floor_y, half=7.0, body_top=BODY_TOP):
        self.origin = np.asarray(center_xz, float) - half
        self.n = int(2 * half / CELL)
        self.top = np.full((self.n, self.n), -np.inf, np.float32)
        self.low = np.full((self.n, self.n), -np.inf, np.float32)
        self.seen = np.zeros((self.n, self.n), bool)
        self.floor_y = floor_y
        self.body_top = float(body_top)
        
    def cells(self, xz):
        ij = np.floor((np.asarray(xz) - self.origin) / CELL).astype(int)
        ok = (ij >= 0).all(-1) & (ij < self.n).all(-1)
        return ij, ok

    def update(self, depth, mask, cam_pos, cam_rot, stride=6):
        valid = (depth > 0.05) & (depth < DEPTH_MAX) & ~mask
        pts = S.backproject(depth, valid, cam_pos, cam_rot, stride=stride)
        if not len(pts):
            return
        ij, ok = self.cells(pts[:, [0, 2]])
        ij, h = ij[ok], (pts[ok, 1] - self.floor_y).astype(np.float32)
        np.maximum.at(self.top, (ij[:, 0], ij[:, 1]), h)
        below = h < self.body_top
        np.maximum.at(self.low, (ij[below, 0], ij[below, 1]), h[below])
        self.seen[ij[:, 0], ij[:, 1]] = True
        # free space: cells crossed by the rays to every 4th hit
        cam = cam_pos[[0, 2]]
        far = pts[ok][::4][:, [0, 2]]
        if not len(far):
            return
        steps = np.arange(0.0, 1.0, CELL / max(np.linalg.norm(far - cam, axis=1).max(), CELL))
        ray = cam[None, None] + steps[None, :, None] * (far - cam)[:, None]
        rij, rok = self.cells(ray.reshape(-1, 2))
        self.seen[rij[rok, 0], rij[rok, 1]] = True
