"""Episode-long RGB-D collision map for the medial last mile (split navigator).

This map is never exported to or populated from the host planner's occupancy grid. Target
pixels remain physical obstacles. Growing the extent preserves observation history.
"""
import numpy as np

from nav.approach import geom as G, scene as S
from nav.lastmile.body_map import BodyMap


class EpisodeCollisionMap:
    def __init__(self, sim):
        p, _, _ = S.agent_pose(sim)
        self.sim = sim
        # Match the evaluated agent's actual 0.88 m body. The view planner's
        # historical 1.6 m band also blocks table tops above this robot's head.
        self.hmap = G.HeightMap(p[[0, 2]], float(p[1]), half=10., body_top=G.H_CAM)
        # Habitat's evaluated embodiment can step up 0.20 m. Surface returns below that height are traversable
        # terrain, not walls.
        self.lmap = BodyMap(self.hmap, float(p[1]), obstacle_low=.20)
        self.last_pos = self.last_yaw = None

    def _grow(self, xz):
        hm, lm = self.hmap, self.lmap
        ij = (np.asarray(xz)-hm.origin)/G.CELL
        if (ij >= 80).all() and (ij < hm.n-80).all():
            return
        # Equal padding on all sides retains square-grid compatibility.
        pad = max(100, int(np.ceil(max(80-ij.min(), ij.max()-hm.n+80)/100))*100)
        hm.top = np.pad(hm.top, pad, constant_values=-np.inf)
        hm.low = np.pad(hm.low, pad, constant_values=-np.inf)
        hm.seen = np.pad(hm.seen, pad, constant_values=False)
        hm.origin = hm.origin-pad*G.CELL
        hm.n = hm.top.shape[0]
        lp = pad//lm.k
        lm.pad_xy(lp)
        lm.bumps = np.pad(lm.bumps, lp, constant_values=False)
        lm.object = np.pad(lm.object, lp, constant_values=False)
        lm.n, lm.origin = hm.n//lm.k, hm.origin

    def update(self, depth, moved_forward, position=None, yaw=None):
        if position is None:
            position, yaw, _ = S.agent_pose(self.sim)
        position = np.asarray(position, float)
        self._grow(position[[0, 2]])
        if moved_forward and self.last_pos is not None:
            self.lmap.forward_feedback(self.last_pos, self.last_yaw, position)
        d = np.asarray(depth, np.float32)
        cam = S.camera(self.sim)
        self.lmap.update(d, *cam, float(position[1]))
        # The native sensor clips at 5 m. Clipped returns are not surfaces.
        d = np.where(np.isfinite(d) & (d < 4.99), d, 0.)
        self.hmap.update(d, np.zeros(d.shape, bool), *cam)
        self.last_pos, self.last_yaw = position.copy(), float(yaw)
