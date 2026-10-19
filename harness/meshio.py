"""GLB -> PyTorch3D, with the canonical transform applied.

PyTorch3D cannot read GLB, so trimesh loads and flattens the scene graph and we
hand over raw tensors. Textures bake to vertex colours by default: it is the
only path that survives thousands of heterogeneous multi-material GLBs. That
costs high-frequency detail, which matters for open-vocabulary detectors, so
all textures are sampled into per-vertex RGB.
"""
from __future__ import annotations

import gzip
import json
import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import trimesh

if TYPE_CHECKING:
    import torch

DEFAULT_ALBEDO = (0.70, 0.70, 0.72)


@dataclass
class CanonicalMesh:
    uid: str
    category: str
    verts: torch.Tensor          # (V, 3), canonical frame
    faces: torch.Tensor          # (F, 3)
    vert_colors: torch.Tensor    # (V, 3) in [0, 1]


def load_mesh(path: str, max_bytes: int | None = None, with_colors: bool = False):
    """Concatenate a GLB scene into one mesh in world coordinates."""
    if not os.path.exists(path) or (max_bytes is not None and os.path.getsize(path) > max_bytes):
        return None
    try:
        # GLBs can contain line primitives, loaded as Path3D. They do not
        # render as surfaces and have no visual to bake into vertex colors.
        parts = [part for part in trimesh.load(path, force="scene").dump()
                 if isinstance(part, trimesh.Trimesh) and len(part.faces)]
        if not parts:
            return None
        colors = ([ObjaverseStore._vertex_colors(part, len(part.vertices)) for part in parts]
                  if with_colors else None)
        g = trimesh.util.concatenate(parts)
        if colors is not None:
            g.visual = trimesh.visual.ColorVisuals(mesh=g, vertex_colors=np.concatenate(colors))
    except Exception as error:
        warnings.warn(f"Skipping {path}: {error}")
        return None
    if not hasattr(g, "vertices") or len(g.vertices) < 4 or len(getattr(g, "faces", [])) < 1:
        return None
    return g


class ObjaverseStore:
    """Resolves uids to canonicalised meshes."""

    def __init__(self, root: str = "/ICONAS/users/marfred/objaverse", canonical: str | None = None):
        self.root = root
        self.ver = os.path.join(root, "hf-objaverse-v1")
        with gzip.open(os.path.join(self.ver, "object-paths.json.gz"), "rb") as f:
            self.object_paths = json.load(f)
        canon_path = canonical or os.path.join(root, "canonical.json")
        with open(canon_path) as stream:
            payload = json.load(stream)
        self.objects = payload["objects"]
        self.convention = payload["convention"]

    def uids(self, categories=None, max_faces=None, require_watertight=False,
             min_azimuth_confidence=0.0, exclude_up_suspect=False):
        out = []
        for uid, r in self.objects.items():
            if categories and r["category"] not in categories:
                continue
            if max_faces and r["face_count"] > max_faces:
                continue
            if require_watertight and not r["is_watertight"]:
                continue
            if r["azimuth_confidence"] < min_azimuth_confidence:
                continue
            if exclude_up_suspect and r["up_axis_suspect"]:
                continue
            out.append(uid)
        return sorted(out)

    def load(self, uid: str, device="cuda") -> CanonicalMesh | None:
        import torch

        rec = self.objects.get(uid)
        rel = self.object_paths.get(uid)
        if rec is None or rel is None:
            return None
        path = os.path.join(self.ver, rel)
        g = load_mesh(path, with_colors=True)
        if g is None:
            return None

        v = np.asarray(g.vertices, dtype=np.float32)
        f = np.asarray(g.faces, dtype=np.int64)
        R = np.asarray(rec["rotation"], dtype=np.float32)
        v = (v @ R.T) * np.float32(rec["scale"]) + np.asarray(rec["translation"], dtype=np.float32)

        colors = self._vertex_colors(g, len(v))
        return CanonicalMesh(
            uid=uid,
            category=rec["category"],
            verts=torch.from_numpy(v).to(device),
            faces=torch.from_numpy(f).to(device),
            vert_colors=torch.from_numpy(colors).to(device),
        )

    @staticmethod
    def _vertex_colors(g, n: int) -> np.ndarray:
        """Bake whatever appearance the GLB carries down to per-vertex RGB."""
        visual = g.visual
        try:
            vis = visual
            # ColorVisuals holds glTF COLOR_0. Converting it can replace those
            # values with a material's single diffuse color.
            if not isinstance(vis, trimesh.visual.ColorVisuals) and hasattr(vis, "to_color"):
                vis = vis.to_color()
            vc = getattr(vis, "vertex_colors", None)
            if vc is not None:
                c = np.asarray(vc, dtype=np.float32)
                if c.ndim == 1 and c.size in (3, 4):
                    c = np.broadcast_to(c, (n, c.size))
                if len(c) != n:
                    raise ValueError("vertex color count does not match vertices")
                c = c[:, :3] / (255.0 if c.max() > 1 else 1.0)
                if np.isfinite(c).all():
                    return np.ascontiguousarray(c)
        except Exception:
            pass
        material = getattr(visual, "material", None)
        factor = getattr(material, "baseColorFactor", None)
        if factor is None:
            factor = getattr(material, "diffuse", None)
        if factor is not None:
            color = np.asarray(factor, dtype=np.float32).reshape(-1)
            if color.size >= 3 and np.isfinite(color[:3]).all():
                color = np.clip(color[:3] / (255.0 if color.max() > 1 else 1.0), 0, 1)
                return np.tile(color, (n, 1))
        return np.tile(np.asarray(DEFAULT_ALBEDO, dtype=np.float32), (n, 1))
