"""Stage 2: viewpoint x occlusion rendering with pixel-exact ground truth.

The amodal mask comes from rasterising the target alone; the modal mask comes
from rasterising target and occluders *together* and attributing pixels by face
ownership. So occlusion_ratio = 1 - |modal|/|amodal| is measured, never
estimated. Occluders rest on the ground plane by default, since a floating
occluder is not a situation a detector will meet in a real scene.
"""
from __future__ import annotations

import numpy as np
import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras, MeshRasterizer, PointLights,
    RasterizationSettings, SoftPhongShader, TexturesVertex, look_at_view_transform,
)
from pytorch3d.structures import Meshes, join_meshes_as_scene

from tools.cameras import HARNESS_LOOK_AT

WORLD_UP = torch.tensor([0.0, 1.0, 0.0])


def _mesh(cm, device, offset_xz=(0.0, 0.0), scale=1.0) -> Meshes:
    offset = cm.verts.new_tensor([offset_xz[0], 0.0, offset_xz[1]])
    return Meshes(
        verts=[cm.verts * scale + offset],
        faces=[cm.faces],
        textures=TexturesVertex(verts_features=[cm.vert_colors]),
    ).to(device)


def _bbox(mask: torch.Tensor):
    idx = mask.nonzero()
    if idx.numel() == 0:
        return []
    y0, x0 = idx.min(0).values.tolist()
    y1, x1 = idx.max(0).values.tolist()
    return [int(x0), int(y0), int(x1) + 1, int(y1) + 1]


class Harness:
    def __init__(self, image_size=512, fov=50.0, device="cuda", faces_per_pixel=1, bin_size=0):
        # bin_size=0 forces naive rasterisation. Binned mode overflows on dense
        # meshes and reports "output may be incomplete" -- incomplete masks would
        # make occlusion_ratio quietly wrong, which defeats the point.
        self.device = torch.device(device)
        self.image_size = image_size
        self.fov = fov
        self.raster = RasterizationSettings(
            image_size=image_size, blur_radius=0.0,
            faces_per_pixel=faces_per_pixel, bin_size=bin_size,
        )
        self.lights = PointLights(device=self.device, location=[[2.0, 3.0, 2.0]])

    @property
    def intrinsics(self):
        focal = self.image_size / (2 * np.tan(np.deg2rad(self.fov) / 2))
        center = (self.image_size - 1) / 2
        return [float(focal), float(focal), center, center]

    def camera(self, dist, elev, azim):
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, up=((0, 1, 0),),
                                      at=(self.LOOK_AT,))
        return FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=self.fov)

    def _rasterize(self, mesh, cameras):
        return MeshRasterizer(cameras=cameras, raster_settings=self.raster)(mesh)

    def lateral_dir(self, cameras):
        """Unit world-space vector perpendicular to the view ray, in the XZ plane."""
        C = cameras.get_camera_center()[0]
        look = -C
        right = torch.cross(look, WORLD_UP.to(C.device), dim=0)
        right[1] = 0.0
        n = right.norm()
        if n < 1e-6:
            right = torch.tensor([1.0, 0.0, 0.0], device=C.device)
            n = 1.0
        return (right / n)[[0, 2]]

    LOOK_AT = HARNESS_LOOK_AT

    def occluder_placement(self, cameras, occluder, offset, along=0.5,
                           height_factor=1.6, scale_mult=1.0):
        """Place and size an occluder so it actually crosses the sight line.

        A ground-resting occluder near the camera sits *below* the ray from the
        camera to the target and occludes nothing, so its scale is derived from
        the sight-line height at its own standoff. Returns (xz position, scale).
        """
        C = cameras.get_camera_center()[0]
        A = torch.tensor(self.LOOK_AT, device=C.device, dtype=C.dtype)
        P = C + along * (A - C)                   # along=0 at camera, 1 at target
        occ_h = float(occluder.verts[:, 1].max())  # canonical frame rests base at y=0
        scale = (float(P[1]) * height_factor) / max(occ_h, 1e-6) * scale_mult
        pos = (P[[0, 2]] + self.lateral_dir(cameras) * offset).tolist()
        return pos, scale

    def occluder_span(self, occluder, scale):
        """Lateral sweep range: wide enough to clear the target on either side."""
        w = float(occluder.verts[:, 0].max() - occluder.verts[:, 0].min()) * scale
        return 0.5 * w + 0.9

    @torch.inference_mode()
    def render(self, target, occluders, dist, elev, azim, offset=0.0,
               occluder_scale=1.0, along=0.5, want_rgb=True, want_depth=False):
        cams = self.camera(dist, elev, azim)
        tgt = _mesh(target, self.device)
        n_tgt_faces = int(tgt.faces_packed().shape[0])

        amodal_frag = self._rasterize(tgt, cams)
        amodal = (amodal_frag.pix_to_face[0, ..., 0] >= 0)

        if occluders:
            placed = []
            for o in occluders:
                pos, sc = self.occluder_placement(cams, o, offset, along,
                                                  scale_mult=occluder_scale)
                placed.append(_mesh(o, self.device, pos, sc))
            scene = join_meshes_as_scene([tgt] + placed)
            frag = self._rasterize(scene, cams)
            front = frag.pix_to_face[0, ..., 0]
            modal = (front >= 0) & (front < n_tgt_faces)   # target owns the nearest face
        else:
            scene, frag, modal = tgt, amodal_frag, amodal

        a = int(amodal.sum())
        m = int(modal.sum())
        ratio = float(1.0 - m / a) if a > 0 else float("nan")

        border = bool(amodal[0, :].any() or amodal[-1, :].any()
                      or amodal[:, 0].any() or amodal[:, -1].any())
        gt = dict(
            uid=target.uid, category=target.category,
            azimuth=float(azim), elevation=float(elev), distance=float(dist),
            fov=self.fov, image_size=self.image_size,
            intrinsics=self.intrinsics, depth_units="canonical",
            amodal_area=a, modal_area=m, occlusion_ratio=ratio,
            amodal_bbox=_bbox(amodal), modal_bbox=_bbox(modal),
            occluder_uids=[o.uid for o in occluders],
            occluder_categories=[o.category for o in occluders],
            occluder_offset=float(offset),
            R=cams.R[0].tolist(), T=cams.T[0].tolist(), truncated=border,
        )
        out = dict(gt=gt, amodal=amodal, modal=modal)
        if want_rgb:
            shader = SoftPhongShader(device=self.device, cameras=cams, lights=self.lights)
            out["rgb"] = shader(frag, scene)[0, ..., :3].clamp(0, 1)
        if want_depth:
            z = frag.zbuf[0, ..., 0].clone()
            z[z < 0] = float("nan")
            out["depth"] = z
        return out

    def _ratio(self, target, occluders, dist, elev, azim, off, occluder_scale, along):
        r = self.render(target, occluders, dist, elev, azim, offset=float(off),
                        occluder_scale=occluder_scale, along=along,
                        want_rgb=False, want_depth=False)
        return r["gt"]["occlusion_ratio"]

    def sweep_occlusion(self, target, occluders, dist, elev, azim, targets,
                        occluder_scale=1.0, along=0.5, steps=13, span=None,
                        tol=0.03, refine=8):
        """Hit each requested occlusion ratio by bracketing it, then bisecting.

        Coverage varies continuously as the occluder slides sideways, but the
        transition can be far sharper than any practical grid, so a coarse sweep
        only locates a bracket; bisection lands the ratio. Levels that stay
        unreachable are reported rather than silently approximated.
        """
        if not targets or not occluders:
            return {}, []

        def ratio(o):
            return self._ratio(target, occluders, dist, elev, azim, o,
                               occluder_scale, along)

        if span is None:
            cams = self.camera(dist, elev, azim)
            _, sc = self.occluder_placement(cams, occluders[0], 0.0, along,
                                            scale_mult=occluder_scale)
            span = self.occluder_span(occluders[0], sc)
        curve = [(float(o), ratio(o)) for o in np.linspace(-span, span, steps)]
        picks = {}
        for want in targets:
            best = None
            for (o1, r1), (o2, r2) in zip(curve, curve[1:]):
                if not (np.isfinite(r1) and np.isfinite(r2)):
                    continue
                if (r1 - want) * (r2 - want) > 0:
                    continue                      # no sign change: not bracketed
                lo, hi, r_lo = o1, o2, r1
                cand = (abs(r1 - want), o1, r1) if abs(r1 - want) < abs(r2 - want) \
                    else (abs(r2 - want), o2, r2)
                for _ in range(refine):
                    mid = 0.5 * (lo + hi)
                    r_mid = ratio(mid)
                    if not np.isfinite(r_mid):
                        break
                    if abs(r_mid - want) < cand[0]:
                        cand = (abs(r_mid - want), mid, r_mid)
                    if cand[0] <= tol:
                        break
                    if (r_lo - want) * (r_mid - want) <= 0:
                        hi = mid
                    else:
                        lo, r_lo = mid, r_mid
                if best is None or cand[0] < best[0]:
                    best = cand
            if best is None:                       # never bracketed: take the closest
                fin = [(abs(r - want), o, r) for o, r in curve if np.isfinite(r)]
                best = min(fin) if fin else None
            if best is not None and best[0] <= tol:
                picks[want] = dict(offset=best[1], achieved=best[2])
        return picks, curve
