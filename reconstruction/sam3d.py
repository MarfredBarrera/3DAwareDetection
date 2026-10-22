"""Reconstruct a foreground-masked object with SAM 3D Objects and render novel views."""
from pathlib import Path

import numpy as np
from PIL import Image

from tools.cameras import SAM3D_FOV, sam3d_gravity


def _synchronize_cuda():
    """Make elapsed wall-clock measurements include queued CUDA work."""
    import torch

    torch.cuda.synchronize()


class SAM3D:
    """Load SAM 3D Objects once; reconstruct masked objects and render candidate views."""

    def __init__(self, checkpoint_dir="/data/checkpoints/sam3d/hf"):
        import torch

        config = Path(checkpoint_dir) / "pipeline.yaml"
        if not config.is_file():
            raise FileNotFoundError(
                f"Missing SAM 3D checkpoint configuration: {config}. Run scripts/download_models.sh."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; run scripts/check_dependencies.sh.")
        from omegaconf import OmegaConf
        from inference import Inference, make_scene, ready_gaussian_for_video_rendering

        config_data = OmegaConf.load(config)
        config_data.workspace_dir = str(config.parent)
        # Fast-SAM3D ships these configs with its code, not the official weights.
        fast_configs = Path("/opt/src/fast-sam3d/checkpoints/hf")
        config_data.ss_generator_config_path = str(fast_configs / "ss_generator_faster.yaml")
        config_data.slat_generator_config_path = str(fast_configs / "slat_generator_faster.yaml")
        config_data.depth_model.model.pretrained_model_name_or_path = str(
            config.parent.parent / "moge-vitl/model.pt")
        from types import SimpleNamespace
        parameters = SimpleNamespace(
            ss_faster_stride=3, ss_warmup=2, ss_order=1, ss_momentum_beta=0.5,
            slat_thresh=0.5, slat_warmup=2, slat_token_ratio=0.15,
            mesh_spectral_threshold_low=0.5, mesh_spectral_threshold_high=0.7,
            enable_ss_faster=True, enable_slat_token=True, enable_mesh_aggregation=True,
            enable_acceleration=True, enable_taylor=False, enable_easy=False,
        )
        self._inference = Inference(config_data, compile=False, args=parameters)
        self._inference.get_params(parameters)
        self._make_scene = make_scene
        self._prepare_gaussian = ready_gaussian_for_video_rendering
        self.last_timing = self.last_ensemble = None

    def _load_input(self, rgb, mask):
        """RGB image and boolean mask; also sets FastSAM3D's mask frequency ratio."""
        with Image.open(rgb) as source:
            image = source.convert("RGB")
        mask_path = Path(mask)
        with Image.open(mask_path) as source:
            mask = np.asarray(source.convert("L")) > 0
        if mask.shape != (image.height, image.width) or not mask.any():
            raise ValueError("mask must match the image and contain foreground pixels")
        from fft.fft2d import calculate_hfer_robust
        self._inference.get_hfer(calculate_hfer_robust(str(mask_path)))
        return image, mask

    def render_views(self, rgb, mask, output_dir, azimuths, elevations, *, input_elevation,
                     distance=2.0, resolution=320, seed=42, ensemble=1, max_uncertainty=None,
                     save_mesh=True):
        """Reconstruct one masked object and render it at every (azimuth, elevation).

        Angles are relative to the input camera ((180, 0) re-renders the input
        view; see ``tools.cameras``). ``input_elevation`` is the input camera's
        elevation in degrees (positive = looking down); candidate cameras use
        gravity as up so renders are level. ``save_mesh`` writes ``mesh.npz``
        for ambiguity scoring. Returns the candidates as written by
        ``render_view_grid``.

        ``ensemble > 1`` reruns stage 1 from ``ensemble - 1`` more seeds, reusing
        the recorded image conditioning, and writes ``stage1.npz``;
        ``last_ensemble["disagreement"]`` is the uncertainty in [0, 1]. When it
        exceeds ``max_uncertainty`` nothing is rendered and ``[]`` is returned.
        ``last_timing`` holds per-step wall-clock seconds.
        """
        from contextlib import nullcontext
        from time import perf_counter

        from reconstruction import stage1_uncertainty as stage1

        image, mask = self._load_input(rgb, mask)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pipeline = self._inference._pipeline
        recorder = stage1.Recorder(pipeline.models["ss_generator"])
        # CUDA work is asynchronous; synchronize so durations cover the model only.
        _synchronize_cuda()
        started = perf_counter()
        with recorder.attached() if ensemble > 1 else nullcontext():
            output = self._inference(np.asarray(image), mask, seed=seed, pointmap=None)
        scene = self._make_scene(output)
        gaussian = self._prepare_gaussian(scene)
        _synchronize_cuda()
        timing = {"model_seconds": perf_counter() - started}

        started = perf_counter()
        self.last_ensemble = None
        if ensemble > 1:
            if recorder.call is None:
                raise RuntimeError("stage 1 did not call the generator")
            self.last_ensemble = stage1.ensemble(recorder, pipeline.models["ss_decoder"],
                                                 pipeline.shape_model_dtype, seed, ensemble)
            stage1.save_ensemble(output_dir / "stage1.npz", self.last_ensemble, ensemble)
        timing["ensemble_seconds"] = perf_counter() - started
        self.last_timing = timing
        uncertainty = (self.last_ensemble or {}).get("disagreement")
        if max_uncertainty is not None and (uncertainty is None or uncertainty > max_uncertainty):
            return []

        started = perf_counter()
        if save_mesh:
            if "mesh" not in output:
                raise RuntimeError("SAM3D returned no mesh for viewpoint scoring")
            _save_render_mesh(output, scene, output_dir)
        candidates = render_view_grid(gaussian, render_view, output_dir / "views", azimuths, elevations,
                                      resolution=resolution, distance=distance,
                                      up=sam3d_gravity(input_elevation))
        timing["render_seconds"] = perf_counter() - started
        return candidates


def _save_render_mesh(output, scene_gaussian, output_dir):
    """Export FastSAM3D's decoded mesh in the normalized Gaussian render frame."""
    from sam3d_objects.utils.visualization import SceneVisualizer

    mesh = output["mesh"][0]
    vertices = mesh.vertices
    positions = scene_gaussian.get_xyz
    active = (scene_gaussian.get_opacity > 0.9).squeeze()
    active_positions = positions[active]
    scale = (active_positions.max(dim=0).values - active_positions.min(dim=0).values).max()
    center = (active_positions.max(dim=0).values + active_positions.min(dim=0).values) / 2
    world = SceneVisualizer.object_pointcloud(
        points_local=vertices.unsqueeze(0), quat_l2c=output["rotation"],
        trans_l2c=output["translation"], scale_l2c=output["scale"],
    ).points_list()[0]
    normalized = ((world - center) / scale).detach().cpu().numpy().astype(np.float32)
    faces = mesh.faces.detach().cpu().numpy().astype(np.int32)
    if not np.isfinite(normalized).all() or len(faces) == 0:
        raise RuntimeError("FastSAM3D returned an invalid mesh")
    np.savez_compressed(output_dir / "mesh.npz", vertices=normalized, faces=faces)


def render_view(gaussian, resolution, bg_color, r, fov, pitch_deg, yaw_start_deg, up):
    """One frame of ``notebook/inference.render_video``, with a choosable camera up.

    That renderer always uses the frame's +Y (the input camera's up) as up; ``up``
    replaces it, normally ``tools.cameras.sam3d_gravity``. Cameras whose view
    direction is parallel to ``up`` fall back to +Y.
    """
    import math

    import torch
    import utils3d
    from sam3d_objects.model.backbone.tdfy_dit.utils import render_utils

    yaw, pitch = math.radians(yaw_start_deg), math.radians(pitch_deg)
    origin = r * np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch),
                           math.cos(yaw) * math.cos(pitch)])
    up = np.asarray(up, dtype=np.float64)
    if np.linalg.norm(np.cross(origin / r, up)) < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
    extrinsics = utils3d.torch.extrinsics_look_at(
        torch.tensor(origin, dtype=torch.float32).cuda(), torch.zeros(3).cuda(),
        torch.tensor(up, dtype=torch.float32).cuda())
    fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
    intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
    return render_utils.render_frames(
        gaussian, [extrinsics], [intrinsics],
        {"resolution": resolution, "bg_color": bg_color, "backend": "gsplat"}, verbose=False)


def render_view_grid(gaussian, render_view, output_dir, azimuths, elevations, resolution=512,
                     distance=1.0, *, up):
    """Render one image at each local azimuth/elevation pair, elevation-major.

    The camera orbits the normalized reconstruction (largest extent 1) at ``distance``
    with a 60-degree FOV; below about 1.6 the object is clipped by the frame.
    ``up`` is the camera up vector in that frame: ``tools.cameras.sam3d_gravity``
    of the input elevation keeps candidates level.
    """
    if resolution < 64:
        raise ValueError("resolution must be at least 64")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for elevation in elevations:
        for azimuth in azimuths:
            def render_at(background):
                result = render_view(
                    gaussian, resolution=resolution, bg_color=background, r=distance,
                    fov=SAM3D_FOV, pitch_deg=float(elevation), yaw_start_deg=float(azimuth), up=up,
                )["color"]
                frame = result.detach().float().cpu().numpy() if hasattr(result, "detach") else np.asarray(result)
                if frame.shape != (1, resolution, resolution, 3):
                    raise RuntimeError(f"Unexpected SAM 3D render shape: {frame.shape}")
                frame = frame[0]
                if frame.dtype != np.uint8:
                    frame = np.clip(frame * (255 if frame.max() <= 1 else 1), 0, 255).astype(np.uint8)
                return frame

            white = render_at((1.0, 1.0, 1.0))
            black = render_at((0.0, 0.0, 0.0))
            # The renderer exposes RGB but no alpha. White minus black gives
            # (1 - alpha) * 255 for the same view, independent of object color.
            alpha = 255 - np.mean(white.astype(np.int16) - black.astype(np.int16), axis=2)
            name = f"azimuth_{azimuth:g}_elevation_{elevation:g}.png"
            mask_name = f"azimuth_{azimuth:g}_elevation_{elevation:g}.mask.png"
            Image.fromarray(white).save(output_dir / name)
            Image.fromarray(np.where(alpha >= 128, 255, 0).astype(np.uint8)).save(output_dir / mask_name)
            candidates.append({"azimuth": azimuth, "elevation": elevation,
                               "image": name, "mask": mask_name})
    return candidates
