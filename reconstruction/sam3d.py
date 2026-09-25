"""Reconstruct a foreground-masked object with SAM 3D Objects and render novel views."""
from pathlib import Path

import numpy as np
from PIL import Image
from .visualization import render_turntable, render_view_grid


def _synchronize_cuda():
    """Make elapsed wall-clock measurements include queued CUDA work."""
    import torch

    torch.cuda.synchronize()


class SAM3D:
    """Load SAM 3D Objects once and reconstruct a masked RGB object into novel views."""

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
        from inference import Inference, make_scene, ready_gaussian_for_video_rendering, render_video

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
        self._render_video = render_video
        self.last_timing = None

    def render_views(self, rgb, mask, output_dir, views=8, resolution=512, seed=42, pointmap=None,
                     view_grid=None):
        """Write a PLY and either turntable frames or an azimuth/elevation view grid.

        ``last_timing["model_seconds"]`` covers only SAM3D inference and
        Gaussian-scene preparation. It excludes input loading, PLY output,
        view rendering, PNG writes, and GIF encoding.
        """
        from time import perf_counter

        with Image.open(rgb) as source:
            image = source.convert("RGB")
        mask_path = Path(mask)
        with Image.open(mask_path) as source:
            mask = np.asarray(source.convert("L")) > 0
        if mask.shape != (image.height, image.width) or not mask.any():
            raise ValueError("mask must match the image and contain foreground pixels")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        from fft.fft2d import calculate_hfer_robust
        self._inference.get_hfer(calculate_hfer_robust(str(mask_path)))

        if pointmap is not None:
            pointmap = _pointmap(pointmap, image.size)
            import torch
            pointmap = torch.from_numpy(pointmap)

        # CUDA work is asynchronous. Synchronizing on either side gives an
        # accurate wall-clock duration for the model portion only.
        _synchronize_cuda()
        model_started = perf_counter()
        output = self._inference(np.asarray(image), mask, seed=seed, pointmap=pointmap)
        scene = self._make_scene(output)
        gaussian = self._prepare_gaussian(scene)
        _synchronize_cuda()
        model_seconds = perf_counter() - model_started

        ply_started = perf_counter()
        gaussian.save_ply(str(output_dir / "object.ply"))
        if view_grid is not None and "mesh" in output:
            _save_render_mesh(output, scene, output_dir)
        ply_write_seconds = perf_counter() - ply_started

        render_started = perf_counter()
        frames = (render_view_grid(gaussian, self._render_video, output_dir / "views",
                                   *view_grid, resolution=resolution)
                  if view_grid is not None else
                  render_turntable(gaussian, self._render_video, output_dir, views, resolution))
        self.last_timing = {
            "model_seconds": model_seconds,
            "ply_write_seconds": ply_write_seconds,
            ("view_grid_seconds" if view_grid is not None else "turntable_seconds"):
                perf_counter() - render_started,
        }
        return frames


def _pointmap(pointmap, image_size):
    if isinstance(pointmap, (str, Path)):
        pointmap = np.load(pointmap)
    pointmap = np.asarray(pointmap, dtype=np.float32)
    width, height = image_size
    if pointmap.shape != (height, width, 3):
        raise ValueError("pointmap must have shape HxWx3 and align with the RGB image")
    if not np.isfinite(pointmap).all():
        raise ValueError("pointmap must contain only finite coordinates")
    return pointmap


def _save_render_mesh(output, scene_gaussian, output_dir):
    """Export FastSAM3D's decoded mesh in the normalized Gaussian render frame."""
    import trimesh
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
    trimesh.Trimesh(vertices=normalized, faces=faces, process=False).export(output_dir / "mesh.ply")
