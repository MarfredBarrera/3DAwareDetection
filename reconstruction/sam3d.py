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

    def stage1_uncertainty(self, rgb, mask, seed=42, t_stars=(0.2, 0.5, 0.8), probes=8,
                           samples=8, ensemble=8, guided=False):
        """Tweedie and seed-ensemble uncertainty of stage 1 (see reconstruction.stage1_uncertainty).

        Runs stage 1 only, with ``seed`` (the reconstruction's own sample), then
        ``ensemble - 1`` more seeds. Returns ``{"ensemble": ..., "tweedie": {t: ...}}``
        with occupancy summaries and layout-token variances.
        """
        import torch
        from reconstruction import stage1_uncertainty as stage1

        image, mask = self._load_input(rgb, mask)
        pipeline = self._inference._pipeline
        generator, decoder = pipeline.models["ss_generator"], pipeline.models["ss_decoder"]
        if not stage1.is_data_at_one(generator):
            raise RuntimeError("stage-1 time convention differs from t = 1 at data")
        dtype = pipeline.shape_model_dtype
        recorder = stage1.Recorder(generator)
        with recorder.attached():
            pipeline.run(self._inference.merge_mask_to_rgba(np.asarray(image), mask), None, seed,
                         stage1_only=True, with_mesh_postprocess=False, with_texture_baking=False,
                         with_layout_postprocess=True, use_vertex_color=True,
                         stage1_inference_steps=None, pointmap=None)
            draws = [recorder.resample(seed + 1000 * k, dtype) for k in range(1, ensemble)]
        if recorder.call is None:
            raise RuntimeError("stage 1 did not call the generator")
        reruns = [recorder.output] + draws  # the reconstruction's own sample first
        shapes = torch.cat([draw[stage1.SHAPE] for draw in reruns])
        layout = {key: torch.cat([draw[key] for draw in reruns]).float().var(dim=0)
                  for key in reruns[0] if key != stage1.SHAPE}
        result = {"ensemble": {**stage1.summarize(stage1.occupancy(decoder, shapes, dtype)),
                               "layout_variance": stage1.pose_variance(layout)},
                  "tweedie": {}}
        for t_star in t_stars:
            t, _, variance, occupied = stage1.tweedie(recorder, generator.reverse_fn, decoder,
                                                      t_star, probes, samples, dtype, seed, guided)
            result["tweedie"][t_star] = {
                **stage1.summarize(occupied), "t": t,
                "shape_variance": float(variance[stage1.SHAPE].mean()),
                "layout_variance": stage1.pose_variance(variance)}
        return result

    def render_views(self, rgb, mask, output_dir, views=8, resolution=512, seed=42, pointmap=None,
                     view_grid=None, save_ply=True, view_distance=1.0, also_turntable=False,
                     input_elevation=None, save_gaussians=False):
        """Write either a turntable or a viewpoint grid from one RGB and mask.

        Grid renders include ``mesh.npz`` for ambiguity scoring. The one-frame
        demo also writes a PLY for its interactive viewer. With the input
        camera's ``input_elevation`` (degrees), grid cameras keep gravity up so
        candidates are level; without it they keep the input camera's up.
        ``save_gaussians`` writes ``gaussians.npz`` (see ``save_gaussians``).

        ``last_timing["model_seconds"]`` covers SAM3D inference and Gaussian
        preparation. It excludes input loading, mesh output, and view rendering.
        """
        from time import perf_counter

        image, mask = self._load_input(rgb, mask)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

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

        mesh_started = perf_counter()
        if save_ply:
            gaussian.save_ply(str(output_dir / "object.ply"))
        if save_gaussians:
            save_gaussian_npz(gaussian, output_dir / "gaussians.npz")
        if view_grid is not None:
            if "mesh" not in output:
                raise RuntimeError("SAM3D returned no mesh for viewpoint scoring")
            _save_render_mesh(output, scene, output_dir, save_ply)
        mesh_write_seconds = perf_counter() - mesh_started

        render_started = perf_counter()
        up = None if input_elevation is None else sam3d_gravity(input_elevation)
        frames = (render_view_grid(gaussian, render_view, output_dir / "views",
                                   *view_grid, resolution=resolution, distance=view_distance, up=up)
                  if view_grid is not None else
                  render_turntable(gaussian, self._render_video, output_dir, views, resolution))
        if view_grid is not None and also_turntable:
            render_turntable(gaussian, self._render_video, output_dir, views, resolution)
        self.last_timing = {
            "model_seconds": model_seconds,
            "mesh_write_seconds": mesh_write_seconds,
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


def save_gaussian_npz(gaussian, path):
    """Activated Gaussians in the candidate render frame, as gsplat takes them.

    Arrays: ``means`` (N, 3), ``scales`` (N, 3), ``quats`` (N, 4, wxyz, unit),
    ``opacities`` (N,), and ``sh`` (N, K, 3) spherical-harmonic coefficients.
    """
    def array(tensor):
        return tensor.detach().float().cpu().numpy()

    np.savez_compressed(path, means=array(gaussian.get_xyz), scales=array(gaussian.get_scaling),
                        quats=array(gaussian.get_rotation),
                        opacities=array(gaussian.get_opacity).reshape(-1),
                        sh=array(gaussian.get_features), sh_degree=gaussian.active_sh_degree)


def _save_render_mesh(output, scene_gaussian, output_dir, save_ply=True):
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
    if save_ply:
        trimesh.Trimesh(vertices=normalized, faces=faces, process=False).export(output_dir / "mesh.ply")


def render_turntable(gaussian, render_video, output_dir, views=8, resolution=512,
                     bg_color=(1.0, 1.0, 1.0)):
    """Render orbiting RGB views and write `views/*.png` plus `turntable.gif`.

    ``bg_color`` is the background the splat is composited over, in [0, 1] RGB
    (the renderer's native scale); it defaults to white.
    """
    if views < 2:
        raise ValueError("views must be at least 2")
    if resolution < 64:
        raise ValueError("resolution must be at least 64")
    output_dir = Path(output_dir)
    views_dir = output_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    frames = render_video(
        gaussian,
        num_frames=views,
        resolution=resolution,
        bg_color=bg_color,
        r=1,
        fov=60,
        pitch_deg=15,
        yaw_start_deg=-45,
    )["color"]
    if hasattr(frames, "detach"):
        frames = frames.detach().float().cpu().numpy()
    else:
        frames = np.asarray(frames)
    if frames.shape != (views, resolution, resolution, 3):
        raise RuntimeError(f"Unexpected SAM 3D render shape: {frames.shape}")
    if frames.dtype != np.uint8:
        frames = np.clip(frames * (255 if frames.max() <= 1 else 1), 0, 255).astype(np.uint8)
    for index, frame in enumerate(frames):
        Image.fromarray(frame).save(views_dir / f"{index:03d}.png")
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(output_dir / "turntable.gif", save_all=True,
                   append_images=images[1:], duration=80, loop=0)
    return frames


def render_view(gaussian, resolution, bg_color, r, fov, pitch_deg, yaw_start_deg, up=None):
    """One frame of ``notebook/inference.render_video``, with a choosable camera up.

    That renderer always uses the frame's +Y as up; ``up`` (None = +Y) replaces it.
    Cameras whose view direction is parallel to ``up`` fall back to +Y.
    """
    import math

    import torch
    import utils3d
    from sam3d_objects.model.backbone.tdfy_dit.utils import render_utils

    yaw, pitch = math.radians(yaw_start_deg), math.radians(pitch_deg)
    origin = r * np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch),
                           math.cos(yaw) * math.cos(pitch)])
    up = np.array([0.0, 1.0, 0.0]) if up is None else np.asarray(up, dtype=np.float64)
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
                     distance=1.0, up=None):
    """Render one image at each local azimuth/elevation pair, elevation-major.

    The camera orbits the normalized reconstruction (largest extent 1) at ``distance``
    with a 60-degree FOV; below about 1.6 the object is clipped by the frame.
    ``up`` is the camera up vector in that frame (None = +Y, the input camera's up).
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


def launch_viewer(ply_paths, host="0.0.0.0", port=7860):
    """Display all objects reconstructed by a pipeline run."""
    ply_paths = [Path(path) for path in ply_paths]
    if not ply_paths:
        raise ValueError("No reconstructed objects to visualize")
    for path in ply_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    import gradio as gr

    with gr.Blocks(title="SAM 3D Gaussian Splat") as app:
        gr.Markdown("# SAM 3D Gaussian Splat")
        for path in ply_paths:
            with gr.Tab(path.parent.name):
                gr.Model3D(value=str(path), label=path.parent.name)
    app.launch(server_name=host, server_port=port, share=False)
