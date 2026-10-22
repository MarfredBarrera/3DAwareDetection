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

    def _record_stage1(self, rgb, mask, seed):
        """Run stage 1 only with ``seed``; returns the recorder and its models."""
        from reconstruction import stage1_uncertainty as stage1

        image, mask = self._load_input(rgb, mask)
        pipeline = self._inference._pipeline
        generator, decoder = pipeline.models["ss_generator"], pipeline.models["ss_decoder"]
        if not stage1.is_data_at_one(generator):
            raise RuntimeError("stage-1 time convention differs from t = 1 at data")
        recorder = stage1.Recorder(generator)
        with recorder.attached():
            pipeline.run(self._inference.merge_mask_to_rgba(np.asarray(image), mask), None, seed,
                         stage1_only=True, with_mesh_postprocess=False, with_texture_baking=False,
                         with_layout_postprocess=True, use_vertex_color=True,
                         stage1_inference_steps=None, pointmap=None)
        if recorder.call is None:
            raise RuntimeError("stage 1 did not call the generator")
        return recorder, generator, decoder, pipeline.shape_model_dtype

    def stage1_ensemble(self, rgb, mask, seed=42, ensemble=3):
        """Seed-ensemble uncertainty of stage 1 alone (see reconstruction.stage1_uncertainty):
        ``seed`` (the reconstruction's own sample) plus ``ensemble - 1`` more seeds."""
        from reconstruction import stage1_uncertainty as stage1

        recorder, _, decoder, dtype = self._record_stage1(rgb, mask, seed)
        return stage1.ensemble(recorder, decoder, dtype, seed, ensemble)

    def stage1_fisher(self, rgb, mask, times, seed=42):
        """Summed squared shape-head NLL gradients (unit noise) at ``times``, plus residual sums."""
        from reconstruction import stage1_uncertainty as stage1

        recorder, generator, _, dtype = self._record_stage1(rgb, mask, seed)
        weight, bias, squared, elements = stage1.fisher_terms(
            recorder, generator.reverse_fn, stage1.shape_head(generator), times, dtype, seed)
        return {"weight": weight.cpu(), "bias": bias.cpu(), "squared": squared, "elements": elements}

    def stage1_prior(self, rgb, mask, seeds=(42,)):
        """Stage-1 occupancy (B, 64, 64, 64) with the image removed: SAM3D's unconditional
        shape prior, one sample per seed, from the same noise as the conditioned run."""
        import torch
        from functools import partial
        from reconstruction import stage1_uncertainty as stage1

        recorder, generator, decoder, dtype = self._record_stage1(rgb, mask, seeds[0])
        reverse_fn, backbone = generator.reverse_fn, generator.reverse_fn.backbone
        saved = (reverse_fn.strength, getattr(reverse_fn, "strength_pm", None))
        reverse_fn.strength = 0.0  # one (now unconditional) branch per step, no guidance
        if saved[1] is not None:
            reverse_fn.strength_pm = 0.0
        backbone.forward = partial(type(backbone).forward, backbone, cfg=True)  # condition zeroed
        try:
            draws = [recorder.resample(seed, dtype)[stage1.SHAPE] for seed in seeds]
        finally:
            del backbone.forward
            reverse_fn.strength = saved[0]
            if saved[1] is not None:
                reverse_fn.strength_pm = saved[1]
        return stage1.occupancy(decoder, torch.cat(draws), dtype)

    def stage1_mean_gap(self, rgb, mask, seed=42):
        """The reconstruction's stage-1 shape (``seed``) vs the one-step mean shapes (guided,
        unguided, and the image-free prior): latent gaps (mean squared difference per dimension),
        occupancies, and the seconds each pass adds (one network pass and one decode)."""
        import time
        import torch
        from reconstruction import stage1_uncertainty as stage1

        recorder, generator, decoder, dtype = self._record_stage1(rgb, mask, seed)
        sample = recorder.output[stage1.SHAPE].float()
        result = {"sample": stage1.occupancy(decoder, sample, dtype)[0]}
        for name in ("guided", "unguided", "prior"):
            torch.cuda.synchronize()
            started = time.perf_counter()
            mean = stage1.mean_prediction(recorder, generator.reverse_fn, dtype, name)[stage1.SHAPE]
            result[f"mean_{name}"] = stage1.occupancy(decoder, mean, dtype)[0]
            torch.cuda.synchronize()
            result[f"seconds_{name}"] = time.perf_counter() - started
            result[f"latent_gap_{name}"] = float(((sample - mean) ** 2).mean())
        return result

    def stage1_information(self, rgb, mask, seed=42, samples=32):
        """Pointwise mutual information between the input and the stage-1 shape (nats)."""
        from reconstruction import stage1_uncertainty as stage1

        recorder, generator, _, dtype = self._record_stage1(rgb, mask, seed)
        return stage1.information(recorder, generator.reverse_fn, samples, dtype, seed)

    def stage1_laplace(self, rgb, mask, std, seed=42, samples=4, noise=1e-3):
        """Last-layer Laplace generative uncertainty of stage 1 (arXiv:2502.20946).

        Same noise ``seed`` for every draw; ``samples`` shape-head weight sets
        from N(MAP, diag ``std``^2). Returns the occupancy summary of the draws
        plus ``entropy`` (nats per shape-latent dimension, sigma^2 = ``noise``).
        """
        import torch
        from reconstruction import stage1_uncertainty as stage1

        recorder, generator, decoder, dtype = self._record_stage1(rgb, mask, seed)
        head = stage1.shape_head(generator)
        weights = torch.Generator(device=head.weight.device).manual_seed(seed)
        draws = []
        for _ in range(samples):
            with stage1.perturbed(head, std, weights):
                draws.append(recorder.resample(seed, dtype)[stage1.SHAPE])
        shapes = torch.cat(draws)
        return {**stage1.summarize(stage1.occupancy(decoder, shapes, dtype)),
                "entropy": stage1.latent_entropy(shapes.float().cpu().numpy(), noise)}

    def render_views(self, rgb, mask, output_dir, views=8, resolution=512, seed=42, pointmap=None,
                     view_grid=None, save_ply=True, view_distance=1.0, also_turntable=False,
                     input_elevation=None, ensemble=1):
        """Write either a turntable or a viewpoint grid from one RGB and mask.

        Grid renders include ``mesh.npz`` for ambiguity scoring. The one-frame
        demo also writes a PLY for its interactive viewer. With the input
        camera's ``input_elevation`` (degrees, required for grids), grid cameras
        keep gravity up so candidates are level. (Keeping the input camera's up
        rolled every off-axis candidate of an elevated input; fixed 2026-10-18.)

        ``last_timing["model_seconds"]`` covers SAM3D inference and Gaussian
        preparation. It excludes input loading, mesh output, and view rendering.

        ``ensemble > 1`` also writes ``stage1.npz``: the seed ensemble of stage 1,
        with this run as its first member and ``ensemble - 1`` reruns of its
        recorded stage-1 call (``ensemble_seconds``), so no second pass over the input.
        """
        from contextlib import nullcontext
        from time import perf_counter

        if view_grid is not None and input_elevation is None:
            raise ValueError("view_grid needs input_elevation; without it candidates are rolled")
        image, mask = self._load_input(rgb, mask)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if pointmap is not None:
            pointmap = _pointmap(pointmap, image.size)
            import torch
            pointmap = torch.from_numpy(pointmap)

        # CUDA work is asynchronous. Synchronizing on either side gives an
        # accurate wall-clock duration for the model portion only.
        from reconstruction import stage1_uncertainty as stage1
        pipeline = self._inference._pipeline
        recorder = stage1.Recorder(pipeline.models["ss_generator"])
        _synchronize_cuda()
        model_started = perf_counter()
        with recorder.attached() if ensemble > 1 else nullcontext():
            output = self._inference(np.asarray(image), mask, seed=seed, pointmap=pointmap)
        scene = self._make_scene(output)
        gaussian = self._prepare_gaussian(scene)
        _synchronize_cuda()
        model_seconds = perf_counter() - model_started

        ensemble_started = perf_counter()
        if ensemble > 1:
            if recorder.call is None:
                raise RuntimeError("stage 1 did not call the generator")
            summary = stage1.ensemble(recorder, pipeline.models["ss_decoder"],
                                      pipeline.shape_model_dtype, seed, ensemble)
            stage1.save_ensemble(output_dir / "stage1.npz", summary, ensemble)
        ensemble_seconds = perf_counter() - ensemble_started

        mesh_started = perf_counter()
        if save_ply:
            gaussian.save_ply(str(output_dir / "object.ply"))
        if view_grid is not None:
            if "mesh" not in output:
                raise RuntimeError("SAM3D returned no mesh for viewpoint scoring")
            _save_render_mesh(output, scene, output_dir, save_ply)
        mesh_write_seconds = perf_counter() - mesh_started

        render_started = perf_counter()
        frames = (render_view_grid(gaussian, render_view, output_dir / "views",
                                   *view_grid, resolution=resolution, distance=view_distance,
                                   up=sam3d_gravity(input_elevation))
                  if view_grid is not None else
                  render_turntable(gaussian, self._render_video, output_dir, views, resolution))
        if view_grid is not None and also_turntable:
            render_turntable(gaussian, self._render_video, output_dir, views, resolution)
        self.last_timing = {
            "model_seconds": model_seconds,
            "ensemble_seconds": ensemble_seconds,
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
