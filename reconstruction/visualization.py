"""Render SAM 3D Gaussian splats into reusable image outputs."""
from pathlib import Path

import numpy as np
from PIL import Image


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


def render_view_grid(gaussian, render_video, output_dir, azimuths, elevations, resolution=512):
    """Render one image at each local azimuth/elevation pair, elevation-major."""
    if resolution < 64:
        raise ValueError("resolution must be at least 64")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for elevation in elevations:
        for azimuth in azimuths:
            def render_at(background):
                result = render_video(
                    gaussian, num_frames=1, resolution=resolution,
                    bg_color=background, r=1, fov=60,
                    pitch_deg=float(elevation), yaw_start_deg=float(azimuth),
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
