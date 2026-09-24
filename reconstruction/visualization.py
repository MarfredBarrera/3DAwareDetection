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
