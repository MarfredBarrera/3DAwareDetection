#!/usr/bin/env bash
# Check the image, CUDA device, and all libraries used by main.py.
set -euo pipefail
cd "$(dirname "$0")/.."

docker compose config --quiet
docker compose run --rm dev sam3d-env core python -c '
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
import sam3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
print(f"SAM3 torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")
'
docker compose run --rm dev sam3d-env sam3d python -c '
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
import pytorch3d, kaolin, gsplat, flash_attn, sam3d_objects
from fft.fft2d import calculate_hfer_robust
from inference import Inference, make_scene, ready_gaussian_for_video_rendering, render_video
print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")
print("SAM 3D dependencies available")
'
