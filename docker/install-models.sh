#!/usr/bin/env bash
set -euo pipefail
export PATH="/opt/envs/sam3d/bin:/usr/local/cuda-12.1/bin:/usr/bin:/bin:/usr/local/bin"
export CUDA_HOME=/usr/local/cuda-12.1
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
export FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.9
pip install --no-cache-dir setuptools==75.8.0 wheel==0.45.1 packaging==24.2 ninja==1.11.1.3
pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir -r /opt/manifests/requirements-sam3d.txt
pip install --no-cache-dir xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir --no-build-isolation flash-attn==2.8.3
pip install --no-cache-dir --no-build-isolation 'pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47'
pip install --no-cache-dir kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html
pip install --no-cache-dir --no-build-isolation 'gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7'
pip install --no-cache-dir --no-deps 'moge @ git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b'
clone-pinned wlfeng0509/Fast-SAM3D c76188fda755b761fe9255eabe87aaad39df829a /opt/src/fast-sam3d
clone-pinned facebookresearch/dinov2 7764ea0f912e53c92e82eb78a2a1631e92725fc8 /opt/src/dinov2
python /opt/patch-fast-sam3d.py /opt/src/fast-sam3d/sam3d_objects
pip install --no-cache-dir --no-deps /opt/src/fast-sam3d
python /opt/src/fast-sam3d/patching/hydra
ln -s /usr/local/cuda-12.1/bin/nvcc /opt/envs/sam3d/bin/nvcc
ln -s /usr/local/cuda-12.1/include /opt/envs/sam3d/include-cuda
cp -as /usr/local/cuda-12.1/targets/x86_64-linux/include/. /opt/envs/sam3d/include/
ln -s /usr/local/cuda-12.1/lib64 /opt/envs/sam3d/lib64
pip freeze > /opt/manifests/sam3d.freeze.txt
