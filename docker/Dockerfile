# syntax=docker/dockerfile:1
# SAM3 and Fast-SAM3D need separate PyTorch/CUDA environments.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04@sha256:7012e535a47883527d402da998384c30b936140c05e2537158c80b8143ee7425 AS cuda121
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04@sha256:50efab398f76258daa91ceebb33b6467e40217c67ea44fb5a2cebc6be7d9cce3 AS core
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_JOBS=4
ENV LANG=C.UTF-8 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 \
    TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=${BUILD_JOBS} CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics EGL_PLATFORM=surfaceless
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git cpulimit build-essential gcc-12 g++-12 pkg-config cmake ninja-build \
    python3-venv python3-dev libjpeg-dev libglm-dev libegl1-mesa-dev \
    libgl1-mesa-dev libglib2.0-0 \
    libgomp1 libxcb1 libxrender1 libsm6 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY docker/clone-pinned.sh /usr/local/bin/clone-pinned
RUN chmod +x /usr/local/bin/clone-pinned && python3 -m venv /opt/envs/core
ENV PATH=/opt/envs/core/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN pip install --no-cache-dir pip==25.0.1 setuptools==75.8.0 wheel==0.45.1 \
    && pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
COPY docker/requirements-core.txt /opt/manifests/requirements-core.txt
RUN pip install --no-cache-dir -r /opt/manifests/requirements-core.txt
ARG SAM3_REF=96914d2425f90a64f45ca977c2b5165418099543
RUN clone-pinned facebookresearch/sam3 "$SAM3_REF" /opt/src/sam3 \
    && pip install --no-cache-dir --no-deps /opt/src/sam3
COPY docker/entrypoint.sh /usr/local/bin/docker-entrypoint
COPY docker/sam3d-env /usr/local/bin/sam3d-env
RUN chmod +x /usr/local/bin/docker-entrypoint /usr/local/bin/sam3d-env && mkdir -p /workspace /data /outputs \
    && pip freeze > /opt/manifests/core.freeze.txt
ENV PYTHONPATH=/workspace
WORKDIR /workspace
ENTRYPOINT ["docker-entrypoint"]
CMD ["bash"]

FROM core AS full
# CUDA 12.1's compiler is needed for the legacy native extensions; the driver is
# supplied by the host. Keeping both toolkits does not create additional images.
COPY --from=cuda121 /usr/local/cuda-12.1 /usr/local/cuda-12.1
ARG BUILD_JOBS=4
ARG MINIFORGE_VERSION=25.3.0-3
# CUDA 12.1 accepts GCC through version 12.  Ubuntu 24.04 defaults to GCC 13,
# so every extension in the legacy SAM3D stack must use this explicit toolchain.
ENV CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 CUDAHOSTCXX=/usr/bin/g++-12 \
    NVCC_CCBIN=/usr/bin/g++-12
RUN curl -fsSL "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda create -y -p /opt/envs/sam3d python=3.11.11 pip=25.0.1 \
    && /opt/conda/bin/conda clean -afy
COPY docker/requirements-sam3d.txt /opt/manifests/
COPY docker/install-models.sh /opt/install-models.sh
COPY docker/patch-fast-sam3d.py /opt/patch-fast-sam3d.py
RUN bash /opt/install-models.sh
CMD ["bash"]
