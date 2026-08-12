# The app's environment. fal builds this server-side (no local Docker, no
# registry push) and caches by the CONTENT of this file - a digest-pinned
# FROM makes every base change a visible edit. (nvidia/cuda tag resolved
# 2026-08-10.)
#
# The stack tracks Meta's tested baseline for sam-3d-objects (doc/setup.md):
# CUDA 12.1, torch 2.5.1+cu121, kaolin/pytorch3d/flash-attn via the repo's
# own pinned requirement files - installing the repo at a fixed SHA pins the
# whole model stack transitively.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04@sha256:21196d81f56b48dbee70494d5f10322e1a77cc47ffe202a3bf68eab81533c20f

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC
# Headless rendering for pyrender/pytorch3d texture ops.
ENV PYOPENGL_PLATFORM=osmesa
# H100 (sm90). Add e.g. "8.9" here if you switch machine_type.
ENV TORCH_CUDA_ARCH_LIST="9.0+PTX"
# Index/links the upstream requirement files expect (doc/setup.md).
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

# python3.11 must match the Python that runs `fal deploy`: fal ships the app
# as cloudpickled bytecode, which is CPython-minor-specific. curl is required
# by fal's resolver. The GL/osmesa libs back headless rendering.
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
        curl ca-certificates git build-essential cmake ninja-build \
        libgl1 libglib2.0-0 libegl1 libglew-dev libosmesa6-dev && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    python3.11 -m ensurepip --upgrade && \
    python3.11 -m pip install --no-cache-dir pip==25.2

RUN python3.11 -m pip install --no-cache-dir torch==2.5.1+cu121 torchvision==0.20.1+cu121

# The model code, pinned by SHA. Its requirements*.txt files supply most of
# the stack (spconv, xformers, kaolin, gsplat, ...) transitively via '.' and
# '.[inference]'. flash-attn and pytorch3d are installed explicitly with
# --no-build-isolation because their setup.py imports torch, which pip's
# isolated build environment cannot see. '[dev]' is skipped (lint/test
# tooling we don't need). Build is cached by fal until this file changes.
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git /opt/sam-3d-objects && \
    cd /opt/sam-3d-objects && git checkout f91db411c50efee93d8db7aeb323885650f6f722
RUN cd /opt/sam-3d-objects && \
    # nvidia-pyindex is a pip-index-config shim incompatible with modern
    # pip's build isolation; PIP_EXTRA_INDEX_URL above covers its job.
    sed -i '/^nvidia-pyindex/d' requirements.txt && \
    # apt ships a distutils-installed blinker 1.4 in the shared dist-packages
    # dir; pip cannot uninstall it, so shadow it (Flask 3.x needs >=1.6).
    python3.11 -m pip install --no-cache-dir --ignore-installed blinker==1.8.2 && \
    python3.11 -m pip install --no-cache-dir '.' && \
    # flash-attn and pytorch3d import torch in setup.py; pip's isolated build
    # env cannot see the installed torch, so build them against the real env.
    python3.11 -m pip install --no-cache-dir ninja==1.11.1.4 psutil==7.0.0 wheel==0.45.1 && \
    MAX_JOBS=8 python3.11 -m pip install --no-cache-dir --no-build-isolation flash_attn==2.8.3 && \
    MAX_JOBS=8 python3.11 -m pip install --no-cache-dir --no-build-isolation "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47" && \
    python3.11 -m pip install --no-cache-dir '.[inference]' && \
    ./patching/hydra

# fal last, so it resolves against what is already installed.
RUN python3.11 -m pip install --no-cache-dir fal==1.78.2

# No ENTRYPOINT/CMD: fal supplies its own.

# sam3d_objects/__init__.py imports sam3d_objects.init, an internal Meta
# module absent from the OSS release; upstream gates that import behind this
# env var. Placed after the RUN layers so adding it never invalidates the
# expensive dependency cache.
ENV LIDRA_SKIP_INIT=true
