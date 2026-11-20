# The app's environment. fal builds this server-side (no local Docker, no
# registry push) and caches by the CONTENT of this file - a digest-pinned
# FROM makes every base change a visible edit. (nvidia/cuda tag resolved
# 2026-11-20.)
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04@sha256:17e2934e1fa96152b14f78078bfbafd0f00f391df995dc6c641a720fce1202bb

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

# python3.11 must match the Python that runs `fal deploy`: fal ships the app
# as cloudpickled bytecode, which is CPython-minor-specific. curl is required
# by fal's resolver. ffmpeg is the system encoder the app writes mp4s with.
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv curl ca-certificates ffmpeg && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    python3.11 -m ensurepip --upgrade && \
    python3.11 -m pip install --no-cache-dir pip==25.2

# The model stack. Keep in sync with requirements.txt.
RUN python3.11 -m pip install --no-cache-dir \
        torch==2.12.1 \
        diffusers==0.39.0 \
        transformers==5.3.0 \
        accelerate==1.14.0 \
        pyzmq==27.1.0 \
        hf-transfer==0.1.9 \
        huggingface_hub==1.27.0 \
        ftfy==6.3.1 \
        imageio==2.37.4 \
        imageio-ffmpeg==0.6.0

# fal last, so it resolves against what is already installed.
RUN python3.11 -m pip install --no-cache-dir fal==1.78.2

# No ENTRYPOINT/CMD: fal supplies its own.
