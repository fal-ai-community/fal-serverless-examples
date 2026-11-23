# Digest-pinned: fal caches builds by this file's content (tag resolved 2026-11-20).
# CUDA 13.0.2 matches the cuda-toolkit that the torch==2.12.1 wheel depends on.
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu22.04@sha256:78badc969903e2aebc515b087f6276a15eae17c6aade19068b18dc252bd6ec90

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

# python3.11 must match the Python that runs `fal deploy`; curl is fal's installer dependency.
# Symlinks go in /usr/local/bin: apt's tooling needs /usr/bin/python3 to stay 3.10.
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv curl ca-certificates ffmpeg && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
    python3.11 -m ensurepip --upgrade && \
    python3.11 -m pip install --no-cache-dir pip==25.2

# Keep in sync with requirements.txt.
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
