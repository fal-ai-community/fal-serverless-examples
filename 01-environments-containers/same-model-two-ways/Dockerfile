# Way 2's environment, built server-side by fal.
#
# Pinned by digest: fal's build cache keys on this file's content, so a moved
# upstream tag would never be picked up. (python:3.11-slim as of 2026-07-25.)
FROM python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93

# python3.11 on PATH must match the Python you run `fal deploy` with: fal
# ships your app as cloudpickled bytecode, which is CPython-minor-specific.
# A mismatch fails at runtime with an opaque connection error, not a clear
# message - this is the single most common custom-container mistake.
#
# curl is required too: fal's default resolver fetches its installer with it.
#
# ffmpeg: transformers decodes audio through it; fal's runtime image ships it,
# this slim base does not.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Dependencies baked into the image. Same versions as the runtime variant's
# App.requirements - the comparison is only honest if the environments match.
RUN pip install --no-cache-dir \
        torch==2.12.1 \
        transformers==5.3.0

# fal last, so it resolves against what is already installed.
RUN pip install --no-cache-dir fal==1.78.2

# No ENTRYPOINT/CMD: fal supplies its own.
