# The pipeline's environment. fal builds this server-side - no local Docker,
# no registry push. Digest-pinned for the same reason as 01-environments-
# containers/same-model-two-ways: fal's build cache keys on the CONTENT of
# this file, so a digest makes every base-image change a visible edit.
# (python:3.11-slim, 2026-07-25.)
FROM python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93

# python3.11 must match the Python that runs `fal deploy` (apps ship as
# cloudpickled bytecode, which is CPython-minor-specific). curl is fal's
# installer dependency.
#
# ffmpeg IS this example: apt pins it to Debian bookworm's 5.1.x line, so
# every worker encodes with the exact same toolchain. The app echoes
# `ffmpeg -version` into every report for provenance.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# fal last, so it resolves against what is already installed.
RUN pip install --no-cache-dir fal==1.78.2

# No ENTRYPOINT/CMD: fal supplies its own.
