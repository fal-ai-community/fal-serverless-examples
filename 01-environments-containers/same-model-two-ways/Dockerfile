# Way 2's environment. fal builds this server-side - you never need Docker
# installed locally, and you never push to a registry.
#
# Pinned by digest, not by tag, and that is not pedantry: fal's build cache
# is keyed on the CONTENT OF THIS FILE, not on what a tag resolves to today.
# With `FROM python:3.11-slim`, a repointed upstream tag would never reach
# your app - fal would keep serving the image it already built. A digest
# makes every base-image change a visible edit to this file, which is also
# what makes the cache do the right thing. (python:3.11-slim, 2026-07-25.)
FROM python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93

# python3.11 on PATH must match the Python you run `fal deploy` with: fal
# ships your app as cloudpickled bytecode, which is CPython-minor-specific.
# A mismatch fails at runtime with an opaque connection error, not a clear
# message - this is the single most common custom-container mistake.
#
# curl is required too: fal's default resolver fetches its installer with it.
#
# ffmpeg is the honest reason this example has a container variant at all:
# transformers decodes audio through it, fal's own runtime image already
# ships it, and this slim base does not. System packages are what you get a
# Dockerfile FOR - without this line the container variant fails to decode a
# single file while the runtime variant sails through.
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
