"""Autoscaler + billing model for catalog slot 02.2 (SERV-1294).

Stdlib only, on purpose: this repo requires requirements.txt to mirror
App.requirements exactly, so any dependency here would ship to the GPU
runner. Being dependency-free also makes the output deterministic, which is
what lets CI regenerate the committed charts and fail on a mismatch.

THE SCALING LAW BELOW IS RECONSTRUCTED, NOT DOCUMENTED. fal publishes no
formula in prose. It was derived from the labelled bar values of the three
diagrams on docs.fal.ai/documentation/deployment/scale-your-application and
matches all three exactly (see test_app.py::PUBLISHED_CHARTS). Treat it as
a model of the autoscaler, and note that scale-down dynamics are published
nowhere at all - which is why this example measures them instead.
"""
from __future__ import annotations

import math

# Published list price for GPU-H100 on fal.ai/pricing. The negotiated
# "as low as" tier is $1.89/h; a reader on that tier should pass --rate.
H100_USD_PER_HOUR = 3.99
USD_PER_SECOND = H100_USD_PER_HOUR / 3600


def desired_runners(
    ongoing: int,
    *,
    max_multiplexing: int = 1,
    concurrency_buffer: int = 0,
    concurrency_buffer_perc: int = 0,
    min_concurrency: int = 0,
    max_concurrency: int | None = None,
) -> int:
    """How many runners the autoscaler wants for `ongoing` requests.

    `ongoing` counts in-flight plus queued requests. The percentage buffer
    is rounded up - fal documents the max() rule but not the rounding, so
    this is an explicit assumption of the model.
    """
    buffer = max(
        concurrency_buffer,
        math.ceil(concurrency_buffer_perc / 100 * ongoing),
    )
    want = math.ceil(ongoing / max_multiplexing) + buffer
    want = max(want, min_concurrency)
    if max_concurrency is not None:
        want = min(want, max_concurrency)
    return want
