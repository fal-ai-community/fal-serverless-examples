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
from dataclasses import dataclass

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


@dataclass(frozen=True)
class RunnerWindow:
    """One runner's billed lifetime.

    `setup_start_s` is when billing starts (the SETUP state). Everything
    before it - PENDING, DOCKER_PULL - is free. `last_completion_s` is when
    it finished its last request; it stays alive, and billing, for
    keep_alive seconds after that.
    """

    runner_id: str
    setup_start_s: float
    last_completion_s: float


def billable_seconds(
    windows: list[RunnerWindow], *, keep_alive: float, granularity: float = 1.0
) -> float:
    """Billed runner-seconds across every runner in `windows`.

    ASSUMPTION: each runner's alive time is rounded up to `granularity`.
    fal publishes no billing granularity or minimum billable duration, so
    the default of 1s is a stated assumption, not a documented fact. It
    matters most at short keep_alive values.
    """
    total = 0.0
    for window in windows:
        alive = (window.last_completion_s + keep_alive) - window.setup_start_s
        if alive <= 0:
            continue
        total += math.ceil(alive / granularity) * granularity
    return total


def cost_usd(billable_s: float, *, usd_per_second: float = USD_PER_SECOND) -> float:
    return billable_s * usd_per_second
