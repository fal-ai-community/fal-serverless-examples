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


TICK_S = 0.1
_MAX_TICKS = 2_000_000  # runaway guard; 0.1s ticks = ~55 hours of sim time


def _grid(value: float) -> float:
    """Quantize a derived timestamp to the tick clock's precision.

    `t` is always round(step * tick_s, 6). Derived times must be rounded the
    same way or float error makes a completion land one tick late - and
    because the next completion is computed from that late value, the error
    compounds instead of cancelling.
    """
    return round(value, 6)


def arrival_schedule(pattern: str, *, requests: int, duration_s: float) -> list[float]:
    """Deterministic arrival times, in seconds from burst start.

    The real burst driver fires this exact schedule, so a simulated run and
    a measured run share their traffic shape.
    """
    if requests < 1:
        raise ValueError("requests must be >= 1")
    if pattern == "spike":
        return [0.0] * requests
    if pattern == "steady":
        step = duration_s / requests
        return [round(i * step, 6) for i in range(requests)]
    if pattern == "ramp":
        return [round(duration_s * (i / requests) ** 2, 6) for i in range(requests)]
    raise ValueError(f"unknown pattern {pattern!r}: use spike, ramp or steady")


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile. No interpolation - these are small samples."""
    if not values:
        raise ValueError("percentile of an empty sample")
    ordered = sorted(values)
    rank = math.ceil(p / 100 * len(ordered)) - 1
    return ordered[max(0, min(rank, len(ordered) - 1))]


@dataclass
class SimResult:
    timeline: list[dict]
    waits: list[float]
    windows: list[RunnerWindow]


def simulate(
    arrivals: list[float],
    *,
    service_s: float,
    cold_start_s: float,
    keep_alive: float,
    min_concurrency: int = 0,
    max_concurrency: int | None = None,
    concurrency_buffer: int = 0,
    concurrency_buffer_perc: int = 0,
    max_multiplexing: int = 1,
    scaling_delay: float = 0.0,
    tick_s: float = TICK_S,
) -> SimResult:
    """Fixed-step simulation of one burst.

    A fixed 0.1s tick rather than an event queue: it is far easier to verify
    against the published diagrams, and 0.1s resolution is well below the
    10s floor of fal's own metrics endpoint anyway.

    Scale-down is modelled as keep_alive expiry only, because fal publishes
    no other scale-down dynamics. That is why this example MEASURES a real
    burst - the measurement is the authority, and the residual chart shows
    where this model disagrees with it.
    """
    pending = sorted(arrivals)
    queue: list[tuple[float, int]] = []
    runners: list[dict] = []
    waits: dict[int, float] = {}
    timeline: list[dict] = []
    next_arrival = 0
    next_request_id = 0

    def start_runner(at: float) -> dict:
        ready_at = _grid(at + cold_start_s)
        runner = {
            "id": f"sim-{len(runners):04d}",
            "setup_start": at,
            "ready_at": ready_at,
            "inflight": [],
            "last_activity": ready_at,
            "terminated_at": None,
        }
        runners.append(runner)
        return runner

    prestart = min_concurrency
    if max_concurrency is not None:
        prestart = min(prestart, max_concurrency)
    for _ in range(prestart):
        start_runner(0.0)  # a min_concurrency runner still pays for SETUP

    for step in range(_MAX_TICKS):
        t = round(step * tick_s, 6)

        while next_arrival < len(pending) and pending[next_arrival] <= t:
            queue.append((pending[next_arrival], next_request_id))
            next_request_id += 1
            next_arrival += 1

        alive = [r for r in runners if r["terminated_at"] is None]

        for runner in alive:
            done = [c for c in runner["inflight"] if c <= t]
            if done:
                runner["inflight"] = [c for c in runner["inflight"] if c > t]
                runner["last_activity"] = max(runner["last_activity"], max(done))

        for runner in alive:
            while (
                queue
                and runner["ready_at"] <= t
                and len(runner["inflight"]) < max_multiplexing
            ):
                arrived_at, request_id = queue.pop(0)
                waits[request_id] = t - arrived_at
                runner["inflight"].append(_grid(t + service_s))
                runner["last_activity"] = max(runner["last_activity"], t)

        in_flight = sum(len(r["inflight"]) for r in alive)
        # scaling_delay: a queued request only counts toward the desired
        # runner count once it has waited that long to be picked up.
        aged = sum(1 for arrived_at, _ in queue if t - arrived_at >= scaling_delay)
        want = desired_runners(
            in_flight + aged,
            max_multiplexing=max_multiplexing,
            concurrency_buffer=concurrency_buffer,
            concurrency_buffer_perc=concurrency_buffer_perc,
            min_concurrency=min_concurrency,
            max_concurrency=max_concurrency,
        )
        while len(alive) < want:
            alive.append(start_runner(t))

        for runner in alive:
            if (
                not runner["inflight"]
                and runner["ready_at"] <= t
                and (t - runner["last_activity"]) >= keep_alive
                and len([r for r in alive if r["terminated_at"] is None]) > want
            ):
                runner["terminated_at"] = t

        live = [r for r in runners if r["terminated_at"] is None]
        timeline.append(
            {
                "t": t,
                "setup": sum(1 for r in live if r["ready_at"] > t),
                "running": sum(1 for r in live if r["inflight"]),
                "idle": sum(1 for r in live if r["ready_at"] <= t and not r["inflight"]),
                "queue_depth": len(queue),
            }
        )

        # Settle when the traffic is gone AND every surviving runner has sat
        # idle for keep_alive. `not live` alone is wrong: min_concurrency
        # runners never terminate, so the loop would run to the guard.
        drained = next_arrival >= len(pending) and not queue
        settled = drained and not any(r["inflight"] for r in live)
        if settled and all((t - r["last_activity"]) >= keep_alive for r in live):
            break
    else:  # pragma: no cover - the guard should never trip in practice
        raise RuntimeError("simulation did not settle; check keep_alive/min_concurrency")

    return SimResult(
        timeline=timeline,
        waits=[waits[i] for i in sorted(waits)],
        windows=[
            RunnerWindow(r["id"], r["setup_start"], r["last_activity"]) for r in runners
        ],
    )


TRACE_SCHEMA = "fal-serverless-examples/tune-autoscaling/trace@1"


@dataclass(frozen=True)
class Calibration:
    """What one measured burst tells us, so the model can extrapolate.

    Everything here is measured except the scale_config, which is declared.
    """

    cold_start_s: float
    service_s: float
    arrivals: tuple[float, ...]
    observed_runners: int
    scale_config: dict
    machine_type: str


def _ok_requests(trace: dict) -> list[dict]:
    """Successful requests only. fal does not bill 5xx, and a failed request
    carries no usable timing."""
    return [r for r in trace["requests"] if 200 <= int(r["status"]) < 300]


def _by_runner(requests: list[dict]) -> dict[str, list[dict]]:
    """Group requests by the runner that served them, in id order."""
    grouped: dict[str, list[dict]] = {}
    for request in requests:
        grouped.setdefault(str(request["served_by"]), []).append(request)
    return dict(sorted(grouped.items()))


def _check_schema(trace: dict) -> None:
    if trace.get("schema") != TRACE_SCHEMA:
        raise ValueError(
            f"unsupported trace schema {trace.get('schema')!r}; expected {TRACE_SCHEMA!r}"
        )


def calibrate(trace: dict) -> Calibration:
    """Extract cold start, service time and traffic shape from a real burst."""
    _check_schema(trace)
    ok = _ok_requests(trace)
    if not ok:
        raise ValueError("trace contains no successful requests")

    cold = [float(r["setup_duration_s"]) for r in ok if r["was_cold"]]
    if not cold:
        raise ValueError("trace contains no cold request; cannot measure cold start")

    # Service time from back-to-back completions on ONE runner. A saturated
    # runner's consecutive completions are separated by exactly one service
    # time. Round-trip minus arrival would NOT work: in a spike, warm
    # requests also queued behind a cold start, so that inflates service time
    # and drives the derived SETUP start negative.
    gaps = []
    for requests in _by_runner(ok).values():
        completions = sorted(float(r["completed_s"]) for r in requests)
        gaps.extend(b - a for a, b in zip(completions, completions[1:]))
    if not gaps:
        raise ValueError(
            "no runner served two requests; cannot measure service time - "
            "fire more requests than max_concurrency so at least one runner "
            "serves back-to-back"
        )

    return Calibration(
        # Nearest-rank p50, the same estimator as every other percentile
        # here - not an interpolated median. One convention, deliberately.
        cold_start_s=percentile(cold, 50),
        service_s=min(gaps),
        arrivals=tuple(float(r["arrived_s"]) for r in ok),
        observed_runners=len({r["served_by"] for r in ok}),
        scale_config=dict(trace["scale_config"]),
        machine_type=str(trace["machine_type"]),
    )


def measured_windows(trace: dict) -> list[RunnerWindow]:
    """One billed window per runner actually observed in the burst.

    Derived, not fetched: SETUP start is recovered as
    (first completion) - setup_duration_s - service_s, using the app's
    self-reported setup duration. /v1/serverless/analytics is a cross-check
    for this, not its source - see burst.py --from-analytics.
    """
    calibration = calibrate(trace)
    windows = []
    for runner_id, requests in _by_runner(_ok_requests(trace)).items():
        completions = [float(r["completed_s"]) for r in requests]
        setup_duration = float(requests[0]["setup_duration_s"])
        windows.append(
            RunnerWindow(
                runner_id=runner_id,
                setup_start_s=min(completions) - setup_duration - calibration.service_s,
                last_completion_s=max(completions),
            )
        )
    return windows


def measured_timeline(trace: dict, *, tick_s: float = TICK_S) -> list[dict]:
    """Runner states and queue depth over time, reconstructed from the trace.

    Same row shape as SimResult.timeline so the two can be overlaid. Note
    this is a reconstruction from per-request observations, not a sampled
    metric - fal's own metrics endpoint caches for 10s, which is coarser
    than this.
    """
    calibration = calibrate(trace)
    ok = _ok_requests(trace)
    windows = {w.runner_id: w for w in measured_windows(trace)}
    grouped = _by_runner(ok)
    # Each runner's own reported setup duration, not the fleet median - this
    # is a reconstruction of what actually happened, runner by runner.
    ready_at = {
        runner_id: windows[runner_id].setup_start_s
        + float(grouped[runner_id][0]["setup_duration_s"])
        for runner_id in windows
    }
    busy: dict[str, list[tuple[float, float]]] = {rid: [] for rid in windows}
    for request in ok:
        runner_id = str(request["served_by"])
        completed = float(request["completed_s"])
        busy[runner_id].append((completed - calibration.service_s, completed))

    start = min(w.setup_start_s for w in windows.values())
    end = max(w.last_completion_s for w in windows.values())
    keep_alive = float(calibration.scale_config.get("keep_alive", 0))

    timeline = []
    steps = int(math.ceil((end - start) / tick_s)) + 1
    for step in range(steps):
        t = round(start + step * tick_s, 6)
        setup = idle = running = 0
        for runner_id, window in windows.items():
            if t < window.setup_start_s or t > window.last_completion_s + keep_alive:
                continue
            if t < ready_at[runner_id]:
                setup += 1
            elif any(begin <= t < finish for begin, finish in busy[runner_id]):
                running += 1
            else:
                idle += 1
        queued = sum(
            1
            for r in ok
            if float(r["arrived_s"]) <= t
            < float(r["completed_s"]) - calibration.service_s
        )
        timeline.append(
            {"t": t, "setup": setup, "idle": idle, "running": running, "queue_depth": queued}
        )
    return timeline
