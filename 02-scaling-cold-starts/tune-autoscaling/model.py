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

import argparse
import json
import math
import sys
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

    `prewarmed` marks a runner that was already alive before the burst (one
    of the min_concurrency floor) rather than one the autoscaler started in
    response to it. The distinction matters for costing: a prewarmed
    runner's real cost is a horizon-wide floor - it is alive around the
    clock, burst or no burst - not a per-burst charge like the rest of the
    fleet. sweep() relies on this flag to keep the two apart; folding a
    prewarmed window into the per-burst demand would double-count it.
    """

    runner_id: str
    setup_start_s: float
    last_completion_s: float
    prewarmed: bool = False


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
    burst - the measurement is the authority, and README.md's "Cost &
    tuning" table is where a reader compares this model's output against
    it (chart.py renders the burst timeline and the cost-latency frontier;
    there is no third, residual chart).
    """
    pending = sorted(arrivals)
    queue: list[tuple[float, int]] = []
    runners: list[dict] = []
    waits: dict[int, float] = {}
    timeline: list[dict] = []
    next_arrival = 0
    next_request_id = 0

    def start_runner(at: float, *, prewarmed: bool = False) -> dict:
        # A min_concurrency runner is alive around the clock, so by the time a
        # burst arrives it booted long ago and is already READY. Only runners
        # the autoscaler starts in response to this burst pay a cold start.
        ready_at = _grid(at) if prewarmed else _grid(at + cold_start_s)
        runner = {
            "id": f"sim-{len(runners):04d}",
            "setup_start": at,
            "ready_at": ready_at,
            "inflight": [],
            "last_activity": ready_at,
            "terminated_at": None,
            "prewarmed": prewarmed,
        }
        runners.append(runner)
        return runner

    prestart = min_concurrency
    if max_concurrency is not None:
        prestart = min(prestart, max_concurrency)
    for _ in range(prestart):
        # Alive around the clock, already warm - it is billed from the
        # window start, not from a cold start paid at t=0.
        start_runner(0.0, prewarmed=True)

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
            if runner["prewarmed"]:
                # A prewarmed runner holds the min_concurrency floor and is
                # alive around the clock - it is not idle-and-disposable, it
                # IS the floor. Terminating it here would just hand the
                # floor's identity to whichever runner the autoscaler
                # started to handle this burst (the next check down would
                # then treat THAT runner as permanently alive instead), so
                # it is never a scale-down candidate.
                continue
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

    end_t = timeline[-1]["t"]
    windows = []
    for runner in runners:
        last = runner["last_activity"]
        if runner["terminated_at"] is None:
            # Never scaled down, so it is alive for the whole window. Two
            # kinds of runner can end up here: a prewarmed one (skipped
            # above by construction - it holds the min_concurrency floor),
            # or an ordinary demand runner that desired_runners() never
            # actually asked to drop below (e.g. concurrency_buffer alone
            # can hold `want` at 1 forever, even at zero load). Either way
            # it is genuinely still alive at end_t, so it gets the same
            # treatment. billable_seconds adds keep_alive to
            # last_completion_s, so subtract it here to make the billed span
            # exactly the window rather than the window plus a tail that
            # never happened.
            last = max(last, end_t - keep_alive)
        windows.append(
            RunnerWindow(
                runner["id"], runner["setup_start"], last, prewarmed=runner["prewarmed"]
            )
        )

    return SimResult(
        timeline=timeline,
        waits=[waits[i] for i in sorted(waits)],
        windows=windows,
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
    """Successful requests with a usable runner identity.

    fal does not bill 5xx, and a failed request carries no usable timing -
    so status must be 2xx. Separately, a 2xx response whose body is missing
    the expected fields yields served_by == "" ; a record with no runner
    identity cannot be grouped into a real runner's window, and keeping it
    would fabricate a phantom runner out of every such record (polluting
    the service_s minimum, contaminating the cold_start_s sample, and
    inflating observed_runners)."""
    return [
        r
        for r in trace["requests"]
        if 200 <= int(r["status"]) < 300 and str(r["served_by"])
    ]


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
        # Select by the cold flag, not list position: _by_runner preserves
        # whatever order it was given, and a runner's cold request is the
        # one whose setup_duration_s is authoritative (set once in setup()
        # and echoed by every response). Fall back to the first request if
        # none is flagged cold.
        cold_first = next((r for r in requests if r["was_cold"]), requests[0])
        setup_duration = float(cold_first["setup_duration_s"])
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
    # Select by the cold flag rather than list position, same as
    # measured_windows: _by_runner preserves whatever order it was given,
    # not necessarily the cold request first. Fall back to the first
    # request if none is flagged cold.
    ready_at = {
        runner_id: windows[runner_id].setup_start_s
        + float(
            next((r for r in requests if r["was_cold"]), requests[0])[
                "setup_duration_s"
            ]
        )
        for runner_id, requests in grouped.items()
    }
    busy: dict[str, list[tuple[float, float]]] = {rid: [] for rid in windows}
    for request in ok:
        runner_id = str(request["served_by"])
        completed = float(request["completed_s"])
        busy[runner_id].append((completed - calibration.service_s, completed))

    start = min(w.setup_start_s for w in windows.values())
    # keep_alive must be looked up before `end`: fal bills a runner from
    # SETUP start until keep_alive seconds after its last request finishes,
    # and the per-tick loop below treats a runner as alive through
    # last_completion_s + keep_alive. `end` has to bound the tick range at
    # that same point, or the idle keep_alive tail - the whole visual point
    # of this example - is never emitted.
    keep_alive = float(calibration.scale_config.get("keep_alive", 0))
    end = max(w.last_completion_s + keep_alive for w in windows.values())

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


FRONTIER_SCHEMA = "fal-serverless-examples/tune-autoscaling/frontier@1"

# Small enough to label legibly on one chart, wide enough to show the
# trade-off: scale-to-zero through always-warm.
DEFAULT_GRID = {
    "keep_alive": (0, 30, 60, 300),
    "min_concurrency": (0, 1, 2),
    "concurrency_buffer": (0, 1),
}


def sweep(
    calibration: Calibration,
    *,
    grid: dict | None = None,
    usd_per_second: float | None = None,
    granularity: float = 1.0,
    bursts_per_hour: float = 1.0,
    horizon_s: float = 3600.0,
) -> list[dict]:
    """Simulate the same measured traffic under every knob combination.

    Cost is charged over an explicit horizon (default one hour), not the
    burst window alone. A min_concurrency runner is alive around the clock -
    that is what the knob means - so its real cost is a horizon-wide floor of
    `min_concurrency * horizon_s` runner-seconds, independent of how often
    the burst repeats. Every OTHER runner only exists because of the burst,
    so it is charged per repeat: its burst-window billable seconds times
    `bursts_in_horizon = bursts_per_hour * horizon_s / 3600`. Prewarmed
    runners (the RunnerWindow.prewarmed flag set by simulate()) are excluded
    from that per-burst demand and charged only via the floor, or their cost
    would be counted twice.

    `billable_s` on each point is that total (floor + demand) over the WHOLE
    horizon, not per burst. `cost_usd` is that same horizon cost in dollars -
    `billable_s * usd_per_second`, not renormalised to a one-hour rate. The
    horizon itself is recorded once, alongside the points, in the frontier
    document (see build_frontier()) rather than on every point, so a reader
    who wants a different normalisation (a one-hour rate, a per-burst
    figure, whatever) can recover it from `horizon_s` without this function
    shipping a redundant field for every possible one.

    `billable_s` is rounded to 3dp for display. `cost_usd` is computed from
    that already-rounded `billable_s`, not the unrounded total - so a reader
    who multiplies the two printed numbers by hand (`billable_s *
    usd_per_second`) reproduces `cost_usd` exactly, rather than
    occasionally landing a cent off in the last decimal.
    """
    grid = grid or DEFAULT_GRID
    rate = USD_PER_SECOND if usd_per_second is None else usd_per_second
    base = calibration.scale_config
    max_multiplexing = int(base.get("max_multiplexing", 1))
    buffer_perc = int(base.get("concurrency_buffer_perc", 0))
    bursts_in_horizon = bursts_per_hour * horizon_s / 3600.0

    points = []
    for keep_alive in grid["keep_alive"]:
        for min_concurrency in grid["min_concurrency"]:
            for concurrency_buffer in grid["concurrency_buffer"]:
                result = simulate(
                    list(calibration.arrivals),
                    service_s=calibration.service_s,
                    cold_start_s=calibration.cold_start_s,
                    keep_alive=float(keep_alive),
                    min_concurrency=int(min_concurrency),
                    max_concurrency=base.get("max_concurrency"),
                    concurrency_buffer=int(concurrency_buffer),
                    concurrency_buffer_perc=buffer_perc,
                    max_multiplexing=max_multiplexing,
                    scaling_delay=float(base.get("scaling_delay", 0)),
                )
                # Prewarmed (min_concurrency) windows are billed only via the
                # horizon-wide floor below - excluding them here is what
                # keeps them from being counted a second time as per-burst
                # demand.
                demand_windows = [w for w in result.windows if not w.prewarmed]
                demand_billable = billable_seconds(
                    demand_windows, keep_alive=float(keep_alive), granularity=granularity
                )
                floor_s = min_concurrency * horizon_s
                demand_s = demand_billable * bursts_in_horizon
                total_s = floor_s + demand_s
                # Round billable_s FIRST, then price that rounded value -
                # not the unrounded total - so cost_usd is exactly
                # billable_s * rate for the numbers this point actually
                # prints (see the docstring above).
                billable_s = round(total_s, 3)
                cost = cost_usd(billable_s, usd_per_second=rate)
                peak = max(
                    row["setup"] + row["idle"] + row["running"] for row in result.timeline
                )
                points.append(
                    {
                        "keep_alive": int(keep_alive),
                        "min_concurrency": int(min_concurrency),
                        "concurrency_buffer": int(concurrency_buffer),
                        "concurrency_buffer_perc": buffer_perc,
                        "max_concurrency": base.get("max_concurrency"),
                        "max_multiplexing": max_multiplexing,
                        "p50_wait_s": round(percentile(result.waits, 50), 3),
                        "p95_wait_s": round(percentile(result.waits, 95), 3),
                        "billable_s": billable_s,
                        "cost_usd": round(cost, 6),
                        "peak_runners": peak,
                        # Beyond the measured envelope this is extrapolation:
                        # we never observed that many runners, so the chart
                        # must not present it as an equal to a validated point.
                        "validated": peak <= calibration.observed_runners,
                        "measured": (
                            int(keep_alive) == int(base.get("keep_alive", -1))
                            and int(min_concurrency) == int(base.get("min_concurrency", -1))
                            and int(concurrency_buffer)
                            == int(base.get("concurrency_buffer", -1))
                        ),
                    }
                )
    return points


def _tick_quantized(value: float, *, tick_s: float = TICK_S) -> float:
    """Round `value` to the nearest multiple of the simulation's own tick.

    p95_wait_s is a simulation output, so any difference smaller than one
    tick (e.g. 139.677s vs 139.676s) is sub-tick numerical noise, not a real
    latency difference - quantizing before comparing is what keeps that
    noise from acting as a tie-breaker.
    """
    return round(value / tick_s) * tick_s


def pareto_front(points: list[dict]) -> list[int]:
    """Indices of points no other point beats on both cost and p95 wait.

    p95_wait_s is quantized to the tick clock (TICK_S = 0.1s) inside the
    dominance test below, so two points that only differ by sub-tick noise
    (a fraction of a millisecond, far below the 0.1s the simulation can
    actually resolve) are treated as tied on wait rather than one silently
    excluding the other from the front. cost_usd needs no such treatment -
    it has no comparable source of noise. The returned ordering is
    unaffected: it still sorts by the raw (unquantized) cost_usd and
    p95_wait_s, exactly as before.
    """
    front = []
    for i, point in enumerate(points):
        point_wait = _tick_quantized(point["p95_wait_s"])
        dominated = any(
            other["cost_usd"] <= point["cost_usd"]
            and _tick_quantized(other["p95_wait_s"]) <= point_wait
            and (
                other["cost_usd"] < point["cost_usd"]
                or _tick_quantized(other["p95_wait_s"]) < point_wait
            )
            for j, other in enumerate(points)
            if j != i
        )
        if not dominated:
            front.append(i)
    return sorted(front, key=lambda i: (points[i]["cost_usd"], points[i]["p95_wait_s"]))


def build_frontier(
    trace: dict,
    *,
    grid: dict | None = None,
    usd_per_second: float | None = None,
    granularity: float = 1.0,
    bursts_per_hour: float = 1.0,
    horizon_s: float = 3600.0,
) -> dict:
    """The whole frontier.json document.

    `billable_s`/`cost_usd` on each point are totals over `horizon_s`
    seconds, not one burst - see sweep()'s docstring for the floor-plus-
    demand arithmetic behind that.

    Deliberately carries NO wall-clock stamp: it inherits the trace's
    recorded_at so that regenerating from an unchanged trace is
    byte-identical, which is what lets CI detect chart and model rot for free.

    `usd_per_second`, when omitted, inherits the trace's own recorded
    `rate_usd_per_second` (set by burst.py's --rate at measurement time) -
    NOT the $3.99/h list price. A reader who ran burst.py on a negotiated
    rate and then calibrates from that trace should get pricing at the rate
    they actually measured, without having to pass --rate a second time.
    Only a trace carrying no rate at all falls back to USD_PER_SECOND.
    """
    calibration = calibrate(trace)
    rate = (
        usd_per_second
        if usd_per_second is not None
        else trace.get("rate_usd_per_second", USD_PER_SECOND)
    )
    points = sweep(
        calibration,
        grid=grid,
        usd_per_second=rate,
        granularity=granularity,
        bursts_per_hour=bursts_per_hour,
        horizon_s=horizon_s,
    )
    return {
        "schema": FRONTIER_SCHEMA,
        "trace_recorded_at": trace["recorded_at"],
        "machine_type": calibration.machine_type,
        "rate_usd_per_second": rate,
        "billing_granularity_s": granularity,
        "bursts_per_hour": bursts_per_hour,
        "horizon_s": horizon_s,
        "calibration": {
            "cold_start_s": calibration.cold_start_s,
            "service_s": calibration.service_s,
            "observed_runners": calibration.observed_runners,
            "requests": len(calibration.arrivals),
        },
        "points": points,
        "pareto_front": pareto_front(points),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", required=True, metavar="TRACE",
                       help="trace.json from burst.py")
    parser.add_argument("--rate", type=float, default=None,
                       help="USD per second; if omitted, inherits the "
                            "trace's own recorded rate_usd_per_second (set "
                            "by burst.py's --rate when the burst was "
                            f"measured), falling back to {USD_PER_SECOND:.9f} "
                            f"(the ${H100_USD_PER_HOUR}/h H100 list price) "
                            "only if the trace carries no rate at all")
    parser.add_argument("--billing-granularity", type=float, default=1.0,
                       help="seconds each runner's billed time rounds up to (assumption)")
    parser.add_argument("--bursts-per-hour", type=float, default=1.0,
                       help="how often this burst repeats; scales cost_usd "
                            "and billable_s directly (see --horizon-s)")
    parser.add_argument("--horizon-s", type=float, default=3600.0,
                       help="seconds the cost is totalled over (default 3600 = "
                            "1h); a min_concurrency runner is billed for all of "
                            "it, since it is alive around the clock regardless "
                            "of when or how often the burst happens")
    parser.add_argument("--out", default="frontier.json")
    args = parser.parse_args(argv)

    # Readers run this directly, so a bad trace should print one clear line,
    # not a raw traceback. Only these three input-shaped failures are
    # handled here - a missing file, unparseable JSON, or a trace whose
    # schema calibrate() rejects; anything else is unexpected and should
    # surface with its full traceback rather than being swallowed.
    try:
        with open(args.calibrate) as handle:
            trace = json.load(handle)
        document = build_frontier(
            trace,
            usd_per_second=args.rate,
            granularity=args.billing_granularity,
            bursts_per_hour=args.bursts_per_hour,
            horizon_s=args.horizon_s,
        )
    except FileNotFoundError:
        print(f"error: trace file not found: {args.calibrate}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"error: {args.calibrate} is not valid JSON ({exc})", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {args.calibrate}: {exc}", file=sys.stderr)
        return 1

    with open(args.out, "w") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    cheapest = document["points"][document["pareto_front"][0]]
    print(f"wrote {args.out}: {len(document['points'])} points, "
          f"{len(document['pareto_front'])} on the front")
    print(f"  cheapest on front: keep_alive={cheapest['keep_alive']} "
          f"min_concurrency={cheapest['min_concurrency']} "
          f"buffer={cheapest['concurrency_buffer']} "
          f"-> ${cheapest['cost_usd']:.4f} over the {args.horizon_s:.0f}s horizon "
          f"(not one burst), p95 wait {cheapest['p95_wait_s']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
