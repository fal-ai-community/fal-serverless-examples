"""Autoscaler and billing model, reconstructed from fal's published scaling diagrams."""

# Stdlib only and deterministic: a dependency here would have to ship in
# App.requirements, and CI regenerates the committed charts and diffs them.
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from dataclasses import dataclass
from typing import TypedDict

# GPU-H100 list price on fal.ai/pricing; negotiated-tier readers pass --rate.
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
    """Runners the autoscaler wants for `ongoing` (in-flight + queued) requests."""
    buffer = max(
        concurrency_buffer,
        # fal documents the max() rule but not the rounding; ceil is assumed.
        math.ceil(concurrency_buffer_perc / 100 * ongoing),
    )
    want = math.ceil(ongoing / max_multiplexing) + buffer
    want = max(want, min_concurrency)
    if max_concurrency is not None:
        want = min(want, max_concurrency)
    return want


@dataclass(frozen=True)
class RunnerWindow:
    """One runner's billed lifetime; billing starts at SETUP (PENDING is free)."""

    runner_id: str
    setup_start_s: float
    last_completion_s: float
    # A min_concurrency floor runner, alive around the clock: sweep() must
    # bill it via the horizon-wide floor only, or its cost double-counts.
    prewarmed: bool = False


def billable_seconds(
    windows: list[RunnerWindow], *, keep_alive: float, granularity: float = 1.0
) -> float:
    """Billed runner-seconds; the per-runner round-up to `granularity` is assumed, not documented."""
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


def _snap_to_clock(value: float) -> float:
    """Round to the tick clock's precision - unrounded times land a tick late and compound."""
    return round(value, 6)


def arrival_schedule(pattern: str, *, requests: int, duration_s: float) -> list[float]:
    """Deterministic arrival offsets; burst.py fires this exact schedule."""
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


class TimelineRow(TypedDict):
    """One tick of runner states, shared by simulate() and measured_timeline()."""

    t: float
    setup: int
    idle: int
    running: int
    queue_depth: int


def live_runners(row: TimelineRow) -> int:
    return row["setup"] + row["idle"] + row["running"]


@dataclass
class SimResult:
    timeline: list[TimelineRow]
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
    """Fixed-step burst simulation; scale-down is keep_alive expiry only - fal publishes no other dynamics."""
    pending = sorted(arrivals)
    queue: deque[tuple[float, int]] = deque()
    runners: list[dict] = []
    waits: dict[int, float] = {}
    timeline: list[TimelineRow] = []
    next_arrival = 0
    next_request_id = 0

    def start_runner(at: float, *, prewarmed: bool = False) -> dict:
        # Only burst-started runners pay a cold start; the prewarmed floor is READY.
        ready_at = (
            _snap_to_clock(at) if prewarmed else _snap_to_clock(at + cold_start_s)
        )
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
                arrived_at, request_id = queue.popleft()
                waits[request_id] = t - arrived_at
                runner["inflight"].append(_snap_to_clock(t + service_s))
                runner["last_activity"] = max(runner["last_activity"], t)

        in_flight = sum(len(r["inflight"]) for r in alive)
        # A queued request counts toward demand only after aging scaling_delay.
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
                # The prewarmed runner IS the floor - never a scale-down candidate.
                continue
            if (
                not runner["inflight"]
                and runner["ready_at"] <= t
                and (t - runner["last_activity"]) >= keep_alive
                and sum(1 for r in alive if r["terminated_at"] is None) > want
            ):
                runner["terminated_at"] = t

        live = [r for r in runners if r["terminated_at"] is None]
        timeline.append(
            {
                "t": t,
                "setup": sum(1 for r in live if r["ready_at"] > t),
                "running": sum(1 for r in live if r["inflight"]),
                "idle": sum(
                    1 for r in live if r["ready_at"] <= t and not r["inflight"]
                ),
                "queue_depth": len(queue),
            }
        )

        # `not live` alone would never settle: min_concurrency runners never terminate.
        drained = next_arrival >= len(pending) and not queue
        settled = drained and not any(r["inflight"] for r in live)
        if settled and all((t - r["last_activity"]) >= keep_alive for r in live):
            break
    else:  # pragma: no cover - the guard should never trip in practice
        raise RuntimeError(
            "simulation did not settle; check keep_alive/min_concurrency"
        )

    end_t = timeline[-1]["t"]
    windows = []
    for runner in runners:
        last = runner["last_activity"]
        if runner["terminated_at"] is None:
            # billable_seconds re-adds keep_alive; subtract it for still-alive runners.
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
    """What one measured burst yields; scale_config is declared, not measured."""

    cold_start_s: float
    service_s: float
    arrivals: tuple[float, ...]
    observed_runners: int
    scale_config: dict
    machine_type: str


def _ok_requests(trace: dict) -> list[dict]:
    """2xx requests with a runner identity; an empty served_by would fabricate a phantom runner."""
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

    # Service = back-to-back completion gap on one runner; round-trip minus
    # arrival would fold queue waits in.
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
        # Nearest-rank p50 - the one estimator used everywhere here.
        cold_start_s=percentile(cold, 50),
        service_s=min(gaps),
        arrivals=tuple(float(r["arrived_s"]) for r in ok),
        observed_runners=len({r["served_by"] for r in ok}),
        scale_config=dict(trace["scale_config"]),
        machine_type=str(trace["machine_type"]),
    )


def _cold_request(requests: list[dict]) -> dict:
    """The runner's authoritative setup_duration_s: the cold-flagged request, not list order."""
    return next((r for r in requests if r["was_cold"]), requests[0])


def measured_windows(trace: dict) -> list[RunnerWindow]:
    """Per-runner billed window; SETUP start = first completion - setup - service."""
    calibration = calibrate(trace)
    windows = []
    for runner_id, requests in _by_runner(_ok_requests(trace)).items():
        completions = [float(r["completed_s"]) for r in requests]
        setup_duration = float(_cold_request(requests)["setup_duration_s"])
        windows.append(
            RunnerWindow(
                runner_id=runner_id,
                setup_start_s=min(completions) - setup_duration - calibration.service_s,
                last_completion_s=max(completions),
            )
        )
    return windows


def measured_timeline(trace: dict, *, tick_s: float = TICK_S) -> list[TimelineRow]:
    """Reconstruct per-tick runner states from the trace, in the sim's row shape."""
    calibration = calibrate(trace)
    ok = _ok_requests(trace)
    windows = {w.runner_id: w for w in measured_windows(trace)}
    grouped = _by_runner(ok)
    # Each runner's own reported setup duration, not the fleet median.
    ready_at = {
        runner_id: windows[runner_id].setup_start_s
        + float(_cold_request(requests)["setup_duration_s"])
        for runner_id, requests in grouped.items()
    }
    busy: dict[str, list[tuple[float, float]]] = {rid: [] for rid in windows}
    for request in ok:
        runner_id = str(request["served_by"])
        completed = float(request["completed_s"])
        busy[runner_id].append((completed - calibration.service_s, completed))

    start = min(w.setup_start_s for w in windows.values())
    # `end` must include the keep_alive tail, or the billed idle tail is never emitted.
    keep_alive = float(calibration.scale_config.get("keep_alive", 0))
    end = max(w.last_completion_s + keep_alive for w in windows.values())

    timeline: list[TimelineRow] = []
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
            if float(r["arrived_s"])
            <= t
            < float(r["completed_s"]) - calibration.service_s
        )
        timeline.append(
            {
                "t": t,
                "setup": setup,
                "idle": idle,
                "running": running,
                "queue_depth": queued,
            }
        )
    return timeline


FRONTIER_SCHEMA = "fal-serverless-examples/tune-autoscaling/frontier@1"

# Small enough to label on one chart; spans scale-to-zero through always-warm.
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
    """Sweep every knob combination over the measured traffic, costed over the whole `horizon_s`."""
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
                # Prewarmed windows bill only via the floor - never as demand.
                demand_windows = [w for w in result.windows if not w.prewarmed]
                demand_billable = billable_seconds(
                    demand_windows,
                    keep_alive=float(keep_alive),
                    granularity=granularity,
                )
                floor_s = min_concurrency * horizon_s
                demand_s = demand_billable * bursts_in_horizon
                total_s = floor_s + demand_s
                # Price the rounded billable_s so cost_usd == billable_s * rate exactly.
                billable_s = round(total_s, 3)
                cost = cost_usd(billable_s, usd_per_second=rate)
                peak = max(live_runners(row) for row in result.timeline)
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
                        # Beyond the measured envelope is extrapolation, not validation.
                        "validated": peak <= calibration.observed_runners,
                        "measured": (
                            int(keep_alive) == int(base.get("keep_alive", -1))
                            and int(min_concurrency)
                            == int(base.get("min_concurrency", -1))
                            and int(concurrency_buffer)
                            == int(base.get("concurrency_buffer", -1))
                        ),
                    }
                )
    return points


def _tick_quantized(value: float, *, tick_s: float = TICK_S) -> float:
    """Round to the sim's own tick; sub-tick differences are numerical noise, not latency."""
    return round(value / tick_s) * tick_s


def pareto_front(points: list[dict]) -> list[int]:
    """Points no other point beats on both cost and tick-quantized p95 wait."""
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
    """The frontier.json document; carries no wall-clock stamp so CI regeneration is byte-identical."""
    calibration = calibrate(trace)
    # Default to the trace's own recorded rate, not the list price.
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
    parser.add_argument(
        "--calibrate", required=True, metavar="TRACE", help="trace.json from burst.py"
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=None,
        help="USD per second; if omitted, inherits the "
        "trace's own recorded rate_usd_per_second (set "
        "by burst.py's --rate when the burst was "
        f"measured), falling back to {USD_PER_SECOND:.9f} "
        f"(the ${H100_USD_PER_HOUR}/h H100 list price) "
        "only if the trace carries no rate at all",
    )
    parser.add_argument(
        "--billing-granularity",
        type=float,
        default=1.0,
        help="seconds each runner's billed time rounds up to (assumption)",
    )
    parser.add_argument(
        "--bursts-per-hour",
        type=float,
        default=1.0,
        help="how often this burst repeats; scales cost_usd "
        "and billable_s directly (see --horizon-s)",
    )
    parser.add_argument(
        "--horizon-s",
        type=float,
        default=3600.0,
        help="seconds the cost is totalled over (default 3600 = "
        "1h); a min_concurrency runner is billed for all of "
        "it, since it is alive around the clock regardless "
        "of when or how often the burst happens",
    )
    parser.add_argument("--out", default="frontier.json")
    args = parser.parse_args(argv)

    # Only input-shaped failures are caught; anything else keeps its traceback.
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
    print(
        f"wrote {args.out}: {len(document['points'])} points, "
        f"{len(document['pareto_front'])} on the front"
    )
    print(
        f"  cheapest on front: keep_alive={cheapest['keep_alive']} "
        f"min_concurrency={cheapest['min_concurrency']} "
        f"buffer={cheapest['concurrency_buffer']} "
        f"-> ${cheapest['cost_usd']:.4f} over the {args.horizon_s:.0f}s horizon "
        f"(not one burst), p95 wait {cheapest['p95_wait_s']:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
