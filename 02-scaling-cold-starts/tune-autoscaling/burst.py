#!/usr/bin/env python3
"""Fire one measured burst at a deployed tune-autoscaling app (SERV-1294).

Stdlib plus fal (already a declared dependency of this example). No httpx:
requirements.txt must mirror App.requirements exactly, so anything imported
here that is not in that list would have to ship to the GPU runner.

What it records, per request: when it was sent, when it came back, and the
app's own report of which runner served it, whether that runner was cold,
and how long that runner's setup() took. That is enough to reconstruct the
runner ramp and the billed windows with no metrics API and no ADMIN key.

Retries are OFF on purpose. A retry would fold two queue waits into one
observation and quietly corrupt the latency distribution.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model  # noqa: E402  - sibling module, stdlib-only

PINNED_MACHINE = "GPU-H100"
SYNC_URL = "https://fal.run/{app}"
ANALYTICS_URL = "https://api.fal.ai/v1/serverless/analytics"

SCALE_KEYS = (
    "keep_alive", "min_concurrency", "max_concurrency", "concurrency_buffer",
    "concurrency_buffer_perc", "scaling_delay", "max_multiplexing",
)


def select_app_config(apps_document: dict, app: str) -> dict:
    """Find one app's live scaling config in `fal apps list --json` output.

    Read live rather than trusted from app.py: keep_alive, min/max
    concurrency, both buffers, scaling_delay and request_timeout are
    INHERITED across deploys, so the code may not be what is running.
    """
    alias = app.split("/")[-1]
    for entry in apps_document.get("apps", []):
        if entry.get("alias") == alias:
            return entry
    known = ", ".join(sorted(e.get("alias", "?") for e in apps_document.get("apps", [])))
    raise ValueError(f"{app!r} is not deployed (aliases seen: {known or 'none'})")


def assert_machine(config: dict, expected: str = PINNED_MACHINE) -> None:
    """Refuse to price a burst we cannot attribute to one machine type."""
    machines = list(config.get("machine_types") or [])
    if machines != [expected]:
        raise RuntimeError(
            f"expected machine_types == [{expected!r}], got {machines!r}. "
            f"The cost math is priced for {expected}; a fallback list means the "
            f"machine actually used is not knowable from here. Redeploy with "
            f"machine_type = {expected!r}, or pass --rate for the machine you got."
        )


def read_live_config(app: str) -> dict:
    try:
        completed = subprocess.run(
            ["fal", "apps", "list", "--json"],
            capture_output=True, text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "the `fal` CLI was not found on PATH. It comes from this "
            "example's requirements.txt - install it (or activate the "
            f"venv that has it) before running a burst. ({error})"
        ) from error
    if completed.returncode != 0:
        raise RuntimeError(
            f"`fal apps list --json` exited {completed.returncode}: "
            f"{completed.stderr.strip() or '(no stderr captured)'}"
        )
    try:
        apps_document = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        snippet = completed.stdout[:200]
        raise RuntimeError(
            f"`fal apps list --json` did not print JSON ({error}); "
            f"stdout started with: {snippet!r}"
        ) from error
    return select_app_config(apps_document, app)


def _auth_headers() -> dict[str, str]:
    """One auth path for FAL_KEY and for a stored `fal auth login` session."""
    from fal.sdk import get_credentials

    return get_credentials().to_headers()


def fire(url: str, payload: dict, headers: dict, timeout: float) -> tuple[int, dict]:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, headers={**headers, "Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, {"error": error.read()[:200].decode("utf-8", "replace")}
    except Exception as error:  # network/timeout - recorded, never retried
        return 0, {"error": f"{type(error).__name__}: {error}"}


def build_trace(
    *,
    app: str,
    config: dict,
    pattern: str,
    requests: int,
    duration_s: float,
    concurrency: int,
    records: list[dict],
    usd_per_second: float,
    recorded_at: str,
) -> dict:
    """Assemble trace@1. model.calibrate is the only reader of this shape."""
    import fal

    return {
        "schema": model.TRACE_SCHEMA,
        "recorded_at": recorded_at,
        "app": app,
        "machine_type": (list(config.get("machine_types") or [None]))[0],
        "fal_version": fal.__version__,
        "rate_usd_per_second": usd_per_second,
        "scale_config": {key: config.get(key) for key in SCALE_KEYS},
        "burst": {
            "pattern": pattern,
            "requests": requests,
            "duration_s": duration_s,
            "concurrency": concurrency,
        },
        "requests": records,
    }


def cross_check_analytics(app: str, headers: dict) -> dict:
    """Optional: compare derived billable seconds against fal's own figure.

    Never required to produce a chart, and must never be ABLE to raise: a
    completed burst has already cost real GPU time and every record still
    lives only in memory when this runs, so a DNS failure, connection
    error, or timeout here (a bare URLError/socket.timeout - not a subclass
    of HTTPError) would otherwise destroy that measurement. Every failure
    mode, HTTPError included, comes back as the same reported-failure shape.

    /v1/serverless/analytics needs a Key-format credential; a stored
    browser session gets 403, which is reported rather than raised.
    """
    url = f"{ANALYTICS_URL}?app={app}&timeframe=minute"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return {"ok": True, "analytics": json.loads(response.read())}
    except urllib.error.HTTPError as error:
        return {
            "ok": False,
            "status": error.code,
            "hint": (
                "403 means this credential lacks the scope; the derived figure "
                "stands on its own. Use a Key-format API key to cross-check."
            ),
        }
    except Exception as error:  # noqa: BLE001 - deliberately catch-all, see above
        return {
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
            "hint": (
                "the cross-check could not reach the analytics endpoint "
                "(network error, DNS, or timeout); the derived figure "
                "stands on its own."
            ),
        }


def run_burst(app: str, arrivals: list[float], payload: dict, concurrency: int,
              timeout: float) -> list[dict]:
    url = SYNC_URL.format(app=app)
    headers = _auth_headers()
    origin = time.monotonic()
    records: list[dict] = []

    def one(index: int, offset: float) -> dict:
        delay = offset - (time.monotonic() - origin)
        if delay > 0:
            time.sleep(delay)
        arrived = time.monotonic() - origin
        status, body = fire(url, payload, headers, timeout)
        completed = time.monotonic() - origin
        return {
            "id": index,
            "arrived_s": round(arrived, 3),
            "completed_s": round(completed, 3),
            "status": status,
            "served_by": body.get("served_by", ""),
            "was_cold": bool(body.get("was_cold", False)),
            "setup_duration_s": float(body.get("setup_duration_s", 0.0)),
        }

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one, i, t) for i, t in enumerate(arrivals)]
        for future in futures:
            records.append(future.result())
    return sorted(records, key=lambda r: r["id"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True, help="e.g. yourname/tune-autoscaling")
    parser.add_argument("--pattern", default="spike", choices=["spike", "ramp", "steady"])
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--duration", type=float, default=0.0,
                       help="seconds the arrivals are spread over (ignored for spike)")
    parser.add_argument("--concurrency", type=int, default=20,
                       help="client-side worker pool size; keep >= requests "
                            "for any pattern whose arrival schedule you want "
                            "honoured - with fewer workers than requests, a "
                            "task only starts waiting out its arrival offset "
                            "once a worker frees up, so ramp and steady "
                            "arrivals fire late and silently stop matching "
                            "the pattern you asked for (not just spikes)")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--rate", type=float, default=model.USD_PER_SECOND)
    parser.add_argument("--prompt",
                       default="a calico cat riding a bicycle through a sunlit "
                               "wheat field, film photo")
    parser.add_argument("--steps", type=int, default=4,
                       help="fixed across the burst so service time is comparable")
    parser.add_argument("--from-analytics", action="store_true",
                       help="also cross-check against /v1/serverless/analytics")
    parser.add_argument("--out", default="trace.json")
    args = parser.parse_args(argv)

    config = read_live_config(args.app)
    assert_machine(config)
    print(f"live scale config: {json.dumps({k: config.get(k) for k in SCALE_KEYS})}")

    arrivals = model.arrival_schedule(
        args.pattern, requests=args.requests, duration_s=args.duration
    )
    print(f"firing {args.requests} requests ({args.pattern}) at {args.app}")
    records = run_burst(
        args.app,
        list(arrivals),
        {"prompt": args.prompt, "num_inference_steps": args.steps},
        args.concurrency,
        args.timeout,
    )

    failed = [r for r in records if not (200 <= r["status"] < 300)]
    trace = build_trace(
        app=args.app, config=config, pattern=args.pattern, requests=args.requests,
        duration_s=args.duration, concurrency=args.concurrency, records=records,
        usd_per_second=args.rate,
        recorded_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    def write_trace() -> None:
        with open(args.out, "w") as handle:
            json.dump(trace, handle, indent=2, sort_keys=True)
            handle.write("\n")

    # Deliberate ordering: the trace is written to disk BEFORE the optional
    # cross-check runs. The burst above is expensive and irreplaceable - every
    # recorded request lives only in memory until this write happens - while
    # the cross-check is neither: it is optional and (per its docstring)
    # cannot raise. Running it first would risk the entire measurement for a
    # feature whose own purpose is to be skippable.
    write_trace()

    if args.from_analytics:
        trace["analytics_cross_check"] = cross_check_analytics(args.app, _auth_headers())
        write_trace()

    print(f"wrote {args.out}: {len(records)} requests, {len(failed)} failed")

    try:
        calibration = model.calibrate(trace)
    except ValueError as error:
        print(
            f'error: could not calibrate {args.out}: "{error}". The trace is '
            f"saved and can be re-analysed once the cause is addressed.",
            file=sys.stderr,
        )
        return 1

    billable = model.billable_seconds(
        model.measured_windows(trace), keep_alive=float(config.get("keep_alive") or 0)
    )
    print(f"  observed {calibration.observed_runners} runner(s), "
          f"cold start {calibration.cold_start_s:.1f}s, "
          f"service {calibration.service_s:.1f}s")
    print(f"  billable {billable:.0f} runner-seconds "
          f"-> ${model.cost_usd(billable, usd_per_second=args.rate):.4f} "
          f"at ${args.rate * 3600:.2f}/h")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
