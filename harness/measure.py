#!/usr/bin/env python3
"""Measurement harness - see harness/README.md and SERV-1310.

Ephemeral-deploys each example (AppClient spawn pattern), measures cold start /
p50 / p95 latency / cost, and writes <example>/numbers.json. Numbers are
CI-owned; humans and assistants must never hand-edit them.

Method (full details in harness/README.md):
  cold_start_s      runner boot -> app-ready duration from the platform's own
                    server-side log timestamps ("Runner scheduled for
                    allocation" -> "Application startup complete"), matching
                    the fal docs' definition of cold start (PENDING -> ready)
  p50/p95_latency_s client-measured wall time of N warm POSTs to the endpoint,
                    payload built from the Input model's Field examples
  cost_per_run_usd  effective $/s for the runner's machine type (fetched from
                    the billing rates endpoint at measure time) x p50 latency;
                    null when the rate is unavailable - never invented

Auth: FAL_KEY env var (CI) or a stored `fal auth login` session (local); the
same code path serves both via fal.sdk.get_credentials().

Written against fal==1.78.2 (pinned in harness/requirements.txt) and verified
against live ephemeral deploys. Version-specific facts this file relies on:
  - fal 1.78.2 discards the spawn worker's Future, so a deploy that fails
    before emitting a URL would hang forever; we guard via SpawnInfo._url_ready.
  - Ephemeral runners are INVISIBLE to `fal runners list` / `fal apps runners`
    (gRPC lists cover registered apps only); they ARE visible to REST
    GET /runners/{runner_id}, with the runner id parsed from the platform's
    "Starting runner <uuid>" log line.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import runpy
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

HARNESS_VERSION = "0.1.0"
REPO_ROOT = Path(__file__).resolve().parent.parent

URL_TIMEOUT_S = float(os.environ.get("HARNESS_URL_TIMEOUT", "900"))
STARTUP_TIMEOUT_S = float(os.environ.get("HARNESS_STARTUP_TIMEOUT", "600"))
REQUEST_TIMEOUT_S = float(os.environ.get("HARNESS_REQUEST_TIMEOUT", "120"))
TEST_TIMEOUT_S = float(os.environ.get("HARNESS_TEST_TIMEOUT", "600"))
DEFAULT_WARM_RUNS = int(os.environ.get("HARNESS_WARM_RUNS", "10"))
POLL_INTERVAL_S = 0.3
TEARDOWN_WAIT_S = 120.0

FAILED_RUNNER_STATES = {"CRASH_BACKOFF", "FAILURE_DELAY", "FAILED", "DEAD"}

# Server-side boot markers streamed in the run logs (with server timestamps).
RUNNER_ID_RE = re.compile(r"Starting runner ([0-9a-f-]{36})")
MARKER_SCHEDULED = "Runner scheduled for allocation"
MARKER_APP_READY = "Application startup complete"


class MeasureError(RuntimeError):
    """A per-example measurement failure (deploy, smoke, or metric)."""


def log(msg: str) -> None:
    print(msg, flush=True)


def auth_headers() -> dict[str, str]:
    """One auth path for local (Bearer via `fal auth login`) and CI (FAL_KEY)."""
    from fal.sdk import get_credentials

    return get_credentials().to_headers()


def rest_url() -> str:
    from fal import flags

    return flags.REST_URL


# --------------------------------------------------------------------------
# Example discovery and app loading
# --------------------------------------------------------------------------

def discover_examples() -> list[Path]:
    dirs = []
    for tier_dir in sorted(REPO_ROOT.glob("[0-9][0-9]-*")):
        if not tier_dir.is_dir():
            continue
        for example in sorted(tier_dir.iterdir()):
            if example.is_dir() and (example / "app.py").is_file():
                dirs.append(example)
    return dirs


def load_app_class(app_file: Path):
    """Load the unique fal.App subclass from an example's app.py.

    Mirrors what `fal run <file>` does (fal/utils.py::load_function_from):
    runpy-execute the file, pick the single App subclass, register the
    enclosing package for serialization.
    """
    import fal
    from fal._serialization import include_package_from_path

    example_dir = str(app_file.parent)
    sys.path.insert(0, example_dir)
    try:
        module = runpy.run_path(str(app_file))
    finally:
        # Avoid one example's dir shadowing another's imports in --all runs.
        if example_dir in sys.path:
            sys.path.remove(example_dir)
    include_package_from_path(str(app_file))

    app_classes = list(
        {
            obj
            for obj in module.values()
            if isinstance(obj, type) and issubclass(obj, fal.App) and obj is not fal.App
        }
    )
    if len(app_classes) != 1:
        raise MeasureError(
            f"{app_file}: expected exactly one fal.App subclass, found {len(app_classes)}"
        )
    return app_classes[0]


def pick_endpoint(app_cls) -> tuple[str, object]:
    """Return (path, function) for the endpoint to measure; prefer '/'. """
    endpoints = []
    for _, fn in inspect.getmembers(app_cls, inspect.isfunction):
        sig = getattr(fn, "route_signature", None)
        if sig is None or getattr(sig, "is_websocket", False):
            continue
        endpoints.append((sig.path, fn))
    if not endpoints:
        raise MeasureError(f"{app_cls.__name__}: no HTTP @fal.endpoint found")
    for path, fn in endpoints:
        if path == "/":
            return path, fn
    return sorted(endpoints, key=lambda e: e[0])[0]


def sample_input(endpoint_fn) -> dict:
    """Build the request payload from the Input model's Field examples.

    Convention (harness/README.md): every required Input field carries
    `examples=[...]` - the same sample the README shows. Optional fields fall
    back to their server-side defaults.
    """
    from pydantic import BaseModel

    params = [
        p for name, p in inspect.signature(endpoint_fn).parameters.items()
        if name != "self"
    ]
    if not params:
        return {}
    model = params[0].annotation
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise MeasureError(
            f"endpoint '{endpoint_fn.__name__}': request parameter must be a pydantic model"
        )
    payload = {}
    for field_name, field_info in model.model_fields.items():
        if field_info.examples:
            payload[field_name] = field_info.examples[0]
        elif field_info.is_required():
            raise MeasureError(
                f"Input field '{field_name}' has no examples=[...]; the harness "
                "builds its sample request from Field examples (see harness/README.md)"
            )
    return payload


# --------------------------------------------------------------------------
# Boot observation: log markers (primary) + REST runner polling (state/type)
# --------------------------------------------------------------------------

class BootWatcher(threading.Thread):
    """Drains the spawn log queue, printing every line and recording the
    server-side timestamps of the boot markers plus the runner id."""

    def __init__(self, spawn_info):
        super().__init__(daemon=True, name="boot-watcher")
        self._logs = spawn_info.logs
        self._stop_event = threading.Event()
        self.runner_id: str | None = None
        # marker -> (server_ts: datetime | None, monotonic_arrival: float)
        self.markers: dict[str, tuple[object, float]] = {}

    def run(self) -> None:
        import queue as queue_mod

        while True:
            try:
                entry = self._logs.get(timeout=0.5)
            except queue_mod.Empty:
                if self._stop_event.is_set():
                    return
                continue
            message = entry.message
            log(f"    [app] {message}")
            now = time.monotonic()
            match = RUNNER_ID_RE.search(message)
            if match and self.runner_id is None:
                self.runner_id = match.group(1)
            for marker in (MARKER_SCHEDULED, MARKER_APP_READY):
                if marker in message and marker not in self.markers:
                    self.markers[marker] = (getattr(entry, "timestamp", None), now)

    def stop(self) -> None:
        self._stop_event.set()


class RestRunnerPoller(threading.Thread):
    """Polls REST GET /runners/{runner_id} - the only API that can see an
    ephemeral runner - for state (crash detection) and machine type."""

    def __init__(self, headers: dict[str, str]):
        super().__init__(daemon=True, name="rest-runner-poller")
        self._headers = headers
        self._stop_event = threading.Event()
        self.runner_id: str | None = None  # set by measure() once known
        self.samples: list[tuple[float, str, str]] = []  # (t, state, machine_type)

    def run(self) -> None:
        base = rest_url()
        with httpx.Client(timeout=10) as client:
            while not self._stop_event.is_set():
                if self.runner_id:
                    try:
                        resp = client.get(
                            f"{base}/runners/{self.runner_id}", headers=self._headers
                        )
                        if resp.is_success:
                            data = resp.json()
                            self.samples.append(
                                (
                                    time.monotonic(),
                                    str(data.get("state", "")),
                                    str(data.get("machine_type", "")),
                                )
                            )
                    except Exception:
                        pass  # transient REST errors
                self._stop_event.wait(POLL_INTERVAL_S)

    def stop(self) -> None:
        self._stop_event.set()

    def failed_state_seen(self) -> str | None:
        for _, state, _ in self.samples:
            if state in FAILED_RUNNER_STATES:
                return state
        return None

    def machine_type(self) -> str | None:
        for _, _, mt in reversed(self.samples):
            if mt:
                return mt
        return None


def cold_start_seconds(watcher: BootWatcher, health_ok_at: float) -> tuple[float, str]:
    """Cold start per the fal docs definition (PENDING -> app ready).

    Primary: server-side log timestamps, one clock domain, no polling error.
    Fallback: client-clock arrival times of the same markers / health check.
    """
    scheduled = watcher.markers.get(MARKER_SCHEDULED)
    ready = watcher.markers.get(MARKER_APP_READY)
    if scheduled and ready and scheduled[0] is not None and ready[0] is not None:
        delta = (ready[0] - scheduled[0]).total_seconds()
        if delta >= 0:
            return delta, "server log timestamps (scheduled -> app startup complete)"
    if scheduled:
        return health_ok_at - scheduled[1], "client clock (scheduled log arrival -> healthy)"
    raise MeasureError(
        "could not observe the runner boot ('Runner scheduled for allocation' "
        "marker never arrived in the run logs)"
    )


# --------------------------------------------------------------------------
# Spawn, health, requests
# --------------------------------------------------------------------------

def spawn_owned(fn):
    """Ephemeral-deploy on a harness-owned daemon thread.

    fn.spawn() (fal 1.78.2) submits the deploy to a non-daemon pool and
    discards the Future: a pre-URL failure is silently swallowed, and a stuck
    stream would block interpreter shutdown. Driving the same public
    Host.run(...) call ourselves keeps the worker's exception observable and
    lets the process exit regardless. Auth defaults to PRIVATE (upstream
    default is public - anyone with the URL could invoke and bill during the
    measurement window).
    """
    from fal.api.api import SpawnInfo, SpawnResultHandler

    info = SpawnInfo()
    failure: dict[str, BaseException] = {}

    def worker() -> None:
        try:
            fn.host.run(
                fn.func,
                fn.options,
                args=(),
                kwargs={},
                application_name=fn.app_name,
                application_auth_mode=fn.app_auth or "private",
                result_handler=SpawnResultHandler(info),
                entrypoint=fn.run_entrypoint,
            )
        except BaseException as exc:  # surfaced by wait_for_url
            failure["exc"] = exc

    threading.Thread(target=worker, daemon=True, name="spawn-worker").start()
    return info, failure


def wait_for_url(spawn_info, failure: dict) -> str:
    """Bounded wait for the app URL, surfacing the deploy worker's real
    exception immediately instead of burning the whole timeout."""
    deadline = time.monotonic() + URL_TIMEOUT_S
    ready = getattr(spawn_info, "_url_ready", None)
    while time.monotonic() < deadline:
        if "exc" in failure:
            exc = failure["exc"]
            raise MeasureError(f"deploy failed: {type(exc).__name__}: {exc}")
        if ready is not None and ready.is_set():
            return spawn_info.url
        time.sleep(0.25)
    raise MeasureError(
        f"deploy produced no URL within {URL_TIMEOUT_S:.0f}s - the env "
        "build or scheduling stalled; see app logs above"
    )


def wait_healthy(url: str, headers: dict, poller: RestRunnerPoller) -> float:
    """Poll /health until 2xx; returns the monotonic time of first success."""
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    last_status: object = None
    with httpx.Client(timeout=10) as client:
        while time.monotonic() < deadline:
            failed = poller.failed_state_seen()
            if failed:
                raise MeasureError(
                    f"runner entered {failed} during startup - see app logs"
                )
            try:
                resp = client.get(f"{url.rstrip('/')}/health", headers=headers)
                if resp.is_success:
                    return time.monotonic()
                last_status = resp.status_code
            except httpx.HTTPError as exc:
                last_status = type(exc).__name__
            time.sleep(0.5)
    raise MeasureError(
        f"app not healthy within {STARTUP_TIMEOUT_S:.0f}s (last: {last_status})"
    )


def run_requests(
    url: str, path: str, payload: dict, headers: dict, runs: int
) -> list[float]:
    """One smoke request (validated, excluded from stats) + N timed warm runs."""
    target = url.rstrip("/") + path
    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
        resp = client.post(target, json=payload, headers=headers)
        if resp.status_code in (401, 403):
            headers = auth_headers()  # token may have expired mid-run; retry once
            resp = client.post(target, json=payload, headers=headers)
        if not resp.is_success:
            raise MeasureError(
                f"smoke request failed: HTTP {resp.status_code}: {resp.text[:300]}"
            )
        log(f"    smoke response: {resp.text[:200]}")

        latencies = []
        for i in range(runs):
            t0 = time.monotonic()
            resp = client.post(target, json=payload, headers=headers)
            elapsed = time.monotonic() - t0
            if not resp.is_success:
                raise MeasureError(
                    f"warm request {i + 1}/{runs} failed: HTTP {resp.status_code}"
                )
            latencies.append(elapsed)
    return latencies


def run_example_tests(example_dir: Path, url: str, endpoint_path: str) -> None:
    """Run the example's test_app.py against the live ephemeral app."""
    test_file = example_dir / "test_app.py"
    if not test_file.is_file():
        return
    import subprocess

    env = os.environ.copy()
    env["HARNESS_APP_URL"] = url
    env["HARNESS_ENDPOINT_PATH"] = endpoint_path
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")])
    )
    log("    running test_app.py against the live app")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-q"],
            cwd=REPO_ROOT,
            env=env,
            timeout=TEST_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise MeasureError(f"test_app.py timed out after {TEST_TIMEOUT_S:.0f}s")
    if result.returncode != 0:
        raise MeasureError(f"test_app.py failed (pytest exit {result.returncode})")


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------

def machine_unit_price(machine_type: str, headers: dict) -> float | None:
    """Effective $/s for a machine type from the billing rates endpoint.

    Returns None when the rate is unavailable (endpoint error, unknown type,
    or the 0.0 'unpriced' sentinel) - the caller must then write null, never 0.
    """
    try:
        resp = httpx.get(
            f"{rest_url()}/billing/v2/machine_prices",
            params={"machine_types": machine_type},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        rows = data if isinstance(data, list) else data.get("prices") or data.get("items") or []
        for row in rows:
            if isinstance(row, dict) and row.get("machine_type") == machine_type:
                price = row.get("price", row.get("unit_price"))
                if price:  # 0.0 is the server's "unpriced" sentinel
                    return float(price)
    except Exception as exc:
        log(f"    warning: machine price lookup failed ({exc}); cost will be null")
        return None
    log(f"    warning: no price for machine type '{machine_type}'; cost will be null")
    return None


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(pct / 100 * len(ordered)) - 1)
    return ordered[index]


# --------------------------------------------------------------------------
# Teardown and result writing
# --------------------------------------------------------------------------

def teardown(spawn_info, poller: RestRunnerPoller, headers: dict) -> str | None:
    """Cancel the run stream (that IS the ephemeral teardown), then confirm
    via REST that the runner actually stopped - it bills until it does.

    Returns the runner id if it could not be confirmed stopped (a money leak
    the caller must escalate), else None.
    """
    poller.stop()
    stream = getattr(spawn_info, "stream", None)
    if stream is not None:
        try:
            stream.cancel()
        except Exception as exc:
            log(f"    warning: stream cancel failed ({exc})")

    runner_id = poller.runner_id
    if runner_id is None:
        return None
    base = rest_url()
    deadline = time.monotonic() + TEARDOWN_WAIT_S
    errors = 0
    with httpx.Client(timeout=10) as client:
        while time.monotonic() < deadline:
            try:
                resp = client.get(f"{base}/runners/{runner_id}", headers=headers)
                if resp.is_success:
                    data = resp.json()
                    state = str(data.get("state", ""))
                    if data.get("finished_at") or state in {
                        "TERMINATED", "TERMINATING", "DEAD", "DRAINING",
                    }:
                        log(f"    runner {runner_id} stopped ({state or 'finished'})")
                        return None
                elif resp.status_code == 404:
                    return None  # unknown to the control plane - nothing billing
            except Exception as exc:
                errors += 1
                if errors == 1:
                    log(f"    warning: teardown confirmation errored ({exc}); retrying")
            time.sleep(2)
    log(
        f"    ERROR: runner {runner_id} still alive after {TEARDOWN_WAIT_S:.0f}s "
        "- it keeps billing until terminated; check `fal runners list`"
    )
    return runner_id


def write_numbers(example_dir: Path, record: dict) -> None:
    payload = {
        "machine_type": record["machine_type"],
        "cold_start_s": record["cold_start_s"],
        "p50_latency_s": record["p50_latency_s"],
        "p95_latency_s": record["p95_latency_s"],
        "cost_per_run_usd": record["cost_per_run_usd"],
        "runs": record["runs"],
        "measured_at": record["measured_at"],
        "harness_version": HARNESS_VERSION,
        "_comment": "CI-generated by harness/measure.py (SERV-1310). Never hand-edit.",
    }
    (example_dir / "numbers.json").write_text(json.dumps(payload, indent=2) + "\n")


# --------------------------------------------------------------------------
# Per-example measurement
# --------------------------------------------------------------------------

def measure(
    example_dir: Path, runs: int = DEFAULT_WARM_RUNS, teardown_report: dict | None = None
) -> dict:
    from fal.app import wrap_app

    app_file = example_dir / "app.py"
    if not app_file.is_file():
        raise MeasureError(f"{example_dir}: no app.py")

    app_cls = load_app_class(app_file)
    endpoint_path, endpoint_fn = pick_endpoint(app_cls)
    payload = sample_input(endpoint_fn)
    headers = auth_headers()

    fn = wrap_app(app_cls)
    log(
        f"  deploying ephemeral app (machine_type={app_cls.machine_type}, "
        f"endpoint {endpoint_path}, sample {json.dumps(payload)})"
    )

    poller = RestRunnerPoller(headers)
    poller.start()
    spawn_info, spawn_failure = spawn_owned(fn)
    watcher = BootWatcher(spawn_info)
    watcher.start()
    try:
        url = wait_for_url(spawn_info, spawn_failure)
        log(f"    url: {url}")
        if watcher.runner_id:
            poller.runner_id = watcher.runner_id
        health_ok_at = wait_healthy(url, headers, poller)
        if poller.runner_id is None and watcher.runner_id:
            poller.runner_id = watcher.runner_id
        time.sleep(0.5)  # let the last boot log lines drain
        cold_start_s, cold_method = cold_start_seconds(watcher, health_ok_at)
        log(f"    cold start: {cold_start_s:.2f}s via {cold_method}")

        latencies = run_requests(url, endpoint_path, payload, headers, runs)
        p50 = percentile(latencies, 50)
        p95 = percentile(latencies, 95)
        log(f"    warm runs: {runs}  p50={p50:.3f}s  p95={p95:.3f}s")

        run_example_tests(example_dir, url, endpoint_path)

        machine_type = poller.machine_type() or (
            app_cls.machine_type
            if isinstance(app_cls.machine_type, str)
            else app_cls.machine_type[0]
        )
        unit_price = machine_unit_price(machine_type, headers)
        cost = round(unit_price * p50, 8) if unit_price is not None else None
        if unit_price is not None:
            log(f"    rate: ${unit_price}/s on {machine_type} -> ${cost} per run (p50)")

        return {
            "machine_type": machine_type,
            "cold_start_s": round(cold_start_s, 3),
            "p50_latency_s": round(p50, 3),
            "p95_latency_s": round(p95, 3),
            "cost_per_run_usd": cost,
            "runs": runs,
            "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    finally:
        if poller.runner_id is None and watcher.runner_id:
            poller.runner_id = watcher.runner_id
        leaked = teardown(spawn_info, poller, headers)
        if leaked and teardown_report is not None:
            teardown_report["leaked_runner"] = leaked
        watcher.stop()


def purge_example_modules(example_dir: Path) -> None:
    """Drop modules imported from this example's dir so a later example's
    same-named helper (e.g. utils.py) can't silently receive stale code."""
    prefix = str(example_dir)
    for name in [
        n for n, mod in list(sys.modules.items())
        if (getattr(mod, "__file__", "") or "").startswith(prefix)
    ]:
        del sys.modules[name]


def preflight(headers: dict) -> None:
    """One cheap authenticated call so a bad/revoked credential fails the run
    in seconds with a real error, not one silent URL timeout per example."""
    resp = httpx.get(f"{rest_url()}/users/current", headers=headers, timeout=30)
    if not resp.is_success:
        raise MeasureError(
            f"auth preflight failed: HTTP {resp.status_code}: {resp.text[:200]} "
            "- check FAL_KEY (CI) or `fal auth login` (local)"
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", default="", help="one example dir; empty = all")
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_WARM_RUNS,
        help=f"warm requests per example (default {DEFAULT_WARM_RUNS})",
    )
    args = parser.parse_args()

    if args.example:
        target = Path(args.example)
        if not target.is_absolute():
            target = REPO_ROOT / target
        if not (target / "app.py").is_file():
            log(f"error: {target} is not an example dir (no app.py)")
            return 2
        examples = [target]
    else:
        examples = discover_examples()
        if not examples:
            log("no examples found under the tier directories - nothing to measure")
            return 0

    try:
        preflight(auth_headers())
    except MeasureError as exc:
        log(f"error: {exc}")
        return 1
    except Exception as exc:
        log(
            f"error: credential setup failed: {type(exc).__name__}: {exc} "
            "- check FAL_KEY (CI) or `fal auth login` (local)"
        )
        return 1

    failures = []
    for example_dir in examples:
        rel = example_dir.relative_to(REPO_ROOT) if example_dir.is_relative_to(REPO_ROOT) else example_dir
        log(f"measuring {rel}")
        teardown_report: dict = {}
        try:
            record = measure(example_dir, runs=args.runs, teardown_report=teardown_report)
        except MeasureError as exc:
            log(f"  FAILED: {exc}")
            failures.append((rel, str(exc)))
            continue
        except Exception as exc:  # unexpected - still fail the example, keep going
            log(f"  FAILED (unexpected {type(exc).__name__}): {exc}")
            failures.append((rel, f"{type(exc).__name__}: {exc}"))
            continue
        finally:
            purge_example_modules(example_dir)
        write_numbers(example_dir, record)
        log(f"  wrote {rel / 'numbers.json'}")
        if teardown_report.get("leaked_runner"):
            # Numbers are real and kept, but a billing runner outlived teardown:
            # the run must go red so someone looks at it.
            failures.append(
                (rel, f"runner {teardown_report['leaked_runner']} not confirmed stopped")
            )

    log("")
    log(f"measured {len(examples) - len(failures)}/{len(examples)} example(s)")
    for rel, reason in failures:
        log(f"  FAILED {rel}: {reason}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
