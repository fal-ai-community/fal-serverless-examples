#!/usr/bin/env python3
"""Measure baseline vs optimized cold starts with real ephemeral deploys.

For each app this script spawns a real ephemeral deploy (same machinery as
the CI harness), records the runner's state timeline (DOCKER_PULL, SETUP, ...)
from the platform's REST API, waits for health, fires the first inference
request, pulls the app's own per-stage setup report, tears the runner down,
and finally prints a before/after table. Every number comes from a live run -
nothing is estimated or hardcoded.

Run from the repo root venv (harness/requirements.txt + `fal auth login`):

    python 02-scaling-cold-starts/cut-cold-starts/bench.py

By default each app runs twice. Run 1 of the optimized app is the account's
first boot: it builds the /data FlashPack pack and populates the shared
inductor cache (a one-time cost); run 2 is the steady-state cold start every
later worker enjoys. Run 1 of the baseline may also pay one-time environment
build; run 2 is its honest steady state. Compare run 2 against run 2.

Ephemeral deploys bill real GPU time (~2-15 min per run on GPU-H100).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import httpx

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from harness.check import (  # noqa: E402
    POLL_INTERVAL_S,
    REQUEST_TIMEOUT_S,
    BootWatcher,
    CheckError,
    auth_headers,
    load_app_class,
    pick_endpoint,
    preflight,
    rest_url,
    sample_input,
    spawn_owned,
    teardown,
    wait_for_url,
    wait_healthy,
)

FIRST_REQUEST_TIMEOUT_S = 300.0


class StateTimelinePoller(threading.Thread):
    """RestRunnerPoller's interface (runner_id / stop / failed_state_seen),
    plus a timestamped timeline of every runner-state transition it observes.

    States occurring before the runner id appears in the logs (early
    DOCKER_PULL, scheduling) are not observable from outside - the table
    reports whatever the platform exposed.
    """

    FAILED_STATES = {"CRASH_BACKOFF", "FAILURE_DELAY", "FAILED", "DEAD"}

    def __init__(self, headers: dict[str, str]):
        super().__init__(daemon=True, name="state-timeline-poller")
        self._headers = headers
        self._stop_event = threading.Event()
        self.runner_id: str | None = None
        self.last_state: str = ""
        self._failed: str | None = None
        self.timeline: list[tuple[float, str]] = []  # (monotonic time, new state)

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
                            state = str(resp.json().get("state", ""))
                            if state and state != self.last_state:
                                self.timeline.append((time.monotonic(), state))
                                self.last_state = state
                            if state in self.FAILED_STATES and self._failed is None:
                                self._failed = state
                    except Exception:
                        pass  # transient REST errors
                self._stop_event.wait(POLL_INTERVAL_S)

    def stop(self) -> None:
        self._stop_event.set()

    def failed_state_seen(self) -> str | None:
        return self._failed

    def state_durations(self, until: float) -> list[tuple[str, float]]:
        """Observed (state, seconds) pairs, cut off at `until` (health time)."""
        durations = []
        for i, (t, state) in enumerate(self.timeline):
            t_end = self.timeline[i + 1][0] if i + 1 < len(self.timeline) else until
            durations.append((state, max(0.0, min(t_end, until) - t)))
        return durations


def bench_once(app_file: Path, label: str) -> dict:
    """One ephemeral deploy of `app_file`; returns every measured number."""
    from fal.app import wrap_app

    app_cls = load_app_class(app_file)
    endpoint_path, endpoint_fn = pick_endpoint(app_cls)
    payload = sample_input(endpoint_fn)
    headers = auth_headers()
    fn = wrap_app(app_cls)

    print(f"[{label}] deploying ephemeral app ({app_cls.__name__}, {app_cls.machine_type})")
    poller = StateTimelinePoller(headers)
    poller.start()
    t_spawn = time.monotonic()
    spawn_info, spawn_failure = spawn_owned(fn)
    watcher = BootWatcher(spawn_info)
    watcher.start()

    # Hand the runner id to the state poller the moment it shows up in the
    # boot logs - the URL only appears when the app is READY, which would be
    # far too late to observe DOCKER_PULL/SETUP.
    def propagate_runner_id() -> None:
        while poller.runner_id is None and not poller._stop_event.is_set():
            if watcher.runner_id:
                poller.runner_id = watcher.runner_id
                return
            time.sleep(0.2)

    threading.Thread(target=propagate_runner_id, daemon=True).start()

    result: dict = {"label": label, "app_file": app_file.name}
    try:
        url = wait_for_url(spawn_info, spawn_failure)
        result["url_after_s"] = round(time.monotonic() - t_spawn, 1)
        print(f"[{label}] url after {result['url_after_s']}s: {url}")
        wait_healthy(url, headers, poller)
        t_healthy = time.monotonic()
        if poller.runner_id is None and watcher.runner_id:
            poller.runner_id = watcher.runner_id
        result["spawn_to_healthy_s"] = round(t_healthy - t_spawn, 1)
        result["runner_states"] = [
            (state, round(seconds, 1))
            for state, seconds in poller.state_durations(until=t_healthy)
        ]

        t0 = time.monotonic()
        resp = httpx.post(
            url.rstrip("/") + endpoint_path,
            json=payload,
            headers=headers,
            timeout=FIRST_REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
        result["first_request_s"] = round(time.monotonic() - t0, 2)
        result["generation_ms"] = resp.json().get("generation_ms")

        report = httpx.post(
            url.rstrip("/") + "/setup-report",
            json={},
            headers=headers,
            timeout=REQUEST_TIMEOUT_S,
        )
        report.raise_for_status()
        result["setup_report"] = report.json()
    finally:
        if poller.runner_id is None and watcher.runner_id:
            poller.runner_id = watcher.runner_id
        leaked = teardown(spawn_info, poller, headers)
        watcher.stop()
        if leaked:
            print(
                f"[{label}] ERROR: runner {leaked} not confirmed stopped - "
                "it bills until terminated; check `fal runners list`"
            )
            result["leaked_runner"] = leaked

    report = result["setup_report"]
    print(
        f"[{label}] healthy in {result['spawn_to_healthy_s']}s "
        f"(in-app setup {report['total_seconds']}s on {report['gpu']}); "
        f"first request {result['first_request_s']}s"
    )
    return result


def stage_seconds(report: dict, prefix: str) -> str:
    for entry in report["stages"]:
        if entry["name"].startswith(prefix):
            return f"{entry['seconds']:.1f}"
    return "-"


def print_table(runs: list[dict]) -> None:
    print("\n## Raw runs (JSON)\n")
    print(json.dumps(runs, indent=2))

    print("\n## Before/after (README-ready)\n")
    headers = ["Setup stage"] + [r["label"] for r in runs]
    rows = [
        ("imports", "imports"),
        ("download weights", "download"),
        ("weights -> GPU", "load weights"),
        ("compile + warmup", "compile + warmup"),
        ("pack to FlashPack (one-time)", "pack weights"),
        ("publish pack to /data (one-time)", "publish pack"),
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for title, prefix in rows:
        cells = [stage_seconds(r["setup_report"], prefix) for r in runs]
        lines.append(f"| {title} | " + " | ".join(f"{c}s" if c != "-" else c for c in cells) + " |")
    lines.append(
        "| **setup() total** | "
        + " | ".join(f"**{r['setup_report']['total_seconds']:.1f}s**" for r in runs)
        + " |"
    )
    lines.append(
        "| spawn -> healthy (incl. scheduling) | "
        + " | ".join(f"{r['spawn_to_healthy_s']}s" for r in runs)
        + " |"
    )
    lines.append(
        "| first request after boot | "
        + " | ".join(f"{r['first_request_s']}s" for r in runs)
        + " |"
    )
    print("\n".join(lines))

    print("\nObserved runner states (from REST /runners/{id}; early states may be unobservable):")
    for r in runs:
        states = ", ".join(f"{s} {d}s" for s, d in r["runner_states"]) or "(none observed)"
        print(f"  {r['label']}: {states}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=2, help="deploys per app (default 2)")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-optimized", action="store_true")
    args = parser.parse_args()

    preflight(auth_headers())

    plan: list[tuple[Path, str]] = []
    for skip, file_name, name in (
        (args.skip_baseline, "baseline_app.py", "baseline"),
        (args.skip_optimized, "app.py", "optimized"),
    ):
        if not skip:
            plan += [
                (EXAMPLE_DIR / file_name, f"{name} run {i + 1}") for i in range(args.runs)
            ]

    runs = []
    for app_file, label in plan:
        try:
            runs.append(bench_once(app_file, label))
        except (CheckError, httpx.HTTPError) as exc:
            print(f"[{label}] FAILED: {exc}")
            return 1

    print_table(runs)
    return 1 if any(r.get("leaked_runner") for r in runs) else 0


if __name__ == "__main__":
    sys.exit(main())
