"""Measured before/after for the README table - real ephemeral deploys.

Usage (from the repo root, with fal credentials configured):

    HARNESS_STARTUP_TIMEOUT=2700 python 02-scaling-cold-starts/cut-cold-starts/bench.py --variant app_baseline --runs 2
    HARNESS_STARTUP_TIMEOUT=2700 python 02-scaling-cold-starts/cut-cold-starts/bench.py --variant app --runs 3

Each run: ephemeral deploy -> wait healthy (wall clock = "spawn -> healthy")
-> POST /setup-report -> teardown with runner-stop confirmation. Prints one
JSON line per run and a markdown table over all runs. The first optimized run
on a fresh account builds the /data pack + kernel cache (one-time); later
runs are the steady state every cold start gets.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import httpx

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness.check import (  # noqa: E402
    BootWatcher,
    RestRunnerPoller,
    auth_headers,
    load_app_class,
    spawn_owned,
    teardown,
    wait_for_url,
    wait_healthy,
)


def bench_once(app_file: Path) -> dict:
    from fal.app import wrap_app

    app_cls = load_app_class(app_file)
    headers = auth_headers()
    fn = wrap_app(app_cls)
    poller = RestRunnerPoller(headers)
    poller.start()
    started = time.monotonic()
    spawn_info, failure = spawn_owned(fn)
    watcher = BootWatcher(spawn_info)
    watcher.start()
    try:
        url = wait_for_url(spawn_info, failure)
        if watcher.runner_id:
            poller.runner_id = watcher.runner_id
        wait_healthy(url, headers, poller)
        spawn_to_healthy_s = round(time.monotonic() - started, 1)
        response = httpx.post(
            url.rstrip("/") + "/setup-report", json={}, headers=headers, timeout=60
        )
        response.raise_for_status()
        return {"spawn_to_healthy_s": spawn_to_healthy_s, **response.json()}
    finally:
        if poller.runner_id is None and watcher.runner_id:
            poller.runner_id = watcher.runner_id
        leaked = teardown(spawn_info, poller, headers)
        watcher.stop()
        if leaked:
            print(f"ERROR: leaked runner {leaked} - terminate it manually")


def format_table(runs: list[dict]) -> str:
    """Markdown: one column per run, stage rows in first-seen order.

    Stages a run did not execute render as '-' (e.g. steady-state boots skip
    the first-boot pack build) - never invented zeros.
    """
    stage_names: list[str] = []
    for run in runs:
        for stage in run["stages"]:
            if stage["name"] not in stage_names:
                stage_names.append(stage["name"])

    def label(i: int, run: dict) -> str:
        kind = "first boot" if run.get("built_caches_this_boot") else "steady"
        return f"run {i + 1} ({run['variant']}, {kind})"

    header = "| stage | " + " | ".join(label(i, r) for i, r in enumerate(runs)) + " |"
    divider = "|---" * (len(runs) + 1) + "|"
    rows = []
    for name in stage_names:
        cells = []
        for run in runs:
            match = [s["seconds"] for s in run["stages"] if s["name"] == name]
            cells.append(str(match[0]) if match else "-")
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    rows.append(
        "| setup() total | "
        + " | ".join(str(r["total_seconds"]) for r in runs) + " |"
    )
    rows.append(
        "| spawn -> healthy | "
        + " | ".join(str(r["spawn_to_healthy_s"]) for r in runs) + " |"
    )
    return "\n".join([header, divider, *rows])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["app", "app_baseline"], required=True)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    app_file = EXAMPLE_DIR / f"{args.variant}.py"
    runs = []
    for index in range(args.runs):
        print(f"--- {args.variant} run {index + 1}/{args.runs}")
        report = bench_once(app_file)
        print(json.dumps(report))
        runs.append(report)
    print()
    print(format_table(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
