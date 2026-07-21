#!/usr/bin/env python3
"""Measurement harness (stub) - see harness/README.md and SERV-1310.

Contract: deploy the example ephemerally, measure cold start / p50 / p95
latency / cost, write <example>/numbers.json. Numbers are CI-owned; humans
and assistants must never hand-edit them.
"""
import argparse
import sys

NUMBERS_SCHEMA_KEYS = [
    "machine_type",
    "cold_start_s",
    "p50_latency_s",
    "p95_latency_s",
    "cost_per_run_usd",
    "runs",
    "measured_at",
    "harness_version",
]


def measure(example_dir: str) -> dict:
    raise NotImplementedError(
        "Implement per SERV-1310: ephemeral deploy via the fal AppClient "
        "testing pattern, timed cold start + warm requests, cost from "
        "machine-type pricing. See harness/README.md."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", default="", help="one example dir; empty = all")
    args = parser.parse_args()
    print(f"harness stub: nothing measured (example={args.example or 'all'})")
    print("implement measure() per SERV-1310 before enabling CI schedule")
    return 0


if __name__ == "__main__":
    sys.exit(main())
