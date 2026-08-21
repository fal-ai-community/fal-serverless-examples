#!/usr/bin/env python3
"""Run mypy once per example dir - examples share app.py/test_app.py basenames,
so a single invocation across dirs dies on duplicate module names."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def group_key(filename: str) -> str:
    parts = Path(filename).parts
    if len(parts) >= 3 and parts[0][:2].isdigit():
        return "/".join(parts[:2])
    return "."


def main(filenames: list[str]) -> int:
    groups: dict[str, list[str]] = {}
    for filename in filenames:
        groups.setdefault(group_key(filename), []).append(filename)

    failed = False
    for key in sorted(groups):
        result = subprocess.run(
            [sys.executable, "-m", "mypy", *sorted(groups[key])], check=False
        )
        if result.returncode:
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
