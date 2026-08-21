#!/usr/bin/env python3
"""Static per-example checks - no secrets, no deploys, safe for fork PRs."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_FILES = ("app.py", "README.md", "requirements.txt", "test_app.py")


def check_example(example_dir: Path) -> list[str]:
    from harness.check import (
        discover_variants,
        load_app_class,
        pick_endpoint,
        sample_input,
    )

    errors = []
    for name in REQUIRED_FILES:
        if not (example_dir / name).is_file():
            errors.append(f"missing {name}")
    if errors:
        return errors

    pinned = [
        line.strip()
        for line in (example_dir / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    for app_file in discover_variants(example_dir):
        where = "" if app_file.name == "app.py" else f"{app_file.name}: "
        try:
            app_cls = load_app_class(app_file)
        except Exception as exc:
            errors.append(f"{where}failed to load: {exc}")
            continue

        if not getattr(app_cls, "app_auth", None):
            errors.append(
                f"{where}app_auth must be set explicitly (deploys fail without it)"
            )

        try:
            _, endpoint_fn = pick_endpoint(app_cls)
            sample_input(endpoint_fn)
        except Exception as exc:
            errors.append(f"{where}{exc}")

        # image-based variants declare no requirements - nothing to mirror
        if app_cls.requirements and pinned != app_cls.requirements:
            errors.append(
                f"{where}requirements.txt does not mirror App.requirements exactly "
                f"(file: {pinned} vs app: {app_cls.requirements})"
            )
        for spec in app_cls.requirements:
            if "==" not in spec:
                errors.append(f"{where}unpinned requirement: {spec!r}")
    return errors


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    from harness.check import discover_examples

    examples = discover_examples()
    if not examples:
        print("no examples found - nothing to check")
        return 0

    failed = False
    for example_dir in examples:
        rel = example_dir.relative_to(REPO_ROOT)
        errors = check_example(example_dir)
        if errors:
            failed = True
            print(f"FAIL {rel}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"ok   {rel}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
