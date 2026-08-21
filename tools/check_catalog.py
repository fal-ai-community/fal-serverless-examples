#!/usr/bin/env python3
"""Fail if an example is missing from llms.txt or its tier README."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    llms = (REPO_ROOT / "llms.txt").read_text()
    errors = []
    for tier_dir in sorted(REPO_ROOT.glob("[0-9][0-9]-*")):
        tier_readme_path = tier_dir / "README.md"
        tier_readme = tier_readme_path.read_text() if tier_readme_path.is_file() else ""
        examples = sorted(
            path for path in tier_dir.iterdir() if (path / "app.py").is_file()
        )
        for example in examples:
            rel = example.relative_to(REPO_ROOT)
            if f"{rel}/README.md" not in llms:
                errors.append(f"{rel}: not listed in llms.txt")
            if example.name not in tier_readme:
                errors.append(f"{rel}: not linked in {tier_dir.name}/README.md")

    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
