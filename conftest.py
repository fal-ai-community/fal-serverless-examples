"""Repo-root pytest config: exposes the harness `app_client` fixture to all examples."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from harness.pytest_plugin import app_client  # noqa: F401
