"""Repo-root pytest config.

Exposes the harness `app_client` fixture (harness/pytest_plugin.py) to every
example's test_app.py regardless of where pytest is invoked from inside the
repo. Outside a harness run the fixture skips, so plain `pytest` costs
nothing. pytest imports this file by path, so no addopts/sys.path setup is
required from the caller.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from harness.pytest_plugin import app_client  # noqa: E402,F401
