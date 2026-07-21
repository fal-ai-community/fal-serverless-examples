"""Pytest plugin providing the `app_client` fixture for example smoke tests.

The harness (measure.py) deploys the example ephemerally, then runs the
example's test_app.py with this plugin (`-p harness.pytest_plugin`) and
HARNESS_APP_URL pointing at the live app - so tests stay tiny:

    def test_smoke(app_client):
        response = app_client.run("/", {"prompt": "..."})
        assert response["result"]

Run outside the harness (plain `pytest` anywhere inside the repo - the root
conftest.py registers this fixture), it skips instead of failing, so example
tests never cost anything by accident. Running an example's tests from a copy
of the folder outside the repo is not supported.
"""

from __future__ import annotations

import os

import httpx
import pytest

REQUEST_TIMEOUT_S = float(os.environ.get("HARNESS_REQUEST_TIMEOUT", "120"))


class HarnessAppClient:
    """Minimal client for the live ephemeral app under test."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        from fal.sdk import get_credentials

        self._headers = get_credentials().to_headers()

    def run(self, path: str = "/", payload: dict | None = None) -> dict:
        response = httpx.post(
            self.url + path,
            json=payload or {},
            headers=self._headers,
            timeout=REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
        return response.json()


@pytest.fixture
def app_client() -> HarnessAppClient:
    url = os.environ.get("HARNESS_APP_URL")
    if not url:
        pytest.skip("no live app under test - run via harness/measure.py")
    return HarnessAppClient(url)
