"""Pytest plugin providing the `app_client` fixture for example smoke tests."""

from __future__ import annotations

import os

import httpx
import pytest

REQUEST_TIMEOUT_S = float(os.environ.get("HARNESS_REQUEST_TIMEOUT", "120"))


class HarnessAppClient:
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
        # skip, never fail: plain `pytest` must cost nothing by accident
        pytest.skip("no live app under test - run via harness/check.py")
    return HarnessAppClient(url)
