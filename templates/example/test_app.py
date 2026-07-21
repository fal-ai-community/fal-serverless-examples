"""Smoke test - run by the harness against the live ephemeral app.

The harness (harness/check.py) deploys app.py, then executes this file with
the `app_client` fixture from harness/pytest_plugin.py. Outside the harness
the fixture skips, so plain `pytest` costs nothing. Keep tests tiny -
deploy/teardown plumbing lives in the harness.
"""


def test_smoke(app_client):
    response = app_client.run("/", {"prompt": "a calico cat riding a bicycle"})
    assert response["result"]
