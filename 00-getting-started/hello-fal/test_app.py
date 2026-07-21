"""Smoke test - run by the harness against the live ephemeral app.

The harness (harness/measure.py) deploys app.py, then executes this file with
the `app_client` fixture from harness/pytest_plugin.py. Outside the harness
the fixture skips, so plain `pytest` costs nothing.
"""


def test_greet_returns_message(app_client):
    response = app_client.run("/", {"name": "fal"})
    assert response["message"] == "Hello, fal!"
