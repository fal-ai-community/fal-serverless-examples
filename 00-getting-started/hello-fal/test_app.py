"""Smoke test - run by the harness against the live ephemeral app."""


def test_greet_returns_message(app_client):
    response = app_client.run("/", {"name": "fal"})
    assert response["message"] == "Hello, fal!"
