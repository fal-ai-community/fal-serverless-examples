"""Smoke test - run by the harness against the live ephemeral app.

The harness (harness/check.py) deploys app.py, then executes this file with
the `app_client` fixture from harness/pytest_plugin.py. Outside the harness
the fixture skips, so plain `pytest` costs nothing.
"""


def test_generate_echoes_seed_and_returns_image(app_client):
    response = app_client.run(
        "/", {"prompt": "a calico cat riding a bicycle", "seed": 42}
    )
    assert response["seed"] == 42  # the seed contract: effective seed is echoed
    assert response["image"]["url"].startswith("https://")
    assert response["image"]["width"] > 0


def test_sync_mode_returns_data_uri(app_client):
    response = app_client.run(
        "/", {"prompt": "a lighthouse at dusk", "sync_mode": True}
    )
    assert response["image"]["url"].startswith("data:image/")
