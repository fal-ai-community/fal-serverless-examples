"""Smoke test - run by the harness against the live ephemeral app."""


def test_generate_echoes_seed_and_returns_image(app_client):
    response = app_client.run(
        "/", {"prompt": "a calico cat riding a bicycle", "seed": 42}
    )
    assert response["seed"] == 42
    assert response["image"]["url"].startswith("https://")
    assert response["image"]["width"] > 0


def test_sync_mode_returns_data_uri(app_client):
    response = app_client.run(
        "/", {"prompt": "a lighthouse at dusk", "sync_mode": True}
    )
    assert response["image"]["url"].startswith("data:image/")
