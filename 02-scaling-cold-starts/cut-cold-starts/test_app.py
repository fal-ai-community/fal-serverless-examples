"""Smoke test - run by the harness against the live ephemeral app.

The harness (harness/check.py) deploys app.py (the optimized variant), then
executes this file with the `app_client` fixture from
harness/pytest_plugin.py. Outside the harness the fixture skips, so plain
`pytest` costs nothing.
"""


def test_generate_echoes_seed_and_returns_image(app_client):
    response = app_client.run(
        "/", {"prompt": "a calico cat riding a bicycle, film photo", "seed": 42}
    )
    assert response["seed"] == 42  # the seed contract: effective seed is echoed
    assert response["image"]["url"].startswith("https://")
    assert response["generation_ms"] > 0


def test_setup_report_shows_all_three_levers(app_client):
    report = app_client.run("/setup-report", {})
    assert report["variant"] == "optimized"
    assert report["total_seconds"] > 0
    stage_names = [stage["name"] for stage in report["stages"]]
    assert any(name.startswith("load weights (FlashPack") for name in stage_names)
    assert any(name.startswith("compile + warmup") for name in stage_names)
