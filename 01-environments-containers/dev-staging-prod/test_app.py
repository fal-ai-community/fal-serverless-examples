"""Harness smoke test for the promotion demo's environment contract."""

SAMPLE = "hello from the promotion pipeline"


def test_echoes_and_reports_its_environment(app_client):
    response = app_client.run("/", {"echo": SAMPLE})
    assert response["echo"] == SAMPLE
    # Keep in sync with app.py's CODE_VERSION.
    assert response["code_version"] == "1.0.0"
    # Must pass with the secret set or missing; only the two fields disagreeing fails.
    assert (response["stage"] == "unset") == (not response["environment_configured"])
