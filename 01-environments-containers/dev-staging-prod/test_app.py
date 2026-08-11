"""Smoke test - run by the harness against the live ephemeral app.

The harness deploys into the default environment, so this asserts the
contract the app must honor everywhere: it echoes, it reports a stage, and
it never crashes just because a secret is missing.
"""

SAMPLE = "hello from the promotion pipeline"


def test_echoes_and_reports_its_environment(app_client):
    response = app_client.run("/", {"echo": SAMPLE})
    assert response["echo"] == SAMPLE
    assert response["code_version"]
    # A missing secret must degrade to "unset", not to a 500 - an app that
    # only boots in the environment you remembered to configure is a
    # promotion hazard.
    assert response["stage"] == "unset" or response["environment_configured"]
