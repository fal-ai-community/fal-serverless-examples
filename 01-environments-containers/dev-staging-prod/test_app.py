"""Smoke test - run by the harness against the live ephemeral app.

The harness deploys into the default environment, so this asserts the
contract the app must honor everywhere: it echoes, it reports a stage, and
it never crashes just because a secret is missing.
"""

SAMPLE = "hello from the promotion pipeline"


def test_echoes_and_reports_its_environment(app_client):
    response = app_client.run("/", {"echo": SAMPLE})
    assert response["echo"] == SAMPLE
    # Pinned to app.py's CODE_VERSION: equality proves which revision answered.
    assert response["code_version"] == "1.0.0"
    # Passes with the secret set or missing; what it must catch is the two
    # fields disagreeing (e.g. an empty-string secret).
    assert (response["stage"] == "unset") == (not response["environment_configured"])
