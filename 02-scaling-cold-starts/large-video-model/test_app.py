"""Smoke tests - run by the harness against the live ephemeral app.

'/' is the harness-smoked JSON endpoint; '/stream' (Task 5) is exercised
with a raw SSE client because the harness only speaks request/response.
"""


def test_invoke_contract(app_client):
    response = app_client.run("/", {"prompt": "a red fox trotting through fresh snow at golden hour"})
    assert response["video"]["url"].startswith("https://")
    assert isinstance(response["seed"], int)
    assert response["timings"]["generate_s"] > 0
