"""Smoke test (stub) - executed by the harness/CI via ephemeral deploy.

Pattern: fal's AppClient testing flow (see fal docs: "test models and endpoints");
the harness (SERV-1310) provides the deploy/teardown plumbing so tests stay tiny.
"""


def test_smoke_placeholder():
    # Replace with a real call through the harness once SERV-1310 lands:
    # response = app_client.run("/", {"prompt": "a calico cat riding a bicycle"})
    # assert response["result"]
    assert True
