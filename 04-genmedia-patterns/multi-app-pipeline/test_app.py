"""Free unit tests plus one live harness test for the multi-app pipeline."""

import importlib.util
from pathlib import Path

import httpx
import pytest

HERE = Path(__file__).parent


def _load(name: str):
    # --import-mode=importlib (duplicate test_app.py basenames): load by path.
    spec = importlib.util.spec_from_file_location(
        f"multi_app_pipeline_{name}", HERE / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent = _load("app")
llm = _load("stage_llm")
dit = _load("stage_dit")
sr = _load("stage_sr")
STAGES = {"llm": llm.PipelineLLM, "dit": dit.PipelineDiT, "sr": sr.PipelineSR}


def test_agent_targets_the_stage_app_names():
    assert {k: cls.app_name for k, cls in STAGES.items()} == agent.STAGE_APPS


def test_every_app_is_private_and_pinned():
    for cls in [agent.VideoPipelineAgent, *STAGES.values()]:
        assert cls.app_auth == "private"
        assert cls.requirements, cls
        for spec in cls.requirements:
            assert "==" in spec, f"{cls.__name__}: unpinned {spec}"


def test_agent_runs_on_cpu_and_stages_on_gpus():
    assert agent.VideoPipelineAgent.machine_type == "S"
    for cls in STAGES.values():
        assert str(cls.machine_type).startswith("GPU-")


def test_stage_owner_prefers_nickname(monkeypatch):
    monkeypatch.setenv("FAL_USER_NICKNAME", "alice")
    monkeypatch.setenv("FAL_APP_NAME", "bob/agent")
    assert agent.stage_owner() == "alice"
    monkeypatch.delenv("FAL_USER_NICKNAME")
    assert agent.stage_owner() == "bob"
    monkeypatch.delenv("FAL_APP_NAME")
    with pytest.raises(RuntimeError):
        agent.stage_owner()


def test_timing_overhead_is_wall_minus_compute():
    t = agent.timing(10.0, {"compute_seconds": 7.5})
    assert (t.wall_seconds, t.compute_seconds, t.overhead_seconds) == (10.0, 7.5, 2.5)


def test_parse_plan_reads_json_and_falls_back():
    p, n = llm.parse_plan('Sure! {"prompt": "a cat", "negative_prompt": "blur"}')
    assert (p, n) == ("a cat", "blur")
    p, n = llm.parse_plan("just a sentence")
    assert (p, n) == ("just a sentence", llm.DEFAULT_NEGATIVE)


def test_agent_and_dit_reject_off_grid_frames():
    # The agent must reject before it spends an LLM call on a doomed request.
    for model in (agent.Input, dit.Input):
        with pytest.raises(ValueError):
            model(idea="x", prompt="x", num_frames=34)


def test_stage_422_passes_through_and_everything_else_is_502():
    class StageError(Exception):
        def __init__(self, status_code):
            self.status_code = status_code

    assert agent.stage_error_status(StageError(422)) == 422
    # A missing stage app (404) is the operator's problem, not the caller's.
    assert agent.stage_error_status(StageError(404)) == 502
    assert agent.stage_error_status(StageError(500)) == 502
    assert agent.stage_error_status(RuntimeError("network")) == 502


def test_agent_timeout_covers_every_stage():
    cls = agent.VideoPipelineAgent
    assert cls.request_timeout >= len(agent.STAGE_APPS) * agent.STAGE_TIMEOUT_S
    assert cls.skip_retry_conditions == ["timeout"]


def test_sr_rejects_oversized_input_before_decoding():
    from fal.exceptions import FieldException

    sr.check_source((1920, 1080), sr.MAX_FRAMES)
    with pytest.raises(FieldException):
        sr.check_source((3840, 2160), 1)
    with pytest.raises(FieldException):
        sr.check_source((832, 480), sr.MAX_FRAMES + 1)


# Live: only under harness/check.py, and only after the three stages are deployed.
def test_pipeline_end_to_end(app_client):
    response = app_client.run(
        "/", {"idea": "a red panda napping on a branch", "num_frames": 17, "seed": 7}
    )
    assert response["seed"] == 7
    assert response["prompt"]
    assert set(response["timings"]) == {"llm", "dit", "sr"}
    for t in response["timings"].values():
        assert t["compute_seconds"] > 0
    assert response["video"]["url"] != response["base_video"]["url"]
    head = httpx.get(
        response["video"]["url"], timeout=60, headers={"Range": "bytes=0-63"}
    ).content
    assert b"ftyp" in head[:16]
