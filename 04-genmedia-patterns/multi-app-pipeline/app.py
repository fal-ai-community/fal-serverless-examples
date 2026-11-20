"""CPU agent app chaining private LLM, DiT, and SR apps, each on its own machine."""

import os
import time
from typing import Any

import fal
from fal.toolkit import File
from fastapi import HTTPException
from pydantic import BaseModel, Field

# Must match the `name=` of each stage class in stage_*.py.
STAGE_APPS = {
    "llm": "video-pipeline-llm",
    "dit": "video-pipeline-dit",
    "sr": "video-pipeline-sr",
}
# Covers a cold DiT boot (weights + warmup) plus generation.
STAGE_TIMEOUT_S = 1800


class Input(BaseModel):
    idea: str = Field(
        description="A short idea for a video; the LLM stage expands it into a prompt.",
        examples=["a red panda napping on a branch"],
    )
    num_frames: int = Field(
        default=33,
        ge=17,
        le=81,
        description="Frames at 16 fps; must be 4k+1 (17, 21, ... 81).",
    )
    num_inference_steps: int = Field(
        default=20, ge=1, le=50, description="DiT denoising steps."
    )
    upscale: bool = Field(
        default=True, description="Run the SR stage (2x) on the generated clip."
    )
    seed: int | None = Field(
        default=None, description="DiT seed; random when omitted, echoed back."
    )


class StageTiming(BaseModel):
    wall_seconds: float = Field(description="Seen from the agent: queue + hop + work.")
    compute_seconds: float = Field(description="Reported by the stage itself.")
    overhead_seconds: float = Field(
        description="wall - compute: queueing, cold start, and network hops."
    )


class Output(BaseModel):
    video: File = Field(description="The final clip (upscaled if `upscale`).")
    base_video: File = Field(description="The DiT output before super-resolution.")
    prompt: str = Field(description="The prompt the LLM stage wrote.")
    negative_prompt: str = Field(description="The negative prompt the LLM wrote.")
    seed: int = Field(description="The DiT seed.")
    timings: dict[str, StageTiming] = Field(description="Per-stage timings.")


def stage_owner() -> str:
    """The account this agent runs under; stages are private apps on the same one."""
    owner = os.environ.get("FAL_USER_NICKNAME")
    if not owner:
        owner = os.environ.get("FAL_APP_NAME", "").partition("/")[0]
    if not owner:
        raise RuntimeError("cannot resolve the app owner from the runner env")
    return owner


def timing(wall: float, result: dict[str, Any]) -> StageTiming:
    compute = float(result.get("compute_seconds", 0.0))
    return StageTiming(
        wall_seconds=round(wall, 2),
        compute_seconds=round(compute, 2),
        overhead_seconds=round(max(wall - compute, 0.0), 2),
    )


class VideoPipelineAgent(fal.App, keep_alive=300):
    machine_type = "S"
    app_auth = "private"
    # Waiting on other apps costs no CPU: one runner can drive many requests.
    max_multiplexing = 20
    requirements = [
        "fal==1.78.2",
        "fal-client==1.0.3",
    ]

    def setup(self) -> None:
        import fal_client

        # FAL_KEY is injected into every runner; it authorizes calls to private apps.
        self.client = fal_client.AsyncClient(default_timeout=STAGE_TIMEOUT_S)
        owner = stage_owner()
        self.apps = {stage: f"{owner}/{name}" for stage, name in STAGE_APPS.items()}

    async def call(self, stage: str, arguments: dict[str, Any]) -> tuple[dict, float]:
        t0 = time.perf_counter()
        try:
            result = await self.client.subscribe(
                self.apps[stage],
                arguments=arguments,
                client_timeout=STAGE_TIMEOUT_S,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"{stage} stage ({self.apps[stage]}) failed: {exc}",
            ) from exc
        return result, time.perf_counter() - t0

    @fal.endpoint("/")
    async def run(self, request: Input) -> Output:
        plan, llm_wall = await self.call("llm", {"idea": request.idea})

        dit_args: dict[str, Any] = {
            "prompt": plan["prompt"],
            "negative_prompt": plan["negative_prompt"],
            "num_frames": request.num_frames,
            "num_inference_steps": request.num_inference_steps,
        }
        if request.seed is not None:
            dit_args["seed"] = request.seed
        clip, dit_wall = await self.call("dit", dit_args)

        timings = {"llm": timing(llm_wall, plan), "dit": timing(dit_wall, clip)}
        final = clip
        if request.upscale:
            final, sr_wall = await self.call("sr", {"video_url": clip["video"]["url"]})
            timings["sr"] = timing(sr_wall, final)

        return Output(
            video=File(**final["video"]),
            base_video=File(**clip["video"]),
            prompt=plan["prompt"],
            negative_prompt=plan["negative_prompt"],
            seed=clip["seed"],
            timings=timings,
        )
