"""<Example name> - fal app.

Template: replace Input/Output, setup(), and the endpoint body.
Conventions: pinned requirements, explicit machine_type, keep_alive chosen
deliberately (and explained in README "Cost & tuning").
"""
import fal
from pydantic import BaseModel, Field


class Input(BaseModel):
    prompt: str = Field(
        description="Describe the field for the playground and OpenAPI schema.",
        examples=["a calico cat riding a bicycle"],
    )


class Output(BaseModel):
    result: str = Field(description="Describe what comes back.")


class App(fal.App, keep_alive=60):
    machine_type = "GPU-H100"  # see docs: machine types
    app_auth = "private"  # required at deploy time; 'public' for playground-open demos
    # Authoritative deploy-time deps; keep requirements.txt an exact pinned mirror for local/CI.
    requirements = [
        "fal==1.78.2",  # keep in sync with harness/requirements.txt
        # pin everything: "torch==x.y.z", ...
    ]

    def setup(self) -> None:
        """Runs once per worker start: load weights here, ideally from /data."""

    @fal.endpoint("/")
    def run(self, request: Input) -> Output:
        return Output(result=request.prompt)
