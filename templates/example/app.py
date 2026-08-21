"""Catalog slot NN.N. <Example name> - fal app.

Template: rename the class and endpoint verb; replace the Input/Output fields,
setup(), the endpoint body, and these notes.
Conventions: pinned requirements, explicit machine_type, keep_alive chosen
deliberately (and explained in README "Cost & tuning").
"""
import fal
from pydantic import BaseModel, Field


class Input(BaseModel):
    prompt: str = Field(
        description="Describe the field for the playground and OpenAPI schema.",
        # Every required field needs examples: the CI smoke payload is built from them.
        examples=["a calico cat riding a bicycle"],
    )


class Output(BaseModel):
    # Name output fields for what they hold (message, images, video, ...), never `result`.
    message: str = Field(description="Describe what comes back.")


class ExampleEcho(fal.App, keep_alive=60):  # rename to match your folder, e.g. HelloFal
    machine_type = "S"  # smallest CPU tier fits this echo app; GPU examples override (e.g. "GPU-H100")
    app_auth = "private"  # required at deploy time; 'public' for playground-open demos
    # Authoritative deploy-time deps; keep requirements.txt an exact pinned mirror for local/CI.
    requirements = [
        "fal==1.78.2",  # keep in sync with harness/requirements.txt
        # pin everything: "torch==x.y.z", ...
    ]

    def setup(self) -> None:
        """Runs once per worker start: load weights (ideally from /data), then run a warmup inference."""

    @fal.endpoint("/")
    def echo(self, request: Input) -> Output:  # rename to a verb naming your action (generate, transcribe, ...)
        return Output(message=request.prompt)
