"""Hello, fal - the smallest deployable fal app.

One pydantic-typed endpoint: run it with
`fal run`, ship it with `fal deploy`, verify it with the CI deploy-check.
"""

import fal
from pydantic import BaseModel, Field


class Input(BaseModel):
    name: str = Field(
        description="Who to greet.",
        examples=["fal"],
    )


class Output(BaseModel):
    message: str = Field(description="The greeting, produced on a fal worker.")


class HelloFal(fal.App, keep_alive=60):
    machine_type = (
        "S"  # 1 GB CPU tier - no GPU needed to greet; headroom for the runtime
    )
    app_auth = (
        "private"  # deploy requires an explicit auth mode; callers send a fal key
    )
    requirements = [
        "fal==1.78.2",
    ]

    def setup(self) -> None:
        """Intentionally empty - real apps load model weights here, once per worker."""

    @fal.endpoint("/")
    def greet(self, request: Input) -> Output:
        return Output(message=f"Hello, {request.name}!")
