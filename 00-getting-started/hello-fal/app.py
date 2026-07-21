"""Hello, fal - the smallest deployable fal app.

Catalog slot 00.1 (SERV-1284). One pydantic-typed endpoint: run it with
`fal run`, ship it with `fal deploy`, measure it with the numbers harness.
The full "one app, run three ways" walkthrough lands with SERV-1284.
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
    machine_type = "S"  # 1 GB CPU tier: same $/s as XS today, headroom for the runtime
    app_auth = "private"  # deploy requires an explicit auth mode; callers send a fal key
    # Authoritative deploy-time deps; requirements.txt mirrors this exactly.
    requirements = [
        "fal==1.78.2",
    ]

    def setup(self) -> None:
        """Nothing to load here - loading real model weights is 00.2's job."""

    @fal.endpoint("/")
    def greet(self, request: Input) -> Output:
        return Output(message=f"Hello, {request.name}!")
