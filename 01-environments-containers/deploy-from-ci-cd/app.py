"""A tiny CPU app deployed by GitHub Actions after local tests pass."""

import fal
from pydantic import BaseModel, Field


class Input(BaseModel):
    name: str = Field(description="Who to greet.", examples=["fal"])


class Output(BaseModel):
    message: str = Field(description="The greeting.")


class CICDDemo(fal.App, keep_alive=60):
    machine_type = "S"
    app_auth = "private"
    requirements = ["fal==1.83.0"]

    @fal.endpoint("/")
    def greet(self, request: Input) -> Output:
        return Output(message=f"Hello, {request.name}!")
