"""Promotion demo: reports the secret and code version each environment gave it."""
import os

import fal
from pydantic import BaseModel, Field

STAGE_SECRET = "DEPLOY_STAGE"
# Bumped by hand so rollbacks are visible (README rollback step).
CODE_VERSION = "1.0.0"


class Input(BaseModel):
    echo: str = Field(
        description="Any string; returned unchanged so you can trace a request.",
        examples=["hello from the promotion pipeline"],
    )


class Output(BaseModel):
    echo: str = Field(description="Your input, unchanged.")
    stage: str = Field(
        description=(
            "Value of the DEPLOY_STAGE secret in THIS environment, "
            "or 'unset' when the secret is missing."
        )
    )
    code_version: str = Field(description="Constant baked into the code.")
    environment_configured: bool = Field(
        description=(
            "False when the environment's DEPLOY_STAGE secret is missing or empty."
        )
    )


# Class name -> app name "promotion-demo"; the README's URLs depend on it.
class PromotionDemo(fal.App, keep_alive=60):
    machine_type = "S"
    app_auth = "private"
    secrets = [STAGE_SECRET]
    requirements = [
        "fal==1.78.2",
    ]

    def setup(self) -> None:
        """Nothing to load - the environment is the subject here."""

    @fal.endpoint("/")
    def whoami(self, request: Input) -> Output:
        stage = os.getenv(STAGE_SECRET) or None  # empty-string secret == unset
        return Output(
            echo=request.echo,
            stage=stage or "unset",
            code_version=CODE_VERSION,
            environment_configured=stage is not None,
        )
