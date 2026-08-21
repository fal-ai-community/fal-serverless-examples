"""Dev -> staging -> prod.

Deliberately boring app: it reports back exactly what each environment gave
it (secret value + code version), so a promotion or rollback can be proven
rather than assumed.
"""

import os

import fal
from pydantic import BaseModel, Field

STAGE_SECRET = "DEPLOY_STAGE"
# Bumped by hand to make rollbacks visible - see the README's rollback step.
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


# fal derives the app name from the class name: PromotionDemo deploys as
# "promotion-demo", the name the README's URLs are written against.
class PromotionDemo(fal.App, keep_alive=60):
    machine_type = "S"
    app_auth = "private"
    # Only this app gets DEPLOY_STAGE injected, even though secrets are
    # account-wide by default. Least privilege costs one line.
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
