"""Dev -> staging -> prod: one codebase, three environments.

Catalog slot 01.3 (SERV-1299). The app is deliberately boring - what it
demonstrates is that IDENTICAL code behaves differently per environment
because its secrets and scale settings are environment-scoped. It reports
back exactly what it was given, so you can prove the promotion worked.

Important framing: fal has no "promote" primitive. An app cannot be moved
between environments; promotion is redeploying the same code into the next
one. Revisions and rollbacks are per-app WITHIN one environment.
"""
import os

import fal
from pydantic import BaseModel, Field

# Set per environment:
#   fal secrets set --env staging DEPLOY_STAGE=staging
#   fal secrets set DEPLOY_STAGE=production        # main is the default env
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
        description="Value of the DEPLOY_STAGE secret in THIS environment."
    )
    code_version: str = Field(description="Constant baked into the code.")
    environment_configured: bool = Field(
        description="False when the environment has no DEPLOY_STAGE secret set."
    )


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
        stage = os.getenv(STAGE_SECRET)
        return Output(
            echo=request.echo,
            stage=stage or "unset",
            code_version=CODE_VERSION,
            environment_configured=stage is not None,
        )
