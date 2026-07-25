"""Same model, way 2: your own Dockerfile.

Catalog slot 01.1 (SERV-1285). Identical model, identical schema, identical
pinned versions as app.py - the environment is the only variable. fal builds
this Dockerfile server-side, so no local Docker and no registry push.
"""
from pathlib import Path

import fal
from fal.container import ContainerImage
from pydantic import BaseModel, Field

MODEL_ID = "openai/whisper-tiny.en"
# Canonical ASR sample used by the transformers docs: a clip of the
# "I have a dream" speech.
SAMPLE_AUDIO = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
DOCKERFILE = Path(__file__).parent / "Dockerfile"


class Input(BaseModel):
    audio_url: str = Field(
        description="Audio file to transcribe.",
        examples=[SAMPLE_AUDIO],
    )


class Output(BaseModel):
    text: str = Field(description="The transcript.")
    built_with: str = Field(description="Which build path served this request.")


class TranscribeContainer(fal.App, keep_alive=60):
    machine_type = "L"
    app_auth = "private"
    # WAY 2: the image carries the dependencies, so `requirements` stays
    # empty - everything this app imports is baked in by the Dockerfile.
    image = ContainerImage.from_dockerfile(str(DOCKERFILE))

    def setup(self) -> None:
        import torch
        from transformers import pipeline

        self.pipe = pipeline(
            "automatic-speech-recognition", model=MODEL_ID, torch_dtype=torch.float32
        )
        self.pipe(SAMPLE_AUDIO)  # warm the graph

    @fal.endpoint("/")
    def transcribe(self, request: Input) -> Output:
        result = self.pipe(request.audio_url)
        return Output(text=result["text"].strip(), built_with="container")
