"""Same model, way 2: your own Dockerfile.

Identical model, identical schema, identical pinned versions as app.py - the
environment is the only variable. fal builds this Dockerfile server-side, so
no local Docker and no registry push.
"""

from pathlib import Path

import fal
from fal.container import ContainerImage
from pydantic import BaseModel, Field

MODEL_ID = "openai/whisper-tiny.en"
# Canonical ASR sample from the transformers docs ("I have a dream" clip).
# setup() fetches it from huggingface.co, so cold starts need that host up.
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
    # Dependencies are baked into the image by the Dockerfile; no `requirements` list.
    image = ContainerImage.from_dockerfile(str(DOCKERFILE))

    def setup(self) -> None:
        import torch
        from transformers import pipeline

        self.pipe = pipeline(
            "automatic-speech-recognition", model=MODEL_ID, dtype=torch.float32
        )
        # Warmup exercises the full audio-decode path; a mis-built environment
        # should fail here at setup, not at the first request.
        warmup = self.pipe(SAMPLE_AUDIO)
        assert warmup["text"].strip(), "warmup transcript is empty"

    @fal.endpoint("/")
    def transcribe(self, request: Input) -> Output:
        result = self.pipe(request.audio_url)
        return Output(text=result["text"].strip(), built_with="container")
