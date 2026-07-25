"""Same model, way 1: the fal runtime.

Catalog slot 01.1 (SERV-1285). This file and app_container.py serve the
IDENTICAL model with the identical schema - the only difference is how the
environment gets built. This one lists dependencies and lets fal build the
environment; the other one ships a Dockerfile.

Both are deploy-checked against the same test_app.py, because a comparison
you can't run both halves of is just an opinion.
"""
import fal
from pydantic import BaseModel, Field

MODEL_ID = "openai/whisper-tiny.en"
# Canonical ASR sample used by the transformers docs: a clip of the
# "I have a dream" speech.
SAMPLE_AUDIO = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"


class Input(BaseModel):
    audio_url: str = Field(
        description="Audio file to transcribe.",
        examples=[SAMPLE_AUDIO],
    )


class Output(BaseModel):
    text: str = Field(description="The transcript.")
    built_with: str = Field(description="Which build path served this request.")


class TranscribeRuntime(fal.App, keep_alive=60):
    machine_type = "L"  # 4 vCPU / 15 GB - whisper-tiny is happy on CPU
    app_auth = "private"
    # WAY 1: declare dependencies, fal builds the environment. Pinned, and
    # mirrored exactly by requirements.txt for local installs and CI.
    requirements = [
        "fal==1.78.2",
        "torch==2.12.1",
        "transformers==5.3.0",
    ]

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
        return Output(text=result["text"].strip(), built_with="fal-runtime")
