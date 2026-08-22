"""Same model, way 1: fal builds the environment (twin of app_container.py)."""

import fal
from pydantic import BaseModel, Field

MODEL_ID = "openai/whisper-tiny.en"
# Fetched at setup(); cold starts need huggingface.co reachable.
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
    machine_type = "L"  # 4 vCPU / 15 GB; whisper-tiny runs fine on CPU
    app_auth = "private"
    # Keep in sync with requirements.txt.
    requirements = [
        "fal==1.78.2",
        "torch==2.12.1",
        "transformers==5.3.0",
    ]

    def setup(self) -> None:
        import torch
        from transformers import pipeline

        self.pipe = pipeline(
            "automatic-speech-recognition", model=MODEL_ID, dtype=torch.float32
        )
        # A mis-built environment must fail here at setup, not at the first request.
        warmup = self.pipe(SAMPLE_AUDIO)
        assert warmup["text"].strip(), "warmup transcript is empty"

    @fal.endpoint("/")
    def transcribe(self, request: Input) -> Output:
        result = self.pipe(request.audio_url)
        return Output(text=result["text"].strip(), built_with="fal-runtime")
