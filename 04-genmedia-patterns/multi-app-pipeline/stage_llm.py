"""Pipeline stage 1: an LLM on its own GPU rewrites a short idea into a video prompt."""

import json
import time

import fal
from pydantic import BaseModel, Field

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
SYSTEM_PROMPT = (
    "You write prompts for a text-to-video model. Rewrite the user's idea into one "
    "vivid, concrete shot description of at most 80 words: subject, action, setting, "
    "lighting, camera. Reply with JSON only: "
    '{"prompt": "...", "negative_prompt": "..."}'
)
DEFAULT_NEGATIVE = "blurry, low quality, distorted, watermark, text"


class Input(BaseModel):
    idea: str = Field(
        description="A short idea for a video, in any wording.",
        examples=["a red panda napping on a branch"],
    )
    max_new_tokens: int = Field(
        default=200, ge=32, le=512, description="Upper bound on generated tokens."
    )


class Output(BaseModel):
    prompt: str = Field(description="The expanded text-to-video prompt.")
    negative_prompt: str = Field(description="What the video model should avoid.")
    compute_seconds: float = Field(description="Time spent inside this stage.")


def parse_plan(text: str) -> tuple[str, str]:
    """Pull the JSON plan out of the reply; fall back to the raw text as the prompt."""
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            plan = json.loads(text[start : end + 1])
            prompt = str(plan.get("prompt", "")).strip()
            if prompt:
                negative = str(plan.get("negative_prompt", "")).strip()
                return prompt, negative or DEFAULT_NEGATIVE
        except json.JSONDecodeError:
            pass
    return text.strip(), DEFAULT_NEGATIVE


class PipelineLLM(fal.App, name="video-pipeline-llm", keep_alive=300):
    # A 7B model in bf16 is ~15 GB: an A100 is enough, no need for an H100.
    machine_type = "GPU-A100"
    app_auth = "private"
    startup_timeout = 1200
    requirements = [
        "fal==1.78.2",
        "torch==2.12.1",
        "transformers==5.3.0",
        "accelerate==1.14.0",
        "huggingface_hub==1.27.0",
    ]

    def setup(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16
        ).to("cuda")
        warmup = self.expand(Input(idea="a lighthouse in a storm", max_new_tokens=64))
        assert warmup.prompt, "warmup produced an empty prompt"

    @fal.endpoint("/")
    def expand(self, request: Input) -> Output:
        import torch

        t0 = time.perf_counter()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.idea},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to("cuda")
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=request.max_new_tokens, do_sample=False
            )
        reply = self.tokenizer.decode(
            out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        prompt, negative = parse_plan(reply)
        return Output(
            prompt=prompt,
            negative_prompt=negative,
            compute_seconds=round(time.perf_counter() - t0, 2),
        )
