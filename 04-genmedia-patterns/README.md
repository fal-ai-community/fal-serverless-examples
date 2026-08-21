# Tier 04 · Genmedia patterns

Production serving patterns for generative media: per-user LoRA fleets, the train-a-LoRA-then-serve-it loop, multi-GPU video inference, and progressive 3D rendering. Every example follows the layout in [`templates/example/`](../templates/example/).

| Slot | Example |
|---|---|
| `04.3` | [**per-user-loras/**](per-user-loras/) — Serve thousands of per-user LoRAs with routing hints (avatar/personalization) |
| `04.4` | [**train-serve-lora/**](train-serve-lora/) — Train a LoRA → serve it (klein-base gradients, distilled serving, /data handoff) |
| `04.5` | [**multi-gpu-video/**](multi-gpu-video/) — Multi-GPU inference on a frontier video model (fal.distributed) |
| `04.7` | [**3d-progressive-rendering/**](3d-progressive-rendering/) — 3D progressive rendering (SAM-3D) |
