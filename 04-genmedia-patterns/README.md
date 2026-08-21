# Tier 04 · Genmedia patterns

Production serving for image, video, and 3D: per-user LoRAs, LoRA training, multi-GPU inference, progressive 3D rendering.

| Example | What it covers |
|---|---|
| [`per-user-loras/`](per-user-loras/) | Serve thousands of per-user LoRAs with routing hints (avatar/personalization) |
| [`train-serve-lora/`](train-serve-lora/) | Train a LoRA → serve it (klein-base gradients, distilled serving, /data handoff) |
| [`multi-gpu-video/`](multi-gpu-video/) | Multi-GPU inference on a frontier video model (fal.distributed) |
| [`3d-progressive-rendering/`](3d-progressive-rendering/) | 3D progressive rendering (SAM-3D), with intermediate states streamed live |
