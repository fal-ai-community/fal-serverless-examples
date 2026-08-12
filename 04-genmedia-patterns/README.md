# Tier 04 · Genmedia patterns

Production serving for image, video, audio, and 3D: ComfyUI, LTX-class video, per-user LoRAs, multi-GPU, TTS/music.

Example folders land here as their tickets complete. Every example follows the layout in [`templates/example/`](../templates/example/) and the quality bar in [`CONTEXT.md`](../CONTEXT.md).

| Slot | Example | Readiness | Ticket |
|---|---|---|---|
| `04.1` | ComfyUI workflow → production API (real workflow, custom nodes, current models) | `needs-deploy` · blocked: [SERV-1274](https://linear.app/features-and-labels/issue/SERV-1274) (upstream coordination) | [SERV-1292](https://linear.app/features-and-labels/issue/SERV-1292) |
| `04.2` | Text-to-video on a current model (LTX-class, 720p+, endpoint + client + numbers) | `needs-deploy` | [SERV-1300](https://linear.app/features-and-labels/issue/SERV-1300) |
| `04.3` | [Serve thousands of per-user LoRAs with routing hints](per-user-loras/) (avatar/personalization) | `needs-deploy` · [SERV-1273](https://linear.app/features-and-labels/issue/SERV-1273) answered: hint limits verified from source + affinity lift measured live at 10-runner scale 2026-08-04 (see ticket) | [SERV-1306](https://linear.app/features-and-labels/issue/SERV-1306) |
| `04.4` | Train a LoRA → serve it (klein-base gradients, distilled serving, /data handoff) — [`train-serve-lora/`](train-serve-lora/) | `needs-deploy` | [SERV-1312](https://linear.app/features-and-labels/issue/SERV-1312) |
| `04.5` | Multi-GPU inference on a frontier video model (fal.distributed) — [`multi-gpu-video/`](multi-gpu-video/) | `needs-deploy` · deploy-check green 2026-08-05; registered-deploy validation pending | [SERV-1316](https://linear.app/features-and-labels/issue/SERV-1316) |
| `04.6` | TTS (Kokoro) + text-to-music (DiffRhythm); refresh models | `needs-deploy` | [SERV-1318](https://linear.app/features-and-labels/issue/SERV-1318) |
| `04.7` | 3D progressive rendering (SAM-3D) | `needs-deploy` | [SERV-1319](https://linear.app/features-and-labels/issue/SERV-1319) |

Readiness: `ready-now` = writable from existing docs today · `needs-deploy` = needs a validation deployment · `net-new` = new code/pattern to design first.
