# Multi-GPU video inference with fal.distributed

Serve WAN 2.2 A14B - a 2x14B-expert text-to-video model whose weights alone
outgrow a single GPU's memory budget - across 4 GPUs with Ulysses sequence
parallelism: one prompt, one video, four GPUs cooperating on every denoising
step.

> Catalog slot `04.5` ([SERV-1316](https://linear.app/features-and-labels/issue/SERV-1316))

_Full walkthrough landing with the validated deploy - see app.py._
