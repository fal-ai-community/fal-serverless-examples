# Tier 02 · Scaling & cold starts

Cold-start engineering with measured before/afters, autoscaling for bursty traffic, serving very large models.

| Example | What it covers |
|---|---|
| [`cut-cold-starts/`](cut-cold-starts/) | Cut cold starts with FlashPack + /data + compiled caches — the same model deployed with and without every lever, so the before/after is a real measured deploy |
| [`tune-autoscaling/`](tune-autoscaling/) | Tune autoscaling for bursty traffic: keep_alive, concurrency, cost math |
| [`large-video-model/`](large-video-model/) | Serve a 20+ GB video model: parallel loading, GPU sizing reasoning, tiny-VAE previews over SSE |
