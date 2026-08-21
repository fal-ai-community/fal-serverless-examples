# Tier 02 · Scaling & cold starts

Cold-start engineering with measured before/afters, autoscaling for bursty traffic, serving very large models. Every example follows the layout in [`templates/example/`](../templates/example/).

| Slot | Example |
|---|---|
| `02.1` | [**cut-cold-starts/**](cut-cold-starts/) — Cut cold starts (FlashPack + /data + compiled caches) |
| `02.2` | [**tune-autoscaling/**](tune-autoscaling/) — Tune autoscaling for bursty traffic (keep_alive, concurrency, cost math) |
| `02.3` | [**large-video-model/**](large-video-model/) — Serve a 20+ GB video model (parallel loading, GPU sizing reasoning, tiny-VAE previews over SSE) |
