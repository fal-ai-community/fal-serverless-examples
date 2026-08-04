# Serve thousands of per-user LoRAs with routing hints

<!-- Stub: the full §4-A page lands with Task 7 of docs/superpowers/plans/2026-08-03-per-user-loras.md. -->

One warm FLUX.2 [klein] 4B fleet, one LoRA per user: adapters on `/data`, a KV
user index, a bounded GPU adapter LRU per runner, and fal routing hints so
repeat traffic lands where the user's adapter is already hot.
