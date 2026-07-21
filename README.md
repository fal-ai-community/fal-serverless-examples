# fal-serverless-examples

Production-grade examples for [fal Serverless](https://fal.ai/docs/documentation/serverless) — a numbered curriculum that takes you from your first app to production operations. Every example ends in a **deployed endpoint**, and CI **re-deploys and smoke-tests each one for real**, so nothing here can silently rot.

> Status: scaffold. Examples land tier by tier — see the catalog board in [Linear](https://linear.app/features-and-labels/project/serverless-examples-and-onboarding-revamp-ab1f7491f6e7) and each tier's README for what's planned.

## Catalog

| Directory | Tier | Entries | Focus |
|---|---|---|---|
| [`00-getting-started/`](00-getting-started/) | Getting started | 4 planned | First app to first production API: hello-fal three ways, your first image API, media in/out, bulletproof inputs. |
| [`01-environments-containers/`](01-environments-containers/) | Environments & containers | 3 planned | Runtimes and Dockerfiles side by side, migrations from other platforms, dev-to-prod environment promotion. |
| [`02-scaling-cold-starts/`](02-scaling-cold-starts/) | Scaling & cold starts | 4 planned | Cold-start engineering with measured before/afters, autoscaling for bursty traffic, serving very large models. |
| [`03-streaming-realtime/`](03-streaming-realtime/) | Streaming & realtime | 4 planned | SSE progressive results, WebSocket interactive editing, WebRTC realtime video and world models. |
| [`04-genmedia-patterns/`](04-genmedia-patterns/) | Genmedia patterns | 7 planned | Production serving for image, video, audio, and 3D: ComfyUI, LTX-class video, per-user LoRAs, multi-GPU, TTS/music. |
| [`05-media-processing/`](05-media-processing/) | Media processing | 6 planned | The CPU/GPU pipelines around the models: transcode ladders, 3D LODs, image variants, audio finishing, provenance stamping, batch fan-out. |
| [`06-pipelines-apps/`](06-pipelines-apps/) | Pipelines & apps | 6 planned | Composing endpoints into products: workflows, queue + webhooks backends, multi-app routing, failover, client-side chaining. |
| [`07-integrations/`](07-integrations/) | Integrations | 1 planned | Frameworks and agents: Next.js/Vercel, n8n, deploy via MCP. |
| [`08-production-ops/`](08-production-ops/) | Production ops | 5 planned | Observability, resilience, CI testing with ephemeral deploys, marketplace publishing, cost tracking. |

## What makes these examples different

- **Journey-ordered**: tiers 00→08 follow the path a real team takes — getting started → scaling → realtime → genmedia → pipelines → production ops.
- **CI-verified, not hoped-for**: the harness in [`harness/`](harness/) ephemeral-deploys every example and runs its smoke test against the live endpoint — a broken example turns CI red instead of rotting quietly.
- **Deploy is the finish line**: each example ends with `fal deploy`, a playground link, and client code that calls the live endpoint.
- **AI-native**: [`llms.txt`](llms.txt) indexes the catalog for agents and LLM tools.

## Layout of every example

```
NN-tier-name/short-example-name/
  README.md          # problem → output first → code → deploy → cost & tuning → next steps
  app.py             # the fal app (or a package for bigger examples)
  requirements.txt   # pinned
  test_app.py        # smoke test (run by CI against a real ephemeral deploy)
```

Start from [`templates/example/`](templates/example/). Folder names are short kebab-case with no slot prefix — the slot ↔ folder mapping lives in each tier's README table.

## Running an example

```bash
pip install fal==1.78.2   # each example's requirements.txt pins the exact version
fal auth login
cd 00-getting-started/<example>
fal run app.py        # iterate
fal deploy app.py     # ship it
```

## Related

- [fal docs](https://fal.ai/docs) — the documentation these examples back ([repo](https://github.com/fal-ai/fal-docs))
- [fal-demos](https://github.com/fal-ai-community/fal-demos) — predecessor demo repo (being superseded by this catalog)

## License

TBD — decided as part of the org-transfer ticket ([SERV-1279](https://linear.app/features-and-labels/issue/SERV-1279)).
