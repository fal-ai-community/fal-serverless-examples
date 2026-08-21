# fal-serverless-examples

Production-grade examples for [fal Serverless](https://fal.ai/docs/documentation/serverless) — a numbered curriculum that takes you from your first deployed app to multi-GPU serving and production media pipelines. Every example ends in a **deployed endpoint**, and CI **re-deploys and smoke-tests each one for real**, so nothing here can silently rot.

## Catalog

| Directory | Tier | Examples | Focus |
|---|---|---|---|
| [`00-getting-started/`](00-getting-started/) | Getting started | 4 | First app to first production API: hello-fal, your first image API, media in/out, bulletproof inputs. |
| [`01-environments-containers/`](01-environments-containers/) | Environments & containers | 2 | fal runtime and your own Dockerfile side by side; dev → staging → prod promotion and rollback. |
| [`02-scaling-cold-starts/`](02-scaling-cold-starts/) | Scaling & cold starts | 3 | Cold-start engineering with measured before/afters, autoscaling tuned from a real burst, serving very large video models. |
| [`04-genmedia-patterns/`](04-genmedia-patterns/) | Genmedia patterns | 4 | Production genmedia serving: per-user LoRA fleets, train a LoRA → serve it, multi-GPU video, progressive 3D streaming. |
| [`05-media-processing/`](05-media-processing/) | Media processing | 1 | The CPU pipelines around the models: transcode ladders, thumbnails, captions, queue + webhooks. |

Tier and slot numbers are stable identifiers, and the numbering is deliberately sparse — gaps between numbers are expected.

## What makes these examples different

- **Journey-ordered**: the tiers build on each other, following the path a real team takes — getting started → environments → scaling → genmedia serving → media pipelines.
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

- [fal docs](https://fal.ai/docs) — the documentation these examples back
- [fal-demos](https://github.com/fal-ai-community/fal-demos) — companion demo repo

## License

[MIT](LICENSE).
