# fal-serverless-examples

Production-grade examples for [fal Serverless](https://fal.ai/docs/documentation/serverless) — a numbered curriculum that takes you from your first app to production-grade serving. Every example ends in a **deployed endpoint**, and CI **re-deploys and smoke-tests each one for real**, so nothing here can silently rot.

> The catalog grows tier by tier — each tier's README lists what's inside.

## Catalog

| Directory | Tier | Entries | Focus |
|---|---|---|---|
| [`00-getting-started/`](00-getting-started/) | Getting started | 4 examples | First app to first production API: hello-fal, your first image API, media in/out, bulletproof inputs. |
| [`01-environments-containers/`](01-environments-containers/) | Environments & containers | 2 examples | Runtimes and Dockerfiles side by side, dev-to-prod environment promotion. |
| [`02-scaling-cold-starts/`](02-scaling-cold-starts/) | Scaling & cold starts | 3 examples | Cold-start engineering with measured before/afters, autoscaling for bursty traffic, serving very large models. |
| [`04-genmedia-patterns/`](04-genmedia-patterns/) | Genmedia patterns | 4 examples | Production serving for image, video, and 3D: per-user LoRAs, LoRA train-and-serve, multi-GPU video, progressive 3D rendering. |
| [`05-media-processing/`](05-media-processing/) | Media processing | 1 example | The CPU pipelines around the models: transcode ladder, thumbnails, and captions in one post-production pipeline. |

## What makes these examples different

- **Journey-ordered**: tiers 00→05 follow the path a real team takes — getting started → environments → scaling → genmedia → media processing.
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

Folder names are short kebab-case.

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

[MIT](LICENSE).
