<p align="center">
  <a href="https://fal.ai">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://docs.fal.ai/logo/dark.svg">
      <img src="https://docs.fal.ai/logo/light.svg" alt="fal" height="96">
    </picture>
    <h3 align="center">fal Serverless Examples</h3>
  </a>
</p>

Runnable examples for [fal Serverless](https://fal.ai/docs/documentation/serverless) — deploying your own models and pipelines on fal GPUs, from a first endpoint to production serving patterns. This repo supersedes [fal-demos](https://github.com/fal-ai-community/fal-demos).

## What's here

The numbered folders, [**`00-getting-started/`**](00-getting-started/) through [**`05-media-processing/`**](05-media-processing/), build on each other — start at 00 and work forward. Each folder's README lists what's inside. New folders land over time, so the numbering has gaps.

Every example is one self-contained folder: an `app.py`, pinned requirements, and a smoke test, on Python 3.11. Every pull request runs static checks — layout, pinned requirements mirrored, the app module loads, a smoke payload is derivable. The same harness ([`harness/`](harness/)) also deploys an example to fal for real, smoke-tests it, runs `test_app.py`, and tears it down; that job spends money, so a maintainer starts it. [`docs/`](docs/) holds Markdown copies of the fal docs pages on pricing, ComfyUI workflows and migrating a Cog model or custom container. [`llms.txt`](llms.txt) indexes the catalog for AI tools. Found something broken? [Open an issue](https://github.com/fal-ai-community/fal-serverless-examples/issues).

## Running an example

fal Serverless is a private offering — [talk to the fal team](https://fal.ai/enterprise#contact-sales) to get access for your account. Then:

```bash
pip install fal==1.78.2
fal auth login
cd 00-getting-started/hello-fal
fal run app.py
```

The app runs on fal's cloud, not your machine. `fal run` prints two URLs — the API endpoint, and a playground page to try it from the browser — and Ctrl-C tears it down. When you're ready to ship, `fal deploy app.py` instead.

## License

MIT — see [LICENSE](LICENSE).
