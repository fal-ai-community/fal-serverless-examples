<p align="center">
  <a href="https://fal.ai">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://docs.fal.ai/logo/dark.svg">
      <img src="https://docs.fal.ai/logo/light.svg" alt="fal" height="96">
    </picture>
    <h3 align="center">fal Serverless Examples</h3>
  </a>
</p>

This is a collection of examples for [fal Serverless](https://fal.ai/docs/documentation/serverless). Use these examples to learn fal and to build and deploy your own models and pipelines. This repo supersedes [fal-demos](https://github.com/fal-ai-community/fal-demos).

## Usage

fal Serverless is a private offering — [talk to the fal team](https://fal.ai/enterprise#contact-sales) to get access for your account. Once you have it, install the `fal` package and log in:

```bash
pip install fal==1.78.2
fal auth login
```

Each example is one folder with an `app.py`. To run one:

```bash
cd 00-getting-started/hello-fal
fal run app.py
```

Your code runs on fal's cloud, not your machine. `fal run` prints two URLs — the API endpoint, and a playground page where you can try it from the browser — and keeps the app alive until you press Ctrl-C. When you're ready to ship, run `fal deploy app.py` instead.

## Examples

The numbered folders, [**`00-getting-started/`**](00-getting-started/) through [**`05-media-processing/`**](05-media-processing/), are a guided tour from your first app to production serving. Each folder's README lists what's inside. New folders are added over time, so the numbering has gaps.

[`llms.txt`](llms.txt) is an index of the catalog for AI tools.

_Every example comes with a test that CI runs against a real fal deploy ([`harness/`](harness/)), on Python **3.11**. Found something broken anyway? [Open an issue](https://github.com/fal-ai-community/fal-serverless-examples/issues)._

## License

The [MIT license](LICENSE).
