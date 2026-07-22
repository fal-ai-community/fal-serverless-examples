# Contributing

Thanks for wanting to add an example! This repo is a curated, numbered curriculum — every example is CI-verified against real fal infrastructure, so the bar is "runs, deploys, and stays running", not "worked once on my machine".

## Before you write code

1. **Propose the example first** — open an issue using the "Propose an example" template. The catalog (tiers 00–08, each tier README has the table) is maintainer-curated; proposals that fill a planned slot or add a genuinely new production pattern are most likely to land.
2. Wait for a maintainer 👍 on the slot/scope before investing in a PR.

## Building an example

- Copy [`templates/example/`](templates/example/) into the right tier directory. Short kebab-case folder name, no slot prefix.
- **Pin everything.** `App.requirements` in `app.py` is authoritative; `requirements.txt` must mirror it exactly. No unpinned installs anywhere, including README snippets.
- Set `app_auth` explicitly on the App class (deploys fail without it).
- Every required `Input` field carries `Field(examples=[...])` — that's what the playground and the CI smoke request are built from.
- README follows the template's section order: problem → output shown first → run it → how it works → deploy → cost & tuning → next steps. Show real output. Never publish a performance claim you can't back with a reproducible run.
- `test_app.py` stays tiny: the `app_client` fixture (provided by the harness) hits your live ephemeral deploy.

## Validating

```bash
pip install -r harness/requirements.txt
python harness/static_check.py                      # free, no account needed
python harness/check.py --example <your-example>    # real ephemeral deploy: needs `fal auth login` and spends a few cents
```

CI runs the same two stages: static checks on every PR, and the real deploy-check after maintainer approval (it spends money, so fork PRs wait for a reviewer).

## PR conventions

- Title: `type(scope): description` — e.g. `feat(03-streaming-realtime): SSE preview frames`.
- One example per PR. Include what you validated (deploy-check output, playground link if you deployed it yourself).
