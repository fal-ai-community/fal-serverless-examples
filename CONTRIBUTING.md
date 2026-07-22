# Contributing

This catalog is **maintained by the fal team** — examples are planned, built, and validated internally against the [Linear board](https://linear.app/features-and-labels/project/serverless-examples-and-onboarding-revamp-ab1f7491f6e7). We're not accepting example contributions from outside fal right now.

**Found a broken example or a mistake?** Please open an issue — CI re-deploys every example against real infrastructure, but reader eyes catch what smoke tests don't. Small correction PRs (typos, broken links) are welcome too.

## For fal folks adding an example

1. **Start from the ticket.** Every catalog slot has a Linear ticket (tier README tables link them); the ticket is the source of truth for scope. Read [`CONTEXT.md`](CONTEXT.md) first.
2. Copy [`templates/example/`](templates/example/) into the tier directory. Short kebab-case folder name, no slot prefix; add the folder link to the tier README table.
3. **Pin everything.** `App.requirements` in `app.py` is authoritative; `requirements.txt` mirrors it exactly. No unpinned installs anywhere, including README snippets.
4. Set `app_auth` explicitly (deploys fail without it). Every required `Input` field carries `Field(examples=[...])` — that powers the playground and the CI smoke request.
5. README follows the template's section order: problem → output first → run it → how it works → deploy → cost & tuning → next steps. Show real output. Never publish a performance claim you can't back with a reproducible run.
6. Validate before the PR:

```bash
pip install -r harness/requirements.txt
python harness/static_check.py                      # free
python harness/check.py --example <your-example>    # real ephemeral deploy (fal auth login; costs a few cents)
```

7. PR title: `type(scope): description` — e.g. `feat(03-streaming-realtime): SSE preview frames`. One example per PR; include the deploy-check output tail. Update `llms.txt`.

CI runs the static job on every PR; the real deploy-check runs behind the maintainer-approved `deploy-check` environment.
