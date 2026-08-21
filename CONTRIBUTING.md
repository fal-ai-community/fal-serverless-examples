# Contributing

This catalog is **built and maintained by the fal team**. We're not accepting example contributions from outside fal right now.

**Found a broken example or a mistake?** Please open an issue — CI re-deploys every example against real infrastructure, but reader eyes catch what smoke tests don't. Small correction PRs (typos, broken links) are welcome too.

## For fal folks adding an example

1. Copy [`templates/example/`](templates/example/) into the tier directory. Short kebab-case folder name, no slot prefix; add the folder link to the tier README table.
2. **Pin everything.** `App.requirements` in `app.py` is authoritative; `requirements.txt` mirrors it exactly. No unpinned installs anywhere, including README snippets.
3. Set `app_auth` explicitly (deploys fail without it). Every required `Input` field carries `Field(examples=[...])` — that powers the playground and the CI smoke request.
4. README follows the template's section order: problem → output first → run it → how it works → deploy → cost & tuning → next steps. Show real output. Never publish a performance claim you can't back with a reproducible run.
5. Validate before the PR:

```bash
pip install -r harness/requirements.txt
python harness/static_check.py                      # free
python harness/check.py --example <your-example>    # real ephemeral deploy (fal auth login; costs a few cents)
```

6. PR title: `type(scope): description` — e.g. `feat(00-getting-started): media in, media out`. One example per PR; include the deploy-check output tail. Update `llms.txt`.

CI runs the static job on every PR; the real deploy-check runs behind the maintainer-approved `deploy-check` environment.
