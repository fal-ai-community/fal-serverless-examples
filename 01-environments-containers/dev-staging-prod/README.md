# Dev → staging → prod

You want to try a change against real infrastructure without touching production, then ship the *same* code forward, then undo it fast when it turns out to be wrong. On fal that's three ideas — environments, environment-scoped secrets, and revisions — and one thing that is **not** an idea: there is no "promote" button. An app cannot move between environments; promotion is redeploying identical code into the next one.

## What you get

The same [`app.py`](app.py), deployed twice, reporting what each environment gave it:

```bash
# staging
{"echo":"hi","stage":"staging","code_version":"1.0.0","environment_configured":true}
# main (production)
{"echo":"hi","stage":"production","code_version":"1.0.0","environment_configured":true}
```

Identical source, identical command, different answer — because `DEPLOY_STAGE` is scoped per environment. Every transcript on this page is copied from a real run.

## Run it

```bash
pip install fal==1.78.2
fal auth login
fal run app.py                    # dev: ephemeral, dies with your terminal
fal run app.py --env staging      # dev, but resolving staging's secrets
```

## How it works

The app is deliberately trivial: it echoes your input and reports the `DEPLOY_STAGE` secret plus a `CODE_VERSION` constant. That's enough to *prove* which environment and which revision served a request — which is the only way to know a promotion or a rollback actually did anything.

Two details in [`app.py`](app.py) worth copying:

- **`secrets = ["DEPLOY_STAGE"]`** — secrets are account-wide and injected into every app by default; listing them explicitly means only this app receives it. One line, least privilege.
- **A missing secret degrades to `"unset"`**, it doesn't crash. An app that only boots in the environment you remembered to configure is a promotion hazard.

## The walkthrough

### 1. Scope a secret per environment

```bash
fal secrets set --env staging DEPLOY_STAGE=staging
fal secrets set DEPLOY_STAGE=production          # no --env: main is the default
```

Secrets are injected as environment variables **when a runner starts**. Runners already running keep the old value until they cycle.

### 2. Deploy to staging

```bash
fal deploy app.py --env staging
```

Staging apps get a suffixed URL — `https://fal.run/<user>/promotion-demo--staging` — and are **completely separate** from main. Verified: while only staging existed, `https://fal.run/<user>/promotion-demo` returned

```json
{"detail": "Application 'promotion-demo' not found"}
```

Environments don't share apps, URLs, secrets, or revisions.

### 3. Promote — which means redeploy the same code

```bash
fal deploy app.py            # same file, same command, no --env
```

Nothing is copied from staging; you are deploying the identical source into a different environment, where it picks up that environment's secrets. If your staging and production code can drift, that's your CI's job to prevent, not the platform's.

### 4. Roll back — three steps, and the third one surprises people

```bash
fal apps list-rev promotion-demo                  # find the revision you want
fal apps set-rev promotion-demo <revision-id>     # point the app at it
fal apps rollout promotion-demo --force           # cycle the runners
```

**`set-rev` alone does not change what your users get.** Verified on a live app: after `set-rev`, the control plane reported the old revision as active while every request still returned the *new* code — three warm runners were happily serving it. A plain `fal apps rollout` (graceful) also kept serving it. Only `--force`, which kills runners immediately, flipped traffic — within ~12s:

```
after deploying v1.1.0:            code_version=1.1.0
set-rev -> v1.0.0, then requests:  code_version=1.1.0   # still the old runners
rollout --force, +12s:             code_version=1.0.0   # rolled back
```

Graceful rollout is the right default when you're shipping *forward* (in-flight requests finish). When you're rolling back because production is broken, `--force` is the point. Without it you also just wait out `keep_alive`.

Revisions are **per app, per environment**: rolling back main does nothing to staging.

## Cost & tuning

- Environments cost nothing by themselves — you pay for runners, wherever they are. A staging app with `keep_alive=60` and no traffic bills nothing once its runner expires.
- Set scale settings per environment: staging usually wants `min_concurrency=0` (cold starts are fine), production often wants ≥1 (they aren't). `fal apps scale <app> --env staging --min-concurrency 0`.
- `fal deploy` **keeps the previous deployment's scale settings** unless you pass `--reset-scale` — so a production `min_concurrency` you set by hand survives your next deploy. Worth knowing before you go hunting for why a change didn't apply.

## Next steps

- **[same-model-two-ways](../same-model-two-ways/)**: choose how the environment gets built before you promote it.
- Concept docs: [managing environments](https://fal.ai/docs/documentation/deployment/manage-environments) · [secrets](https://fal.ai/docs/documentation/deployment/manage-secrets-securely) · [deployments & revisions](https://fal.ai/docs/documentation/deployment/manage-deployments)
