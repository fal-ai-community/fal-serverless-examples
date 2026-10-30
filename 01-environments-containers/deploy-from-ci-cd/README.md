# Deploy from CI/CD

You want a pull request to prove an app's behavior before deploying it, then deploy the merged code to production without an interactive login. This example keeps the app tiny so the GitHub Actions workflow is the interesting part.

## What you get

The private CPU app accepts `{"name": "fal"}` at its `/` endpoint and returns:

```json
{"message": "Hello, fal!"}
```

The root [`deploy.yml`](../../.github/workflows/deploy.yml) runs local tests before any deployment:

| Event | Tests | Deployment after tests pass and opt-in is enabled |
|---|---|---|
| Same-repository pull request | Yes | fal `staging` |
| Fork pull request | Yes | Never |
| Push to `main` | Yes | fal `main` (production) |
| Manual `workflow_dispatch` | Yes | fal `staging` |

Pull request and push runs are filtered to this example's directory and the workflow file. Publishing or copying the workflow does not enable deployments: the repository variable `FAL_CI_CD_ENABLED` must be the string `true`.

## Run it

From the repository root:

```bash
cd 01-environments-containers/deploy-from-ci-cd
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt pytest==9.1.1
python -m pytest -q test_app.py
```

These tests require no credentials and provision no fal compute. They call the endpoint method directly, check the response, and verify required string input. They do not exercise HTTP transport, the remote runtime, or a deployed app. For infrastructure-level testing later, see [Test Models and Endpoints](https://fal.ai/docs/documentation/development/test-models-and-endpoints).

## How it works

- [`app.py`](app.py) defines `CICDDemo`, a private `S` CPU app with one typed greeting endpoint. There are no model weights or GPU requirements.
- `requirements = ["fal==1.83.0"]` is the remote runtime dependency list. [`requirements.txt`](requirements.txt) mirrors it exactly. The workflow installs the same pinned SDK plus `pytest==9.1.1` for its separate test job.
- [`test_app.py`](test_app.py) loads `app.py` by its path with a unique module name. Tests still run with the repository's `--import-mode=importlib` setting and do not depend on the live-only `app_client` fixture.
- The deployment job depends on the test job. Test steps never receive the deployment secret. The deploy step maps `FAL_DEPLOY_KEY` to `FAL_KEY` for noninteractive authentication, and fails if the key is missing.
- Deployments use the stable app name `ci-cd-demo`, explicit `--env`, `--strategy rolling`, and `--attach --yes`. A deployment failure fails the job rather than reporting success as soon as the deployment is submitted.

## Deploy

### 1. Use the exact workflow

Fork this repository and enable GitHub Actions in your fork. The workflow must remain at the repository root as `.github/workflows/deploy.yml`; GitHub does not discover workflows placed inside the example folder.

If you prefer a new repository, copy this entire example directory to `01-environments-containers/deploy-from-ci-cd/` and copy [`deploy.yml`](../../.github/workflows/deploy.yml) to `.github/workflows/deploy.yml`. Keep those paths unchanged, or update both the workflow's `defaults.run.working-directory` and its event path filters. The copied test file works without the rest of the examples repository. Use `main` as your production branch, or deliberately update the workflow's branch filter and deployment guard together.

### 2. Set up fal and GitHub

1. Create the fal `staging` environment in the account or team you will deploy to, following [Manage Environments](https://fal.ai/docs/documentation/deployment/manage-environments). fal `main` is the production target in this workflow.
2. Create a key with the `DEPLOY` preset using [Key-Based Authentication](https://fal.ai/docs/documentation/model-apis/authentication/key-based#permissions). Save its complete value as a GitHub Actions secret named `FAL_DEPLOY_KEY`.
3. The workflow uses GitHub deployment environments named `staging` and `production`. Create and configure them in repository settings if you want required reviewers or deployment branch restrictions. You can put `FAL_DEPLOY_KEY` in each environment instead of using a repository secret. These GitHub environments are approval and secret boundaries, not the fal environments themselves; no approval policy is assumed by the workflow.
4. Add the repository Actions variable `FAL_CI_CD_ENABLED` with value `true` only when you are ready for deployments. Leave it unset to run tests without touching fal.
5. Protect `main` with pull request review and a required **Test app** check before merging. In a larger repository, account for GitHub's behavior that required path-filtered checks can remain pending on unrelated changes; adjust the filters before requiring this check globally.

**Treat same-repository pull request authors as trusted deployers.** The `DEPLOY` preset can manage apps, secrets, and environments; it is not restricted to staging by the workflow's `--env` flag. Even though tests and dependency installation do not receive `FAL_KEY`, deploying the checked-out Python app executes its code with deployment credentials available. Review app and workflow changes accordingly, and configure GitHub environment protection for approvals where your team needs them. The workflow uses `pull_request`, never `pull_request_target`, and fork pull requests cannot enter the deployment job or receive its secret.

### 3. Test the pipeline in staging

Once the workflow exists on your default branch, open **Actions > deploy-ci-cd-demo > Run workflow** and select the branch to test. Manual runs always target staging, including when you select `main`. First leave the opt-in variable unset to exercise only local tests; enable it when you deliberately want the staging deployment.

A same-repository pull request affecting this example also tests and deploys to staging after opt-in. Every run targets the same `ci-cd-demo` app in the same fal `staging` environment. The next successful deployment overwrites that shared staging version; this is not a per-PR preview environment.

Deployments are serialized separately for fal `staging` and `main`, with `cancel-in-progress: false`. An active deployment is not canceled by a newer run. GitHub concurrency groups are not a FIFO queue: a newer pending run can replace an older pending run.

After the job finishes, open the app in the [fal Apps dashboard](https://fal.ai/dashboard/apps) and follow [Testing Your Own Apps](https://fal.ai/docs/documentation/model-apis/playground#your-own-apps). A green local test job alone is not evidence that the deployed endpoint works.

### 4. Merge to production

Merge the reviewed change into `main`. The push workflow tests that merged commit again, then deploys to fal `main` through the GitHub `production` environment. It redeploys the merged source; it does not promote or copy a staging revision. If you configure required reviewers on the GitHub environment, the deployment waits for that approval.

The underlying deployment command, run from this example folder with `FAL_KEY` supplied, is:

```bash
fal deploy app.py::CICDDemo --app-name ci-cd-demo --env main --strategy rolling --attach --yes
```

The workflow passes `--env staging` for pull requests and manual runs. No `fal auth login` step is needed in CI.

### Roll back

Use [Rollbacks & Revisions](https://fal.ai/docs/documentation/deployment/rollbacks) for the revision and rollout model. For this app, select a known-good production revision and roll back explicitly in `main`:

```bash
fal apps list-rev ci-cd-demo --env main
fal apps set-rev ci-cd-demo REVISION_ID --env main --strategy rolling
```

Replace `REVISION_ID` with an ID from the first command. Rolling back `main` does not change staging or the code in GitHub. Revert or fix the source before the next production deployment so it does not deploy the broken version again.

## Cost & tuning

- Local tests use no fal compute. Deployments and requests use real infrastructure and can incur charges.
- `machine_type = "S"` keeps this example on CPU. `keep_alive=60` keeps a runner warm for 60 seconds after a request; reduce it when warm responses are less important than idle cost.
- Set `FAL_CI_CD_ENABLED` to `false` to stop future workflow deployments. This does not delete or stop an already deployed app. Manage existing apps through the [deployment documentation](https://fal.ai/docs/documentation/deployment/manage-deployments).

## Next steps

- [Dev, staging, and production](../dev-staging-prod/): explore environment-scoped configuration and promotion.
- [CI/CD walkthrough](https://fal.ai/docs/examples/serverless/deploy-from-ci-cd): follow the accompanying documentation.
- [Test Models and Endpoints](https://fal.ai/docs/documentation/development/test-models-and-endpoints): add real infrastructure checks when your app needs them.
