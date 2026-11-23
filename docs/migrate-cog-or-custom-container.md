# Migrating a Cog model or custom container to fal

> Mirrored from [the fal docs page on migrating a Cog model or custom container](https://fal.ai/docs/documentation/development/migrate-from-replicate). The docs page is the source of truth.

fal Serverless runs custom containers. You can build your app from your own Dockerfile, or you can run an existing container image from a public or private registry. A model that you packaged with Cog is a container image, so it can run on fal.

This guide shows three migration paths for a Cog model:

1. Run the existing Cog image without code changes.
2. Build from a Dockerfile, such as the Dockerfile that Cog generates.
3. Convert the Cog `Predictor` class to a `fal.App`.

Each path deploys with the `fal` CLI and gives you autoscaling, logs, and [app analytics](https://fal.ai/docs/documentation/serverless/observability/app-analytics). For other Docker servers that do not use Cog, see [Migrate an External Docker Server](https://fal.ai/docs/documentation/development/migrate-external-docker-server).

## Choose a Migration Path

| Path | What you change | Typical time | Use it when |
|------|-----------------|--------------|-------------|
| [1. Run the Cog image](#path-1-run-the-existing-cog-image) | A `pyproject.toml` entry only | Under an hour | You want the model live on fal quickly and keep the Cog API |
| [2. Build from a Dockerfile](#path-2-build-from-a-dockerfile) | The Dockerfile and a small `fal.App` | A few hours | You need system packages or a specific CUDA version |
| [3. Convert to `fal.App`](#path-3-convert-the-predictor-to-a-falapp) | `predict.py` becomes a `fal.App` class | Half a day to a day | You want typed inputs, CDN file outputs, and the full Playground |

The time includes the first image build and a test deployment. Large model weights add download time to the first build and the first runner start.

## Before You Start

1. Install the `fal` CLI and log in. See [Installation](https://fal.ai/docs/documentation/development/getting-started/installation).
2. Make sure your account has Serverless access. Request access on the [Serverless get-started page](https://fal.ai/dashboard/serverless-get-started).
3. Choose a GPU. See [Machine Types](https://fal.ai/docs/documentation/deployment/machine-types) for the list, for example `GPU-H100` or `GPU-B200`.

## Path 1: Run the Existing Cog Image

A Cog image contains its own HTTP server. On fal, an image-only app starts the container and sends requests to the port that you expose. You do not write a `fal.App` class.

### Build and Push the Image

Build the image with Cog and push it to a registry that fal can pull from:

```bash
cog build -t docker.io/my-org/my-cog-model:v1
docker push docker.io/my-org/my-cog-model:v1
```

If the registry is private, configure credentials. See [Private Docker Registries](https://fal.ai/docs/documentation/development/private-registries) for the registries fal supports and how to add credentials.

### Configure the App

Add the app to `pyproject.toml`. Do not set `ref` or `python_entry_point`. Without them, fal treats the container as an image-only app and does not load the fal Python runtime.

```toml
[tool.fal.apps.my-cog-model]
auth = "private"
machine_type = "GPU-H100"
exposed_port = 5000
keep_alive = 300

[tool.fal.apps.my-cog-model.image]
image = "docker.io/my-org/my-cog-model:v1"

[tool.fal.apps.my-cog-model.health_check]
path = "/health-check"
start_period_seconds = 600
```

The Cog image already defines the command that starts its server, so you do not set `cmd`. To override it, add `cmd` under `[tool.fal.apps.my-cog-model.image]`. See the [pyproject.toml reference](https://fal.ai/docs/api-reference/python-sdk/pyproject-toml#container-image).

### Test and Deploy

Run an ephemeral deployment to test the image:

```bash
fal run my-cog-model --auth private
```

> **Warning:** `fal run` ignores the `auth` value in `pyproject.toml` and defaults to `public`. Pass `--auth private` unless you want the test deployment to be public. `fal deploy` uses the configured value.

Deploy the app to a persistent URL:

```bash
fal deploy my-cog-model
```

### Call the App

fal forwards each request to the container without changes. The Cog API stays the same, so send the Cog request body to the `/predictions` path under your endpoint ID:

```bash
curl -X POST "https://queue.fal.run/<your-username>/my-cog-model/predictions" \
  -H "Authorization: Key $FAL_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "a sunset over mountains"}}'
```

The queue returns a `status_url` and a `response_url`. Poll the status URL, then fetch the result from the response URL. See [Calling Your Endpoints](https://fal.ai/docs/documentation/development/calling-your-endpoints) for queue, webhook, and streaming flows.

The Cog server returns files in the format that Cog uses, such as data URIs or paths. They do not go to the [fal CDN](https://fal.ai/docs/documentation/model-apis/fal-cdn). To return CDN URLs, use path 3.

## Path 2: Build From a Dockerfile

Use this path when the model needs system packages, a specific CUDA version, or build steps that a `requirements` list cannot express. fal builds the image from your Dockerfile and caches the result.

### Get a Dockerfile

If you already have a Dockerfile, use it. If you only have a `cog.yaml`, print the Dockerfile that Cog generates:

```bash
cog debug > Dockerfile
```

Then edit the generated Dockerfile:

1. Remove the `COPY . /src`, `EXPOSE`, and `CMD` lines at the end. fal starts the app.
2. Remove the Cog wheel installation (`cog-*.whl`). fal does not use the Cog runtime.
3. Replace the Cog requirements step with your pip packages.
4. Make sure that `curl` is installed. fal uses it when a runner starts.
5. Install `fal` as the last step. This gives compatible versions of `pydantic`, `protobuf`, and `boto3`.

> **Note:** `cog debug` is a hidden debugging command, and Cog does not guarantee its output format. The generated Dockerfile can change between Cog versions.

### Reference the Dockerfile From a `fal.App`

```python
import fal
from fal.container import ContainerImage
from fal.toolkit import Image
from pydantic import BaseModel, Field

class Input(BaseModel):
    prompt: str = Field(description="Text prompt")

class Output(BaseModel):
    image: Image

class MyApp(fal.App):
    machine_type = "GPU-H100"
    image = ContainerImage.from_dockerfile("Dockerfile")

    def setup(self):
        import torch
        from diffusers import StableDiffusionXLPipeline

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
        ).to("cuda")

    @fal.endpoint("/")
    def predict(self, input: Input) -> Output:
        image = self.pipe(input.prompt).images[0]
        return Output(image=Image.from_pil(image))
```

`ContainerImage.from_dockerfile()` reads a file from disk. To keep the Dockerfile in your Python file, use `ContainerImage.from_dockerfile_str()`. Both accept `build_args`, `secrets`, `registries`, and a build `context_dir` for `COPY`. See [Use a Custom Container Image](https://fal.ai/docs/documentation/development/use-custom-container-image) for all options and the build cache rules.

You can also keep the Dockerfile and its settings in `pyproject.toml`. Set `dockerfile = "Dockerfile"` under `[tool.fal.apps.<app-name>.image]`. See the [pyproject.toml reference](https://fal.ai/docs/api-reference/python-sdk/pyproject-toml#container-image).

## Path 3: Convert the Predictor to a `fal.App`

The Cog `Predictor` class maps directly to a `fal.App`. The `setup()` method stays the same. The `predict()` method becomes an `@fal.endpoint` method with Pydantic input and output models.

| Cog | fal | Notes |
|-----|-----|-------|
| `cog.yaml` | `requirements = [...]` or `ContainerImage` | Environment definition |
| `class Predictor(BasePredictor)` | `class MyApp(fal.App)` | App class |
| `def setup(self)` | `def setup(self)` | One-time model load per runner |
| `def predict(self, ...)` | `@fal.endpoint("/")` | Request handler |
| `cog.Path` (file output) | `fal.toolkit.Image` or `fal.toolkit.File` | fal uploads output files to the CDN |
| `Input(...)` type hints | Pydantic `BaseModel` | Standard Pydantic validation |
| `cog push` | `fal deploy` | One CLI command |
| Webhooks | `webhook_url` parameter | Both deliver results to a webhook |

#### Cog

```yaml
# cog.yaml
build:
  python_version: "3.11"
  python_packages:
    - torch==2.1.0
    - diffusers==0.30.0
    - transformers
    - accelerate
  gpu: true
predict: "predict.py:Predictor"
```

```python
# predict.py
from cog import BasePredictor, Input, Path
import torch
from diffusers import StableDiffusionXLPipeline

class Predictor(BasePredictor):
    def setup(self):
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
        ).to("cuda")

    def predict(self, prompt: str = Input(description="Text prompt")) -> Path:
        image = self.pipe(prompt).images[0]
        output_path = "/tmp/output.png"
        image.save(output_path)
        return Path(output_path)
```

#### fal

```python
import fal
from pydantic import BaseModel, Field
from fal.toolkit import Image

class Input(BaseModel):
    prompt: str = Field(description="Text prompt")

class Output(BaseModel):
    image: Image

class MyApp(fal.App):
    machine_type = "GPU-H100"
    requirements = ["torch==2.1.0", "diffusers==0.30.0", "transformers", "accelerate"]

    def setup(self):
        import torch
        from diffusers import StableDiffusionXLPipeline

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
        ).to("cuda")

    @fal.endpoint("/")
    def predict(self, input: Input) -> Output:
        image = self.pipe(input.prompt).images[0]
        return Output(image=Image.from_pil(image))
```

In the fal version, class attributes replace `cog.yaml`. `fal.toolkit.Image` replaces `cog.Path` and uploads the image to the [fal CDN](https://fal.ai/docs/documentation/model-apis/fal-cdn), so the response contains a URL. The imports are inside `setup()` so that they run on the runner, not on your machine. See [Serialization and Build](https://fal.ai/docs/documentation/development/app-lifecycle#serialization-and-build) for the reason.

Test the app on a temporary runner, then deploy it:

```bash
fal run my_app.py::MyApp
fal deploy my_app.py::MyApp
```

`fal run` executes `setup()` and your endpoints as production does. Import errors and model-load failures show in your terminal, not as a failed production deployment.

Call the deployed app:

```python
import fal_client

result = fal_client.subscribe("<your-username>/my-app", arguments={
    "prompt": "a sunset over mountains"
})
print(result["image"]["url"])
```

## What Changes After the Migration

| Area | Cog setup | On fal |
|------|-----------|--------|
| Build | `cog build` and `cog push` | fal builds from your Dockerfile or `requirements`, or pulls your existing image |
| Configuration | `cog.yaml` | `fal.App` class attributes or `pyproject.toml` |
| API shape | `POST /predictions` with an `input` object | Path 1 keeps it. Paths 2 and 3 use your Pydantic models |
| Output files | Data URIs or paths from the Cog server | Path 1 keeps them. Paths 2 and 3 return fal CDN URLs |
| Scaling | Set by the platform | You set `keep_alive`, `min_concurrency`, and `max_concurrency`. See [Scale Your Application](https://fal.ai/docs/documentation/deployment/scale-your-application) |
| Secrets | Platform environment variables | [fal secrets](https://fal.ai/docs/documentation/development/manage-secrets-securely) |
| Auth | Platform API token | fal API key (`Authorization: Key $FAL_KEY`) |

## How Billing Works on fal

fal bills Serverless apps per second of runner time, at the rate for the machine type. A runner is a container on a GPU. You pay while a runner is alive, not per prediction.

| Runner state | Billed |
|--------------|--------|
| PENDING (waiting for hardware) | No |
| DOCKER_PULL (pulling your image) | No |
| SETUP (running `setup()` or the server start) | Yes |
| IDLE (ready, including `keep_alive` time) | Yes |
| RUNNING (processing requests) | Yes |
| DRAINING and TERMINATING | Yes |

fal does not charge for requests that fail with an HTTP 5xx status. fal bills a multi-GPU runner as the GPU count multiplied by the duration. See [Serverless Pricing](https://fal.ai/docs/documentation/serverless/pricing) for the full rules, and [fal.ai/pricing](https://fal.ai/pricing) for the hourly rate of each GPU.

Three settings decide most of the cost:

- **`keep_alive`** keeps an idle runner warm for a number of seconds after the last request. A longer value reduces cold starts and adds IDLE time.
- **`min_concurrency`** keeps a minimum number of runners alive at all times. Set it to `0` to scale to zero, so that an app with no traffic has no compute charge.
- **`machine_type`** sets the per-second rate. Use the smallest GPU that fits the model.

To estimate the cost, add up the seconds that your runners are alive, not the number of predictions. [App Analytics](https://fal.ai/docs/documentation/serverless/observability/app-analytics) shows runner time and cost for each app.

## Test Your Deployment

Open your app from the [Apps dashboard](https://fal.ai/dashboard/apps). Follow [Testing Your Own Apps](https://fal.ai/docs/documentation/model-apis/playground#your-own-apps) to test an endpoint. Use the app's Logs and Requests pages to inspect recorded requests.

## Next Steps

- [App Lifecycle](https://fal.ai/docs/documentation/development/app-lifecycle) explains each stage, from code serialization to runner shutdown.
- [Use a Custom Container Image](https://fal.ai/docs/documentation/development/use-custom-container-image) covers every Dockerfile option.
- [Optimize Container Images](https://fal.ai/docs/documentation/serverless/optimizations/optimize-container-images) reduces build and start times.
- [Scale Your Application](https://fal.ai/docs/documentation/deployment/scale-your-application) explains the scaling settings.
