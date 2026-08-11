# Cut cold starts by Nx: FlashPack + /data + compiled caches

<!-- Section order is the quality bar (design doc S4-A). Keep it. -->

One paragraph: the problem this solves and who hits it.

## What you get

<!-- OUTPUT FIRST: show the result before any code — image/video/audio sample or response JSON. -->

## Run it

```bash
pip install fal==1.78.2   # keep pinned to this example's requirements.txt
fal auth login
fal run app.py        # local iteration against real GPUs
```

## How it works

<!-- Walk the interesting parts of app.py: setup()/weights, endpoint I/O schema, machine_type choice. Link concept docs rather than re-explaining them. -->

## Deploy

```bash
fal deploy app.py
```

<!-- Playground link + curl / fal_client / @fal-ai/client call against the deployed endpoint. -->

## Cost & tuning

<!-- keep_alive / concurrency / machine-type tradeoffs specific to this example. -->

## Next steps

<!-- 2-3 links: the next example in the journey + relevant concept docs. -->
