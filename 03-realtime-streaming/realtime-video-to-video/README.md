# Real-time video-to-video: YOLO boxes on your live webcam

Queue-and-poll APIs are the wrong shape for live video: by the time a request
round-trips, the frame it described is gone. This example serves a
video-to-video pipeline at interactive latency - your browser streams webcam
frames to a fal GPU over WebRTC, YOLO draws detection boxes on every frame,
and the annotated stream comes back on the same connection. The fal endpoint
(`@fal.realtime`) is the WebRTC *signaling* channel; the video itself travels
over SRTP, a transport built for lossy real-time media. YOLO is a
stand-in brain: any model that maps frame → frame drops into the same
skeleton.

## What you get

In the browser: two video panes - your webcam, and the same stream with
detection boxes tracking objects live, with FPS/bitrate counters.

The same model also answers one-shot HTTP requests (this is what CI smokes):

```json
{
  "detections": [
    {"label": "bus", "confidence": 0.87, "box": [22.9, 231.3, 805.0, 756.9]},
    {"label": "person", "confidence": 0.87, "box": [48.6, 398.6, 245.3, 902.7]},
    {"label": "person", "confidence": 0.85, "box": [669.5, 392.2, 809.7, 877.0]},
    {"label": "person", "confidence": 0.83, "box": [221.5, 405.8, 345.0, 857.5]}
  ],
  "image": {"url": "https://v3b.fal.media/files/…/annotated.png"}
}
```

## Run it

```bash
pip install fal==1.78.2   # keep pinned to this example's requirements.txt
fal auth login
fal run app.py::RealtimeVideoToVideo
```

Then open the bundled viewer against the printed URL:

```bash
python -m http.server 8000   # serve viewer.html next to this README
```

Open `http://localhost:8000/viewer.html`, paste the app URL + a FAL key
(local dev only - the key mints a 120-second JWT in the page; only the JWT
goes on the wire), hit **Start**, allow the webcam.

No webcam or on a server? The headless client streams a synthetic pan over a
street photo and reports round-trip FPS (it reuses the app env's
`requirements.txt`, which already carries aiortc):

```bash
python client.py --url <printed app URL> --seconds 20 --save-dir frames/
```

## How it works

Four ideas, in the order app.py builds them:

1. **One model, two transports.** `/` is a plain HTTP endpoint: one image in,
   detections + annotated image out. `/realtime` is the WebRTC session. Both
   run the same loaded YOLO instance, so the cheap HTTP path doubles as the
   deploy smoke test - CI proves the model loads and detects before any
   WebRTC session exists.
2. **The realtime endpoint is a signaling channel, not a media pipe.**
   `@fal.realtime("/realtime")` gives you a bidirectional WebSocket that
   speaks msgpack-encoded frames (that is fal's realtime wire format - send
   binary msgpack, not JSON text). The message protocol is five small
   pydantic models: the server sends `iceservers` on connect, the client
   sends an `offer` (SDP) and trickled `icecandidate`s, the server replies
   with an `answer` (or an `error`). After that the WebSocket goes quiet -
   video never touches it.
3. **NAT reality: STUN by default, TURN when it matters.** Both peers (your
   browser and the fal worker) sit behind NAT. With plain STUN, hole-punching
   works from most home/office networks; symmetric NATs need a TURN relay.
   Without configuration the app signals public STUN and works when the
   network allows; for production reliability, add free-tier
   [Metered](https://www.metered.ca/stun-turn) TURN credentials -
   `resolve_ice_servers()` then mints short-lived TURN credentials per
   session, so the long-lived secret never reaches a client:

   ```bash
   fal secrets set METERED_TURN_SECRET_KEY "…"
   fal secrets set METERED_TURN_LABEL "your-metered-subdomain"
   ```

4. **The pipeline is a track wrapper.** aiortc hands the incoming video as a
   `MediaStreamTrack`; `create_yolo_track` wraps it with a `recv()` that
   pulls a frame, runs YOLO, draws boxes, and returns the annotated frame
   with the *source* timestamps - so added processing time shows up as lag
   (which WebRTC's jitter buffer absorbs), never as clock drift. One aiortc
   quirk worth knowing: the server does not trickle ICE - `setLocalDescription`
   gathers first, so the `answer` SDP already carries the worker's candidates.

   Inference runs in a worker thread (`asyncio.to_thread`), because a blocking
   GPU call on the event loop also stalls ICE keepalives, signaling, and
   outbound packet pacing. The wrapper still annotates every frame it gets. If
   inference is slower than capture, frames queue and latency grows for as long
   as the session lasts. A production pipeline drops stale frames: it keeps the
   newest frame and skips the rest. That needs aiortc queue internals, so this
   example does not do it.

Concept docs: [Realtime Endpoints](https://fal.ai/docs/documentation/development/realtime) ·
[Real-Time Inference (clients)](https://fal.ai/docs/documentation/model-apis/inference/real-time) ·
[handling inputs & outputs](https://fal.ai/docs/documentation/development/handle-inputs-and-outputs)

## Deploy

```bash
fal deploy app.py
```

The realtime endpoint is `wss://fal.run/<username>/realtime-video-to-video/realtime`;
paste `<username>/realtime-video-to-video` into the viewer, or point the
headless client at `https://fal.run/<username>/realtime-video-to-video`.
Browsers authenticate with a short-lived JWT minted from
`POST https://rest.alpha.fal.ai/tokens/` (the viewer shows the pattern);
server-side clients just send the usual `Authorization: Key` header.

## Cost & tuning

Measured on this example's CI-style ephemeral deploy (2026-08-24): with the
env already built and weights cached on `/data`, cold start - scheduling,
container boot, `setup()` with model load + warmup - measured **≈114s to
healthy** (the very first deploy also pip-builds a torch-sized env before
any runner starts, which takes several minutes more). A warm one-shot `/`
detection round-trips in **≈1.5s** end to end from a dev box - that is
image fetch + inference + annotation + CDN upload, not inference alone.

One honest NAT data point: from a WSL2 dev box (double NAT), signaling
succeeded but STUN-only media did not flow - exactly the case TURN
credentials exist for. Browsers on ordinary home networks usually punch
through; corporate networks and VM setups usually do not.

YOLOv8-nano is a ~6 MB model that barely registers on a datacenter GPU - for
this workload you are paying for *session time*, not compute intensity. That
changes the tuning math:

| Knob | Tradeoff |
|---|---|
| `keep_alive=300` | A realtime app is interactive: users reconnect in bursts. Five idle minutes costs a few cents on L40; a cold start mid-demo costs the demo. |
| `machine_type = ["GPU-L40", "GPU-H100"]` | yolov8n saturates neither; the list is a capacity fallback, not a performance choice. Swap in a heavier per-frame model (style transfer, restoration) and this becomes the knob that sets your frame rate. |
| `buffering=5` | Outgoing WebSocket queue depth. Signaling is bursty (a handful of ICE messages) and tiny; the default is fine for signaling-rate traffic. |
| `max_multiplexing` | Concurrent sessions *per runner* - each WebRTC session occupies one slot for its whole duration. yolov8n leaves GPU headroom for several; a heavier per-frame model wants `max_multiplexing = 1` and more runners (`max_concurrency`) instead. |
| `TURN_EXPIRY_SECONDS = 600` | Lifetime of the per-session TURN credentials; longer sessions need a refresh or a longer expiry. |

## Next steps

- Swap the brain: replace YOLO in `create_yolo_track` with any frame→frame
  model - a fast style-transfer or restoration net turns this into live
  generative video, no signaling changes needed.
- [`3d-progressive-rendering/`](../../04-genmedia-patterns/3d-progressive-rendering/)
  streams *progress* instead of media - SSE is the right tool when results
  arrive in seconds, not 30 times a second.

---
YOLOv8 weights and the `ultralytics` package are
[AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE);
fine for this open example, but check the license (or Ultralytics'
commercial terms) before shipping it inside a closed product.
