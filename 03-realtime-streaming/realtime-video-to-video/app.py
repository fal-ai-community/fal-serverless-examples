"""Real-time video-to-video over WebRTC: YOLO detection boxes on a live stream."""

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import fal
from fal.exceptions import FieldException
from fal.toolkit import FAL_PERSISTENT_DIR
from fal.toolkit import Image as FalImage
from pydantic import BaseModel, Field, RootModel

if TYPE_CHECKING:
    import numpy as np

# Pinned release asset: the /data cache below can never go stale or drift.
WEIGHTS_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
)
WEIGHTS_PATH = FAL_PERSISTENT_DIR / "yolo" / "yolov8n-v8.3.0.pt"

# Smoke payload for CI (DetectInput.examples below) and for client.py;
# test_app.py asserts the "person" and "bus" labels, so another image breaks it.
SAMPLE_IMAGE = "https://ultralytics.com/images/bus.jpg"

# Without TURN secrets the app signals public STUN: direct peer connections
# work from most home/office networks; see README for when you need TURN.
STUN_SERVERS: list[dict] = [{"urls": ["stun:stun.l.google.com:19302"]}]
TURN_EXPIRY_SECONDS = 600


# --- signaling messages (the WebSocket protocol, both directions) -----------


class IceCandidate(BaseModel):
    # field names are camelCase on purpose: they mirror the WebRTC wire format
    candidate: str
    sdpMid: str | None = None
    sdpMLineIndex: int | None = None


class OfferInput(BaseModel):
    type: Literal["offer"]
    sdp: str


class IceCandidateInput(BaseModel):
    type: Literal["icecandidate"]
    candidate: IceCandidate | None = None


RealtimeInputMessage = Annotated[
    OfferInput | IceCandidateInput,
    Field(discriminator="type"),
]


class RealtimeInput(RootModel):
    root: RealtimeInputMessage


class IceServersOutput(BaseModel):
    type: Literal["iceservers"]
    iceservers: list[dict]


class AnswerOutput(BaseModel):
    type: Literal["answer"]
    sdp: str


class ErrorOutput(BaseModel):
    type: Literal["error"]
    error: str


RealtimeOutputMessage = IceServersOutput | AnswerOutput | ErrorOutput


class RealtimeOutput(RootModel):
    root: RealtimeOutputMessage


# --- HTTP detection endpoint models ------------------------------------------


class DetectInput(BaseModel):
    image_url: str = Field(
        description="Image to run detection on.",
        examples=[SAMPLE_IMAGE],
    )
    confidence: float = Field(
        default=0.25,
        ge=0.05,
        le=0.95,
        description="Minimum detection confidence to keep a box.",
    )


class Detection(BaseModel):
    label: str = Field(description="COCO class name, e.g. 'person'.")
    confidence: float = Field(description="Detection confidence, 0-1.")
    box: list[float] = Field(description="Pixel box [x1, y1, x2, y2].")


class DetectOutput(BaseModel):
    detections: list[Detection] = Field(description="Everything found in the image.")
    image: FalImage = Field(description="The input, annotated with detection boxes.")


def resolve_ice_servers(secret_key: str | None, label: str | None) -> list[dict]:
    """Metered TURN when both secrets are set; public STUN otherwise."""
    if not secret_key or not label:
        return [dict(server) for server in STUN_SERVERS]

    import json
    import urllib.parse
    import urllib.request

    # Metered's two-step flow: the long-lived secret mints a short-lived API
    # key, which returns time-boxed TURN credentials - the secret never
    # reaches a client.
    mint_url = (
        f"https://{label}.metered.live/api/v1/turn/credential?"
        + urllib.parse.urlencode({"secretKey": secret_key})
    )
    body = json.dumps({"expiryInSeconds": TURN_EXPIRY_SECONDS}).encode()
    mint_request = urllib.request.Request(
        mint_url, data=body, method="POST", headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(mint_request, timeout=5) as response:
        api_key = json.loads(response.read()).get("apiKey")
    if not api_key:
        raise RuntimeError("Metered credential response missing apiKey")

    list_url = (
        f"https://{label}.metered.live/api/v1/turn/credentials?"
        + urllib.parse.urlencode({"apiKey": api_key})
    )
    with urllib.request.urlopen(list_url, timeout=5) as response:
        raw_servers = json.loads(response.read())
    servers = [
        {
            "urls": item["urls"],
            "username": item.get("username"),
            "credential": item.get("credential", item.get("password")),
        }
        for item in raw_servers
        if item.get("urls")
    ]
    if not servers:
        raise RuntimeError("Metered returned an empty ICE server list")
    return servers


def draw_boxes(image: "np.ndarray", detections) -> "np.ndarray":
    """Draw label + box for each detection onto a BGR frame, in place."""
    import cv2

    for box in detections.boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        label = detections.names.get(int(box.cls[0]), "obj")
        text = f"{label} {float(box.conf[0]):.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            text,
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return image


def create_yolo_track(source_track, yolo_model):
    """Wrap an incoming video track: every frame comes back annotated."""
    from aiortc.mediastreams import MediaStreamTrack
    from av import VideoFrame

    def annotate(image: "np.ndarray") -> "np.ndarray":
        return draw_boxes(image, yolo_model(image, verbose=False)[0])

    class YoloTrack(MediaStreamTrack):
        kind = "video"

        async def recv(self):
            frame = await source_track.recv()
            image = frame.to_ndarray(format="bgr24")
            try:
                # Blocking CUDA call: run it off the event loop, which also
                # drives ICE keepalives, signaling, and outbound RTP pacing.
                annotated = await asyncio.to_thread(annotate, image)
            except Exception as exc:
                print(f"YOLO inference error: {exc}")
                annotated = image
            new_frame = VideoFrame.from_ndarray(annotated, format="bgr24")
            # Keep the source timestamps: the browser plays frames on the
            # original capture clock, so added latency shows up as lag, not drift.
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame

    return YoloTrack()


class RealtimeVideoToVideo(
    fal.App,
    name="realtime-video-to-video",
    keep_alive=300,  # realtime sessions come in bursts; stay warm between them
):
    machine_type = ["GPU-L40", "GPU-H100"]  # tried in order until capacity is found
    app_auth = "private"
    requirements = [
        "fal==1.78.2",
        "aiortc==1.15.0",
        "av==17.1.0",  # aiortc 1.15 caps av at <18
        "numpy==2.5.2",
        "opencv-python==4.14.0.94",
        "pillow==11.3.0",
        "torch==2.13.0",
        "ultralytics==8.4.127",
    ]

    def setup(self):
        import urllib.request

        import numpy as np
        from ultralytics import YOLO

        weights = Path(str(WEIGHTS_PATH))
        if not weights.is_file():
            import uuid

            weights.parent.mkdir(parents=True, exist_ok=True)
            # Unique staging name + atomic replace: /data is shared, so
            # concurrent cold boots may both download, but no boot can ever
            # see (or cache forever) a half-written file.
            partial = weights.with_name(f".{weights.name}.{uuid.uuid4().hex}.part")
            urllib.request.urlretrieve(WEIGHTS_URL, partial)
            partial.replace(weights)
        self.model = YOLO(str(weights))
        # Warmup: pay CUDA init + first-inference graph capture now, not on frame 1.
        self.model(np.zeros((480, 640, 3), dtype=np.uint8), verbose=False)

        self._turn_secret = os.getenv("METERED_TURN_SECRET_KEY")
        self._turn_label = os.getenv("METERED_TURN_LABEL")
        mode = "TURN (Metered)" if self._turn_secret and self._turn_label else "STUN"
        print(f"[realtime-video-to-video] model ready, ICE mode: {mode}")

    @fal.endpoint("/")
    def detect(self, request: DetectInput) -> DetectOutput:
        """One-shot detection over HTTP - same model the realtime track runs."""
        import cv2
        import numpy as np
        from PIL import Image as PILImage

        try:
            # url is the only required field; mypy misses the pydantic defaults
            pil = FalImage(url=request.image_url).to_pil().convert("RGB")  # type: ignore[call-arg]
        except Exception as exc:
            raise FieldException(
                field="image_url",
                message="Could not fetch or decode the image.",
                type="file_download_error",
            ) from exc

        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        detections = self.model(bgr, verbose=False, conf=request.confidence)[0]
        annotated = draw_boxes(bgr, detections)
        return DetectOutput(
            detections=[
                Detection(
                    label=detections.names.get(int(box.cls[0]), "obj"),
                    confidence=round(float(box.conf[0]), 4),
                    box=[round(v, 1) for v in box.xyxy[0].tolist()],
                )
                for box in detections.boxes
            ],
            image=FalImage.from_pil(
                PILImage.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            ),
        )

    @fal.realtime("/realtime", buffering=5)
    async def webrtc(
        self, inputs: AsyncIterator[RealtimeInput]
    ) -> AsyncIterator[RealtimeOutput]:
        """WebSocket signaling for one WebRTC session: the client streams video
        to the worker, the worker streams the annotated video back."""
        from aiortc import (
            RTCConfiguration,
            RTCIceServer,
            RTCPeerConnection,
            RTCSessionDescription,
        )
        from aiortc.contrib.media import MediaBlackhole
        from aiortc.sdp import candidate_from_sdp

        ice_servers = await asyncio.to_thread(
            resolve_ice_servers, self._turn_secret, self._turn_label
        )
        pc = RTCPeerConnection(
            RTCConfiguration(
                iceServers=[
                    RTCIceServer(
                        urls=server["urls"],
                        username=server.get("username"),
                        credential=server.get("credential"),
                    )
                    for server in ice_servers
                ]
            )
        )
        blackhole = MediaBlackhole()  # swallow non-video tracks (e.g. audio)
        stop_event = asyncio.Event()
        outgoing: asyncio.Queue[RealtimeOutput | None] = asyncio.Queue()
        background: set[asyncio.Task] = set()

        async def send(message: RealtimeOutputMessage) -> None:
            if not stop_event.is_set():
                await outgoing.put(RealtimeOutput(root=message))

        async def send_error(prefix: str, exc: Exception) -> None:
            await send(ErrorOutput(type="error", error=f"{prefix}: {exc}"))
            stop_event.set()
            await outgoing.put(None)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            # "disconnected" stays out: it is transient (brief packet loss) and
            # ICE recovers to "connected" on its own - tearing down loses the
            # session for good.
            if pc.connectionState in ("failed", "closed"):
                stop_event.set()
                await outgoing.put(None)

        @pc.on("track")
        def on_track(track) -> None:
            # Must stay synchronous: aiortc fires this during
            # setRemoteDescription, and the answer only includes the return
            # track if addTrack already happened when createAnswer runs.
            if track.kind == "video":
                pc.addTrack(create_yolo_track(track, self.model))
            else:
                blackhole.addTrack(track)
                task = asyncio.ensure_future(blackhole.start())
                background.add(task)
                task.add_done_callback(background.discard)

        async def handle_message(message: RealtimeInputMessage) -> None:
            if isinstance(message, OfferInput):
                try:
                    await pc.setRemoteDescription(
                        RTCSessionDescription(sdp=message.sdp, type=message.type)
                    )
                    await pc.setLocalDescription(await pc.createAnswer())
                    # aiortc gathers ICE inside setLocalDescription, so this
                    # answer already carries the worker's candidates - the
                    # server never trickles.
                    await send(AnswerOutput(type="answer", sdp=pc.localDescription.sdp))
                except Exception as exc:
                    await send_error("offer_failed", exc)
            else:
                try:
                    # End-of-candidates arrives as null OR as an empty
                    # candidate string (RFC 8838 trickle; Firefox sends both).
                    if message.candidate is None or not message.candidate.candidate:
                        await pc.addIceCandidate(None)
                        return
                    candidate = candidate_from_sdp(message.candidate.candidate)
                    candidate.sdpMid = message.candidate.sdpMid
                    candidate.sdpMLineIndex = message.candidate.sdpMLineIndex
                    await pc.addIceCandidate(candidate)
                except Exception as exc:
                    await send_error("ice_failed", exc)

        async def input_loop() -> None:
            try:
                async for message in inputs:
                    if stop_event.is_set():
                        break
                    await handle_message(message.root)
            except Exception as exc:
                # Anything a client can send wrong lands here (text frames,
                # bad msgpack, schema violations) - answer, don't crash.
                await send_error("invalid_payload", exc)
            finally:
                stop_event.set()
                await outgoing.put(None)

        input_task: asyncio.Task | None = None
        try:
            await send(IceServersOutput(type="iceservers", iceservers=ice_servers))
            input_task = asyncio.create_task(input_loop())
            while True:
                message = await outgoing.get()
                if message is None:
                    # Ending the generator ends the session: fal's runtime
                    # flushes every frame already yielded, then closes the
                    # socket (raising instead would drop queued frames).
                    return
                yield message
        finally:
            stop_event.set()
            if input_task is not None:
                input_task.cancel()
                # Never let a dead input loop's exception abort this cleanup:
                # pc.close() below must run or the worker keeps annotating.
                with suppress(asyncio.CancelledError, Exception):
                    await input_task
            await blackhole.stop()
            await pc.close()
