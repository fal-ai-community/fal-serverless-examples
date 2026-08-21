"""Queue client for the deployed pipeline - the raw HTTP a backend really sends.

Deliberately httpx-only (no fal_client) so every URL and header for calling a
PRIVATE serverless app through fal's queue is visible and verifiable:

    submit  POST https://queue.fal.run/{owner}/{app}[?fal_webhook=...]
    status  GET  https://queue.fal.run/{owner}/{app}/requests/{id}/status?logs=1
    result  GET  https://queue.fal.run/{owner}/{app}/requests/{id}

Statuses are IN_QUEUE -> IN_PROGRESS -> COMPLETED (there is no FAILED status:
failures arrive as COMPLETED with an `error`, and the result URL then returns
a non-2xx with the error body).

Usage:
    export FAL_KEY=<key_id>:<key_secret>
    python client.py --app <username>/post-production-pipeline \
        [--video-url URL] [--srt-file captions.srt] [--ladder 1080,720,480] \
        [--webhook https://your-host/fal-webhook]

Dependency: pip install httpx==0.28.1
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import httpx

QUEUE = "https://queue.fal.run"
DEFAULT_VIDEO = (
    "https://storage.googleapis.com/falserverless/example_outputs/pika/"
    "pika_i2v_v22_output.mp4"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--app", required=True, help="owner/app-name of your deploy")
    parser.add_argument("--video-url", default=DEFAULT_VIDEO)
    parser.add_argument(
        "--ladder", default="1080,720,480", help="comma-separated rung heights"
    )
    parser.add_argument(
        "--srt-file", default=None, help="local .srt sent as inline captions"
    )
    parser.add_argument(
        "--webhook", default=None, help="public URL fal POSTs the result to"
    )
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    args = parser.parse_args()

    fal_key = os.environ.get("FAL_KEY")
    if not fal_key:
        sys.exit("set FAL_KEY=<key_id>:<key_secret> (dashboard -> Keys)")
    headers = {"Authorization": f"Key {fal_key}", "Content-Type": "application/json"}

    payload = {
        "video_url": args.video_url,
        "ladder": [int(h) for h in args.ladder.split(",")],
    }
    if args.srt_file:
        payload["captions_srt"] = Path(args.srt_file).read_text(encoding="utf-8")

    # SUBMIT - the queue answers immediately with request handles; the
    # optional fal_webhook parameter must be URL-encoded.
    submit_url = f"{QUEUE}/{args.app}"
    if args.webhook:
        submit_url += "?" + urlencode({"fal_webhook": args.webhook})
    submitted = httpx.post(submit_url, json=payload, headers=headers, timeout=30)
    submitted.raise_for_status()
    handles = submitted.json()
    request_id = handles["request_id"]
    print(f"queued: request_id={request_id}")
    if args.webhook:
        print(
            f"fal will POST the result to {args.webhook} "
            "(signed; see webhook_receiver.py)"
        )
        print("polling too, as the fallback a real backend keeps:")

    # POLL - ?logs=1 streams the app's own log lines into the status body.
    status_url = f"{QUEUE}/{args.app}/requests/{request_id}/status"
    result_url = f"{QUEUE}/{args.app}/requests/{request_id}"
    seen_logs = 0
    last_status = ""
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        status = httpx.get(
            status_url, params={"logs": "1"}, headers=headers, timeout=30
        )
        status.raise_for_status()
        body = status.json()
        logs = body.get("logs") or []
        for line in logs[seen_logs:]:
            print(f"  [app] {line.get('message', line)}")
        seen_logs = len(logs)
        if body["status"] != last_status:
            last_status = body["status"]
            position = body.get("queue_position")
            print(
                f"  status: {last_status}"
                + (f" (queue position {position})" if position is not None else "")
            )
        if last_status == "COMPLETED":
            if body.get("error"):
                print(
                    "  completed with error: "
                    f"{body['error']} ({body.get('error_type')})"
                )
            break
        time.sleep(args.poll_interval)
    else:
        sys.exit(f"timed out after {args.timeout:.0f}s; check {status_url}")

    # RESULT - a failed request returns non-2xx here, with the error body.
    result = httpx.get(result_url, headers=headers, timeout=60)
    if not result.is_success:
        print(f"FAILED: HTTP {result.status_code}")
        print(result.text[:500])
        return 1
    print(json.dumps(result.json(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
