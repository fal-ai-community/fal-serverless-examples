"""Queue client for the deployed pipeline - the raw HTTP a backend really sends."""

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

    # fal_webhook must be URL-encoded
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

    # ?logs=1 includes the app's own log lines in the status body
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

    # no FAILED status: failures arrive COMPLETED; result URL gives non-2xx + error
    result = httpx.get(result_url, headers=headers, timeout=60)
    if not result.is_success:
        print(f"FAILED: HTTP {result.status_code}")
        print(result.text[:500])
        return 1
    print(json.dumps(result.json(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
