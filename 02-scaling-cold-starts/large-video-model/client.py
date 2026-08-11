"""Watch a generation live from the terminal.

Usage:
    pip install fal-client==1.0.0
    export FAL_KEY=...
    python client.py <owner>/large-video-model "a red fox trotting through fresh snow"

Writes preview_latest.mp4 as each preview arrives (open it in a looping
player) and final.mp4 (with audio) when generation completes.
"""
import base64
import pathlib
import sys
import urllib.request

import fal_client


def main() -> None:
    app_id, prompt = sys.argv[1], sys.argv[2]
    for event in fal_client.stream(app_id, {"prompt": prompt}, path="/stream"):
        if event["type"] == "preview":
            data = base64.b64decode(event["video"].split(",", 1)[1])
            pathlib.Path("preview_latest.mp4").write_bytes(data)
            bar = "#" * round(20 * event["progress"]) + "-" * (20 - round(20 * event["progress"]))
            print(f"[{bar}] step {event['step'] + 1}/{event['total_steps']}"
                  f" - preview_latest.mp4 updated ({len(data) // 1024} KB)")
        elif event["type"] == "result":
            urllib.request.urlretrieve(event["video"]["url"], "final.mp4")
            print(f"done: final.mp4 (seed {event['seed']}, {event['timings']})")
        elif event["type"] == "error":
            print(f"generation failed ({event['status_code']}): {event['detail']}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
