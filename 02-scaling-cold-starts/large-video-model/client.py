"""Usage: FAL_KEY=... python client.py <owner>/large-video-model "<prompt>"."""

import base64
import pathlib
import shutil
import sys
import urllib.request

import fal_client


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    app_id, prompt = sys.argv[1:3]
    for event in fal_client.stream(app_id, {"prompt": prompt}, path="/stream"):
        if event["type"] == "preview":
            data = base64.b64decode(event["video"]["url"].split(",", 1)[1])
            pathlib.Path("preview_latest.mp4").write_bytes(data)
            bar = "#" * round(20 * event["progress"]) + "-" * (
                20 - round(20 * event["progress"])
            )
            print(
                f"[{bar}] step {event['step'] + 1}/{event['total_steps']}"
                f" - preview_latest.mp4 updated ({len(data) // 1024} KB)"
            )
        elif event["type"] == "result":
            with (
                urllib.request.urlopen(event["video"]["url"], timeout=60) as resp,
                open("final.mp4", "wb") as out,
            ):
                shutil.copyfileobj(resp, out)
            print(f"done: final.mp4 (seed {event['seed']}, {event['timings']})")
        elif event["type"] == "error":
            print(f"generation failed ({event['status_code']}): {event['detail']}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
