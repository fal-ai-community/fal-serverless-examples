"""Generate the paper-diorama style training set with klein itself.

16 subjects x 1 fixed style suffix x fixed seeds -> 16 images at 1024px,
each with a sidecar .txt caption describing CONTENT ONLY (the style lives
in the trigger phrase at training time, so it transfers to new subjects).

Runs anywhere with a FAL_KEY (hosted endpoint) - or point --endpoint at
your own 00.2 deployment (e.g. <username>/first-image-api).

    pip install fal-client
    python make_dataset.py --out paper-diorama-16.zip
"""
import argparse
import io
import zipfile
from pathlib import Path

import fal_client
import httpx

DEFAULT_ENDPOINT = "fal-ai/flux-2-klein/4b"
STYLE_SUFFIX = (
    ", handcrafted paper-cutout diorama, layered colored cardstock, "
    "visible paper texture, soft studio lighting, shallow depth"
)
SUBJECTS = [
    "a lighthouse on a rocky island at dusk",
    "a red fox in a snowy forest",
    "a sailboat on a calm turquoise sea",
    "a hot air balloon over rolling hills",
    "a cottage with a smoking chimney in a meadow",
    "a whale breaching next to a fishing boat",
    "a desert canyon with a winding river",
    "a night market street with lanterns",
    "a mountain climber on a snowy ridge",
    "an owl perched on a birch branch",
    "a tram crossing a city bridge at sunset",
    "a diver exploring a coral reef",
    "a windmill among tulip fields",
    "a campsite with a tent under the aurora",
    "a lion resting under an acacia tree",
    "a rocket launching from a jungle clearing",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--out", default="paper-diorama-16.zip")
    args = parser.parse_args()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for i, subject in enumerate(SUBJECTS):
            result = fal_client.subscribe(
                args.endpoint,
                arguments={
                    "prompt": subject + STYLE_SUFFIX,
                    "image_size": {"width": 1024, "height": 1024},
                    "num_inference_steps": 4,
                    "seed": 1000 + i,
                },
            )
            url = result["images"][0]["url"] if "images" in result else result["image"]["url"]
            png = httpx.get(url, timeout=120).content
            zf.writestr(f"{i:03d}.png", png)
            zf.writestr(f"{i:03d}.txt", subject)
            print(f"[{i + 1}/{len(SUBJECTS)}] {subject}")
    Path(args.out).write_bytes(buf.getvalue())
    print(f"wrote {args.out} ({len(buf.getvalue()) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
