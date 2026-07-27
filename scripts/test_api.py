from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the Warehouse GroundingDINO API")
    parser.add_argument("image", type=Path, help="Path to the image to detect")
    parser.add_argument(
        "--prompt",
        default="box . pallet . carton . bottle .",
        help="Objects to detect, separated by dots",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000/detect")
    parser.add_argument("--output", type=Path, default=Path("result.jpg"))
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    with args.image.open("rb") as image_file:
        response = httpx.post(
            args.url,
            files={"image": (args.image.name, image_file, "image/jpeg")},
            data={
                "prompt": args.prompt,
                "box_threshold": "0.35",
                "text_threshold": "0.25",
            },
            timeout=600.0,
        )

    response.raise_for_status()
    payload = response.json()
    args.output.write_bytes(base64.b64decode(payload["annotated_image_base64"]))

    print(json.dumps({
        "count": payload["count"],
        "device": payload["device"],
        "detections": payload["detections"],
        "annotated_image": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
