from __future__ import annotations

import argparse
import json
import mimetypes
from pathlib import Path
from urllib.parse import urljoin

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
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    mime_type = mimetypes.guess_type(args.image.name)[0] or "application/octet-stream"
    headers = {"X-API-Key": args.api_key} if args.api_key else {}

    with args.image.open("rb") as image_file:
        response = httpx.post(
            args.url,
            headers=headers,
            files={"image": (args.image.name, image_file, mime_type)},
            data={
                "prompt": args.prompt,
                "box_threshold": "0.35",
                "text_threshold": "0.25",
                "include_image_base64": "false",
            },
            timeout=600.0,
        )

    response.raise_for_status()
    payload = response.json()

    result_url = payload["result_url"]
    if result_url.startswith("/"):
        result_url = urljoin(args.url, result_url)
    image_response = httpx.get(result_url, headers=headers, timeout=60.0)
    image_response.raise_for_status()
    args.output.write_bytes(image_response.content)

    print(
        json.dumps(
            {
                "request_id": payload["request_id"],
                "count": payload["count"],
                "counts_by_label": payload["counts_by_label"],
                "average_confidence_by_label": payload[
                    "average_confidence_by_label"
                ],
                "processing_time_ms": payload["processing_time_ms"],
                "device": payload["device"],
                "detections": payload["detections"],
                "annotated_image": str(args.output.resolve()),
                "result_url": payload["result_url"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
