from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.grounding_service import GroundingDinoService

app = FastAPI(
    title="Warehouse GroundingDINO API",
    version="1.0.0",
    description="Open-vocabulary object detection for warehouse images.",
)
service = GroundingDinoService()

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Warehouse GroundingDINO API",
        "health": "/health",
        "documentation": "/docs",
    }


@app.get("/health")
def health() -> dict:
    status = service.status()
    ready = all(
        status[key]
        for key in (
            "grounding_dino_root_exists",
            "config_exists",
            "checkpoint_exists",
        )
    )
    return {"status": "ready" if ready else "not_ready", **status}


@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    prompt: str = Form("box . pallet . carton . bottle ."),
    box_threshold: float = Form(0.35),
    text_threshold: float = Form(0.25),
) -> dict:
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type: {image.content_type}",
        )
    if not 0.0 <= box_threshold <= 1.0:
        raise HTTPException(status_code=422, detail="box_threshold must be 0 to 1")
    if not 0.0 <= text_threshold <= 1.0:
        raise HTTPException(status_code=422, detail="text_threshold must be 0 to 1")

    image_bytes = await image.read(MAX_UPLOAD_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The uploaded image is too large")

    suffix = Path(image.filename or "image.jpg").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".jpg"

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(image_bytes)
            temp_path = Path(temp_file.name)

        detections, annotated_jpeg = service.detect(
            image_path=temp_path,
            prompt=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"GroundingDINO inference failed: {exc}",
        ) from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    return {
        "filename": image.filename,
        "prompt": service.normalize_prompt(prompt),
        "device": service.device,
        "count": len(detections),
        "detections": detections,
        "annotated_image_mime_type": "image/jpeg",
        "annotated_image_base64": base64.b64encode(annotated_jpeg).decode("ascii"),
    }
