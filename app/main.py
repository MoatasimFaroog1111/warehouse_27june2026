from __future__ import annotations

import asyncio
import base64
import logging
import tempfile
import time
import uuid
import warnings
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Annotated

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    Security,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import settings
from app.grounding_service import GroundingDinoService
from app.logging_config import configure_logging
from app.schemas import DetectionResponse, ErrorResponse, HealthResponse, ImageMetadata
from app.storage import ResultStorage

configure_logging(settings.log_level)
logger = logging.getLogger("warehouse_api")
started_at = time.monotonic()

service = GroundingDinoService(
    max_prompt_chars=settings.max_prompt_chars,
    max_prompt_terms=settings.max_prompt_terms,
)
storage = ResultStorage(settings.result_dir, settings.result_ttl_seconds)
inference_slots = asyncio.Semaphore(settings.max_concurrent_requests)

Image.MAX_IMAGE_PIXELS = settings.max_image_pixels


@asynccontextmanager
async def lifespan(_: FastAPI):
    removed = storage.cleanup()
    logger.info("service_started", extra={"count": removed, "device": service.device})
    if settings.preload_model:
        await asyncio.to_thread(service.get_model)
        logger.info("model_preloaded", extra={"device": service.device})
    yield
    logger.info("service_stopped")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-oriented open-vocabulary warehouse object detection.",
    lifespan=lifespan,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)

app.add_middleware(GZipMiddleware, minimum_size=1024)
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
    )
if settings.allowed_hosts != ("*",):
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(settings.allowed_hosts),
    )


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", uuid.uuid4().hex)


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": _request_id(request),
            }
        },
    )


@app.middleware("http")
async def request_context(request: Request, call_next):
    incoming = request.headers.get("X-Request-ID", "").strip()
    request.state.request_id = (
        incoming[:128] if incoming and incoming.isprintable() else uuid.uuid4().hex
    )
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "unhandled_request_error",
            extra={
                "request_id": request.state.request_id,
                "method": request.method,
                "path": request.url.path,
            },
        )
        return _error_response(
            request,
            500,
            "internal_error",
            "An unexpected server error occurred.",
        )

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-Process-Time-Ms"] = str(duration_ms)
    logger.info(
        "request_completed",
        extra={
            "request_id": request.state.request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, str) else "The request was rejected."
    code_by_status = {
        400: "invalid_request",
        401: "unauthorized",
        404: "not_found",
        413: "upload_too_large",
        415: "unsupported_media_type",
        422: "validation_error",
        503: "service_unavailable",
    }
    return _error_response(
        request,
        exc.status_code,
        code_by_status.get(exc.status_code, "request_error"),
        detail,
    )


@app.exception_handler(RequestValidationError)
async def request_validation_handler(request: Request, _: RequestValidationError):
    return _error_response(
        request,
        422,
        "validation_error",
        "Request validation failed.",
    )


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    x_api_key: str | None = Security(api_key_header),
) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="A valid API key is required.")


def _validate_and_normalize_image(image_bytes: bytes) -> tuple[bytes, ImageMetadata]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as opened:
                opened.load()
                normalized_source = ImageOps.exif_transpose(opened)
                width, height = normalized_source.size
                image_format = (opened.format or "unknown").upper()
                if width * height > settings.max_image_pixels:
                    raise ValueError(
                        f"The image exceeds the {settings.max_image_pixels}-pixel limit."
                    )
                if image_format not in {"JPEG", "PNG", "WEBP", "BMP"}:
                    raise ValueError(f"Unsupported image format: {image_format}")
                rgb = normalized_source.convert("RGB")
                normalized = BytesIO()
                rgb.save(normalized, format="JPEG", quality=95, optimize=True)
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=415,
            detail=f"The uploaded file is not a supported image: {exc}",
        ) from exc

    return (
        normalized.getvalue(),
        ImageMetadata(width=width, height=height, format=image_format),
    )


def _result_url(request: Request, result_id: str) -> str:
    if settings.public_base_url:
        return f"{settings.public_base_url}/results/{result_id}"
    return f"/results/{result_id}"


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "live": "/live",
        "ready": "/ready",
        "health": "/health",
        "documentation": "/docs",
        "detection": "/detect",
    }


@app.get("/live")
def live() -> dict[str, str]:
    return {"status": "alive"}


def _health_payload() -> HealthResponse:
    status = service.status()
    ready = all(
        status[key]
        for key in (
            "grounding_dino_root_exists",
            "config_exists",
            "checkpoint_exists",
        )
    )
    return HealthResponse(
        status="ready" if ready else "not_ready",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        uptime_seconds=round(time.monotonic() - started_at, 2),
        **status,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return _health_payload()


@app.get("/ready", response_model=HealthResponse)
def ready():
    payload = _health_payload()
    if payload.status != "ready":
        return JSONResponse(status_code=503, content=payload.model_dump())
    return payload


@app.get(
    "/results/{result_id}",
    name="get_result",
    responses={200: {"content": {"image/jpeg": {}}}},
)
def get_result(
    result_id: str,
    _: Annotated[None, Depends(require_api_key)],
):
    path = storage.get_jpeg(result_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Detection result not found.")
    return FileResponse(
        path,
        media_type="image/jpeg",
        filename=f"{result_id}.jpg",
        headers={"Cache-Control": "private, max-age=300"},
    )


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    request: Request,
    _: Annotated[None, Depends(require_api_key)],
    image: UploadFile = File(...),
    prompt: str = Form("box . pallet . carton . bottle ."),
    box_threshold: float = Form(0.35),
    text_threshold: float = Form(0.25),
    include_image_base64: bool = Form(settings.include_base64_default),
) -> DetectionResponse:
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type: {image.content_type}",
        )
    if not 0.0 <= box_threshold <= 1.0:
        raise HTTPException(status_code=422, detail="box_threshold must be 0 to 1.")
    if not 0.0 <= text_threshold <= 1.0:
        raise HTTPException(status_code=422, detail="text_threshold must be 0 to 1.")

    image_bytes = await image.read(settings.max_upload_bytes + 1)
    await image.close()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(image_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"The upload exceeds the {settings.max_upload_bytes}-byte limit.",
        )

    normalized_jpeg, metadata = await asyncio.to_thread(
        _validate_and_normalize_image,
        image_bytes,
    )

    temp_path: Path | None = None
    started = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file.write(normalized_jpeg)
            temp_path = Path(temp_file.name)

        async with inference_slots:
            (
                detections,
                counts_by_label,
                average_confidence,
                annotated_jpeg,
                normalized_prompt,
            ) = await asyncio.to_thread(
                service.detect,
                temp_path,
                prompt,
                box_threshold,
                text_threshold,
            )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "inference_failed",
            extra={"request_id": _request_id(request), "device": service.device},
        )
        raise HTTPException(
            status_code=500,
            detail="GroundingDINO inference failed.",
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    result_id, _ = await asyncio.to_thread(storage.save_jpeg, annotated_jpeg)
    processing_time_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "detection_completed",
        extra={
            "request_id": _request_id(request),
            "count": len(detections),
            "duration_ms": processing_time_ms,
            "device": service.device,
        },
    )

    return DetectionResponse(
        request_id=_request_id(request),
        filename=image.filename,
        prompt=normalized_prompt,
        device=service.device,
        processing_time_ms=processing_time_ms,
        count=len(detections),
        counts_by_label=counts_by_label,
        average_confidence_by_label=average_confidence,
        image=metadata,
        detections=detections,
        result_id=result_id,
        result_url=_result_url(request, result_id),
        annotated_image_base64=(
            base64.b64encode(annotated_jpeg).decode("ascii")
            if include_image_base64
            else None
        ),
    )
