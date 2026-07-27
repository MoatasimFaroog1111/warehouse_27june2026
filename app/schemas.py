from __future__ import annotations

from pydantic import BaseModel, Field


class Detection(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    box_xyxy: list[float] = Field(min_length=4, max_length=4)


class ImageMetadata(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    format: str


class DetectionResponse(BaseModel):
    request_id: str
    filename: str | None
    prompt: str
    device: str
    processing_time_ms: float
    count: int = Field(ge=0)
    counts_by_label: dict[str, int]
    average_confidence_by_label: dict[str, float]
    image: ImageMetadata
    detections: list[Detection]
    result_id: str
    result_url: str
    annotated_image_mime_type: str = "image/jpeg"
    annotated_image_base64: str | None = None


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str
    uptime_seconds: float
    device: str
    model_loaded: bool
    grounding_dino_root_exists: bool
    config_exists: bool
    checkpoint_exists: bool


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody
