from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _as_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _as_csv(name: str, default: str = "") -> tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str
    app_version: str
    environment: str
    host: str
    port: int
    log_level: str
    api_key: str | None
    max_upload_bytes: int
    max_image_pixels: int
    max_prompt_chars: int
    max_prompt_terms: int
    max_concurrent_requests: int
    result_dir: Path
    result_ttl_seconds: int
    public_base_url: str | None
    preload_model: bool
    include_base64_default: bool
    cors_origins: tuple[str, ...]
    allowed_hosts: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("API_KEY")
        public_base_url = os.getenv("PUBLIC_BASE_URL")
        return cls(
            app_name=os.getenv("APP_NAME", "Warehouse GroundingDINO API"),
            app_version=os.getenv("APP_VERSION", "2.0.0"),
            environment=os.getenv("APP_ENV", "development"),
            host=os.getenv("HOST", "0.0.0.0"),
            port=_as_int("PORT", 8000),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            api_key=api_key.strip() if api_key and api_key.strip() else None,
            max_upload_bytes=_as_int("MAX_UPLOAD_BYTES", 15 * 1024 * 1024),
            max_image_pixels=_as_int("MAX_IMAGE_PIXELS", 40_000_000),
            max_prompt_chars=_as_int("MAX_PROMPT_CHARS", 500),
            max_prompt_terms=_as_int("MAX_PROMPT_TERMS", 30),
            max_concurrent_requests=_as_int("MAX_CONCURRENT_REQUESTS", 1),
            result_dir=Path(
                os.getenv("RESULT_DIR", PROJECT_ROOT / "runtime" / "results")
            ).resolve(),
            result_ttl_seconds=_as_int("RESULT_TTL_SECONDS", 24 * 60 * 60),
            public_base_url=(
                public_base_url.rstrip("/")
                if public_base_url and public_base_url.strip()
                else None
            ),
            preload_model=_as_bool("PRELOAD_MODEL", False),
            include_base64_default=_as_bool("INCLUDE_BASE64_DEFAULT", False),
            cors_origins=_as_csv("CORS_ORIGINS"),
            allowed_hosts=_as_csv("ALLOWED_HOSTS", "*"),
        )


settings = Settings.from_env()
