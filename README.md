# warehouse_27june2026

Production-oriented warehouse object-detection API powered by GroundingDINO.

## Current capabilities

- Open-vocabulary detection through `POST /detect`.
- Counts grouped by detected label.
- Average confidence grouped by label.
- Pixel coordinates in `xyxy` format.
- Annotated results saved as JPEG files and returned through short result URLs.
- Optional Base64 image output for compatibility.
- Optional `X-API-Key` protection shown in Swagger's **Authorize** button.
- Upload-size, image-pixel, prompt-length, and prompt-term limits.
- Real image decoding and normalization before inference.
- One reusable model instance with bounded inference concurrency.
- JSON application logs with request IDs and processing times.
- Liveness, readiness, and health endpoints.
- Automatic expiration and cleanup of saved result images.
- CPU fallback when CUDA is unavailable.

## Project layout

```text
app/
  config.py               Environment configuration
  grounding_service.py    GroundingDINO model wrapper
  logging_config.py       JSON logging
  main.py                 FastAPI endpoints and middleware
  schemas.py              Response models
  storage.py              Result-image storage and cleanup
scripts/
  run_api.sh              Production-style launcher
  test_api.py             End-to-end smoke test
GroundingDINO/             Local upstream clone, ignored by Git
runtime/results/           Generated images, ignored by Git
requirements-api.txt
.env.example
```

## Update and install

From `/workspaces/warehouse_27june2026`:

```bash
git pull origin main
source GroundingDINO/.venv/bin/activate
python -m pip install -r requirements-api.txt
```

Copy the configuration template:

```bash
cp -n .env.example .env
```

For Codespaces development, an empty `API_KEY` keeps the API open. Before exposing the API publicly, set a strong secret:

```text
API_KEY=replace-with-a-long-random-secret
APP_ENV=production
```

## Run

```bash
bash scripts/run_api.sh
```

The launcher activates `GroundingDINO/.venv`, loads `.env`, and starts one Uvicorn worker. Keep `WEB_CONCURRENCY=1` unless the machine has enough memory to load a separate model in every worker.

Codespaces forwards port `8000`. Open the forwarded URL and append:

```text
/docs
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/live` | Process liveness |
| `GET` | `/ready` | Dependency readiness; returns 503 when model files are missing |
| `GET` | `/health` | Detailed health and model status |
| `POST` | `/detect` | Run object detection |
| `GET` | `/results/{result_id}` | Download an annotated result |
| `GET` | `/docs` | Swagger UI |

When `API_KEY` is configured, click **Authorize** in Swagger and enter the key. Programmatic clients must send:

```text
X-API-Key: your-secret
```

## Detection request

```bash
curl -X POST http://127.0.0.1:8000/detect \
  -H 'X-API-Key: your-secret' \
  -F 'image=@GroundingDINO/.asset/cat_dog.jpeg' \
  -F 'prompt=cat . dog .' \
  -F 'box_threshold=0.35' \
  -F 'text_threshold=0.25' \
  -F 'include_image_base64=false'
```

Example response structure:

```json
{
  "request_id": "7cb2fdfe50e84bdb9a230941fe7aeeca",
  "filename": "cat_dog.jpeg",
  "prompt": "cat . dog .",
  "device": "cpu",
  "processing_time_ms": 25137.42,
  "count": 2,
  "counts_by_label": {
    "cat": 1,
    "dog": 1
  },
  "average_confidence_by_label": {
    "cat": 0.8206,
    "dog": 0.5658
  },
  "image": {
    "width": 1200,
    "height": 900,
    "format": "JPEG"
  },
  "detections": [
    {
      "label": "cat",
      "confidence": 0.8206,
      "box_xyxy": [0.59, 319.48, 562.74, 877.14]
    }
  ],
  "result_id": "18a34c2f20ea44e2aa3c42e60bf11e2d",
  "result_url": "/results/18a34c2f20ea44e2aa3c42e60bf11e2d",
  "annotated_image_mime_type": "image/jpeg",
  "annotated_image_base64": null
}
```

`include_image_base64=false` is recommended because it keeps responses small. Use `result_url` to retrieve the annotated JPEG.

## Smoke test

Run the API in one terminal, then execute in a second terminal:

```bash
cd /workspaces/warehouse_27june2026
source GroundingDINO/.venv/bin/activate
python scripts/test_api.py GroundingDINO/.asset/cat_dog.jpeg \
  --prompt "cat . dog ." \
  --output result.jpg
```

When `API_KEY` is enabled:

```bash
python scripts/test_api.py GroundingDINO/.asset/cat_dog.jpeg \
  --prompt "cat . dog ." \
  --api-key 'your-secret' \
  --output result.jpg
```

The script prints the totals and downloads the annotated result.

## Suggested warehouse prompts

```text
box . pallet . carton . bottle .
forklift . worker . safety helmet .
damaged box . open carton .
wooden pallet . plastic pallet . cardboard box .
```

GroundingDINO is open-vocabulary, so prompt wording affects results. Keep object names short, concrete, and separated by dots.

## Environment variables

| Variable | Default | Meaning |
|---|---:|---|
| `API_KEY` | empty | Optional API authentication |
| `APP_ENV` | `development` | Environment label |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | API port |
| `LOG_LEVEL` | `INFO` | Application log level |
| `GROUNDING_DINO_DEVICE` | automatic | `cpu` or `cuda` |
| `PRELOAD_MODEL` | `false` | Load the model during startup |
| `MAX_UPLOAD_BYTES` | `15728640` | Maximum upload size |
| `MAX_IMAGE_PIXELS` | `40000000` | Decompression-bomb limit |
| `MAX_PROMPT_CHARS` | `500` | Prompt character limit |
| `MAX_PROMPT_TERMS` | `30` | Maximum object terms |
| `MAX_CONCURRENT_REQUESTS` | `1` | Requests admitted to inference |
| `RESULT_DIR` | `runtime/results` | Saved result directory |
| `RESULT_TTL_SECONDS` | `86400` | Result retention period |
| `INCLUDE_BASE64_DEFAULT` | `false` | Embed images in JSON by default |
| `PUBLIC_BASE_URL` | empty | Optional absolute public API URL |
| `CORS_ORIGINS` | empty | Comma-separated allowed browser origins |
| `ALLOWED_HOSTS` | `*` | Comma-separated trusted hosts |
| `WEB_CONCURRENCY` | `1` | Uvicorn worker count |

## Production notes

- Place the service behind HTTPS and a reverse proxy.
- Set `API_KEY`, `APP_ENV=production`, and restrictive `ALLOWED_HOSTS`.
- Keep one worker per loaded model unless the host has sufficient RAM or GPU memory.
- Use persistent object storage instead of the local `runtime/results` directory when deploying multiple replicas.
- Use an external gateway for distributed rate limiting.
- Do not commit `.env`, checkpoint files, generated images, or the local GroundingDINO clone.
