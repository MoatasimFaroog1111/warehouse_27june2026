# warehouse_27june2026

Warehouse object-detection API powered by GroundingDINO.

## What the integration provides

- `GET /health` checks the local GroundingDINO configuration and model weights.
- `POST /detect` accepts an image and an open-vocabulary prompt.
- The response contains the object count, labels, confidence scores, pixel coordinates, and the annotated image as Base64 JPEG.
- The model is loaded once and reused between requests.
- CPU mode is selected automatically when CUDA is unavailable.

## Project layout

```text
app/
  main.py                 FastAPI endpoints
  grounding_service.py    GroundingDINO model wrapper
scripts/
  run_api.sh              API launcher
  test_api.py             End-to-end smoke test
GroundingDINO/             Local upstream clone, ignored by Git
requirements-api.txt
```

## Run in the current Codespace

From `/workspaces/warehouse_27june2026`:

```bash
git pull origin main
source GroundingDINO/.venv/bin/activate
python -m pip install -r requirements-api.txt
bash scripts/run_api.sh
```

The API will be available on port `8000`:

- API documentation: `/docs`
- Health check: `/health`
- Detection endpoint: `/detect`

## Test with the GroundingDINO sample image

Open a second terminal in the same Codespace:

```bash
cd /workspaces/warehouse_27june2026
source GroundingDINO/.venv/bin/activate
python scripts/test_api.py GroundingDINO/.asset/cat_dog.jpeg \
  --prompt "cat . dog ." \
  --output result.jpg
```

The test prints JSON containing:

```json
{
  "count": 2,
  "device": "cpu",
  "detections": [
    {
      "label": "cat",
      "confidence": 0.0,
      "box_xyxy": [0.0, 0.0, 0.0, 0.0]
    }
  ],
  "annotated_image": "/workspaces/warehouse_27june2026/result.jpg"
}
```

The values above illustrate the response structure; actual confidence scores and coordinates come from the uploaded image.

## Direct API request

```bash
curl -X POST http://127.0.0.1:8000/detect \
  -F 'image=@GroundingDINO/.asset/cat_dog.jpeg' \
  -F 'prompt=cat . dog .' \
  -F 'box_threshold=0.35' \
  -F 'text_threshold=0.25'
```

## Warehouse prompts

Examples:

```text
box . pallet . carton . bottle .
forklift . worker . safety helmet .
damaged box . open carton .
```

## Configuration

Optional environment variables:

```text
GROUNDING_DINO_ROOT
GROUNDING_DINO_CONFIG
GROUNDING_DINO_CHECKPOINT
GROUNDING_DINO_DEVICE
MAX_UPLOAD_BYTES
HOST
PORT
```

The default checkpoint is expected at:

```text
GroundingDINO/weights/groundingdino_swint_ogc.pth
```

The first CPU request can take longer because the model is loaded lazily. Later requests reuse the loaded model.
