from __future__ import annotations

import os
import sys
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import torch
from torchvision.ops import box_convert

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUNDING_DINO_ROOT = Path(
    os.getenv("GROUNDING_DINO_ROOT", PROJECT_ROOT / "GroundingDINO")
).resolve()
CONFIG_PATH = Path(
    os.getenv(
        "GROUNDING_DINO_CONFIG",
        GROUNDING_DINO_ROOT
        / "groundingdino"
        / "config"
        / "GroundingDINO_SwinT_OGC.py",
    )
).resolve()
CHECKPOINT_PATH = Path(
    os.getenv(
        "GROUNDING_DINO_CHECKPOINT",
        GROUNDING_DINO_ROOT / "weights" / "groundingdino_swint_ogc.pth",
    )
).resolve()

if str(GROUNDING_DINO_ROOT) not in sys.path:
    sys.path.insert(0, str(GROUNDING_DINO_ROOT))

from groundingdino.util.inference import annotate, load_image, load_model, predict  # noqa: E402


class GroundingDinoService:
    """Thread-safe, lazy-loaded GroundingDINO inference service."""

    def __init__(self, max_prompt_chars: int = 500, max_prompt_terms: int = 30) -> None:
        self.device = os.getenv(
            "GROUNDING_DINO_DEVICE",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        self.max_prompt_chars = max_prompt_chars
        self.max_prompt_terms = max_prompt_terms
        self._model = None
        self._model_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def normalize_prompt(self, prompt: str) -> str:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("The prompt must contain at least one object name.")
        if len(prompt) > self.max_prompt_chars:
            raise ValueError(
                f"The prompt exceeds the {self.max_prompt_chars}-character limit."
            )

        terms = [
            term.strip()
            for term in prompt.replace(",", ".").split(".")
            if term.strip()
        ]
        if not terms:
            raise ValueError("The prompt must contain at least one object name.")
        if len(terms) > self.max_prompt_terms:
            raise ValueError(
                f"The prompt exceeds the {self.max_prompt_terms}-term limit."
            )

        unique_terms = list(dict.fromkeys(terms))
        return " . ".join(unique_terms) + " ."

    def status(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "model_loaded": self._model is not None,
            "grounding_dino_root_exists": GROUNDING_DINO_ROOT.is_dir(),
            "config_exists": CONFIG_PATH.is_file(),
            "checkpoint_exists": CHECKPOINT_PATH.is_file(),
        }

    def _validate_files(self) -> None:
        missing = [
            str(path)
            for path in (GROUNDING_DINO_ROOT, CONFIG_PATH, CHECKPOINT_PATH)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "GroundingDINO is not ready. Missing: " + ", ".join(missing)
            )

    def get_model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._validate_files()
                    self._model = load_model(
                        str(CONFIG_PATH),
                        str(CHECKPOINT_PATH),
                        device=self.device,
                    )
        return self._model

    def detect(
        self,
        image_path: Path,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, int],
        dict[str, float],
        bytes,
        str,
    ]:
        normalized_prompt = self.normalize_prompt(prompt)
        model = self.get_model()
        image_source, image = load_image(str(image_path))

        with self._inference_lock:
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=normalized_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device,
            )

        height, width = image_source.shape[:2]
        scale = torch.tensor([width, height, width, height], dtype=boxes.dtype)
        pixel_boxes = box_convert(
            boxes=boxes * scale,
            in_fmt="cxcywh",
            out_fmt="xyxy",
        )

        detections: list[dict[str, Any]] = []
        label_counts: Counter[str] = Counter()
        confidence_totals: defaultdict[str, float] = defaultdict(float)

        for phrase, confidence, box in zip(phrases, logits, pixel_boxes):
            label = phrase.strip().lower()
            score = round(float(confidence), 4)
            x1, y1, x2, y2 = [round(float(value), 2) for value in box.tolist()]
            detections.append(
                {
                    "label": label,
                    "confidence": score,
                    "box_xyxy": [x1, y1, x2, y2],
                }
            )
            label_counts[label] += 1
            confidence_totals[label] += score

        average_confidence = {
            label: round(confidence_totals[label] / count, 4)
            for label, count in label_counts.items()
        }

        annotated_frame = annotate(
            image_source=image_source,
            boxes=boxes,
            logits=logits,
            phrases=phrases,
        )
        encoded_ok, encoded_image = cv2.imencode(".jpg", annotated_frame)
        if not encoded_ok:
            raise RuntimeError("Failed to encode the annotated image.")

        return (
            detections,
            dict(label_counts),
            average_confidence,
            encoded_image.tobytes(),
            normalized_prompt,
        )
