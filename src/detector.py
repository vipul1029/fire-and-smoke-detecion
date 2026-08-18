# pyright: reportMissingImports=false
"""Fire and smoke detector backed by YOLOv8 weights."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# Apply PyTorch weights_only safety patch for compatibility with PyTorch 2.6+
try:
    import torch
    original_load = torch.load
    def patched_load(*args, **kwargs):
        if "weights_only" in kwargs:
            kwargs["weights_only"] = False
        else:
            kwargs["weights_only"] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load
except Exception:
    pass

logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path(__file__).parent / "weights"
FIRE_WEIGHTS = WEIGHTS_DIR / "fire_detection_yolov8m.pt"


@dataclass
class Detection:
    bbox: List[int]
    confidence: float
    class_id: int = 0
    label: str = "fire"
    crop: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


class FireDetector:
    def __init__(
        self,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        device: str = "cpu",
        weights_url: Optional[str] = None,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.device = device
        self.weights_url = weights_url or os.getenv(
            "FIRE_MODEL_HF_URL",
            "https://huggingface.co/rabahdev/fire-smoke-yolov8n/resolve/main/best.pt"
        )
        self.model = None
        self.backend = "yolo"
        self.class_names = {0: "smoke", 1: "fire"}
        self.available = False

        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        self._load_model()

    def _load_model(self) -> None:
        if not FIRE_WEIGHTS.exists():
            self._download_weights()

        if not FIRE_WEIGHTS.exists():
            logger.warning(
                "Fire detector disabled because weights are missing at %s",
                FIRE_WEIGHTS,
            )
            self.backend = "unavailable"
            self.available = False
            return

        from ultralytics import YOLO

        self.model = YOLO(str(FIRE_WEIGHTS))
        self.class_names = getattr(self.model, "names", {0: "smoke", 1: "fire"}) or {0: "smoke", 1: "fire"}
        self.available = True
        logger.info("Loaded fire detector weights from %s", FIRE_WEIGHTS)

    def _download_weights(self) -> None:
        if not self.weights_url:
            logger.warning("Fire weights missing and FIRE_MODEL_HF_URL is not set")
            return

        self.download_from_hf(self.weights_url, FIRE_WEIGHTS)

    @staticmethod
    def download_from_hf(url: str, output_path: Path) -> None:
        import requests
        logger.info("Downloading fire detection weights from Hugging Face: %s", url)
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info("Successfully downloaded weights to %s", output_path)
        except Exception as e:
            logger.error("Failed to download weights from Hugging Face: %s", e)
            if output_path.exists():
                output_path.unlink()
            raise RuntimeError(f"Failed to download weights: {e}") from e

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self.model is None:
            return []

        results = self.model.predict(
            source=frame,
            imgsz=self.input_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            stream=False,
        )

        detections: List[Detection] = []
        height, width = frame.shape[:2]
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0]) if box.cls is not None else 0
                label = self.class_names.get(class_id, "fire" if class_id == 1 else "smoke")

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2].copy()
                detections.append(
                    Detection(
                        bbox=[x1, y1, x2, y2],
                        confidence=confidence,
                        class_id=class_id,
                        label=label,
                        crop=crop,
                    )
                )

        return detections

    @property
    def info(self) -> dict:
        return {
            "backend": self.backend,
            "model": "fire-detection",
            "weights": str(FIRE_WEIGHTS),
            "available": self.available,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
        }
