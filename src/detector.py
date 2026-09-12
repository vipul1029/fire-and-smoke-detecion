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
        enable_verification: bool = True,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.device = device
        self.enable_verification = enable_verification
        self.weights_url = weights_url or os.getenv(
            "FIRE_MODEL_HF_URL",
            "https://huggingface.co/rabahdev/fire-smoke-yolov8n/resolve/main/best.pt"
        )
        self.model = None
        self.backend = "yolo"
        self.class_names = {0: "smoke", 1: "fire"}
        self.available = False

        # Verification rejection counters (for research metrics)
        self.color_rejected = 0
        self.texture_rejected = 0

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

    # ── Stage 1: HSV Color Verification ──────────────────────────────────────

    def _verify_color(self, crop: np.ndarray, label: str) -> bool:
        """
        Stage 1 — HSV Color Verification.

        Fire occupies a specific hue range (red-orange-yellow) with high
        saturation and brightness. Smoke appears as low-saturation grey.
        Rejects false alarms from sunsets, tail lights, and reflections.
        """
        try:
            import cv2
        except ImportError:
            return True  # skip if cv2 unavailable

        if crop is None or crop.size == 0:
            return False

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        if label == "fire":
            # Fire: red-orange-yellow hues (0-35 and wraparound 160-180)
            # with high saturation (>100) and high brightness (>100)
            mask1 = cv2.inRange(hsv,
                                np.array([0, 100, 100]),
                                np.array([35, 255, 255]))
            mask2 = cv2.inRange(hsv,
                                np.array([160, 100, 100]),
                                np.array([180, 255, 255]))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            # Smoke: very low saturation (grey/white), medium-high brightness
            mask = cv2.inRange(hsv,
                               np.array([0, 0, 80]),
                               np.array([180, 60, 220]))

        pixel_ratio = np.count_nonzero(mask) / mask.size
        threshold = 0.12 if label == "fire" else 0.10
        return pixel_ratio > threshold

    # ── Stage 3: Texture Chaos Verification ──────────────────────────────────

    def _verify_texture(self, crop: np.ndarray, label: str) -> bool:
        """
        Stage 3 — Laplacian Texture Chaos Verification.

        Fire and smoke have high-frequency, chaotic textures. Uniform objects
        such as red signs, indicator lights, and walls produce low Laplacian
        variance. Rejects false alarms from static, smooth-surfaced objects.
        """
        try:
            import cv2
        except ImportError:
            return True

        if crop is None or crop.size == 0:
            return False

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Fire is more chaotic than smoke; smoke can be diffuse
        threshold = 200.0 if label == "fire" else 80.0
        return laplacian_var > threshold

    # ── Inference ─────────────────────────────────────────────────────────────

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

                if self.enable_verification:
                    # Stage 1: HSV color verification
                    if not self._verify_color(crop, label):
                        self.color_rejected += 1
                        logger.debug("Detection rejected by color check: %s", label)
                        continue

                    # Stage 3: Texture chaos verification
                    if not self._verify_texture(crop, label):
                        self.texture_rejected += 1
                        logger.debug("Detection rejected by texture check: %s", label)
                        continue

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
            "verification_enabled": self.enable_verification,
            "color_rejected": self.color_rejected,
            "texture_rejected": self.texture_rejected,
        }
