# pyright: reportMissingImports=false
"""Fire and smoke detection pipeline for live camera and uploaded videos."""

from __future__ import annotations

import logging
import time
from typing import Callable, List, Optional

import numpy as np

from .detector import Detection, FireDetector

logger = logging.getLogger(__name__)


class FireAIPipeline:
    def __init__(
        self,
        detector_conf: float = 0.25,
        device: str = "cpu",
        target_fps: int = 5,
        on_results: Optional[Callable] = None,
    ):
        self.on_results = on_results
        self.device = device
        self.target_fps = max(1, int(target_fps))
        self.process_interval = 1.0 / self.target_fps

        logger.info("Initializing fire detection pipeline...")
        self.detector = FireDetector(conf_threshold=detector_conf, device=device)

        self.frame_idx = 0
        self.fps = 0.0
        self.total_detected = 0
        self._fps_times = []
        self._latest_detections: List[Detection] = []

    def process_frame(self, frame: np.ndarray) -> List[Detection]:
        self.frame_idx += 1
        t0 = time.perf_counter()
        detections = self.detector.detect(frame)
        self._latest_detections = detections
        self.total_detected += len(detections)

        self._fps_times.append(time.perf_counter())
        if len(self._fps_times) > 30:
            self._fps_times = self._fps_times[-30:]
        if len(self._fps_times) > 1:
            elapsed = self._fps_times[-1] - self._fps_times[0]
            self.fps = round(len(self._fps_times) / elapsed, 1) if elapsed > 0 else 0.0

        if self.on_results:
            self.on_results(detections, frame)

        logger.debug(
            "Fire frame %d processed in %.3fs (%d detections)",
            self.frame_idx,
            time.perf_counter() - t0,
            len(detections),
        )
        return detections

    def annotate_frame(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        try:
            import cv2
        except ImportError:
            return frame

        out = frame.copy()

        # Class colors: 0 (smoke) -> Orange (0, 165, 255), 1 (fire) -> Red (0, 0, 255)
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            color = (0, 0, 255) if detection.label == "fire" else (0, 165, 255)
            label = f"{detection.label} {detection.confidence * 100:.0f}%"
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_y = max(18, y1 - 6)
            cv2.rectangle(out, (x1, text_y - th - 6), (x1 + tw + 8, text_y + 2), (0, 0, 0), -1)
            cv2.putText(out, label, (x1 + 4, text_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        fire_count = sum(1 for d in detections if d.label == "fire")
        smoke_count = sum(1 for d in detections if d.label == "smoke")
        hud = f"FPS:{self.fps:.0f}  Fire:{fire_count}  Smoke:{smoke_count}  Model:fire-detection"
        hud_color = (0, 0, 255) if fire_count > 0 else ((0, 165, 255) if smoke_count > 0 else (0, 255, 0))
        cv2.putText(out, hud, (8, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, hud_color, 1, cv2.LINE_AA)
        return out

    def get_stats(self) -> dict:
        detections = self._latest_detections
        fire_count = sum(1 for d in detections if d.label == "fire")
        smoke_count = sum(1 for d in detections if d.label == "smoke")
        total_hazards = len(detections)

        severity_score = 0.0
        if total_hazards > 0:
            severity_score = min(100.0, round((fire_count * 45) + (smoke_count * 20) + (self.fps * 2), 1))

        if severity_score >= 70:
            severity_label = "critical"
        elif severity_score >= 40:
            severity_label = "high"
        elif severity_score >= 15:
            severity_label = "medium"
        else:
            severity_label = "low"

        hazard_detections = [
            {
                "id": index + 1,
                "bbox": detection.bbox,
                "label": detection.label,
                "confidence": round(detection.confidence, 3),
            }
            for index, detection in enumerate(detections)
        ]

        return {
            "fire_count": fire_count,
            "smoke_count": smoke_count,
            "total_hazards": total_hazards,
            "detections": hazard_detections,
            "total_potholes": 0,  # compatible naming
            "fps": self.fps,
            "frame_count": self.frame_idx,
            "avg_confidence": round(sum(d.confidence for d in detections) / len(detections), 3) if detections else 0.0,
            "severity_score": severity_score,
            "severity_label": severity_label,
            "model": "fire-detection",
            "detector": self.detector.info,
            "pipeline_type": "fire",
            "target_fps": self.target_fps,
        }

    @property
    def info(self) -> dict:
        return {
            "detector": self.detector.info,
            "tracker": {"backend": "none"},
            "inference": {"backend": "none"},
            "model": "fire-detection",
            "pipeline_type": "fire",
            "target_fps": self.target_fps,
        }
