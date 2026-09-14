# pyright: reportMissingImports=false
"""Fire and smoke detection pipeline for live camera and uploaded videos."""

from __future__ import annotations

import logging
import time
from typing import Callable, List, Optional

import numpy as np

from .detector import Detection, FireDetector
from .tracker import FireFeatureTracker
from .predictor import FireProgressionPredictor
from .reporter import IncidentReporter

logger = logging.getLogger(__name__)


class FireAIPipeline:
    def __init__(
        self,
        detector_conf: float = 0.25,
        device: str = "cpu",
        target_fps: int = 5,
        on_results: Optional[Callable] = None,
        enable_verification: bool = True,
        persistence_window: int = 3,
    ):
        self.on_results = on_results
        self.device = device
        self.target_fps = max(1, int(target_fps))
        self.process_interval = 1.0 / self.target_fps
        self.enable_verification = enable_verification

        # Stage 2: Temporal persistence window (number of frames)
        self._persistence_window = max(1, persistence_window)
        self._frame_history: List[List[List[int]]] = []  # list of bbox lists per frame
        self.persistence_rejected = 0

        logger.info("Initializing fire detection pipeline...")
        self.detector = FireDetector(
            conf_threshold=detector_conf,
            device=device,
            enable_verification=enable_verification,
        )

        # Temporal Transformer predictor
        self.predictor = FireProgressionPredictor()
        self.tracker = FireFeatureTracker()
        self._latest_prediction: dict = {}

        # Natural language incident reporter
        self.reporter = IncidentReporter(report_interval=30)
        self._latest_report: dict = {}

        self.frame_idx = 0
        self.fps = 0.0
        self.total_detected = 0
        self._fps_times = []
        self._latest_detections: List[Detection] = []

    # ── Stage 2: Temporal Persistence Filter ─────────────────────────────────

    @staticmethod
    def _iou(box_a: List[int], box_b: List[int]) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])

        inter = max(0, x_b - x_a) * max(0, y_b - y_a)
        if inter == 0:
            return 0.0

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _filter_by_persistence(self, detections: List[Detection]) -> List[Detection]:
        """
        Stage 2 — Temporal Persistence Filter.

        A detection is confirmed only if a spatially overlapping detection
        (IoU > 0.10) existed in at least one of the previous
        `persistence_window` frames. This rejects single-frame false alarms
        caused by camera flashes, sudden lighting changes, and passing
        reflections — all of which disappear within 1-2 frames.

        During the startup grace period (fewer frames than the window),
        all detections are accepted to avoid missing real fire events.
        """
        # Grace period: not enough history yet — accept everything
        if len(self._frame_history) < self._persistence_window - 1:
            return detections

        confirmed: List[Detection] = []
        for det in detections:
            is_persistent = False
            for past_bboxes in self._frame_history:
                for past_bbox in past_bboxes:
                    if self._iou(det.bbox, past_bbox) > 0.10:
                        is_persistent = True
                        break
                if is_persistent:
                    break

            if is_persistent:
                confirmed.append(det)
            else:
                self.persistence_rejected += 1
                logger.debug("Detection rejected by persistence check: %s", det.label)

        return confirmed

    # ── Frame Processing ──────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> List[Detection]:
        self.frame_idx += 1
        t0 = time.perf_counter()

        # Run YOLO inference (Stage 1 + Stage 3 applied inside detector)
        detections = self.detector.detect(frame)

        # Stage 2: Temporal persistence filtering
        if self.enable_verification:
            detections = self._filter_by_persistence(detections)

        # Update frame history for next frame's persistence check
        self._frame_history.append([d.bbox for d in detections])
        if len(self._frame_history) > self._persistence_window:
            self._frame_history.pop(0)

        self._latest_detections = detections
        self.total_detected += len(detections)

        # Temporal Transformer: update tracker and run prediction when ready
        sequence = self.tracker.update(detections, frame.shape)
        if sequence is not None and self.predictor.available:
            self._latest_prediction = self.predictor.predict(sequence)

        # Incident reporter: generate natural language description
        stats_snapshot = self._quick_stats(detections)
        self._latest_report = self.reporter.update(
            detections, stats_snapshot, self.frame_idx,
            frame.shape, self._latest_prediction or None,
        )

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

    def _quick_stats(self, detections: List[Detection]) -> dict:
        fire_count = sum(1 for d in detections if d.label == "fire")
        smoke_count = sum(1 for d in detections if d.label == "smoke")
        total = len(detections)
        severity_score = min(100.0, (fire_count * 45) + (smoke_count * 20))
        if severity_score >= 70:
            severity_label = "critical"
        elif severity_score >= 40:
            severity_label = "high"
        elif severity_score >= 15:
            severity_label = "medium"
        else:
            severity_label = "low"
        avg_conf = sum(d.confidence for d in detections) / total if total else 0.0
        return {
            "fire_count": fire_count,
            "smoke_count": smoke_count,
            "avg_confidence": avg_conf,
            "severity_label": severity_label,
        }

    # ── Annotation ────────────────────────────────────────────────────────────

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

        # Prediction overlay (shown only when predictor is trained and ready)
        pred = self._latest_prediction
        if pred.get("available"):
            label = pred["growth_label"].upper()
            risk = pred["risk_score"]
            conf = pred["growth_confidence"]
            pred_text = f"PREDICT:{label} ({conf:.0f}%)  Risk:{risk:.0f}/100"
            p_color = (0, 0, 255) if label == "CRITICAL" else (
                (0, 165, 255) if label == "GROWING" else (0, 200, 0)
            )
            cv2.rectangle(out, (6, 6), (len(pred_text) * 7 + 10, 22), (0, 0, 0), -1)
            cv2.putText(out, pred_text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, p_color, 1, cv2.LINE_AA)

        # Incident report overlay — shown at bottom when active
        fh, fw = out.shape[:2]
        report  = self._latest_report
        has_det = (fire_count + smoke_count) > 0

        if has_det and not report.get("active") and self.reporter._generating:
            # Thread is running — show placeholder
            overlay = out.copy()
            cv2.rectangle(overlay, (0, fh - 30), (fw, fh), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.70, out, 0.30, 0, out)
            cv2.putText(out, "INCIDENT REPORTER: Analyzing scene...",
                        (8, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (180, 180, 180), 1, cv2.LINE_AA)

        elif report.get("active"):
            severity = report.get("severity", "low")
            src      = report.get("source", "rule-based")
            r_color  = (
                (0, 0, 255)   if severity == "critical" else
                (0, 100, 255) if severity == "high"     else
                (0, 165, 255) if severity == "medium"   else
                (0, 200, 100)
            )

            # Semi-transparent black bar
            overlay = out.copy()
            cv2.rectangle(overlay, (0, fh - 95), (fw, fh), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)

            # Header bar
            src_label = "LLM" if src == "llm" else "Rule-Based"
            header = f"  INCIDENT REPORT [{src_label}]  |  Severity: {severity.upper()}"
            cv2.rectangle(out, (0, fh - 95), (fw, fh - 75), (40, 10, 10), -1)
            cv2.putText(out, header, (6, fh - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 180, 255), 1, cv2.LINE_AA)

            # Description
            desc     = report.get("description", "")
            max_ch   = max(1, fw // 8)
            line1    = desc[:max_ch]
            line2    = desc[max_ch: max_ch * 2] if len(desc) > max_ch else ""

            cv2.putText(out, line1, (6, fh - 57),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1, cv2.LINE_AA)
            if line2:
                cv2.putText(out, line2, (6, fh - 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1, cv2.LINE_AA)

            # Recommendation
            rec = report.get("recommendation", "")
            cv2.putText(out, rec[:max_ch], (6, fh - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, r_color, 1, cv2.LINE_AA)

        return out

    # ── Statistics ────────────────────────────────────────────────────────────

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

        total_rejected = (
            self.detector.color_rejected
            + self.detector.texture_rejected
            + self.persistence_rejected
        )

        return {
            "fire_count": fire_count,
            "smoke_count": smoke_count,
            "total_hazards": total_hazards,
            "detections": hazard_detections,
            "total_potholes": 0,
            "fps": self.fps,
            "frame_count": self.frame_idx,
            "avg_confidence": round(sum(d.confidence for d in detections) / len(detections), 3) if detections else 0.0,
            "severity_score": severity_score,
            "severity_label": severity_label,
            "model": "fire-detection",
            "detector": self.detector.info,
            "pipeline_type": "fire",
            "target_fps": self.target_fps,
            # Verification stats (for research metrics)
            "verification": {
                "enabled": self.enable_verification,
                "color_rejected": self.detector.color_rejected,
                "texture_rejected": self.detector.texture_rejected,
                "persistence_rejected": self.persistence_rejected,
                "total_rejected": total_rejected,
                "persistence_window": self._persistence_window,
            },
            "prediction": self._latest_prediction if self._latest_prediction else {
                "available": self.predictor.available,
                "growth_label": "warming up" if self.predictor.available else "not trained",
            },
            "incident_report": self._latest_report,
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
            "verification_enabled": self.enable_verification,
            "persistence_window": self._persistence_window,
        }
