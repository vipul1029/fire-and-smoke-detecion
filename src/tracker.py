# pyright: reportMissingImports=false
"""Per-frame feature extraction and sliding window tracker for fire progression."""

from __future__ import annotations

from typing import List, Optional
import numpy as np

from .detector import Detection

FEATURE_DIM = 8
DEFAULT_SEQ_LEN = 15


class FireFeatureTracker:
    """
    Extracts 8 numerical features from YOLO detections each frame and
    maintains a sliding window sequence for the Temporal Transformer.

    Features per frame:
        0 - fire_area       : total fire bbox area normalised by frame area
        1 - smoke_area      : total smoke bbox area normalised by frame area
        2 - fire_count      : number of fire detections
        3 - smoke_count     : number of smoke detections
        4 - avg_confidence  : mean detection confidence (0 if none)
        5 - growth_rate     : change in fire_area since previous frame
        6 - centroid_x      : normalised horizontal centroid of fire (0 if none)
        7 - centroid_y      : normalised vertical centroid of fire (0 if none)
    """

    def __init__(self, seq_len: int = DEFAULT_SEQ_LEN):
        self.seq_len = seq_len
        self._history: List[List[float]] = []
        self._prev_fire_area: float = 0.0

    def extract_features(
        self, detections: List[Detection], frame_shape: tuple
    ) -> List[float]:
        h, w = frame_shape[:2]
        frame_area = max(float(h * w), 1.0)

        fire_dets = [d for d in detections if d.label == "fire"]
        smoke_dets = [d for d in detections if d.label == "smoke"]

        fire_area = sum(d.width * d.height for d in fire_dets) / frame_area
        smoke_area = sum(d.width * d.height for d in smoke_dets) / frame_area

        fire_count = float(len(fire_dets))
        smoke_count = float(len(smoke_dets))

        all_confs = [d.confidence for d in detections]
        avg_confidence = float(np.mean(all_confs)) if all_confs else 0.0

        growth_rate = fire_area - self._prev_fire_area
        self._prev_fire_area = fire_area

        if fire_dets:
            cx = float(np.mean([(d.bbox[0] + d.bbox[2]) / 2.0 for d in fire_dets])) / w
            cy = float(np.mean([(d.bbox[1] + d.bbox[3]) / 2.0 for d in fire_dets])) / h
        else:
            cx, cy = 0.0, 0.0

        return [
            fire_area,
            smoke_area,
            fire_count,
            smoke_count,
            avg_confidence,
            growth_rate,
            cx,
            cy,
        ]

    def update(
        self, detections: List[Detection], frame_shape: tuple
    ) -> Optional[List[List[float]]]:
        """
        Update history with current frame features.
        Returns the full sequence (list of FEATURE_DIM-length lists) when
        the window is full, otherwise returns None.
        """
        features = self.extract_features(detections, frame_shape)
        self._history.append(features)
        if len(self._history) > self.seq_len:
            self._history.pop(0)
        if len(self._history) == self.seq_len:
            return [list(f) for f in self._history]
        return None

    def reset(self) -> None:
        self._history.clear()
        self._prev_fire_area = 0.0

    @property
    def ready(self) -> bool:
        return len(self._history) == self.seq_len
