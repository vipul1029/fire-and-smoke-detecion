# pyright: reportMissingImports=false
"""
Temporal Transformer for real-time fire progression prediction.

Architecture:
    Input  : (batch, seq_len=15, features=8)
    Layers : Linear projection → Positional Encoding →
             Transformer Encoder (3 layers, 4 heads) → Mean pooling → MLP heads
    Outputs: growth_label (3-class), risk_score (0-100),
             predicted_area_5s, predicted_area_10s
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

PREDICTOR_WEIGHTS = Path(__file__).parent / "weights" / "fire_predictor.pt"

FEATURE_DIM = 8
SEQ_LEN = 15
D_MODEL = 64
N_HEAD = 4
NUM_LAYERS = 3
DIM_FF = 256
DROPOUT = 0.1

GROWTH_LABELS = {0: "stable", 1: "growing", 2: "critical"}


# ── Model definition ──────────────────────────────────────────────────────────

def _build_model():
    import torch
    import torch.nn as nn

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, dropout: float, max_len: int = 512):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            x = x + self.pe[:, : x.size(1)]
            return self.dropout(x)

    class TemporalTransformerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(FEATURE_DIM, D_MODEL)
            self.pos_enc = PositionalEncoding(D_MODEL, DROPOUT)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=D_MODEL,
                nhead=N_HEAD,
                dim_feedforward=DIM_FF,
                dropout=DROPOUT,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, NUM_LAYERS)
            self.norm = nn.LayerNorm(D_MODEL)

            # Output heads
            self.growth_head = nn.Sequential(
                nn.Linear(D_MODEL, 32), nn.ReLU(), nn.Linear(32, 3)
            )
            self.risk_head = nn.Sequential(
                nn.Linear(D_MODEL, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
            )
            self.area_5s_head = nn.Sequential(
                nn.Linear(D_MODEL, 32), nn.ReLU(), nn.Linear(32, 1), nn.ReLU()
            )
            self.area_10s_head = nn.Sequential(
                nn.Linear(D_MODEL, 32), nn.ReLU(), nn.Linear(32, 1), nn.ReLU()
            )

        def forward(self, x):
            x = self.input_proj(x)
            x = self.pos_enc(x)
            x = self.transformer(x)
            x = self.norm(x)
            x = x.mean(dim=1)  # mean pooling across time steps
            return {
                "growth_logits": self.growth_head(x),
                "risk_score": self.risk_head(x).squeeze(-1) * 100.0,
                "area_5s": self.area_5s_head(x).squeeze(-1),
                "area_10s": self.area_10s_head(x).squeeze(-1),
            }

    return TemporalTransformerModel()


# ── Inference wrapper ─────────────────────────────────────────────────────────

class FireProgressionPredictor:
    """
    Loads trained Temporal Transformer weights and runs real-time inference.
    Gracefully disabled if weights are not yet trained.
    """

    def __init__(self, weights_path: Path = PREDICTOR_WEIGHTS):
        self.weights_path = weights_path
        self.model = None
        self.available = False
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None
        self._load()

    def _load(self) -> None:
        if not self.weights_path.exists():
            logger.warning(
                "Predictor weights not found at %s. Run train_predictor.py first.",
                self.weights_path,
            )
            return
        try:
            import torch
            ckpt = torch.load(str(self.weights_path), map_location="cpu", weights_only=False)
            self.model = _build_model()
            self.model.load_state_dict(ckpt["model_state"])
            self.model.eval()
            self._scaler_mean = np.array(ckpt["scaler_mean"], dtype=np.float32)
            self._scaler_std = np.array(ckpt["scaler_std"], dtype=np.float32)
            self.available = True
            logger.info("Loaded fire predictor weights from %s", self.weights_path)
        except Exception as e:
            logger.error("Failed to load predictor: %s", e)

    def predict(self, sequence: List[List[float]]) -> Dict[str, Any]:
        """
        Run prediction on a sequence of SEQ_LEN feature vectors.
        Returns a dict with growth_label, risk_score, area_5s, area_10s.
        Returns a safe default dict if model is unavailable.
        """
        if not self.available or self.model is None:
            return {
                "growth_label": "unknown",
                "growth_confidence": 0.0,
                "risk_score": 0.0,
                "area_5s": 0.0,
                "area_10s": 0.0,
                "available": False,
            }

        import torch
        import torch.nn.functional as F

        x = np.array(sequence, dtype=np.float32)

        # Normalise using training statistics
        if self._scaler_mean is not None and self._scaler_std is not None:
            x = (x - self._scaler_mean) / (self._scaler_std + 1e-8)

        x_t = torch.tensor(x).unsqueeze(0)  # (1, seq_len, features)

        with torch.no_grad():
            out = self.model(x_t)
            probs = F.softmax(out["growth_logits"], dim=-1)[0]
            class_id = int(probs.argmax().item())
            confidence = float(probs[class_id].item())
            risk_score = float(out["risk_score"][0].item())
            area_5s = float(out["area_5s"][0].item())
            area_10s = float(out["area_10s"][0].item())

        return {
            "growth_label": GROWTH_LABELS[class_id],
            "growth_confidence": round(confidence * 100, 1),
            "risk_score": round(min(risk_score, 100.0), 1),
            "area_5s": round(area_5s, 6),
            "area_10s": round(area_10s, 6),
            "available": True,
        }

    @property
    def info(self) -> dict:
        return {
            "available": self.available,
            "weights": str(self.weights_path),
            "architecture": "TemporalTransformer",
            "seq_len": SEQ_LEN,
            "feature_dim": FEATURE_DIM,
            "d_model": D_MODEL,
            "num_layers": NUM_LAYERS,
            "nhead": N_HEAD,
        }
