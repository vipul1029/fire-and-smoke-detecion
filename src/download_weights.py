"""Download fire model weights into this package."""

from __future__ import annotations

import os
from pathlib import Path

from .detector import FireDetector, FIRE_WEIGHTS


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


_load_env_file()


def download_weights(weights_url: str | None = None) -> Path:
    url = weights_url or os.getenv(
        "FIRE_MODEL_HF_URL",
        "https://huggingface.co/rabahdev/fire-smoke-yolov8n/resolve/main/best.pt"
    )
    FireDetector.download_from_hf(url, FIRE_WEIGHTS)
    return FIRE_WEIGHTS


def ensure_weights(weights_url: str | None = None) -> Path:
    if FIRE_WEIGHTS.exists():
        return FIRE_WEIGHTS
    return download_weights(weights_url)


if __name__ == "__main__":
    path = ensure_weights()
    print(path)
