# Fire and Smoke Detection System

An AI-powered fire and smoke detection system built with YOLOv8m, a Three-Stage False Alarm Reduction Framework, a Temporal Transformer for fire progression prediction, and an automated natural language incident reporting module powered by a local LLM.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Image](#image)
  - [Video](#video)
  - [Webcam](#webcam)
  - [CLI Options](#cli-options)
- [Model Info](#model-info)
- [Three-Stage False Alarm Reduction](#three-stage-false-alarm-reduction)
- [Fire Progression Prediction](#fire-progression-prediction)
- [Incident Report Generation](#incident-report-generation)
- [Training](#training)
  - [Train YOLOv8m](#train-yolov8m)
  - [Train Temporal Transformer](#train-temporal-transformer)
- [Output & Annotations](#output--annotations)
- [Severity Scoring](#severity-scoring)
- [Using as a Library](#using-as-a-library)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

---

## Features

- Detects **fire** and **smoke** in images, videos, and live camera streams using YOLOv8m
- **Three-Stage False Alarm Reduction** — HSV color verification, temporal persistence filtering, and texture chaos analysis
- **Fire Progression Prediction** — Temporal Transformer predicts whether fire is stable, growing, or critical within the next 5–10 seconds
- **Automated Incident Reporting** — Local LLM (Ollama) converts detection data into natural language descriptions and safety recommendations; falls back to rule-based NLG if Ollama is offline
- Color-coded bounding boxes with confidence percentages
- Live HUD overlay with FPS, detection counts, severity score, and prediction label
- Calculates severity score (0–100) per frame
- Automatically downloads model weights on first run

---

## Project Structure

```
fire-smoke-detection/
├── src/
│   ├── __init__.py            # Package exports
│   ├── detector.py            # YOLOv8m inference + Stage 1 & Stage 3 verification
│   ├── pipeline.py            # Full pipeline: detection → filtering → prediction → reporting
│   ├── tracker.py             # Per-frame feature extraction for Temporal Transformer
│   ├── predictor.py           # Temporal Transformer model and inference
│   ├── reporter.py            # Natural language incident report generator (Ollama + fallback)
│   ├── download_weights.py    # Weight download helpers
│   └── weights/               # Model weights (auto-created)
│       ├── fire_detection_yolov8m.pt   # YOLOv8m fire/smoke detector
│       └── fire_predictor.pt           # Trained Temporal Transformer (after training)
├── train_yolov8m_local.py     # Train YOLOv8m on local GPU (RTX 4050 / 6GB VRAM)
├── train_predictor.py         # Train Temporal Transformer on fire videos
├── run.py                     # Main entry point
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.9+
- Windows, macOS, or Linux
- NVIDIA GPU with CUDA recommended (CPU supported)
- ~3 GB disk space for model weights
- [Ollama](https://ollama.com) (optional) for LLM-powered incident reports

---

## Installation

**1. Create and activate a virtual environment:**

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Install PyTorch with CUDA support (recommended):**

```bash
# For CUDA 12.4 (adjust cu124 to match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**4. (Optional) Install Ollama for AI-powered incident reports:**

Download from [ollama.com](https://ollama.com), then pull a model:

```bash
ollama pull phi3:mini
```

The system auto-detects Ollama at startup. If it is not running, it falls back to rule-based report generation silently.

---

## Usage

### Image

```bash
python run.py --source path/to/image.jpg --output result.jpg
```

### Video

```bash
python run.py --source path/to/video.mp4 --output result.mp4
```

### Webcam

```bash
python run.py --source webcam
```

Press **`q`** to quit.

### CLI Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--source` | `str` | *(required)* | Image/video file path or `webcam` |
| `--output` | `str` | `output.jpg` | Output file path |
| `--conf` | `float` | `0.25` | Confidence threshold (0.0–1.0) |
| `--device` | `str` | `cpu` | Inference device: `cpu`, `cuda`, or GPU index |

---

## Model Info

| Property | Value |
|----------|-------|
| Architecture | YOLOv8m |
| Parameters | ~25 million |
| Input size | 640 × 640 |
| Classes | `0` → smoke, `1` → fire |
| Confidence threshold | 0.25 |
| IOU threshold | 0.45 |
| Weights file | `src/weights/fire_detection_yolov8m.pt` |
| Training dataset | D-Fire (smoke + fire, ~17,000 images) |

---

## Three-Stage False Alarm Reduction

A novel three-stage verification framework filters YOLO detections before they are accepted, significantly reducing false positives from sunsets, tail lights, reflections, and camera flashes.

### Stage 1 — HSV Color Verification (`detector.py`)

Each detected region is converted to HSV color space and checked for fire/smoke-specific hue ranges:

- **Fire:** hue 0–35° or 160–180° (red-orange-yellow), saturation > 100, brightness > 100. Pixel ratio must exceed 12%.
- **Smoke:** low saturation (0–60), medium brightness (80–220). Pixel ratio must exceed 10%.

### Stage 2 — Temporal Persistence Filter (`pipeline.py`)

A detection is confirmed only if a spatially overlapping detection (IoU > 0.10) existed in at least one of the previous `persistence_window` frames (default: 3). Single-frame detections caused by camera flashes or sudden lighting changes are rejected.

### Stage 3 — Texture Chaos Analysis (`detector.py`)

Fire and smoke have high-frequency, chaotic textures. Laplacian variance of the cropped region is computed:

- **Fire threshold:** variance > 200
- **Smoke threshold:** variance > 80

Uniform surfaces (red signs, walls, indicator lights) are rejected.

---

## Fire Progression Prediction

A Temporal Transformer model predicts fire behavior over the next 5–10 seconds using a 15-frame sliding window of detection features.

### Architecture

```
Input: (batch, 15 frames, 8 features)
  → Linear(8 → 64)
  → Sinusoidal Positional Encoding
  → TransformerEncoder (3 layers, 4 heads, ff=256)
  → LayerNorm → Mean pooling
  → Output heads:
      growth_head   : 3-class (stable / growing / critical)
      risk_head     : risk score 0–100
      area_5s_head  : predicted fire area in 5 seconds
      area_10s_head : predicted fire area in 10 seconds
```

### Features per frame (8-dimensional)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `fire_area` | Total fire bbox area normalised by frame area |
| 1 | `smoke_area` | Total smoke bbox area normalised by frame area |
| 2 | `fire_count` | Number of fire detections |
| 3 | `smoke_count` | Number of smoke detections |
| 4 | `avg_confidence` | Mean detection confidence |
| 5 | `growth_rate` | Change in fire area since previous frame |
| 6 | `centroid_x` | Normalised horizontal centroid of fire |
| 7 | `centroid_y` | Normalised vertical centroid of fire |

### Prediction overlay

When trained weights are present, the top of the frame shows:

```
PREDICT:CRITICAL (84%)  Risk:91/100
```

---

## Incident Report Generation

Every 30 frames the system generates a structured natural language incident report from detection data.

### How it works

1. **Detection data is extracted:** fire count, smoke count, confidence, location in frame, consecutive frame count, area trend.
2. **Ollama LLM is called** (background thread, non-blocking) with a structured prompt.
3. **Previous report stays on screen** while the new one is being generated — no flickering.
4. If Ollama responds within 10 seconds → LLM-generated report is displayed.
5. If Ollama is unavailable or times out → rule-based NLG generates the report instead.

### Example output

```
High-confidence fire and smoke detected in the northeast region of the monitored area.
Smoke density is increasing and the fire region has expanded across 18 consecutive frames.

[RED] Possible rapidly developing fire. Immediate evacuation of the affected zone is
      recommended. Contact emergency services without delay.
```

### Ollama setup

```bash
# Install Ollama from https://ollama.com, then:
ollama pull phi3:mini        # 2.2 GB — fast, good quality
# or
ollama pull llama3.2:3b      # 2.0 GB — slightly better quality
```

The model can be changed in `src/reporter.py` by updating `OLLAMA_MODEL`.

---

## Training

### Train YOLOv8m

**1. Download D-Fire dataset:**

```
https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo
```

Extract to any folder, then run:

```bash
python train_yolov8m_local.py --dataset_dir "path/to/extracted/dataset"
```

The script auto-selects batch size based on available VRAM (batch=8 for 6 GB), trains for 50 epochs with mixed precision, evaluates, and saves best weights to `src/weights/fire_detection_yolov8m.pt`.

**Estimated training time:** 3–5 hours on RTX 4050 (6 GB VRAM).

### Train Temporal Transformer

**1. Download fire videos:**

```
https://www.kaggle.com/datasets/unidpro/fire-and-smoke-dataset
```

**2. Run training:**

```bash
python train_predictor.py --videos_dir "path/to/videos" --device cuda
```

The script self-labels sequences using fire area growth rate, applies Gaussian noise augmentation (×5), trains with early stopping (patience=10), and saves to `src/weights/fire_predictor.pt`.

---

## Output & Annotations

**Bounding box colors:**

| Class | Color |
|-------|-------|
| Fire | Red |
| Smoke | Orange |

**HUD overlay (bottom of frame):**

```
FPS:24  Fire:1  Smoke:1  Model:fire-detection
```

**Prediction overlay (top of frame, when predictor is trained):**

```
PREDICT:GROWING (76%)  Risk:63/100
```

**Incident report overlay (bottom bar, when detection is active):**

```
[white] High-confidence fire and smoke detected in the northeast region...
[white] Smoke density is increasing across 18 consecutive frames.
[RED]   Possible rapidly developing fire. Immediate evacuation recommended.
```

---

## Severity Scoring

```
severity_score = min(100, (fire_count × 45) + (smoke_count × 20) + (fps × 2))
```

| Score | Label |
|-------|-------|
| 70–100 | `critical` |
| 40–69 | `high` |
| 15–39 | `medium` |
| 0–14 | `low` |

---

## Using as a Library

```python
import cv2
from src.pipeline import FireAIPipeline

pipeline = FireAIPipeline(
    detector_conf=0.25,
    device="cuda",
    enable_verification=True,
    persistence_window=3,
)

frame = cv2.imread("image.jpg")
detections = pipeline.process_frame(frame)
annotated  = pipeline.annotate_frame(frame, detections)
stats      = pipeline.get_stats()

# stats includes:
# {
#   "fire_count": 1,
#   "smoke_count": 1,
#   "severity_score": 65.0,
#   "severity_label": "high",
#   "fps": 24.0,
#   "avg_confidence": 0.83,
#   "verification": {
#       "color_rejected": 3,
#       "texture_rejected": 1,
#       "persistence_rejected": 2,
#       "total_rejected": 6,
#   },
#   "prediction": {
#       "growth_label": "growing",
#       "growth_confidence": 76.0,
#       "risk_score": 63.0,
#       "area_5s": 0.021,
#       "available": True,
#   },
#   "incident_report": {
#       "active": True,
#       "description": "High-confidence fire and smoke detected...",
#       "recommendation": "Alert emergency services immediately...",
#       "severity": "high",
#       "region": "northeast region",
#       "consecutive_frames": 18,
#       "trend": "increasing",
#       "source": "llm",         # "llm" or "rule-based"
#       "timestamp": "14:32:07",
#   },
# }

cv2.imwrite("output.jpg", annotated)
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FIRE_MODEL_HF_URL` | Hugging Face URL | Override the URL used to download model weights |

---

## Troubleshooting

**CUDA not available**
- Reinstall PyTorch with CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
- Use `--cache-dir E:\pip_cache` if C drive is full.

**Ollama not generating reports**
- Make sure Ollama is running: `ollama serve`
- Make sure the model is pulled: `ollama pull phi3:mini`
- The system will fall back to rule-based NLG automatically if Ollama is offline.

**Temporal Transformer not active**
- Train it first: `python train_predictor.py --videos_dir path/to/videos`
- Weights must exist at `src/weights/fire_predictor.pt`

**Cannot read image or video**
- Check the file path and format (`.mp4`, `.avi`, `.mov`, `.mkv` for video).

**Webcam not opening**
- Ensure the webcam is not in use by another application.

**Slow inference**
- Use `--device cuda` if you have an NVIDIA GPU.
- Reduce input resolution if needed.
