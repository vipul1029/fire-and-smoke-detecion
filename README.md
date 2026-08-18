
Fire Detection and smoke detection

# MarkMyAd — Fire Detection


An AI-powered fire and smoke detection system built with YOLOv8 and OpenCV. It can analyze images, videos, or a live webcam feed and annotate detections in real time.

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
- [Output & Annotations](#output--annotations)
- [Severity Scoring](#severity-scoring)
- [Using as a Library](#using-as-a-library)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

---

## Features

- Detects **fire** and **smoke** in images, videos, and live camera streams
- Draws color-coded bounding boxes with confidence percentages on each detection
- Shows a live HUD overlay with FPS, fire count, smoke count, and model name
- Calculates a severity score (0–100) and severity label per frame
- Automatically downloads model weights from Hugging Face on first run

---

## Project Structure

```
markmyad-fire-detection/
├── src/
│   ├── __init__.py            # Package init
│   ├── detector.py            # YOLOv8 model loading and inference
│   ├── pipeline.py            # FPS tracking, annotation, severity scoring
│   └── download_weights.py    # Weight download helpers
│   └── weights/               # Created automatically; stores the .pt model
├── run.py                     # Main entry point
├── requirements.txt           # Python dependencies
└── README.md
```

---

## Requirements

- Python 3.9+
- Windows, macOS, or Linux
- CPU or CUDA-capable GPU (optional, for faster inference)
- ~200 MB disk space for model weights

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

| Package | Version | Purpose |
|---------|---------|---------|
| `ultralytics` | 8.2.18 | YOLOv8 inference |
| `torch` | latest | PyTorch backend |
| `opencv-python` | 4.9.0.80 | Frame I/O and annotation |
| `numpy` | 1.26.4 | Array operations |
| `requests` | latest | Downloading model weights |

**3. (Optional) Pre-download model weights:**

Model weights download automatically on first run. To download them manually:

```bash
python -m src.download_weights
```

Weights are saved to `src/weights/fire_detection_yolov8m.pt`.

---

## Usage

### Image

Run detection on a single image and save the annotated result:

```bash
python run.py --source path/to/image.jpg --output result.jpg
```

### Video

Process a video file frame by frame:

```bash
python run.py --source path/to/video.mp4 --output result.mp4
```

Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`

### Webcam

Run live detection from your webcam. Press **`q`** to quit:

```bash
python run.py --source webcam
```

### CLI Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--source` | `str` | *(required)* | Image/video file path or `webcam` |
| `--output` | `str` | `output.jpg` | Output file path |
| `--conf` | `float` | `0.25` | Confidence threshold (0.0–1.0) |
| `--device` | `str` | `cpu` | Inference device: `cpu`, `cuda`, or GPU index |

**Examples:**

```bash
# Higher confidence threshold on GPU
python run.py --source image.jpg --conf 0.5 --device cuda

# Video with custom output path
python run.py --source footage.mp4 --output out.mp4 --conf 0.3

# Webcam with CUDA
python run.py --source webcam --device cuda
```

---

## Model Info

| Property | Value |
|----------|-------|
| Architecture | YOLOv8 medium |
| Input size | 640 × 640 |
| Classes | `0` → smoke, `1` → fire |
| IOU threshold | 0.45 |
| Weights file | `src/weights/fire_detection_yolov8m.pt` |
| Source | [Hugging Face — rabahdev/fire-smoke-yolov8n](https://huggingface.co/rabahdev/fire-smoke-yolov8n) |

---

## Output & Annotations

Each detection contains:

- Bounding box coordinates `[x1, y1, x2, y2]`
- Confidence score (0.0 – 1.0)
- Class label: `"fire"` or `"smoke"`
- Cropped region of the detected area

**Bounding box colors:**

| Class | Color |
|-------|-------|
| Fire | Red |
| Smoke | Orange |

**HUD overlay** (bottom-left corner of every frame):

```
FPS:24  Fire:1  Smoke:0  Model:fire-detection
```

HUD color changes based on what is detected — red for fire, orange for smoke only, green for nothing detected.

---

## Severity Scoring

The pipeline computes a severity score per frame:

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
    detector_conf=0.3,
    device="cpu",
    target_fps=30,
)

frame = cv2.imread("image.jpg")

detections = pipeline.process_frame(frame)
annotated = pipeline.annotate_frame(frame, detections)

stats = pipeline.get_stats()
# {
#   "fire_count": 1,
#   "smoke_count": 0,
#   "severity_score": 47.0,
#   "severity_label": "high",
#   "fps": 24.0,
#   "avg_confidence": 0.812,
#   ...
# }

cv2.imwrite("output.jpg", annotated)
```

**Using a callback:**

```python
def on_detection(detections, frame):
    print(f"{len(detections)} hazard(s) detected")

pipeline = FireAIPipeline(on_results=on_detection)
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FIRE_MODEL_HF_URL` | Hugging Face URL | Override the URL used to download model weights |

You can set this in a `.env` file at the project root:

```env
FIRE_MODEL_HF_URL=https://your-host.com/custom_weights.pt
```

---

## Troubleshooting

**Cannot read image or video**
- Check that the file path is correct and the file format is supported.

**Webcam not opening**
- Make sure the webcam is connected and not in use by another app.

**Slow inference**
- Use `--device cuda` if you have an NVIDIA GPU with CUDA installed.

**Weight download fails**
- Check your internet connection.
- Place the `.pt` file manually at `src/weights/fire_detection_yolov8m.pt`, or set `FIRE_MODEL_HF_URL` to an alternative download URL.
