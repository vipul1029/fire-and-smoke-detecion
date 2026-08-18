Fire Detection and smoke detection

This is a standalone, independent extraction of the Fire Detection system from the `markmyad-supervision` project.

## Purpose
This project provides a runnable, zero-dependency (other than Python packages) version of the original Fire Detection AI pipeline. It is entirely self-contained and does not require the original `markmyad-supervision` backend or frontend to run.

## What It Detects
- **Fire**
- **Smoke**

## Model Details
- **Architecture**: YOLOv8 (specifically `yolov8n` or `yolov8m` depending on weights)
- **Framework**: Ultralytics / PyTorch
- **Weights File**: `fire_detection_yolov8m.pt`
- **Model Source**: Downloaded dynamically from Hugging Face (`https://huggingface.co/rabahdev/fire-smoke-yolov8n/resolve/main/best.pt`) on first run if not present locally.

## Project Structure
```text
markmyad-fire-detection/
├── src/
│   ├── __init__.py          # Package init
│   ├── detector.py          # YOLOv8 loading and inference wrapper
│   ├── pipeline.py          # FPS calculation, hazard logic, bounding box drawing
│   ├── download_weights.py  # Script for downloading weights from Hugging Face
│   └── weights/             # (Created dynamically) Where the .pt model is stored
├── requirements.txt         # Python dependencies
├── run.py                   # Main entry point for standalone execution
└── README.md                # This file
```

## System Requirements
- Windows, macOS, or Linux
- Python 3.9+
- CPU or CUDA-compatible GPU

## Installation
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   
   # Windows:
   .\.venv\Scripts\activate
   
   # macOS/Linux:
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
The standalone implementation supports Image, Video, and Webcam input modes. The model weights will automatically be downloaded on the first run.

### Image Inference
Run fire detection on a single image. The output will be saved as `output.jpg` by default.
```bash
python run.py --source path/to/image.jpg --output result.jpg
```

### Video Inference
Run fire detection on a video file. The output will be saved as `output.mp4` by default.
```bash
python run.py --source path/to/video.mp4 --output result.mp4
```

### Webcam Inference
Run live fire detection using your computer's webcam. A window will open showing real-time detection. Press `q` to quit.
```bash
python run.py --source webcam
```

### Advanced Configuration
You can pass additional arguments to control the model behavior:
- `--conf`: Confidence threshold (default: 0.25). Higher values reduce false positives.
- `--device`: Target device for inference, e.g., `cpu`, `cuda`, `0` (default: `cpu`).

Example:
```bash
python run.py --source webcam --conf 0.4 --device cuda
```

## Independence & Integrity
This extraction guarantees that:
- It maintains the exact same prediction logic, thresholds, class mapping, and bounding box drawing as the original system.
- It does **not** rely on `markmyad-supervision` files or imports.
- It can be moved anywhere on your system and will continue to function independently.
