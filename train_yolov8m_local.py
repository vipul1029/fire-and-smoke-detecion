# pyright: reportMissingImports=false
"""
Train YOLOv8m for Fire and Smoke Detection — Local GPU Training
GPU    : NVIDIA RTX 4050 Laptop (6GB VRAM)
Dataset: D-Fire (smoke + fire, 2 classes)

STEPS BEFORE RUNNING:
1. Download D-Fire dataset from Kaggle:
   https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo
2. Extract the zip file to any folder on your PC
3. Run:
   python train_yolov8m_local.py --dataset_dir "path/to/extracted/folder"

OUTPUT:
   src/weights/fire_detection_yolov8m.pt  (replaces existing YOLOv8n weights)
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import shutil
import sys
from pathlib import Path

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_WEIGHTS = Path("src/weights/fire_detection_yolov8m.pt")


# ── GPU check ─────────────────────────────────────────────────────────────────

def check_gpu():
    if not torch.cuda.is_available():
        logger.error("CUDA not available. Check your PyTorch installation.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    logger.info("GPU    : %s", gpu_name)
    logger.info("VRAM   : %.1f GB", vram_gb)
    logger.info("CUDA   : %s", torch.version.cuda)

    # Auto batch size based on VRAM
    if vram_gb >= 8:
        batch = 16
    elif vram_gb >= 5.5:   # 6 GB cards report ~5.99 GB
        batch = 8
    elif vram_gb >= 4:
        batch = 4
    else:
        batch = 2

    logger.info("Batch  : %d (auto-selected for %.1f GB VRAM)", batch, vram_gb)
    return batch


# ── Dataset preparation ───────────────────────────────────────────────────────

def prepare_dataset(dataset_dir: str) -> str:
    dataset_path = Path(dataset_dir).resolve()

    if not dataset_path.exists():
        logger.error("Dataset directory not found: %s", dataset_path)
        logger.error("Download from: https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo")
        sys.exit(1)

    logger.info("Dataset dir: %s", dataset_path)

    # Find train/val image directories
    train_dirs = (
        glob.glob(str(dataset_path / "**/train/images"), recursive=True)
        or glob.glob(str(dataset_path / "train/images"))
    )
    val_dirs = (
        glob.glob(str(dataset_path / "**/valid/images"), recursive=True)
        or glob.glob(str(dataset_path / "**/val/images"), recursive=True)
        or glob.glob(str(dataset_path / "valid/images"))
    )

    if not train_dirs:
        logger.error("Could not find train/images folder inside %s", dataset_path)
        logger.error("Expected structure: dataset_dir/train/images/  and  dataset_dir/valid/images/")
        sys.exit(1)

    train_path = train_dirs[0]
    val_path   = val_dirs[0] if val_dirs else train_path

    train_count = len(glob.glob(train_path + "/*.*"))
    val_count   = len(glob.glob(val_path   + "/*.*"))
    logger.info("Train images : %d", train_count)
    logger.info("Val images   : %d", val_count)

    # Write clean data.yaml guaranteeing class order: 0=smoke, 1=fire
    data_yaml = {
        "path"  : str(dataset_path),
        "train" : train_path,
        "val"   : val_path,
        "nc"    : 2,
        "names" : {0: "smoke", 1: "fire"},
    }

    yaml_path = str(dataset_path / "fire_data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    logger.info("data.yaml written: %s", yaml_path)
    return yaml_path


# ── Training ──────────────────────────────────────────────────────────────────

def train(dataset_dir: str, epochs: int, batch: int):
    from ultralytics import YOLO

    yaml_path = prepare_dataset(dataset_dir)

    logger.info("Loading YOLOv8m base weights (pretrained on COCO)...")
    model = YOLO("yolov8m.pt")

    logger.info("Starting training — %d epochs, batch=%d, imgsz=640", epochs, batch)
    logger.info("This will take approximately 3-5 hours on RTX 4050.")
    logger.info("You can monitor progress below. Press Ctrl+C to stop early.")

    results = model.train(
        data         = yaml_path,
        epochs       = epochs,
        imgsz        = 640,
        batch        = batch,
        device       = 0,           # RTX 4050
        project      = "runs",
        name         = "fire_yolov8m",
        patience     = 15,          # early stopping
        amp          = True,        # mixed precision — critical for 6GB VRAM
        cache        = False,       # set to 'ram' if you have 16GB+ RAM
        workers      = 2,          # 4 causes spawn issues on Windows
        save         = True,
        plots        = True,
        pretrained   = True,
        optimizer    = "auto",
        lr0          = 0.01,
        lrf          = 0.01,
        momentum     = 0.937,
        weight_decay = 0.0005,
        warmup_epochs= 3,
        mosaic       = 1.0,
        close_mosaic = 10,
        hsv_h        = 0.015,
        hsv_s        = 0.7,
        hsv_v        = 0.4,
        fliplr       = 0.5,
        scale        = 0.5,
        erasing      = 0.4,
        verbose      = True,
    )

    return results


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(best_weights: str, yaml_path: str):
    from ultralytics import YOLO

    logger.info("Evaluating trained model...")
    model = YOLO(best_weights)
    metrics = model.val(data=yaml_path, device=0, verbose=False)

    print()
    print("=" * 50)
    print("  FINAL EVALUATION RESULTS")
    print("=" * 50)
    print(f"  mAP@0.5       : {metrics.box.map50:.4f}  ({metrics.box.map50 * 100:.2f}%)")
    print(f"  mAP@0.5:0.95  : {metrics.box.map:.4f}  ({metrics.box.map * 100:.2f}%)")
    print(f"  Precision     : {metrics.box.mp:.4f}  ({metrics.box.mp * 100:.2f}%)")
    print(f"  Recall        : {metrics.box.mr:.4f}  ({metrics.box.mr * 100:.2f}%)")
    print("=" * 50)
    return metrics


# ── Verify architecture ───────────────────────────────────────────────────────

def verify_architecture(weights_path: str):
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    m = ckpt.get("model")
    yaml_info = getattr(m, "yaml", {})
    total_params = sum(p.numel() for p in m.parameters())

    print()
    print("  ARCHITECTURE VERIFICATION")
    print(f"  depth_multiple : {yaml_info.get('depth_multiple')}  (expected 0.67 for YOLOv8m)")
    print(f"  width_multiple : {yaml_info.get('width_multiple')}  (expected 0.75 for YOLOv8m)")
    print(f"  Total params   : {total_params:,}  (expected ~25M for YOLOv8m)")
    print(f"  Classes (nc)   : {yaml_info.get('nc')}")
    print(f"  Class names    : {getattr(m, 'names', 'unknown')}")

    if abs(yaml_info.get('depth_multiple', 0) - 0.67) < 0.01:
        print("  Confirmed: YOLOv8m architecture")
    else:
        print("  WARNING: Architecture mismatch — check weights")


# ── Save to project ───────────────────────────────────────────────────────────

def save_to_project(best_weights: str):
    OUTPUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)

    # Backup existing weights
    if OUTPUT_WEIGHTS.exists():
        backup = OUTPUT_WEIGHTS.with_suffix(".backup.pt")
        shutil.copy(str(OUTPUT_WEIGHTS), str(backup))
        logger.info("Backed up old weights to %s", backup)

    shutil.copy(best_weights, str(OUTPUT_WEIGHTS))
    logger.info("Saved new YOLOv8m weights to %s", OUTPUT_WEIGHTS)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8m locally on RTX 4050")
    parser.add_argument(
        "--dataset_dir", type=str, required=True,
        help="Path to extracted D-Fire dataset folder"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs (default: 50)"
    )
    args = parser.parse_args()

    print()
    print("=" * 50)
    print("  YOLOv8m LOCAL TRAINING — RTX 4050")
    print("=" * 50)

    # Step 1: Check GPU
    auto_batch = check_gpu()

    # Step 2: Train
    results = train(args.dataset_dir, args.epochs, auto_batch)

    # Step 3: Get best weights path
    best_weights = str(Path(results.save_dir) / "weights" / "best.pt")
    logger.info("Best weights saved at: %s", best_weights)

    # Step 4: Evaluate
    yaml_path = prepare_dataset(args.dataset_dir)
    metrics = evaluate(best_weights, yaml_path)

    # Step 5: Verify architecture
    verify_architecture(best_weights)

    # Step 6: Copy to project
    save_to_project(best_weights)

    print()
    print("=" * 50)
    print("  TRAINING COMPLETE")
    print("=" * 50)
    print(f"  Model saved : {OUTPUT_WEIGHTS}")
    print(f"  mAP@0.5     : {metrics.box.map50 * 100:.2f}%")
    print()
    print("  NEXT STEPS:")
    print("  1. Download 85 fire videos:")
    print("     kaggle.com/datasets/unidpro/fire-and-smoke-dataset")
    print("  2. Run Temporal Transformer training:")
    print("     python train_predictor.py --videos_dir path/to/videos --device cuda")
    print("=" * 50)
