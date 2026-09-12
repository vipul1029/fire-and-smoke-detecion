# ============================================================
#  YOLOv8m Fire & Smoke Detection — Google Colab Training
#  Dataset : D-Fire (sayedgamal99/smoke-fire-detection-yolo)
#  Model   : YOLOv8m
#  Classes : 0=smoke  1=fire
#
#  HOW TO USE:
#  1. Open Google Colab (colab.research.google.com)
#  2. Runtime → Change runtime type → T4 GPU
#  3. Upload this file to Colab
#  4. Run each cell block (marked with # %%) top to bottom
#  5. At the end, download fire_yolov8m_best.pt
#  6. Drop it into  src/weights/fire_detection_yolov8m.pt
# ============================================================


# %% ── Cell 1: Check GPU ─────────────────────────────────────────────────────
import subprocess
result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(result.stdout if result.returncode == 0 else "No GPU found — change runtime to T4 GPU")


# %% ── Cell 2: Install dependencies ─────────────────────────────────────────
subprocess.run(["pip", "install", "-q", "ultralytics==8.2.18", "kaggle"], check=True)
print("Dependencies installed.")


# %% ── Cell 3: Kaggle API setup ───────────────────────────────────────────────
# Option A — Upload kaggle.json
#   Download kaggle.json from: kaggle.com → Your Profile → Settings → API → Create New Token
#   Then run the cell below to upload it

from google.colab import files
print("Upload your kaggle.json file now:")
uploaded = files.upload()

import os, json, shutil
os.makedirs("/root/.kaggle", exist_ok=True)
shutil.copy("kaggle.json", "/root/.kaggle/kaggle.json")
os.chmod("/root/.kaggle/kaggle.json", 0o600)
print("Kaggle API configured.")


# %% ── Cell 4: Download D-Fire dataset ───────────────────────────────────────
os.makedirs("/content/dataset", exist_ok=True)
os.chdir("/content/dataset")

subprocess.run([
    "kaggle", "datasets", "download",
    "-d", "sayedgamal99/smoke-fire-detection-yolo",
    "--unzip"
], check=True)

print("Dataset downloaded. Contents:")
for item in os.listdir("/content/dataset"):
    print(" ", item)


# %% ── Cell 5: Inspect and prepare data.yaml ─────────────────────────────────
import glob, yaml
from pathlib import Path

# Find existing data.yaml
yaml_files = glob.glob("/content/dataset/**/*.yaml", recursive=True)
print("Found yaml files:", yaml_files)

# Find train/val/test image directories
train_dirs = glob.glob("/content/dataset/**/train/images", recursive=True)
val_dirs   = glob.glob("/content/dataset/**/valid/images", recursive=True)
if not val_dirs:
    val_dirs = glob.glob("/content/dataset/**/val/images", recursive=True)

print("Train dirs:", train_dirs)
print("Val dirs  :", val_dirs)

train_path = train_dirs[0] if train_dirs else "/content/dataset/train/images"
val_path   = val_dirs[0]   if val_dirs   else "/content/dataset/valid/images"

# Write a clean data.yaml that guarantees class order: 0=smoke, 1=fire
data_yaml = {
    "path"  : "/content/dataset",
    "train" : train_path,
    "val"   : val_path,
    "nc"    : 2,
    "names" : {0: "smoke", 1: "fire"},
}

yaml_out = "/content/dataset/fire_data.yaml"
with open(yaml_out, "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"\ndata.yaml written to {yaml_out}")
print(open(yaml_out).read())


# %% ── Cell 6: Count dataset images ──────────────────────────────────────────
train_imgs = glob.glob(train_path + "/*.*")
val_imgs   = glob.glob(val_path   + "/*.*")
print(f"Train images : {len(train_imgs)}")
print(f"Val images   : {len(val_imgs)}")
print(f"Total        : {len(train_imgs) + len(val_imgs)}")


# %% ── Cell 7: Train YOLOv8m ─────────────────────────────────────────────────
from ultralytics import YOLO

model = YOLO("yolov8m.pt")  # downloads official YOLOv8m pretrained on COCO

results = model.train(
    data    = yaml_out,
    epochs  = 50,
    imgsz   = 640,
    batch   = 16,
    device  = 0,               # GPU
    name    = "fire_yolov8m",
    project = "/content/runs",
    patience= 15,              # early stopping
    amp     = True,            # mixed precision — speeds up training
    plots   = True,
    save    = True,
    pretrained = True,
    optimizer  = "auto",
    lr0        = 0.01,
    lrf        = 0.01,
    momentum   = 0.937,
    weight_decay = 0.0005,
    warmup_epochs = 3,
    mosaic  = 1.0,
    hsv_h   = 0.015,
    hsv_s   = 0.7,
    hsv_v   = 0.4,
    fliplr  = 0.5,
    scale   = 0.5,
    erasing = 0.4,
)

print("\nTraining complete.")
print(f"Best weights: {results.save_dir}/weights/best.pt")


# %% ── Cell 8: Evaluate the trained model ─────────────────────────────────────
best_weights = f"{results.save_dir}/weights/best.pt"
trained_model = YOLO(best_weights)

metrics = trained_model.val(data=yaml_out, device=0)

print("\n=== Final Evaluation Metrics ===")
print(f"mAP@0.5      : {metrics.box.map50:.4f}  ({metrics.box.map50*100:.2f}%)")
print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}  ({metrics.box.map*100:.2f}%)")
print(f"Precision    : {metrics.box.p.mean():.4f}  ({metrics.box.p.mean()*100:.2f}%)")
print(f"Recall       : {metrics.box.r.mean():.4f}  ({metrics.box.r.mean()*100:.2f}%)")


# %% ── Cell 9: Verify model architecture ─────────────────────────────────────
import torch

ckpt = torch.load(best_weights, map_location="cpu", weights_only=False)
m = ckpt.get("model")
yaml_info = getattr(m, "yaml", {})
total_params = sum(p.numel() for p in m.parameters())

print("=== Model Architecture Verification ===")
print(f"depth_multiple : {yaml_info.get('depth_multiple')}  (YOLOv8m = 0.67)")
print(f"width_multiple : {yaml_info.get('width_multiple')}  (YOLOv8m = 0.75)")
print(f"Total params   : {total_params:,}  (YOLOv8m ≈ 25M)")
print(f"Classes (nc)   : {yaml_info.get('nc')}")
print(f"Class names    : {getattr(m, 'names', 'unknown')}")


# %% ── Cell 10: Copy and download weights ────────────────────────────────────
import shutil

output_name = "fire_yolov8m_best.pt"
shutil.copy(best_weights, f"/content/{output_name}")

print(f"Downloading {output_name} ...")
files.download(f"/content/{output_name}")
print("Done. Save this file as:  src/weights/fire_detection_yolov8m.pt")


# %% ── Cell 11: Summary ───────────────────────────────────────────────────────
print("=" * 55)
print("TRAINING COMPLETE — SUMMARY")
print("=" * 55)
print(f"Model         : YOLOv8m")
print(f"Classes       : smoke (0), fire (1)")
print(f"mAP@0.5       : {metrics.box.map50*100:.2f}%")
print(f"mAP@0.5:0.95  : {metrics.box.map*100:.2f}%")
print(f"Precision     : {metrics.box.p.mean()*100:.2f}%")
print(f"Recall        : {metrics.box.r.mean()*100:.2f}%")
print(f"Saved as      : {output_name}")
print("=" * 55)
print()
print("NEXT STEPS:")
print("1. Save fire_yolov8m_best.pt as src/weights/fire_detection_yolov8m.pt")
print("2. Download 85 fire videos from Kaggle:")
print("   kaggle.com/datasets/unidpro/fire-and-smoke-dataset")
print("3. Run: python train_predictor.py --videos_dir path/to/videos --device cuda")
