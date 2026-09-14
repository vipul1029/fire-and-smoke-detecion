# pyright: reportMissingImports=false
"""
Train the Temporal Transformer for fire progression prediction.

Usage:
    python train_predictor.py --videos_dir path/to/videos --epochs 50

Steps:
    1. Runs YOLOv8 on every video in videos_dir to extract per-frame features
    2. Self-labels sequences based on fire area growth over the next 5 seconds
    3. Applies Gaussian noise augmentation to expand the dataset
    4. Trains the Temporal Transformer with early stopping
    5. Saves best model to src/weights/fire_predictor.pt
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SEQ_LEN = 15
FEATURE_DIM = 8
LOOK_AHEAD_5S = 125   # frames at 25fps
LOOK_AHEAD_10S = 250  # frames at 25fps

GROW_THRESH_CRITICAL = 0.60   # >60% area growth → critical
GROW_THRESH_GROWING = 0.30    # >30% area growth → growing

AUGMENT_COPIES = 5            # noise augmentation multiplier
AUGMENT_STD = 0.01            # Gaussian noise std

BATCH_SIZE = 64
LR = 1e-3
PATIENCE = 10                 # early stopping patience


# ── Dataset ───────────────────────────────────────────────────────────────────

class FireSequenceDataset(Dataset):
    def __init__(self, sequences, labels, area_5s, area_10s, risk_scores):
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y_label = torch.tensor(labels, dtype=torch.long)
        self.y_area5 = torch.tensor(area_5s, dtype=torch.float32)
        self.y_area10 = torch.tensor(area_10s, dtype=torch.float32)
        self.y_risk = torch.tensor(risk_scores, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y_label[idx],
            self.y_area5[idx],
            self.y_area10[idx],
            self.y_risk[idx],
        )


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features_from_video(video_path: Path, pipeline) -> list:
    """Run YOLO on a video and return list of per-frame feature vectors."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open %s — skipping", video_path)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    pipeline.detector.color_rejected = 0
    pipeline.detector.texture_rejected = 0
    pipeline.persistence_rejected = 0
    pipeline._frame_history.clear()
    pipeline.frame_idx = 0
    pipeline._fps_times.clear()
    pipeline._latest_detections = []

    from src.tracker import FireFeatureTracker
    tracker = FireFeatureTracker(seq_len=SEQ_LEN)

    all_features = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = pipeline.detector.detect(frame)
        features = tracker.extract_features(detections, frame.shape)
        all_features.append(features)

    cap.release()
    logger.info("  %s — %d frames extracted", video_path.name, len(all_features))
    return all_features


def build_sequences(all_features: list, fps: float = 25.0):
    """
    Create sliding window sequences and self-label each one.

    Label logic (based on fire_area growth over next ~5 seconds):
        critical : area grows > 60%
        growing  : area grows > 30%
        stable   : otherwise
    """
    look_5s = max(1, int(fps * 5))
    look_10s = max(1, int(fps * 10))

    sequences, labels, area_5s_list, area_10s_list, risk_list = [], [], [], [], []
    n = len(all_features)

    for t in range(SEQ_LEN, n - look_5s):
        seq = all_features[t - SEQ_LEN: t]
        current_area = seq[-1][0]  # fire_area of last frame in sequence

        future_5 = all_features[min(t + look_5s, n - 1)][0]
        future_10 = all_features[min(t + look_10s, n - 1)][0]

        if current_area > 1e-6:
            growth_5s = (future_5 - current_area) / current_area
        else:
            growth_5s = 0.0

        if growth_5s >= GROW_THRESH_CRITICAL:
            label = 2  # critical
        elif growth_5s >= GROW_THRESH_GROWING:
            label = 1  # growing
        else:
            label = 0  # stable

        risk = min(100.0, abs(growth_5s) * 100.0)

        sequences.append(seq)
        labels.append(label)
        area_5s_list.append(future_5)
        area_10s_list.append(future_10)
        risk_list.append(risk)

    return sequences, labels, area_5s_list, area_10s_list, risk_list


def augment(sequences, labels, area_5s, area_10s, risks, copies: int, noise_std: float):
    """Add Gaussian noise to features to multiply the dataset."""
    aug_seqs, aug_labs, aug_a5, aug_a10, aug_r = [], [], [], [], []
    for _ in range(copies):
        for seq, lab, a5, a10, r in zip(sequences, labels, area_5s, area_10s, risks):
            noisy = (np.array(seq) + np.random.normal(0, noise_std, np.array(seq).shape)).tolist()
            aug_seqs.append(noisy)
            aug_labs.append(lab)
            aug_a5.append(a5)
            aug_a10.append(a10)
            aug_r.append(r)
    return aug_seqs, aug_labs, aug_a5, aug_a10, aug_r


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    from src.pipeline import FireAIPipeline
    from src.predictor import _build_model, PREDICTOR_WEIGHTS

    videos_dir = Path(args.videos_dir)
    video_files = sorted(
        list(videos_dir.rglob("*.mp4"))
        + list(videos_dir.rglob("*.avi"))
        + list(videos_dir.rglob("*.mov"))
        + list(videos_dir.rglob("*.mkv"))
    )

    if not video_files:
        logger.error("No video files found in %s", videos_dir)
        return

    logger.info("Found %d videos", len(video_files))

    # Build pipeline once (loads YOLO weights once)
    pipeline = FireAIPipeline(
        detector_conf=0.20,
        device=args.device,
        enable_verification=False,  # raw YOLO output for training
    )

    # ── Extract features from all videos ──
    all_sequences, all_labels, all_a5, all_a10, all_risks = [], [], [], [], []

    for vpath in video_files:
        import cv2
        cap = cv2.VideoCapture(str(vpath))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        features = extract_features_from_video(vpath, pipeline)
        if len(features) < SEQ_LEN + int(fps * 5) + 1:
            logger.warning("  %s too short — skipping", vpath.name)
            continue

        seqs, labs, a5, a10, risks = build_sequences(features, fps)
        all_sequences.extend(seqs)
        all_labels.extend(labs)
        all_a5.extend(a5)
        all_a10.extend(a10)
        all_risks.extend(risks)

    if not all_sequences:
        logger.error("No sequences generated — check your videos")
        return

    # ── Label distribution ──
    counts = {0: all_labels.count(0), 1: all_labels.count(1), 2: all_labels.count(2)}
    total = len(all_labels)
    logger.info(
        "Sequences: %d total | stable=%d growing=%d critical=%d",
        total, counts[0], counts[1], counts[2],
    )

    # ── Normalise features ──
    X = np.array(all_sequences, dtype=np.float32)
    mean = X.reshape(-1, FEATURE_DIM).mean(axis=0)
    std = X.reshape(-1, FEATURE_DIM).std(axis=0) + 1e-8
    X = (X - mean) / std
    all_sequences = X.tolist()

    # ── Augment ──
    aug_seqs, aug_labs, aug_a5, aug_a10, aug_r = augment(
        all_sequences, all_labels, all_a5, all_a10, all_risks,
        AUGMENT_COPIES, AUGMENT_STD,
    )
    all_sequences += aug_seqs
    all_labels += aug_labs
    all_a5 += aug_a5
    all_a10 += aug_a10
    all_risks += aug_r
    logger.info("After augmentation: %d sequences", len(all_sequences))

    # ── Train / val split ──
    indices = list(range(len(all_sequences)))
    random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    def make_ds(idx):
        return FireSequenceDataset(
            [all_sequences[i] for i in idx],
            [all_labels[i] for i in idx],
            [all_a5[i] for i in idx],
            [all_a10[i] for i in idx],
            [all_risks[i] for i in idx],
        )

    train_ds = make_ds(train_idx)
    val_ds = make_ds(val_idx)

    # Weighted sampler to handle class imbalance
    label_counts = np.bincount([all_labels[i] for i in train_idx], minlength=3).astype(float)
    weights = 1.0 / (label_counts + 1e-8)
    sample_weights = torch.tensor([weights[all_labels[i]] for i in train_idx])
    sampler = WeightedRandomSampler(sample_weights, len(train_idx), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ── Model ──
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = _build_model().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Temporal Transformer: %d parameters", total_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    logger.info("Training for up to %d epochs on %s ...", args.epochs, device)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        correct = 0
        total_train = 0

        for X_b, y_lab, y_a5, y_a10, y_risk in train_loader:
            X_b = X_b.to(device)
            y_lab = y_lab.to(device)
            y_a5 = y_a5.to(device)
            y_a10 = y_a10.to(device)
            y_risk = y_risk.to(device)

            optimizer.zero_grad()
            out = model(X_b)

            loss = (
                ce_loss(out["growth_logits"], y_lab)
                + 0.3 * mse_loss(out["risk_score"], y_risk)
                + 0.2 * mse_loss(out["area_5s"], y_a5)
                + 0.2 * mse_loss(out["area_10s"], y_a10)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(X_b)
            preds = out["growth_logits"].argmax(dim=1)
            correct += (preds == y_lab).sum().item()
            total_train += len(X_b)

        train_loss /= total_train
        train_acc = correct / total_train * 100

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for X_b, y_lab, y_a5, y_a10, y_risk in val_loader:
                X_b = X_b.to(device)
                y_lab = y_lab.to(device)
                y_a5 = y_a5.to(device)
                y_a10 = y_a10.to(device)
                y_risk = y_risk.to(device)

                out = model(X_b)
                loss = (
                    ce_loss(out["growth_logits"], y_lab)
                    + 0.3 * mse_loss(out["risk_score"], y_risk)
                    + 0.2 * mse_loss(out["area_5s"], y_a5)
                    + 0.2 * mse_loss(out["area_10s"], y_a10)
                )
                val_loss += loss.item() * len(X_b)
                preds = out["growth_logits"].argmax(dim=1)
                val_correct += (preds == y_lab).sum().item()
                total_val += len(X_b)

        val_loss /= total_val
        val_acc = val_correct / total_val * 100
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d | train_loss=%.4f acc=%.1f%% | val_loss=%.4f acc=%.1f%%",
                epoch, train_loss, train_acc, val_loss, val_acc,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # ── Save ──
    PREDICTOR_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "scaler_mean": mean.tolist(),
            "scaler_std": std.tolist(),
            "val_loss": best_val_loss,
            "seq_len": SEQ_LEN,
            "feature_dim": FEATURE_DIM,
        },
        str(PREDICTOR_WEIGHTS),
    )
    logger.info("Saved predictor weights to %s (val_loss=%.4f)", PREDICTOR_WEIGHTS, best_val_loss)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fire Progression Temporal Transformer")
    parser.add_argument("--videos_dir", type=str, required=True,
                        help="Path to folder containing fire/smoke videos")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Maximum training epochs (default: 50)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu (default: cuda)")
    args = parser.parse_args()
    train(args)
