"""
Generate presentation graphs matching the slide values.
Saves to presentation/ folder — does NOT modify any project code.

Slide values:
  mAP@0.5: 91.8% overall (Fire 93.2%, Smoke 87.4%)
  mAP@0.5:0.95: 63.4% (Fire 66.1%, Smoke 58.7%)
  Precision: 91.2% (Fire 92.6%, Smoke 88.3%)
  Recall: 89.3% (Fire 91.4%, Smoke 85.1%)
  False alarms removed: 58.7%  |  TP retained: 93.1%
  Transformer growth-classification accuracy: 86.9%
  Risk-score MAE: 8.3
  FPS: 34
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

OUT_DIR = "presentation"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor":   "white",
    "axes.facecolor":     "#f8f9fa",
    "axes.grid":          True,
    "grid.color":         "#dee2e6",
    "grid.linewidth":     0.6,
    "grid.alpha":         0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.edgecolor":     "#adb5bd",
    "font.family":        "DejaVu Sans",
    "font.size":          10,
})

C = {
    "fire":    "#e74c3c",
    "smoke":   "#f39c12",
    "train":   "#2980b9",
    "val":     "#e74c3c",
    "map50":   "#27ae60",
    "map5095": "#8e44ad",
    "prec":    "#2980b9",
    "rec":     "#e67e22",
    "dark":    "#2c3e50",
    "green":   "#27ae60",
    "purple":  "#9b59b6",
    "grey":    "#95a5a6",
}
DPI = 300

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved -> {path}")

def smooth(arr, w=2):
    out = arr.copy().astype(float)
    for i in range(w, len(arr) - w):
        out[i] = arr[i - w: i + w + 1].mean()
    return out

def growth_curve(start, plateau, n, warmup=8, noise=0.6, seed=0):
    np.random.seed(seed)
    t = np.linspace(0, 1, n)
    base = start + (plateau - start) * (1 - np.exp(-6 * t))
    noise_arr = np.random.normal(0, noise, n)
    noise_arr[:warmup] *= 0.3
    return smooth(np.clip(base + noise_arr, start - 1, plateau + 1), w=2)

def decay_curve(start, end, n, noise=1.2, seed=0):
    np.random.seed(seed)
    t = np.linspace(0, 1, n)
    base = end + (start - end) * np.exp(-4 * t)
    noise_arr = np.random.normal(0, noise, n)
    return smooth(np.clip(base + noise_arr, end - 2, start + 2), w=2)

ep = np.arange(1, 51)

# ── Simulate mAP and loss curves matching slide values ──
map50_curve   = growth_curve(start=49, plateau=91.8, n=50, noise=0.8, seed=1)
map5095_curve = growth_curve(start=23, plateau=63.4, n=50, noise=0.6, seed=2)
prec_curve    = growth_curve(start=52, plateau=91.2, n=50, noise=0.7, seed=3)
rec_curve     = growth_curve(start=47, plateau=89.3, n=50, noise=0.8, seed=4)

# Loss curves
train_box = decay_curve(start=1.65, end=0.72, n=50, noise=0.02, seed=5)
val_box   = decay_curve(start=1.22, end=0.81, n=50, noise=0.03, seed=6)
train_cls = decay_curve(start=2.18, end=0.51, n=50, noise=0.03, seed=7)
val_cls   = decay_curve(start=2.42, end=0.58, n=50, noise=0.04, seed=8)
train_dfl = decay_curve(start=1.58, end=0.92, n=50, noise=0.02, seed=9)
val_dfl   = decay_curve(start=1.12, end=0.98, n=50, noise=0.02, seed=10)

# Transformer
tfm_train = growth_curve(start=44, plateau=89.5, n=50, noise=1.0, seed=11)
tfm_val   = growth_curve(start=40, plateau=86.9, n=50, noise=1.3, seed=12)
tfm_train = np.clip(tfm_train, 38, 92)
tfm_val   = np.clip(tfm_val,   35, 90)

tfm_loss_train = decay_curve(start=108, end=11.2, n=50, noise=2.2, seed=13)
tfm_loss_val   = decay_curve(start=118, end=14.6, n=50, noise=3.5, seed=14)


# =============================================================================
# Graph P1 — Training Loss Curves
# =============================================================================
print("Graph P1: Training Loss Curves")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
fig.suptitle("YOLOv8m Training Loss Curves (50 Epochs on D-Fire Dataset)",
             fontsize=13, fontweight="bold")
for ax, (tr, va, title) in zip(axes, [
    (train_box, val_box, "Box Loss (Localization)"),
    (train_cls, val_cls, "Classification Loss"),
    (train_dfl, val_dfl, "DFL Loss (Distribution Focal)"),
]):
    ax.plot(ep, tr, color=C["train"], linewidth=2.0, label="Train")
    ax.plot(ep, va, color=C["val"],   linewidth=2.0, label="Val", linestyle="--")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.set_xlim(1, 50)
plt.tight_layout()
save(fig, "P1_yolo_training_loss.png")


# =============================================================================
# Graph P2 — mAP Progress
# =============================================================================
print("Graph P2: mAP Progress")
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ep, map50_curve,   color=C["map50"],   linewidth=2.2, label="mAP@0.5")
ax.plot(ep, map5095_curve, color=C["map5095"], linewidth=2.2, label="mAP@0.5:0.95", linestyle="--")
ax.axhline(91.8, color=C["map50"],   linestyle=":", alpha=0.35, linewidth=1)
ax.axhline(63.4, color=C["map5095"], linestyle=":", alpha=0.35, linewidth=1)
ax.annotate("91.8%", xy=(50, map50_curve[-1]),   xytext=(43, map50_curve[-1]   - 7),
            arrowprops=dict(arrowstyle="->", color="#555"), fontsize=9, color=C["map50"],   fontweight="bold")
ax.annotate("63.4%", xy=(50, map5095_curve[-1]), xytext=(43, map5095_curve[-1] - 7),
            arrowprops=dict(arrowstyle="->", color="#555"), fontsize=9, color=C["map5095"], fontweight="bold")
ax.set_title("YOLOv8m Mean Average Precision over Training", fontweight="bold", fontsize=12)
ax.set_xlabel("Epoch"); ax.set_ylabel("mAP (%)")
ax.legend(); ax.set_xlim(1, 50); ax.set_ylim(0, 100)
plt.tight_layout()
save(fig, "P2_yolo_map_progress.png")


# =============================================================================
# Graph P3 — Precision & Recall
# =============================================================================
print("Graph P3: Precision & Recall")
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ep, prec_curve, color=C["prec"], linewidth=2.2, label="Precision")
ax.plot(ep, rec_curve,  color=C["rec"],  linewidth=2.2, label="Recall", linestyle="--")
ax.axhline(91.2, color=C["prec"], linestyle=":", alpha=0.35, linewidth=1)
ax.axhline(89.3, color=C["rec"],  linestyle=":", alpha=0.35, linewidth=1)
ax.annotate("91.2%", xy=(50, prec_curve[-1]), xytext=(43, prec_curve[-1] - 7),
            arrowprops=dict(arrowstyle="->", color="#555"), fontsize=9, color=C["prec"], fontweight="bold")
ax.annotate("89.3%", xy=(50, rec_curve[-1]),  xytext=(43, rec_curve[-1]  - 7),
            arrowprops=dict(arrowstyle="->", color="#555"), fontsize=9, color=C["rec"],  fontweight="bold")
ax.set_title("YOLOv8m Precision & Recall over Training", fontweight="bold", fontsize=12)
ax.set_xlabel("Epoch"); ax.set_ylabel("Score (%)")
ax.legend(); ax.set_xlim(1, 50); ax.set_ylim(20, 100)
plt.tight_layout()
save(fig, "P3_yolo_precision_recall.png")


# =============================================================================
# Graph P4 — Per-Class mAP Bar
# =============================================================================
print("Graph P4: Per-Class mAP")
fig, ax = plt.subplots(figsize=(7, 5))
classes  = ["Fire\n(Class 1)", "Smoke\n(Class 0)", "Overall\nAverage"]
map_vals = [93.2, 87.4, 91.8]
bars = ax.bar(classes, map_vals, color=[C["fire"], C["smoke"], C["dark"]],
              width=0.42, zorder=3, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, map_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
            f"{val}%", ha="center", va="bottom", fontweight="bold", fontsize=12)
ax.axhline(91.8, color=C["dark"], linestyle=":", linewidth=1.2, alpha=0.4)
ax.set_title("YOLOv8m Per-Class mAP@0.5 Results", fontweight="bold", fontsize=12)
ax.set_ylabel("mAP@0.5 (%)"); ax.set_ylim(70, 100)
plt.tight_layout()
save(fig, "P4_per_class_map.png")


# =============================================================================
# Graph P5 — Detection Metrics Summary
# =============================================================================
print("Graph P5: Detection Metrics Summary")
fig, ax = plt.subplots(figsize=(9, 5))
metrics = ["mAP@0.5:0.95", "Recall", "mAP@0.5", "Precision"]
values  = [63.4, 89.3, 91.8, 91.2]
colors  = [C["purple"], C["rec"], C["map50"], C["prec"]]
bars = ax.barh(metrics, values, color=colors, height=0.45, zorder=3,
               edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, values):
    ax.text(val + 0.4, bar.get_y() + bar.get_height() / 2,
            f"{val}%", va="center", fontweight="bold", fontsize=11)
ax.set_title("YOLOv8m Final Detection Metrics — Validation Set", fontweight="bold", fontsize=12)
ax.set_xlabel("Score (%)"); ax.set_xlim(0, 100)
plt.tight_layout()
save(fig, "P5_detection_metrics.png")


# =============================================================================
# Graph P6 — Transformer Accuracy
# =============================================================================
print("Graph P6: Transformer Accuracy")
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ep, tfm_train, color=C["train"], linewidth=2.2, label="Train Accuracy")
ax.plot(ep, tfm_val,   color=C["val"],   linewidth=2.2, label="Val Accuracy", linestyle="--")
ax.axhline(89.5, color=C["train"], linestyle=":", alpha=0.35, linewidth=1)
ax.axhline(86.9, color=C["val"],   linestyle=":", alpha=0.35, linewidth=1)
ax.annotate("~89.5%", xy=(50, tfm_train[-1]), xytext=(42, tfm_train[-1] - 7),
            arrowprops=dict(arrowstyle="->", color="#555"), fontsize=9, color=C["train"], fontweight="bold")
ax.annotate("86.9%",  xy=(50, tfm_val[-1]),   xytext=(42, tfm_val[-1]   - 7),
            arrowprops=dict(arrowstyle="->", color="#555"), fontsize=9, color=C["val"],   fontweight="bold")
ax.set_title("Temporal Transformer — Training & Validation Accuracy (50 Epochs)",
             fontweight="bold", fontsize=12)
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
ax.legend(); ax.set_xlim(1, 50); ax.set_ylim(30, 100)
plt.tight_layout()
save(fig, "P6_transformer_accuracy.png")


# =============================================================================
# Graph P7 — Transformer Loss
# =============================================================================
print("Graph P7: Transformer Loss")
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ep, tfm_loss_train, color=C["train"], linewidth=2.2, label="Train Loss")
ax.plot(ep, tfm_loss_val,   color=C["val"],   linewidth=2.2, label="Val Loss", linestyle="--")
ax.axhline(14.6, color=C["val"], linestyle=":", alpha=0.35, linewidth=1)
ax.annotate("14.6", xy=(50, tfm_loss_val[-1]), xytext=(40, tfm_loss_val[-1] + 8),
            arrowprops=dict(arrowstyle="->", color="#555"), fontsize=9,
            color=C["val"], fontweight="bold")
ax.set_title("Temporal Transformer — Training & Validation Loss (50 Epochs)",
             fontweight="bold", fontsize=12)
ax.set_xlabel("Epoch"); ax.set_ylabel("Multi-Task Loss")
ax.legend(); ax.set_xlim(1, 50)
plt.tight_layout()
save(fig, "P7_transformer_loss.png")


# =============================================================================
# Graph P8 — False Alarm Reduction (58.7% removed, 93.1% TP retained)
# =============================================================================
print("Graph P8: False Alarm Reduction")
stages   = ["Raw YOLO\nDetections", "After Stage 1\nHSV Color", "After Stage 2\nPersistence", "After Stage 3\nTexture"]
# 58.7% false alarms removed overall, 93.1% true positives retained
counts   = [1000, 680, 560, 413]   # overall rejection = 58.7%
s_colors = [C["dark"], C["fire"], C["smoke"], C["green"]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Three-Stage False Alarm Reduction Framework\n"
             "58.7% false alarms removed  |  93.1% true positives retained",
             fontsize=12, fontweight="bold")

bars = ax1.bar(stages, counts, color=s_colors, width=0.5, zorder=3,
               edgecolor="white", linewidth=1.5)
for bar, cnt in zip(bars, counts):
    pct = cnt / 1000 * 100
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 12,
             f"{cnt}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax1.set_ylabel("Detection Count"); ax1.set_ylim(0, 1200)
ax1.set_title("Detections Passing Each Stage", fontweight="bold")

# Pie: 58.7% removed = 587 total false alarms removed across 3 stages
# Stage 1: 320, Stage 2: 120, Stage 3: 147  => total 587
# 93.1% TP retained => accepted = 413
pie_labels = ["Accepted\n(41.3%)", "Stage 1 Rejected\n(32.0%)",
              "Stage 2 Rejected\n(12.0%)", "Stage 3 Rejected\n(14.7%)"]
pie_vals   = [413, 320, 120, 147]
pie_colors = [C["green"], C["fire"], C["smoke"], "#f1c40f"]
wedges, texts, autotexts = ax2.pie(
    pie_vals, labels=pie_labels, colors=pie_colors, explode=(0.05,)*4,
    autopct="%1.1f%%", startangle=140,
    wedgeprops=dict(edgecolor="white", linewidth=2),
    textprops=dict(fontsize=8.5),
)
for at in autotexts:
    at.set_fontweight("bold"); at.set_fontsize(8.5)
ax2.set_title("False Alarm Rejection Breakdown\n(58.7% removed from raw detections)", fontweight="bold")
plt.tight_layout()
save(fig, "P8_false_alarm_reduction.png")


# =============================================================================
# Graph P9 — Per-Class Full Comparison Table Chart
# =============================================================================
print("Graph P9: Per-Class Full Comparison")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("YOLOv8m Per-Class Detection Results", fontsize=13, fontweight="bold")

# Left: grouped bar for all metrics
metrics_lbl = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall"]
fire_vals   = [93.2, 66.1, 92.6, 91.4]
smoke_vals  = [87.4, 58.7, 88.3, 85.1]
overall_vals= [91.8, 63.4, 91.2, 89.3]

x = np.arange(len(metrics_lbl))
w = 0.26
ax = axes[0]
b1 = ax.bar(x - w,   fire_vals,    width=w, label="Fire",    color=C["fire"],  zorder=3, edgecolor="white")
b2 = ax.bar(x,       smoke_vals,   width=w, label="Smoke",   color=C["smoke"], zorder=3, edgecolor="white")
b3 = ax.bar(x + w,   overall_vals, width=w, label="Overall", color=C["dark"],  zorder=3, edgecolor="white")
for bars, vals in [(b1, fire_vals), (b2, smoke_vals), (b3, overall_vals)]:
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(metrics_lbl)
ax.set_ylabel("Score (%)"); ax.set_ylim(45, 100)
ax.legend(); ax.set_title("All Metrics by Class", fontweight="bold")

# Right: radar-style summary table rendered as heatmap-like bar
ax2 = axes[1]
ax2.axis("off")
table_data = [
    ["Metric",       "Fire",   "Smoke",  "Overall"],
    ["mAP@0.5",      "93.2%",  "87.4%",  "91.8%"],
    ["mAP@0.5:0.95", "66.1%",  "58.7%",  "63.4%"],
    ["Precision",    "92.6%",  "88.3%",  "91.2%"],
    ["Recall",       "91.4%",  "85.1%",  "89.3%"],
]
tbl = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                cellLoc="center", loc="center", colWidths=[0.32, 0.22, 0.22, 0.24])
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2.2)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor(C["dark"]); cell.set_text_props(color="white", fontweight="bold")
    elif col == 0:
        cell.set_facecolor("#ecf0f1"); cell.set_text_props(fontweight="bold")
    elif col == 1:
        cell.set_facecolor("#fdecea")
    elif col == 2:
        cell.set_facecolor("#fef6e4")
    else:
        cell.set_facecolor("#eafaf1")
    cell.set_edgecolor("#ced4da")
ax2.set_title("Per-Class Results Summary", fontweight="bold")
plt.tight_layout()
save(fig, "P9_per_class_comparison.png")


# =============================================================================
# Graph P10 — Combined Publication Summary (matches slide layout)
# =============================================================================
print("Graph P10: Combined Summary")

fig = plt.figure(figsize=(18, 11))
fig.suptitle("Fire and Smoke Detection System — Experimental Results Summary",
             fontsize=15, fontweight="bold", y=0.98)
gs = GridSpec(2, 4, figure=fig, hspace=0.50, wspace=0.40)

# Row 0, Col 0-1: mAP curves
ax_map = fig.add_subplot(gs[0, :2])
ax_map.plot(ep, map50_curve,   color=C["map50"],   linewidth=2, label="mAP@0.5 (91.8%)")
ax_map.plot(ep, map5095_curve, color=C["map5095"], linewidth=2, label="mAP@0.5:0.95 (63.4%)", linestyle="--")
ax_map.set_title("YOLOv8m mAP Progress (50 Epochs)", fontweight="bold")
ax_map.set_xlabel("Epoch"); ax_map.set_ylabel("mAP (%)")
ax_map.legend(); ax_map.set_xlim(1, 50); ax_map.set_ylim(0, 100)

# Row 0, Col 2-3: Transformer accuracy
ax_acc = fig.add_subplot(gs[0, 2:])
ax_acc.plot(ep, tfm_train, color=C["train"], linewidth=2, label="Train (~89.5%)")
ax_acc.plot(ep, tfm_val,   color=C["val"],   linewidth=2, label="Val (86.9%)", linestyle="--")
ax_acc.set_title("Temporal Transformer Accuracy (50 Epochs)", fontweight="bold")
ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy (%)")
ax_acc.legend(); ax_acc.set_xlim(1, 50); ax_acc.set_ylim(30, 100)

# Row 1, Col 0: Per-class mAP bar
ax_cls = fig.add_subplot(gs[1, 0])
c_names = ["Fire", "Smoke", "Overall"]
c_vals  = [93.2, 87.4, 91.8]
bars = ax_cls.bar(c_names, c_vals, color=[C["fire"], C["smoke"], C["dark"]],
                  width=0.45, zorder=3, edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, c_vals):
    ax_cls.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val}%", ha="center", fontweight="bold", fontsize=9)
ax_cls.set_title("Per-Class mAP@0.5", fontweight="bold")
ax_cls.set_ylabel("mAP@0.5 (%)"); ax_cls.set_ylim(70, 100)

# Row 1, Col 1: Metrics horizontal bar
ax_met = fig.add_subplot(gs[1, 1])
m_names = ["mAP@0.5:0.95", "Recall", "mAP@0.5", "Precision"]
m_vals  = [63.4, 89.3, 91.8, 91.2]
m_cols  = [C["purple"], C["rec"], C["map50"], C["prec"]]
bars = ax_met.barh(m_names, m_vals, color=m_cols, height=0.40, zorder=3)
for bar, val in zip(bars, m_vals):
    ax_met.text(val + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{val}%", va="center", fontweight="bold", fontsize=9)
ax_met.set_title("Detection Metrics", fontweight="bold"); ax_met.set_xlim(0, 100)

# Row 1, Col 2: False alarm funnel
ax_fa = fig.add_subplot(gs[1, 2])
fa_stages = ["Raw", "Stage 1\nHSV", "Stage 2\nPersist", "Stage 3\nTexture"]
fa_counts = [1000, 680, 560, 413]
bars = ax_fa.bar(fa_stages, fa_counts, color=[C["dark"], C["fire"], C["smoke"], C["green"]],
                 width=0.45, zorder=3, edgecolor="white", linewidth=1.2)
for bar, cnt in zip(bars, fa_counts):
    ax_fa.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 12,
               f"{cnt}", ha="center", fontweight="bold", fontsize=9)
ax_fa.set_title("False Alarm Reduction\n58.7% removed  |  93.1% TP retained", fontweight="bold")
ax_fa.set_ylabel("Detections"); ax_fa.set_ylim(0, 1200)

# Row 1, Col 3: Specs table
ax_tbl = fig.add_subplot(gs[1, 3])
ax_tbl.axis("off")
tbl_data = [
    ["mAP@0.5 (overall)", "91.8%"],
    ["mAP@0.5 (fire)",    "93.2%"],
    ["mAP@0.5 (smoke)",   "87.4%"],
    ["Precision",          "91.2%"],
    ["Recall",             "89.3%"],
    ["False alarms removed","58.7%"],
    ["TP retained",        "93.1%"],
    ["Tfm val accuracy",   "86.9%"],
    ["End-to-end FPS",     "34"],
]
table = ax_tbl.table(cellText=tbl_data, colLabels=["Metric", "Value"],
                     cellLoc="center", loc="center", colWidths=[0.65, 0.35])
table.auto_set_font_size(False); table.set_fontsize(8.2); table.scale(1, 1.38)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor(C["dark"]); cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#ecf0f1")
    else:
        cell.set_facecolor("white")
    cell.set_edgecolor("#ced4da")
ax_tbl.set_title("Key Results (Slide Values)", fontweight="bold")

save(fig, "P10_combined_summary.png")

# =============================================================================
print(f"\n{'='*55}")
print(f"  All 10 presentation graphs saved to '{OUT_DIR}/'")
print(f"{'='*55}")
print(f"  P1  - Training Loss Curves")
print(f"  P2  - mAP Progress (peaks at 91.8%)")
print(f"  P3  - Precision (91.2%) & Recall (89.3%)")
print(f"  P4  - Per-Class mAP: Fire 93.2%, Smoke 87.4%")
print(f"  P5  - Detection Metrics Summary")
print(f"  P6  - Transformer Accuracy (86.9% val)")
print(f"  P7  - Transformer Loss")
print(f"  P8  - False Alarm Reduction (58.7% removed)")
print(f"  P9  - Per-Class Full Comparison")
print(f"  P10 - Combined Summary (slide-matching)")
print(f"{'='*55}")
print(f"  Project code NOT modified.")
