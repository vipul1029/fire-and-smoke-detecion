# Algorithms — Fire and Smoke Detection System

This document presents the formal algorithms for every component of the system.
Suitable for use directly in a research paper methodology section.

---

## Table of Contents

1. [Master Pipeline Algorithm](#1-master-pipeline-algorithm)
2. [Algorithm 1 — YOLOv8m Fire and Smoke Detection](#algorithm-1--yolov8m-fire-and-smoke-detection)
3. [Algorithm 2 — Stage 1: HSV Color Verification](#algorithm-2--stage-1-hsv-color-verification)
4. [Algorithm 3 — Stage 2: Temporal Persistence Filter](#algorithm-3--stage-2-temporal-persistence-filter)
5. [Algorithm 4 — Stage 3: Laplacian Texture Chaos Analysis](#algorithm-4--stage-3-laplacian-texture-chaos-analysis)
6. [Algorithm 5 — Three-Stage False Alarm Reduction (Combined)](#algorithm-5--three-stage-false-alarm-reduction-combined)
7. [Algorithm 6 — Feature Extraction for Temporal Transformer](#algorithm-6--feature-extraction-for-temporal-transformer)
8. [Algorithm 7 — Temporal Transformer Inference](#algorithm-7--temporal-transformer-inference)
9. [Algorithm 8 — Self-Supervised Labeling for Training](#algorithm-8--self-supervised-labeling-for-training)
10. [Algorithm 9 — Temporal Transformer Training](#algorithm-9--temporal-transformer-training)
11. [Algorithm 10 — Spatial Region Mapping](#algorithm-10--spatial-region-mapping)
12. [Algorithm 11 — Fire Area Trend Analysis](#algorithm-11--fire-area-trend-analysis)
13. [Algorithm 12 — Incident Report Generation](#algorithm-12--incident-report-generation)
14. [Algorithm 13 — Severity Scoring](#algorithm-13--severity-scoring)
15. [Algorithm 14 — YOLOv8m Training](#algorithm-14--yolov8m-training)
16. [Computational Complexity Summary](#computational-complexity-summary)

---

## 1. Master Pipeline Algorithm

This is the top-level algorithm that runs on every incoming video frame.

```
ALGORITHM: FireDetectionPipeline
INPUT:  frame F (H x W x 3 BGR image)
OUTPUT: annotated_frame, detections, stats

────────────────────────────────────────────────────
BEGIN

  frame_idx <- frame_idx + 1

  // MODULE 1: Object Detection
  raw_detections <- YOLOv8m_Detect(F)

  // MODULE 2: False Alarm Reduction
  verified <- []
  FOR EACH det IN raw_detections DO
      IF HSV_Color_Verify(det.crop, det.label) = TRUE THEN
          IF Texture_Chaos_Verify(det.crop, det.label) = TRUE THEN
              verified <- verified + [det]
          ELSE
              texture_rejected <- texture_rejected + 1
      ELSE
          color_rejected <- color_rejected + 1
  END FOR

  confirmed <- Temporal_Persistence_Filter(verified, frame_history)
  frame_history <- UPDATE(frame_history, confirmed)

  // MODULE 3: Fire Progression Prediction
  features <- Extract_Features(confirmed, F.shape)
  feature_window <- APPEND(feature_window, features)

  IF LEN(feature_window) = SEQ_LEN THEN
      prediction <- TemporalTransformer_Predict(feature_window)
  END IF

  // MODULE 4: Incident Report Generation
  IF frame_idx MOD REPORT_INTERVAL = 0 THEN
      LAUNCH_BACKGROUND_THREAD(
          Generate_Incident_Report(confirmed, prediction)
      )
  END IF

  // OUTPUT
  annotated_frame <- Draw_Overlays(F, confirmed, prediction, current_report)
  stats <- Compute_Stats(confirmed, prediction, current_report)

  RETURN annotated_frame, confirmed, stats

END
────────────────────────────────────────────────────
```

---

## Algorithm 1 — YOLOv8m Fire and Smoke Detection

```
ALGORITHM: YOLOv8m_Detect
INPUT:  frame F (H x W x 3), conf_threshold θ_c = 0.25, iou_threshold θ_iou = 0.45
OUTPUT: detections D = {(bbox, confidence, class_id, label, crop)}

────────────────────────────────────────────────────
BEGIN

  // Preprocessing
  F_resized <- RESIZE(F, 640, 640)
  F_norm    <- F_resized / 255.0

  // Forward pass through YOLOv8m
  feature_maps <- Backbone_C2f(F_norm)          // CSP-Darknet + C2f modules
  neck_out     <- PANet_FPN(feature_maps)        // Path Aggregation Network
  raw_boxes    <- Decoupled_Head(neck_out)       // Anchor-free detection head

  // Post-processing
  filtered <- {b IN raw_boxes | b.confidence >= θ_c}
  final    <- Non_Maximum_Suppression(filtered, θ_iou)

  D <- []
  FOR EACH box IN final DO
      x1, y1, x2, y2 <- CLAMP(box.xyxy, 0, W, 0, H)
      IF (x2 > x1) AND (y2 > y1) THEN
          crop  <- F[y1:y2, x1:x2]
          label <- class_names[box.class_id]   // 0->smoke, 1->fire
          D <- D + [{bbox:[x1,y1,x2,y2], conf:box.conf, class_id:box.cls,
                     label:label, crop:crop}]
      END IF
  END FOR

  RETURN D

END
────────────────────────────────────────────────────

Model parameters:
  depth_multiple  = 0.67
  width_multiple  = 0.75
  total_params    ≈ 25.9 million
  input_size      = 640 x 640
  classes         = {0: smoke, 1: fire}
```

---

## Algorithm 2 — Stage 1: HSV Color Verification

```
ALGORITHM: HSV_Color_Verify
INPUT:  crop C (h x w x 3 BGR),  label L ∈ {"fire", "smoke"}
OUTPUT: ACCEPT (True) or REJECT (False)

────────────────────────────────────────────────────
BEGIN

  IF C is empty OR C has zero pixels THEN
      RETURN REJECT
  END IF

  C_hsv <- BGR_to_HSV(C)         // H:[0,180], S:[0,255], V:[0,255]

  IF L = "fire" THEN

      // Fire: red-orange-yellow spectrum with high saturation and brightness
      mask_1 <- PIXELS WHERE:
          H ∈ [0,  35]   AND
          S ∈ [100, 255] AND
          V ∈ [100, 255]

      // Deep red wraparound in OpenCV HSV
      mask_2 <- PIXELS WHERE:
          H ∈ [160, 180] AND
          S ∈ [100, 255] AND
          V ∈ [100, 255]

      mask      <- mask_1 OR mask_2
      threshold <- 0.12

  ELSE IF L = "smoke" THEN

      // Smoke: achromatic grey/white, low saturation, medium brightness
      mask <- PIXELS WHERE:
          H ∈ [0,  180] AND
          S ∈ [0,   60] AND
          V ∈ [80,  220]

      threshold <- 0.10

  END IF

  pixel_ratio <- COUNT_NONZERO(mask) / TOTAL_PIXELS(C)

  IF pixel_ratio > threshold THEN
      RETURN ACCEPT
  ELSE
      RETURN REJECT
  END IF

END
────────────────────────────────────────────────────

Rejects:
  - Sunsets         (fire hue but wrong saturation/spatial extent)
  - Tail lights     (small red area, ratio too low)
  - Orange barriers (correct hue but fails Stage 3 texture check)
  - White walls     (V too high for smoke, S too high for smoke)
```

---

## Algorithm 3 — Stage 2: Temporal Persistence Filter

```
ALGORITHM: Temporal_Persistence_Filter
INPUT:  detections D_t at frame t,
        frame_history H = [bboxes_{t-W}, ..., bboxes_{t-1}],
        window_size W = 3,
        iou_threshold θ = 0.10
OUTPUT: confirmed subset of D_t

────────────────────────────────────────────────────
BEGIN

  // Grace period: not enough history yet
  IF LEN(H) < W - 1 THEN
      RETURN D_t          // accept all during startup
  END IF

  confirmed <- []

  FOR EACH det IN D_t DO

      is_persistent <- FALSE

      FOR EACH past_frame_bboxes IN H DO
          FOR EACH past_bbox IN past_frame_bboxes DO

              iou <- Compute_IoU(det.bbox, past_bbox)

              IF iou > θ THEN
                  is_persistent <- TRUE
                  BREAK
              END IF

          END FOR
          IF is_persistent THEN BREAK
      END FOR

      IF is_persistent THEN
          confirmed <- confirmed + [det]
      ELSE
          persistence_rejected <- persistence_rejected + 1
      END IF

  END FOR

  // Update history (sliding window)
  H <- H + [bboxes(confirmed)]
  IF LEN(H) > W THEN
      H <- H[1:]          // remove oldest frame
  END IF

  RETURN confirmed

END

────────────────────────────────────────────────────

SUB-ALGORITHM: Compute_IoU
INPUT:  box_a = [x1_a, y1_a, x2_a, y2_a]
        box_b = [x1_b, y1_b, x2_b, y2_b]
OUTPUT: iou ∈ [0, 1]

  inter_x1 <- MAX(x1_a, x1_b)
  inter_y1 <- MAX(y1_a, y1_b)
  inter_x2 <- MIN(x2_a, x2_b)
  inter_y2 <- MIN(y2_a, y2_b)

  inter_w  <- MAX(0, inter_x2 - inter_x1)
  inter_h  <- MAX(0, inter_y2 - inter_y1)
  inter    <- inter_w × inter_h

  IF inter = 0 THEN RETURN 0.0

  area_a   <- (x2_a - x1_a) × (y2_a - y1_a)
  area_b   <- (x2_b - x1_b) × (y2_b - y1_b)
  union    <- area_a + area_b - inter

  RETURN inter / union

────────────────────────────────────────────────────

Rejects:
  - Camera flashes       (disappear after 1 frame)
  - Passing reflections  (1-2 frames only)
  - Transient glare      (brief lighting spike)
```

---

## Algorithm 4 — Stage 3: Laplacian Texture Chaos Analysis

```
ALGORITHM: Texture_Chaos_Verify
INPUT:  crop C (h x w x 3 BGR),  label L ∈ {"fire", "smoke"}
OUTPUT: ACCEPT (True) or REJECT (False)

────────────────────────────────────────────────────
BEGIN

  IF C is empty OR C has zero pixels THEN
      RETURN REJECT
  END IF

  C_gray <- BGR_to_GRAY(C)

  // Laplacian operator: second-order spatial derivative
  // Measures rate of intensity change — proxy for texture complexity
  L <- Laplacian(C_gray)    // cv2.Laplacian(C_gray, CV_64F)

  texture_score <- VARIANCE(L)

  // Fire has more chaotic texture than smoke
  IF L = "fire" THEN
      threshold <- 200.0
  ELSE IF L = "smoke" THEN
      threshold <- 80.0
  END IF

  IF texture_score > threshold THEN
      RETURN ACCEPT
  ELSE
      RETURN REJECT
  END IF

END
────────────────────────────────────────────────────

Mathematical definition:
  Laplacian: ∇²I(x,y) = ∂²I/∂x² + ∂²I/∂y²
  texture_score = Var(∇²I) = E[(∇²I)²] - (E[∇²I])²

Rejects:
  - Painted walls          (uniform, very low variance)
  - Traffic signs          (smooth surface, low variance)
  - Indicator LEDs         (solid color pixel block)
  - Sunlit building faces  (gradual uniform gradient)
```

---

## Algorithm 5 — Three-Stage False Alarm Reduction (Combined)

```
ALGORITHM: Three_Stage_False_Alarm_Reduction
INPUT:  raw detections D from YOLOv8m, frame F, frame_history H
OUTPUT: verified detections V, updated frame_history H

────────────────────────────────────────────────────
BEGIN

  // Stage 1 + Stage 3 applied per-detection
  stage_1_3_passed <- []

  FOR EACH det IN D DO

      // Stage 1: HSV Color Check
      color_ok <- HSV_Color_Verify(det.crop, det.label)

      IF NOT color_ok THEN
          color_rejected <- color_rejected + 1
          CONTINUE              // skip Stage 3, reject immediately
      END IF

      // Stage 3: Texture Chaos Check
      // (Stage 3 run after Stage 1 — saves computation on already-rejected boxes)
      texture_ok <- Texture_Chaos_Verify(det.crop, det.label)

      IF NOT texture_ok THEN
          texture_rejected <- texture_rejected + 1
          CONTINUE
      END IF

      stage_1_3_passed <- stage_1_3_passed + [det]

  END FOR

  // Stage 2: Temporal Persistence
  V, H <- Temporal_Persistence_Filter(stage_1_3_passed, H)

  RETURN V, H

END
────────────────────────────────────────────────────

Stage ordering rationale:
  Stage 1 (HSV) runs first   — fastest, eliminates most false alarms
  Stage 3 (Texture) runs second — medium cost, applied to Stage 1 survivors
  Stage 2 (Persistence) runs last — requires history, applied to survivors of 1+3
  Stage 2 position: in pipeline.py (not detector.py) because it needs frame history
```

---

## Algorithm 6 — Feature Extraction for Temporal Transformer

```
ALGORITHM: Extract_Features
INPUT:  detections D (list), frame_shape (H, W, C)
OUTPUT: feature_vector f ∈ R^8

────────────────────────────────────────────────────
BEGIN

  H, W    <- frame_shape[0], frame_shape[1]
  F_area  <- H × W

  fire_dets  <- {d IN D | d.label = "fire"}
  smoke_dets <- {d IN D | d.label = "smoke"}

  // Feature 0: normalized total fire area
  fire_area  <- SUM(d.width × d.height FOR d IN fire_dets) / F_area

  // Feature 1: normalized total smoke area
  smoke_area <- SUM(d.width × d.height FOR d IN smoke_dets) / F_area

  // Feature 2: fire detection count
  fire_count  <- LEN(fire_dets)

  // Feature 3: smoke detection count
  smoke_count <- LEN(smoke_dets)

  // Feature 4: mean confidence across all detections
  IF LEN(D) > 0 THEN
      avg_conf <- MEAN(d.confidence FOR d IN D)
  ELSE
      avg_conf <- 0.0
  END IF

  // Feature 5: fire area growth rate since previous frame
  growth_rate <- fire_area - prev_fire_area
  prev_fire_area <- fire_area

  // Features 6-7: normalized centroid of fire detections
  IF LEN(fire_dets) > 0 THEN
      cx <- MEAN((d.bbox[0] + d.bbox[2]) / 2 FOR d IN fire_dets) / W
      cy <- MEAN((d.bbox[1] + d.bbox[3]) / 2 FOR d IN fire_dets) / H
  ELSE
      cx <- 0.0
      cy <- 0.0
  END IF

  f <- [fire_area, smoke_area, fire_count, smoke_count,
        avg_conf, growth_rate, cx, cy]

  RETURN f

END

────────────────────────────────────────────────────

ALGORITHM: Update_Sliding_Window
INPUT:  feature_vector f ∈ R^8, window W (list), SEQ_LEN = 15
OUTPUT: sequence S ∈ R^(15×8) if full, else NULL

  W <- APPEND(W, f)
  IF LEN(W) > SEQ_LEN THEN
      W <- W[1:]          // drop oldest frame
  END IF

  IF LEN(W) = SEQ_LEN THEN
      RETURN W            // sequence ready for Temporal Transformer
  ELSE
      RETURN NULL
  END IF
```

---

## Algorithm 7 — Temporal Transformer Inference

```
ALGORITHM: TemporalTransformer_Predict
INPUT:  sequence S ∈ R^(15×8), scaler_mean μ ∈ R^8, scaler_std σ ∈ R^8
OUTPUT: {growth_label, growth_confidence, risk_score, area_5s, area_10s}

────────────────────────────────────────────────────
BEGIN

  // Normalize using training statistics
  S_norm <- (S - μ) / (σ + ε)     where ε = 1e-8

  X <- TENSOR(S_norm).unsqueeze(0)    // shape: (1, 15, 8)

  // Forward pass
  // 1. Linear projection
  X <- Linear(8 -> 64)(X)            // shape: (1, 15, 64)

  // 2. Sinusoidal positional encoding
  FOR pos IN 0..14 DO
      FOR i IN 0..31 DO
          X[0, pos, 2i]   <- X[0, pos, 2i]   + sin(pos / 10000^(2i/64))
          X[0, pos, 2i+1] <- X[0, pos, 2i+1] + cos(pos / 10000^(2i/64))
      END FOR
  END FOR
  X <- Dropout(0.1)(X)               // inference: dropout disabled

  // 3. Transformer encoder (3 layers)
  FOR layer IN 1..3 DO
      // Multi-head self-attention (4 heads)
      Q, K, V  <- Linear(64, 64) each applied to X
      Q, K, V  <- SPLIT into 4 heads of dim 16 each
      attn     <- SOFTMAX(Q × K^T / √16) × V
      attn     <- CONCAT(heads) -> Linear(64, 64)
      X        <- LayerNorm(X + attn)           // residual + norm

      // Feed-forward
      ff <- Linear(64, 256)(X) -> ReLU -> Linear(256, 64)
      X  <- LayerNorm(X + ff)                   // residual + norm
  END FOR

  // 4. Final norm + mean pooling over time
  X <- LayerNorm(64)(X)              // shape: (1, 15, 64)
  X <- MEAN(X, dim=1)                // shape: (1, 64)

  // 5. Output heads
  growth_logits <- Linear(64,32) -> ReLU -> Linear(32,3)  applied to X
  risk_raw      <- Linear(64,32) -> ReLU -> Linear(32,1) -> Sigmoid  applied to X
  area_5s_raw   <- Linear(64,32) -> ReLU -> Linear(32,1) -> ReLU  applied to X
  area_10s_raw  <- Linear(64,32) -> ReLU -> Linear(32,1) -> ReLU  applied to X

  // 6. Post-process
  probs              <- Softmax(growth_logits)
  class_id           <- ARGMAX(probs)
  growth_label       <- {0:"stable", 1:"growing", 2:"critical"}[class_id]
  growth_confidence  <- probs[class_id] × 100
  risk_score         <- MIN(risk_raw × 100, 100.0)
  area_5s            <- area_5s_raw
  area_10s           <- area_10s_raw

  RETURN {
      growth_label:      growth_label,
      growth_confidence: ROUND(growth_confidence, 1),
      risk_score:        ROUND(risk_score, 1),
      area_5s:           ROUND(area_5s, 6),
      area_10s:          ROUND(area_10s, 6),
      available:         True
  }

END
────────────────────────────────────────────────────

Architecture summary:
  Input:   (B, T=15, F=8)
  d_model: 64
  heads:   4   (head_dim = 64/4 = 16)
  layers:  3
  ff_dim:  256
  params:  ~85,000
```

---

## Algorithm 8 — Self-Supervised Labeling for Training

```
ALGORITHM: Self_Supervised_Label
INPUT:  all_features (list of per-frame feature vectors for one video),
        fps = video frame rate,
        GROW_THRESH_CRITICAL = 0.60,
        GROW_THRESH_GROWING  = 0.30,
        SEQ_LEN = 15
OUTPUT: sequences, labels, area_5s_targets, area_10s_targets, risk_scores

────────────────────────────────────────────────────
BEGIN

  look_5s  <- MAX(1, INT(fps × 5))    // frames = 5 seconds
  look_10s <- MAX(1, INT(fps × 10))   // frames = 10 seconds
  N        <- LEN(all_features)

  sequences <- []
  labels    <- []
  area_5s   <- []
  area_10s  <- []
  risks     <- []

  FOR t IN SEQ_LEN .. (N - look_5s - 1) DO

      // Extract sliding window sequence
      seq <- all_features[t - SEQ_LEN : t]   // shape (15, 8)

      // Current and future fire area (feature index 0)
      current_area <- seq[-1][0]
      future_5s    <- all_features[MIN(t + look_5s,  N-1)][0]
      future_10s   <- all_features[MIN(t + look_10s, N-1)][0]

      // Compute growth ratio over next 5 seconds
      IF current_area > 1e-6 THEN
          growth_5s <- (future_5s - current_area) / current_area
      ELSE
          growth_5s <- 0.0
      END IF

      // Assign class label
      IF growth_5s >= GROW_THRESH_CRITICAL THEN
          label <- 2    // critical
      ELSE IF growth_5s >= GROW_THRESH_GROWING THEN
          label <- 1    // growing
      ELSE
          label <- 0    // stable
      END IF

      // Risk score as continuous target
      risk <- MIN(100.0, ABS(growth_5s) × 100.0)

      sequences <- sequences + [seq]
      labels    <- labels    + [label]
      area_5s   <- area_5s   + [future_5s]
      area_10s  <- area_10s  + [future_10s]
      risks     <- risks     + [risk]

  END FOR

  RETURN sequences, labels, area_5s, area_10s, risks

END

────────────────────────────────────────────────────

ALGORITHM: Gaussian_Noise_Augmentation
INPUT:  sequences S, labels L, targets (a5, a10, risks),
        COPIES = 5, NOISE_STD = 0.01
OUTPUT: augmented versions of all inputs (appended to originals)

  aug_S, aug_L, aug_a5, aug_a10, aug_r <- []

  FOR copy IN 1..COPIES DO
      FOR EACH (seq, lab, a5, a10, r) IN zip(S, L, a5_list, a10_list, risks) DO
          noise     <- GAUSSIAN(mean=0, std=NOISE_STD, shape=seq.shape)
          noisy_seq <- seq + noise
          aug_S     <- aug_S  + [noisy_seq]
          aug_L     <- aug_L  + [lab]
          aug_a5    <- aug_a5  + [a5]
          aug_a10   <- aug_a10 + [a10]
          aug_r     <- aug_r   + [r]
      END FOR
  END FOR

  RETURN aug_S, aug_L, aug_a5, aug_a10, aug_r
```

---

## Algorithm 9 — Temporal Transformer Training

```
ALGORITHM: Train_TemporalTransformer
INPUT:  video_dir, epochs=50, batch_size=64, lr=0.001
OUTPUT: trained model saved to src/weights/fire_predictor.pt

────────────────────────────────────────────────────
BEGIN

  // Step 1: Extract features from all videos
  all_features_per_video <- []
  FOR EACH video IN video_dir DO
      pipeline <- FireDetector(conf=0.20, verification=False)
      frame_features <- []
      FOR EACH frame IN video DO
          dets     <- pipeline.detect(frame)
          features <- Extract_Features(dets, frame.shape)
          frame_features <- frame_features + [features]
      END FOR
      all_features_per_video <- all_features_per_video + [frame_features]
  END FOR

  // Step 2: Build sequences with self-supervised labels
  all_seqs, all_labels, all_a5, all_a10, all_risks <- [], [], [], [], []
  FOR EACH (features, fps) IN all_features_per_video DO
      IF LEN(features) < SEQ_LEN + fps*5 THEN CONTINUE  // too short
      seqs, labs, a5, a10, risks <- Self_Supervised_Label(features, fps)
      all_seqs   <- all_seqs   + seqs
      all_labels <- all_labels + labs
      all_a5     <- all_a5     + a5
      all_a10    <- all_a10    + a10
      all_risks  <- all_risks  + risks
  END FOR

  // Step 3: Normalize features
  X <- ARRAY(all_seqs)          // shape (N, 15, 8)
  μ <- MEAN(X.reshape(-1, 8), axis=0)
  σ <- STD(X.reshape(-1, 8),  axis=0) + 1e-8
  X <- (X - μ) / σ

  // Step 4: Augmentation
  aug_seqs, aug_labs, aug_a5, aug_a10, aug_r <-
      Gaussian_Noise_Augmentation(X, all_labels, all_a5, all_a10, all_risks)
  X_all      <- CONCAT(X, aug_seqs)
  labels_all <- CONCAT(all_labels, aug_labs)
  // (similarly for area and risk targets)

  // Step 5: Train/val split
  indices <- SHUFFLE(0..LEN(X_all)-1)
  split   <- INT(0.8 × LEN(indices))
  train_idx, val_idx <- indices[:split], indices[split:]

  // Step 6: Class-balanced sampler
  class_counts <- BINCOUNT(labels_all[train_idx])
  weights      <- 1.0 / (class_counts + 1e-8)
  sample_wts   <- [weights[labels_all[i]] FOR i IN train_idx]
  sampler      <- WeightedRandomSampler(sample_wts)

  // Step 7: Initialize model and optimizer
  model     <- TemporalTransformerModel()
  optimizer <- Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
  scheduler <- ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

  best_val_loss     <- INF
  patience_counter  <- 0
  PATIENCE          <- 10

  // Step 8: Training loop
  FOR epoch IN 1..epochs DO

      // Training
      model.train()
      train_loss <- 0.0
      FOR EACH batch IN DataLoader(train_data, sampler=sampler) DO
          X_b, y_lab, y_a5, y_a10, y_risk <- batch

          logits, risk, a5, a10 <- model(X_b)

          // Multi-task loss
          L_class <- CrossEntropyLoss(logits, y_lab)
          L_risk  <- MSELoss(risk,  y_risk)
          L_a5    <- MSELoss(a5,    y_a5)
          L_a10   <- MSELoss(a10,   y_a10)
          L_total <- L_class + 0.3×L_risk + 0.2×L_a5 + 0.2×L_a10

          optimizer.zero_grad()
          L_total.backward()
          CLIP_GRAD_NORM(model.parameters(), max_norm=1.0)
          optimizer.step()

          train_loss <- train_loss + L_total × LEN(batch)
      END FOR
      train_loss <- train_loss / LEN(train_data)

      // Validation
      model.eval()
      val_loss <- 0.0
      FOR EACH batch IN DataLoader(val_data) DO
          WITH no_grad:
              logits, risk, a5, a10 <- model(X_b)
              L_total <- CrossEntropyLoss + 0.3*MSE_risk + 0.2*MSE_a5 + 0.2*MSE_a10
              val_loss <- val_loss + L_total × LEN(batch)
      END FOR
      val_loss <- val_loss / LEN(val_data)

      scheduler.step(val_loss)

      // Early stopping
      IF val_loss < best_val_loss THEN
          best_val_loss    <- val_loss
          best_state       <- COPY(model.state_dict())
          patience_counter <- 0
      ELSE
          patience_counter <- patience_counter + 1
          IF patience_counter >= PATIENCE THEN
              BREAK    // early stopping
          END IF
      END IF

  END FOR

  // Step 9: Save checkpoint
  SAVE {
      model_state: best_state,
      scaler_mean: μ.tolist(),
      scaler_std:  σ.tolist(),
      val_loss:    best_val_loss,
      seq_len:     15,
      feature_dim: 8,
  } -> "src/weights/fire_predictor.pt"

END
────────────────────────────────────────────────────

Loss weights:
  Classification (cross-entropy): 1.0   (primary task)
  Risk regression (MSE):          0.3   (secondary)
  Area-5s regression (MSE):       0.2   (secondary)
  Area-10s regression (MSE):      0.2   (secondary)
```

---

## Algorithm 10 — Spatial Region Mapping

```
ALGORITHM: Map_to_Region
INPUT:  detections D, frame_shape (H, W, C)
OUTPUT: region_name ∈ {"northwest", "northern", "northeast",
                        "western",  "central",  "eastern",
                        "southwest","southern", "southeast",
                        "monitored area"}

────────────────────────────────────────────────────
BEGIN

  IF LEN(D) = 0 THEN
      RETURN "monitored area"
  END IF

  H, W <- frame_shape[0], frame_shape[1]

  // Prefer fire bbox centroid over smoke
  fire_dets <- {d IN D | d.label = "fire"}
  ref       <- fire_dets[0] IF fire_dets ELSE D[0]

  // Compute normalized centroid
  cx <- (ref.bbox[0] + ref.bbox[2]) / 2 / W    // ∈ [0, 1]
  cy <- (ref.bbox[1] + ref.bbox[3]) / 2 / H    // ∈ [0, 1]

  // Map to 3x3 grid index
  col <- MIN(INT(cx × 3), 2)    // 0=left, 1=center, 2=right
  row <- MIN(INT(cy × 3), 2)    // 0=top,  1=middle, 2=bottom

  // Grid lookup
  GRID <- {
      (0,0):"northwest", (0,1):"northern",  (0,2):"northeast",
      (1,0):"western",   (1,1):"central",   (1,2):"eastern",
      (2,0):"southwest", (2,1):"southern",  (2,2):"southeast"
  }

  RETURN GRID[(row, col)] + " region"

END
```

---

## Algorithm 11 — Fire Area Trend Analysis

```
ALGORITHM: Compute_Trend
INPUT:  area_history A (list of last 60 normalized fire area values)
OUTPUT: trend ∈ {"increasing", "decreasing", "stable"}

────────────────────────────────────────────────────
BEGIN

  IF LEN(A) < 10 THEN
      RETURN "stable"    // insufficient history
  END IF

  // Linear slope over last 10 frames
  window <- A[-10:]      // most recent 10 values

  slope <- (window[-1] - window[0]) / LEN(window)

  IF slope > +0.005 THEN
      RETURN "increasing"
  ELSE IF slope < -0.005 THEN
      RETURN "decreasing"
  ELSE
      RETURN "stable"
  END IF

END

────────────────────────────────────────────────────

Threshold rationale:
  0.005 per frame ≈ 0.5% fire area change per frame
  At 30fps: 0.005 × 10 frames = 5% area change over 10 frames
  Below this is considered camera noise rather than real growth.
```

---

## Algorithm 12 — Incident Report Generation

```
ALGORITHM: Generate_Incident_Report
INPUT:  detections D, stats S, frame_idx t, frame_shape,
        prediction P (from Temporal Transformer),
        report_interval = 30
OUTPUT: report {description, recommendation, severity, region,
                consecutive_frames, trend, timestamp, source}

────────────────────────────────────────────────────
BEGIN

  // Maintain consecutive frame counter
  IF S.total_hazards > 0 THEN
      consecutive_frames <- consecutive_frames + 1
  ELSE
      consecutive_frames <- 0
      RETURN {active: False}
  END IF

  // Track fire area for trend
  fire_px  <- SUM((d.x2-d.x1)*(d.y2-d.y1) FOR d IN D IF d.label="fire")
  area_norm <- fire_px / (H * W)
  area_history <- APPEND(area_history, area_norm)

  // Check if interval has elapsed
  IF (t - last_report_frame) < report_interval THEN
      RETURN last_report           // keep showing previous report
  END IF

  // Do not start new generation if one is in progress
  IF generating = TRUE THEN
      RETURN last_report
  END IF

  last_report_frame <- t
  generating        <- TRUE

  // Compute report inputs
  region     <- Map_to_Region(D, frame_shape)
  trend      <- Compute_Trend(area_history)
  avg_conf   <- S.avg_confidence
  severity   <- S.severity_label
  fire_count <- S.fire_count
  smoke_count<- S.smoke_count

  // Build prompt for LLM
  prompt <- Build_Prompt(fire_count, smoke_count, avg_conf,
                          region, trend, consecutive_frames,
                          severity, P)

  // Launch background thread (non-blocking)
  LAUNCH_THREAD: Worker(prompt, context)

  RETURN last_report     // previous report stays visible while thread runs

END

────────────────────────────────────────────────────

ALGORITHM: Worker (runs in background thread)
INPUT:  prompt, context (fire_count, smoke_count, severity, region, ...)
OUTPUT: sets ready_report

  // Try Ollama LLM first
  response <- Call_Ollama(prompt, timeout=10s)

  IF response IS NOT NULL THEN
      lines           <- SPLIT(response, by newline)
      description     <- lines[0]
      recommendation  <- lines[1] IF LEN(lines)>1 ELSE Rule_Recommendation(severity)
      source          <- "llm"
  ELSE
      // Fallback: rule-based NLG
      description    <- Rule_Description(context)
      recommendation <- Rule_Recommendation(severity, fire_count)
      source         <- "rule-based"
  END IF

  ready_report <- {
      active:             True,
      description:        description,
      recommendation:     recommendation,
      severity:           severity,
      region:             region,
      consecutive_frames: consecutive_frames,
      trend:              trend,
      timestamp:          CURRENT_TIME(),
      source:             source
  }
  generating <- FALSE

────────────────────────────────────────────────────

ALGORITHM: Call_Ollama
INPUT:  prompt (str), timeout = 10 seconds
OUTPUT: response text OR NULL on failure

  payload <- JSON {
      model:   "phi3:mini",
      prompt:  prompt,
      stream:  False,
      options: {temperature: 0.3, num_predict: 120}
  }

  TRY:
      response <- HTTP_POST("http://localhost:11434/api/generate",
                             body=payload, timeout=timeout)
      RETURN response["response"].strip()
  CATCH timeout OR connection_error:
      RETURN NULL
```

---

## Algorithm 13 — Severity Scoring

```
ALGORITHM: Compute_Severity
INPUT:  fire_count F, smoke_count S, fps
OUTPUT: severity_score ∈ [0, 100], severity_label ∈ {critical, high, medium, low}

────────────────────────────────────────────────────
BEGIN

  IF F + S = 0 THEN
      severity_score <- 0.0
      severity_label <- "low"
      RETURN severity_score, severity_label
  END IF

  severity_score <- MIN(100.0, (F × 45) + (S × 20) + (fps × 2))

  IF severity_score >= 70 THEN
      severity_label <- "critical"
  ELSE IF severity_score >= 40 THEN
      severity_label <- "high"
  ELSE IF severity_score >= 15 THEN
      severity_label <- "medium"
  ELSE
      severity_label <- "low"
  END IF

  RETURN severity_score, severity_label

END

────────────────────────────────────────────────────

Weight rationale:
  fire  × 45: fire is more severe than smoke (45 = critical threshold / 2)
  smoke × 20: smoke indicates early-stage or concealed fire
  fps   × 2:  higher FPS means more frames analyzed = higher confidence
              (max fps contribution ≈ 60fps × 2 = 120, capped at 100)
```

---

## Algorithm 14 — YOLOv8m Training

```
ALGORITHM: Train_YOLOv8m
INPUT:  dataset_dir, epochs=50, batch=8, imgsz=640
OUTPUT: best weights saved to src/weights/fire_detection_yolov8m.pt

────────────────────────────────────────────────────
BEGIN

  // Step 1: Dataset preparation
  train_path <- FIND("dataset_dir/**/train/images")
  val_path   <- FIND("dataset_dir/**/valid/images")

  data_yaml <- {
      path:  dataset_dir,
      train: train_path,
      val:   val_path,
      nc:    2,
      names: {0: "smoke", 1: "fire"}
  }
  WRITE(data_yaml -> "fire_data.yaml")

  // Step 2: Load pretrained model
  model <- YOLOv8m(pretrained="yolov8m.pt")   // COCO pretrained

  // Step 3: Train
  results <- model.train(
      data          = "fire_data.yaml",
      epochs        = 50,
      imgsz         = 640,
      batch         = 8,
      device        = 0,           // GPU
      amp           = True,        // mixed precision (BF16/FP16)
      patience      = 15,          // early stopping
      optimizer     = "auto",      // SGD
      lr0           = 0.01,
      lrf           = 0.01,
      momentum      = 0.937,
      weight_decay  = 0.0005,
      warmup_epochs = 3,
      mosaic        = 1.0,
      close_mosaic  = 10,
      augment       = True
  )

  // Step 4: Evaluate best weights
  best_weights <- results.save_dir / "weights/best.pt"
  metrics      <- model.val(data="fire_data.yaml")

  // Step 5: Verify architecture
  ckpt <- LOAD(best_weights)
  ASSERT ckpt.model.yaml.depth_multiple ≈ 0.67  // YOLOv8m check
  ASSERT ckpt.model.yaml.width_multiple ≈ 0.75

  // Step 6: Copy to project
  COPY(best_weights -> "src/weights/fire_detection_yolov8m.pt")

END

────────────────────────────────────────────────────

YOLO loss functions (standard):
  L_box  = IoU loss on bounding box regression
  L_cls  = Binary cross-entropy on class probabilities
  L_dfl  = Distribution Focal Loss on box regression
  L_total = L_box + L_cls + L_dfl
```

---

## Computational Complexity Summary

| Algorithm | Complexity | Notes |
|-----------|-----------|-------|
| YOLOv8m detection | O(H × W) | ~640² operations, GPU-parallelized |
| Stage 1: HSV color | O(w × h) per box | w,h = bbox dimensions |
| Stage 2: Persistence | O(D × W × B) | D=detections, W=window, B=past bboxes |
| Stage 3: Texture | O(w × h) per box | Laplacian = 2D convolution |
| Feature extraction | O(D) | Linear in number of detections |
| Sliding window update | O(1) amortized | Queue append + pop |
| Temporal Transformer | O(T² × d) per layer | T=15, d=64, 3 layers |
| Incident reporter | O(1) | Text template or async LLM call |
| **Total per frame** | **O(H × W)** | **Dominated by YOLO** |

**Runtime target:** ≥ 24 FPS on NVIDIA RTX 4050 (6 GB VRAM) with all modules active.

---

## Data Flow Diagram

```
VIDEO INPUT
    |
    | frame F (H x W x 3)
    v
[YOLO INFERENCE]
    | D_raw = {bbox, conf, class_id, label, crop}
    v
[STAGE 1: HSV COLOR] -----> REJECT -> color_rejected++
    | D_color_passed
    v
[STAGE 3: TEXTURE]   -----> REJECT -> texture_rejected++
    | D_texture_passed
    v
[STAGE 2: PERSISTENCE] ---> REJECT -> persistence_rejected++
    | D_confirmed (verified detections)
    v
    +---> [FEATURE EXTRACTION]
    |         | f_t ∈ R^8
    |         v
    |     [SLIDING WINDOW] (15 frames)
    |         | S ∈ R^(15×8) when full
    |         v
    |     [TEMPORAL TRANSFORMER]
    |         | {growth_label, risk_score, area_5s, area_10s}
    |         v
    +-------> PREDICTION P
    |
    +---> [INCIDENT REPORTER] (every 30 frames, background thread)
    |         | {description, recommendation, source}
    |         v
    |     REPORT R
    |
    v
[ANNOTATE FRAME]
    | - bounding boxes (fire=red, smoke=orange)
    | - HUD: FPS, fire count, smoke count
    | - prediction overlay: PREDICT:CRITICAL (84%) Risk:91/100
    | - report overlay: description + recommendation
    v
ANNOTATED OUTPUT FRAME
```
