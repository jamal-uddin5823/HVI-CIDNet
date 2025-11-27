#!/bin/bash

# Training Script with Hard Negative Mining and Identity-Balanced Sampling
# This script demonstrates the new improved training strategies

set -e
set -o pipefail

echo "================================================================================"
echo "DISCRIMINATIVE LOSS TRAINING WITH HARD NEGATIVE MINING"
echo "================================================================================"
echo ""

# Configuration
DATASET="lapaface"  # or "lfw"
EXPERIMENT_NAME="discriminative_fr0.5_hardneg"
OUTPUT_DIR="./weights/${EXPERIMENT_NAME}"
LOG_DIR="./logs/${EXPERIMENT_NAME}"

# Create directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

# Timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

echo "Starting training at $(date)" | tee "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Training with baseline (no discriminative loss)
echo "================================================================================" | tee -a "$LOG_FILE"
echo "BASELINE TRAINING (D_weight=1.5, no face loss)" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"

python train.py \
    --lapaface \
    --batchSize 8 \
    --cropSize 256 \
    --nEpochs 100 \
    --lr 1e-4 \
    --snapshots 10 \
    --D_weight 1.5 \
    --L1_weight 1.0 \
    --E_weight 50.0 \
    --P_weight 0.01 \
    --HVI_weight 1.0 \
    2>&1 | tee -a "$LOG_FILE"

# Move weights
mv ./weights/train ./weights/baseline_d1.5
echo "✓ Baseline training complete" | tee -a "$LOG_FILE"

# Training with discriminative loss (standard circular shift)
echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "DISCRIMINATIVE TRAINING (FR=0.5, circular shift impostors)" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"

python train.py \
    --lapaface \
    --batchSize 8 \
    --cropSize 256 \
    --nEpochs 100 \
    --lr 1e-4 \
    --snapshots 10 \
    --D_weight 1.5 \
    --L1_weight 1.0 \
    --E_weight 50.0 \
    --P_weight 0.01 \
    --HVI_weight 1.0 \
    --use_face_loss \
    --FR_weight 0.5 \
    --FR_model_path ./weights/adaface/adaface_ir50_webface4m.ckpt \
    --contrastive_margin 0.4 \
    --contrastive_weight 1.0 \
    --triplet_margin 0.2 \
    --triplet_weight 0.5 \
    --face_temperature 0.07 \
    2>&1 | tee -a "$LOG_FILE"

# Move weights
mv ./weights/train ./weights/discriminative_fr0.5_circular
echo "✓ Discriminative training (circular shift) complete" | tee -a "$LOG_FILE"

# Training with hard negative mining (NEW!)
echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "DISCRIMINATIVE TRAINING WITH HARD NEGATIVE MINING (FR=0.5)" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"

python train.py \
    --lapaface \
    --batchSize 8 \
    --cropSize 256 \
    --nEpochs 100 \
    --lr 1e-4 \
    --snapshots 10 \
    --D_weight 1.5 \
    --L1_weight 1.0 \
    --E_weight 50.0 \
    --P_weight 0.01 \
    --HVI_weight 1.0 \
    --use_face_loss \
    --FR_weight 0.5 \
    --FR_model_path ./weights/adaface/adaface_ir50_webface4m.ckpt \
    --contrastive_margin 0.4 \
    --contrastive_weight 1.0 \
    --triplet_margin 0.2 \
    --triplet_weight 0.5 \
    --face_temperature 0.07 \
    --use_hard_negatives \
    --hard_neg_memory_size 1000 \
    --hard_neg_topk 5 \
    --hard_neg_strategy mixed \
    2>&1 | tee -a "$LOG_FILE"

# Move weights
mv ./weights/train ./weights/discriminative_fr0.5_hardneg
echo "✓ Discriminative training (hard negatives) complete" | tee -a "$LOG_FILE"

# Training with identity-balanced sampling (NEW!)
echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "DISCRIMINATIVE TRAINING WITH IDENTITY-BALANCED SAMPLING (FR=0.5)" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"

python train.py \
    --lapaface \
    --batchSize 8 \
    --cropSize 256 \
    --nEpochs 100 \
    --lr 1e-4 \
    --snapshots 10 \
    --D_weight 1.5 \
    --L1_weight 1.0 \
    --E_weight 50.0 \
    --P_weight 0.01 \
    --HVI_weight 1.0 \
    --use_face_loss \
    --FR_weight 0.5 \
    --FR_model_path ./weights/adaface/adaface_ir50_webface4m.ckpt \
    --contrastive_margin 0.4 \
    --contrastive_weight 1.0 \
    --triplet_margin 0.2 \
    --triplet_weight 0.5 \
    --face_temperature 0.07 \
    --use_identity_balanced \
    --images_per_identity 2 \
    2>&1 | tee -a "$LOG_FILE"

# Move weights
mv ./weights/train ./weights/discriminative_fr0.5_identitybal
echo "✓ Discriminative training (identity-balanced) complete" | tee -a "$LOG_FILE"

# Training with BOTH hard negatives AND identity-balanced sampling (BEST!)
echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "DISCRIMINATIVE TRAINING WITH HARD NEGATIVES + IDENTITY-BALANCED (FR=0.5)" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"

python train.py \
    --lapaface \
    --batchSize 8 \
    --cropSize 256 \
    --nEpochs 100 \
    --lr 1e-4 \
    --snapshots 10 \
    --D_weight 1.5 \
    --L1_weight 1.0 \
    --E_weight 50.0 \
    --P_weight 0.01 \
    --HVI_weight 1.0 \
    --use_face_loss \
    --FR_weight 0.5 \
    --FR_model_path ./weights/adaface/adaface_ir50_webface4m.ckpt \
    --contrastive_margin 0.4 \
    --contrastive_weight 1.0 \
    --triplet_margin 0.2 \
    --triplet_weight 0.5 \
    --face_temperature 0.07 \
    --use_hard_negatives \
    --hard_neg_memory_size 1000 \
    --hard_neg_topk 5 \
    --hard_neg_strategy mixed \
    --use_identity_balanced \
    --images_per_identity 2 \
    2>&1 | tee -a "$LOG_FILE"

# Move weights
mv ./weights/train ./weights/discriminative_fr0.5_hardneg_identitybal
echo "✓ Discriminative training (hard negatives + identity-balanced) complete" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "ALL TRAINING COMPLETE" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Trained models:" | tee -a "$LOG_FILE"
echo "  1. baseline_d1.5" | tee -a "$LOG_FILE"
echo "  2. discriminative_fr0.5_circular" | tee -a "$LOG_FILE"
echo "  3. discriminative_fr0.5_hardneg" | tee -a "$LOG_FILE"
echo "  4. discriminative_fr0.5_identitybal" | tee -a "$LOG_FILE"
echo "  5. discriminative_fr0.5_hardneg_identitybal (BEST)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Training complete at $(date)" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
