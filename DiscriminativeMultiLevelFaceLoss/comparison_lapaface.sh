#!/bin/bash

# Comparison Study: Baseline vs Discriminative Multi-Level Face Loss on LaPa-Face
# This script trains 3 models for comparison on LaPa-Face dataset
# - Baseline (no face loss)
# - Discriminative FR loss with weight 0.3
# - Discriminative FR loss with weight 0.5

# Exit immediately on any error or when Ctrl+C is pressed
set -e
set -o pipefail

# Trap signals to ensure clean exit on interrupt
trap 'echo "Script interrupted! Exiting..."; exit 130' INT TERM

# Enable CUDA error debugging
export CUDA_LAUNCH_BLOCKING=1

echo "=========================================="
echo "Model Comparison Study (LaPa-Face)"
echo "=========================================="
echo "Dataset: LaPa-Face"
echo "Epochs: 50"
echo "Models to train: 3"
echo "CUDA_LAUNCH_BLOCKING: enabled"
echo "=========================================="

# Create directories for storing models
mkdir -p ./weights/lapaface_baseline_d1.5_reference
mkdir -p ./weights/lapaface_discriminative_fr0.3_d1.5
mkdir -p ./weights/lapaface_discriminative_fr0.5_d1.5
mkdir -p logs
mkdir -p logs/lapaface

# Model 1: Current best baseline (reference)
echo ""
echo "Training Model 1: Baseline (D=1.5, no face loss)"
echo "=========================================="
python train.py \
    --lapaface \
    --data_train_lapaface=./datasets/LaPa-Face/train \
    --data_val_lapaface=./datasets/LaPa-Face/test \
    --data_valgt_lapaface=./datasets/LaPa-Face/test/normal/ \
    --pretrained_model=./weights/LOLv2_real/best_PSNR.pth \
    --D_weight=1.5 \
    --nEpochs=50 \
    --batchSize=8 \
    --snapshots=5 > logs/lapaface/baseline_d1.5_$(date +%Y%m%d_%H%M%S).log

# Move weights to baseline folder
echo "Saving baseline model weights..."
cp -r ./weights/train/* ./weights/lapaface_baseline_d1.5_reference/
echo "Model 1 complete!"
echo "=========================================="

# Clean up for next model
rm -rf ./weights/train/*

# Model 2: Discriminative FR loss (FR=0.3)
echo ""
echo "Training Model 2: Discriminative FR Loss (FR=0.3, D=1.5)"
echo "=========================================="
python train.py \
    --lapaface \
    --data_train_lapaface=./datasets/LaPa-Face/train \
    --data_val_lapaface=./datasets/LaPa-Face/test \
    --data_valgt_lapaface=./datasets/LaPa-Face/test/normal/ \
    --pretrained_model=./weights/LOLv2_real/best_PSNR.pth \
    --use_face_loss \
    --FR_weight=0.3 \
    --FR_model_path=./weights/adaface/adaface_ir50_webface4m.ckpt \
    --D_weight=1.5 \
    --nEpochs=50 \
    --batchSize=8 \
    --contrastive_margin=0.4 \
    --contrastive_weight=1.0 \
    --triplet_margin=0.2 \
    --triplet_weight=0.5 \
    --snapshots=5 > logs/lapaface/discriminative_fr0.3_d1.5_$(date +%Y%m%d_%H%M%S).log

# Move weights to FR 0.3 folder
echo "Saving FR=0.3 model weights..."
cp -r ./weights/train/* ./weights/lapaface_discriminative_fr0.3_d1.5/
echo "Model 2 complete!"
echo "=========================================="

# Clean up for next model
rm -rf ./weights/train/*

# Model 3: Discriminative FR loss (FR=0.5)
echo ""
echo "Training Model 3: Discriminative FR Loss (FR=0.5, D=1.5)"
echo "=========================================="
python train.py \
    --lapaface \
    --data_train_lapaface=./datasets/LaPa-Face/train \
    --data_val_lapaface=./datasets/LaPa-Face/test \
    --data_valgt_lapaface=./datasets/LaPa-Face/test/normal/ \
    --pretrained_model=./weights/LOLv2_real/best_PSNR.pth \
    --use_face_loss \
    --FR_weight=0.5 \
    --FR_model_path=./weights/adaface/adaface_ir50_webface4m.ckpt \
    --D_weight=1.5 \
    --nEpochs=50 \
    --batchSize=8 \
    --contrastive_margin=0.4 \
    --contrastive_weight=1.0 \
    --triplet_margin=0.2 \
    --triplet_weight=0.5 \
    --snapshots=5 > logs/lapaface/discriminative_fr0.5_d1.5_$(date +%Y%m%d_%H%M%S).log

# Move weights to FR 0.5 folder
echo "Saving FR=0.5 model weights..."
cp -r ./weights/train/* ./weights/lapaface_discriminative_fr0.5_d1.5/
echo "Model 3 complete!"
echo "=========================================="

echo ""
echo "All models trained successfully!"
echo "Check the following directories for results:"
echo "  - ./weights/lapaface_baseline_d1.5_reference"
echo "  - ./weights/lapaface_discriminative_fr0.3_d1.5"
echo "  - ./weights/lapaface_discriminative_fr0.5_d1.5"
echo "=========================================="
