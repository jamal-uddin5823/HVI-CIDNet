#!/bin/bash

# Comparison Study: Baseline vs Discriminative Multi-Level Face Loss
# This script trains 3 models for comparison on LFW_lowlight dataset
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
echo "Model Comparison Study"
echo "=========================================="
echo "Dataset: LFW_lowlight"
echo "Epochs: 50"
echo "Models to train: 3"
echo "CUDA_LAUNCH_BLOCKING: enabled"
echo "=========================================="

# Create directories for storing models
mkdir -p ./weights/baseline_d1.5_reference
mkdir -p ./weights/discriminative_fr0.3_d1.5
mkdir -p ./weights/discriminative_fr0.5_d1.5
mkdir -p ./logs/discriminative

# Model 1: Current best baseline (reference)
echo ""
echo "Training Model 1: Baseline (D=1.5, no face loss)"
echo "=========================================="
python train.py \
    --lfw \
    --data_train_lfw=./datasets/LFW_lowlight/train \
    --data_val_lfw=./datasets/LFW_lowlight/val \
    --pretrained_model=./weights/LOLv2_real/best_PSNR.pth \
    --D_weight=1.5 \
    --nEpochs=50 \
    --snapshots=5 > logs/discriminative/baseline_d1.5_reference_$(date +%Y%m%d_%H%M%S).log

# Move weights to baseline folder
echo "Saving baseline model weights..."
cp -r ./weights/train/* ./weights/baseline_d1.5_reference/
echo "Model 1 complete!"
echo "=========================================="

# Clean up for next model
rm -rf ./weights/train/*

# Model 2: Discriminative FR loss (FR=0.3)
echo ""
echo "Training Model 2: Discriminative FR Loss (FR=0.3, D=1.5)"
echo "=========================================="
python train.py \
    --lfw \
    --data_train_lfw=./datasets/LFW_lowlight/train \
    --data_val_lfw=./datasets/LFW_lowlight/val \
    --pretrained_model=./weights/LOLv2_real/best_PSNR.pth \
    --use_face_loss \
    --FR_weight=0.3 \
    --FR_model_path=./weights/adaface/adaface_ir50_webface4m.ckpt \
    --D_weight=1.5 \
    --nEpochs=50 \
    --contrastive_margin=0.4 \
    --contrastive_weight=1.0 \
    --triplet_margin=0.2 \
    --triplet_weight=0.5 \
    --snapshots=5 > logs/discriminative/discriminative_fr0.3_d1.5_$(date +%Y%m%d_%H%M%S).log

# Move weights to FR 0.3 folder
echo "Saving FR=0.3 model weights..."
cp -r ./weights/train/* ./weights/discriminative_fr0.3_d1.5/
echo "Model 2 complete!"
echo "=========================================="

# Clean up for next model
rm -rf ./weights/train/*

# Model 3: Discriminative FR loss (FR=0.5)
echo ""
echo "Training Model 3: Discriminative FR Loss (FR=0.5, D=1.5)"
echo "=========================================="
python train.py \
    --lfw \
    --data_train_lfw=./datasets/LFW_lowlight/train \
    --data_val_lfw=./datasets/LFW_lowlight/val \
    --pretrained_model=./weights/LOLv2_real/best_PSNR.pth \
    --use_face_loss \
    --FR_weight=0.5 \
    --FR_model_path=./weights/adaface/adaface_ir50_webface4m.ckpt \
    --D_weight=1.5 \
    --nEpochs=50 \
    --contrastive_margin=0.4 \
    --contrastive_weight=1.0 \
    --triplet_margin=0.2 \
    --triplet_weight=0.5 \
    --snapshots=5 > logs/discriminative/discriminative_fr0.5_d1.5_$(date +%Y%m%d_%H%M%S).log

# Move weights to FR 0.5 folder
echo "Saving FR=0.5 model weights..."
cp -r ./weights/train/* ./weights/discriminative_fr0.5_d1.5/
echo "Model 3 complete!"
echo "=========================================="

echo ""
echo "All models trained successfully!"
echo "Check the following directories for results:"
echo "  - ./weights/baseline_d1.5_reference"
echo "  - ./weights/discriminative_fr0.3_d1.5"
echo "  - ./weights/discriminative_fr0.5_d1.5"
echo "=========================================="
