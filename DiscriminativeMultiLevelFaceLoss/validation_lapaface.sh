#!/bin/bash

# Validation Script for Discriminative Multi-Level Face Loss on LaPa-Face
# This script trains the model on LaPa-Face dataset for 10 epochs
# to validate that the discriminative face loss is working correctly

echo "=========================================="
echo "Discriminative Face Loss Validation"
echo "=========================================="
echo "Dataset: LaPa-Face"
echo "Epochs: 10"
echo "Batch Size: 8"
echo "Face Loss Weight: 0.5"
echo "Contrastive Margin: 0.4, Weight: 1.0"
echo "Triplet Margin: 0.2, Weight: 0.5"
echo "=========================================="

# Create log directory if it doesn't exist
mkdir -p logs

# Train for 10 epochs on LaPa-Face to validate
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
    --nEpochs=10 \
    --batchSize=8 \
    --contrastive_margin=0.4 \
    --contrastive_weight=1.0 \
    --triplet_margin=0.2 \
    --triplet_weight=0.5 \
    --snapshots=5 > logs/validation_lapaface_discriminative_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=========================================="
echo "Validation complete!"
echo "Check ./results/lapaface/ for outputs"
echo "Check ./weights/train/ for checkpoints"
echo "=========================================="
