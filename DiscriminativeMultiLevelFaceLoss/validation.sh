#!/bin/bash

# Validation Script for Discriminative Multi-Level Face Loss
# This script trains the model on a small subset of LFW for 10 epochs
# to validate that the discriminative face loss is working correctly

echo "=========================================="
echo "Discriminative Face Loss Validation"
echo "=========================================="
echo "Dataset: LFW_small"
echo "Epochs: 10"
echo "Batch Size: 8"
echo "Face Loss Weight: 0.5"
echo "Contrastive Margin: 0.4, Weight: 1.0"
echo "Triplet Margin: 0.2, Weight: 0.5"
echo "=========================================="

# Train for 10 epochs on small subset to validate
python train.py \
    --lfw \
    --data_train_lfw=./datasets/LFW_small/train \
    --data_val_lfw=./datasets/LFW_small/val \
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
    --snapshots=5 > logs/validation_discriminative_face_loss_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=========================================="
echo "Validation complete!"
echo "Check ./results/lfw/ for outputs"
echo "Check ./weights/train/ for checkpoints"
echo "=========================================="
