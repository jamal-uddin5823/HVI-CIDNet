#!/bin/bash

# Validation Script for Discriminative Multi-Level Face Loss + NEW Improvements
# This script trains the model on a small subset of LFW for 10 epochs
# to validate that the discriminative face loss is working correctly
# WITH the new hard negative mining and identity-balanced sampling

echo "=========================================="
echo "Discriminative Face Loss Validation (WITH NEW IMPROVEMENTS)"
echo "=========================================="
echo "Dataset: LFW_small"
echo "Epochs: 10"
echo "Batch Size: 8"
echo "Face Loss Weight: 0.5"
echo "Contrastive Margin: 0.4, Weight: 1.0"
echo "Triplet Margin: 0.2, Weight: 0.5"
echo "Hard Negative Mining: ENABLED"
echo "Identity-Balanced Sampling: ENABLED"
echo "=========================================="

# Train for 10 epochs on small subset to validate ALL new features
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
    --use_hard_negatives \
    --hard_neg_memory_size=100 \
    --hard_neg_topk=5 \
    --hard_neg_strategy=mixed \
    --use_identity_balanced \
    --images_per_identity=2 \
    --snapshots=5 > logs/validation_discriminative_with_improvements_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=========================================="
echo "Validation complete!"
echo "Check ./results/lfw/ for outputs"
echo "Check ./weights/train/ for checkpoints"
echo ""
echo "The log should show:"
echo "  - Hard Neg Memory: growing from 0 to ~100 identities"
echo "  - Reconstruction, Contrastive, Triplet loss components"
echo "  - Identity-balanced batches (4 identities x 2 images)"
echo "=========================================="
