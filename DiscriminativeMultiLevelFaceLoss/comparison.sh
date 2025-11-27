#!/bin/bash

# Comparison Study: Baseline vs Discriminative Multi-Level Face Loss + NEW Improvements
# This script trains 6 models for comparison on LFW_lowlight dataset:
# - Baseline (no face loss)
# - Discriminative FR loss with weight 0.3
# - Discriminative FR loss with weight 0.5
# - Discriminative FR=0.5 + Hard Negative Mining (NEW)
# - Discriminative FR=0.5 + Identity-Balanced Sampling (NEW)
# - Discriminative FR=0.5 + Hard Negatives + Identity-Balanced (BEST) (NEW)

# Exit immediately on any error or when Ctrl+C is pressed
set -e
set -o pipefail

# Trap signals to ensure clean exit on interrupt
trap 'echo "Script interrupted! Exiting..."; exit 130' INT TERM

# Enable CUDA error debugging
export CUDA_LAUNCH_BLOCKING=1

echo "=========================================="
echo "Model Comparison Study (WITH NEW IMPROVEMENTS)"
echo "=========================================="
echo "Dataset: LFW_lowlight"
echo "Epochs: 50"
echo "Models to train: 6"
echo "CUDA_LAUNCH_BLOCKING: enabled"
echo "=========================================="

# Create directories for storing models
mkdir -p ./weights/baseline_d1.5_reference
mkdir -p ./weights/discriminative_fr0.3_d1.5
mkdir -p ./weights/discriminative_fr0.5_d1.5
mkdir -p ./weights/discriminative_fr0.5_hardneg
mkdir -p ./weights/discriminative_fr0.5_identitybal
mkdir -p ./weights/discriminative_fr0.5_hardneg_identitybal
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

# Model 2: Discriminative FR loss (FR=0.3, circular shift)
echo ""
echo "Training Model 2: Discriminative FR Loss (FR=0.3, D=1.5, circular shift)"
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

# Model 3: Discriminative FR loss (FR=0.5, circular shift)
echo ""
echo "Training Model 3: Discriminative FR Loss (FR=0.5, D=1.5, circular shift)"
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

# Clean up for next model
rm -rf ./weights/train/*

# Model 4: Discriminative FR=0.5 + Hard Negative Mining (NEW)
echo ""
echo "Training Model 4: Discriminative FR=0.5 + Hard Negative Mining (NEW)"
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
    --use_hard_negatives \
    --hard_neg_memory_size=1000 \
    --hard_neg_topk=5 \
    --hard_neg_strategy=mixed \
    --snapshots=5 > logs/discriminative/discriminative_fr0.5_hardneg_$(date +%Y%m%d_%H%M%S).log

# Move weights
echo "Saving FR=0.5 + Hard Negatives model weights..."
cp -r ./weights/train/* ./weights/discriminative_fr0.5_hardneg/
echo "Model 4 complete!"
echo "=========================================="

# Clean up for next model
rm -rf ./weights/train/*

# Model 5: Discriminative FR=0.5 + Identity-Balanced Sampling (NEW)
echo ""
echo "Training Model 5: Discriminative FR=0.5 + Identity-Balanced Sampling (NEW)"
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
    --use_identity_balanced \
    --images_per_identity=2 \
    --snapshots=5 > logs/discriminative/discriminative_fr0.5_identitybal_$(date +%Y%m%d_%H%M%S).log

# Move weights
echo "Saving FR=0.5 + Identity-Balanced model weights..."
cp -r ./weights/train/* ./weights/discriminative_fr0.5_identitybal/
echo "Model 5 complete!"
echo "=========================================="

# Clean up for next model
rm -rf ./weights/train/*

# Model 6: Discriminative FR=0.5 + Hard Negatives + Identity-Balanced (BEST) (NEW)
echo ""
echo "Training Model 6: Discriminative FR=0.5 + Hard Negatives + Identity-Balanced (BEST)"
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
    --use_hard_negatives \
    --hard_neg_memory_size=1000 \
    --hard_neg_topk=5 \
    --hard_neg_strategy=mixed \
    --use_identity_balanced \
    --images_per_identity=2 \
    --snapshots=5 > logs/discriminative/discriminative_fr0.5_hardneg_identitybal_$(date +%Y%m%d_%H%M%S).log

# Move weights
echo "Saving FR=0.5 + Hard Negatives + Identity-Balanced model weights..."
cp -r ./weights/train/* ./weights/discriminative_fr0.5_hardneg_identitybal/
echo "Model 6 complete!"
echo "=========================================="

echo ""
echo "All 6 models trained successfully!"
echo "Check the following directories for results:"
echo "  1. ./weights/baseline_d1.5_reference"
echo "  2. ./weights/discriminative_fr0.3_d1.5"
echo "  3. ./weights/discriminative_fr0.5_d1.5"
echo "  4. ./weights/discriminative_fr0.5_hardneg (NEW)"
echo "  5. ./weights/discriminative_fr0.5_identitybal (NEW)"
echo "  6. ./weights/discriminative_fr0.5_hardneg_identitybal (BEST) (NEW)"
echo "=========================================="
