#!/bin/bash
# Train Baseline Model (without Face Recognition Loss)
# This establishes your baseline for comparison

echo "========================================================================"
echo "Training Baseline Model (No Face Recognition Loss)"
echo "========================================================================"

python train.py \
    --lfw=True \
    --data_train_lfw=./datasets/LFW_lowlight/train \
    --data_val_lfw=./datasets/LFW_lowlight/val \
    --data_valgt_lfw=./datasets/LFW_lowlight/val/high \
    --batchSize=8 \
    --cropSize=256 \
    --nEpochs=100 \
    --lr=0.0001 \
    --L1_weight=1.0 \
    --D_weight=0.5 \
    --E_weight=50.0 \
    --P_weight=0.01 \
    --threads=8 \
    --snapshots=10

echo ""
echo "========================================================================"
echo "Baseline training complete!"
echo "Checkpoints saved to: ./weights/train/"
echo "========================================================================"
