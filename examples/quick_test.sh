#!/bin/bash
# Quick Test Script for Rapid Iteration
# Uses small dataset and few epochs to verify everything works

echo "========================================================================"
echo "Quick Test: Face Recognition Loss Implementation"
echo "========================================================================"

# Step 1: Create small test dataset if needed
if [ ! -d "./datasets/LFW_small" ]; then
    echo "[1/3] Creating small test dataset (1000 images)..."
    python prepare_lfw_dataset.py \
        --max_images=1000 \
        --output_dir=./datasets/LFW_small
else
    echo "[1/3] Using existing small dataset: ./datasets/LFW_small"
fi

# Step 2: Quick training test (10 epochs)
echo ""
echo "[2/3] Running quick training test (10 epochs)..."
echo "This will verify that the face recognition loss is working correctly."
echo ""

python train.py \
    --lfw=True \
    --data_train_lfw=./datasets/LFW_small/train \
    --data_val_lfw=./datasets/LFW_small/val \
    --data_valgt_lfw=./datasets/LFW_small/val/high \
    --batchSize=4 \
    --cropSize=128 \
    --nEpochs=10 \
    --lr=0.0001 \
    --use_face_loss=True \
    --FR_weight=0.5 \
    --snapshots=5 \
    --threads=4

# Step 3: Quick evaluation
echo ""
echo "[3/3] Running quick evaluation (100 pairs)..."
echo ""

LATEST_MODEL=$(ls -t ./weights/train/*.pth | head -1)

if [ -f "$LATEST_MODEL" ]; then
    python eval_face_verification.py \
        --model=$LATEST_MODEL \
        --test_dir=./datasets/LFW_small/test \
        --max_pairs=100
else
    echo "⚠ No model found in ./weights/train/"
fi

echo ""
echo "========================================================================"
echo "Quick test complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "1. Review results in: ./results/"
echo "2. If everything looks good, run full training with:"
echo "   bash examples/train_with_face_loss.sh"
echo ""
