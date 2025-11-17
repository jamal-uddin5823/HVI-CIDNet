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

# Check for pretrained weights
CIDNET_WEIGHTS="./weights/LOLv2_real/best_PSNR.pth"
ADAFACE_WEIGHTS="./pretrained/adaface/adaface_ir50_webface4m.ckpt"

if [ -f "$CIDNET_WEIGHTS" ]; then
    echo "✓ Using pretrained CIDNet: $CIDNET_WEIGHTS"
    CIDNET_ARG="--pretrained_model=$CIDNET_WEIGHTS"
else
    echo "⚠ Training from scratch (no pretrained CIDNet)"
    CIDNET_ARG=""
fi

if [ -f "$ADAFACE_WEIGHTS" ]; then
    echo "✓ Using pretrained AdaFace: $ADAFACE_WEIGHTS"
    ADAFACE_ARG="--FR_model_path=$ADAFACE_WEIGHTS"
else
    echo "⚠ Using randomly initialized AdaFace (not recommended)"
    ADAFACE_ARG=""
fi

# Step 2: Quick training test (10 epochs)
echo ""
echo "[2/3] Running quick training test (10 epochs)..."
echo "This will verify that the face recognition loss is working correctly."
echo ""

python train.py \
    --lfw \
    --data_train_lfw=./datasets/LFW_small/train \
    --data_val_lfw=./datasets/LFW_small/val \
    --data_valgt_lfw=./datasets/LFW_small/val/high \
    $CIDNET_ARG \
    --batchSize=4 \
    --cropSize=256 \
    --nEpochs=10 \
    --lr=0.00001 \
    --use_face_loss \
    --FR_weight=0.5 \
    --snapshots=5 \
    --threads=4 \
    $ADAFACE_ARG

# Step 3: Quick evaluation (Face Recognition Metrics!)
echo ""
echo "[3/3] Running face verification evaluation (100 pairs)..."
echo "This is the KEY METRIC for your thesis!"
echo ""

LATEST_MODEL=$(ls -t ./weights/train/*.pth | head -1)

if [ -f "$LATEST_MODEL" ]; then
    if [ -f "$ADAFACE_WEIGHTS" ]; then
        python eval_face_verification.py \
            --model=$LATEST_MODEL \
            --test_dir=./datasets/LFW_small/test \
            --face_weights=$ADAFACE_WEIGHTS \
            --face_model=ir_50 \
            --max_pairs=100 \
            --output_dir=./results/quick_test_face_verification
    else
        echo "⚠ AdaFace weights not found. Running without face verification metrics."
        echo "  Download from: https://github.com/mk-minchul/AdaFace/releases"
    fi
else
    echo "⚠ No model found in ./weights/train/"
fi

echo ""
echo "========================================================================"
echo "Quick test complete!"
echo "========================================================================"
echo ""
echo "Results:"
echo "  - Enhanced images: ./results/lfw/"
echo "  - Face verification metrics: ./results/quick_test_face_verification/"
echo ""
echo "Next steps:"
echo "1. Review face similarity improvements in: ./results/quick_test_face_verification/"
echo "2. If everything looks good, run full training with:"
echo "   bash examples/train_with_face_loss.sh"
echo ""
