#!/bin/bash
# Ablation Study: Test Different Face Recognition Loss Weights
# This will help determine the optimal FR loss weight for your thesis

echo "========================================================================"
echo "Ablation Study: Face Recognition Loss Weights"
echo "========================================================================"
echo ""
echo "This script will train models with different FR loss weights:"
echo "  - Baseline (no FR loss)"
echo "  - FR weight = 0.3"
echo "  - FR weight = 0.5"
echo "  - FR weight = 1.0"
echo ""
echo "Each training run takes ~X hours on a single GPU."
echo "Total estimated time: ~X hours"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Configuration
DATASET_DIR="./datasets/LFW_lowlight"
EPOCHS=100
BATCH_SIZE=8
CROP_SIZE=256

# Check if dataset exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset not found at $DATASET_DIR"
    echo "Please run: python prepare_lfw_dataset.py --download"
    exit 1
fi

# Function to train a single configuration
train_config() {
    NAME=$1
    USE_FR=$2
    FR_WEIGHT=$3
    WEIGHTS_DIR="./weights/ablation/$NAME"

    echo ""
    echo "========================================================================"
    echo "Training: $NAME"
    echo "========================================================================"

    mkdir -p $WEIGHTS_DIR

    python train.py \
        --lfw=True \
        --data_train_lfw=$DATASET_DIR/train \
        --data_val_lfw=$DATASET_DIR/val \
        --data_valgt_lfw=$DATASET_DIR/val/high \
        --batchSize=$BATCH_SIZE \
        --cropSize=$CROP_SIZE \
        --nEpochs=$EPOCHS \
        --lr=0.0001 \
        --use_face_loss=$USE_FR \
        --FR_weight=$FR_WEIGHT \
        --snapshots=10

    # Move weights to ablation directory
    mv ./weights/train/* $WEIGHTS_DIR/

    echo "✓ $NAME training complete. Weights saved to: $WEIGHTS_DIR"
}

# Run ablation experiments
echo ""
echo "Starting ablation study..."
echo ""

# 1. Baseline (no FR loss)
train_config "baseline" False 0.0

# 2. FR weight = 0.3
train_config "fr_weight_0.3" True 0.3

# 3. FR weight = 0.5 (recommended)
train_config "fr_weight_0.5" True 0.5

# 4. FR weight = 1.0
train_config "fr_weight_1.0" True 1.0

# Evaluation
echo ""
echo "========================================================================"
echo "Running Evaluations..."
echo "========================================================================"

for config in baseline fr_weight_0.3 fr_weight_0.5 fr_weight_1.0; do
    echo ""
    echo "Evaluating: $config"

    MODEL="./weights/ablation/$config/epoch_$EPOCHS.pth"

    if [ -f "$MODEL" ]; then
        python eval_face_verification.py \
            --model=$MODEL \
            --test_dir=$DATASET_DIR/test \
            --output_dir=./results/ablation/$config
    else
        echo "⚠ Model not found: $MODEL"
    fi
done

# Generate comparison table
echo ""
echo "========================================================================"
echo "Ablation Study Results"
echo "========================================================================"
echo ""

for config in baseline fr_weight_0.3 fr_weight_0.5 fr_weight_1.0; do
    RESULT_FILE="./results/ablation/$config/face_verification_results.txt"
    if [ -f "$RESULT_FILE" ]; then
        echo "[$config]"
        grep -A 2 "Face Similarity Metrics:" $RESULT_FILE | tail -2
        grep "Improvement:" $RESULT_FILE | head -1
        echo ""
    fi
done

echo ""
echo "========================================================================"
echo "Ablation study complete!"
echo "========================================================================"
echo ""
echo "Detailed results saved to: ./results/ablation/"
echo ""
echo "Use these results for your thesis to show:"
echo "1. Impact of different FR loss weights"
echo "2. Optimal weight selection"
echo "3. Trade-offs between image quality and face similarity"
echo ""
