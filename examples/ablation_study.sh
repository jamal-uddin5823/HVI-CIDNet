#!/bin/bash
# Ablation Study: Test Different Face Recognition Loss Weights
# This will help determine the optimal FR loss weight for your thesis

echo "========================================================================"
echo "Ablation Study: Face Recognition Loss Weights"
echo "========================================================================"

# Configuration
DATASET_DIR="./datasets/LFW_lowlight"
BATCH_SIZE=8
CROP_SIZE=256

# Check if dataset exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset not found at $DATASET_DIR"
    echo "Please run: python prepare_lfw_dataset.py --download"
    exit 1
fi

# Check for pretrained weights
CIDNET_WEIGHTS="./weights/LOLv2_real/best_PSNR.pth"
ADAFACE_WEIGHTS="./weights/adaface/adaface_ir50_webface4m.ckpt"

if [ -f "$CIDNET_WEIGHTS" ]; then
    echo "✓ Fine-tuning from pretrained CIDNet: $CIDNET_WEIGHTS"
    CIDNET_ARG="--pretrained_model=$CIDNET_WEIGHTS"
    LR="0.00001"
    EPOCHS=50
else
    echo "⚠ Training from scratch (no pretrained CIDNet)"
    CIDNET_ARG=""
    LR="0.0001"
    EPOCHS=100
fi

if [ -f "$ADAFACE_WEIGHTS" ]; then
    echo "✓ Using pretrained AdaFace: $ADAFACE_WEIGHTS"
    ADAFACE_ARG="--FR_model_path=$ADAFACE_WEIGHTS"
else
    echo "⚠ Using randomly initialized AdaFace (not recommended)"
    ADAFACE_ARG=""
fi

echo ""
echo "This script will train 4 models with different configurations:"
echo "  - Baseline (no FR loss)"
echo "  - FR weight = 0.3"
echo "  - FR weight = 0.5"
echo "  - FR weight = 1.0"
echo ""
echo "Training mode: $([ -f "$CIDNET_WEIGHTS" ] && echo "Fine-tuning ($EPOCHS epochs)" || echo "From scratch ($EPOCHS epochs)")"
echo "Each run trains for $EPOCHS epochs"
echo "Total runs: 4"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Function to train baseline (no FR loss)
train_baseline() {
    NAME=$1
    WEIGHTS_DIR="./weights/ablation/$NAME"

    echo ""
    echo "========================================================================"
    echo "Training: $NAME"
    echo "========================================================================"

    mkdir -p $WEIGHTS_DIR

    python train.py \
        --lfw \
        --data_train_lfw=$DATASET_DIR/train \
        --data_val_lfw=$DATASET_DIR/val \
        --data_valgt_lfw=$DATASET_DIR/val/high \
        $CIDNET_ARG \
        --batchSize=$BATCH_SIZE \
        --cropSize=$CROP_SIZE \
        --nEpochs=$EPOCHS \
        --lr=$LR \
        --snapshots=10

    # Move weights to ablation directory
    mv ./weights/train/* $WEIGHTS_DIR/

    echo "✓ $NAME training complete. Weights saved to: $WEIGHTS_DIR"
}

# Function to train with FR loss
train_with_fr() {
    NAME=$1
    FR_WEIGHT=$2
    WEIGHTS_DIR="./weights/ablation/$NAME"

    echo ""
    echo "========================================================================"
    echo "Training: $NAME (FR weight=$FR_WEIGHT)"
    echo "========================================================================"

    mkdir -p $WEIGHTS_DIR

    python train.py \
        --lfw \
        --data_train_lfw=$DATASET_DIR/train \
        --data_val_lfw=$DATASET_DIR/val \
        --data_valgt_lfw=$DATASET_DIR/val/high \
        $CIDNET_ARG \
        --batchSize=$BATCH_SIZE \
        --cropSize=$CROP_SIZE \
        --nEpochs=$EPOCHS \
        --lr=$LR \
        --use_face_loss \
        --FR_weight=$FR_WEIGHT \
        $ADAFACE_ARG \
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
train_baseline "baseline"

# 2. FR weight = 0.3
train_with_fr "fr_weight_0.3" 0.3

# 3. FR weight = 0.5 (recommended)
train_with_fr "fr_weight_0.5" 0.5

# 4. FR weight = 1.0
train_with_fr "fr_weight_1.0" 1.0

# Evaluation - Face Recognition Metrics (KEY for thesis!)
echo ""
echo "========================================================================"
echo "Running Face Verification Evaluations..."
echo "========================================================================"
echo "This will compute face similarity and verification accuracy improvements"
echo "which are the KEY METRICS for demonstrating your thesis contribution!"
echo ""

if [ ! -f "$ADAFACE_WEIGHTS" ]; then
    echo "⚠ WARNING: AdaFace weights not found at: $ADAFACE_WEIGHTS"
    echo "  Face verification evaluation requires AdaFace weights."
    echo "  Download from: https://github.com/mk-minchul/AdaFace/releases"
    echo ""
    read -p "Continue without face verification? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. Please download AdaFace weights first."
        exit 1
    fi
fi

for config in baseline fr_weight_0.3 fr_weight_0.5 fr_weight_1.0; do
    echo ""
    echo "Evaluating: $config"

    MODEL="./weights/ablation/$config/epoch_$EPOCHS.pth"

    if [ -f "$MODEL" ]; then
        if [ -f "$ADAFACE_WEIGHTS" ]; then
            python eval_face_verification.py \
                --model=$MODEL \
                --test_dir=$DATASET_DIR/test \
                --face_weights=$ADAFACE_WEIGHTS \
                --face_model=ir_50 \
                --output_dir=./results/ablation/$config
        else
            echo "⚠ Skipping face verification for $config (no AdaFace weights)"
        fi
    else
        echo "⚠ Model not found: $MODEL"
    fi
done

# Generate comparison table for thesis
echo ""
echo "========================================================================"
echo "Ablation Study Results - Face Recognition Metrics"
echo "========================================================================"
echo ""
echo "Configuration         | Face Similarity | Improvement | PSNR  | SSIM"
echo "---------------------+----------------+-------------+-------+------"

for config in baseline fr_weight_0.3 fr_weight_0.5 fr_weight_1.0; do
    RESULT_FILE="./results/ablation/$config/face_verification_results.txt"
    if [ -f "$RESULT_FILE" ]; then
        # Extract metrics (this is a simple parsing, you may need to adjust based on actual output format)
        FACE_SIM=$(grep "Enhanced avg similarity:" $RESULT_FILE | awk '{print $4}' || echo "N/A")
        IMPROVEMENT=$(grep "Similarity improvement:" $RESULT_FILE | awk '{print $3}' || echo "N/A")
        PSNR=$(grep "Average PSNR:" $RESULT_FILE | awk '{print $3}' || echo "N/A")
        SSIM=$(grep "Average SSIM:" $RESULT_FILE | awk '{print $3}' || echo "N/A")

        printf "%-20s | %-14s | %-11s | %-5s | %-5s\n" "$config" "$FACE_SIM" "$IMPROVEMENT" "$PSNR" "$SSIM"
    else
        printf "%-20s | Results not found\n" "$config"
    fi
done

echo ""
echo "========================================================================"
echo "Ablation study complete!"
echo "========================================================================"
echo ""
echo "Results saved to: ./results/ablation/"
echo ""
echo "For your thesis, use these results to demonstrate:"
echo "1. Face similarity improvement with FR loss (KEY CONTRIBUTION)"
echo "2. Optimal FR loss weight selection (e.g., 0.5)"
echo "3. Trade-offs between image quality (PSNR/SSIM) and face similarity"
echo "4. Comparison: Baseline vs. different FR loss weights"
echo ""
echo "Next steps:"
echo "1. Analyze detailed results in ./results/ablation/*/face_verification_results.txt"
echo "2. Create graphs showing FR loss weight vs. face similarity"
echo "3. Include visual comparisons in thesis (enhanced face images)"
echo ""
