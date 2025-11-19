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
# read -p "Continue? (y/n) " -n 1 -r
# echo
# if [[ ! $REPLY =~ ^[Yy]$ ]]; then
#     echo "Aborted."
#     exit 1
# fi

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
        --snapshots=10 > logs/baseline_$NAME_$(date +"%Y%m%d_%H%M%S").log 2>&1

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
        --snapshots=10 > logs/fr_$NAME_$(date +"%Y%m%d_%H%M%S").log 2>&1

    # Move weights to ablation directory
    mv ./weights/train/* $WEIGHTS_DIR/

    echo "✓ $NAME training complete. Weights saved to: $WEIGHTS_DIR"
}

# Run ablation experiments
echo ""
echo "Starting ablation study..."
echo ""

# 1. Baseline (no FR loss)
# train_baseline "baseline"

# 2. FR weight = 0.3
# train_with_fr "fr_weight_0.3" 0.3

# 3. FR weight = 0.5 (recommended)
# train_with_fr "fr_weight_0.5" 0.5

# 4. FR weight = 1.0
# train_with_fr "fr_weight_1.0" 1.0

# Generate pairs for evaluation
echo ""
echo "========================================================================"
echo "Generating Pairs for Face Verification Evaluation"
echo "========================================================================"
echo "Creating genuine and impostor pairs following LFW protocol..."
echo ""

PAIRS_FILE="./pairs_lfw.txt"

if [ ! -f "$PAIRS_FILE" ]; then
    python generate_lfw_pairs.py \
        --test_dir=$DATASET_DIR/test \
        --num_pairs=1000 \
        --output=$PAIRS_FILE

    if [ $? -ne 0 ]; then
        echo "⚠ Failed to generate pairs file"
        echo "  Will proceed without pairs-based evaluation"
        PAIRS_FILE=""
    else
        echo "✓ Pairs file generated: $PAIRS_FILE"
    fi
else
    echo "✓ Using existing pairs file: $PAIRS_FILE"
fi

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
    exit 1
    # read -p "Continue without face verification? (y/n) " -n 1 -r
    # echo
    # if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    #     echo "Aborted. Please download AdaFace weights first."
    #     exit 1
    # fi
fi

for config in baseline fr_weight_0.3 fr_weight_0.5 fr_weight_1.0; do
    echo ""
    echo "========================================================================"
    echo "Evaluating: $config"
    echo "========================================================================"

    MODEL="./weights/ablation/$config/epoch_$EPOCHS.pth"
    echo "  Looking for model: $MODEL"

    if [ -f "$MODEL" ]; then
        echo "  ✓ Model found"
        if [ -f "$ADAFACE_WEIGHTS" ]; then
            echo "  ✓ AdaFace weights found"

            # Create output directory
            mkdir -p ./results/ablation/$config

            # Use pairs-based evaluation if pairs file exists
            if [ -n "$PAIRS_FILE" ] && [ -f "$PAIRS_FILE" ]; then
                echo "  ✓ Pairs file found ($PAIRS_FILE)"
                echo "  → Running pairs-based face verification evaluation..."
                echo ""

                python eval_face_verification.py \
                    --model=$MODEL \
                    --test_dir=$DATASET_DIR/test \
                    --pairs_file=$PAIRS_FILE \
                    --face_weights=$ADAFACE_WEIGHTS \
                    --face_model=ir_50 \
                    --output_dir=./results/ablation/$config

                if [ $? -eq 0 ]; then
                    echo ""
                    echo "  ✓ Evaluation completed successfully"
                    echo "  Results saved to: ./results/ablation/$config"
                else
                    echo ""
                    echo "  ✗ Evaluation failed with exit code $?"
                    echo "  Check error messages above"
                fi
            else
                echo "  ⚠ Pairs file not found, using legacy evaluation mode"
                echo "  → Running legacy face verification evaluation..."
                echo "    (Note: This only computes genuine pair metrics, not EER/TAR)"
                echo ""

                python eval_face_verification.py \
                    --model=$MODEL \
                    --test_dir=$DATASET_DIR/test \
                    --face_weights=$ADAFACE_WEIGHTS \
                    --face_model=ir_50 \
                    --output_dir=./results/ablation/$config

                if [ $? -eq 0 ]; then
                    echo ""
                    echo "  ✓ Evaluation completed"
                else
                    echo ""
                    echo "  ✗ Evaluation failed with exit code $?"
                fi
            fi
        else
            echo "  ✗ AdaFace weights not found: $ADAFACE_WEIGHTS"
            echo "  Skipping face verification evaluation"
        fi
    else
        echo "  ✗ Model not found - did training complete successfully?"
        echo "     Expected location: $MODEL"
    fi
done

# Generate comparison table for thesis
echo ""
echo "========================================================================"
echo "Ablation Study Results - Face Recognition Metrics"
echo "========================================================================"
echo ""

# Check if ANY results exist
RESULTS_FOUND=0
for config in baseline fr_weight_0.3 fr_weight_0.5 fr_weight_1.0; do
    RESULT_FILE="./results/ablation/$config/face_verification_results.txt"
    if [ -f "$RESULT_FILE" ]; then
        RESULTS_FOUND=1
        break
    fi
done

if [ $RESULTS_FOUND -eq 0 ]; then
    echo "⚠ WARNING: No evaluation results found!"
    echo ""
    echo "Possible causes:"
    echo "1. Training did not complete successfully (models not saved)"
    echo "2. Evaluation failed (check error messages above)"
    echo "3. Test dataset not available at: $DATASET_DIR/test"
    echo "4. AdaFace weights not found at: $ADAFACE_WEIGHTS"
    echo ""
    echo "Please check:"
    echo "  - Model weights should be at: ./weights/ablation/<config>/epoch_$EPOCHS.pth"
    echo "  - Test dataset should be at: $DATASET_DIR/test/low and $DATASET_DIR/test/high"
    echo "  - AdaFace weights at: $ADAFACE_WEIGHTS"
    echo ""
    exit 1
fi

# Check if using pairs-based evaluation (has EER metrics)
if [ -n "$PAIRS_FILE" ] && [ -f "$PAIRS_FILE" ]; then
    echo "Pairs-Based Verification Results:"
    echo ""
    echo "Configuration         | Genuine Sim | EER (%) | TAR@1% (%) | PSNR  | SSIM"
    echo "---------------------+-------------+---------+------------+-------+------"

    for config in baseline fr_weight_0.3 fr_weight_0.5 fr_weight_1.0; do
        RESULT_FILE="./results/ablation/$config/face_verification_results.txt"
        if [ -f "$RESULT_FILE" ]; then
            # Extract metrics from pairs-based evaluation
            GENUINE_SIM=$(grep "Enhanced avg similarity:" $RESULT_FILE | head -1 | awk '{print $4}' || echo "N/A")
            EER=$(grep "Enhanced:" $RESULT_FILE | grep "EER" | awk '{print $2}' | tr -d '%' || echo "N/A")
            TAR=$(grep "Enhanced:" $RESULT_FILE | grep "TAR @ FAR=1%" -A 1 | tail -1 | awk '{print $2}' | tr -d '%' || echo "N/A")
            PSNR=$(grep "Average PSNR:" $RESULT_FILE | awk '{print $3}' || echo "N/A")
            SSIM=$(grep "Average SSIM:" $RESULT_FILE | awk '{print $3}' || echo "N/A")

            printf "%-20s | %-11s | %-7s | %-10s | %-5s | %-5s\n" "$config" "$GENUINE_SIM" "$EER" "$TAR" "$PSNR" "$SSIM"
        else
            printf "%-20s | Results not found\n" "$config"
        fi
    done
else
    echo "Legacy Evaluation Results:"
    echo ""
    echo "Configuration         | Face Similarity | Improvement | PSNR  | SSIM"
    echo "---------------------+----------------+-------------+-------+------"

    for config in baseline fr_weight_0.3 fr_weight_0.5 fr_weight_1.0; do
        RESULT_FILE="./results/ablation/$config/face_verification_results.txt"
        if [ -f "$RESULT_FILE" ]; then
            # Extract metrics from legacy evaluation
            FACE_SIM=$(grep "Enhanced.*GT:" $RESULT_FILE | awk '{print $3}' || echo "N/A")
            IMPROVEMENT=$(grep "Improvement:" $RESULT_FILE | head -1 | awk '{print $2}' || echo "N/A")
            PSNR=$(grep "PSNR:" $RESULT_FILE | awk '{print $2}' | head -1 || echo "N/A")
            SSIM=$(grep "SSIM:" $RESULT_FILE | awk '{print $2}' | head -1 || echo "N/A")

            printf "%-20s | %-14s | %-11s | %-5s | %-5s\n" "$config" "$FACE_SIM" "$IMPROVEMENT" "$PSNR" "$SSIM"
        else
            printf "%-20s | Results not found\n" "$config"
        fi
    done
fi

echo ""
echo "========================================================================"
echo "Ablation study complete!"
echo "========================================================================"
echo ""
echo "Results saved to: ./results/ablation/"
echo ""

if [ -n "$PAIRS_FILE" ] && [ -f "$PAIRS_FILE" ]; then
    echo "For your thesis, use these results to demonstrate:"
    echo "1. Face verification accuracy improvement (EER reduction) - KEY CONTRIBUTION"
    echo "2. Genuine pair similarity improvement with FR loss"
    echo "3. Impostor pair discrimination (should remain low)"
    echo "4. TAR@FAR metrics showing real-world verification performance"
    echo "5. Optimal FR loss weight selection (compare EER across weights)"
    echo "6. Trade-offs between image quality (PSNR/SSIM) and verification accuracy"
    echo ""
    echo "Key Metrics Explanation:"
    echo "  • Genuine Similarity: Higher is better (same person recognition)"
    echo "  • EER (Equal Error Rate): Lower is better (overall accuracy)"
    echo "  • TAR@FAR=1%: Higher is better (true accept rate at 1% false accepts)"
    echo "  • PSNR/SSIM: Standard image quality metrics"
else
    echo "For your thesis, use these results to demonstrate:"
    echo "1. Face similarity improvement with FR loss (KEY CONTRIBUTION)"
    echo "2. Optimal FR loss weight selection (e.g., 0.5)"
    echo "3. Trade-offs between image quality (PSNR/SSIM) and face similarity"
    echo "4. Comparison: Baseline vs. different FR loss weights"
    echo ""
    echo "⚠ Note: For proper verification metrics (EER, TAR@FAR), ensure pairs.txt exists"
fi

echo ""
echo "Next steps:"
echo "1. Analyze detailed results in ./results/ablation/*/face_verification_results.txt"
echo "2. Use generated figures in ./results/ablation/figures/ for thesis"
echo "3. Include visual comparisons in thesis (enhanced face images)"
echo "4. Report verification improvements in results section"
echo ""

# Generate visualizations
echo ""
echo "========================================================================"
echo "Generating Visualizations for Thesis"
echo "========================================================================"
echo ""

if command -v python &> /dev/null; then
    echo "Creating publication-quality figures from ablation study results..."
    echo ""

    python plot_ablation_results.py \
        --results_dir=./results/ablation \
        --output_dir=./results/ablation/figures > ./results/ablation/figures/plot_generation_$(date +"%Y%m%d_%H%M%S").txt 2>&1

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Visualizations generated successfully!"
        echo ""
        echo "Figures saved to: ./results/ablation/figures/"
        echo ""
        echo "Generated plots:"
        echo "  1. fr_weight_vs_verification_metrics.png - Key verification metrics vs FR weight"
        echo "  2. fr_weight_vs_image_quality.png       - PSNR/SSIM vs FR weight"
        echo "  3. comparison_bars.png                  - Bar chart comparison of all configs"
        echo "  4. quality_vs_verification_tradeoff.png - Trade-off analysis"
        echo "  5. summary_figure.png                   - Comprehensive 6-panel summary"
        echo ""
        echo "These figures are publication-ready for your thesis!"
    else
        echo ""
        echo "⚠ Failed to generate visualizations"
        echo "  You can manually run: python plot_ablation_results.py --results_dir=./results/ablation"
    fi
else
    echo "⚠ Python not found in PATH. Skipping visualization generation."
    echo "  Install Python and run: python plot_ablation_results.py --results_dir=./results/ablation"
fi

echo ""
