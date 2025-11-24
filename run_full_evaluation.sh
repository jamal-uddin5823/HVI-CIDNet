#!/bin/bash
# Comprehensive Face Verification Evaluation
# Runs proper pairs-based evaluation on ALL models with full 2000 pairs

set -e  # Exit on error

echo "========================================================================"
echo "COMPREHENSIVE FACE VERIFICATION EVALUATION"
echo "========================================================================"
echo ""
echo "This will evaluate all 4 trained models on the full 2000-pair dataset"
echo "using proper verification protocol (genuine + impostor pairs)"
echo ""

# Configuration
DATASET_DIR="./datasets/LFW_lowlight"
TEST_DIR="$DATASET_DIR/test"
PAIRS_FILE="./pairs.txt"
ADAFACE_WEIGHTS="./weights/adaface/adaface_ir50_webface4m.ckpt"
RESULTS_BASE="./results/full_evaluation"
EPOCH=50

# D_weight variations to evaluate
D_WEIGHTS=("0.5" "1" "1.5")

# Verify prerequisites
echo "Checking prerequisites..."
echo ""

if [ ! -d "$TEST_DIR" ]; then
    echo "✗ Error: Test directory not found: $TEST_DIR"
    exit 1
fi
echo "✓ Test directory found: $TEST_DIR"

if [ ! -f "$PAIRS_FILE" ]; then
    echo "✗ Error: Pairs file not found: $PAIRS_FILE"
    echo "  Generate with: python generate_lfw_pairs.py --test_dir=$TEST_DIR --num_pairs=1000"
    exit 1
fi

# Count actual pairs in file
NUM_PAIRS=$(grep -v "^#" "$PAIRS_FILE" | grep -v "^$" | wc -l)
echo "✓ Pairs file found: $PAIRS_FILE ($NUM_PAIRS pairs)"

if [ ! -f "$ADAFACE_WEIGHTS" ]; then
    echo "✗ Error: AdaFace weights not found: $ADAFACE_WEIGHTS"
    echo "  Download from: https://github.com/mk-minchul/AdaFace/releases"
    exit 1
fi
echo "✓ AdaFace weights found: $ADAFACE_WEIGHTS"

# Models to evaluate
MODELS=(
    "baseline"
    "fr_weight_0.3"
    "fr_weight_0.5"
    "fr_weight_1.0"
)

# Check all models exist
echo ""
echo "Checking model checkpoints..."
MISSING_MODELS=0
FOUND_MODELS=()

for config in "${MODELS[@]}"; do
    for d_weight in "${D_WEIGHTS[@]}"; do
        MODEL_PATH="./weights/ablation/$config/d_${d_weight}/epoch_$EPOCH.pth"
        if [ ! -f "$MODEL_PATH" ]; then
            echo "⚠ Missing: $MODEL_PATH"
            MISSING_MODELS=$((MISSING_MODELS + 1))
        else
            echo "✓ Found: $MODEL_PATH"
            FOUND_MODELS+=("$config/d_$d_weight")
        fi
    done
done

if [ ${#FOUND_MODELS[@]} -eq 0 ]; then
    echo ""
    echo "✗ Error: No model checkpoints found"
    echo "  Train models first or adjust EPOCH variable"
    exit 1
fi

echo ""
echo "========================================================================"
echo "All prerequisites met! Starting evaluation..."
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Model configs: ${MODELS[@]}"
echo "  D weights: ${D_WEIGHTS[@]}"
echo "  Found models: ${#FOUND_MODELS[@]}"
echo "  Epoch: $EPOCH"
echo "  Pairs: $NUM_PAIRS"
echo "  Output: $RESULTS_BASE"
echo ""

# Create results directory
mkdir -p "$RESULTS_BASE"

# Evaluate each model with each d_weight
for config in "${MODELS[@]}"; do
    for d_weight in "${D_WEIGHTS[@]}"; do
        MODEL_PATH="./weights/ablation/$config/d_${d_weight}/epoch_$EPOCH.pth"
        
        # Skip if model doesn't exist
        if [ ! -f "$MODEL_PATH" ]; then
            echo ""
            echo "⚠ Skipping: $config (d=$d_weight) - model not found"
            continue
        fi
        
        echo ""
        echo "========================================================================"
        echo "Evaluating: $config (D_weight=$d_weight)"
        echo "========================================================================"
        echo ""

        OUTPUT_DIR="$RESULTS_BASE/${config}_d${d_weight}"

        mkdir -p "$OUTPUT_DIR"

        echo "Model: $MODEL_PATH"
        echo "Output: $OUTPUT_DIR"
        echo ""
        echo "Running evaluation (this may take 10-30 minutes)..."
        echo ""

        # Run evaluation with full pairs
        python eval_face_verification.py \
            --model="$MODEL_PATH" \
            --test_dir="$TEST_DIR" \
            --pairs_file="$PAIRS_FILE" \
            --face_weights="$ADAFACE_WEIGHTS" \
            --face_model=ir_50 \
            --output_dir="$OUTPUT_DIR" \
            --device=cuda 2>&1 | tee "$OUTPUT_DIR/evaluation.log"

        EXIT_CODE=${PIPESTATUS[0]}

        if [ $EXIT_CODE -eq 0 ]; then
            echo ""
            echo "✓ Evaluation completed successfully"
            echo ""

            # Display key results
            if [ -f "$OUTPUT_DIR/face_verification_results.txt" ]; then
                echo "Key Results:"
                echo "------------"
                grep -E "Genuine pair|Impostor pair|Equal Error Rate|TAR @ FAR=1%" "$OUTPUT_DIR/face_verification_results.txt" | head -20
            fi
        else
            echo ""
            echo "✗ Evaluation failed with exit code: $EXIT_CODE"
            echo "  Check log: $OUTPUT_DIR/evaluation.log"
            echo ""
        fi

        echo ""
        echo "------------------------------------------------------------------------"
    done
done

echo ""
echo "========================================================================"
echo "GENERATING COMPARISON TABLE"
echo "========================================================================"
echo ""

# Create comparison table
COMPARISON_FILE="$RESULTS_BASE/comparison_table.txt"

cat > "$COMPARISON_FILE" << 'EOF'
================================================================================
FACE VERIFICATION EVALUATION - ABLATION STUDY RESULTS
================================================================================

Comparing 4 configurations on LFW low-light face verification task
Dataset: LFW synthetic low-light images
Pairs: Genuine (same person) + Impostor (different people)
Evaluation Protocol: Standard face verification with EER, TAR@FAR metrics

================================================================================
EOF

echo "" >> "$COMPARISON_FILE"
echo "RESULTS SUMMARY" >> "$COMPARISON_FILE"
echo "===============" >> "$COMPARISON_FILE"
echo "" >> "$COMPARISON_FILE"

# Table header
printf "%-25s | %-12s | %-12s | %-10s | %-12s | %-12s | %-8s | %-8s\n" \
    "Configuration" "Genuine Sim" "Impostor Sim" "EER (%)" "TAR@0.1% (%)" "TAR@1% (%)" "PSNR" "SSIM" >> "$COMPARISON_FILE"
echo "--------------------------+--------------+--------------+------------+--------------+--------------+----------+----------" >> "$COMPARISON_FILE"

# Extract and display results for each model and d_weight
for config in "${MODELS[@]}"; do
    for d_weight in "${D_WEIGHTS[@]}"; do
        RESULT_FILE="$RESULTS_BASE/${config}_d${d_weight}/face_verification_results.txt"
        
        if [ ! -f "$RESULT_FILE" ]; then
            continue
        fi
        
        CONFIG_LABEL="${config}_d${d_weight}"

        # Extract metrics (Enhanced values only)
        GENUINE_SIM=$(grep "Enhanced avg similarity:" "$RESULT_FILE" | head -1 | awk '{print $4}' | tr -d '\n')
        IMPOSTOR_SIM=$(grep "Enhanced avg similarity:" "$RESULT_FILE" | tail -1 | awk '{print $4}' | tr -d '\n')

        # Extract EER
        EER=$(grep "Enhanced:" "$RESULT_FILE" | grep -A 1 "Equal Error Rate" | tail -1 | awk '{print $2}' | tr -d '%' | tr -d '\n')

        # Extract TAR values
        TAR_001=$(grep "Enhanced:" "$RESULT_FILE" | grep -A 1 "TAR @ FAR=0.1%" | tail -1 | awk '{print $2}' | tr -d '%' | tr -d '\n')
        TAR_1=$(grep "Enhanced:" "$RESULT_FILE" | grep -A 1 "TAR @ FAR=1%" | tail -1 | awk '{print $2}' | tr -d '%' | tr -d '\n')

        # Extract image quality
        PSNR=$(grep "Average PSNR:" "$RESULT_FILE" | awk '{print $3}' | tr -d '\n')
        SSIM=$(grep "Average SSIM:" "$RESULT_FILE" | awk '{print $3}' | tr -d '\n')

        # Handle empty values
        GENUINE_SIM=${GENUINE_SIM:-N/A}
        IMPOSTOR_SIM=${IMPOSTOR_SIM:-N/A}
        EER=${EER:-N/A}
        TAR_001=${TAR_001:-N/A}
        TAR_1=${TAR_1:-N/A}
        PSNR=${PSNR:-N/A}
        SSIM=${SSIM:-N/A}

        printf "%-25s | %-12s | %-12s | %-10s | %-12s | %-12s | %-8s | %-8s\n" \
            "$CONFIG_LABEL" "$GENUINE_SIM" "$IMPOSTOR_SIM" "$EER" "$TAR_001" "$TAR_1" "$PSNR" "$SSIM" >> "$COMPARISON_FILE"
    done
done

echo "" >> "$COMPARISON_FILE"
echo "================================================================================\n" >> "$COMPARISON_FILE"
echo "METRIC DEFINITIONS:" >> "$COMPARISON_FILE"
echo "  • Genuine Similarity: Face similarity for same person (HIGHER is better)" >> "$COMPARISON_FILE"
echo "  • Impostor Similarity: Face similarity for different people (LOWER is better)" >> "$COMPARISON_FILE"
echo "  • EER: Equal Error Rate where FRR=FAR (LOWER is better)" >> "$COMPARISON_FILE"
echo "  • TAR@FAR: True Accept Rate at False Accept Rate threshold (HIGHER is better)" >> "$COMPARISON_FILE"
echo "  • PSNR: Peak Signal-to-Noise Ratio in dB (HIGHER is better)" >> "$COMPARISON_FILE"
echo "  • SSIM: Structural Similarity Index (HIGHER is better, max=1.0)" >> "$COMPARISON_FILE"
echo "" >> "$COMPARISON_FILE"
echo "NOTE: All metrics computed on enhanced images compared to ground truth." >> "$COMPARISON_FILE"
echo "      For genuine pairs, low-light and enhanced images are compared to GT." >> "$COMPARISON_FILE"
echo "      For impostor pairs, different identities are compared." >> "$COMPARISON_FILE"
echo "" >> "$COMPARISON_FILE"

# Display comparison table
echo ""
cat "$COMPARISON_FILE"
echo ""
echo "Comparison table saved to: $COMPARISON_FILE"

echo ""
echo "========================================================================"
echo "EVALUATION COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved to: $RESULTS_BASE/"
echo ""
echo "Individual result files:"
for config in "${MODELS[@]}"; do
    for d_weight in "${D_WEIGHTS[@]}"; do
        RESULT_FILE="$RESULTS_BASE/${config}_d${d_weight}/face_verification_results.txt"
        if [ -f "$RESULT_FILE" ]; then
            echo "  • $RESULT_FILE"
        fi
    done
done
echo ""
echo "Next steps:"
echo "  1. Review comparison table above"
echo "  2. Run extended analysis: python extended_analysis.py"
echo "  3. Generate plots: python plot_ablation_results.py --results_dir=$RESULTS_BASE"
echo ""
