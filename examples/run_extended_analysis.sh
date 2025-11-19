#!/bin/bash
# Extended Analysis Script
# Runs statistical significance tests, per-identity analysis, and failure case analysis

echo "========================================================================"
echo "Extended Analysis for Ablation Study"
echo "========================================================================"
echo ""
echo "This script will perform:"
echo "  1. Statistical Significance Test (McNemar's test, paired t-test)"
echo "  2. Per-Identity Analysis (which identities benefit most)"
echo "  3. Failure Case Analysis (baseline fails, FR succeeds)"
echo ""

# Configuration
DATASET_DIR="./datasets/LFW_lowlight"
PAIRS_FILE="./pairs_lfw.txt"
BASELINE_MODEL="./weights/ablation/baseline/epoch_50.pth"
FR_MODEL="./weights/ablation/fr_weight_0.5/epoch_50.pth"
ADAFACE_WEIGHTS="./weights/adaface/adaface_ir50_webface4m.ckpt"
OUTPUT_DIR="./results/extended_analysis"

# Check if models exist
if [ ! -f "$BASELINE_MODEL" ]; then
    echo "Error: Baseline model not found at: $BASELINE_MODEL"
    exit 1
fi

if [ ! -f "$FR_MODEL" ]; then
    echo "Error: FR model not found at: $FR_MODEL"
    exit 1
fi

if [ ! -f "$ADAFACE_WEIGHTS" ]; then
    echo "Error: AdaFace weights not found at: $ADAFACE_WEIGHTS"
    echo "Please download from: https://github.com/mk-minchul/AdaFace/releases"
    exit 1
fi

if [ ! -f "$PAIRS_FILE" ]; then
    echo "Error: Pairs file not found at: $PAIRS_FILE"
    echo "Please generate pairs first using: python generate_lfw_pairs.py"
    exit 1
fi

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Configuration:"
echo "  Baseline model: $BASELINE_MODEL"
echo "  FR model: $FR_MODEL"
echo "  Test directory: $DATASET_DIR/test"
echo "  Pairs file: $PAIRS_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Run extended analysis
echo "========================================================================"
echo "Running Extended Analysis..."
echo "========================================================================"
echo ""

python extended_analysis.py \
    --baseline_model=$BASELINE_MODEL \
    --fr_model=$FR_MODEL \
    --test_dir=$DATASET_DIR/test \
    --pairs_file=$PAIRS_FILE \
    --face_weights=$ADAFACE_WEIGHTS \
    --face_model=ir_50 \
    --output_dir=$OUTPUT_DIR \
    --analyses significance identity failures

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "Extended Analysis Complete!"
    echo "========================================================================"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    echo "  1. statistical_significance.txt - p-values and significance tests"
    echo "  2. per_identity_analysis.csv - per-identity performance data"
    echo "  3. per_identity_analysis.png - visualization of identity-level results"
    echo "  4. failure_cases.png - visual comparison of failure cases"
    echo "  5. failure_cases_summary.txt - list of failure cases"
    echo ""
    echo "For your thesis:"
    echo "  • Use statistical_significance.txt to report p-values"
    echo "  • Use per_identity_analysis.png to show FR loss helps difficult identities"
    echo "  • Use failure_cases.png to show qualitative examples"
    echo ""
else
    echo ""
    echo "========================================================================"
    echo "Extended Analysis Failed!"
    echo "========================================================================"
    echo ""
    echo "Please check the error messages above."
    exit 1
fi
