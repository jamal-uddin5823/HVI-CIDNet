#!/bin/bash

# Evaluation Script for SS_Face (Eval-only split)
# Uses the same pipeline as evaluation.sh but points to SS_Face_lowlight

set -e
set -o pipefail

echo "================================================================================"
echo "SS_FACE - MODEL EVALUATION (EVAL-ONLY DATASET)"
echo "================================================================================"
echo ""

# Configuration
DATASET_DIR="./datasets/SS_Face_lowlight/test"
PAIRS_FILE="./pairs_ssface.txt"
FACE_WEIGHTS="./weights/adaface/adaface_ir50_webface4m.ckpt"
RESULTS_BASE="./results/ss_face"
LOG_DIR="./logs/discriminative"

# Create directories
mkdir -p ${RESULTS_BASE}
mkdir -p ${LOG_DIR}

# Timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/evaluation_ssface_${TIMESTAMP}.log"

echo "Starting evaluation at $(date)" | tee "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "Checking prerequisites..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ ! -d "$DATASET_DIR" ]; then
    echo "✗ Error: Test dataset not found at $DATASET_DIR" | tee -a "$LOG_FILE"
    echo "  Run: python prepare_ss_face_dataset.py --download" | tee -a "$LOG_FILE"
    exit 1
fi

if [ ! -f "$FACE_WEIGHTS" ]; then
    echo "✗ Error: AdaFace weights not found at $FACE_WEIGHTS" | tee -a "$LOG_FILE"
    exit 1
fi

# Auto-generate pairs if missing
if [ ! -f "$PAIRS_FILE" ]; then
    echo "Generating pairs file at $PAIRS_FILE ..." | tee -a "$LOG_FILE"
    python generate_ssface_pairs.py --test_dir="$DATASET_DIR" --output="$PAIRS_FILE" --num_pairs 2000 2>&1 | tee -a "$LOG_FILE"
fi

if [ ! -f "$PAIRS_FILE" ]; then
    echo "✗ Failed to generate pairs file: $PAIRS_FILE" | tee -a "$LOG_FILE"
    exit 1
fi

echo "✓ All prerequisites found" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Models to evaluate (adjust as needed)
MODELS=(
    "baseline_d1.5_reference"
    "discriminative_fr0.5_d1.5"
    "no_enhance_baseline"
)

for model in "${MODELS[@]}"; do
    # Special handling for no-enhance baseline
    if [ "$model" = "no_enhance_baseline" ]; then
        MODEL_PATH="dummy"  # Not used when --no-enhance is set
        OUTPUT_DIR="${RESULTS_BASE}/${model}"

        echo "------------------------------------------------------------------------" | tee -a "$LOG_FILE"
        echo "Evaluating: ${model} (NO ENHANCEMENT - Direct Recognition)" | tee -a "$LOG_FILE"
        echo "------------------------------------------------------------------------" | tee -a "$LOG_FILE"
        echo "  Mode: Running AdaFace directly on low-light images" | tee -a "$LOG_FILE"
        echo "  Output: ${OUTPUT_DIR}" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"

        python eval_face_verification.py \
            --model="${MODEL_PATH}" \
            --test_dir="${DATASET_DIR}" \
            --pairs_file="${PAIRS_FILE}" \
            --face_weights="${FACE_WEIGHTS}" \
            --output_dir="${OUTPUT_DIR}" \
            --no-enhance 2>&1 | tee -a "$LOG_FILE"

        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "" | tee -a "$LOG_FILE"
            echo "✗ Evaluation failed for ${model}" | tee -a "$LOG_FILE"
            exit 1
        fi

        echo "" | tee -a "$LOG_FILE"
        echo "✓ Completed: ${model}" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        continue
    fi

    MODEL_PATH="./weights/discriminative_0.01/${model}/epoch_50.pth"
    OUTPUT_DIR="${RESULTS_BASE}/${model}"

    echo "------------------------------------------------------------------------" | tee -a "$LOG_FILE"
    echo "Evaluating: ${model}" | tee -a "$LOG_FILE"
    echo "------------------------------------------------------------------------" | tee -a "$LOG_FILE"
    echo "  Model: ${MODEL_PATH}" | tee -a "$LOG_FILE"
    echo "  Output: ${OUTPUT_DIR}" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "⚠ Warning: Model not found, skipping: $MODEL_PATH" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        continue
    fi

    python eval_face_verification.py \
        --model="${MODEL_PATH}" \
        --test_dir="${DATASET_DIR}" \
        --pairs_file="${PAIRS_FILE}" \
        --face_weights="${FACE_WEIGHTS}" \
        --output_dir="${OUTPUT_DIR}" 2>&1 | tee -a "$LOG_FILE"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "" | tee -a "$LOG_FILE"
        echo "✗ Evaluation failed for ${model}" | tee -a "$LOG_FILE"
        exit 1
    fi

    echo "" | tee -a "$LOG_FILE"
    echo "✓ Completed: ${model}" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

done

echo "✓ SS_Face evaluation completed" | tee -a "$LOG_FILE"
echo "Results saved to: ${RESULTS_BASE}" | tee -a "$LOG_FILE"
