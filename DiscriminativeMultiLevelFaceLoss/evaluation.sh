#!/bin/bash

# Evaluation Script for Discriminative Multi-Level Face Loss
# This script evaluates trained models and generates comprehensive comparison results

set -e  # Exit on any error
set -o pipefail

echo "================================================================================"
echo "DISCRIMINATIVE MULTI-LEVEL FACE LOSS - MODEL EVALUATION"
echo "================================================================================"
echo ""

# Configuration
DATASET_DIR="./datasets/LFW_lowlight/test"
PAIRS_FILE="./pairs.txt"
FACE_WEIGHTS="./weights/adaface/adaface_ir50_webface4m.ckpt"
RESULTS_BASE="./results/discriminative"
LOG_DIR="./logs/discriminative"

# Create directories
mkdir -p ${RESULTS_BASE}
mkdir -p ${LOG_DIR}

# Timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/evaluation_${TIMESTAMP}.log"

echo "Starting evaluation at $(date)" | tee "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Check if required files exist
echo "Checking prerequisites..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ ! -d "$DATASET_DIR" ]; then
    echo "✗ Error: Test dataset not found at $DATASET_DIR" | tee -a "$LOG_FILE"
    exit 1
fi

if [ ! -f "$PAIRS_FILE" ]; then
    echo "✗ Error: Pairs file not found at $PAIRS_FILE" | tee -a "$LOG_FILE"
    exit 1
fi

if [ ! -f "$FACE_WEIGHTS" ]; then
    echo "✗ Error: AdaFace weights not found at $FACE_WEIGHTS" | tee -a "$LOG_FILE"
    exit 1
fi

echo "✓ All prerequisites found" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Define models array for later use in summary
MODELS=(
    "baseline_d1.5_reference"
    "discriminative_fr0.3_d1.5"
    "discriminative_fr0.5_d1.5"
)

# ============================================================================
# STEP 1: Evaluate Individual Models
# ============================================================================
echo "================================================================================" | tee -a "$LOG_FILE"
echo "STEP 1: Evaluating Individual Models" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Models to evaluate
MODELS=(
    "baseline_d1.5_reference"
    "discriminative_fr0.3_d1.5"
    "discriminative_fr0.5_d1.5"
)

# Evaluate each model
for model in "${MODELS[@]}"; do
    MODEL_PATH="./weights/${model}/epoch_50.pth"
    OUTPUT_DIR="${RESULTS_BASE}/${model}"
    
    echo "------------------------------------------------------------------------" | tee -a "$LOG_FILE"
    echo "Evaluating: ${model}" | tee -a "$LOG_FILE"
    echo "------------------------------------------------------------------------" | tee -a "$LOG_FILE"
    echo "  Model: ${MODEL_PATH}" | tee -a "$LOG_FILE"
    echo "  Output: ${OUTPUT_DIR}" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    # Check if model exists
    if [ ! -f "$MODEL_PATH" ]; then
        echo "⚠ Warning: Model not found, skipping: $MODEL_PATH" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        continue
    fi
    
    # Run evaluation
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

echo "✓ All model evaluations completed" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================================
# STEP 2: Generate Comparison and Thesis Results
# ============================================================================
echo "================================================================================" | tee -a "$LOG_FILE"
echo "STEP 2: Generating Comparison and Thesis Results" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ -f "generate_thesis_results.py" ]; then
    echo "Generating comprehensive comparison..." | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # Create symbolic links with expected names for the script to find
    mkdir -p "${RESULTS_BASE}/temp_links"

    # Link baseline
    if [ -d "${RESULTS_BASE}/baseline_d1.5_reference" ]; then
        ln -sf "$(cd "${RESULTS_BASE}/baseline_d1.5_reference" && pwd)" "${RESULTS_BASE}/temp_links/baseline_d1.5"
    fi

    # Link discriminative models
    if [ -d "${RESULTS_BASE}/discriminative_fr0.3_d1.5" ]; then
        ln -sf "$(cd "${RESULTS_BASE}/discriminative_fr0.3_d1.5" && pwd)" "${RESULTS_BASE}/temp_links/fr_weight_0.3_d1.5"
    fi

    if [ -d "${RESULTS_BASE}/discriminative_fr0.5_d1.5" ]; then
        ln -sf "$(cd "${RESULTS_BASE}/discriminative_fr0.5_d1.5" && pwd)" "${RESULTS_BASE}/temp_links/fr_weight_0.5_d1.5"
    fi

    # Run the script on the temp_links directory
    python generate_thesis_results.py \
        --results_dir="${RESULTS_BASE}/temp_links" \
        --output_dir="${RESULTS_BASE}" 2>&1 | tee -a "$LOG_FILE"

    # Clean up symbolic links
    rm -rf "${RESULTS_BASE}/temp_links"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "" | tee -a "$LOG_FILE"
        echo "✗ Thesis results generation failed" | tee -a "$LOG_FILE"
        exit 1
    fi
    
    echo "" | tee -a "$LOG_FILE"
    echo "✓ Thesis results generated" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
else
    echo "⚠ Warning: generate_thesis_results.py not found, skipping comparison" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
fi

# ============================================================================
# STEP 3: Extended Analysis (Optional)
# ============================================================================
echo "================================================================================" | tee -a "$LOG_FILE"
echo "STEP 3: Extended Analysis (Optional)" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ ! -f "extended_analysis.py" ]; then
    echo "⚠ Skipping extended analysis (extended_analysis.py not found)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
else
    # Run extended analysis comparing baseline vs best FR model
    BASELINE_MODEL="./weights/baseline_d1.5_reference/epoch_50.pth"
    FR_MODEL="./weights/discriminative_fr0.5_d1.5/epoch_50.pth"  # Best performing FR model
    
    if [ -f "$BASELINE_MODEL" ] && [ -f "$FR_MODEL" ]; then
        echo "Running extended analysis: Baseline vs FR=0.5" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        
        OUTPUT_DIR="${RESULTS_BASE}/extended_analysis"
        mkdir -p "$OUTPUT_DIR"
        
        python extended_analysis.py \
            --baseline_model "$BASELINE_MODEL" \
            --fr_model "$FR_MODEL" \
            --test_dir "${DATASET_DIR}" \
            --pairs_file "${PAIRS_FILE}" \
            --face_weights "${FACE_WEIGHTS}" \
            --output_dir "$OUTPUT_DIR" \
            --analyses significance identity failures 2>&1 | tee -a "$LOG_FILE"
        
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "" | tee -a "$LOG_FILE"
            echo "⚠ Extended analysis failed (non-critical)" | tee -a "$LOG_FILE"
        else
            echo "" | tee -a "$LOG_FILE"
            echo "✓ Extended analysis completed" | tee -a "$LOG_FILE"
        fi
        echo "" | tee -a "$LOG_FILE"
    else
        echo "⚠ Skipping extended analysis (baseline or FR model not found)" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
    fi
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo "================================================================================" | tee -a "$LOG_FILE"
echo "EVALUATION COMPLETE!" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Completed at: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Results saved to: ${RESULTS_BASE}/" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Generated files:" | tee -a "$LOG_FILE"
echo "----------------" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# List individual model results
echo "1. Individual model evaluations:" | tee -a "$LOG_FILE"
for model in "${MODELS[@]}"; do
    RESULT_FILE="${RESULTS_BASE}/${model}/face_verification_results.txt"
    if [ -f "$RESULT_FILE" ]; then
        echo "   • ${RESULT_FILE}" | tee -a "$LOG_FILE"
    fi
done
echo "" | tee -a "$LOG_FILE"

# List comparison files
echo "2. Comparison and statistics:" | tee -a "$LOG_FILE"
if [ -f "${RESULTS_BASE}/comparison_table.txt" ]; then
    echo "   • ${RESULTS_BASE}/comparison_table.txt" | tee -a "$LOG_FILE"
fi
if [ -f "${RESULTS_BASE}/thesis_results_summary.txt" ]; then
    echo "   • ${RESULTS_BASE}/thesis_results_summary.txt" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# List visualization files
echo "3. Visualizations:" | tee -a "$LOG_FILE"
if [ -d "${RESULTS_BASE}/plots" ]; then
    for plot in "${RESULTS_BASE}/plots/"*.png; do
        if [ -f "$plot" ]; then
            echo "   • ${plot}" | tee -a "$LOG_FILE"
        fi
    done
fi
echo "" | tee -a "$LOG_FILE"

# List extended analysis files
if [ -d "${RESULTS_BASE}/extended_analysis" ]; then
    echo "4. Extended analysis:" | tee -a "$LOG_FILE"
    for file in "${RESULTS_BASE}/extended_analysis/"*.{txt,csv,png}; do
        if [ -f "$file" ]; then
            echo "   • ${file}" | tee -a "$LOG_FILE"
        fi
    done
    echo "" | tee -a "$LOG_FILE"
fi

echo "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Show quick preview of results if available
if [ -f "${RESULTS_BASE}/thesis_results_summary.txt" ]; then
    echo "================================================================================" | tee -a "$LOG_FILE"
    echo "QUICK PREVIEW OF KEY FINDINGS" | tee -a "$LOG_FILE"
    echo "================================================================================" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    grep -A 20 "KEY FINDINGS" "${RESULTS_BASE}/thesis_results_summary.txt" 2>/dev/null | tee -a "$LOG_FILE" || \
    head -30 "${RESULTS_BASE}/thesis_results_summary.txt" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
fi

echo "================================================================================" | tee -a "$LOG_FILE"
echo "NEXT STEPS" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "1. Review the comparison table:" | tee -a "$LOG_FILE"
echo "   cat ${RESULTS_BASE}/thesis_results_summary.txt" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "2. Check which FR weight performs best (0.3, 0.5, or 1.0)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "3. Verify statistical significance (if p-values are available)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "4. Include the generated plots in your thesis" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "✓ Evaluation complete! 🎓" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
