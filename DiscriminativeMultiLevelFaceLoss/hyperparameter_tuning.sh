#!/bin/bash

# Hyperparameter Tuning Script for Discriminative Multi-Level Face Loss
# This script systematically tests different contrastive margin and weight values
# to find the optimal configuration for face recognition perceptual loss

# Configuration
DATASET="--lfw"  # Using LFW dataset for face-focused training
EPOCHS=100
BATCH_SIZE=8
BASE_WEIGHTS_DIR="./weights/hyperparameter_tuning"
LOG_DIR="./logs/hyperparameter_tuning"
RESULTS_FILE="${LOG_DIR}/tuning_results.txt"

# Create directories
mkdir -p ${BASE_WEIGHTS_DIR}
mkdir -p ${LOG_DIR}

# Initialize results file
echo "Hyperparameter Tuning Results - $(date)" > ${RESULTS_FILE}
echo "================================================" >> ${RESULTS_FILE}
echo "" >> ${RESULTS_FILE}

# Base training arguments (modify as needed)
BASE_ARGS="--use_face_loss --nEpochs=${EPOCHS} --batchSize=${BATCH_SIZE} ${DATASET} --FR_weight=0.5"

echo "Starting Hyperparameter Tuning..."
echo "=================================="
echo ""

# Phase 1: Tune Contrastive Margin (with default weight=1.0)
echo "Phase 1: Tuning Contrastive Margin (weight=1.0)"
echo "------------------------------------------------"

for MARGIN in 0.3 0.4 0.5; do
    RUN_NAME="margin_${MARGIN}_weight_1.0"
    WEIGHTS_DIR="${BASE_WEIGHTS_DIR}/${RUN_NAME}"
    LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
    
    echo "Training with margin=${MARGIN}, weight=1.0"
    echo "  Weights: ${WEIGHTS_DIR}"
    echo "  Log: ${LOG_FILE}"
    
    # Run training
    python train.py ${BASE_ARGS} \
        --contrastive_margin=${MARGIN} \
        --contrastive_weight=1.0 \
        --val_folder="${WEIGHTS_DIR}/" \
        > ${LOG_FILE} 2>&1
    
    # Extract best metrics from log
    BEST_PSNR=$(grep "Best PSNR" ${LOG_FILE} | tail -1 | awk '{print $NF}')
    BEST_SSIM=$(grep "Best SSIM" ${LOG_FILE} | tail -1 | awk '{print $NF}')
    
    # Log results
    echo "Run: ${RUN_NAME}" >> ${RESULTS_FILE}
    echo "  Margin: ${MARGIN}" >> ${RESULTS_FILE}
    echo "  Weight: 1.0" >> ${RESULTS_FILE}
    echo "  Best PSNR: ${BEST_PSNR}" >> ${RESULTS_FILE}
    echo "  Best SSIM: ${BEST_SSIM}" >> ${RESULTS_FILE}
    echo "" >> ${RESULTS_FILE}
    
    echo "Completed: ${RUN_NAME} (PSNR: ${BEST_PSNR}, SSIM: ${BEST_SSIM})"
    echo ""
done

echo ""
echo "Phase 2: Tuning Contrastive Weight (with default margin=0.4)"
echo "------------------------------------------------------------"

# Phase 2: Tune Contrastive Weight (with default margin=0.4)
for C_WEIGHT in 0.5 1.0 1.5; do
    RUN_NAME="margin_0.4_weight_${C_WEIGHT}"
    WEIGHTS_DIR="${BASE_WEIGHTS_DIR}/${RUN_NAME}"
    LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
    
    echo "Training with margin=0.4, weight=${C_WEIGHT}"
    echo "  Weights: ${WEIGHTS_DIR}"
    echo "  Log: ${LOG_FILE}"
    
    # Run training
    python train.py ${BASE_ARGS} \
        --contrastive_margin=0.4 \
        --contrastive_weight=${C_WEIGHT} \
        --val_folder="${WEIGHTS_DIR}/" \
        > ${LOG_FILE} 2>&1
    
    # Extract best metrics from log
    BEST_PSNR=$(grep "Best PSNR" ${LOG_FILE} | tail -1 | awk '{print $NF}')
    BEST_SSIM=$(grep "Best SSIM" ${LOG_FILE} | tail -1 | awk '{print $NF}')
    
    # Log results
    echo "Run: ${RUN_NAME}" >> ${RESULTS_FILE}
    echo "  Margin: 0.4" >> ${RESULTS_FILE}
    echo "  Weight: ${C_WEIGHT}" >> ${RESULTS_FILE}
    echo "  Best PSNR: ${BEST_PSNR}" >> ${RESULTS_FILE}
    echo "  Best SSIM: ${BEST_SSIM}" >> ${RESULTS_FILE}
    echo "" >> ${RESULTS_FILE}
    
    echo "Completed: ${RUN_NAME} (PSNR: ${BEST_PSNR}, SSIM: ${BEST_SSIM})"
    echo ""
done

echo ""
echo "Phase 3: Grid Search on Best Candidates"
echo "----------------------------------------"

# Phase 3: Fine-grained grid search (optional, based on Phase 1 & 2 results)
# Test combinations of the best margins and weights
BEST_MARGINS=(0.3 0.4)  # Adjust based on Phase 1 results
BEST_WEIGHTS=(1.0 1.5)  # Adjust based on Phase 2 results

for MARGIN in "${BEST_MARGINS[@]}"; do
    for C_WEIGHT in "${BEST_WEIGHTS[@]}"; do
        # Skip if already tested in Phase 1 or 2
        if [[ "${MARGIN}" == "0.4" && "${C_WEIGHT}" == "1.0" ]]; then
            echo "Skipping margin=${MARGIN}, weight=${C_WEIGHT} (already tested)"
            continue
        fi
        
        RUN_NAME="margin_${MARGIN}_weight_${C_WEIGHT}"
        WEIGHTS_DIR="${BASE_WEIGHTS_DIR}/${RUN_NAME}"
        LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
        
        echo "Training with margin=${MARGIN}, weight=${C_WEIGHT}"
        echo "  Weights: ${WEIGHTS_DIR}"
        echo "  Log: ${LOG_FILE}"
        
        # Run training
        python train.py ${BASE_ARGS} \
            --contrastive_margin=${MARGIN} \
            --contrastive_weight=${C_WEIGHT} \
            --val_folder="${WEIGHTS_DIR}/" \
            > ${LOG_FILE} 2>&1
        
        # Extract best metrics from log
        BEST_PSNR=$(grep "Best PSNR" ${LOG_FILE} | tail -1 | awk '{print $NF}')
        BEST_SSIM=$(grep "Best SSIM" ${LOG_FILE} | tail -1 | awk '{print $NF}')
        
        # Log results
        echo "Run: ${RUN_NAME}" >> ${RESULTS_FILE}
        echo "  Margin: ${MARGIN}" >> ${RESULTS_FILE}
        echo "  Weight: ${C_WEIGHT}" >> ${RESULTS_FILE}
        echo "  Best PSNR: ${BEST_PSNR}" >> ${RESULTS_FILE}
        echo "  Best SSIM: ${BEST_SSIM}" >> ${RESULTS_FILE}
        echo "" >> ${RESULTS_FILE}
        
        echo "Completed: ${RUN_NAME} (PSNR: ${BEST_PSNR}, SSIM: ${BEST_SSIM})"
        echo ""
    done
done

# Summary
echo ""
echo "================================================"
echo "Hyperparameter Tuning Complete!"
echo "================================================"
echo ""
echo "Results saved to: ${RESULTS_FILE}"
echo ""
echo "Summary of all runs:"
cat ${RESULTS_FILE}

# Find best configuration
echo ""
echo "Analyzing best configuration..."
echo "Check ${RESULTS_FILE} for detailed results"
echo ""
echo "To analyze results programmatically, you can parse ${RESULTS_FILE}"
echo "or check the validation metrics in each model's weights directory."
