#!/bin/bash
# MASTER SCRIPT: Complete Face Recognition Ablation Study Analysis
#
# This script runs the full evaluation pipeline:
# 1. Full evaluation on all 2000 pairs for all 4 models
# 2. Statistical analysis and comparison
# 3. Publication-quality visualizations
#
# Usage on HPC:
#   bash RUN_COMPLETE_ANALYSIS.sh
#
# Or run steps individually (see below)

set -e  # Exit on any error
set -o pipefail

# Trap to handle interrupts and errors
trap 'echo ""; echo "✗ Script interrupted or failed at line $LINENO. Exiting..."; exit 130' INT TERM ERR

echo "================================================================================"
echo "COMPLETE FACE RECOGNITION ABLATION STUDY ANALYSIS"
echo "================================================================================"
echo ""
echo "This will:"
echo "  1. Evaluate 4 models (baseline, FR=0.3, FR=0.5, FR=1.0) on 2000 pairs"
echo "  2. Generate statistical significance tests"
echo "  3. Create publication-ready comparison tables"
echo "  4. Generate thesis-quality visualizations"
echo ""
echo "Estimated time: 1-2 hours (depending on GPU)"
echo ""
echo "================================================================================"
echo ""

# Check if running on HPC
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "⚠ Warning: CUDA_VISIBLE_DEVICES not set"
    echo "  If you have multiple GPUs, you may want to set this"
    echo ""
fi

# Create logs directory
mkdir -p logs

# Timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/complete_analysis_${TIMESTAMP}.log"

echo "Starting analysis at $(date)"
echo "Logging to: $LOG_FILE"
echo ""

# ============================================================================
# STEP 1: Full Evaluation on All Models
# ============================================================================
echo "================================================================================"
echo "STEP 1/3: Full Evaluation on All Models (2000 pairs each)"
echo "================================================================================"
echo ""

if [ -f "run_full_evaluation.sh" ]; then
    echo "Running full evaluation script..."
    echo ""
    bash run_full_evaluation.sh 2>&1 | tee -a "$LOG_FILE"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo ""
        echo "✗ Step 1 failed. Check log: $LOG_FILE"
        exit 1
    fi
    
    echo ""
    echo "✓ Step 1 completed successfully"
    echo ""
else
    echo "✗ Error: run_full_evaluation.sh not found"
    exit 1
fi

# ============================================================================
# STEP 2: Generate Comprehensive Results and Statistics
# ============================================================================
echo "================================================================================"
echo "STEP 2/3: Statistical Analysis and Comparison Tables"
echo "================================================================================"
echo ""

if [ -f "generate_thesis_results.py" ]; then
    echo "Generating thesis results..."
    echo ""
    python generate_thesis_results.py \
        --results_dir ./results/full_evaluation \
        --output_dir ./results/full_evaluation 2>&1 | tee -a "$LOG_FILE"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo ""
        echo "✗ Step 2 failed. Check log: $LOG_FILE"
        exit 1
    fi
    
    echo ""
    echo "✓ Step 2 completed successfully"
    echo ""
else
    echo "⚠ Warning: generate_thesis_results.py not found, skipping statistical analysis"
    echo ""
fi

# ============================================================================
# STEP 3: Extended Analysis (if available)
# ============================================================================
echo "================================================================================"
echo "STEP 3/3: Extended Analysis (Optional)"
echo "================================================================================"
echo ""
echo "Running extended analysis for each D_weight:"
echo "  • Compares each baseline vs its best FR model (same D_weight)"
echo "  • Generates statistical tests, per-identity analysis, failure cases"
echo ""

if [ ! -f "extended_analysis.py" ]; then
    echo "⚠ Skipping extended analysis (extended_analysis.py not found)"
    echo ""
else
    # D_weights to analyze
    D_WEIGHTS=(0.5 1 1.5)
    FR_WEIGHTS=(0.3 0.5 1.0)
    
    ANALYSIS_COUNT=0
    
    for d_weight in "${D_WEIGHTS[@]}"; do
        BASELINE_MODEL="./weights/ablation/baseline/d_${d_weight}/epoch_50.pth"
        
        # Skip if baseline doesn't exist
        if [ ! -f "$BASELINE_MODEL" ]; then
            echo "⚠ Skipping d_weight=$d_weight: baseline model not found"
            continue
        fi
        
        # Find best FR model for this d_weight (by checking which exists)
        # Priority: FR=0.5 > FR=1.0 > FR=0.3
        BEST_FR_MODEL=""
        BEST_FR_WEIGHT=""
        
        for fr_weight in 0.5 1.0 0.3; do
            FR_MODEL="./weights/ablation/fr_weight_${fr_weight}/d_${d_weight}/epoch_50.pth"
            if [ -f "$FR_MODEL" ]; then
                BEST_FR_MODEL="$FR_MODEL"
                BEST_FR_WEIGHT="$fr_weight"
                break
            fi
        done
        
        if [ -z "$BEST_FR_MODEL" ]; then
            echo "⚠ Skipping d_weight=$d_weight: no FR models found"
            continue
        fi
        
        echo ""
        echo "------------------------------------------------------------------------"
        echo "Extended Analysis: D_weight=$d_weight"
        echo "------------------------------------------------------------------------"
        echo "  Baseline: baseline/d_${d_weight}"
        echo "  FR Model: fr_weight_${BEST_FR_WEIGHT}/d_${d_weight}"
        echo ""
        
        OUTPUT_DIR="./results/extended_analysis/d_${d_weight}"
        mkdir -p "$OUTPUT_DIR"
        
        python extended_analysis.py \
            --baseline_model "$BASELINE_MODEL" \
            --fr_model "$BEST_FR_MODEL" \
            --test_dir ./datasets/LFW_lowlight/test \
            --pairs_file ./pairs.txt \
            --face_weights ./weights/adaface/adaface_ir50_webface4m.ckpt \
            --output_dir "$OUTPUT_DIR" \
            --analyses significance identity failures 2>&1 | tee -a "$LOG_FILE"

        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo ""
            echo "✗ Extended analysis failed for d_weight=$d_weight (check log: $LOG_FILE)"
            exit 1
        fi
        
        echo ""
        echo "✓ Extended analysis completed for d_weight=$d_weight"
        ANALYSIS_COUNT=$((ANALYSIS_COUNT + 1))
        echo ""
    done
    
    if [ $ANALYSIS_COUNT -eq 0 ]; then
        echo "⚠ No extended analyses were run (missing models)"
    else
        echo "✓ Completed $ANALYSIS_COUNT extended analysis/analyses"
    fi
    echo ""
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "================================================================================"
echo "ANALYSIS COMPLETE!"
echo "================================================================================"
echo ""
echo "Completed at: $(date)"
echo ""
echo "Results location: ./results/"
echo ""
echo "Generated files:"
echo "----------------"
echo ""
echo "1. Individual model evaluations (for each FR weight × D weight combination):"
echo "   Example locations:"
echo "   • ./results/full_evaluation/baseline_d0.5/face_verification_results.txt"
echo "   • ./results/full_evaluation/baseline_d1/face_verification_results.txt"
echo "   • ./results/full_evaluation/baseline_d1.5/face_verification_results.txt"
echo "   • ./results/full_evaluation/fr_weight_0.3_d0.5/face_verification_results.txt"
echo "   • ./results/full_evaluation/fr_weight_0.5_d1/face_verification_results.txt"
echo "   • ./results/full_evaluation/fr_weight_1.0_d1.5/face_verification_results.txt"
echo "   (Up to 12 combinations evaluated: 4 configs × 3 d_weights)"
echo ""
echo "2. Comparison and statistics:"
echo "   • ./results/full_evaluation/comparison_table.txt"
echo "   • ./results/full_evaluation/thesis_results_summary.txt"
echo ""
echo "3. Visualizations:"
echo "   • ./results/full_evaluation/plots/verification_metrics.png"
echo "   • ./results/full_evaluation/plots/image_quality.png"
echo "   • ./results/full_evaluation/plots/tradeoff_analysis.png"
echo ""

if [ -d "./results/extended_analysis" ]; then
    echo "4. Extended analysis (per D_weight):"
    for d_weight in 0.5 1 1.5; do
        if [ -f "./results/extended_analysis/d_${d_weight}/statistical_significance.txt" ]; then
            echo "   • ./results/extended_analysis/d_${d_weight}/statistical_significance.txt"
            echo "   • ./results/extended_analysis/d_${d_weight}/per_identity_analysis.csv"
            echo "   • ./results/extended_analysis/d_${d_weight}/failure_cases.png"
        fi
    done
    echo ""
fi

echo "Log file: $LOG_FILE"
echo ""
echo "================================================================================"
echo "NEXT STEPS FOR YOUR THESIS"
echo "================================================================================"
echo ""
echo "1. Review the comparison table:"
echo "   cat ./results/full_evaluation/thesis_results_summary.txt"
echo ""
echo "2. Check which configuration is best (typically FR=0.5 or FR=1.0)"
echo ""
echo "3. Verify statistical significance (p < 0.05)"
echo ""
echo "4. Include the generated plots in your thesis"
echo ""
echo "5. If results look good, copy to your local machine:"
echo "   scp -r hpc4090@hpc:/path/to/results ./thesis_results/"
echo ""
echo "================================================================================"
echo ""

# Final check
if [ -f "./results/full_evaluation/thesis_results_summary.txt" ]; then
    echo "Quick preview of key findings:"
    echo ""
    grep -A 20 "KEY FINDINGS" ./results/full_evaluation/thesis_results_summary.txt || echo "(Key findings section not found)"
    echo ""
fi

echo "✓ All done! Good luck with your thesis! 🎓"
echo ""
