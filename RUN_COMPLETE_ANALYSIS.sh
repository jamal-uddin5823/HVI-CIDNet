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

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo ""
        echo "✓ Step 1 completed successfully"
        echo ""
    else
        echo ""
        echo "✗ Step 1 failed. Check log: $LOG_FILE"
        exit 1
    fi
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

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo ""
        echo "✓ Step 2 completed successfully"
        echo ""
    else
        echo ""
        echo "⚠ Step 2 had issues (check log), continuing..."
        echo ""
    fi
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

BASELINE_MODEL="./weights/ablation/baseline/epoch_50.pth"
FR_MODEL="./weights/ablation/fr_weight_0.5/epoch_50.pth"

if [ -f "extended_analysis.py" ] && [ -f "$BASELINE_MODEL" ] && [ -f "$FR_MODEL" ]; then
    echo "Running extended analysis (baseline vs FR=0.5)..."
    echo "This will generate:"
    echo "  • Statistical significance tests (McNemar's, t-test)"
    echo "  • Per-identity analysis"
    echo "  • Failure case visualizations"
    echo ""

    python extended_analysis.py \
        --baseline_model "$BASELINE_MODEL" \
        --fr_model "$FR_MODEL" \
        --test_dir ./datasets/LFW_lowlight/test \
        --pairs_file ./pairs.txt \
        --face_weights ./weights/adaface/adaface_ir50_webface4m.ckpt \
        --output_dir ./results/extended_analysis \
        --analyses significance identity failures 2>&1 | tee -a "$LOG_FILE"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo ""
        echo "✓ Step 3 completed successfully"
        echo ""
    else
        echo ""
        echo "⚠ Step 3 had issues (check log)"
        echo ""
    fi
else
    echo "⚠ Skipping extended analysis (missing files)"
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
echo "1. Individual model evaluations:"
echo "   • ./results/full_evaluation/baseline/face_verification_results.txt"
echo "   • ./results/full_evaluation/fr_weight_0.3/face_verification_results.txt"
echo "   • ./results/full_evaluation/fr_weight_0.5/face_verification_results.txt"
echo "   • ./results/full_evaluation/fr_weight_1.0/face_verification_results.txt"
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
    echo "4. Extended analysis:"
    echo "   • ./results/extended_analysis/statistical_significance.txt"
    echo "   • ./results/extended_analysis/per_identity_analysis.csv"
    echo "   • ./results/extended_analysis/failure_cases.png"
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
