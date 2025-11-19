#!/bin/bash
# Pre-flight validation script
# Run this BEFORE the full evaluation to check everything is ready

echo "================================================================================"
echo "PRE-FLIGHT VALIDATION CHECK"
echo "================================================================================"
echo ""
echo "This script verifies that everything is ready for the full evaluation."
echo "Run this BEFORE 'RUN_COMPLETE_ANALYSIS.sh' to avoid wasting time."
echo ""

CHECKS_PASSED=0
CHECKS_FAILED=0

# Function to check file/directory
check_exists() {
    local path=$1
    local name=$2
    local type=$3  # "file" or "dir"

    if [ "$type" = "file" ]; then
        if [ -f "$path" ]; then
            echo "✓ $name"
            CHECKS_PASSED=$((CHECKS_PASSED + 1))
            return 0
        else
            echo "✗ $name - NOT FOUND"
            echo "  Expected: $path"
            CHECKS_FAILED=$((CHECKS_FAILED + 1))
            return 1
        fi
    elif [ "$type" = "dir" ]; then
        if [ -d "$path" ]; then
            echo "✓ $name"
            CHECKS_PASSED=$((CHECKS_PASSED + 1))
            return 0
        else
            echo "✗ $name - NOT FOUND"
            echo "  Expected: $path"
            CHECKS_FAILED=$((CHECKS_FAILED + 1))
            return 1
        fi
    fi
}

# 1. Check dataset
echo "1. Dataset Structure"
echo "--------------------"
check_exists "./datasets/LFW_lowlight" "Dataset directory" "dir"
check_exists "./datasets/LFW_lowlight/test" "Test directory" "dir"
check_exists "./datasets/LFW_lowlight/test/low" "Test low-light images" "dir"
check_exists "./datasets/LFW_lowlight/test/high" "Test ground truth images" "dir"

# Count images in test set
if [ -d "./datasets/LFW_lowlight/test/low" ]; then
    NUM_LOW=$(find ./datasets/LFW_lowlight/test/low -name "*.png" -o -name "*.jpg" | wc -l)
    NUM_HIGH=$(find ./datasets/LFW_lowlight/test/high -name "*.png" -o -name "*.jpg" | wc -l)
    echo "  → Found $NUM_LOW low-light images, $NUM_HIGH ground truth images"
    if [ $NUM_LOW -lt 100 ]; then
        echo "  ⚠ Warning: Very few test images ($NUM_LOW)"
        CHECKS_FAILED=$((CHECKS_FAILED + 1))
    fi
fi
echo ""

# 2. Check pairs file
echo "2. Pairs File"
echo "-------------"
if check_exists "./pairs.txt" "Pairs file" "file"; then
    NUM_PAIRS=$(grep -v "^#" ./pairs.txt | grep -v "^$" | wc -l)
    NUM_GENUINE=$(grep -v "^#" ./pairs.txt | grep " 1$" | wc -l)
    NUM_IMPOSTOR=$(grep -v "^#" ./pairs.txt | grep " 0$" | wc -l)

    echo "  → Total pairs: $NUM_PAIRS"
    echo "  → Genuine pairs (label=1): $NUM_GENUINE"
    echo "  → Impostor pairs (label=0): $NUM_IMPOSTOR"

    if [ $NUM_PAIRS -lt 1000 ]; then
        echo "  ⚠ Warning: Fewer than 1000 pairs ($NUM_PAIRS)"
        echo "    Recommended: At least 1000 genuine + 1000 impostor"
    fi

    if [ $NUM_GENUINE -eq 0 ] || [ $NUM_IMPOSTOR -eq 0 ]; then
        echo "  ✗ ERROR: Must have both genuine AND impostor pairs!"
        echo "    Genuine: $NUM_GENUINE, Impostor: $NUM_IMPOSTOR"
        CHECKS_FAILED=$((CHECKS_FAILED + 1))
    fi
fi
echo ""

# 3. Check trained models
echo "3. Trained Models (epoch 50)"
echo "----------------------------"
EPOCH=50
check_exists "./weights/ablation/baseline/epoch_$EPOCH.pth" "Baseline model" "file"
check_exists "./weights/ablation/fr_weight_0.3/epoch_$EPOCH.pth" "FR weight 0.3 model" "file"
check_exists "./weights/ablation/fr_weight_0.5/epoch_$EPOCH.pth" "FR weight 0.5 model" "file"
check_exists "./weights/ablation/fr_weight_1.0/epoch_$EPOCH.pth" "FR weight 1.0 model" "file"
echo ""

# 4. Check AdaFace weights
echo "4. AdaFace Weights"
echo "------------------"
if check_exists "./weights/adaface/adaface_ir50_webface4m.ckpt" "AdaFace IR-50 weights" "file"; then
    FILESIZE=$(stat -f%z "./weights/adaface/adaface_ir50_webface4m.ckpt" 2>/dev/null || stat -c%s "./weights/adaface/adaface_ir50_webface4m.ckpt" 2>/dev/null || echo "0")
    if [ $FILESIZE -gt 100000000 ]; then  # > 100MB
        echo "  → File size: $(echo "scale=1; $FILESIZE/1024/1024" | bc 2>/dev/null || echo "OK") MB"
    else
        echo "  ⚠ Warning: File may be corrupted (too small: $FILESIZE bytes)"
        CHECKS_FAILED=$((CHECKS_FAILED + 1))
    fi
fi
echo ""

# 5. Check Python environment
echo "5. Python Environment"
echo "---------------------"
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo "✓ Python: $PYTHON_VERSION"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))

    # Check key packages
    echo ""
    echo "  Checking Python packages..."

    python -c "import torch; print('  ✓ torch:', torch.__version__)" 2>/dev/null || {
        echo "  ✗ torch - NOT FOUND"
        CHECKS_FAILED=$((CHECKS_FAILED + 1))
    }

    python -c "import torchvision; print('  ✓ torchvision:', torchvision.__version__)" 2>/dev/null || {
        echo "  ✗ torchvision - NOT FOUND"
        CHECKS_FAILED=$((CHECKS_FAILED + 1))
    }

    python -c "import numpy; print('  ✓ numpy:', numpy.__version__)" 2>/dev/null || {
        echo "  ✗ numpy - NOT FOUND"
        CHECKS_FAILED=$((CHECKS_FAILED + 1))
    }

    python -c "import scipy; print('  ✓ scipy:', scipy.__version__)" 2>/dev/null || {
        echo "  ⚠ scipy - NOT FOUND (optional, for statistical tests)"
    }

    python -c "import matplotlib; print('  ✓ matplotlib:', matplotlib.__version__)" 2>/dev/null || {
        echo "  ⚠ matplotlib - NOT FOUND (optional, for plots)"
    }
else
    echo "✗ Python not found"
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
fi
echo ""

# 6. Check CUDA
echo "6. CUDA/GPU"
echo "-----------"
if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi available"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))

    # Check GPU
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    echo "  → GPUs detected: $GPU_COUNT"

    if [ $GPU_COUNT -gt 0 ]; then
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read line; do
            echo "  → $line"
        done
    fi

    # Check if PyTorch can use CUDA
    python -c "import torch; print('  → CUDA available in PyTorch:', torch.cuda.is_available())" 2>/dev/null
    python -c "import torch; print('  → CUDA device count:', torch.cuda.device_count())" 2>/dev/null
else
    echo "⚠ nvidia-smi not found - will run on CPU (very slow!)"
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
fi
echo ""

# 7. Check evaluation scripts
echo "7. Evaluation Scripts"
echo "---------------------"
check_exists "./eval_face_verification.py" "Main evaluation script" "file"
check_exists "./run_full_evaluation.sh" "Full evaluation runner" "file"
check_exists "./generate_thesis_results.py" "Results generator" "file"
check_exists "./RUN_COMPLETE_ANALYSIS.sh" "Master script" "file"
echo ""

# 8. Check disk space
echo "8. Disk Space"
echo "-------------"
AVAILABLE_GB=$(df -BG . | tail -1 | awk '{print $4}' | tr -d 'G')
echo "  Available: ${AVAILABLE_GB}GB"
if [ $AVAILABLE_GB -lt 10 ]; then
    echo "  ⚠ Warning: Low disk space (< 10GB)"
    echo "    Results may need ~5-10GB"
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
else
    echo "  ✓ Sufficient disk space"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
fi
echo ""

# Summary
echo "================================================================================"
echo "VALIDATION SUMMARY"
echo "================================================================================"
echo ""
echo "Checks passed: $CHECKS_PASSED"
echo "Checks failed: $CHECKS_FAILED"
echo ""

if [ $CHECKS_FAILED -eq 0 ]; then
    echo "✅ ALL CHECKS PASSED - READY TO RUN FULL EVALUATION!"
    echo ""
    echo "Next step:"
    echo "  bash RUN_COMPLETE_ANALYSIS.sh"
    echo ""
    echo "This will take approximately 1-2 hours."
    echo ""
    exit 0
else
    echo "❌ SOME CHECKS FAILED"
    echo ""
    echo "Please fix the issues above before running the full evaluation."
    echo ""
    echo "Common fixes:"
    echo "  • Missing dataset: python prepare_lfw_dataset.py --download"
    echo "  • Missing pairs: python generate_lfw_pairs.py --test_dir=./datasets/LFW_lowlight/test --num_pairs=1000"
    echo "  • Missing AdaFace: Download from https://github.com/mk-minchul/AdaFace/releases"
    echo "  • Missing models: Check that training completed successfully"
    echo ""
    exit 1
fi
