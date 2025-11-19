#!/bin/bash
# Ablation Study Setup Diagnostic Script
# Run this to check if everything is ready for the ablation study

echo "========================================================================"
echo "Ablation Study Setup Diagnostic"
echo "========================================================================"
echo ""

# Configuration (should match ablation_study.sh)
EPOCHS=50
DATASET_DIR="./datasets/LFW_lowlight"
ADAFACE_WEIGHTS="./weights/adaface/adaface_ir50_webface4m.ckpt"

ISSUES_FOUND=0

echo "[1/4] Checking trained models..."
echo ""
for config in baseline fr_weight_0.3 fr_weight_0.5 fr_weight_1.0; do
    MODEL="./weights/ablation/$config/epoch_$EPOCHS.pth"
    if [ -f "$MODEL" ]; then
        SIZE=$(du -h "$MODEL" | cut -f1)
        echo "  ✓ $config: Found ($SIZE)"
    else
        echo "  ✗ $config: NOT FOUND"
        echo "     Expected: $MODEL"
        ISSUES_FOUND=$((ISSUES_FOUND + 1))
    fi
done

echo ""
echo "[2/4] Checking test dataset..."
echo ""

LOW_DIR="$DATASET_DIR/test/low"
HIGH_DIR="$DATASET_DIR/test/high"

if [ -d "$LOW_DIR" ]; then
    LOW_COUNT=$(ls -1 "$LOW_DIR" 2>/dev/null | wc -l)
    echo "  ✓ Low-light images: Found ($LOW_COUNT files)"
else
    echo "  ✗ Low-light images: Directory not found"
    echo "     Expected: $LOW_DIR"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
fi

if [ -d "$HIGH_DIR" ]; then
    HIGH_COUNT=$(ls -1 "$HIGH_DIR" 2>/dev/null | wc -l)
    echo "  ✓ High-quality images: Found ($HIGH_COUNT files)"
else
    echo "  ✗ High-quality images: Directory not found"
    echo "     Expected: $HIGH_DIR"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
fi

echo ""
echo "[3/4] Checking AdaFace weights..."
echo ""

if [ -f "$ADAFACE_WEIGHTS" ]; then
    SIZE=$(du -h "$ADAFACE_WEIGHTS" | cut -f1)
    echo "  ✓ AdaFace weights: Found ($SIZE)"
else
    echo "  ✗ AdaFace weights: NOT FOUND"
    echo "     Expected: $ADAFACE_WEIGHTS"
    echo ""
    echo "  Download from:"
    echo "    https://github.com/mk-minchul/AdaFace/releases/download/v1.0/adaface_ir50_webface4m.ckpt"
    echo ""
    echo "  Or run:"
    echo "    mkdir -p ./weights/adaface"
    echo "    wget -O $ADAFACE_WEIGHTS \\"
    echo "      https://github.com/mk-minchul/AdaFace/releases/download/v1.0/adaface_ir50_webface4m.ckpt"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
fi

echo ""
echo "[4/4] Checking pairs file..."
echo ""

PAIRS_FILE="./pairs_lfw.txt"
if [ -f "$PAIRS_FILE" ]; then
    PAIRS_COUNT=$(grep -v "^#" "$PAIRS_FILE" | grep -v "^$" | wc -l)
    echo "  ✓ Pairs file: Found ($PAIRS_COUNT pairs)"
else
    echo "  ⚠ Pairs file: Not found (will be generated)"
    echo "     Expected: $PAIRS_FILE"
    echo "     This is OK - the script will generate it automatically"
fi

echo ""
echo "========================================================================"
echo "Summary"
echo "========================================================================"
echo ""

if [ $ISSUES_FOUND -eq 0 ]; then
    echo "✓ All checks passed! You're ready to run the ablation study."
    echo ""
    echo "To run the evaluation only (if models are already trained):"
    echo "  1. Comment out the training sections in examples/ablation_study.sh"
    echo "  2. Run: bash examples/ablation_study.sh"
    echo ""
    echo "Or to re-run everything:"
    echo "  bash examples/ablation_study.sh"
else
    echo "✗ Found $ISSUES_FOUND issue(s) that need to be fixed."
    echo ""
    echo "Common solutions:"
    echo ""
    echo "1. If models are missing:"
    echo "   - Did training complete successfully?"
    echo "   - Check for error messages in training output"
    echo "   - Training should create: ./weights/ablation/<config>/epoch_$EPOCHS.pth"
    echo ""
    echo "2. If dataset is missing:"
    echo "   - Run: python prepare_lfw_dataset.py --download"
    echo "   - Or check if dataset is in a different location"
    echo ""
    echo "3. If AdaFace weights are missing:"
    echo "   - Download from the URL shown above"
    echo "   - Save to: $ADAFACE_WEIGHTS"
    echo ""
fi

echo ""
exit $ISSUES_FOUND
