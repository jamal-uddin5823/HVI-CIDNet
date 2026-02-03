#!/bin/bash
# Master script to generate all thesis figures
# Run on HPC where dataset is available

echo "========================================================================"
echo "Generating All Thesis Figures"
echo "========================================================================"

# Step 1: Extract and validate data
echo "[1/11] Extracting and validating data..."
python extract_thesis_data.py
if [ $? -ne 0 ]; then
    echo "ERROR: Data extraction failed!"
    exit 1
fi

# Step 2: Generate figures
echo "[2/11] Generating Figure 1: Dataset methodology..."
python generate_figure1_dataset.py

echo "[3/11] Skipping Figure 2 (Mermaid diagram - render separately)..."

echo "[4/11] Skipping Figure 3 (Mermaid diagram - render separately)..."

echo "[5/11] Generating Figure 4: Training curves..."
python generate_figure4_training.py

echo "[6/11] Generating Figure 6: Verification performance..."
python generate_figure6_verification.py

echo "[7/11] Generating Figure 7: Quality-verification trade-off..."
python generate_figure7_tradeoff.py

echo "[8/11] Generating Figure 8: ROC curves..."
python generate_figure8_roc.py

echo "[9/11] Generating Figure 9: Score distributions..."
python generate_figure9_distributions.py

echo "[10/11] Generating Figure 10: Enhancement impact..."
python generate_figure10_baseline_comparison.py

echo "========================================================================"
echo "All figures generated successfully!"
echo "========================================================================"
echo ""
echo "Output directory: figures/"
echo "PDF files: figures/*.pdf"
echo "PNG files: figures/*.png"
echo ""
echo "Mermaid diagrams (render separately):"
echo "  - figure2_loss_architecture.mmd"
echo "  - figure3_data_splitting.mmd"
echo ""
echo "Use https://mermaid.live or mermaid-cli to render .mmd files"
