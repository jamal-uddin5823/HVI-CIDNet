#!/bin/bash

# Script to run complete multi-level training pipeline:
# 1. Generate training sets (with mixed)
# 2. Generate test sets (with mixed)
# 3. Train baseline, face_loss3, and face_loss5
# 4. Evaluate all models on mixed test set

set -e  # Exit on error

# Disable TQDM progress bars
export TQDM_DISABLE=1

# Create multilevel weights directory
mkdir -p weights/multilevel

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Complete Multi-Level Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"

# ============================================================
# STEP 1: Generate Training Sets (with mixed)
# ============================================================
echo -e "\n${BLUE}${BLUE}[STEP 1] Generating Training Sets...${NC}"
echo -e "${YELLOW}Command: python generate_multilevel_training_sets.py${NC}\n"

if python generate_multilevel_training_sets.py \
    --lfw_dir=./datasets/LFW_original/lfw \
    --output_base_dir=./datasets/LFW_multilevel \
    --generate_mixed \
    --use_symlinks; then
    echo -e "${GREEN}[$(date)] Training sets generated successfully!${NC}"
    echo -e "${GREEN}Output:${NC}"
    echo -e "  - train_easy/, train_medium/, train_hard/ (individual levels)"
    echo -e "  - train_mixed/ (combined via symlinks)"
    echo -e "  - val_easy/, val_medium/, val_hard/"
    echo -e "  - val_mixed/ (combined via symlinks)"
else
    echo -e "${RED}[$(date)] Failed to generate training sets${NC}"
    exit 1
fi

# ============================================================
# STEP 2: Generate Test Pairs and Mixed Test Set
# ============================================================
echo -e "\n${BLUE}[STEP 2] Generating Test Pairs and Mixed Set...${NC}"

# Test sets were already created in Step 1, now generate pairs for each level
test_levels=("easy" "medium" "hard")

for level in "${test_levels[@]}"; do
    test_dir="./datasets/LFW_multilevel/test_${level}"
    pairs_file="${test_dir}/pairs.txt"
    
    if [ -f "$pairs_file" ]; then
        echo -e "${YELLOW}pairs.txt already exists for test_${level}${NC}"
    else
        echo -e "${YELLOW}Generating pairs.txt for test_${level}...${NC}"
        python generate_lfw_pairs.py \
            --test_dir="${test_dir}" \
            --output="${pairs_file}" \
            --num_pairs=1000 \
            --seed=42
    fi
done

# Create test_mixed combining all difficulty levels via symlinks
echo -e "\n${YELLOW}Creating test_mixed with all difficulty levels...${NC}"

test_mixed_dir="./datasets/LFW_multilevel/test_mixed"
mkdir -p "${test_mixed_dir}/low"
mkdir -p "${test_mixed_dir}/high"

# Create symlinks for each person in mixed set
for person_dir in ./datasets/LFW_multilevel/test_easy/high/*/; do
    person_name=$(basename "$person_dir")
    mkdir -p "${test_mixed_dir}/high/${person_name}"
    mkdir -p "${test_mixed_dir}/low/${person_name}"
    
    # Link easy level
    for img in ./datasets/LFW_multilevel/test_easy/high/"${person_name}"/*; do
        if [ -f "$img" ]; then
            filename=$(basename "$img")
            ln -sf "$(realpath "$img")" "${test_mixed_dir}/high/${person_name}/${filename%.png}_easy.png"
            img_low="./datasets/LFW_multilevel/test_easy/low/${person_name}/$(basename "$img")"
            ln -sf "$(realpath "$img_low")" "${test_mixed_dir}/low/${person_name}/${filename%.png}_easy.png"
        fi
    done
    
    # Link medium level
    for img in ./datasets/LFW_multilevel/test_medium/high/"${person_name}"/*; do
        if [ -f "$img" ]; then
            filename=$(basename "$img")
            ln -sf "$(realpath "$img")" "${test_mixed_dir}/high/${person_name}/${filename%.png}_medium.png"
            img_low="./datasets/LFW_multilevel/test_medium/low/${person_name}/$(basename "$img")"
            ln -sf "$(realpath "$img_low")" "${test_mixed_dir}/low/${person_name}/${filename%.png}_medium.png"
        fi
    done
    
    # Link hard level
    for img in ./datasets/LFW_multilevel/test_hard/high/"${person_name}"/*; do
        if [ -f "$img" ]; then
            filename=$(basename "$img")
            ln -sf "$(realpath "$img")" "${test_mixed_dir}/high/${person_name}/${filename%.png}_hard.png"
            img_low="./datasets/LFW_multilevel/test_hard/low/${person_name}/$(basename "$img")"
            ln -sf "$(realpath "$img_low")" "${test_mixed_dir}/low/${person_name}/${filename%.png}_hard.png"
        fi
    done
done

# Generate pairs.txt for mixed set
pairs_mixed="${test_mixed_dir}/pairs.txt"
if [ -f "$pairs_mixed" ]; then
    echo -e "${YELLOW}pairs.txt already exists for test_mixed${NC}"
else
    echo -e "${YELLOW}Generating pairs.txt for test_mixed...${NC}"
    python generate_lfw_pairs.py \
        --test_dir="${test_mixed_dir}" \
        --output="${pairs_mixed}" \
        --num_pairs=1000 \
        --seed=42
fi

echo -e "${GREEN}[$(date)] Test pairs and mixed set generated successfully!${NC}"
echo -e "${GREEN}Output:${NC}"
echo -e "  - test_easy/pairs.txt"
echo -e "  - test_medium/pairs.txt"
echo -e "  - test_hard/pairs.txt"
echo -e "  - test_mixed/ (combined via symlinks)"
echo -e "  - test_mixed/pairs.txt"

# ============================================================
# STEP 3: Train Models
# ============================================================
echo -e "\n${BLUE}[STEP 3] Training Models...${NC}"

# Function to run training and organize weights
run_training() {
    local model_name=$1
    local make_target=$2
    
    echo -e "\n${GREEN}[$(date)] Starting training: ${model_name}${NC}"
    echo -e "${YELLOW}Running: make ${make_target}${NC}\n"
    
    # Run the training
    if TQDM_DISABLE=1 make ${make_target}; then
        echo -e "\n${GREEN}[$(date)] Training completed: ${model_name}${NC}"
        
        # Create model-specific directory
        mkdir -p weights/multilevel/${model_name}
        
        # Move all trained weights to the model-specific folder
        if [ -d "weights/train" ] && [ "$(ls -A weights/train)" ]; then
            echo -e "${GREEN}Moving weights to weights/multilevel/${model_name}/${NC}"
            mv weights/train/* weights/multilevel/${model_name}/
            echo -e "${GREEN}Weights saved in weights/multilevel/${model_name}/${NC}"
        else
            echo -e "${RED}Warning: No weights found in weights/train/${NC}"
        fi
    else
        echo -e "${RED}[$(date)] Training failed: ${model_name}${NC}"
        exit 1
    fi
}

# Run each training in sequence
run_training "baseline" "baseline"
run_training "face_loss3" "face_loss3"
run_training "face_loss5" "face_loss5"

# Also copy loss history to model-specific directories for easy parsing
echo -e "\n${BLUE}[STEP 3.5] Organizing Training Metrics...${NC}"
for model_name in baseline face_loss3 face_loss5; do
    model_dir="weights/multilevel/${model_name}"
    mkdir -p "${model_dir}"
    # Copy latest loss history and metrics to model directory
    latest_loss=$(ls -t ./results/training/loss_history*.json 2>/dev/null | head -1)
    latest_metrics=$(ls -t ./results/training/metrics*.md 2>/dev/null | head -1)
    if [ -n "$latest_loss" ]; then
        cp "$latest_loss" "${model_dir}/loss_history.json"
        echo "  Copied loss history to ${model_dir}/"
    fi
    if [ -n "$latest_metrics" ]; then
        cp "$latest_metrics" "${model_dir}/metrics.md"
        echo "  Copied metrics to ${model_dir}/"
    fi
done
echo -e "${GREEN}Training metrics organized!${NC}"

# ============================================================
# STEP 4: Evaluate Models on All Difficulty Levels
# ============================================================
echo -e "\n${BLUE}[STEP 4] Evaluating Models on All Difficulty Levels...${NC}"

# Create results directory for evaluations
mkdir -p results/multilevel_evaluations

# Function to run evaluation on a specific difficulty level
run_evaluation_on_difficulty() {
    local model_name=$1
    local weights_path=$2
    local difficulty=$3

    local output_dir="results/multilevel_evaluations/${model_name}/${difficulty}"
    mkdir -p "${output_dir}"

    echo -e "\n${GREEN}[$(date)] Evaluating: ${model_name} on ${difficulty}${NC}"
    echo -e "${YELLOW}Model: ${weights_path}${NC}"
    echo -e "${YELLOW}Test set: ./datasets/LFW_multilevel/test_${difficulty}${NC}\n"

    if python eval_face_verification.py \
        --model="${weights_path}" \
        --test_dir=./datasets/LFW_multilevel/test_${difficulty} \
        --pairs_file=./datasets/LFW_multilevel/test_${difficulty}/pairs.txt \
        --output_dir="${output_dir}"; then
        echo -e "${GREEN}[$(date)] Evaluation completed: ${model_name} on ${difficulty}${NC}"
    else
        echo -e "${RED}[$(date)] Evaluation failed: ${model_name} on ${difficulty}${NC}"
        exit 1
    fi
}

# Function to run evaluation on all difficulties for a model
run_all_difficulties() {
    local model_name=$1
    local weights_path=$2

    for difficulty in easy medium hard mixed; do
        run_evaluation_on_difficulty "${model_name}" "${weights_path}" "${difficulty}"
    done
}

# Run evaluations for all models on all difficulties
run_all_difficulties "baseline" "./weights/multilevel/baseline/epoch_70.pth"
run_all_difficulties "face_loss3" "./weights/multilevel/face_loss3/epoch_70.pth"
run_all_difficulties "face_loss5" "./weights/multilevel/face_loss5/epoch_70.pth"

# ============================================================
# STEP 5: Generate Thesis Visualizations
# ============================================================
echo -e "\n${BLUE}[STEP 5] Generating Thesis Visualizations...${NC}"

mkdir -p figures/thesis

# Generate main thesis figures
python plot_thesis_summary.py \
    --results_dir=./results/multilevel_evaluations \
    --output_dir=figures/thesis

# Generate multi-level comparison plots
python plot_multilevel_results.py \
    --results_dir=./results/multilevel_evaluations \
    --output_dir=figures

# Generate verification analysis (ROC curves, distributions)
python plot_verification_analysis.py \
    --results_dir=./results/multilevel_evaluations \
    --output_dir=figures

echo -e "${GREEN}[$(date)] Thesis figures generated!${NC}"

# ============================================================
# Summary
# ============================================================
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}All steps completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"

echo -e "\n${YELLOW}Training Weights:${NC}"
echo -e "  - weights/multilevel/baseline/"
echo -e "  - weights/multilevel/face_loss3/"
echo -e "  - weights/multilevel/face_loss5/"

echo -e "\n${YELLOW}Evaluation Results (all difficulties):${NC}"
echo -e "  - results/multilevel_evaluations/baseline/{easy,medium,hard,mixed}/"
echo -e "  - results/multilevel_evaluations/face_loss3/{easy,medium,hard,mixed}/"
echo -e "  - results/multilevel_evaluations/face_loss5/{easy,medium,hard,mixed}/"

echo -e "\n${YELLOW}Thesis Figures:${NC}"
echo -e "  - figures/thesis/thesis_main_results.png"
echo -e "  - figures/thesis/thesis_comprehensive_summary.png"
echo -e "  - figures/thesis/thesis_tradeoff_analysis.png"
echo -e "  - figures/multilevel_model_comparison.png"
echo -e "  - figures/degradation_curves.png"
echo -e "  - figures/roc_curves_*.png"
echo -e "  - figures/score_distributions_*.png"

echo -e "\n${GREEN}Pipeline complete!${NC}"
