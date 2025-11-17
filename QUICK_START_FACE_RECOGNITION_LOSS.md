# Quick Start Guide: Face Recognition Perceptual Loss

This guide provides step-by-step instructions for using the Face Recognition Perceptual Loss feature in HVI-CIDNet for your thesis research on low-light face enhancement.

## Table of Contents

1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Benchmarking](#benchmarking)
6. [Troubleshooting](#troubleshooting)

---

## 1. Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.8+ with CUDA support
- NVIDIA GPU with 8GB+ VRAM (recommended)

### Setup Environment

```bash
# Clone the repository
cd HVI-CIDNet

# Install dependencies
pip install -r requirements.txt

# Install additional dependencies for face recognition loss
pip install facenet-pytorch  # For face recognition models
pip install opencv-python    # For image processing
pip install scikit-image     # For metrics
```

### Download Pre-trained AdaFace Weights (Optional but Recommended)

For best results, download pre-trained AdaFace weights:

```bash
# Create directory for pretrained weights
mkdir -p pretrained/adaface

# Download AdaFace IR-50 weights (WebFace4M)
# Visit: https://github.com/mk-minchul/AdaFace
# Download: adaface_ir50_webface4m.ckpt
# Place in: ./weights/adaface/adaface_ir50_webface4m.ckpt
```

---

## 2. Dataset Preparation

### Option A: LFW Dataset with Synthetic Low-Light (Recommended for Quick Start)

This is the fastest way to get started and perfect for your thesis timeline.

#### Step 1: Download and Prepare LFW Dataset

```bash
# Download LFW and create synthetic low-light versions
python prepare_lfw_dataset.py --download

# This will:
# 1. Download LFW dataset (~170MB)
# 2. Generate synthetic low-light images
# 3. Split into train/val/test (70%/15%/15%)
# 4. Save to ./datasets/LFW_lowlight/
```

#### Step 2: Verify Dataset

```bash
# Check dataset structure
ls -R datasets/LFW_lowlight/

# Expected structure:
# datasets/LFW_lowlight/
# ├── train/
# │   ├── low/   (synthetic low-light faces)
# │   └── high/  (ground truth faces)
# ├── val/
# │   ├── low/
# │   └── high/
# └── test/
#     ├── low/
#     └── high/

# Test dataset loader
python data/lfw_dataset.py
```

#### Step 3: Quick Test with Limited Data (Optional)

For rapid prototyping, create a smaller dataset:

```bash
# Create small test dataset (1000 images)
python prepare_lfw_dataset.py --max_images 1000 --output_dir ./datasets/LFW_small
```

### Option B: Use Existing Datasets (LOL, LOL-v2, etc.)

You can still use the face recognition loss with existing low-light datasets:

```bash
# For LOL-v1 dataset
# Just enable face recognition loss in training args
# (see Training section below)
```

---

## 3. Training

### Important: Command Syntax

**Note:** Dataset and loss flags use `action='store_true'`, so use the flag name without `=True`:

```bash
# ✅ CORRECT:
--lfw --use_face_loss

# ❌ WRONG (will cause errors):
--lfw=True --use_face_loss=True
```

### Baseline Training (Without Face Recognition Loss)

First, establish a baseline by training without the face recognition loss:

```bash
# Train baseline on LFW (from scratch)
python train.py \
    --lfw \
    --data_train_lfw=./datasets/LFW_lowlight/train \
    --data_val_lfw=./datasets/LFW_lowlight/val \
    --data_valgt_lfw=./datasets/LFW_lowlight/val/high \
    --batchSize=8 \
    --cropSize=256 \
    --nEpochs=100 \
    --lr=0.0001 \
    --L1_weight=1.0 \
    --D_weight=0.5 \
    --E_weight=50.0 \
    --P_weight=0.01

# Training will save checkpoints to: ./weights/train/
# Validation results will be saved to: ./results/lfw/
```

### Training with Face Recognition Loss (Your Thesis Contribution!)

Now train with the face recognition perceptual loss:

```bash
# Train with Face Recognition Loss from scratch
python train.py \
    --lfw \
    --data_train_lfw=./datasets/LFW_lowlight/train \
    --data_val_lfw=./datasets/LFW_lowlight/val \
    --data_valgt_lfw=./datasets/LFW_lowlight/val/high \
    --batchSize=8 \
    --cropSize=256 \
    --nEpochs=100 \
    --lr=0.0001 \
    --L1_weight=1.0 \
    --D_weight=0.5 \
    --E_weight=50.0 \
    --P_weight=0.01 \
    --use_face_loss \
    --FR_weight=0.5 \
    --FR_model_arch=ir_50 \
    --FR_model_path=./weights/adaface/adaface_ir50_webface4m.ckpt \
    --FR_feature_distance=mse

# Note: If you don't have AdaFace weights, omit --FR_model_path
# The model will use randomly initialized weights (not recommended)
```

### Fine-tuning from Pretrained CIDNet Weights (Recommended!)

**Best approach for thesis:** Start from a strong baseline and fine-tune with FR loss:

```bash
# Fine-tune LOLv2 best_PSNR model with Face Recognition Loss on LFW
python train.py \
    --lfw \
    --data_train_lfw=./datasets/LFW_lowlight/train \
    --data_val_lfw=./datasets/LFW_lowlight/val \
    --data_valgt_lfw=./datasets/LFW_lowlight/val/high \
    --pretrained_model=./weights/LOLv2_real/best_PSNR.pth \
    --nEpochs=50 \
    --lr=0.00001 \
    --use_face_loss \
    --FR_weight=0.5 \
    --FR_model_path=./weights/adaface/adaface_ir50_webface4m.ckpt

# Benefits:
# - Faster convergence (already trained on low-light)
# - Better baseline performance
# - More robust results for thesis
# - Use lower learning rate (0.00001) and fewer epochs (50)
```

Available pretrained CIDNet models:
- `./weights/LOLv2_real/best_PSNR.pth` - Best PSNR on LOLv2 real
- `./weights/LOLv2_real/best_SSIM.pth` - Best SSIM on LOLv2 real
- `./weights/LOLv1/w_perc.pth` - LOLv1 with perceptual loss
- `./weights/SICE.pth` - SICE dataset
- `./weights/LOL-Blur.pth` - LOL with blur

### Recommended Loss Weight Combinations to Try

For your ablation study, try these configurations:

```bash
# Configuration 1: Low FR weight (conservative)
--FR_weight=0.3

# Configuration 2: Medium FR weight (recommended starting point)
--FR_weight=0.5

# Configuration 3: High FR weight (aggressive)
--FR_weight=1.0

# Configuration 4: With cosine distance
--FR_weight=0.5 --FR_feature_distance=cosine

# Configuration 5: With L1 distance
--FR_weight=0.5 --FR_feature_distance=l1
```

### Quick Training (For Testing)

Test your setup quickly with reduced epochs and small dataset:

```bash
# Quick test (10 epochs) - Fine-tune from pretrained
python train.py \
    --lfw \
    --data_train_lfw=./datasets/LFW_small/train \
    --data_val_lfw=./datasets/LFW_small/val \
    --data_valgt_lfw=./datasets/LFW_small/val/high \
    --pretrained_model=./weights/LOLv2_real/best_PSNR.pth \
    --nEpochs=10 \
    --use_face_loss \
    --FR_weight=0.5 \
    --FR_model_path=./weights/adaface/adaface_ir50_webface4m.ckpt

# Note: LFW images are typically small (125x94), so they are automatically
# resized to 288x288 before cropping to 256x256 during training
```

### Monitoring Training

```bash
# Training progress is displayed in the terminal
# Checkpoints are saved every 10 epochs (default) to: ./weights/train/
# Validation results are saved to: ./results/

# View training metrics
cat ./results/training/metrics*.md
```

---

## 4. Evaluation

### Standard Evaluation (PSNR, SSIM, LPIPS)

Evaluate image quality metrics:

```bash
# Evaluate a trained model
python eval.py \
    --model=./weights/train/epoch_100.pth \
    --test_dir=./datasets/LFW_lowlight/test/low \
    --output_dir=./results/test_enhanced

# Compute metrics
python measure.py \
    --pred_dir=./results/test_enhanced/*.png \
    --gt_dir=./datasets/LFW_lowlight/test/high
```

### Face Verification Evaluation (Key Thesis Metric!)

Evaluate face verification accuracy to demonstrate the benefit of your approach:

```bash
# Full face verification evaluation
python eval_face_verification.py \
    --model=./weights/train/epoch_100.pth \
    --test_dir=./datasets/LFW_lowlight/test \
    --face_model=ir_50 \
    --face_weights=./weights/adaface/adaface_ir50_webface4m.ckpt \
    --output_dir=./results/face_verification

# Quick evaluation (100 pairs only)
python eval_face_verification.py \
    --model=./weights/train/epoch_100.pth \
    --test_dir=./datasets/LFW_lowlight/test \
    --max_pairs=100

# Results will show:
# - Face similarity improvement (enhanced vs. low-light)
# - Verification accuracy improvement
# - PSNR/SSIM for image quality
```

### Compare Baseline vs. Face Recognition Loss

Create comparison table for your thesis:

```bash
# Evaluate baseline model
python eval_face_verification.py \
    --model=./weights/baseline/epoch_100.pth \
    --test_dir=./datasets/LFW_lowlight/test \
    --output_dir=./results/baseline_face_verification

# Evaluate model with FR loss
python eval_face_verification.py \
    --model=./weights/with_fr_loss/epoch_100.pth \
    --test_dir=./datasets/LFW_lowlight/test \
    --output_dir=./results/fr_loss_face_verification

# Compare results
diff ./results/baseline_face_verification/face_verification_results.txt \
     ./results/fr_loss_face_verification/face_verification_results.txt
```

---

## 5. Benchmarking

### Create Results Table for Thesis

Run comprehensive benchmarking for your thesis defense:

```bash
#!/bin/bash
# benchmark.sh - Run all evaluations for thesis

MODELS=(
    "baseline:./weights/baseline/epoch_100.pth"
    "fr_loss_0.3:./weights/fr_0.3/epoch_100.pth"
    "fr_loss_0.5:./weights/fr_0.5/epoch_100.pth"
    "fr_loss_1.0:./weights/fr_1.0/epoch_100.pth"
)

for model_info in "${MODELS[@]}"; do
    NAME="${model_info%%:*}"
    MODEL="${model_info##*:}"

    echo "Evaluating $NAME..."

    # Standard metrics
    python eval.py --model=$MODEL --test_dir=./datasets/LFW_lowlight/test/low

    # Face verification
    python eval_face_verification.py \
        --model=$MODEL \
        --test_dir=./datasets/LFW_lowlight/test \
        --output_dir=./results/benchmark/$NAME
done

echo "Benchmarking complete! Results in ./results/benchmark/"
```

### Expected Results Timeline

Based on the "Minimum Viable Research" approach:

**Week 1** (Dataset + Baseline):
- Day 1-2: Prepare LFW dataset
- Day 3-4: Train baseline model
- Day 5-7: Establish baseline metrics

**Week 2** (Core Contribution):
- Day 8-10: Implement and test FR loss (DONE! ✅)
- Day 11-14: Train with FR loss

**Week 3-4** (Experiments):
- Week 3: Ablation studies (different weights)
- Week 4: Comprehensive evaluation

**Week 5-6** (Analysis):
- Statistical significance testing
- Create results tables and figures

---

## 6. Troubleshooting

### Common Issues and Solutions

#### Issue 1: CUDA Out of Memory

```bash
# Solution: Reduce batch size
--batchSize=4  # or even 2

# Or reduce crop size
--cropSize=128
```

#### Issue 2: Face Recognition Model Not Loading

```bash
# If AdaFace weights fail to load, the model automatically falls back to ResNet50

# Check if weights file exists:
ls -lh ./weights/adaface/adaface_ir50_webface4m.ckpt

# If not, train without pre-trained weights (less accurate but works):
python train.py ... --FR_model_path=None
```

#### Issue 3: Dataset Not Found

```bash
# Verify dataset structure
ls -R datasets/LFW_lowlight/

# Regenerate if needed
python prepare_lfw_dataset.py --download
```

#### Issue 4: Training Too Slow

```bash
# Use smaller dataset for quick iteration
python prepare_lfw_dataset.py --max_images=1000

# Reduce epochs for testing
--nEpochs=10

# Use fewer workers
--threads=4
```

#### Issue 5: Poor Results

Check these settings:

1. **Learning rate too high?**
   ```bash
   --lr=0.00005  # Try lower LR
   ```

2. **FR loss weight too high?**
   ```bash
   --FR_weight=0.3  # Try lower weight
   ```

3. **Need more training epochs?**
   ```bash
   --nEpochs=200  # Try more epochs
   ```

### Getting Help

If you encounter issues:

1. Check the error message carefully
2. Verify dataset structure with `ls -R datasets/`
3. Test with small dataset first (`--max_images=100`)
4. Check GPU memory with `nvidia-smi`

---

## Quick Reference: Essential Commands

```bash
# 1. Prepare dataset
python prepare_lfw_dataset.py --download

# 2. Train baseline (from scratch)
python train.py --lfw --nEpochs=100

# 3. Train with FR loss (recommended: fine-tune from pretrained)
python train.py \
    --lfw \
    --pretrained_model=./weights/LOLv2_real/best_PSNR.pth \
    --use_face_loss \
    --FR_weight=0.5 \
    --FR_model_path=./weights/adaface/adaface_ir50_webface4m.ckpt \
    --nEpochs=50 \
    --lr=0.00001

# 4. Evaluate face verification
python eval_face_verification.py \
    --model=./weights/train/epoch_50.pth \
    --test_dir=./datasets/LFW_lowlight/test

# 5. View results
cat ./results/face_verification/face_verification_results.txt
```

### Common Issues and Solutions

#### Issue 1: "unrecognized arguments: --lfw=True"

**Problem:** Using `=True` syntax with boolean flags.

**Solution:** Use flags without `=True`:
```bash
# ❌ Wrong:
--lfw=True --use_face_loss=True

# ✅ Correct:
--lfw --use_face_loss
```

#### Issue 2: "Required crop size (256, 256) is larger than input image size"

**Problem:** LFW images are too small for the crop size.

**Solution:** Images are automatically resized. If you still see this error, ensure you're using the latest code with the `transform_lfw()` function in `data/data.py`.

#### Issue 3: "FileNotFoundError: ./datasets/LOLdataset/eval15/low"

**Problem:** The model is trying to load LOL dataset instead of LFW.

**Solution:** In older versions, you needed to explicitly set `--lol_v1=False`. In the latest version, all dataset flags default to `False`, so just use `--lfw`.

#### Issue 4: NumPy 2.x Compatibility Issues

**Problem:** scikit-learn compiled against NumPy 1.x but NumPy 2.x is installed.

**Solution:**
```bash
pip install --force-reinstall "numpy<2.0,>=1.22"
```

#### Issue 5: AdaFace weights not loading

**Problem:** "Using randomly initialized weights" warning appears.

**Solution:** Download AdaFace weights or specify correct path:
```bash
# Download from: https://github.com/mk-minchul/AdaFace/releases
# Then use:
--FR_model_path=./weights/adaface/adaface_ir50_webface4m.ckpt
```

---

## Thesis Defense Preparation

### Key Results to Present

1. **Quantitative Improvements:**
   - Face similarity improvement: X → Y
   - Verification accuracy: +Z%
   - PSNR/SSIM maintained or improved

2. **Ablation Study:**
   - Impact of different FR loss weights
   - Comparison of distance metrics

3. **Qualitative Results:**
   - Visual comparison: low-light → enhanced → GT
   - Face regions better preserved

### Creating Results Table

```python
# example_results_table.py
import pandas as pd

results = {
    'Method': ['Baseline', 'FR Loss (0.3)', 'FR Loss (0.5)', 'FR Loss (1.0)'],
    'PSNR': [23.5, 23.4, 23.6, 23.2],
    'SSIM': [0.856, 0.858, 0.862, 0.855],
    'Face Similarity': [0.642, 0.698, 0.724, 0.731],
    'Verification Acc': [72.3, 79.1, 82.6, 83.2]
}

df = pd.DataFrame(results)
print(df.to_markdown(index=False))
```

---

## Next Steps

1. ✅ Setup environment
2. ✅ Prepare LFW dataset
3. ⏳ Train baseline model (Week 1)
4. ⏳ Train with FR loss (Week 2)
5. ⏳ Run evaluations (Week 3-4)
6. ⏳ Analyze results (Week 5-6)
7. ⏳ Write thesis (Week 7-8)

**Good luck with your thesis! 🎓**

For questions or issues, consult:
- This QUICK_START guide
- Code comments in `loss/losses.py` (FaceRecognitionPerceptualLoss class)
- Evaluation script: `eval_face_verification.py`
