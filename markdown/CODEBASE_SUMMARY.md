# HVI-CIDNet Thesis Codebase Summary

## Quick Reference: HPC Connection

```bash
# SSH to HPC
ssh hpc4090@hpc4090

# Navigate to project directory
cd ~/jamal_fr/
```

**HPC Specs**: RTX 4090 (24GB), i7-14700K (28 cores), 64GB RAM, CUDA 12.4

---

## Project Overview

This is a **thesis research project** focused on **low-light image enhancement with face recognition preservation**. The project builds upon the HVI-CIDNet architecture (CVPR 2025) and extends it with novel discriminative face loss techniques.

### Core Research Problem

Low-light image enhancement typically optimizes for pixel-level reconstruction (PSNR, SSIM), but this can **compress the feature space** and harm downstream face recognition accuracy. This thesis develops methods to preserve **facial identity information** during enhancement.

---

## Project Structure

```
D:\Prog_Stuffs\Thesis\code/
├── net/                          # Core neural network architectures
│   ├── CIDNet.py                 # Main HVI-CIDNet architecture
│   ├── HVI_transform.py          # HVI color space transformations
│   └── LCA.py                    # Lighten Cross-Attention blocks
│
├── loss/                         # Loss function implementations
│   ├── discriminative_face_loss.py   # ⭐ Thesis contribution: Multi-level discriminative loss
│   ├── adaface_model.py          # Pre-trained AdaFace face recognition model
│   └── losses.py                 # Standard losses (L1, SSIM, Perceptual)
│
├── data/                         # Dataset handling and sampling
│   ├── hard_negative_sampler.py  # Smart impostor sampling for training
│   ├── identity_balanced_sampler.py  # Ensures diverse identities per batch
│   ├── options.py                # Training configuration
│   └── [dataset loaders]         # LFW, LaPa-Face, SS-Face datasets
│
├── DiscriminativeMultiLevelFaceLoss/  # Evaluation & analysis pipeline
│   ├── RUN_COMPLETE_ANALYSIS.sh  # Automated evaluation script
│   └── [test protocols]          # Easy, medium, hard test sets
│
├── markdown/                     # ⭐ Documentation and experimental results
│   ├── PLAN.md                   # 1-month research roadmap
│   ├── QUICK_START_FACE_RECOGNITION_LOSS.md
│   ├── DISCRIMINATIVE_EASY_OBSERVATIONS.md
│   ├── DISCRIMINATIVE_0.01_OBSERVATIONS.md
│   ├── HARD_NEGATIVES_README.md
│   └── [other analysis docs]
│
├── train.py                      # Main training script
├── train_with_improvements_PATCH.py  # Enhanced training with improvements
├── eval.py                       # Standard evaluation (PSNR/SSIM/LPIPS)
├── eval_face_verification.py     # Face recognition evaluation
├── generate_lfw_pairs.py         # Generate verification test pairs
├── generate_thesis_results.py    # Automated results generation
└── extended_analysis.py          # Statistical analysis tools
```

---

## Key Contributions

### 1. Discriminative Multi-Level Face Loss

**File**: `loss/discriminative_face_loss.py`

A three-component loss that explicitly preserves face recognition capabilities:

```python
L_FR = L_reconstruction + contrastive_weight * L_contrastive + triplet_weight * L_triplet
```

- **L_reconstruction**: Multi-level feature matching (layers 2, 3, 4, fc of AdaFace)
- **L_contrastive**: InfoNCE-style loss pushes impostor pairs apart, pulls genuine together
- **L_triplet**: Margin enforcement for better feature space separation

**Hyperparameters**:
- Feature layers: `['layer2', 'layer3', 'layer4', 'fc']`
- Layer weights: `[0.2, 0.4, 0.8, 1.0]`
- Contrastive margin: 0.4, Triplet margin: 0.2
- Temperature: 0.07

### 2. Hard Negative Mining

**File**: `data/hard_negative_sampler.py`

Replaces random circular shift with smart impostor sampling:

- Maintains memory bank of identity features
- Samples "challenging but solvable" impostor pairs
- Expected 20-40% improvement in convergence and final performance

### 3. Identity-Balanced Sampling

**File**: `data/identity_balanced_sampler.py`

Ensures diverse identities in each batch:
- Each batch contains N/2 different identities
- Each identity appears K times (default: 2)
- Prevents inefficient all-same or all-different batches

---

## Recent Experimental Findings

### Easy Test Set Results (1000 genuine + 1000 impostor pairs)

| Configuration | Genuine Sim | Impostor Sim | EER (%) | TAR@FAR=1% |
|--------------|-------------|--------------|---------|------------|
| baseline (no face loss) | 0.9855 | 0.6314 | 0.85 | 99.20% |
| FR_weight=0.3 | 0.9811 | 0.5720 | 0.85 | 99.30% |
| **FR_weight=0.5** | **0.9863** | **0.6262** | **0.35** | **99.60%** |

**Key Insight**: FR_weight=0.5 achieves the best EER (0.35%) with 59% improvement over baseline's 0.85%.

### 0.01 Dark Level Results

**Finding**: All models achieve perfect ceiling performance (EER=0.00%) at extreme darkness levels.

**Implication**: Discriminative loss provides no benefit when baseline already achieves perfect performance. Evaluation protocols must be challenging enough to measure improvements.

---

## Important Configuration Files

### Training Configuration: `data/options.py`

Key hyperparameters:
- `FR_weight`: Face recognition loss weight (0.3-0.5 recommended)
- `D_weight`: SSIM weight (1.5 recommended)
- `E_weight`: Edge loss weight (50.0)
- `P_weight`: Perceptual loss weight (0.01)
- `HVI_weight`: HVI color space loss weight (1.0)

### Datasets Supported

**Paired (with ground truth)**:
- LOL-v1/v2, Sony Total Dark, SICE, FiveK, LOL-Blur

**Unpaired (for generalization testing)**:
- DICM, LIME, MEF, NPE, VV

**Face-focused**:
- LFW (with synthetic low-light), LaPa-Face, SS-Face

---

## Quick Start Commands

### Training

```bash
# Train with discriminative face loss
python train.py \
    --lfw \
    --data_train_lfw=./datasets/LFW_lowlight/train \
    --use_face_loss \
    --FR_weight=0.5 \
    --FR_model_path=./weights/adaface/adaface_ir50_webface4m.ckpt \
    --nEpochs=100
```

### Evaluation

```bash
# Face verification evaluation
python eval_face_verification.py \
    --model=./weights/train/epoch_100.pth \
    --test_dir=./datasets/LFW_lowlight/test \
    --pairs_file=pairs.txt \
    --face_weights=./weights/adaface/adaface_ir50_webface4m.ckpt
```

### Generate Test Pairs

```bash
python generate_lfw_pairs.py \
    --test_dir=./datasets/LFW_lowlight/test \
    --num_pairs=1000 \
    --output=pairs.txt
```

---

## Pre-trained Weights Location

Base HVI-CIDNet weights (from CVPR 2025 paper):
- `./weights/LOLv2_real/best_PSNR.pth` - Best PSNR on LOLv2 real
- `./weights/LOLv2_real/best_SSIM.pth` - Best SSIM on LOLv2 real

AdaFace face recognition weights:
- `./weights/adaface/adaface_ir50_webface4m.ckpt` (download from AdaFace repo)

---

## Git Status

Current branch: `master`
Recent commits:
- `26e2215` Save local configuration and scripts
- `8dd7b0d` Latest
- `f822369` Implement hard negative mining and identity-balanced sampling
- `d406f30` Add comprehensive analysis of discriminative face loss at 0.01 dark level
- `f6f427a` Ran 0.01 dark

---

## Current Research Focus

1. Understanding when discriminative loss helps vs. when it provides no benefit
2. Improving training efficiency with hard negative mining
3. Extending to real-world applications with cross-dataset generalization
4. Automated evaluation pipelines for reproducible research

---

## Key Papers Referenced

- **HVI-CIDNet** (CVPR 2025): Base low-light enhancement architecture
- **Low-FaceNet** (2024): Face recognition-driven low-light enhancement
- **Beyond Image SR for Recognition** (CVPR 2024): Task-driven perceptual loss
- **AdaFace**: Adaptive margin for face recognition

---

## Environment

### Development Environment (Local)
- Platform: Windows (MSYS_NT-10.0-26200)
- Python: 3.7+
- PyTorch: 1.13.1+ with CUDA support
- GPU: NVIDIA with 8GB+ VRAM recommended

### HPC Training Environment (Remote)
- **Location**: `~/jamal_fr/` on `hpc4090`
- **OS**: Ubuntu 24.04.2 LTS x86_64
- **Shell**: bash 5.2.21
- **CPU**: Intel i7-14700K (28 cores) @ 5.500 GHz
- **GPU**: NVIDIA GeForce RTX 4090 (24GB VRAM)
- **RAM**: 64GB (64066MiB total)
- **CUDA Version**: 12.4
- **NVIDIA Driver**: 550.163.01

---

*Generated: 2026-01-22*
