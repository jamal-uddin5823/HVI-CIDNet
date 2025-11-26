# LaPa-Face Dataset Setup and Training Guide

This guide explains how to use the LaPa-Face dataset for low-light face enhancement with discriminative face recognition loss.

## Dataset Structure

The LaPa-Face dataset contains paired underexposed (low-light) and normal-exposure face images:

```
datasets/LaPa-Face/
├── train/
│   ├── underexposed/   # Low-light/underexposed training images
│   ├── normal/         # Ground truth normal-exposure training images
│   └── seg/            # Face segmentation masks (optional, for future use)
└── test/
    ├── underexposed/   # Low-light test images
    ├── normal/         # Ground truth test images
    └── seg/            # Segmentation masks (optional)
```

## Dataset Setup

### 1. Download Dataset

Download the LaPa-Face dataset:
- Train: https://drive.google.com/file/d/1bmFIy1In-OnTv-Fb1kvsk46hojJrpINb/view?usp=sharing
- Test: https://drive.google.com/file/d/1neJZq1C9HkXCO_eqdPRijiX4jvggF7mF/view?usp=sharing

### 2. Extract Dataset

```bash
cd datasets/
unzip LaPa-Face-train.zip
unzip LaPa-Face-test.zip

# Verify structure
ls -la LaPa-Face/train/
# Should show: underexposed/, normal/, seg/

ls -la LaPa-Face/test/
# Should show: underexposed/, normal/, seg/
```

### 3. Verify Dataset Loader

Test the LaPa-Face dataset loader:

```bash
python -m data.lapaface_dataset
```

Expected output:
```
Testing LaPa-Face Dataset Loader...
======================================================================

[1] Testing Training Dataset...
[LaPa-Face Dataset] Loaded XXXX image pairs from ./datasets/LaPa-Face/train
  ✓ Dataset loaded: XXXX image pairs
  ✓ Sample loaded:
    Underexposed shape: torch.Size([3, 256, 256])
    Normal-exposure shape: torch.Size([3, 256, 256])
    ...

[2] Testing Evaluation Dataset...
[LaPa-Face Eval Dataset] Loaded XXXX image pairs from ./datasets/LaPa-Face/test
  ✓ Eval dataset loaded: XXXX image pairs
  ...
```

## Training on LaPa-Face

### Quick Start: Validation Run (10 epochs)

Test that the discriminative face loss works correctly on LaPa-Face:

```bash
cd DiscriminativeMultiLevelFaceLoss/
./validation_lapaface.sh
```

This will:
- Train for 10 epochs
- Use discriminative face loss with FR_weight=0.5
- Save results to `./results/lapaface/`
- Save checkpoints to `./weights/train/`

### Full Training: Comparison Study (50 epochs)

Train 3 models (baseline, FR=0.3, FR=0.5) for comparison:

```bash
cd DiscriminativeMultiLevelFaceLoss/
./comparison_lapaface.sh
```

This will train:
1. **Baseline** (D_weight=1.5, no face loss)
2. **Discriminative FR=0.3** (D_weight=1.5, FR_weight=0.3)
3. **Discriminative FR=0.5** (D_weight=1.5, FR_weight=0.5)

Results will be saved to:
- `./weights/lapaface_baseline_d1.5_reference/`
- `./weights/lapaface_discriminative_fr0.3_d1.5/`
- `./weights/lapaface_discriminative_fr0.5_d1.5/`

### Hyperparameter Tuning

Systematically tune contrastive margin and weight:

```bash
cd DiscriminativeMultiLevelFaceLoss/
./hyperparameter_tuning_lapaface.sh
```

This will:
- Test contrastive margins: 0.3, 0.4, 0.5
- Test contrastive weights: 0.5, 1.0, 1.5
- Save results to `./logs/lapaface_hyperparameter_tuning/tuning_results.txt`

## Manual Training

Train a single model with custom parameters:

```bash
python train.py \
    --lapaface \
    --data_train_lapaface=./datasets/LaPa-Face/train \
    --data_val_lapaface=./datasets/LaPa-Face/test \
    --data_valgt_lapaface=./datasets/LaPa-Face/test/normal/ \
    --pretrained_model=./weights/LOLv2_real/best_PSNR.pth \
    --use_face_loss \
    --FR_weight=0.5 \
    --FR_model_path=./weights/adaface/adaface_ir50_webface4m.ckpt \
    --D_weight=1.5 \
    --nEpochs=50 \
    --batchSize=8 \
    --contrastive_margin=0.4 \
    --contrastive_weight=1.0 \
    --triplet_margin=0.2 \
    --triplet_weight=0.5 \
    --snapshots=5
```

### Baseline Training (No Face Loss)

Train without discriminative face loss:

```bash
python train.py \
    --lapaface \
    --data_train_lapaface=./datasets/LaPa-Face/train \
    --data_val_lapaface=./datasets/LaPa-Face/test \
    --data_valgt_lapaface=./datasets/LaPa-Face/test/normal/ \
    --pretrained_model=./weights/LOLv2_real/best_PSNR.pth \
    --D_weight=1.5 \
    --nEpochs=50 \
    --batchSize=8 \
    --snapshots=5
```

## Configuration Parameters

### Dataset Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lapaface` | - | Enable LaPa-Face dataset |
| `--data_train_lapaface` | `./datasets/LaPa-Face/train` | Training data directory |
| `--data_val_lapaface` | `./datasets/LaPa-Face/test` | Validation data directory |
| `--data_valgt_lapaface` | `./datasets/LaPa-Face/test/normal/` | Ground truth validation directory |

### Discriminative Face Loss Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_face_loss` | - | Enable discriminative face loss |
| `--FR_weight` | 0.5 | Face recognition loss weight |
| `--FR_model_path` | None | Path to AdaFace weights |
| `--FR_model_arch` | ir_50 | AdaFace architecture (ir_50 or ir_101) |
| `--contrastive_margin` | 0.4 | Margin for contrastive loss |
| `--contrastive_weight` | 1.0 | Weight for contrastive loss |
| `--triplet_margin` | 0.2 | Margin for triplet loss |
| `--triplet_weight` | 0.5 | Weight for triplet loss |
| `--face_temperature` | 0.07 | Temperature for contrastive learning |

### Loss Weight Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--D_weight` | 0.5 | SSIM loss weight |
| `--L1_weight` | 1.0 | L1 loss weight |
| `--E_weight` | 50.0 | Edge loss weight |
| `--P_weight` | 0.01 | Perceptual (VGG) loss weight |
| `--HVI_weight` | 1.0 | HVI color space loss weight |

## Expected Results

After training on LaPa-Face, you should observe:

1. **Image Quality Metrics** (on test set):
   - PSNR: ~32-33 dB
   - SSIM: ~0.988-0.990

2. **Face Verification Metrics** (on face pairs protocol):
   - EER: < 1%
   - TAR@FAR=1%: > 99%
   - Genuine similarity: > 0.98

3. **Discriminative Loss Effects**:
   - FR_weight=0.5 typically achieves best overall performance
   - Slight PSNR trade-off (0.5-1.0 dB) for improved face recognition
   - Significantly better discrimination between genuine and impostor pairs

## Evaluation

### Face Verification Evaluation

Evaluate face recognition performance on enhanced images:

```bash
python eval_face_verification.py \
    --model=./weights/lapaface_discriminative_fr0.5_d1.5/epoch_50.pth \
    --test_dir=./datasets/LaPa-Face/test \
    --pairs_file=./pairs.txt \
    --face_weights=./weights/adaface/adaface_ir50_webface4m.ckpt \
    --output_dir=./results/lapaface_evaluation
```

### Image Quality Metrics

Compute PSNR/SSIM on enhanced images:

```bash
python measure.py \
    --enhanced_dir=./results/lapaface/ \
    --gt_dir=./datasets/LaPa-Face/test/normal/
```

## Differences from LFW Dataset

| Aspect | LFW | LaPa-Face |
|--------|-----|-----------|
| **Source** | Labeled Faces in the Wild | Large-scale face parsing |
| **Low-light** | Synthetic degradation | Real underexposed captures |
| **Pairs** | Curated identity pairs | Requires pair generation |
| **Segmentation** | No | Yes (face parsing masks) |
| **Challenge** | Pose/expression variation | Exposure/lighting variation |
| **Size** | ~13K images | Larger (train + test) |

### Key Advantages of LaPa-Face

1. **Real underexposure**: Not synthetic, more realistic low-light conditions
2. **Segmentation masks**: Enables face-aware losses and analysis
3. **Larger scale**: More training data for better generalization
4. **Exposure diversity**: Wide range of underexposure levels

## Troubleshooting

### Dataset Not Found

```
FileNotFoundError: Underexposed directory not found
```

**Solution**: Verify dataset paths and ensure zip files are extracted:
```bash
ls datasets/LaPa-Face/train/underexposed/
ls datasets/LaPa-Face/train/normal/
```

### No Matching Pairs

```
ValueError: No matching filenames found between underexposed and normal folders!
```

**Solution**: Check that image filenames match between `underexposed/` and `normal/`:
```bash
# Compare file counts
ls datasets/LaPa-Face/train/underexposed/ | wc -l
ls datasets/LaPa-Face/train/normal/ | wc -l

# Check for naming mismatches
diff <(ls datasets/LaPa-Face/train/underexposed/) <(ls datasets/LaPa-Face/train/normal/)
```

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size:
```bash
python train.py --lapaface --batchSize=4 ...  # Instead of 8
```

### AdaFace Weights Not Found

```
FileNotFoundError: ./weights/adaface/adaface_ir50_webface4m.ckpt not found
```

**Solution**: Download AdaFace pretrained weights:
```bash
mkdir -p ./weights/adaface/
# Download from AdaFace GitHub repository
# Place adaface_ir50_webface4m.ckpt in ./weights/adaface/
```

## Next Steps

1. **Train baseline and discriminative models** using `comparison_lapaface.sh`
2. **Evaluate face verification performance** on test pairs
3. **Compare results** with LFW experiments
4. **Analyze per-identity improvements** using extended analysis
5. **Document findings** in observations markdown

## References

- **LaPa-Face Dataset**: [Citation needed]
- **Discriminative Multi-Level Face Loss**: See `DISCRIMINATIVE_EASY_OBSERVATIONS.md`
- **AdaFace**: Kim et al., "AdaFace: Quality Adaptive Margin for Face Recognition", CVPR 2022
- **CIDNet**: Base architecture for low-light enhancement

---

**Document Version**: 1.0
**Last Updated**: 2025-11-26
**Status**: Active Development
