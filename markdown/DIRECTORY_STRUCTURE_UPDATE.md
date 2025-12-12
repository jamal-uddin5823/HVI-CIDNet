# Directory Structure Update - D_weight Variations

## Overview
The project has been updated to handle D_weight variations (0.5, 1.0, 1.5) for all model configurations in the ablation study.

## Weights Directory Structure

```
weights/
└── ablation/
    ├── baseline/
    │   ├── d_0.5/
    │   │   ├── epoch_10.pth
    │   │   ├── epoch_20.pth
    │   │   ├── ...
    │   │   └── epoch_50.pth
    │   ├── d_1/
    │   │   ├── epoch_10.pth
    │   │   ├── ...
    │   │   └── epoch_50.pth
    │   └── d_1.5/
    │       ├── epoch_10.pth
    │       ├── ...
    │       └── epoch_50.pth
    ├── fr_weight_0.3/
    │   ├── d_0.5/
    │   ├── d_1/
    │   └── d_1.5/
    ├── fr_weight_0.5/
    │   ├── d_0.5/
    │   ├── d_1/
    │   └── d_1.5/
    └── fr_weight_1.0/
        ├── d_0.5/
        ├── d_1/
        └── d_1.5/
```

**Total combinations**: 4 configurations × 3 d_weights = **12 model variations**

## Results Directory Structure

```
results/
├── full_evaluation/
│   ├── baseline_d0.5/
│   │   ├── face_verification_results.txt
│   │   ├── evaluation.log
│   │   └── sample_results/
│   ├── baseline_d1/
│   ├── baseline_d1.5/
│   ├── fr_weight_0.3_d0.5/
│   ├── fr_weight_0.3_d1/
│   ├── fr_weight_0.3_d1.5/
│   ├── fr_weight_0.5_d0.5/
│   ├── fr_weight_0.5_d1/
│   ├── fr_weight_0.5_d1.5/
│   ├── fr_weight_1.0_d0.5/
│   ├── fr_weight_1.0_d1/
│   ├── fr_weight_1.0_d1.5/
│   ├── comparison_table.txt          # Comparison of all 12 configurations
│   ├── thesis_results_summary.txt    # Statistical analysis summary
│   └── plots/
│       ├── verification_metrics.png
│       ├── image_quality.png
│       └── tradeoff_analysis.png
│
├── extended_analysis/
│   ├── statistical_significance.txt   # McNemar's & t-test results
│   ├── per_identity_analysis.csv      # Identity-wise performance
│   └── failure_cases.png              # Visualization of failure cases
│
└── ablation/
    ├── baseline/
    │   ├── training_curves.png
    │   └── logs/
    ├── fr_weight_0.3/
    ├── fr_weight_0.5/
    └── fr_weight_1.0/
```

## Updated Scripts

### 1. `run_full_evaluation.sh`
**Changes:**
- Added `D_WEIGHTS=("0.5" "1" "1.5")` array
- Nested loops: iterate over both `MODELS` and `D_WEIGHTS`
- Output directories: `$RESULTS_BASE/${config}_d${d_weight}/`
- Model paths: `./weights/ablation/$config/d_${d_weight}/epoch_$EPOCH.pth`
- Comparison table now shows all 12 configurations

**Usage:**
```bash
bash run_full_evaluation.sh
```

**Output:**
- Evaluates all available model/d_weight combinations
- Creates `comparison_table.txt` with all results
- Individual results in `results/full_evaluation/{config}_d{d_weight}/`

### 2. `generate_thesis_results.py`
**Changes:**
- Dynamically loads all `{config}_d{d_weight}` variations
- Searches for `d_weights = ['0.5', '1', '1.5']`
- Configuration keys: `baseline_d0.5`, `baseline_d1`, `fr_weight_0.5_d1`, etc.
- Plots adapt to number of configurations found

**Usage:**
```bash
python generate_thesis_results.py \
    --results_dir ./results/full_evaluation \
    --output_dir ./results/full_evaluation
```

**Output:**
- `thesis_results_summary.txt` - comprehensive comparison
- `plots/verification_metrics.png` - bar charts for all configs
- `plots/image_quality.png` - PSNR/SSIM comparison
- `plots/tradeoff_analysis.png` - improvement visualization

### 3. `RUN_COMPLETE_ANALYSIS.sh`
**Changes:**
- Updated model search to include `d_weight in 0.5 1 1.5`
- Tracks which d_weight is used for extended analysis
- Shows `baseline: d_weight=X` and `FR=0.5: d_weight=Y` in logs
- Updated example paths to show all 12 combinations

**Usage:**
```bash
bash RUN_COMPLETE_ANALYSIS.sh
```

**Pipeline:**
1. Runs `run_full_evaluation.sh` → 12 model evaluations
2. Runs `generate_thesis_results.py` → comparison tables & plots
3. Runs `extended_analysis.py` → statistical tests (baseline vs FR=0.5)

### 4. `extended_analysis.py`
**Notes:**
- No changes needed - it accepts model paths as arguments
- `RUN_COMPLETE_ANALYSIS.sh` finds the best available models automatically
- Will use first available d_weight (0.5, then 1, then 1.5)

## Naming Convention

### Configuration Keys
Format: `{fr_config}_d{d_weight}`

Examples:
- `baseline_d0.5` - Baseline model with D_weight=0.5
- `baseline_d1` - Baseline model with D_weight=1.0
- `baseline_d1.5` - Baseline model with D_weight=1.5
- `fr_weight_0.3_d1` - FR weight=0.3, D_weight=1.0
- `fr_weight_0.5_d1.5` - FR weight=0.5, D_weight=1.5
- `fr_weight_1.0_d0.5` - FR weight=1.0, D_weight=0.5

### Labels in Plots
Format: `{config} D={d_weight}`

Examples:
- `baseline D=0.5`
- `fr weight 0.5 D=1`
- `fr weight 1.0 D=1.5`

## Migration Notes

### If you have old weights (flat structure)
```bash
# Old structure
weights/ablation/baseline/epoch_50.pth

# New structure (move to d_1/ subdirectory)
mkdir -p weights/ablation/baseline/d_1
mv weights/ablation/baseline/epoch_*.pth weights/ablation/baseline/d_1/
```

### If you have old results
```bash
# Old structure
results/full_evaluation/baseline/face_verification_results.txt

# New structure
mkdir -p results/full_evaluation/baseline_d1
mv results/full_evaluation/baseline/* results/full_evaluation/baseline_d1/
rmdir results/full_evaluation/baseline
```

## Comparison Table Format

The updated comparison table shows:
```
Configuration          | Genuine Sim | Impostor Sim | EER (%)   | TAR@0.1% (%) | TAR@1% (%) | PSNR    | SSIM
-----------------------|-------------|--------------|-----------|--------------|------------|---------|--------
baseline_d0.5          | 0.8234      | 0.3421       | 8.23      | 89.45        | 92.12      | 18.45   | 0.8234
baseline_d1            | 0.8456      | 0.3312       | 7.89      | 90.12        | 92.89      | 19.12   | 0.8456
baseline_d1.5          | 0.8521      | 0.3287       | 7.56      | 90.56        | 93.21      | 19.45   | 0.8521
fr_weight_0.3_d0.5     | 0.8378      | 0.3398       | 7.98      | 89.78        | 92.45      | 18.67   | 0.8345
...
```

## Workflow

1. **Train models** with different d_weights:
   ```bash
   python train.py --fr_weight 0 --D_weight 0.5 --save_dir weights/ablation/baseline/d_0.5
   python train.py --fr_weight 0 --D_weight 1.0 --save_dir weights/ablation/baseline/d_1
   python train.py --fr_weight 0.5 --D_weight 1.5 --save_dir weights/ablation/fr_weight_0.5/d_1.5
   ```

2. **Run full evaluation**:
   ```bash
   bash RUN_COMPLETE_ANALYSIS.sh
   ```

3. **Review results**:
   - Comparison table: `results/full_evaluation/comparison_table.txt`
   - Statistical analysis: `results/full_evaluation/thesis_results_summary.txt`
   - Plots: `results/full_evaluation/plots/*.png`
   - Extended analysis: `results/extended_analysis/`

## Benefits

1. **Comprehensive comparison**: All FR weights × all D weights
2. **Flexible**: Works even if some model combinations are missing
3. **Automatic**: Scripts auto-detect available models
4. **Organized**: Clear naming convention for results
5. **Thesis-ready**: Publication-quality tables and plots

## Questions?

See also:
- `QUICK_START_FACE_RECOGNITION_LOSS.md` - Training guide
- `HPC_EVALUATION_INSTRUCTIONS.md` - Evaluation guide
- `EXTENDED_ANALYSIS_README.md` - Statistical analysis guide
