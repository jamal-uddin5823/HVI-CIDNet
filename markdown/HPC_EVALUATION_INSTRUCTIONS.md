# Complete Evaluation Instructions for HPC

## ✅ Status Check

Based on your HPC setup, you have:
- ✅ Dataset ready: `datasets/LFW_lowlight/` (train/val/test)
- ✅ All 4 trained models in `weights/ablation/`
- ✅ 2,000 pairs file: `pairs.txt`
- ✅ Evaluation scripts ready

**You are ready to run the full evaluation!**

---

## 🚀 Quick Start (Recommended)

On your HPC, run this single command:

```bash
cd ~/jamal_fr  # Or wherever your project is
bash RUN_COMPLETE_ANALYSIS.sh
```

This will:
1. Evaluate all 4 models on 2,000 pairs (~1-2 hours)
2. Generate statistical comparisons
3. Create publication-ready plots
4. Save everything to `./results/`

**Then go get coffee ☕ and check back in 1-2 hours.**

---

## 📋 What Gets Evaluated

### Models (all at epoch 50):
1. **Baseline** - No face recognition loss
2. **FR weight=0.3** - Conservative FR loss
3. **FR weight=0.5** - Moderate FR loss (typically best)
4. **FR weight=1.0** - Aggressive FR loss

### Metrics Computed:
- **Genuine Pair Similarity** (same person) - Higher is better
- **Impostor Pair Similarity** (different people) - Lower is better
- **EER** (Equal Error Rate) - Lower is better
- **TAR@FAR=0.1%** and **TAR@FAR=1%** - Higher is better
- **PSNR** and **SSIM** (image quality)

### Statistical Tests:
- Significance testing (p-values)
- Confidence intervals
- Best configuration identification

---

## 📊 Expected Timeline

| Step | Duration | What Happens |
|------|----------|--------------|
| Baseline evaluation | 15-30 min | 2000 pairs processed |
| FR=0.3 evaluation | 15-30 min | 2000 pairs processed |
| FR=0.5 evaluation | 15-30 min | 2000 pairs processed |
| FR=1.0 evaluation | 15-30 min | 2000 pairs processed |
| Statistical analysis | 1-2 min | Tables and plots generated |
| **TOTAL** | **1-2 hours** | All results ready |

---

## 📂 Output Files

After completion, you'll have:

```
results/
├── full_evaluation/
│   ├── baseline/
│   │   └── face_verification_results.txt
│   ├── fr_weight_0.3/
│   │   └── face_verification_results.txt
│   ├── fr_weight_0.5/
│   │   └── face_verification_results.txt
│   ├── fr_weight_1.0/
│   │   └── face_verification_results.txt
│   ├── comparison_table.txt            ← Read this first!
│   ├── thesis_results_summary.txt      ← Use for thesis!
│   └── plots/
│       ├── verification_metrics.png     ← Thesis figure
│       ├── image_quality.png            ← Thesis figure
│       └── tradeoff_analysis.png        ← Thesis figure
└── extended_analysis/                   (optional)
    ├── statistical_significance.txt
    ├── per_identity_analysis.csv
    └── failure_cases.png
```

---

## 🔍 How to Check Progress

While it's running, you can monitor:

```bash
# Check current status
tail -f logs/complete_analysis_*.log

# See which model is being evaluated
ps aux | grep eval_face_verification

# Check GPU usage
nvidia-smi
```

---

## 📖 Reading the Results

### 1. Quick Overview
```bash
cat results/full_evaluation/comparison_table.txt
```

Look for:
- Which FR weight has **lowest EER**
- Which has **highest TAR@FAR=1%**
- Improvements over baseline

### 2. Detailed Results
```bash
cat results/full_evaluation/thesis_results_summary.txt
```

This includes:
- Complete comparison tables
- Statistical significance tests
- Key findings summary
- Recommended configuration

### 3. Individual Model Results
```bash
# Detailed results for specific model
cat results/full_evaluation/fr_weight_0.5/face_verification_results.txt
```

---

## ✅ Validation Checklist

After evaluation completes, verify:

- [ ] All 4 result files exist
- [ ] Each processed 1000 genuine + 1000 impostor pairs
- [ ] Genuine similarity > Impostor similarity (for all models)
- [ ] EER values are reasonable (5-30%)
- [ ] FR models show improvement over baseline
- [ ] Statistical significance p < 0.05 for best model

---

## 🎯 What to Report in Your Thesis

### Quantitative Results (from comparison table):

**Example text:**

> "We evaluated four configurations on 2,000 LFW face pairs (1,000 genuine + 1,000 impostor).
> The baseline model (without face recognition loss) achieved an EER of XX.X% and TAR@FAR=1%
> of XX.X%. Adding face recognition perceptual loss with weight 0.5 improved performance to
> EER of XX.X% (ΔX.X%) and TAR@FAR=1% of XX.X% (ΔX.X%). This improvement was statistically
> significant (p < 0.05), demonstrating that face recognition loss effectively preserves
> identity information during low-light enhancement."

### Figures to Include:

1. **Figure 1**: `verification_metrics.png`
   - Caption: "Face verification performance across different FR loss weights"

2. **Figure 2**: `image_quality.png`
   - Caption: "Image quality (PSNR/SSIM) comparison"

3. **Figure 3**: `tradeoff_analysis.png`
   - Caption: "Trade-off between EER and TAR@FAR=1%"

### Key Talking Points:

1. **Identity Preservation**: Genuine pair similarity improvement
2. **Discrimination**: Impostor similarity should remain low/similar
3. **Verification Accuracy**: EER reduction, TAR@FAR improvement
4. **Image Quality**: PSNR/SSIM should not degrade significantly
5. **Optimal Weight**: Which FR weight gives best balance

---

## 🔧 Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in evaluation (edit `eval_face_verification.py`)

### Issue: "AdaFace weights not found"
**Solution**:
```bash
# Download AdaFace weights
mkdir -p weights/adaface
cd weights/adaface
wget https://github.com/mk-minchul/AdaFace/releases/download/v1.0/adaface_ir50_webface4m.ckpt
```

### Issue: Evaluation is very slow
**Check**:
- GPU is being used (`nvidia-smi`)
- Not running on CPU
- Dataset is on fast storage (not network drive)

### Issue: Results look wrong (too good or too bad)
**Verify**:
1. Check that 2,000 pairs were processed (not 10!)
2. Verify genuine + impostor pairs both evaluated
3. Check for error messages in log files

---

## 🚨 Common Mistakes to Avoid

1. ❌ **Running on only 10 pairs** (like the test)
   - ✅ Script uses ALL 2,000 pairs from pairs.txt

2. ❌ **Using legacy evaluation mode** (same-person only)
   - ✅ Script uses pairs-based protocol with genuine + impostor

3. ❌ **Not checking statistical significance**
   - ✅ Script automatically computes p-values

4. ❌ **Cherry-picking best epoch without justification**
   - ✅ We use epoch 50 consistently for all models

5. ❌ **Reporting accuracy without standard deviations**
   - ✅ Results include means ± stdev

---

## 📞 Getting Results Off HPC

After evaluation completes, copy results to your local machine:

```bash
# On your local machine
scp -r hpc4090@hpc4090:~/jamal_fr/results ./thesis_results/

# Or compress first (on HPC):
tar -czf results_$(date +%Y%m%d).tar.gz results/
# Then download:
scp hpc4090@hpc4090:~/jamal_fr/results_*.tar.gz ./
```

---

## 🎓 For Thesis Defense

Be prepared to answer:

1. **Why 2,000 pairs?**
   - Standard practice in face verification (LFW uses 6,000)
   - Sufficient for statistical significance (n≥1000)

2. **Why these FR weights?**
   - Ablation study: 0.3 (conservative), 0.5 (moderate), 1.0 (aggressive)
   - Demonstrates impact of different loss contributions

3. **Why epoch 50?**
   - Models were fine-tuned from pretrained CIDNet
   - 50 epochs sufficient for convergence with low learning rate

4. **How do you know improvements are significant?**
   - Statistical tests (p-values < 0.05)
   - Consistent improvement across metrics
   - Tested on 2,000 pairs (robust sample size)

5. **Trade-offs?**
   - May show slight PSNR/SSIM decrease for better face recognition
   - This is acceptable: task-specific optimization

---

## 📚 Related Files

- `eval_face_verification.py` - Main evaluation script
- `generate_lfw_pairs.py` - Generates genuine/impostor pairs
- `extended_analysis.py` - Deep-dive statistical analysis
- `QUICK_START_FACE_RECOGNITION_LOSS.md` - Complete documentation

---

## ⏱️ Timeline Summary

**If starting now:**

| Time | Action |
|------|--------|
| Now | Start `RUN_COMPLETE_ANALYSIS.sh` |
| +1-2 hours | Evaluation completes |
| +10 minutes | Review results |
| +30 minutes | Draft thesis results section |
| +1 hour | Create thesis figures and tables |

**Total: ~3-4 hours from start to thesis-ready results**

---

## ✨ Good Luck!

This evaluation will give you:
- ✅ Publishable results
- ✅ Statistical rigor
- ✅ Publication-quality figures
- ✅ Complete comparison across all configurations

**You're ready to demonstrate your thesis contribution!** 🎓

---

*Last updated: 2025-11-19*
