# Low-Light Image Enhancement for Face Recognition: Ablation Study Observations

**Project:** HVI-CIDNet with Face Recognition Perceptual Loss
**Task:** Low-light image enhancement optimized for face recognition from low-light frames
**Date:** November 24, 2025
**Evaluation:** Complete analysis on 2000 image pairs (1000 genuine + 1000 impostor)

---

## Executive Summary

This ablation study investigated the impact of **Face Recognition (FR) loss weight** and **SSIM loss (D_weight)** on low-light image enhancement for face recognition tasks. We trained **12 models** with combinations of:
- **FR weights:** 0 (baseline), 0.3, 0.5, 1.0
- **D_weights (SSIM):** 0.5, 1.0, 1.5

**Key Finding:** Adding Face Recognition perceptual loss (AdaFace) shows **mixed results**:
- ✅ **Improves genuine pair similarity** (better same-person recognition)
- ❌ **Worsens impostor pair similarity** (worse different-person discrimination)
- ❌ **Often increases EER** (worse overall verification accuracy)
- ⚠️ **Weak statistical significance** (only 2/3 McNemar tests significant, 0/3 t-tests significant)

**Overall:** The baseline model (no FR loss) often performs **best** on verification metrics (EER, TAR), suggesting FR loss may have **limited practical benefit** for this task.

---

## 1. Experimental Setup

### 1.1 Dataset
- **Training/Validation:** LFW (Labeled Faces in the Wild) synthetic low-light dataset
- **Test Set:** 2000 evaluation pairs following LFW verification protocol
  - 1000 genuine pairs (same identity)
  - 1000 impostor pairs (different identities)

### 1.2 Model Architecture
- **Base Model:** CIDNet (HVI-CIDNet)
- **FR Loss Model:** AdaFace (IR-50, pretrained on WebFace4M)
- **Training:** 50 epochs with pretrained CIDNet weights

### 1.3 Loss Function Components
- **L1 Loss:** Pixel-wise reconstruction (weight = 1.0)
- **SSIM Loss (D_weight):** Structural similarity (varied: 0.5, 1.0, 1.5)
- **Edge Loss:** Preserves image edges (weight = 1.0)
- **Perceptual Loss (VGG):** Perceptual quality (weight = 1.0)
- **HVI Loss:** Color space transformation loss
- **Face Recognition Loss (FR_weight):** AdaFace feature similarity (varied: 0, 0.3, 0.5, 1.0)

---

## 2. Complete Results Table

### 2.1 All 12 Model Configurations

| Configuration      | D_weight | Genuine Sim ↑ | Impostor Sim ↓ | EER ↓    | TAR@FAR=1% ↑ | PSNR (dB) ↑ | SSIM ↑ |
|-------------------|----------|---------------|----------------|----------|--------------|-------------|--------|
| **BASELINES**      |          |               |                |          |              |             |        |
| baseline           | 0.5      | 0.9532        | 0.6620         | 2.35%    | 95.80%       | 23.30       | 0.7805 |
| baseline           | 1.0      | 0.9388        | **0.5623** ✓  | 2.30%    | 96.20%       | 23.23       | 0.7796 |
| baseline           | 1.5      | 0.9437        | 0.5921         | **1.80%** ✓ | **97.30%** ✓| 23.28       | 0.7808 |
| **FR = 0.3**       |          |               |                |          |              |             |        |
| fr_weight_0.3      | 0.5      | 0.9437        | 0.5707         | 2.70%    | 94.30%       | 23.27       | 0.7811 |
| fr_weight_0.3      | 1.0      | **0.9551** ✓ | 0.6723         | 2.00%    | 96.80%       | 23.23       | 0.7794 |
| fr_weight_0.3      | 1.5      | 0.9546        | 0.6732         | 3.10%    | 94.10%       | 23.30       | **0.7812** ✓ |
| **FR = 0.5**       |          |               |                |          |              |             |        |
| fr_weight_0.5      | 0.5      | 0.9432        | 0.5940         | 2.85%    | 96.20%       | 23.22       | 0.7768 |
| fr_weight_0.5      | 1.0      | 0.9422        | 0.5714         | 3.00%    | 95.00%       | 23.26       | 0.7801 |
| fr_weight_0.5      | 1.5      | 0.9500        | 0.6245         | 2.45%    | **97.30%** ✓| 23.29       | 0.7803 |
| **FR = 1.0**       |          |               |                |          |              |             |        |
| fr_weight_1.0      | 0.5      | 0.9538        | 0.6758         | 3.80%    | 93.00%       | 23.21       | 0.7776 |
| fr_weight_1.0      | 1.0      | 0.9437        | 0.5919         | 3.10%    | 94.50%       | 23.25       | 0.7791 |
| fr_weight_1.0      | 1.5      | 0.9490        | 0.6319         | 2.85%    | 95.70%       | 23.28       | 0.7806 |

**✓ = Best performer for that metric**

**Metric Definitions:**
- **Genuine Similarity:** Cosine similarity for same-person pairs (HIGHER = better recognition)
- **Impostor Similarity:** Cosine similarity for different-person pairs (LOWER = better discrimination)
- **EER (Equal Error Rate):** Error rate where false rejection = false acceptance (LOWER = better)
- **TAR@FAR=1%:** True Accept Rate at 1% False Accept Rate (HIGHER = better)
- **PSNR/SSIM:** Image quality metrics (HIGHER = better)

---

## 3. Key Observations

### 3.1 The Trade-Off Problem

**FR Loss Creates a Problematic Trade-Off:**

1. **✅ Genuine Similarity Improvement (Good)**
   - FR=0.3, D=1.0: **+1.63%** improvement (0.9388 → 0.9551)
   - FR=0.3, D=1.5: **+1.09%** improvement (0.9437 → 0.9546)
   - FR=1.0, D=0.5: **+0.06%** improvement (0.9532 → 0.9538)

2. **❌ Impostor Similarity Degradation (Bad)**
   - FR=0.3, D=1.0: **+19.5%** increase (0.5623 → 0.6723) - Much worse discrimination!
   - FR=0.3, D=1.5: **+13.7%** increase (0.5921 → 0.6732)
   - FR=1.0, D=0.5: **+2.1%** increase (0.6620 → 0.6758)

3. **❌ EER Often Increases (Bad)**
   - FR=0.3, D=0.5: **+0.35%** worse (2.35% → 2.70%)
   - FR=0.3, D=1.5: **+1.30%** worse (1.80% → 3.10%)
   - FR=1.0, D=0.5: **+1.45%** worse (2.35% → 3.80%)

### 3.2 Best Performing Configurations

**For Overall Verification Accuracy (EER + TAR):**
1. **baseline_d1.5** - EER: 1.80%, TAR@1%: 97.30% ⭐ **BEST OVERALL**
2. **fr_weight_0.5_d1.5** - EER: 2.45%, TAR@1%: 97.30%
3. **baseline_d1.0** - EER: 2.30%, TAR@1%: 96.20%

**For Discrimination (Low Impostor Similarity):**
1. **baseline_d1.0** - Impostor Sim: 0.5623 ⭐ **BEST DISCRIMINATION**
2. **fr_weight_0.3_d0.5** - Impostor Sim: 0.5707
3. **fr_weight_0.5_d1.0** - Impostor Sim: 0.5714

**For Face Similarity (Genuine Pairs Only):**
1. **fr_weight_0.3_d1.0** - Genuine Sim: 0.9551 ⭐ **BEST GENUINE SIM**
2. **fr_weight_0.3_d1.5** - Genuine Sim: 0.9546
3. **fr_weight_1.0_d0.5** - Genuine Sim: 0.9538

### 3.3 Impact of D_weight (SSIM Loss)

**D_weight = 1.5 consistently performs best across all FR configurations:**
- Achieves best EER for baseline (1.80%)
- Achieves best TAR@1% (97.30% for both baseline and FR=0.5)
- Best SSIM values (structural similarity)

**D_weight = 1.0 shows best discrimination:**
- Lowest impostor similarity for baseline (0.5623)
- Good balance between genuine and impostor scores

**D_weight = 0.5 is least consistent:**
- Higher impostor similarities
- More variable performance

---

## 4. Statistical Significance Analysis

### 4.1 Extended Analysis Results (FR=0.5 vs Baseline)

The extended_analysis compared **FR weight = 0.5** models against their corresponding baselines using two statistical tests:

| D_weight | McNemar Test        | Paired t-test       | Overall Significant? |
|----------|---------------------|---------------------|---------------------|
| 0.5      | p=0.3105 (❌ NO)    | p=0.2407 (❌ NO)    | ❌ **NOT SIGNIFICANT** |
| 1.0      | p=0.0133 (✅ YES)   | p=0.1665 (❌ NO)    | ⚠️ **MIXED** |
| 1.5      | p=0.0159 (✅ YES)   | p=0.2414 (❌ NO)    | ⚠️ **MIXED** |

**Key Findings:**
- **McNemar's Test:** Only 2 out of 3 tests show significance (D=1.0, D=1.5)
- **Paired t-test:** NONE of the t-tests show significance
- **Conclusion:** **Statistical evidence for FR loss benefit is WEAK**

### 4.2 What This Means

**McNemar's Test (Classification Accuracy):**
- Tests whether FR model makes **different classification decisions** than baseline
- Significant for D=1.0 and D=1.5 (p < 0.05)
- Means: FR model correctly classifies some pairs that baseline gets wrong
- **BUT:** This doesn't mean overall accuracy improves (could be swapping errors)

**Paired t-test (Similarity Scores):**
- Tests whether **mean similarity scores** differ significantly
- NOT significant for any D_weight (all p > 0.13)
- Means: **Average similarity improvement is NOT statistically meaningful**
- The +1.63% genuine similarity gain for FR=0.3 is within noise/variance

### 4.3 Thesis Implications

⚠️ **For your thesis, you CANNOT claim:**
- "FR loss significantly improves face verification" (EER often worse)
- "Statistically significant improvement in recognition" (t-tests all non-significant)
- "Better overall performance" (baselines win on EER/TAR)

✅ **What you CAN claim:**
- "FR loss improves genuine pair similarity by up to 1.63%"
- "Classification accuracy shows marginal significance for some configurations (McNemar p < 0.05)"
- "Trade-off exists: genuine similarity ↑ but impostor similarity ↑ (discrimination ↓)"
- "Baseline models often achieve better verification performance"

---

## 5. Detailed Analysis by FR Weight

### 5.1 FR Weight = 0.3 (Moderate FR Loss)

**Performance:**
- **Best genuine similarity:** 0.9551 (D=1.0) - highest across all models
- **High impostor similarity:** 0.6723 (D=1.0) - poor discrimination
- **Mixed EER:** 2.00% to 3.10% depending on D_weight
- **Variable TAR:** 94.10% to 96.80%

**Observations:**
- Achieves highest genuine pair recognition
- But sacrifices impostor discrimination
- Only competitive with baseline when D=1.0
- Not recommended for deployment (worse EER and TAR than baseline in 2/3 cases)

### 5.2 FR Weight = 0.5 (Balanced FR Loss)

**Performance:**
- **Moderate genuine similarity:** 0.9422 to 0.9500
- **Moderate impostor similarity:** 0.5714 to 0.6245
- **EER:** 2.45% to 3.00%
- **Best result:** D=1.5 achieves 97.30% TAR (ties baseline)

**Statistical Significance:**
- McNemar significant for D=1.0, D=1.5
- t-test NOT significant for any D_weight
- Classification accuracy improves but not similarity scores

**Observations:**
- More balanced than FR=0.3
- D=1.5 configuration performs well (ties baseline TAR)
- Still generally underperforms baseline on EER

### 5.3 FR Weight = 1.0 (Strong FR Loss)

**Performance:**
- **High genuine similarity:** 0.9437 to 0.9538
- **High impostor similarity:** 0.5919 to 0.6758
- **Worst EER:** 2.85% to 3.80%
- **Worst TAR:** 93.00% to 95.70%

**Observations:**
- **Overfitting to face features**
- Consistently worst verification performance
- Highest EER (3.80% for D=0.5) - 111% worse than best baseline!
- **NOT RECOMMENDED**

### 5.4 FR Weight = 0 (Baseline - No FR Loss)

**Performance:**
- **Good genuine similarity:** 0.9388 to 0.9532
- **Best impostor similarity:** 0.5623 (D=1.0)
- **Best EER:** 1.80% (D=1.5)
- **Best TAR:** 97.30% (D=1.5)

**Observations:**
- **Consistently best or near-best** on verification metrics
- Best discrimination (lowest impostor similarity)
- No trade-off issues
- **RECOMMENDED for deployment**

---

## 6. Why Does FR Loss Hurt Performance?

### 6.1 Feature Space Compression Hypothesis

**Problem:** FR loss pushes all enhanced images toward "face-like" features in AdaFace space

**Effect:**
1. **Genuine pairs move closer** (good) - higher similarity
2. **Impostor pairs ALSO move closer** (bad) - harder to distinguish different people
3. **Overall discriminability decreases** - verification threshold becomes less effective

**Analogy:**
- Baseline: Spreads people across large feature space (easy to tell apart)
- FR loss: Compresses everyone into smaller "face-like" region (harder to tell apart)

### 6.2 Task Mismatch

**Training objective:** Minimize L1 + SSIM + VGG + FR losses jointly

**Testing objective:** Maximize verification accuracy (EER, TAR)

**Mismatch:**
- FR loss optimizes for "looking like the GT in AdaFace space"
- But verification needs "maximizing same-person similarity WHILE minimizing different-person similarity"
- FR loss doesn't explicitly optimize the discriminative aspect

### 6.3 Pretrained AdaFace May Not Generalize

**AdaFace trained on:** WebFace4M (normal-light, diverse faces)

**Our task:** LFW low-light synthetic images

**Potential issues:**
- AdaFace features may not be optimal for low-light enhanced images
- Domain gap between training and application
- FR loss guidance may be misaligned with actual verification needs

---

## 7. Recommendations

### 7.1 For Thesis Presentation

**DO:**
- Present complete results table showing trade-offs
- Acknowledge that baseline often performs best
- Discuss genuine vs. impostor similarity trade-off
- Report weak statistical significance honestly
- Frame as exploratory study with mixed results

**DON'T:**
- Claim "significant improvement" (not supported by data)
- Cherry-pick only genuine similarity results
- Ignore EER/TAR degradation
- Overstate statistical significance

### 7.2 Honest Thesis Narrative

**Suggested framing:**

> "We investigated whether face recognition perceptual loss (AdaFace) could improve low-light enhancement for face verification. Our ablation study on 12 model configurations revealed a complex trade-off: FR loss improves genuine pair similarity (same-person recognition) by up to 1.63%, but simultaneously degrades impostor pair discrimination, leading to higher EER in most cases. Statistical analysis showed weak significance (2/3 McNemar tests significant, 0/3 t-tests significant). Surprisingly, baseline models without FR loss achieved the best overall verification performance (EER: 1.80%, TAR@1%: 97.30%). This suggests that task-specific perceptual losses may not always improve downstream task performance, and careful evaluation of trade-offs is essential."

### 7.3 For Deployment

**Recommended Configuration: baseline_d1.5**
- No FR loss (simpler, faster, better performance)
- EER: 1.80% (best)
- TAR@1%: 97.30% (tied best)
- SSIM: 0.7808 (good image quality)

**Alternative: fr_weight_0.5_d1.5** (if you must use FR loss for novelty)
- Ties baseline on TAR@1%: 97.30%
- Moderate EER: 2.45% (acceptable, though 36% worse than baseline)
- Good genuine similarity: 0.9500
- Statistical significance: McNemar p=0.0159

---

## 8. What Went Well vs. What Didn't

### 8.1 What Went Well ✅

1. **Comprehensive evaluation:** 12 models, 2000 pairs, rigorous metrics
2. **Low-light enhancement works:** All models dramatically improve over raw low-light (41% EER → 2-4% EER)
3. **Statistical rigor:** Multiple statistical tests, per-identity analysis
4. **Some genuine similarity gains:** FR=0.3 achieves +1.63% for D=1.0
5. **Good image quality:** All models maintain PSNR ~23.2-23.3 dB, SSIM ~0.77-0.78

### 8.2 What Didn't Work ❌

1. **FR loss doesn't improve verification:** EER and TAR often worse
2. **Impostor discrimination degrades:** Higher impostor similarity with FR loss
3. **Weak statistical significance:** t-tests all non-significant
4. **Inconsistent across D_weights:** No clear winner
5. **Baseline outperforms:** Simpler model achieves best results

---

## 9. Future Work (If Continuing This Research)

### 9.1 Architectural Changes

1. **Discriminative FR loss:**
   - Maximize genuine similarity AND minimize impostor similarity explicitly
   - Use contrastive loss or triplet loss instead of reconstruction loss

2. **Identity-aware training:**
   - Sample mini-batches with multiple images per identity
   - Optimize intra-class compactness AND inter-class separation

3. **Multi-task learning:**
   - Joint loss: L_reconstruction + L_genuine_sim - L_impostor_sim
   - Balance weights to avoid feature space compression

### 9.2 Evaluation Improvements

1. **Real low-light data:** Test on actual low-light captures, not synthetic
2. **Cross-dataset evaluation:** Evaluate on different face databases
3. **Different FR models:** Try ArcFace, CosFace, or ElasticFace
4. **Larger test set:** 2000 pairs may have high variance

### 9.3 Alternative Approaches

1. **Domain-specific FR model:** Fine-tune AdaFace on low-light faces
2. **Attention mechanisms:** Focus enhancement on facial regions only
3. **Test-time optimization:** Adjust threshold based on deployment data
4. **Ensemble methods:** Combine baseline + FR models

---

## 10. Conclusion

### 10.1 Main Findings

This ablation study provides **cautionary evidence** about using face recognition perceptual losses for low-light image enhancement:

1. ⚠️ **Marginal Benefits:** FR loss improves genuine similarity (+0.06% to +1.63%) but gains are statistically weak
2. ❌ **Significant Costs:** Impostor discrimination degrades (+2.1% to +19.5%), increasing EER
3. ✅ **Baseline Wins:** Models without FR loss achieve best verification accuracy (1.80% EER, 97.30% TAR)
4. 📊 **Weak Significance:** Only 2/3 McNemar tests significant, 0/3 t-tests significant

### 10.2 Answer to Research Question

**Q: Does face recognition perceptual loss improve low-light enhancement for face verification?**

**A: No, not in this experimental setup.** While FR loss improves genuine pair similarity, it simultaneously degrades impostor discrimination, resulting in worse overall verification performance. The baseline model (no FR loss) achieves the best EER and TAR metrics. Statistical evidence is weak (non-significant t-tests), and the trade-off is unfavorable.

### 10.3 Value of This Work

Despite negative results, this study contributes:

✅ **Rigorous evaluation methodology** (12 models, 2000 pairs, statistical tests)
✅ **Honest reporting of trade-offs** (genuine vs. impostor performance)
✅ **Clear recommendation:** Use baseline_d1.5 for deployment
✅ **Lessons for future work:** Need discriminative losses, not just reconstruction

**Negative results are publishable and valuable** - they prevent others from pursuing ineffective approaches.

---

## Appendix A: Result File Locations

### Individual Model Results
```
./results/full_evaluation/baseline_d{0.5,1,1.5}/face_verification_results.txt
./results/full_evaluation/fr_weight_0.3_d{0.5,1,1.5}/face_verification_results.txt
./results/full_evaluation/fr_weight_0.5_d{0.5,1,1.5}/face_verification_results.txt
./results/full_evaluation/fr_weight_1.0_d{0.5,1,1.5}/face_verification_results.txt
```

### Summary and Analysis
```
./results/full_evaluation/comparison_table.txt
./results/full_evaluation/thesis_results_summary.txt
./results/extended_analysis/d_{0.5,1,1.5}/statistical_significance.txt
./results/extended_analysis/d_{0.5,1,1.5}/per_identity_analysis.csv
```

### Visualizations
```
./results/ablation/figures/*.png
./results/full_evaluation/plots/*.png
```

---

## Appendix B: Per-D_weight Comparisons

### Baseline vs. Best FR (Same D_weight)

**D_weight = 0.5:**
- Baseline: Gen=0.9532, Imp=0.6620, EER=2.35%, TAR=95.80%
- Best FR (1.0): Gen=0.9538, Imp=0.6758, EER=3.80%, TAR=93.00%
- **Verdict:** Baseline wins (much better EER and TAR)

**D_weight = 1.0:**
- Baseline: Gen=0.9388, Imp=0.5623, EER=2.30%, TAR=96.20%
- Best FR (0.3): Gen=0.9551, Imp=0.6723, EER=2.00%, TAR=96.80%
- **Verdict:** FR=0.3 marginally better EER (-0.30%), slightly better TAR (+0.60%)

**D_weight = 1.5:**
- Baseline: Gen=0.9437, Imp=0.5921, EER=1.80%, TAR=97.30%
- Best FR (0.3): Gen=0.9546, Imp=0.6732, EER=3.10%, TAR=94.10%
- **Verdict:** Baseline wins (much better EER and TAR)

**Overall:** Baseline wins 2/3 comparisons decisively, loses 1/3 marginally.

---

**Document prepared by:** Claude (AI Assistant)
**For:** HVI-CIDNet Low-Light Face Recognition Thesis Project
**Version:** 2.0 (CORRECTED - Based on Actual Data)
**Last Updated:** November 24, 2025
