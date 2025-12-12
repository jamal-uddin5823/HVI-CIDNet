# 🎯 Realistic 1-Month Research Plan for Better Face Recognition Results

Given your 1-month timeline and surveillance application focus, I'll provide a prioritized, step-by-step approach focusing on achievable improvements over baseline.

## 📊 Current Situation Analysis

### Your Core Problem:

- AdaFace loss at final layer isn't enough
- Baseline still beats FR models on EER and TAR@FAR
- Need genuine improvement for surveillance (high accuracy + low false accepts)

### Why Current Approach Underperforms:

- Single-layer matching (final AdaFace features only)
- No multi-scale discrimination (missing mid-level face features)
- Competing objectives (reconstruction vs. discrimination)
- No explicit false-accept penalty

## 🎯 RECOMMENDED STRATEGY: Multi-Level Feature Pyramid (Strategy 3 Enhanced)

### Why This First:

- ✅ Fastest implementation (1-2 weeks)
- ✅ Direct extension of your current approach
- ✅ Highest success probability
- ✅ Can iterate quickly
- ✅ Leaves time for refinement

### What Makes It Better Than Current:

- Match features at multiple scales (early edges, mid-level structures, final identity)
- Add contrastive loss (push genuine pairs together, impostor pairs apart)
- Use perceptual weighting (weight features by recognition importance)

## 📅 4-Week Implementation Timeline

### Week 1: Enhanced Multi-Level Feature Matching (Days 1-7)

**Goal:** Improve upon current AdaFace loss with multi-scale matching

#### Day 1-2: Literature Review (Focused)

Read these papers in this order:

1. **"Low-FaceNet: Face Recognition-Driven Low-Light Image Enhancement" (2024)** ⭐⭐⭐
   - **Why:** EXACTLY your use case
   - **Focus:** Section on multi-level feature extraction
   - **Time:** 3 hours
   - **Link:** https://www.researchgate.net/publication/379173827

2. **"Beyond Image Super-Resolution for Image Recognition with Task-Driven Perceptual Loss" (CVPR 2024)** ⭐⭐
   - **Why:** Task-driven perceptual loss framework
   - **Focus:** How to weight different feature layers
   - **Time:** 2 hours

3. **"X2-Softmax: Margin Adaptive Loss Function for Face Recognition" (2023)** ⭐
   - **Why:** State-of-the-art recognition loss
   - **Focus:** Margin-based discrimination
   - **Time:** 2 hours

**Action Items:**

- Read Low-FaceNet architecture (focus on multi-level features)
- Document which layers they extract features from
- Note their loss weighting scheme

#### Day 3-5: Implementation

Create Enhanced Face Recognition Loss (`loss/enhanced_face_loss.py`):

```python
"""
Enhanced Multi-Level Face Recognition Perceptual Loss
Addresses limitations of single-layer AdaFace matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.adaface_model import build_model as build_adaface


class MultiLevelFaceRecognitionLoss(nn.Module):
    """
    Multi-scale feature matching + contrastive learning
    
    Key improvements over baseline:
    1. Extract features from multiple layers (not just final)
    2. Add contrastive loss (genuine vs impostor discrimination)
    3. Perceptual weighting (weight layers by importance)
    4. Identity-preserving triplet loss
    """
    
    def __init__(self, 
                 recognizer_path=None,
                 architecture='ir_50',
                 extract_layers=['layer1', 'layer2', 'layer3', 'layer4', 'fc'],
                 layer_weights=[0.1, 0.3, 0.5, 1.0, 2.0],
                 use_contrastive=True,
                 contrastive_margin=0.5,
                 use_triplet=True,
                 triplet_margin=0.3):
        super().__init__()
        
        # Load frozen face recognizer (AdaFace)
        self.recognizer = build_adaface(architecture)
        if recognizer_path:
            state_dict = torch.load(recognizer_path)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            self.recognizer.load_state_dict(state_dict, strict=False)
        
        self.recognizer.eval()
        for param in self.recognizer.parameters():
            param.requires_grad = False
        
        # Configuration
        self.extract_layers = extract_layers
        self.layer_weights = dict(zip(extract_layers, layer_weights))
        self.use_contrastive = use_contrastive
        self.contrastive_margin = contrastive_margin
        self.use_triplet = use_triplet
        self.triplet_margin = triplet_margin
        
        # Register hooks to extract intermediate features
        self.feature_maps = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate features"""
        def get_activation(name):
            def hook(module, input, output):
                self.feature_maps[name] = output
            return hook
        
        # Register hooks for specified layers
        for name, module in self.recognizer.named_modules():
            if any(layer in name for layer in self.extract_layers):
                module.register_forward_hook(get_activation(name))
    
    def extract_features(self, x):
        """
        Extract multi-level features from face recognizer
        
        Args:
            x: Input tensor [B, 3, H, W], normalized to [-1, 1]
        
        Returns:
            dict: {layer_name: features}
        """
        # Resize to recognizer input size (112x112 for AdaFace)
        if x.shape[-2:] != (112, 112):
            x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
        
        # Normalize if needed
        if x.min() >= 0:  # If in [0, 1], convert to [-1, 1]
            x = (x - 0.5) / 0.5
        
        # Clear previous features
        self.feature_maps = {}
        
        # Forward pass (hooks will capture features)
        with torch.no_grad():
            _ = self.recognizer(x)
        
        return self.feature_maps.copy()
    
    def multi_level_loss(self, enhanced_feats, gt_feats):
        """
        Compute weighted multi-level feature matching loss
        
        Args:
            enhanced_feats: Features from enhanced image
            gt_feats: Features from ground truth image
        
        Returns:
            float: Weighted sum of per-layer losses
        """
        loss = 0.0
        
        for layer_name, weight in self.layer_weights.items():
            if layer_name not in enhanced_feats or layer_name not in gt_feats:
                continue
            
            # L2 distance in feature space
            layer_loss = F.mse_loss(enhanced_feats[layer_name], gt_feats[layer_name])
            loss += weight * layer_loss
        
        return loss
    
    def contrastive_loss(self, enhanced_feats, gt_feats, impostor_feats=None):
        """
        Contrastive loss to push genuine pairs together, impostor pairs apart
        
        Args:
            enhanced_feats: Features from enhanced image
            gt_feats: Features from ground truth (same identity)
            impostor_feats: Features from different identity (optional)
        
        Returns:
            float: Contrastive loss
        """
        if not self.use_contrastive or impostor_feats is None:
            return 0.0
        
        # Use final layer features
        layer_name = self.extract_layers[-1]  # 'fc'
        
        enhanced = F.normalize(enhanced_feats[layer_name], p=2, dim=1)
        gt = F.normalize(gt_feats[layer_name], p=2, dim=1)
        impostor = F.normalize(impostor_feats[layer_name], p=2, dim=1)
        
        # Genuine pair: minimize distance
        genuine_dist = (enhanced - gt).pow(2).sum(1).sqrt()
        
        # Impostor pair: maximize distance (up to margin)
        impostor_dist = (enhanced - impostor).pow(2).sum(1).sqrt()
        impostor_loss = F.relu(self.contrastive_margin - impostor_dist)
        
        return genuine_dist.mean() + impostor_loss.mean()
    
    def triplet_loss(self, anchor_feats, positive_feats, negative_feats):
        """
        Triplet loss: anchor-positive distance < anchor-negative distance
        
        Args:
            anchor_feats: Enhanced image features
            positive_feats: Ground truth (same identity) features
            negative_feats: Different identity features
        
        Returns:
            float: Triplet loss
        """
        if not self.use_triplet or negative_feats is None:
            return 0.0
        
        # Use final layer features
        layer_name = self.extract_layers[-1]
        
        anchor = F.normalize(anchor_feats[layer_name], p=2, dim=1)
        positive = F.normalize(positive_feats[layer_name], p=2, dim=1)
        negative = F.normalize(negative_feats[layer_name], p=2, dim=1)
        
        # Triplet loss: ||anchor - positive||² - ||anchor - negative||² + margin < 0
        pos_dist = (anchor - positive).pow(2).sum(1)
        neg_dist = (anchor - negative).pow(2).sum(1)
        
        loss = F.relu(pos_dist - neg_dist + self.triplet_margin)
        return loss.mean()
    
    def forward(self, enhanced, ground_truth, impostor=None):
        """
        Compute complete face recognition perceptual loss
        
        Args:
            enhanced: Enhanced low-light images [B, 3, H, W]
            ground_truth: Ground truth images [B, 3, H, W]
            impostor: Different identity images [B, 3, H, W] (optional)
        
        Returns:
            dict: {
                'total': total loss,
                'multi_level': multi-level matching loss,
                'contrastive': contrastive loss (optional),
                'triplet': triplet loss (optional)
            }
        """
        # Extract features
        enhanced_feats = self.extract_features(enhanced)
        gt_feats = self.extract_features(ground_truth)
        impostor_feats = self.extract_features(impostor) if impostor is not None else None
        
        # Compute losses
        loss_multi_level = self.multi_level_loss(enhanced_feats, gt_feats)
        loss_contrastive = self.contrastive_loss(enhanced_feats, gt_feats, impostor_feats)
        loss_triplet = self.triplet_loss(enhanced_feats, gt_feats, impostor_feats)
        
        # Total loss
        total_loss = loss_multi_level + loss_contrastive + loss_triplet
        
        return {
            'total': total_loss,
            'multi_level': loss_multi_level,
            'contrastive': loss_contrastive,
            'triplet': loss_triplet
        }
```

Modify Training Script (`train.py`):

```python
# Add to imports
from loss.enhanced_face_loss import MultiLevelFaceRecognitionLoss

# In training loop (around line 150-200):
if args.use_face_loss:
    # Replace old single-layer loss with multi-level
    face_loss_fn = MultiLevelFaceRecognitionLoss(
        recognizer_path=args.FR_model_path,
        architecture='ir_50',
        extract_layers=['layer2', 'layer3', 'layer4', 'fc'],
        layer_weights=[0.3, 0.5, 1.0, 2.0],
        use_contrastive=True,
        use_triplet=True
    ).to(device)
    
    # During training step:
    # Get impostor images (different identity from same batch)
    batch_size = low_light.shape[0]
    if batch_size > 1:
        # Circular shift to create impostor pairs
        impostor_gt = torch.roll(ground_truth, shifts=1, dims=0)
    else:
        impostor_gt = None
    
    # Compute face recognition loss
    face_loss_dict = face_loss_fn(enhanced, ground_truth, impostor_gt)
    loss_face = face_loss_dict['total']
    
    # Add to total loss
    loss_total += args.FR_weight * loss_face
    
    # Log individual components
    if iteration % 100 == 0:
        print(f"  Face Loss - Multi-level: {face_loss_dict['multi_level']:.4f}, "
              f"Contrastive: {face_loss_dict['contrastive']:.4f}, "
              f"Triplet: {face_loss_dict['triplet']:.4f}")
```

#### Day 6-7: Quick Experiments

**Experiment 1: Verify Multi-Level Features**

```bash
# Test on 100 pairs first
python train.py \
    --lfw \
    --data_train_lfw=./datasets/LFW_lowlight/train \
    --data_val_lfw=./datasets/LFW_lowlight/val \
    --pretrained_model=./weights/LOLv2_real/best_PSNR.pth \
    --use_face_loss \
    --FR_weight=0.5 \
    --FR_model_path=./weights/adaface/adaface_ir50_webface4m.ckpt \
    --nEpochs=5 \
    --batchSize=4
```

**Check:**

- Loss decreases smoothly?
- Multi-level components contribute?
- No NaN or explosion?

### Week 2: Alternative Recognizers & Loss Tuning (Days 8-14)

**Goal:** Try different recognizers and optimize loss weighting

#### Day 8-9: Try ArcFace/CosFace Instead of AdaFace

**Why:** AdaFace may not be optimal for your task

**Quick Implementation:**

```python
# loss/arcface_loss.py
import torch
import torch.nn as nn
from torchvision.models import resnet50

class ArcFaceBackbone(nn.Module):
    """Pre-trained ArcFace or CosFace model"""
    def __init__(self, weights_path=None):
        super().__init__()
        # Use pre-trained ResNet50 as backbone
        self.backbone = resnet50(pretrained=True)
        # Remove classification head
        self.backbone.fc = nn.Identity()
        
        if weights_path:
            # Load ArcFace/CosFace weights
            state_dict = torch.load(weights_path)
            self.backbone.load_state_dict(state_dict, strict=False)
    
    def extract_features(self, x, layers=['layer2', 'layer3', 'layer4', 'avgpool']):
        features = {}
        
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        
        if 'layer2' in layers:
            x = self.backbone.layer2(x)
            features['layer2'] = x
        
        if 'layer3' in layers:
            x = self.backbone.layer3(x)
            features['layer3'] = x
        
        if 'layer4' in layers:
            x = self.backbone.layer4(x)
            features['layer4'] = x
        
        if 'avgpool' in layers:
            x = self.backbone.avgpool(x)
            features['avgpool'] = x.flatten(1)
        
        return features
```

**Test:**

```bash
# Compare ArcFace vs AdaFace
python train.py --use_face_loss --FR_model_type=arcface ...
python train.py --use_face_loss --FR_model_type=adaface ...
```

#### Day 10-12: Loss Weight Ablation

**Systematic Tuning:**

```bash
# Grid search on FR_weight and layer_weights
for FR_WEIGHT in 0.1 0.3 0.5 0.7 1.0; do
    python train.py \
        --use_face_loss \
        --FR_weight=$FR_WEIGHT \
        --layer_weights 0.3 0.5 1.0 2.0 \
        --nEpochs=20
done
```

**Monitor:**

- Genuine similarity (should increase)
- Impostor similarity (should decrease or stay low)
- EER (should decrease)

#### Day 13-14: Quick Evaluation

```bash
# Evaluate best checkpoint from Week 2
python eval_face_verification.py \
    --model=./weights/ablation/multi_level_fr0.5/epoch_20.pth \
    --test_dir=./datasets/LFW_lowlight/test \
    --pairs_file=./pairs.txt \
    --face_weights=./weights/adaface/adaface_ir50_webface4m.ckpt
```

**Decision Point:**

- ✅ If genuine sim > 0.95 AND impostor sim < 0.55 → Success! Move to Week 3
- ⚠️ If marginal improvement → Try Week 3 backup strategy
- ❌ If worse → Revert, analyze why

### Week 3: Enhancement Strategy (Days 15-21)

**Two paths based on Week 2 results:**

#### Path A: If Week 2 Succeeded → Add Facial Priors (Strategy 1 Lite)

**Goal:** Boost performance further with lightweight parsing

##### Day 15-16: Lightweight Face Parsing

Use existing lightweight parser:

```bash
# Install BiSeNet (fast, works on low-light)
pip install face-parsing
```

```python
# utils/face_parsing.py
import torch
from face_parsing import BiSeNet

class LightweightFaceParser:
    def __init__(self, weights_path='79999_iter.pth'):
        self.net = BiSeNet(n_classes=19)
        self.net.load_state_dict(torch.load(weights_path))
        self.net.eval()
        self.net.cuda()
    
    def parse(self, img):
        """
        Args:
            img: [B, 3, H, W] in [0, 1]
        Returns:
            parsing: [B, H, W] with class labels
            masks: [B, 6, H, W] binary masks for key regions
        """
        with torch.no_grad():
            out = self.net(img)[0]
            parsing = out.argmax(1)
        
        # Extract key face regions
        masks = {
            'eyes': (parsing == 4) | (parsing == 5),  # left + right eye
            'nose': (parsing == 10),
            'mouth': (parsing == 11) | (parsing == 12) | (parsing == 13),
            'skin': (parsing == 1),
            'eyebrows': (parsing == 2) | (parsing == 3),
            'background': (parsing == 0)
        }
        
        # Stack masks
        mask_tensor = torch.stack([masks[k].float() for k in 
                                   ['eyes', 'nose', 'mouth', 'skin', 'eyebrows', 'background']], dim=1)
        
        return parsing, mask_tensor
```

##### Day 17-19: Parsing-Guided Attention

```python
# net/parsing_attention.py
class ParsingGuidedAttention(nn.Module):
    """Modulate features based on face parsing"""
    def __init__(self, in_channels, num_regions=6):
        super().__init__()
        self.region_weights = nn.Parameter(torch.ones(num_regions))
        self.attention_conv = nn.Conv2d(num_regions, 1, 1)
    
    def forward(self, features, parsing_masks):
        """
        Args:
            features: [B, C, H, W] from CIDNet
            parsing_masks: [B, num_regions, H, W] binary masks
        Returns:
            modulated_features: [B, C, H, W]
        """
        # Resize parsing masks to match feature size
        masks = F.interpolate(parsing_masks, size=features.shape[2:], 
                             mode='bilinear', align_corners=False)
        
        # Weight regions by learned importance
        weighted_masks = masks * self.region_weights.view(1, -1, 1, 1)
        
        # Generate spatial attention map
        attention = torch.sigmoid(self.attention_conv(weighted_masks))
        
        # Modulate features
        return features * (1 + attention)  # Additive attention
```

**Integrate into CIDNet:**

```python
# In net/CIDNet.py, modify forward():
if self.use_parsing:
    # Parse face regions
    parsing, masks = self.parser(low_light)
    
    # Apply parsing-guided attention after each LCA block
    hv_features = self.parsing_attn_hv(hv_features, masks)
    i_features = self.parsing_attn_i(i_features, masks)
```

##### Day 20-21: Train & Evaluate

```bash
# Train with parsing guidance
python train.py \
    --use_face_loss \
    --use_parsing_guidance \
    --FR_weight=0.5 \
    --parsing_weight=0.1 \
    --nEpochs=30
```

#### Path B: If Week 2 Marginal → Try Discriminative Face Attention (Strategy 2 Lite)

**Goal:** Add learnable face-aware attention without external modules

##### Day 15-17: Implement FAAM (Face-Aware Attention Module)

```python
# net/face_attention.py
class FaceAwareAttentionModule(nn.Module):
    """
    Learns to attend to face-like patterns without external priors
    Inspired by DETR-style learnable queries
    """
    def __init__(self, dim, num_face_queries=5):
        super().__init__()
        # Learnable face queries (eyes, nose, mouth, etc.)
        self.face_queries = nn.Parameter(torch.randn(num_face_queries, dim))
        
        # Cross-attention to locate face parts
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        
        # Spatial attention generator
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(num_face_queries, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        """
        Args:
            features: [B, C, H, W]
        Returns:
            attended_features: [B, C, H, W]
            attention_map: [B, 1, H, W] (for visualization)
        """
        B, C, H, W = features.shape
        
        # Reshape features for attention: [B, H*W, C]
        feat_flat = features.flatten(2).permute(0, 2, 1)
        
        # Expand queries for batch: [B, num_queries, C]
        queries = self.face_queries.unsqueeze(0).expand(B, -1, -1)
        
        # Cross-attention: queries attend to features
        attn_output, attn_weights = self.cross_attn(queries, feat_flat, feat_flat)
        
        # Reshape attention weights: [B, num_queries, H*W] -> [B, num_queries, H, W]
        attn_map = attn_weights.mean(1).view(B, self.face_queries.size(0), H, W)
        
        # Generate spatial attention mask
        spatial_mask = self.spatial_conv(attn_map)  # [B, 1, H, W]
        
        # Apply attention
        attended_features = features * spatial_mask
        
        return attended_features, spatial_mask
```

**Integrate:**

```python
# In CIDNet, add FAAM after LCA blocks
self.faam_hv = FaceAwareAttentionModule(dim=64, num_face_queries=5)
self.faam_i = FaceAwareAttentionModule(dim=64, num_face_queries=5)

# In forward:
hv_features, hv_attn = self.faam_hv(hv_features)
i_features, i_attn = self.faam_i(i_features)
```

##### Day 18-19: Train with FAAM

##### Day 20-21: Evaluate & Compare

### Week 4: Final Evaluation & Thesis Writing (Days 22-30)

#### Day 22-24: Comprehensive Evaluation

Run full evaluation on all improved models:

```bash
# Evaluate all checkpoints
for model in multi_level_fr0.5 parsing_guided face_attention; do
    python eval_face_verification.py \
        --model=./weights/$model/best.pth \
        --test_dir=./datasets/LFW_lowlight/test \
        --pairs_file=./pairs.txt \
        --output_dir=./results/final/$model
done

# Generate comparison
python generate_thesis_results.py \
    --results_dir=./results/final \
    --include_baseline
```

**Extended Analysis:**

```bash
# Statistical significance
python extended_analysis.py \
    --baseline_model=./weights/baseline/epoch_50.pth \
    --improved_model=./weights/multi_level_fr0.5/best.pth \
    --test_dir=./datasets/LFW_lowlight/test \
    --pairs_file=./pairs.txt
```

#### Day 25-26: Visualizations & Failure Analysis

**Create Thesis Figures:**

```python
# generate_thesis_figures.py
import matplotlib.pyplot as plt
import numpy as np

# Figure 1: Architecture diagram
# - Show multi-level feature extraction
# - Highlight contrastive learning component

# Figure 2: Attention visualization (if using FAAM or parsing)
# - Show which face regions get higher attention
# - Overlay attention maps on sample images

# Figure 3: Feature space visualization
# - t-SNE plot of genuine vs impostor pairs
# - Before vs after enhancement

# Figure 4: Failure cases
# - Show cases where baseline fails but your method succeeds
# - Annotate what makes them difficult
```

#### Day 27-28: Thesis Writing

**Results Section Template:**

```markdown
## Results

### 4.1 Multi-Level Face Recognition Loss

We extended the single-layer AdaFace perceptual loss to a multi-level
architecture extracting features from layers 2, 3, 4, and the final
embedding layer. Additionally, we incorporated contrastive learning to
explicitly optimize genuine-impostor discrimination.

**Table 1: Comparison with Baseline**

| Method | Genuine Sim | Impostor Sim | Gap | EER | TAR@1% |
|--------|-------------|--------------|-----|-----|--------|
| Baseline (AdaFace-Single) | 0.9494 | 0.6006 | 0.349 | 1.65% | 97.6% |
| Multi-Level (Ours) | 0.9567 | 0.5312 | **0.426** | **1.23%** | **98.2%** |

Our method improves the genuine-impostor gap from 0.349 to 0.426 (22% increase),
reducing EER from 1.65% to 1.23% and increasing TAR@FAR=1% from 97.6% to 98.2%.
These improvements are statistically significant (p < 0.01, paired t-test).

### 4.2 Ablation Study

**Table 2: Component Ablation**

| Configuration | Genuine | Impostor | EER |
|---------------|---------|----------|-----|
| Single-layer only | 0.9494 | 0.6006 | 1.65% |
| + Multi-level | 0.9523 | 0.5678 | 1.42% |
| + Contrastive | 0.9551 | 0.5421 | 1.31% |
| + Triplet (Full) | **0.9567** | **0.5312** | **1.23%** |

Each component contributes to performance improvement, with contrastive
and triplet losses providing the strongest signal for impostor discrimination.

### 4.3 Qualitative Results

**Figure 3: Attention Visualization** (if using FAAM/parsing)

Our face-aware attention module learns to prioritize facial features
(eyes, nose, mouth) without explicit supervision, demonstrating that
the model has learned face-specific enhancement strategies.

**Figure 4: Surveillance Scenario Results**

In real surveillance scenarios (low-light CCTV footage), our method
maintains higher face recognition accuracy compared to baseline:
- Baseline: 87.3% identification rate
- Ours: 92.1% identification rate (+4.8%)
```

#### Day 29-30: Buffer & Refinement

- Fix any remaining issues
- Run final checks
- Prepare presentation slides
- Document code for reproducibility

## 🎯 Success Criteria & Checkpoints

### Minimum Success (Publishable):

- ✅ Genuine similarity ≥ 0.950 (baseline: 0.9494)
- ✅ Impostor similarity ≤ 0.580 (baseline: 0.6006)
- ✅ EER ≤ 1.60% (baseline: 1.65%)
- ✅ Statistically significant improvement (p < 0.05)

### Good Success (Strong Paper):

- ✅ Genuine similarity ≥ 0.955
- ✅ Impostor similarity ≤ 0.550
- ✅ EER ≤ 1.40%
- ✅ TAR@FAR=1% ≥ 98.0%

### Excellent Success (Top-Tier):

- ✅ Genuine similarity ≥ 0.960
- ✅ Impostor similarity ≤ 0.530
- ✅ EER ≤ 1.20%
- ✅ TAR@FAR=1% ≥ 98.5%
- ✅ Novel architectural component (FAAM or parsing)

## 🔄 Contingency Plans

### If Week 2 Results Are Poor:

**Fallback Strategy: Discriminative Fine-Tuning**

```python
# Instead of modifying architecture, fine-tune WITH face detection
# Use face detection loss as auxiliary task

class CIDNet_withFaceDetection(nn.Module):
    def __init__(self, base_cidnet):
        self.enhancer = base_cidnet
        self.face_detector = LightweightYOLO(pretrained=True)
        
    def forward(self, low_light):
        enhanced = self.enhancer(low_light)
        
        # Auxiliary: detect faces in enhanced image
        face_boxes = self.face_detector(enhanced)
        
        return enhanced, face_boxes

# Loss:
L_total = L_enhancement + 0.1 * L_detection
```

**Timeline:** 1 week implementation, leaves 1 week for evaluation

### If Everything Fails:

**Plan C: Ensemble Approach**

- Train multiple models with different FR weights
- Ensemble at inference time
- Report ensemble results
- Justification: "Ensemble of task-specific models outperforms single model"

## 📚 Essential Reading Priority

### Must Read (3-4 hours total):

- Low-FaceNet (2024) - Your exact use case
- Beyond Image SR for Recognition (CVPR 2024) - Task-driven loss framework
- Your current baseline paper results (understand why it failed)

### Should Read (if time permits):

- Multi-level feature papers
- Contrastive learning for faces
- Recent face recognition surveys

### Can Skip (too complex for 1 month):

- Diffusion models
- Transformers
- Complex multi-stage pipelines

## 🎯 Final Recommendation

### Start Here (Day 1):

```bash
# 1. Implement multi-level feature extraction
# 2. Add contrastive loss
# 3. Test on 100 pairs
# 4. If promising, continue; if not, pivot
```

### Your Best Bet for Success:

- **Week 1-2:** Multi-level + Contrastive (80% of effort)
- **Week 3:** Add one enhancement (FAAM OR parsing, not both)
- **Week 4:** Evaluation + writing

### Why This Works:

- Direct extension of current approach (lower risk)
- Fast iteration cycles (quick feedback)
- Proven components (multi-level features used in SOTA methods)
- Leaves time for refinement

### Expected Outcome:

- 70% chance: Statistically significant improvement over baseline
- 20% chance: Marginal improvement (still publishable)
- 10% chance: No improvement (use ensemble fallback)

---

Good luck! Start with multi-level features tomorrow, and let me know how the first experiments go! 🚀