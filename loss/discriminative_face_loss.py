"""
Discriminative Multi-Level Face Recognition Loss

Key Differences from Current Approach:
1. Multi-level features (not just final layer)
2. Explicit contrastive loss (push impostors apart)
3. Identity-preserving triplet loss
4. Margin-based discrimination
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.adaface_model import build_model as build_adaface


class DiscriminativeMultiLevelFaceLoss(nn.Module):
    """
    Addresses feature space compression by explicitly optimizing discrimination
    
    Loss Components:
    1. Multi-level reconstruction (like current, but multiple layers)
    2. Contrastive loss (NEW: push impostors apart)
    3. Triplet loss (NEW: anchor-positive < anchor-negative)
    4. Margin enforcement (NEW: maintain separation margin)
    """
    
    def __init__(self,
                 recognizer_path,
                 architecture='ir_50',
                 # Multi-level configuration
                 feature_layers=['layer2', 'layer3', 'layer4', 'fc'],
                 layer_weights=[0.2, 0.4, 0.8, 1.0],
                 # Discriminative configuration
                 use_contrastive=True,
                 contrastive_margin=0.4,  # Impostors must be > 0.4 apart
                 contrastive_weight=1.0,
                 use_triplet=True,
                 triplet_margin=0.2,
                 triplet_weight=0.5,
                 # Temperature for contrastive learning
                 temperature=0.07):
        super().__init__()
        
        # Load frozen face recognizer
        self.recognizer = build_adaface(architecture)
        if recognizer_path:
            state_dict = torch.load(recognizer_path, map_location='cuda')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            self.recognizer.load_state_dict(state_dict, strict=False)
        
        self.recognizer.eval()
        for param in self.recognizer.parameters():
            param.requires_grad = False
        
        # Configuration
        self.feature_layers = feature_layers
        self.layer_weights = dict(zip(feature_layers, layer_weights))
        self.use_contrastive = use_contrastive
        self.contrastive_margin = contrastive_margin
        self.contrastive_weight = contrastive_weight
        self.use_triplet = use_triplet
        self.triplet_margin = triplet_margin
        self.triplet_weight = triplet_weight
        self.temperature = temperature
        
        # Hook storage
        self.feature_maps = {}
        self._register_hooks()
        
        print(f"Initialized DiscriminativeMultiLevelFaceLoss:")
        print(f"  Layers: {feature_layers}")
        print(f"  Contrastive: {use_contrastive} (margin={contrastive_margin})")
        print(f"  Triplet: {use_triplet} (margin={triplet_margin})")
    
    def _register_hooks(self):
        """Register hooks to capture intermediate features"""
        def get_activation(name):
            def hook(module, input, output):
                self.feature_maps[name] = output
            return hook
        
        for name, module in self.recognizer.named_modules():
            for layer in self.feature_layers:
                if layer in name:
                    module.register_forward_hook(get_activation(name))
                    break
    
    def extract_multi_level_features(self, x):
        """
        Extract features from multiple layers
        
        Args:
            x: [B, 3, H, W] in [0, 1]
        Returns:
            dict: {layer_name: features}
        """
        # Resize to 112x112
        if x.shape[-2:] != (112, 112):
            x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
        
        # Normalize to [-1, 1]
        if x.min() >= 0:
            x = (x - 0.5) / 0.5
        
        # Clear previous features
        self.feature_maps = {}
        
        # Forward pass (hooks capture intermediate features)
        with torch.no_grad():
            final_feat = self.recognizer(x)
        
        # Add final feature if not already captured
        if 'fc' in self.feature_layers and 'fc' not in self.feature_maps:
            self.feature_maps['fc'] = final_feat
        
        return self.feature_maps.copy()
    
    def reconstruction_loss(self, enhanced_feats, gt_feats):
        """
        Multi-level feature reconstruction (similar to current approach)
        
        This is the "pull enhanced → GT" component
        """
        loss = 0.0
        count = 0
        
        for layer_name, weight in self.layer_weights.items():
            if layer_name not in enhanced_feats or layer_name not in gt_feats:
                continue
            
            # Normalize features before comparison
            enh_norm = F.normalize(enhanced_feats[layer_name].flatten(1), p=2, dim=1)
            gt_norm = F.normalize(gt_feats[layer_name].flatten(1), p=2, dim=1)
            
            # Cosine distance (1 - similarity)
            layer_loss = 1.0 - (enh_norm * gt_norm).sum(dim=1).mean()
            
            loss += weight * layer_loss
            count += weight
        
        return loss / count if count > 0 else loss
    
    def supervised_contrastive_loss(self, anchor_feats, positive_feats, negative_feats):
        """
        Supervised contrastive loss - the KEY discriminative component
        
        Goal: Pull genuine pairs together, push impostor pairs apart
        
        Args:
            anchor_feats: Enhanced image features
            positive_feats: Ground truth (same identity) features
            negative_feats: Different identity features [B, D]
        
        Returns:
            Contrastive loss value
        """
        if not self.use_contrastive or negative_feats is None:
            return torch.tensor(0.0).to(anchor_feats['fc'].device)
        
        # Use final layer features
        anchor = F.normalize(anchor_feats['fc'].flatten(1), p=2, dim=1)
        positive = F.normalize(positive_feats['fc'].flatten(1), p=2, dim=1)
        negative = F.normalize(negative_feats['fc'].flatten(1), p=2, dim=1)
        
        # Compute similarities
        pos_sim = (anchor * positive).sum(dim=1)  # [B]
        neg_sim = (anchor * negative).sum(dim=1)  # [B]
        
        # InfoNCE-style contrastive loss
        # Numerator: exp(sim(anchor, positive) / temp)
        # Denominator: exp(sim(anchor, positive) / temp) + exp(sim(anchor, negative) / temp)
        
        logits_pos = pos_sim / self.temperature
        logits_neg = neg_sim / self.temperature
        
        # Log-sum-exp for numerical stability
        max_logit = torch.max(logits_pos, logits_neg)
        exp_pos = torch.exp(logits_pos - max_logit)
        exp_neg = torch.exp(logits_neg - max_logit)
        
        # Contrastive loss: -log(exp_pos / (exp_pos + exp_neg))
        loss = -torch.log(exp_pos / (exp_pos + exp_neg + 1e-8))
        
        return loss.mean()
    
    def triplet_margin_loss(self, anchor_feats, positive_feats, negative_feats):
        """
        Triplet loss with margin
        
        Ensures: d(anchor, positive) + margin < d(anchor, negative)
        
        This creates a "buffer zone" between genuine and impostor pairs
        """
        if not self.use_triplet or negative_feats is None:
            return torch.tensor(0.0).to(anchor_feats['fc'].device)
        
        # Use final layer
        anchor = F.normalize(anchor_feats['fc'].flatten(1), p=2, dim=1)
        positive = F.normalize(positive_feats['fc'].flatten(1), p=2, dim=1)
        negative = F.normalize(negative_feats['fc'].flatten(1), p=2, dim=1)
        
        # Euclidean distances in normalized space
        pos_dist = (anchor - positive).pow(2).sum(dim=1).sqrt()
        neg_dist = (anchor - negative).pow(2).sum(dim=1).sqrt()
        
        # Triplet loss: max(0, pos_dist - neg_dist + margin)
        loss = F.relu(pos_dist - neg_dist + self.triplet_margin)
        
        return loss.mean()
    
    def forward(self, enhanced, ground_truth, impostor=None):
        """
        Complete discriminative loss computation
        
        Args:
            enhanced: Enhanced low-light images [B, 3, H, W]
            ground_truth: Ground truth images [B, 3, H, W]
            impostor: Different identity images [B, 3, H, W] (REQUIRED for discrimination)
        
        Returns:
            dict: {
                'total': total loss,
                'reconstruction': multi-level reconstruction,
                'contrastive': supervised contrastive loss,
                'triplet': triplet margin loss
            }
        """
        # Extract multi-level features
        enhanced_feats = self.extract_multi_level_features(enhanced)
        gt_feats = self.extract_multi_level_features(ground_truth)
        impostor_feats = self.extract_multi_level_features(impostor) if impostor is not None else None
        
        # Component losses
        loss_recon = self.reconstruction_loss(enhanced_feats, gt_feats)
        loss_contrastive = self.supervised_contrastive_loss(enhanced_feats, gt_feats, impostor_feats)
        loss_triplet = self.triplet_margin_loss(enhanced_feats, gt_feats, impostor_feats)
        
        # Total loss
        total = (loss_recon + 
                 self.contrastive_weight * loss_contrastive + 
                 self.triplet_weight * loss_triplet)
        
        return {
            'total': total,
            'reconstruction': loss_recon,
            'contrastive': loss_contrastive,
            'triplet': loss_triplet
        }
