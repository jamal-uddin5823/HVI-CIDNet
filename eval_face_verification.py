"""
Face Verification Evaluation for Low-Light Enhancement

This script evaluates how well the enhanced images preserve facial identity
by measuring face verification accuracy on LFW pairs. This is a key metric
for assessing the effectiveness of face recognition-aware enhancement.

Metrics computed:
1. Face Verification Accuracy on LFW pairs
2. Feature Similarity (cosine similarity) before/after enhancement
3. Face Detection Success Rate
4. Image Quality Metrics (PSNR, SSIM)

Usage:
    # Evaluate a trained model on LFW test set
    python eval_face_verification.py --model weights/train/epoch_100.pth \\
                                      --test_dir datasets/LFW_lowlight/test \\
                                      --pairs_file datasets/LFW_lowlight/pairs.txt

    # Quick evaluation (first 100 pairs only)
    python eval_face_verification.py --model weights/train/epoch_100.pth \\
                                      --test_dir datasets/LFW_lowlight/test \\
                                      --max_pairs 100
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Import model and loss
from net.CIDNet import CIDNet
from loss.adaface_model import build_model as build_adaface
from measure import calculate_psnr, calculate_ssim


def compute_psnr(pred_tensor, target_tensor):
    """Compute PSNR between two tensors

    Args:
        pred_tensor: Predicted image tensor (B, C, H, W) in range [0, 1]
        target_tensor: Target image tensor (B, C, H, W) in range [0, 1]

    Returns:
        float: PSNR value
    """
    # Ensure same size by resizing pred to match target
    if pred_tensor.shape != target_tensor.shape:
        pred_tensor = F.interpolate(pred_tensor, size=target_tensor.shape[2:], mode='bilinear', align_corners=False)

    # Convert tensors to numpy arrays in range [0, 255]
    pred_np = (pred_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    target_np = (target_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Convert to PIL Images for calculate_psnr
    pred_img = Image.fromarray(pred_np)
    target_img = Image.fromarray(target_np)

    return calculate_psnr(pred_img, target_img)


def compute_ssim(pred_tensor, target_tensor):
    """Compute SSIM between two tensors

    Args:
        pred_tensor: Predicted image tensor (B, C, H, W) in range [0, 1]
        target_tensor: Target image tensor (B, C, H, W) in range [0, 1]

    Returns:
        float: SSIM value
    """
    # Ensure same size by resizing pred to match target
    if pred_tensor.shape != target_tensor.shape:
        pred_tensor = F.interpolate(pred_tensor, size=target_tensor.shape[2:], mode='bilinear', align_corners=False)

    # Convert tensors to numpy arrays in range [0, 255]
    pred_np = (pred_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    target_np = (target_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Convert to PIL Images for calculate_ssim
    pred_img = Image.fromarray(pred_np)
    target_img = Image.fromarray(target_np)

    return calculate_ssim(pred_img, target_img)


def load_enhancement_model(model_path, device='cuda'):
    """Load CIDNet enhancement model"""
    print(f"Loading enhancement model from {model_path}...")
    model = CIDNet().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("  ✓ Enhancement model loaded")
    return model


def load_face_recognition_model(arch='ir_50', weights_path=None, device='cuda'):
    """Load AdaFace face recognition model"""
    print(f"Loading face recognition model ({arch})...")
    model = build_adaface(arch).to(device)

    if weights_path:
        if os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location=device)
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                model.load_state_dict(state_dict, strict=False)
                print(f"  ✓ Loaded weights from {weights_path}")
            except Exception as e:
                print(f"  ✗ Error loading weights: {e}")
                print(f"  ! Using random initialization")
        else:
            print(f"  ✗ Weights file not found: {weights_path}")
            print(f"  ! Using random initialization (for testing only)")
            print(f"  WARNING: For real evaluation, download AdaFace weights")
    else:
        print("  ! No weights path provided, using random initialization")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


def preprocess_for_face_recognizer(img_tensor, size=112):
    """Preprocess image for AdaFace model

    Args:
        img_tensor: Image tensor in range [0, 1], shape (C, H, W) or (B, C, H, W)
        size: Target size for face recognition model

    Returns:
        Preprocessed tensor
    """
    # Resize
    if img_tensor.shape[-2:] != (size, size):
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0) if img_tensor.dim() == 3 else img_tensor,
            size=(size, size),
            mode='bilinear',
            align_corners=False
        )
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.squeeze(0)

    # Normalize to [-1, 1]
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(img_tensor.device)
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(img_tensor.device)
    img_tensor = (img_tensor - mean) / std

    return img_tensor


def compute_face_similarity(feat1, feat2):
    """Compute cosine similarity between face features

    Args:
        feat1: Face embeddings (B, D)
        feat2: Face embeddings (B, D)

    Returns:
        Cosine similarities (B,)
    """
    # Normalize features
    feat1 = F.normalize(feat1, p=2, dim=1)
    feat2 = F.normalize(feat2, p=2, dim=1)

    # Compute cosine similarity
    similarity = (feat1 * feat2).sum(dim=1)

    return similarity


def enhance_image(model, img_tensor, device='cuda', target_size=None):
    """Enhance a low-light image using CIDNet

    Args:
        model: CIDNet model
        img_tensor: Input image tensor (1, 3, H, W)
        device: Device
        target_size: Optional (H, W) to resize output to match

    Returns:
        Enhanced image tensor (1, 3, H, W)
    """
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        enhanced = model(img_tensor)
        enhanced = torch.clamp(enhanced, 0, 1)

        # Resize to target size if specified (to match GT dimensions)
        if target_size is not None and enhanced.shape[2:] != target_size:
            enhanced = F.interpolate(enhanced, size=target_size, mode='bilinear', align_corners=False)

    return enhanced


def load_pairs_file(pairs_file):
    """Load pairs from pairs.txt file

    Args:
        pairs_file: Path to pairs.txt

    Returns:
        list: List of (low_name, high_name, label) tuples
    """
    pairs = []
    with open(pairs_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) != 3:
                continue

            low_name, high_name, label = parts
            pairs.append((low_name, high_name, int(label)))

    return pairs


def evaluate_face_verification_with_pairs(
    enhancement_model,
    face_model,
    test_dir,
    pairs_file,
    device='cuda',
    max_pairs=None,
    save_results=True,
    output_dir='./results/face_verification'
):
    """
    Evaluate face verification accuracy using pairs.txt protocol

    This properly evaluates genuine pairs (same person) vs impostor pairs (different people)
    and computes standard verification metrics like TAR@FAR, EER, etc.

    Args:
        enhancement_model: CIDNet model
        face_model: Face recognition model
        test_dir: Directory with low/high subdirectories
        pairs_file: Path to pairs.txt file
        device: Device to use
        max_pairs: Maximum number of pairs to evaluate (None = all)
        save_results: Save detailed results to file
        output_dir: Output directory for results

    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "="*70)
    print("Face Verification Evaluation (Pairs Protocol)")
    print("="*70)

    # Load pairs
    print(f"\nLoading pairs from: {pairs_file}")
    pairs = load_pairs_file(pairs_file)

    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]

    print(f"Loaded {len(pairs)} pairs")

    # Separate genuine and impostor pairs
    genuine_pairs = [(l, h) for l, h, label in pairs if label == 1]
    impostor_pairs = [(l, h) for l, h, label in pairs if label == 0]

    print(f"  Genuine pairs (same person):  {len(genuine_pairs)}")
    print(f"  Impostor pairs (different):   {len(impostor_pairs)}")

    # Setup directories
    low_dir = os.path.join(test_dir, 'low')
    high_dir = os.path.join(test_dir, 'high')

    # Get file extension mapping
    low_files = {os.path.splitext(f)[0]: f for f in os.listdir(low_dir)
                 if f.endswith(('.png', '.jpg', '.jpeg'))}
    high_files = {os.path.splitext(f)[0]: f for f in os.listdir(high_dir)
                  if f.endswith(('.png', '.jpg', '.jpeg'))}

    # Storage for scores
    genuine_scores_low = []
    genuine_scores_enhanced = []
    impostor_scores_low = []
    impostor_scores_enhanced = []
    psnr_values = []
    ssim_values = []

    # Transform
    to_tensor = transforms.ToTensor()

    # Evaluate genuine pairs
    print("\nEvaluating genuine pairs (same person)...")
    for low_name, high_name in tqdm(genuine_pairs, desc="Genuine pairs"):
        try:
            # Get filenames
            low_file = low_files.get(low_name)
            high_file = high_files.get(high_name)

            if low_file is None or high_file is None:
                continue

            # Load images
            low_path = os.path.join(low_dir, low_file)
            high_path = os.path.join(high_dir, high_file)

            low_img = Image.open(low_path).convert('RGB')
            high_img = Image.open(high_path).convert('RGB')

            low_tensor = to_tensor(low_img).unsqueeze(0).to(device)
            high_tensor = to_tensor(high_img).unsqueeze(0).to(device)

            # Resize to dimensions that are multiples of 32 (prevents ResNet feature map misalignment)
            target_h = ((high_tensor.shape[2] + 31) // 32) * 32
            target_w = ((high_tensor.shape[3] + 31) // 32) * 32
            low_tensor = F.interpolate(low_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)
            high_tensor = F.interpolate(high_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)

            # Enhance (resize to match GT dimensions)
            enhanced_tensor = enhance_image(enhancement_model, low_tensor, device, target_size=high_tensor.shape[2:])

            # Compute PSNR/SSIM (only for genuine pairs where they match)
            if low_name == high_name:
                psnr = compute_psnr(enhanced_tensor, high_tensor)
                ssim = compute_ssim(enhanced_tensor, high_tensor)
                psnr_values.append(psnr)
                ssim_values.append(ssim)

            # Preprocess for face recognition
            low_face = preprocess_for_face_recognizer(low_tensor)
            enhanced_face = preprocess_for_face_recognizer(enhanced_tensor)
            high_face = preprocess_for_face_recognizer(high_tensor)

            # Extract features
            with torch.no_grad():
                feat_low = face_model(low_face)
                feat_enhanced = face_model(enhanced_face)
                feat_high = face_model(high_face)

                if feat_low.dim() > 2:
                    feat_low = feat_low.view(feat_low.size(0), -1)
                    feat_enhanced = feat_enhanced.view(feat_enhanced.size(0), -1)
                    feat_high = feat_high.view(feat_high.size(0), -1)

            # Compute similarities (genuine pair: should be high)
            sim_low = compute_face_similarity(feat_low, feat_high).item()
            sim_enhanced = compute_face_similarity(feat_enhanced, feat_high).item()

            genuine_scores_low.append(sim_low)
            genuine_scores_enhanced.append(sim_enhanced)

        except Exception as e:
            print(f"\n  Error processing genuine pair ({low_name}, {high_name}): {e}")
            continue

    # Evaluate impostor pairs
    print("\nEvaluating impostor pairs (different people)...")
    for low_name, high_name in tqdm(impostor_pairs, desc="Impostor pairs"):
        try:
            # Get filenames
            low_file = low_files.get(low_name)
            high_file = high_files.get(high_name)

            if low_file is None or high_file is None:
                continue

            # Load images
            low_path = os.path.join(low_dir, low_file)
            high_path = os.path.join(high_dir, high_file)

            low_img = Image.open(low_path).convert('RGB')
            high_img = Image.open(high_path).convert('RGB')

            low_tensor = to_tensor(low_img).unsqueeze(0).to(device)
            high_tensor = to_tensor(high_img).unsqueeze(0).to(device)

            # Resize to dimensions that are multiples of 32 (prevents ResNet feature map misalignment)
            target_h = ((high_tensor.shape[2] + 31) // 32) * 32
            target_w = ((high_tensor.shape[3] + 31) // 32) * 32
            low_tensor = F.interpolate(low_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)
            high_tensor = F.interpolate(high_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)

            # Enhance (resize to match GT dimensions)
            enhanced_tensor = enhance_image(enhancement_model, low_tensor, device, target_size=high_tensor.shape[2:])

            # Preprocess for face recognition
            low_face = preprocess_for_face_recognizer(low_tensor)
            enhanced_face = preprocess_for_face_recognizer(enhanced_tensor)
            high_face = preprocess_for_face_recognizer(high_tensor)

            # Extract features
            with torch.no_grad():
                feat_low = face_model(low_face)
                feat_enhanced = face_model(enhanced_face)
                feat_high = face_model(high_face)

                if feat_low.dim() > 2:
                    feat_low = feat_low.view(feat_low.size(0), -1)
                    feat_enhanced = feat_enhanced.view(feat_enhanced.size(0), -1)
                    feat_high = feat_high.view(feat_high.size(0), -1)

            # Compute similarities (impostor pair: should be low)
            sim_low = compute_face_similarity(feat_low, feat_high).item()
            sim_enhanced = compute_face_similarity(feat_enhanced, feat_high).item()

            impostor_scores_low.append(sim_low)
            impostor_scores_enhanced.append(sim_enhanced)

        except Exception as e:
            print(f"\n  Error processing impostor pair ({low_name}, {high_name}): {e}")
            continue

    # Compute verification metrics
    print("\n" + "="*70)
    print("Verification Metrics")
    print("="*70)

    # Find optimal threshold using EER
    all_scores_low = genuine_scores_low + impostor_scores_low
    all_labels = [1] * len(genuine_scores_low) + [0] * len(impostor_scores_low)

    all_scores_enhanced = genuine_scores_enhanced + impostor_scores_enhanced

    # Compute metrics at various thresholds
    thresholds = np.linspace(0, 1, 100)
    tar_low_list = []
    far_low_list = []
    tar_enhanced_list = []
    far_enhanced_list = []

    for thresh in thresholds:
        # Low-light
        tp_low = sum([1 for s in genuine_scores_low if s >= thresh])
        fn_low = len(genuine_scores_low) - tp_low
        fp_low = sum([1 for s in impostor_scores_low if s >= thresh])
        tn_low = len(impostor_scores_low) - fp_low

        tar_low = tp_low / len(genuine_scores_low) if len(genuine_scores_low) > 0 else 0
        far_low = fp_low / len(impostor_scores_low) if len(impostor_scores_low) > 0 else 0
        tar_low_list.append(tar_low)
        far_low_list.append(far_low)

        # Enhanced
        tp_enh = sum([1 for s in genuine_scores_enhanced if s >= thresh])
        fn_enh = len(genuine_scores_enhanced) - tp_enh
        fp_enh = sum([1 for s in impostor_scores_enhanced if s >= thresh])
        tn_enh = len(impostor_scores_enhanced) - fp_enh

        tar_enh = tp_enh / len(genuine_scores_enhanced) if len(genuine_scores_enhanced) > 0 else 0
        far_enh = fp_enh / len(impostor_scores_enhanced) if len(impostor_scores_enhanced) > 0 else 0
        tar_enhanced_list.append(tar_enh)
        far_enhanced_list.append(far_enh)

    # Find EER (where FRR = FAR, or TAR = 1 - FAR)
    eer_idx_low = np.argmin(np.abs(np.array(tar_low_list) - (1 - np.array(far_low_list))))
    eer_low = (far_low_list[eer_idx_low] + (1 - tar_low_list[eer_idx_low])) / 2
    eer_thresh_low = thresholds[eer_idx_low]

    eer_idx_enh = np.argmin(np.abs(np.array(tar_enhanced_list) - (1 - np.array(far_enhanced_list))))
    eer_enhanced = (far_enhanced_list[eer_idx_enh] + (1 - tar_enhanced_list[eer_idx_enh])) / 2
    eer_thresh_enh = thresholds[eer_idx_enh]

    # Find TAR @ FAR = 0.1%, 1%
    tar_at_far_001_low = tar_low_list[np.argmin(np.abs(np.array(far_low_list) - 0.001))]
    tar_at_far_01_low = tar_low_list[np.argmin(np.abs(np.array(far_low_list) - 0.01))]

    tar_at_far_001_enh = tar_enhanced_list[np.argmin(np.abs(np.array(far_enhanced_list) - 0.001))]
    tar_at_far_01_enh = tar_enhanced_list[np.argmin(np.abs(np.array(far_enhanced_list) - 0.01))]

    # Compile results
    results = {
        'num_genuine': len(genuine_scores_low),
        'num_impostor': len(impostor_scores_low),
        'genuine_mean_low': np.mean(genuine_scores_low),
        'genuine_std_low': np.std(genuine_scores_low),
        'genuine_mean_enhanced': np.mean(genuine_scores_enhanced),
        'genuine_std_enhanced': np.std(genuine_scores_enhanced),
        'impostor_mean_low': np.mean(impostor_scores_low),
        'impostor_std_low': np.std(impostor_scores_low),
        'impostor_mean_enhanced': np.mean(impostor_scores_enhanced),
        'impostor_std_enhanced': np.std(impostor_scores_enhanced),
        'eer_low': eer_low,
        'eer_threshold_low': eer_thresh_low,
        'eer_enhanced': eer_enhanced,
        'eer_threshold_enhanced': eer_thresh_enh,
        'tar_at_far_0.1%_low': tar_at_far_001_low,
        'tar_at_far_1%_low': tar_at_far_01_low,
        'tar_at_far_0.1%_enhanced': tar_at_far_001_enh,
        'tar_at_far_1%_enhanced': tar_at_far_01_enh,
        'psnr_mean': np.mean(psnr_values) if psnr_values else 0,
        'ssim_mean': np.mean(ssim_values) if ssim_values else 0,
    }

    # Print results
    print(f"\nGenuine Pair Scores (same person - should be HIGH):")
    print(f"  Low-light:  {results['genuine_mean_low']:.4f} ± {results['genuine_std_low']:.4f}")
    print(f"  Enhanced:   {results['genuine_mean_enhanced']:.4f} ± {results['genuine_std_enhanced']:.4f}")
    print(f"  Improvement: {results['genuine_mean_enhanced'] - results['genuine_mean_low']:.4f}")

    print(f"\nImpostor Pair Scores (different people - should be LOW):")
    print(f"  Low-light:  {results['impostor_mean_low']:.4f} ± {results['impostor_std_low']:.4f}")
    print(f"  Enhanced:   {results['impostor_mean_enhanced']:.4f} ± {results['impostor_std_enhanced']:.4f}")

    print(f"\nVerification Performance:")
    print(f"  Equal Error Rate (EER):")
    print(f"    Low-light:  {eer_low*100:.2f}% (threshold={eer_thresh_low:.3f})")
    print(f"    Enhanced:   {eer_enhanced*100:.2f}% (threshold={eer_thresh_enh:.3f})")
    print(f"    Improvement: {(eer_low - eer_enhanced)*100:.2f}%")

    print(f"\n  True Accept Rate (TAR) @ FAR=0.1%:")
    print(f"    Low-light:  {tar_at_far_001_low*100:.2f}%")
    print(f"    Enhanced:   {tar_at_far_001_enh*100:.2f}%")
    print(f"    Improvement: {(tar_at_far_001_enh - tar_at_far_001_low)*100:.2f}%")

    print(f"\n  True Accept Rate (TAR) @ FAR=1%:")
    print(f"    Low-light:  {tar_at_far_01_low*100:.2f}%")
    print(f"    Enhanced:   {tar_at_far_01_enh*100:.2f}%")
    print(f"    Improvement: {(tar_at_far_01_enh - tar_at_far_01_low)*100:.2f}%")

    if psnr_values:
        print(f"\nImage Quality:")
        print(f"  PSNR: {results['psnr_mean']:.2f} dB")
        print(f"  SSIM: {results['ssim_mean']:.4f}")

    # Save results
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, 'face_verification_results.txt')

        with open(results_file, 'w') as f:
            f.write("Face Verification Evaluation Results (Pairs Protocol)\n")
            f.write("="*70 + "\n\n")
            f.write(f"Pairs file: {pairs_file}\n")
            f.write(f"Genuine pairs: {results['num_genuine']}\n")
            f.write(f"Impostor pairs: {results['num_impostor']}\n\n")

            f.write("Genuine Pair Scores (same person):\n")
            f.write(f"  Low-light avg similarity:  {results['genuine_mean_low']:.4f} ± {results['genuine_std_low']:.4f}\n")
            f.write(f"  Enhanced avg similarity:   {results['genuine_mean_enhanced']:.4f} ± {results['genuine_std_enhanced']:.4f}\n")
            f.write(f"  Similarity improvement:    {results['genuine_mean_enhanced'] - results['genuine_mean_low']:.4f}\n\n")

            f.write("Impostor Pair Scores (different people):\n")
            f.write(f"  Low-light avg similarity:  {results['impostor_mean_low']:.4f} ± {results['impostor_std_low']:.4f}\n")
            f.write(f"  Enhanced avg similarity:   {results['impostor_mean_enhanced']:.4f} ± {results['impostor_std_enhanced']:.4f}\n\n")

            f.write("Verification Performance:\n")
            f.write(f"  Equal Error Rate (EER):\n")
            f.write(f"    Low-light:  {eer_low*100:.2f}% (threshold={eer_thresh_low:.3f})\n")
            f.write(f"    Enhanced:   {eer_enhanced*100:.2f}% (threshold={eer_thresh_enh:.3f})\n")
            f.write(f"    Improvement: {(eer_low - eer_enhanced)*100:.2f}%\n\n")

            f.write(f"  True Accept Rate @ FAR=0.1%:\n")
            f.write(f"    Low-light:  {tar_at_far_001_low*100:.2f}%\n")
            f.write(f"    Enhanced:   {tar_at_far_001_enh*100:.2f}%\n")
            f.write(f"    Improvement: {(tar_at_far_001_enh - tar_at_far_001_low)*100:.2f}%\n\n")

            f.write(f"  True Accept Rate @ FAR=1%:\n")
            f.write(f"    Low-light:  {tar_at_far_01_low*100:.2f}%\n")
            f.write(f"    Enhanced:   {tar_at_far_01_enh*100:.2f}%\n")
            f.write(f"    Improvement: {(tar_at_far_01_enh - tar_at_far_01_low)*100:.2f}%\n\n")

            if psnr_values:
                f.write("Image Quality:\n")
                f.write(f"  Average PSNR: {results['psnr_mean']:.2f} dB\n")
                f.write(f"  Average SSIM: {results['ssim_mean']:.4f}\n")

        print(f"\nResults saved to: {results_file}")

    return results


def evaluate_face_verification(
    enhancement_model,
    face_model,
    test_dir,
    device='cuda',
    max_pairs=None,
    threshold=0.5,
    save_results=True,
    output_dir='./results/face_verification'
):
    """
    Legacy evaluation: Evaluate face verification accuracy on enhanced images
    (Only uses same-person pairs - not proper verification)

    Args:
        enhancement_model: CIDNet model
        face_model: Face recognition model
        test_dir: Directory with low/high subdirectories
        device: Device to use
        max_pairs: Maximum number of pairs to evaluate (None = all)
        threshold: Similarity threshold for verification
        save_results: Save detailed results to file
        output_dir: Output directory for results

    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "="*70)
    print("Face Verification Evaluation")
    print("="*70)

    # Setup directories
    low_dir = os.path.join(test_dir, 'low')
    high_dir = os.path.join(test_dir, 'high')

    if not os.path.exists(low_dir) or not os.path.exists(high_dir):
        raise FileNotFoundError(f"Test directory structure invalid: {test_dir}")

    # Get all image files
    low_files = sorted([f for f in os.listdir(low_dir)
                       if f.endswith(('.png', '.jpg', '.jpeg'))])
    high_files = sorted([f for f in os.listdir(high_dir)
                        if f.endswith(('.png', '.jpg', '.jpeg'))])

    # Limit number of pairs if specified
    if max_pairs is not None:
        low_files = low_files[:max_pairs]
        high_files = high_files[:max_pairs]

    num_pairs = min(len(low_files), len(high_files))
    print(f"\nEvaluating {num_pairs} image pairs...")

    # Metrics storage
    similarities_low = []
    similarities_enhanced = []
    psnr_values = []
    ssim_values = []

    # Transform for loading images
    to_tensor = transforms.ToTensor()

    # Evaluate each pair
    for idx in tqdm(range(num_pairs), desc="Evaluating"):
        try:
            # Load low-light image
            low_path = os.path.join(low_dir, low_files[idx])
            low_img = Image.open(low_path).convert('RGB')
            low_tensor = to_tensor(low_img).unsqueeze(0).to(device)

            # Load ground truth (high-quality) image
            high_path = os.path.join(high_dir, high_files[idx])
            high_img = Image.open(high_path).convert('RGB')
            high_tensor = to_tensor(high_img).unsqueeze(0).to(device)

            # Debug: Check if sizes match
            if idx == 0:  # Only print for first image
                print(f"\nDebug - Original image sizes:")
                print(f"  Low:  {low_tensor.shape}")
                print(f"  High: {high_tensor.shape}")

            # CRITICAL FIX: Resize to dimensions that are multiples of 32
            # This ensures consistent feature map sizes in ResNet layers (stride-2 operations)
            # 125x94 -> 128x96 (nearest multiples of 32)
            target_h = ((high_tensor.shape[2] + 31) // 32) * 32
            target_w = ((high_tensor.shape[3] + 31) // 32) * 32

            if idx == 0:
                print(f"  Resizing to {target_h}x{target_w} (multiples of 32)")

            low_tensor = F.interpolate(low_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)
            high_tensor = F.interpolate(high_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)

            # Enhance low-light image (resize to match GT dimensions)
            enhanced_tensor = enhance_image(enhancement_model, low_tensor, device, target_size=high_tensor.shape[2:])

            # Debug: Verify all sizes match
            if idx == 0:
                print(f"  Enhanced: {enhanced_tensor.shape}")

            # Compute PSNR/SSIM between enhanced and GT
            psnr = compute_psnr(enhanced_tensor, high_tensor)
            ssim = compute_ssim(enhanced_tensor, high_tensor)
            psnr_values.append(psnr)
            ssim_values.append(ssim)

            # Preprocess for face recognition
            low_face = preprocess_for_face_recognizer(low_tensor)
            enhanced_face = preprocess_for_face_recognizer(enhanced_tensor)
            high_face = preprocess_for_face_recognizer(high_tensor)

            # Debug: Check preprocessed sizes
            if idx == 0:
                print(f"\nDebug - Preprocessed for AdaFace:")
                print(f"  Low face:  {low_face.shape}")
                print(f"  Enh face:  {enhanced_face.shape}")
                print(f"  High face: {high_face.shape}")

            # Extract face features
            with torch.no_grad():
                feat_low = face_model(low_face)
                feat_enhanced = face_model(enhanced_face)
                feat_high = face_model(high_face)

                # Flatten if needed
                if feat_low.dim() > 2:
                    feat_low = feat_low.view(feat_low.size(0), -1)
                    feat_enhanced = feat_enhanced.view(feat_enhanced.size(0), -1)
                    feat_high = feat_high.view(feat_high.size(0), -1)

            # Compute similarities
            sim_low = compute_face_similarity(feat_low, feat_high).item()
            sim_enhanced = compute_face_similarity(feat_enhanced, feat_high).item()

            similarities_low.append(sim_low)
            similarities_enhanced.append(sim_enhanced)

        except Exception as e:
            print(f"\n  Error processing pair {idx}: {e}")
            continue

    # Compute aggregate metrics
    print("\n" + "="*70)
    print("Evaluation Results")
    print("="*70)

    results = {
        'num_pairs': len(similarities_low),
        'similarity_low_mean': np.mean(similarities_low),
        'similarity_low_std': np.std(similarities_low),
        'similarity_enhanced_mean': np.mean(similarities_enhanced),
        'similarity_enhanced_std': np.std(similarities_enhanced),
        'similarity_improvement': np.mean(similarities_enhanced) - np.mean(similarities_low),
        'psnr_mean': np.mean(psnr_values),
        'psnr_std': np.std(psnr_values),
        'ssim_mean': np.mean(ssim_values),
        'ssim_std': np.std(ssim_values),
    }

    # Print results
    print(f"\nImage Quality Metrics:")
    print(f"  PSNR: {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
    print(f"  SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")

    print(f"\nFace Similarity Metrics:")
    print(f"  Low-light → GT:  {results['similarity_low_mean']:.4f} ± {results['similarity_low_std']:.4f}")
    print(f"  Enhanced → GT:   {results['similarity_enhanced_mean']:.4f} ± {results['similarity_enhanced_std']:.4f}")
    print(f"  Improvement:     {results['similarity_improvement']:.4f}")

    # Verification accuracy (using threshold)
    if len(similarities_low) > 0 and len(similarities_enhanced) > 0:
        low_correct = sum([1 for s in similarities_low if s >= threshold])
        enhanced_correct = sum([1 for s in similarities_enhanced if s >= threshold])

        results['verification_acc_low'] = low_correct / len(similarities_low) * 100
        results['verification_acc_enhanced'] = enhanced_correct / len(similarities_enhanced) * 100
        results['verification_improvement'] = results['verification_acc_enhanced'] - results['verification_acc_low']

        print(f"\nVerification Accuracy (threshold={threshold}):")
        print(f"  Low-light:  {results['verification_acc_low']:.2f}%")
        print(f"  Enhanced:   {results['verification_acc_enhanced']:.2f}%")
        print(f"  Improvement: {results['verification_improvement']:.2f}%")
    else:
        print(f"\n⚠ Warning: No valid similarity scores computed")
        print(f"  All image pairs failed processing - check error messages above")
        results['verification_acc_low'] = 0.0
        results['verification_acc_enhanced'] = 0.0
        results['verification_improvement'] = 0.0

    # Save results
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, 'face_verification_results.txt')

        with open(results_file, 'w') as f:
            f.write("Face Verification Evaluation Results\n")
            f.write("="*70 + "\n\n")
            f.write(f"Test directory: {test_dir}\n")
            f.write(f"Number of pairs: {results['num_pairs']}\n\n")

            f.write("Image Quality Metrics:\n")
            f.write(f"  PSNR: {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB\n")
            f.write(f"  SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}\n\n")

            f.write("Face Similarity Metrics:\n")
            f.write(f"  Low-light → GT:  {results['similarity_low_mean']:.4f} ± {results['similarity_low_std']:.4f}\n")
            f.write(f"  Enhanced → GT:   {results['similarity_enhanced_mean']:.4f} ± {results['similarity_enhanced_std']:.4f}\n")
            f.write(f"  Improvement:     {results['similarity_improvement']:.4f}\n\n")

            f.write(f"Verification Accuracy (threshold={threshold}):\n")
            f.write(f"  Low-light:  {results['verification_acc_low']:.2f}%\n")
            f.write(f"  Enhanced:   {results['verification_acc_enhanced']:.2f}%\n")
            f.write(f"  Improvement: {results['verification_improvement']:.2f}%\n")

        print(f"\nResults saved to: {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate face verification on enhanced images'
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained CIDNet model')
    parser.add_argument('--test_dir', type=str, required=True,
                       help='Test directory with low/high subdirectories')
    parser.add_argument('--pairs_file', type=str, default=None,
                       help='Path to pairs.txt for proper verification evaluation (recommended)')
    parser.add_argument('--face_model', type=str, default='ir_50',
                       choices=['ir_50', 'ir_101'],
                       help='Face recognition model architecture')
    parser.add_argument('--face_weights', type=str, default=None,
                       help='Path to AdaFace weights (optional)')
    parser.add_argument('--max_pairs', type=int, default=None,
                       help='Maximum pairs to evaluate (for quick testing)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Similarity threshold for verification (legacy mode only)')
    parser.add_argument('--output_dir', type=str, default='./results/face_verification',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'

    # Load models
    enhancement_model = load_enhancement_model(args.model, args.device)
    face_model = load_face_recognition_model(
        args.face_model,
        args.face_weights,
        args.device
    )

    # Run evaluation
    if args.pairs_file is not None:
        print("\n[Using pairs-based verification protocol]")
        print("This will properly evaluate genuine vs. impostor pairs")
        results = evaluate_face_verification_with_pairs(
            enhancement_model,
            face_model,
            args.test_dir,
            args.pairs_file,
            device=args.device,
            max_pairs=args.max_pairs,
            save_results=True,
            output_dir=args.output_dir
        )
    else:
        print("\n[Using legacy evaluation mode]")
        print("⚠ Warning: This only tests same-person pairs")
        print("  For proper verification metrics, use --pairs_file")
        print("  Generate pairs with: python generate_lfw_pairs.py")
        results = evaluate_face_verification(
            enhancement_model,
            face_model,
            args.test_dir,
            device=args.device,
            max_pairs=args.max_pairs,
            threshold=args.threshold,
            save_results=True,
            output_dir=args.output_dir
        )

    print("\n" + "="*70)
    print("Evaluation complete!")
    print("="*70)


if __name__ == '__main__':
    main()
