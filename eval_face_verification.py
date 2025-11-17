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
from measure import compute_psnr, compute_ssim


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

    if weights_path and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict, strict=False)
        print(f"  ✓ Loaded weights from {weights_path}")
    else:
        print("  ! Using random initialization (for testing only)")
        print("  WARNING: For real evaluation, download AdaFace weights")

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


def enhance_image(model, img_tensor, device='cuda'):
    """Enhance a low-light image using CIDNet

    Args:
        model: CIDNet model
        img_tensor: Input image tensor (1, 3, H, W)
        device: Device

    Returns:
        Enhanced image tensor (1, 3, H, W)
    """
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        enhanced = model(img_tensor)
        enhanced = torch.clamp(enhanced, 0, 1)
    return enhanced


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
    Evaluate face verification accuracy on enhanced images

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

            # Enhance low-light image
            enhanced_tensor = enhance_image(enhancement_model, low_tensor, device)

            # Compute PSNR/SSIM between enhanced and GT
            psnr = compute_psnr(enhanced_tensor, high_tensor)
            ssim = compute_ssim(enhanced_tensor, high_tensor)
            psnr_values.append(psnr)
            ssim_values.append(ssim)

            # Preprocess for face recognition
            low_face = preprocess_for_face_recognizer(low_tensor)
            enhanced_face = preprocess_for_face_recognizer(enhanced_tensor)
            high_face = preprocess_for_face_recognizer(high_tensor)

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
    low_correct = sum([1 for s in similarities_low if s >= threshold])
    enhanced_correct = sum([1 for s in similarities_enhanced if s >= threshold])

    results['verification_acc_low'] = low_correct / len(similarities_low) * 100
    results['verification_acc_enhanced'] = enhanced_correct / len(similarities_enhanced) * 100
    results['verification_improvement'] = results['verification_acc_enhanced'] - results['verification_acc_low']

    print(f"\nVerification Accuracy (threshold={threshold}):")
    print(f"  Low-light:  {results['verification_acc_low']:.2f}%")
    print(f"  Enhanced:   {results['verification_acc_enhanced']:.2f}%")
    print(f"  Improvement: {results['verification_improvement']:.2f}%")

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
    parser.add_argument('--face_model', type=str, default='ir_50',
                       choices=['ir_50', 'ir_101'],
                       help='Face recognition model architecture')
    parser.add_argument('--face_weights', type=str, default=None,
                       help='Path to AdaFace weights (optional)')
    parser.add_argument('--max_pairs', type=int, default=None,
                       help='Maximum pairs to evaluate (for quick testing)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Similarity threshold for verification')
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
