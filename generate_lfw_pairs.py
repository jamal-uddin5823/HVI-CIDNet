"""
Generate LFW Pairs for Face Verification Evaluation

This script generates pairs.txt following the standard LFW verification protocol:
- Genuine pairs (same person): Enhanced low-light image vs. ground truth
- Impostor pairs (different people): Enhanced image of person A vs. GT of person B

This enables proper face verification evaluation to measure:
1. True Accept Rate (TAR): % of genuine pairs correctly verified
2. False Accept Rate (FAR): % of impostor pairs incorrectly accepted
3. Verification Accuracy at various thresholds
4. Equal Error Rate (EER): Where TAR = 1 - FAR

Usage:
    # Generate 1000 genuine + 1000 impostor pairs
    python generate_lfw_pairs.py --test_dir=./datasets/LFW_lowlight/test --num_pairs=1000

    # Custom output location
    python generate_lfw_pairs.py --test_dir=./datasets/LFW_small/test --output=./pairs_small.txt
"""

import os
import argparse
import random
from pathlib import Path


def generate_pairs(test_dir, num_pairs=1000, output_file='pairs.txt', seed=42):
    """
    Generate genuine and impostor pairs for face verification evaluation.

    Args:
        test_dir: Directory containing test/low and test/high subdirectories
        num_pairs: Number of pairs of each type (genuine and impostor)
        output_file: Output file path for pairs.txt
        seed: Random seed for reproducibility

    Returns:
        tuple: (num_genuine, num_impostor) pairs generated
    """
    random.seed(seed)

    low_dir = os.path.join(test_dir, 'low')
    high_dir = os.path.join(test_dir, 'high')

    # Check if directories exist
    if not os.path.exists(low_dir):
        raise FileNotFoundError(f"Low-light directory not found: {low_dir}")
    if not os.path.exists(high_dir):
        raise FileNotFoundError(f"High-quality directory not found: {high_dir}")

    # Get all image filenames (without extension)
    low_files = sorted([f for f in os.listdir(low_dir)
                       if f.endswith(('.png', '.jpg', '.jpeg'))])
    high_files = sorted([f for f in os.listdir(high_dir)
                        if f.endswith(('.png', '.jpg', '.jpeg'))])

    print(f"Found {len(low_files)} low-light images")
    print(f"Found {len(high_files)} high-quality images")

    # Verify matching files
    if len(low_files) != len(high_files):
        print("⚠ Warning: Number of low and high images don't match")

    # Get base names (assuming matching filenames)
    image_names = [os.path.splitext(f)[0] for f in low_files]

    if len(image_names) < 2:
        raise ValueError("Need at least 2 images to generate pairs")

    # Adjust num_pairs if not enough images
    max_genuine_pairs = len(image_names)
    max_impostor_pairs = len(image_names) * (len(image_names) - 1) // 2

    if num_pairs > max_genuine_pairs:
        print(f"⚠ Warning: Requested {num_pairs} genuine pairs, but only {max_genuine_pairs} images available")
        num_pairs = max_genuine_pairs

    if num_pairs > max_impostor_pairs:
        print(f"⚠ Warning: Requested {num_pairs} impostor pairs, but only {max_impostor_pairs} combinations possible")
        num_pairs = min(num_pairs, max_impostor_pairs)

    # Generate genuine pairs (same person: low-light enhanced vs. ground truth)
    print(f"\nGenerating {num_pairs} genuine pairs (same person)...")
    genuine_pairs = []
    sampled_indices = random.sample(range(len(image_names)), num_pairs)

    for idx in sampled_indices:
        # Format: low_image_name high_image_name label
        genuine_pairs.append((image_names[idx], image_names[idx], 1))

    print(f"  ✓ Generated {len(genuine_pairs)} genuine pairs")

    # Generate impostor pairs (different people)
    print(f"Generating {num_pairs} impostor pairs (different people)...")
    impostor_pairs = []
    attempts = 0
    max_attempts = num_pairs * 10  # Avoid infinite loop

    while len(impostor_pairs) < num_pairs and attempts < max_attempts:
        idx1, idx2 = random.sample(range(len(image_names)), 2)
        pair = (image_names[idx1], image_names[idx2], 0)

        # Avoid duplicates
        if pair not in impostor_pairs:
            impostor_pairs.append(pair)

        attempts += 1

    print(f"  ✓ Generated {len(impostor_pairs)} impostor pairs")

    # Combine and shuffle
    all_pairs = genuine_pairs + impostor_pairs
    random.shuffle(all_pairs)

    # Write to file
    print(f"\nWriting pairs to: {output_file}")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    with open(output_file, 'w') as f:
        # Write header
        f.write(f"# LFW Face Verification Pairs\n")
        f.write(f"# Format: low_image_name high_image_name label\n")
        f.write(f"#   label=1: genuine pair (same person)\n")
        f.write(f"#   label=0: impostor pair (different people)\n")
        f.write(f"# Total pairs: {len(all_pairs)} ({len(genuine_pairs)} genuine + {len(impostor_pairs)} impostor)\n")
        f.write(f"#\n")

        # Write pairs
        for low_name, high_name, label in all_pairs:
            f.write(f"{low_name} {high_name} {label}\n")

    print(f"  ✓ Wrote {len(all_pairs)} pairs")
    print(f"\nSummary:")
    print(f"  Genuine pairs (same person):     {len(genuine_pairs)}")
    print(f"  Impostor pairs (different people): {len(impostor_pairs)}")
    print(f"  Total pairs:                      {len(all_pairs)}")
    print(f"\nPairs file ready for evaluation!")

    return len(genuine_pairs), len(impostor_pairs)


def main():
    parser = argparse.ArgumentParser(description='Generate LFW pairs for face verification')
    parser.add_argument('--test_dir', type=str, default='./datasets/LFW_lowlight/test',
                       help='Test directory containing low/ and high/ subdirectories')
    parser.add_argument('--num_pairs', type=int, default=1000,
                       help='Number of pairs of each type (genuine and impostor)')
    parser.add_argument('--output', type=str, default='pairs.txt',
                       help='Output file path for pairs.txt')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    print("="*70)
    print("LFW Pairs Generation for Face Verification Evaluation")
    print("="*70)
    print(f"Test directory: {args.test_dir}")
    print(f"Pairs per type: {args.num_pairs}")
    print(f"Output file:    {args.output}")
    print(f"Random seed:    {args.seed}")
    print("="*70)
    print()

    try:
        generate_pairs(
            test_dir=args.test_dir,
            num_pairs=args.num_pairs,
            output_file=args.output,
            seed=args.seed
        )
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
