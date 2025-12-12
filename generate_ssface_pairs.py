"""
Generate SS_Face Pairs for Face Verification Evaluation

This script is a convenience wrapper around generate_lfw_pairs.py
specifically for SS_Face dataset evaluation. It uses the same pair
generation logic (genuine + impostor pairs) but with SS_Face defaults.

SS_Face uses numeric IDs as identities (e.g., 1/, 10/, 12/), and the
pair generator automatically detects this subdirectory structure.

Usage:
    # Generate pairs for SS_Face (auto-detects test set)
    python generate_ssface_pairs.py

    # Custom number of pairs
    python generate_ssface_pairs.py --num_pairs 2000

    # Custom paths
    python generate_ssface_pairs.py --test_dir=./datasets/SS_Face_lowlight/test --output=./pairs_custom.txt
"""

import sys
import os

# Import the generic LFW pairs generator (works for any identity-based structure)
from generate_lfw_pairs import generate_pairs


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate SS_Face pairs for face verification (wrapper for generate_lfw_pairs.py)'
    )
    parser.add_argument('--test_dir', type=str, default='./datasets/SS_Face_lowlight/test',
                       help='SS_Face test directory containing low/ and high/ subdirectories')
    parser.add_argument('--num_pairs', type=int, default=2000,
                       help='Number of pairs of each type (genuine and impostor)')
    parser.add_argument('--output', type=str, default='./pairs_ssface.txt',
                       help='Output file path for pairs.txt')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    print("="*70)
    print("SS_Face Pairs Generation for Face Verification Evaluation")
    print("="*70)
    print(f"Test directory: {args.test_dir}")
    print(f"Pairs per type: {args.num_pairs}")
    print(f"Output file:    {args.output}")
    print(f"Random seed:    {args.seed}")
    print("="*70)
    print()
    print("Note: This uses the same logic as generate_lfw_pairs.py")
    print("      Identity subdirectories (e.g., 1/, 10/, 12/) are auto-detected")
    print()

    try:
        generate_pairs(
            test_dir=args.test_dir,
            num_pairs=args.num_pairs,
            output_file=args.output,
            seed=args.seed,
            min_images_per_person=1  # SS_Face may have single-image identities
        )
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print()
    print("="*70)
    print("✓ SS_Face pairs ready for evaluation!")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Run evaluation: bash DiscriminativeMultiLevelFaceLoss/evaluation_ssface.sh")
    print(f"  2. Or manually: python eval_face_verification.py --pairs_file={args.output} ...")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
