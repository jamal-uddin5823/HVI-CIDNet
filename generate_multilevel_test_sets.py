"""
Generate Multi-Level Low-Light Test Sets for Face Recognition

This script creates test sets with multiple difficulty levels (easy, medium, hard)
to evaluate the generalization capability of face recognition-aware low-light
enhancement models.

Each difficulty level represents physically-accurate low-light conditions:
- Easy: Clean degradation (1% light, no noise) - ceiling performance baseline
- Medium: Moderate darkness with minimal sensor noise (5% light, Poisson-Gaussian noise)
- Hard: Challenging realistic conditions (10% light, higher noise, white balance shift)

Usage:
    python datasets/generate_multilevel_test_sets.py \
        --source_test_dir=./datasets/LFW_lowlight/test \
        --output_base_dir=./datasets/LFW_multilevel \
        --num_pairs=1000
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.lowlight_synthesis import synthesize_low_light_image
from generate_lfw_pairs import generate_pairs


# ============================================================================
# Difficulty Level Configurations
# ============================================================================

DIFFICULTY_LEVELS = {
    'easy': {
        'reduction_factor': 0.01,
        'apply_noise': False,
        'shot_noise': 1.0,
        'read_noise': 0.005,
        'gain': 1.5,
        'apply_white_balance': False,
        'wb_variation': 0.1,
        'apply_blur': False,
        'raw_sensor_mode': True
    },
    'medium': {
        'reduction_factor': 0.05,
        'apply_noise': True,
        'shot_noise': 1.0,
        'read_noise': 0.005,
        'gain': 1.5,
        'apply_white_balance': False,
        'wb_variation': 0.1,
        'apply_blur': False,
        'raw_sensor_mode': True
    },
    'hard': {
        'reduction_factor': 0.10,
        'apply_noise': True,
        'shot_noise': 2.0,
        'read_noise': 0.015,
        'gain': 3.0,
        'apply_white_balance': True,
        'wb_variation': 0.1,
        'apply_blur': False,
        'raw_sensor_mode': True
    }
}


# ============================================================================
# Main Generation Function
# ============================================================================

def generate_multilevel_test_sets(
    source_test_dir: str,
    output_base_dir: str,
    num_pairs: int = 1000,
    seed: int = 42
):
    """
    Generate multi-level test sets with varying difficulty.

    Args:
        source_test_dir: Path to source test set with high/ subdirectories
        output_base_dir: Base output directory for multi-level test sets
        num_pairs: Number of pairs per difficulty level
        seed: Random seed for reproducibility

    Creates:
        - output_base_dir/test_easy/
        - output_base_dir/test_medium/
        - output_base_dir/test_hard/

        Each with low/, high/ subdirectories and pairs.txt
    """
    import numpy as np
    np.random.seed(seed)

    print("=" * 70)
    print("Multi-Level Low-Light Test Set Generation")
    print("=" * 70)
    print(f"Source:      {source_test_dir}")
    print(f"Output:      {output_base_dir}")
    print(f"Pairs/level: {num_pairs}")
    print(f"Seed:        {seed}")
    print("=" * 70)
    print()

    # Verify source directory exists
    source_high_dir = os.path.join(source_test_dir, 'high')
    if not os.path.exists(source_high_dir):
        raise FileNotFoundError(f"Source high directory not found: {source_high_dir}")

    # Scan source images
    print("[Step 1/3] Scanning source images...")
    person_images = {}
    for person_name in sorted(os.listdir(source_high_dir)):
        person_dir = os.path.join(source_high_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        images = sorted([f for f in os.listdir(person_dir)
                        if f.endswith(('.png', '.jpg', '.jpeg'))])
        if images:
            person_images[person_name] = images

    total_images = sum(len(imgs) for imgs in person_images.values())
    print(f"  Found {total_images} images from {len(person_images)} people")
    print()

    # Generate each difficulty level
    for level_name, level_params in DIFFICULTY_LEVELS.items():
        print(f"[Step 2/3] Generating {level_name.upper()} test set...")
        print(f"  Reduction factor: {level_params['reduction_factor']}")
        print(f"  Apply noise:       {level_params['apply_noise']}")
        print(f"  Apply WB:          {level_params['apply_white_balance']}")

        # Create output directories
        output_test_dir = os.path.join(output_base_dir, f'test_{level_name}')
        output_low_dir = os.path.join(output_test_dir, 'low')
        output_high_dir = os.path.join(output_test_dir, 'high')

        os.makedirs(output_low_dir, exist_ok=True)
        os.makedirs(output_high_dir, exist_ok=True)

        # Process each person
        person_count = 0
        image_count = 0

        for person_name, images in tqdm(person_images.items(),
                                       desc=f"  {level_name.capitalize()}"):
            person_low_dir = os.path.join(output_low_dir, person_name)
            person_high_dir = os.path.join(output_high_dir, person_name)
            os.makedirs(person_low_dir, exist_ok=True)
            os.makedirs(person_high_dir, exist_ok=True)
            person_count += 1

            for img_filename in images:
                # Source image path
                src_path = os.path.join(source_high_dir, person_name, img_filename)

                # Destination paths
                img_basename = os.path.splitext(img_filename)[0]
                img_ext = '.png'  # Always save as PNG
                dest_high_path = os.path.join(person_high_dir, img_basename + img_ext)
                dest_low_path = os.path.join(person_low_dir, img_basename + img_ext)

                # Copy high-quality ground truth
                shutil.copy2(src_path, dest_high_path)

                # Generate low-light version with level-specific parameters
                from PIL import Image
                import numpy as np

                img = Image.open(src_path).convert('RGB')
                img_array = np.array(img).astype(np.float32) / 255.0

                # Use deterministic seed for reproducibility
                img_seed = seed + hash(f"{person_name}_{img_filename}") % 1000000

                low_light_array = synthesize_low_light_image(
                    img_array,
                    apply_light_reduction=True,
                    apply_noise=level_params['apply_noise'],
                    apply_white_balance=level_params['apply_white_balance'],
                    apply_blur=level_params['apply_blur'],
                    reduction_factor=level_params['reduction_factor'],
                    shot_noise_scale=level_params['shot_noise'],
                    read_noise_std=level_params['read_noise'],
                    gain=level_params['gain'],
                    wb_variation=level_params['wb_variation'],
                    raw_sensor_mode=level_params['raw_sensor_mode'],
                    seed=img_seed,
                    output_format='numpy'
                )

                # Save low-light image
                low_light_img = (low_light_array * 255).astype(np.uint8)
                Image.fromarray(low_light_img).save(dest_low_path)
                image_count += 1

        print(f"  ✓ Processed {image_count} images from {person_count} people")
        print()

    # Generate pairs for each level
    print("[Step 3/3] Generating verification pairs...")
    for level_name in DIFFICULTY_LEVELS.keys():
        output_test_dir = os.path.join(output_base_dir, f'test_{level_name}')
        pairs_file = os.path.join(output_test_dir, 'pairs.txt')

        print(f"  Generating pairs for {level_name}...")
        num_genuine, num_impostor = generate_pairs(
            test_dir=output_test_dir,
            num_pairs=num_pairs,
            output_file=pairs_file,
            seed=seed
        )
        print(f"    ✓ {num_genuine} genuine + {num_impostor} impostor pairs")

    print()
    print("=" * 70)
    print("Multi-level test set generation complete!")
    print("=" * 70)
    print()
    print("Output structure:")
    print(f"  {output_base_dir}/")
    print(f"  ├── test_easy/")
    print(f"  │   ├── low/")
    print(f"  │   ├── high/")
    print(f"  │   └── pairs.txt")
    print(f"  ├── test_medium/")
    print(f"  │   └── ...")
    print(f"  └── test_hard/")
    print(f"      └── ...")
    print()
    print("Difficulty level specifications:")
    print("  Easy:   1% light, no noise (ceiling performance)")
    print("  Medium: 5% light, minimal Poisson-Gaussian noise")
    print("  Hard:   10% light, higher noise, white balance shift")
    print()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate multi-level low-light test sets for face recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 1000 pairs per level
  python datasets/generate_multilevel_test_sets.py \\
      --source_test_dir=./datasets/LFW_lowlight/test \\
      --output_base_dir=./datasets/LFW_multilevel \\
      --num_pairs=1000

  # Use custom number of pairs
  python datasets/generate_multilevel_test_sets.py \\
      --source_test_dir=./datasets/LFW_lowlight/test \\
      --output_base_dir=./datasets/LFW_multilevel \\
      --num_pairs=500
        """
    )

    parser.add_argument(
        '--source_test_dir',
        type=str,
        default='./datasets/LFW_lowlight/test',
        help='Path to source test set containing high/ subdirectory (default: ./datasets/LFW_lowlight/test)'
    )

    parser.add_argument(
        '--output_base_dir',
        type=str,
        default='./datasets/LFW_multilevel',
        help='Base output directory for multi-level test sets (default: ./datasets/LFW_multilevel)'
    )

    parser.add_argument(
        '--num_pairs',
        type=int,
        default=1000,
        help='Number of pairs of each type per level (default: 1000)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    try:
        generate_multilevel_test_sets(
            source_test_dir=args.source_test_dir,
            output_base_dir=args.output_base_dir,
            num_pairs=args.num_pairs,
            seed=args.seed
        )
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
