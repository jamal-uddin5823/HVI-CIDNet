"""
Generate Multi-Level Low-Light Training Sets for Face Recognition

This script creates training/validation/test sets with multiple difficulty levels
(easy, medium, hard) to match the distribution of multi-level test sets.

Each difficulty level uses the EXACT same parameters as generate_multilevel_test_sets.py:
- Easy: 1% light, no noise, gamma correction ON (raw_sensor_mode=False)
- Medium: 5% light, Poisson-Gaussian noise, raw sensor mode
- Hard: 10% light, higher noise, white balance shift, raw sensor mode

Dataset Structure (preserves LFW person identity):
    datasets/LFW_multilevel/
    ├── train_easy/
    │   ├── low/
    │   │   ├── George_W_Bush/
    │   │   │   ├── George_W_Bush_0001.png
    │   │   │   └── George_W_Bush_0002.png
    │   │   └── Colin_Powell/
    │   │       └── Colin_Powell_0001.png
    │   └── high/ (same structure - ground truth)
    ├── train_medium/ (same structure)
    ├── train_hard/ (same structure)
    ├── train_mixed/ (optional - combines all levels via symlinks)
    ├── val_easy/, val_medium/, val_hard/
    └── test_easy/, test_medium/, test_hard/

Usage:
    python generate_multilevel_training_sets.py \\
        --lfw_dir=./datasets/LFW_original/lfw \\
        --output_base_dir=./datasets/LFW_multilevel \\
        --generate_mixed \\
        --use_symlinks
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import shutil
from collections import defaultdict


# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.lowlight_synthesis import synthesize_low_light_image


# ============================================================================
# Difficulty Level Configurations
# MUST match generate_multilevel_test_sets.py EXACTLY
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
        'raw_sensor_mode': False  # Gamma needed to make 1% light visible
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
# Dataset Scanning and Splitting
# ============================================================================

def scan_lfw_dataset(lfw_dir, min_images_per_person=2):
    """
    Scan LFW dataset and organize images by person.

    Args:
        lfw_dir: Path to LFW dataset (e.g., ./datasets/LFW_original/lfw)
        min_images_per_person: Minimum images per person to include

    Returns:
        dict: {person_name: [image_paths]}
    """
    print(f"  Scanning: {lfw_dir}")

    person_images = {}
    person_dirs = sorted([d for d in os.listdir(lfw_dir)
                         if os.path.isdir(os.path.join(lfw_dir, d))])

    for person_name in tqdm(person_dirs, desc="  Scanning people"):
        person_dir = os.path.join(lfw_dir, person_name)
        images = sorted([f for f in os.listdir(person_dir)
                        if f.endswith(('.jpg', '.png', '.jpeg'))])

        # Filter by minimum images per person
        if len(images) >= min_images_per_person:
            image_paths = [os.path.join(person_dir, img) for img in images]
            person_images[person_name] = image_paths

    total_images = sum(len(imgs) for imgs in person_images.values())
    print(f"  Found {total_images} images from {len(person_images)} people")

    return person_images


def split_people_by_ratio(people_list, train_ratio=0.7, val_ratio=0.15,
                          test_ratio=0.15, seed=42):
    """
    Split people into train/val/test sets (PERSON-BASED to prevent data leakage).

    Args:
        people_list: List of person names
        train_ratio: Fraction of people for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed

    Returns:
        tuple: (train_people, val_people, test_people) - lists of person names
    """
    import random
    random.seed(seed)

    n_people = len(people_list)
    n_train = int(n_people * train_ratio)
    n_val = int(n_people * val_ratio)

    # Shuffle people
    shuffled_people = people_list.copy()
    random.shuffle(shuffled_people)

    # Split
    train_people = shuffled_people[:n_train]
    val_people = shuffled_people[n_train:n_train + n_val]
    test_people = shuffled_people[n_train + n_val:]

    return train_people, val_people, test_people


# ============================================================================
# Multi-Level Dataset Generation
# ============================================================================

def generate_level_for_split(
    people_list,
    person_images,
    output_split_dir,
    level_params,
    level_name,
    seed=42
):
    """
    Generate low-light images for a specific split and difficulty level.

    Args:
        people_list: List of person names in this split
        person_images: Dict mapping person_name -> [image_paths]
        output_split_dir: Output directory (e.g., ./datasets/LFW_multilevel/train_easy)
        level_params: Dict of synthesis parameters for this difficulty level
        level_name: Name of difficulty level (for progress display)
        seed: Random seed
    """
    os.makedirs(output_split_dir, exist_ok=True)

    low_dir = os.path.join(output_split_dir, 'low')
    high_dir = os.path.join(output_split_dir, 'high')
    os.makedirs(low_dir, exist_ok=True)
    os.makedirs(high_dir, exist_ok=True)

    image_count = 0
    person_count = 0

    for person_name in tqdm(people_list, desc=f"    {level_name.capitalize()}"):
        if person_name not in person_images:
            continue

        # Create person subdirectories
        person_low_dir = os.path.join(low_dir, person_name)
        person_high_dir = os.path.join(high_dir, person_name)
        os.makedirs(person_low_dir, exist_ok=True)
        os.makedirs(person_high_dir, exist_ok=True)
        person_count += 1

        for img_path in person_images[person_name]:
            # Load image
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img).astype(np.float32) / 255.0

            # Get filename
            original_filename = os.path.basename(img_path)
            img_basename = os.path.splitext(original_filename)[0]
            img_name = img_basename + '.png'

            # Paths
            high_path = os.path.join(person_high_dir, img_name)
            low_path = os.path.join(person_low_dir, img_name)

            # Copy ground truth
            shutil.copy2(img_path, high_path)

            # Generate low-light version
            # Use deterministic seed for reproducibility
            img_seed = seed + hash(f"{person_name}_{img_basename}") % 1000000

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
            Image.fromarray(low_light_img).save(low_path)
            image_count += 1

    return image_count, person_count


def generate_mixed_set(output_base_dir, split_name, use_symlinks=True):
    """
    Generate mixed training set combining all difficulty levels.

    Args:
        output_base_dir: Base output directory
        split_name: Split name (e.g., 'train', 'val')
        use_symlinks: If True, use symlinks; otherwise copy files
    """
    mixed_dir = os.path.join(output_base_dir, f'{split_name}_mixed')
    mixed_low_dir = os.path.join(mixed_dir, 'low')
    mixed_high_dir = os.path.join(mixed_dir, 'high')

    os.makedirs(mixed_low_dir, exist_ok=True)
    os.makedirs(mixed_high_dir, exist_ok=True)

    levels = ['easy', 'medium', 'hard']
    total_files = 0

    print(f"    Generating {split_name}_mixed...")

    for level in levels:
        source_dir = os.path.join(output_base_dir, f'{split_name}_{level}')

        for subdir in ['low', 'high']:
            source_subdir = os.path.join(source_dir, subdir)
            target_subdir = os.path.join(mixed_dir, subdir)

            if not os.path.exists(source_subdir):
                continue

            # Copy/symlink all person directories
            for person_name in os.listdir(source_subdir):
                source_person_dir = os.path.join(source_subdir, person_name)
                target_person_dir = os.path.join(target_subdir, person_name)

                if not os.path.isdir(source_person_dir):
                    continue

                os.makedirs(target_person_dir, exist_ok=True)

                for img_file in os.listdir(source_person_dir):
                    source_file = os.path.join(source_person_dir, img_file)
                    target_file = os.path.join(target_person_dir, img_file)

                    # Skip if exists
                    if os.path.exists(target_file):
                        continue

                    if use_symlinks:
                        # Create relative symlink
                        rel_source = os.path.relpath(source_file, target_person_dir)
                        os.symlink(rel_source, target_file)
                    else:
                        shutil.copy2(source_file, target_file)

                    total_files += 1

    method = "symlinks" if use_symlinks else "copies"
    print(f"    ✓ Created {total_files} {method}")


# ============================================================================
# Statistics Generation
# ============================================================================

def count_images(base_dir):
    """Count images in a directory tree."""
    count = 0
    for root, dirs, files in os.walk(base_dir):
        count += len([f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))])
    return count


def get_people_in_split(base_dir):
    """Get list of people in a split."""
    high_dir = os.path.join(base_dir, 'high')
    if not os.path.exists(high_dir):
        return []
    return [d for d in os.listdir(high_dir) if os.path.isdir(os.path.join(high_dir, d))]


def generate_statistics(output_base_dir, splits_dict, stats_file):
    """Generate dataset statistics file."""
    with open(stats_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("LFW Multi-Level Training Dataset Statistics\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Output Base Directory: {output_base_dir}\n\n")

        # Split summary
        f.write("Person-Based Split Summary:\n")
        f.write("-" * 70 + "\n")
        for split_name, people_list in splits_dict.items():
            f.write(f"  {split_name.capitalize()}: {len(people_list)} people\n")
        f.write("\n")

        # Difficulty level breakdown
        f.write("Difficulty Level Breakdown:\n")
        f.write("-" * 70 + "\n")

        for split_name in ['train', 'val', 'test']:
            f.write(f"\n{split_name.capitalize()} Set:\n")
            for level in ['easy', 'medium', 'hard']:
                level_dir = os.path.join(output_base_dir, f"{split_name}_{level}")
                if os.path.exists(level_dir):
                    people = get_people_in_split(level_dir)
                    images = count_images(level_dir) // 2  # Divide by 2 for low/high
                    f.write(f"  {split_name}_{level}: {images} images from {len(people)} people\n")

            # Check for mixed set
            mixed_dir = os.path.join(output_base_dir, f"{split_name}_mixed")
            if os.path.exists(mixed_dir):
                people = get_people_in_split(mixed_dir)
                images = count_images(mixed_dir) // 2
                f.write(f"  {split_name}_mixed: {images} images from {len(people)} people\n")

        f.write("\n")

        # Difficulty specifications
        f.write("Difficulty Level Specifications:\n")
        f.write("-" * 70 + "\n")
        f.write("  Easy:   1% light, no noise, gamma correction ON\n")
        f.write("  Medium: 5% light, Poisson-Gaussian noise, raw sensor mode\n")
        f.write("  Hard:   10% light, high noise, white balance shift, raw mode\n\n")

        # Split strategy
        f.write("Split Strategy: Person-based (prevents data leakage)\n")
        f.write("-" * 70 + "\n")
        f.write("  ✓ All images of same person stay in same split\n")
        f.write("  ✓ No person appears in multiple splits\n")
        f.write("  ✓ Identity preserved in directory structure\n\n")

        # Training recommendations
        f.write("Training Recommendations:\n")
        f.write("-" * 70 + "\n")
        f.write("  1. Mixed Training (Recommended):\n")
        f.write("     - Train on train_mixed for robust generalization\n")
        f.write("     - Model sees random difficulty each batch\n")
        f.write("     - Best for real-world deployment\n\n")
        f.write("  2. Curriculum Learning:\n")
        f.write("     - Start with train_easy (epoch 1-100)\n")
        f.write("     - Progress to train_medium (epoch 101-200)\n")
        f.write("     - Finish with train_hard (epoch 201+)\n\n")
        f.write("  3. Evaluation:\n")
        f.write("     - Evaluate on val_easy, val_medium, val_hard separately\n")
        f.write("     - Track generalization gap across levels\n")
        f.write("     - Final evaluation on test_easy/medium/hard\n")


# ============================================================================
# Main Generation Function
# ============================================================================

def generate_multilevel_training_sets(
    lfw_dir='./datasets/LFW_original/lfw',
    output_base_dir='./datasets/LFW_multilevel',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    min_images_per_person=2,
    generate_mixed=True,
    use_symlinks=True,
    seed=42
):
    """
    Generate multi-level low-light training sets.

    Args:
        lfw_dir: Path to original LFW dataset
        output_base_dir: Base output directory
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        min_images_per_person: Minimum images per person to include
        generate_mixed: Generate mixed training set
        use_symlinks: Use symlinks for mixed set (saves disk space)
        seed: Random seed
    """
    print("=" * 70)
    print("Multi-Level Low-Light Training Set Generation")
    print("=" * 70)
    print(f"Source:      {lfw_dir}")
    print(f"Output:      {output_base_dir}")
    print(f"Train/Val/Test: {train_ratio:.1%} / {val_ratio:.1%} / {test_ratio:.1%}")
    print(f"Seed:        {seed}")
    print("=" * 70)
    print()

    # Check if LFW directory exists
    if not os.path.exists(lfw_dir):
        raise FileNotFoundError(f"LFW directory not found: {lfw_dir}")

    # Step 1: Scan LFW dataset
    print("[Step 1/4] Scanning LFW dataset...")
    person_images = scan_lfw_dataset(lfw_dir, min_images_per_person)
    all_people = list(person_images.keys())
    print()

    # Step 2: Split by person
    print("[Step 2/4] Splitting people into train/val/test...")
    train_people, val_people, test_people = split_people_by_ratio(
        all_people, train_ratio, val_ratio, test_ratio, seed
    )

    splits = {
        'train': train_people,
        'val': val_people,
        'test': test_people
    }

    # Calculate image counts per split
    for split_name, people_list in splits.items():
        img_count = sum(len(person_images[p]) for p in people_list)
        print(f"  {split_name.capitalize()}: {img_count} images from {len(people_list)} people")

    print()

    # Step 3: Generate multi-level versions for each split
    print("[Step 3/4] Generating multi-level low-light images...")

    for split_name, people_list in splits.items():
        print(f"\n  Processing {split_name} set ({len(people_list)} people)...")

        for level_name, level_params in DIFFICULTY_LEVELS.items():
            output_split_dir = os.path.join(
                output_base_dir, f"{split_name}_{level_name}"
            )

            image_count, person_count = generate_level_for_split(
                people_list,
                person_images,
                output_split_dir,
                level_params,
                level_name,
                seed
            )

            print(f"    ✓ {level_name.capitalize()}: {image_count} images from {person_count} people")

    # Step 4: Generate mixed training set
    if generate_mixed:
        print("\n[Step 4/4] Generating mixed training sets...")
        for split_name in ['train', 'val']:
            if os.path.exists(os.path.join(output_base_dir, f'{split_name}_easy')):
                generate_mixed_set(output_base_dir, split_name, use_symlinks)

    # Generate statistics
    print("\n[Generating statistics...]")
    stats_file = os.path.join(output_base_dir, 'dataset_stats.txt')
    generate_statistics(output_base_dir, splits, stats_file)
    print(f"  ✓ Statistics saved to: {stats_file}")

    # Done
    print("\n" + "=" * 70)
    print("Multi-level training set generation complete!")
    print("=" * 70)
    print(f"\nDataset location: {output_base_dir}\n")

    print("Directory structure:")
    print(f"  {output_base_dir}/")
    print(f"  ├── train_easy/")
    print(f"  │   ├── low/")
    print(f"  │   └── high/")
    print(f"  ├── train_medium/")
    print(f"  ├── train_hard/")
    if generate_mixed:
        print(f"  ├── train_mixed/  (all levels combined)")
    print(f"  ├── val_easy/, val_medium/, val_hard/")
    if generate_mixed:
        print(f"  └── val_mixed/")

    print("\nDifficulty level specifications:")
    print("  Easy:   1% light, no noise, gamma correction ON")
    print("  Medium: 5% light, Poisson-Gaussian noise, raw mode")
    print("  Hard:   10% light, high noise, white balance shift, raw mode")

    print("\nRecommended training:")
    print("  1. Mixed training (recommended):")
    print(f"     python train.py --lfw=True \\")
    print(f"       --data_train_lfw=./datasets/LFW_multilevel/train_mixed \\")
    print(f"       --data_val_lfw=./datasets/LFW_multilevel/val_mixed")

    print("\n  2. Curriculum learning:")
    print(f"     # Epochs 1-100: train_easy")
    print(f"     # Epochs 101-200: train_medium")
    print(f"     # Epochs 201+: train_hard")

    print("\n  3. Evaluation on multi-level test set:")
    print(f"     python eval_face_verification.py \\")
    print(f"       --test_dir=./datasets/LFW_multilevel/test_easy \\")
    print(f"       --pairs_file=./datasets/LFW_multilevel/test_easy/pairs.txt \\")
    print(f"       --checkpoint=./checkpoints/your_model.pth")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate multi-level low-light training sets for face recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all multi-level training sets with mixed training
  python generate_multilevel_training_sets.py \\
      --lfw_dir=./datasets/LFW_original/lfw \\
      --output_base_dir=./datasets/LFW_multilevel \\
      --generate_mixed \\
      --use_symlinks

  # Without mixed training set (separate levels only)
  python generate_multilevel_training_sets.py \\
      --lfw_dir=./datasets/LFW_original/lfw \\
      --output_base_dir=./datasets/LFW_multilevel

  # Custom split ratios
  python generate_multilevel_training_sets.py \\
      --train_ratio 0.8 \\
      --val_ratio 0.1 \\
      --test_ratio 0.1
        """
    )

    parser.add_argument(
        '--lfw_dir',
        type=str,
        default='./datasets/LFW_original/lfw',
        help='Path to original LFW dataset (default: ./datasets/LFW_original/lfw)'
    )

    parser.add_argument(
        '--output_base_dir',
        type=str,
        default='./datasets/LFW_multilevel',
        help='Base output directory (default: ./datasets/LFW_multilevel)'
    )

    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Training set ratio (default: 0.7)'
    )

    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15)'
    )

    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.15,
        help='Test set ratio (default: 0.15)'
    )

    parser.add_argument(
        '--min_images_per_person',
        type=int,
        default=2,
        help='Minimum images per person to include (default: 2)'
    )

    parser.add_argument(
        '--generate_mixed',
        action='store_true',
        help='Generate mixed training set (all levels combined)'
    )

    parser.add_argument(
        '--use_symlinks',
        action='store_true',
        help='Use symbolic links for mixed set (saves disk space)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    try:
        generate_multilevel_training_sets(
            lfw_dir=args.lfw_dir,
            output_base_dir=args.output_base_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            min_images_per_person=args.min_images_per_person,
            generate_mixed=args.generate_mixed,
            use_symlinks=args.use_symlinks,
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
