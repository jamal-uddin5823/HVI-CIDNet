"""
LFW Dataset Preparation Script for Low-Light Face Recognition Research

This script prepares the Labeled Faces in the Wild (LFW) dataset for training
face recognition-aware low-light image enhancement models.

It uses physics-based low-light synthesis to create realistic low-light versions
of LFW face images, enabling supervised training without requiring real low-light
face data collection.

Dataset Structure:
    datasets/LFW_lowlight/
    ├── train/
    │   ├── low/      # Synthetic low-light faces
    │   └── high/     # Original faces (ground truth)
    ├── val/
    │   ├── low/
    │   └── high/
    └── test/
        ├── low/
        └── high/

Usage:
    python prepare_lfw_dataset.py --download  # Download and process LFW
    python prepare_lfw_dataset.py             # Process existing LFW dataset
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import urllib.request
import tarfile
import shutil
import random

# Import low-light synthesis module (full physics-based pipeline)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data'))
from lowlight_synthesis import synthesize_low_light_image


def download_lfw(data_dir='./datasets/LFW_original', min_faces_per_person=2):
    """
    Download LFW dataset using scikit-learn (most reliable method!)

    This uses sklearn.datasets.fetch_lfw_people which:
    - Handles downloading automatically with retry logic
    - Provides properly aligned faces (better than raw LFW)
    - Works reliably even behind firewalls
    - Caches downloads automatically in ~/.scikit_learn_data
    - Falls back to manual download if sklearn unavailable

    Args:
        data_dir: Directory to save organized LFW dataset
        min_faces_per_person: Minimum images per person (default: 2)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from sklearn.datasets import fetch_lfw_people
        use_sklearn = True
    except ImportError:
        print("⚠ Warning: scikit-learn not found")
        print("  Install with: pip install scikit-learn")
        print("  Falling back to manual download...")
        use_sklearn = False

    if use_sklearn:
        return _download_lfw_sklearn(data_dir, min_faces_per_person)
    else:
        return _download_lfw_manual(data_dir)


def _download_lfw_sklearn(data_dir='./datasets/LFW_original', min_faces_per_person=2):
    """Download LFW using scikit-learn (preferred method)"""
    from sklearn.datasets import fetch_lfw_people

    print(f"[1/2] Downloading LFW dataset using scikit-learn...")
    print(f"  ✓ Reliable download with automatic retry")
    print(f"  ✓ Properly aligned faces included")
    print(f"  ✓ This may take a few minutes on first run...")

    try:
        # Download using sklearn (downloads to ~/scikit_learn_data by default)
        lfw_people = fetch_lfw_people(
            min_faces_per_person=min_faces_per_person,
            resize=1.0,  # Keep original size (~250x250)
            color=True,  # RGB images
            download_if_missing=True
        )

        print(f"\n  ✓ Downloaded {len(lfw_people.images)} images")
        print(f"  ✓ {len(lfw_people.target_names)} unique people")
        print(f"  ✓ Image shape: {lfw_people.images[0].shape}")

    except Exception as e:
        print(f"\n  ✗ Error downloading with sklearn: {e}")
        print("  Trying manual download method...")
        return _download_lfw_manual(data_dir)

    # Organize into directory structure expected by prepare_lfw_lowlight
    print(f"\n[2/2] Organizing dataset into directory structure...")
    lfw_dir = os.path.join(data_dir, 'lfw')
    os.makedirs(lfw_dir, exist_ok=True)

    # Group images by person
    from collections import defaultdict
    person_images = defaultdict(list)

    for img, target in zip(lfw_people.images, lfw_people.target):
        person_name = lfw_people.target_names[target]
        person_images[person_name].append(img)

    # Save images to disk
    saved_count = 0
    for person_name, images in tqdm(person_images.items(), desc="  Saving images"):
        person_dir = os.path.join(lfw_dir, person_name.replace(' ', '_'))
        os.makedirs(person_dir, exist_ok=True)

        for idx, img in enumerate(images):
            # Convert to PIL and save
            img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            img_pil = Image.fromarray(img_uint8)

            img_path = os.path.join(person_dir, f"{person_name.replace(' ', '_')}_{idx:04d}.jpg")
            img_pil.save(img_path)
            saved_count += 1

    print(f"\n  ✓ Saved {saved_count} images to {lfw_dir}")
    print(f"  ✓ Dataset ready for low-light synthesis!")

    return True


def _download_lfw_manual(data_dir='./datasets/LFW_original'):
    """Fallback: Manual download method (if sklearn fails)"""
    os.makedirs(data_dir, exist_ok=True)

    # LFW download URLs (try multiple mirrors)
    lfw_urls = [
        'http://vis-www.cs.umass.edu/lfw/lfw.tgz',
        'https://ndownloader.figshare.com/files/5976018',  # Alternative mirror
    ]

    print(f"\n[Manual Download Method]")
    print(f"  Attempting download from multiple mirrors...")

    tar_path = os.path.join(data_dir, 'lfw.tgz')

    # Try each URL
    for idx, lfw_url in enumerate(lfw_urls):
        print(f"\n  Trying mirror {idx+1}/{len(lfw_urls)}: {lfw_url}")

        try:
            urllib.request.urlretrieve(lfw_url, tar_path)
            print("  ✓ Download complete!")
            break
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            if idx == len(lfw_urls) - 1:
                print("\n  ✗ All download attempts failed")
                print("\n  Please download manually:")
                print(f"    1. Visit: http://vis-www.cs.umass.edu/lfw/lfw.tgz")
                print(f"    2. Save to: {tar_path}")
                print(f"    3. Run: python prepare_lfw_dataset.py (without --download)")
                return False

    print(f"\n  Extracting dataset...")
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(data_dir)
        print("  ✓ Extraction complete!")
    except Exception as e:
        print(f"  ✗ Error extracting: {e}")
        return False

    # Clean up tar file
    os.remove(tar_path)

    print(f"  ✓ Dataset ready at: {data_dir}/lfw")
    return True


def prepare_lfw_lowlight(
    lfw_dir='./datasets/LFW_original/lfw',
    output_dir='./datasets/LFW_lowlight',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    min_images_per_person=2,
    max_images=None,
    enable_blur=False,
    seed=42
):
    """
    Prepare LFW dataset with synthetic low-light versions

    Args:
        lfw_dir: Path to original LFW dataset
        output_dir: Output directory for processed dataset
        train_ratio: Fraction of data for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        min_images_per_person: Minimum images per person to include
        max_images: Maximum total images (for quick testing), None = all
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)

    print("="*70)
    print("LFW Low-Light Dataset Preparation")
    print("="*70)

    # Check if LFW directory exists
    if not os.path.exists(lfw_dir):
        print(f"Error: LFW directory not found: {lfw_dir}")
        print("Run with --download to download LFW dataset first")
        return False

    # Collect all image paths
    print("\n[Step 1/5] Scanning LFW directory...")
    all_images = []
    person_dirs = sorted([d for d in os.listdir(lfw_dir)
                         if os.path.isdir(os.path.join(lfw_dir, d))])

    for person_name in person_dirs:
        person_dir = os.path.join(lfw_dir, person_name)
        images = sorted([f for f in os.listdir(person_dir)
                        if f.endswith(('.jpg', '.png'))])

        # Filter by minimum images per person
        if len(images) >= min_images_per_person:
            for img_name in images:
                all_images.append(os.path.join(person_dir, img_name))

    print(f"  Found {len(all_images)} images from {len(person_dirs)} people")

    # Limit dataset size if specified
    if max_images is not None and len(all_images) > max_images:
        print(f"  Limiting to {max_images} images (for quick testing)")
        random.shuffle(all_images)
        all_images = all_images[:max_images]

    # Split into train/val/test
    print("\n[Step 2/5] Splitting dataset...")
    random.shuffle(all_images)

    n_total = len(all_images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_images = all_images[:n_train]
    val_images = all_images[n_train:n_train + n_val]
    test_images = all_images[n_train + n_val:]

    print(f"  Train: {len(train_images)} images ({train_ratio*100:.0f}%)")
    print(f"  Val:   {len(val_images)} images ({val_ratio*100:.0f}%)")
    print(f"  Test:  {len(test_images)} images ({test_ratio*100:.0f}%)")

    # Create output directories
    print("\n[Step 3/5] Creating output directories...")
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }

    for split_name in splits.keys():
        os.makedirs(os.path.join(output_dir, split_name, 'low'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split_name, 'high'), exist_ok=True)

    # Process images
    print("\n[Step 4/5] Generating synthetic low-light images...")
    print("  This may take several minutes...")

    for split_name, image_list in splits.items():
        print(f"\n  Processing {split_name} set ({len(image_list)} images)...")

        for idx, img_path in enumerate(tqdm(image_list, desc=f"  {split_name}")):
            try:
                # Load image
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img).astype(np.float32) / 255.0

                # Generate filename (use index to avoid conflicts)
                img_name = f"{split_name}_{idx:05d}.png"

                # Save high-quality version (ground truth)
                high_path = os.path.join(output_dir, split_name, 'high', img_name)
                img.save(high_path)

                # Generate low-light version using full physics-based synthesis
                # Use varied parameters for diversity
                reduction_factor = random.uniform(0.05, 0.15)

                # Vary noise parameters
                shot_noise = random.uniform(1.0, 2.0)
                read_noise = random.uniform(0.005, 0.015)
                gain = random.uniform(1.5, 3.0)
                wb_variation = random.uniform(0.1, 0.2)

                # Optional blur parameters (only used if enable_blur=True)
                blur_sigma = random.uniform(0.3, 0.6) if enable_blur else 0.0

                # Note: Blur is disabled by default for face recognition research
                # Blur destroys facial identity features, making enhancement too hard
                # Use --enable_blur flag for general low-light datasets
                low_light_array = synthesize_low_light_image(
                    img_array,
                    apply_light_reduction=True,
                    apply_noise=True,
                    apply_white_balance=True,
                    apply_blur=enable_blur,  # Default: False (preserves identity)
                    reduction_factor=reduction_factor,
                    shot_noise_scale=shot_noise,
                    read_noise_std=read_noise,
                    gain=gain,
                    wb_variation=wb_variation,
                    blur_sigma=blur_sigma,
                    blur_type='gaussian',
                    seed=seed + idx,
                    output_format='numpy'
                )

                # Save low-light version
                low_light_img = (low_light_array * 255).astype(np.uint8)
                low_path = os.path.join(output_dir, split_name, 'low', img_name)
                Image.fromarray(low_light_img).save(low_path)

            except Exception as e:
                print(f"    Error processing {img_path}: {e}")
                continue

    # Generate dataset statistics
    print("\n[Step 5/5] Generating dataset statistics...")
    stats_file = os.path.join(output_dir, 'dataset_stats.txt')
    with open(stats_file, 'w') as f:
        f.write("LFW Low-Light Dataset Statistics\n")
        f.write("="*70 + "\n\n")
        f.write(f"Source: {lfw_dir}\n")
        f.write(f"Output: {output_dir}\n\n")
        f.write(f"Total images: {len(all_images)}\n")
        f.write(f"  Train: {len(train_images)} ({train_ratio*100:.1f}%)\n")
        f.write(f"  Val:   {len(val_images)} ({val_ratio*100:.1f}%)\n")
        f.write(f"  Test:  {len(test_images)} ({test_ratio*100:.1f}%)\n\n")
        f.write("Low-light synthesis parameters:\n")
        f.write("  Reduction factor: 0.05 - 0.15 (random per image)\n")
        f.write("  Noise level: low/medium/high (random per image)\n")
        f.write("  Physics-based: Poisson-Gaussian sensor noise\n")
        f.write("  White balance variation: ±10%\n")

    print(f"\n  Statistics saved to: {stats_file}")

    print("\n" + "="*70)
    print("Dataset preparation complete!")
    print("="*70)
    print(f"\nDataset location: {output_dir}")
    print("\nDirectory structure:")
    print("  LFW_lowlight/")
    print("  ├── train/")
    print("  │   ├── low/   (synthetic low-light)")
    print("  │   └── high/  (ground truth)")
    print("  ├── val/")
    print("  │   ├── low/")
    print("  │   └── high/")
    print("  └── test/")
    print("      ├── low/")
    print("      └── high/")
    print("\nYou can now train with: --lfw=True --data_train_lfw=./datasets/LFW_lowlight/train")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Prepare LFW dataset with synthetic low-light images'
    )
    parser.add_argument('--download', action='store_true',
                       help='Download LFW dataset first')
    parser.add_argument('--lfw_dir', type=str, default='./datasets/LFW_original/lfw',
                       help='Path to original LFW dataset')
    parser.add_argument('--output_dir', type=str, default='./datasets/LFW_lowlight',
                       help='Output directory for processed dataset')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test set ratio')
    parser.add_argument('--min_images', type=int, default=2,
                       help='Minimum images per person')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum total images (for testing)')
    parser.add_argument('--enable_blur', action='store_true',
                       help='Enable blur in synthesis (NOT recommended for face recognition research)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Download if requested
    if args.download:
        download_dir = os.path.dirname(args.lfw_dir)
        success = download_lfw(download_dir)
        if not success:
            return

    # Prepare dataset
    success = prepare_lfw_lowlight(
        lfw_dir=args.lfw_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_images_per_person=args.min_images,
        max_images=args.max_images,
        enable_blur=args.enable_blur,
        seed=args.seed
    )

    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
