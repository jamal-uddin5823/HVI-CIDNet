#!/usr/bin/env python3
"""
LaPa-Face Dataset Preparation Script

This script prepares the LaPa-Face dataset for training by:
1. Extracting train and test zip files
2. Verifying directory structure
3. Checking for matching pairs between underexposed and normal images
4. Providing dataset statistics

LaPa-Face contains real underexposed images (no synthetic generation needed).

Dataset Download Links:
- Train: https://drive.google.com/file/d/1bmFIy1In-OnTv-Fb1kvsk46hojJrpINb/view?usp=sharing
- Test:  https://drive.google.com/file/d/1neJZq1C9HkXCO_eqdPRijiX4jvggF7mF/view?usp=sharing

Usage:
    python prepare_lapaface_dataset.py --data_dir ./datasets/LaPa-Face
    python prepare_lapaface_dataset.py --data_dir ./datasets/LaPa-Face --verify_only
"""

import os
import argparse
import zipfile
from pathlib import Path
from collections import defaultdict


def extract_zip(zip_path, extract_to):
    """Extract a zip file to the specified directory"""
    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"  ✓ Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"  ✗ Error extracting {zip_path}: {e}")
        return False


def get_image_files(directory):
    """Get list of image files in a directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []

    if not os.path.exists(directory):
        return []

    for filename in sorted(os.listdir(directory)):
        if Path(filename).suffix.lower() in image_extensions:
            image_files.append(filename)

    return image_files


def verify_directory_structure(base_dir, split='train'):
    """Verify that the expected directory structure exists"""
    print(f"\nVerifying {split} directory structure...")

    if split == 'train':
        expected_dir = os.path.join(base_dir, 'LaPa-Face')
    else:  # test
        expected_dir = os.path.join(base_dir, 'LaPa-Test')

    required_dirs = ['underexposed', 'normal', 'seg']
    all_exist = True

    for subdir in required_dirs:
        full_path = os.path.join(expected_dir, subdir)
        if os.path.exists(full_path):
            num_files = len(get_image_files(full_path))
            print(f"  ✓ {subdir:15s} exists ({num_files} images)")
        else:
            print(f"  ✗ {subdir:15s} NOT FOUND")
            all_exist = False

    return all_exist, expected_dir


def check_matching_pairs(underexposed_dir, normal_dir):
    """Check for matching filenames between underexposed and normal directories"""
    print("\nChecking for matching pairs...")

    underexposed_files = set(get_image_files(underexposed_dir))
    normal_files = set(get_image_files(normal_dir))

    print(f"  Underexposed images: {len(underexposed_files)}")
    print(f"  Normal images:       {len(normal_files)}")

    # Find matching pairs
    matching = underexposed_files.intersection(normal_files)
    only_underexposed = underexposed_files - normal_files
    only_normal = normal_files - underexposed_files

    print(f"  Matching pairs:      {len(matching)}")

    if only_underexposed:
        print(f"  ⚠ Only in underexposed: {len(only_underexposed)}")
        if len(only_underexposed) <= 10:
            for f in list(only_underexposed)[:10]:
                print(f"    - {f}")

    if only_normal:
        print(f"  ⚠ Only in normal: {len(only_normal)}")
        if len(only_normal) <= 10:
            for f in list(only_normal)[:10]:
                print(f"    - {f}")

    return len(matching), len(only_underexposed), len(only_normal)


def reorganize_dataset(base_dir):
    """Reorganize extracted dataset into standard structure"""
    print("\nReorganizing dataset structure...")

    # Expected structure after extraction
    train_source = os.path.join(base_dir, 'LaPa-Face')
    test_source = os.path.join(base_dir, 'LaPa-Test')

    # Target structure
    train_target = os.path.join(base_dir, 'train')
    test_target = os.path.join(base_dir, 'test')

    # Create target directories if they don't exist
    os.makedirs(train_target, exist_ok=True)
    os.makedirs(test_target, exist_ok=True)

    # Move/symlink train data
    if os.path.exists(train_source) and not os.path.exists(os.path.join(train_target, 'underexposed')):
        for subdir in ['underexposed', 'normal', 'seg']:
            source = os.path.join(train_source, subdir)
            target = os.path.join(train_target, subdir)
            if os.path.exists(source) and not os.path.exists(target):
                os.rename(source, target)
                print(f"  ✓ Moved train/{subdir}")

    # Move/symlink test data
    if os.path.exists(test_source) and not os.path.exists(os.path.join(test_target, 'underexposed')):
        for subdir in ['underexposed', 'normal', 'seg']:
            source = os.path.join(test_source, subdir)
            target = os.path.join(test_target, subdir)
            if os.path.exists(source) and not os.path.exists(target):
                os.rename(source, target)
                print(f"  ✓ Moved test/{subdir}")

    # Clean up empty directories
    for old_dir in [train_source, test_source]:
        if os.path.exists(old_dir) and not os.listdir(old_dir):
            os.rmdir(old_dir)
            print(f"  ✓ Removed empty directory {old_dir}")

    return train_target, test_target


def get_dataset_statistics(base_dir):
    """Get comprehensive statistics about the dataset"""
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)

    stats = {
        'train': {'underexposed': 0, 'normal': 0, 'seg': 0, 'matching': 0},
        'test': {'underexposed': 0, 'normal': 0, 'seg': 0, 'matching': 0}
    }

    for split in ['train', 'test']:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            continue

        print(f"\n{split.upper()}:")
        print("-" * 70)

        for subdir in ['underexposed', 'normal', 'seg']:
            full_path = os.path.join(split_dir, subdir)
            if os.path.exists(full_path):
                num_files = len(get_image_files(full_path))
                stats[split][subdir] = num_files
                print(f"  {subdir:15s}: {num_files:6d} images")

        # Check matching pairs
        underexposed_dir = os.path.join(split_dir, 'underexposed')
        normal_dir = os.path.join(split_dir, 'normal')

        if os.path.exists(underexposed_dir) and os.path.exists(normal_dir):
            underexposed_files = set(get_image_files(underexposed_dir))
            normal_files = set(get_image_files(normal_dir))
            matching = len(underexposed_files.intersection(normal_files))
            stats[split]['matching'] = matching
            print(f"  {'Matching pairs':15s}: {matching:6d} pairs")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    total_train = stats['train']['matching']
    total_test = stats['test']['matching']
    print(f"Total training pairs:   {total_train:6d}")
    print(f"Total test pairs:       {total_test:6d}")
    print(f"Total dataset size:     {total_train + total_test:6d} pairs")
    print("="*70)

    return stats


def main():
    parser = argparse.ArgumentParser(description='Prepare LaPa-Face dataset for training')
    parser.add_argument('--data_dir', type=str, default='./datasets/LaPa-Face',
                        help='Base directory for LaPa-Face dataset')
    parser.add_argument('--verify_only', action='store_true',
                        help='Only verify existing dataset without extraction')
    parser.add_argument('--train_zip', type=str, default=None,
                        help='Path to LaPa-Face-train.zip (if not in data_dir)')
    parser.add_argument('--test_zip', type=str, default=None,
                        help='Path to LaPa-Face-test.zip (if not in data_dir)')

    args = parser.parse_args()

    print("="*70)
    print("LaPa-Face Dataset Preparation")
    print("="*70)
    print(f"Base directory: {args.data_dir}")
    print("="*70)

    # Create base directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)

    # Determine zip file paths
    train_zip = args.train_zip or os.path.join(os.path.dirname(args.data_dir), 'LaPa-Face-train.zip')
    test_zip = args.test_zip or os.path.join(os.path.dirname(args.data_dir), 'LaPa-Face-test.zip')

    # Step 1: Extract zip files (if not verify_only mode)
    if not args.verify_only:
        print("\nStep 1: Extracting dataset...")
        print("-" * 70)

        if os.path.exists(train_zip):
            extract_zip(train_zip, args.data_dir)
        else:
            print(f"⚠ Train zip not found at: {train_zip}")
            print(f"  Please download from: https://drive.google.com/file/d/1bmFIy1In-OnTv-Fb1kvsk46hojJrpINb/view?usp=sharing")

        if os.path.exists(test_zip):
            extract_zip(test_zip, args.data_dir)
        else:
            print(f"⚠ Test zip not found at: {test_zip}")
            print(f"  Please download from: https://drive.google.com/file/d/1neJZq1C9HkXCO_eqdPRijiX4jvggF7mF/view?usp=sharing")

        # Step 2: Reorganize directory structure
        print("\nStep 2: Reorganizing directory structure...")
        print("-" * 70)
        train_dir, test_dir = reorganize_dataset(args.data_dir)
    else:
        print("\nVerify-only mode: Skipping extraction")
        train_dir = os.path.join(args.data_dir, 'train')
        test_dir = os.path.join(args.data_dir, 'test')

    # Step 3: Verify structure
    print("\nStep 3: Verifying dataset structure...")
    print("-" * 70)

    train_valid = os.path.exists(train_dir)
    test_valid = os.path.exists(test_dir)

    if train_valid:
        print("\nTrain directory:")
        for subdir in ['underexposed', 'normal', 'seg']:
            full_path = os.path.join(train_dir, subdir)
            if os.path.exists(full_path):
                num_files = len(get_image_files(full_path))
                print(f"  ✓ {subdir:15s}: {num_files:6d} images")
            else:
                print(f"  ✗ {subdir:15s}: NOT FOUND")
    else:
        print(f"✗ Train directory not found: {train_dir}")

    if test_valid:
        print("\nTest directory:")
        for subdir in ['underexposed', 'normal', 'seg']:
            full_path = os.path.join(test_dir, subdir)
            if os.path.exists(full_path):
                num_files = len(get_image_files(full_path))
                print(f"  ✓ {subdir:15s}: {num_files:6d} images")
            else:
                print(f"  ✗ {subdir:15s}: NOT FOUND")
    else:
        print(f"✗ Test directory not found: {test_dir}")

    # Step 4: Check matching pairs
    if train_valid:
        print("\nStep 4: Checking training pairs...")
        print("-" * 70)
        underexposed_dir = os.path.join(train_dir, 'underexposed')
        normal_dir = os.path.join(train_dir, 'normal')
        if os.path.exists(underexposed_dir) and os.path.exists(normal_dir):
            check_matching_pairs(underexposed_dir, normal_dir)

    if test_valid:
        print("\nStep 5: Checking test pairs...")
        print("-" * 70)
        underexposed_dir = os.path.join(test_dir, 'underexposed')
        normal_dir = os.path.join(test_dir, 'normal')
        if os.path.exists(underexposed_dir) and os.path.exists(normal_dir):
            check_matching_pairs(underexposed_dir, normal_dir)

    # Step 5: Dataset statistics
    get_dataset_statistics(args.data_dir)

    # Final status
    print("\n" + "="*70)
    print("PREPARATION COMPLETE")
    print("="*70)

    if train_valid and test_valid:
        print("✓ Dataset is ready for training!")
        print("\nNext steps:")
        print("  1. Test dataset loader:")
        print(f"     python -m data.lapaface_dataset")
        print("\n  2. Run validation training:")
        print(f"     cd DiscriminativeMultiLevelFaceLoss/")
        print(f"     ./validation_lapaface.sh")
        print("\n  3. Train comparison models:")
        print(f"     ./comparison_lapaface.sh")
    else:
        print("✗ Dataset preparation incomplete")
        print("\nPlease ensure zip files are downloaded:")
        print("  - Train: https://drive.google.com/file/d/1bmFIy1In-OnTv-Fb1kvsk46hojJrpINb/view?usp=sharing")
        print("  - Test:  https://drive.google.com/file/d/1neJZq1C9HkXCO_eqdPRijiX4jvggF7mF/view?usp=sharing")

    print("="*70)


if __name__ == '__main__':
    main()
