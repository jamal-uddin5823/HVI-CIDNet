"""
SS_Face Dataset Preparation Script for Low-Light Face Enhancement/Recognition

This script prepares the SS_Face dataset into a train/val/test split with
synthetic low-light counterparts, following the same structure and pipeline
style as `prepare_lfw_dataset.py`.

Expected source structure (inside the provided zip):
    datasets/SS_Face/SS_Face/<ID>/face/*.jpg
    datasets/SS_Face/SS_Face/<ID>/photos/*.jpg   # optional, ignored by default

Output structure (preserves identity as the numeric <ID>):
    datasets/SS_Face_lowlight/
    ├── train/
    │   ├── low/
    │   │   ├── 1/
    │   │   │   ├── 1_1.png  # synthetic low-light
    │   │   └── 10/
    │   └── high/             # originals as ground truth
    │       ├── 1/
    │       │   ├── 1_1.png
    │       └── 10/
    ├── val/  (same structure)
    └── test/ (same structure)

Usage:
    python prepare_ss_face_dataset.py --download         # unzip SS_Face.zip if needed (alias of --unzip)
    python prepare_ss_face_dataset.py --unzip            # unzip SS_Face.zip if needed
    python prepare_ss_face_dataset.py                    # process existing extracted folder
    python prepare_ss_face_dataset.py --enable_blur      # optional blur (not recommended)
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import zipfile
import random

# Import low-light synthesis module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data'))
from lowlight_synthesis import synthesize_low_light_image


def unzip_ss_face(zip_path='./datasets/SS_Face/SS_Face.zip', extract_to='./datasets/SS_Face') -> bool:
    """Unzip SS_Face.zip if SS_Face root folder is not present."""
    try:
        extract_to = os.path.abspath(extract_to)
        os.makedirs(extract_to, exist_ok=True)

        # If already extracted, skip
        root_candidate = os.path.join(extract_to, 'SS_Face')
        if os.path.isdir(root_candidate) and any(os.scandir(root_candidate)):
            print(f"✓ Found extracted folder: {root_candidate} — skipping unzip")
            return True

        if not os.path.isfile(zip_path):
            print(f"✗ Zip file not found: {zip_path}")
            return False

        print(f"[1/1] Extracting {zip_path} → {extract_to} ...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_to)
        print("  ✓ Extraction complete")
        return True
    except Exception as e:
        print(f"✗ Unzip failed: {e}")
        return False


def prepare_ss_face_lowlight(
    src_root='./datasets/SS_Face/SS_Face',
    output_dir='./datasets/SS_Face_lowlight',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    min_images_per_identity=1,
    max_images=None,
    enable_blur=False,
    seed=42,
    eval_only=True
):
    """
    Prepare SS_Face dataset with synthetic low-light versions (like LFW pipeline).

    We use images from `<ID>/face/*.jpg` as the "high" reference and synthesize
    their low-light counterparts. Identity is the numeric ID directory name.
    """
    random.seed(seed)
    np.random.seed(seed)

    print("="*70)
    print("SS_Face Low-Light Dataset Preparation")
    print("="*70)

    if not os.path.isdir(src_root):
        print(f"Error: SS_Face root not found: {src_root}")
        print("If you have SS_Face.zip, run with --unzip or unzip it manually.")
        return False

    # Collect all face images organized by identity (ID directory)
    print("\n[Step 1/5] Scanning SS_Face directory (face/*.jpg)...")
    from collections import defaultdict
    id_images = defaultdict(list)

    # Identity directories are immediate children under src_root
    for entry in sorted(os.listdir(src_root)):
        id_dir = os.path.join(src_root, entry)
        if not os.path.isdir(id_dir):
            continue
        face_dir = os.path.join(id_dir, 'face')
        if not os.path.isdir(face_dir):
            continue

        files = sorted([f for f in os.listdir(face_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if len(files) >= min_images_per_identity:
            for f in files:
                id_images[entry].append(os.path.join(face_dir, f))

    total_images = sum(len(v) for v in id_images.values())
    print(f"  Found {total_images} images from {len(id_images)} identities")
    print("  Split is identity-based to avoid leakage across splits")

    # Identity-based split
    print("\n[Step 2/5] Splitting identities into train/val/test...")
    ids = list(id_images.keys())
    random.shuffle(ids)

    if eval_only:
        # No split: use all images for test-only evaluation
        train_ids, val_ids, test_ids = [], [], ids
    else:
        n_ids = len(ids)
        n_train = int(n_ids * train_ratio)
        n_val = int(n_ids * val_ratio)
        train_ids = ids[:n_train]
        val_ids = ids[n_train:n_train+n_val]
        test_ids = ids[n_train+n_val:]

    def collect(split_ids):
        out = []
        for _id in split_ids:
            out.extend(id_images[_id])
        return out

    train_images = collect(train_ids)
    val_images = collect(val_ids)
    test_images = collect(test_ids)

    if max_images is not None:
        grand_total = len(train_images) + len(val_images) + len(test_images)
        if grand_total > max_images:
            print(f"  Limiting to {max_images} images for quick testing")
            scale = max_images / grand_total
            train_images = train_images[:int(len(train_images) * scale)]
            val_images = val_images[:int(len(val_images) * scale)]
            test_images = test_images[:int(len(test_images) * scale)]

    if eval_only:
        print(f"  Eval-only: using ALL {len(test_images)} images from {len(test_ids)} identities as test set")
    else:
        print(f"  Train: {len(train_images)} images from {len(train_ids)} identities")
        print(f"  Val:   {len(val_images)} images from {len(val_ids)} identities")
        print(f"  Test:  {len(test_images)} images from {len(test_ids)} identities")

    # Create output directories
    print("\n[Step 3/5] Creating output directories...")
    splits = ({ 'test': test_images } if eval_only else { 'train': train_images, 'val': val_images, 'test': test_images })
    for split in splits.keys():
        os.makedirs(os.path.join(output_dir, split, 'low'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'high'), exist_ok=True)

    # Process images
    print("\n[Step 4/5] Generating synthetic low-light images...")
    print("  This may take time depending on dataset size")

    for split_name, image_list in splits.items():
        print(f"\n  Processing {split_name} set ({len(image_list)} images)...")
        for idx, img_path in enumerate(tqdm(image_list, desc=f"  {split_name}")):
            try:
                img = Image.open(img_path).convert('RGB')
                img_array = np.asarray(img).astype(np.float32) / 255.0

                # identity is the parent folder name before 'face'
                # .../SS_Face/<ID>/face/<file>.jpg → <ID>
                identity = Path(img_path).parent.parent.name
                img_name = Path(img_path).stem + '.png'

                # Create identity subfolders
                person_low_dir = os.path.join(output_dir, split_name, 'low', identity)
                person_high_dir = os.path.join(output_dir, split_name, 'high', identity)
                os.makedirs(person_low_dir, exist_ok=True)
                os.makedirs(person_high_dir, exist_ok=True)

                # Save original as high
                high_path = os.path.join(person_high_dir, img_name)
                img.save(high_path)

                # Low-light synthesis (aligned with LFW defaults)
                low_arr = synthesize_low_light_image(
                    img_array,
                    apply_light_reduction=True,
                    apply_noise=False,
                    apply_white_balance=False,
                    apply_blur=False if not enable_blur else True,
                    reduction_factor=0.01,
                    seed=seed + idx,
                    output_format='numpy'
                )

                low_img = (np.clip(low_arr, 0.0, 1.0) * 255).astype(np.uint8)
                low_path = os.path.join(person_low_dir, img_name)
                Image.fromarray(low_img).save(low_path)
            except Exception as e:
                print(f"    Error processing {img_path}: {e}")
                continue

    # Stats
    print("\n[Step 5/5] Writing dataset statistics...")
    os.makedirs(output_dir, exist_ok=True)
    stats_file = os.path.join(output_dir, 'dataset_stats.txt')
    total = len(train_images) + len(val_images) + len(test_images)
    with open(stats_file, 'w') as f:
        f.write("SS_Face Low-Light Dataset Statistics\n")
        f.write("="*70 + "\n\n")
        f.write(f"Source: {src_root}\n")
        f.write(f"Output: {output_dir}\n\n")
        f.write(f"Total images: {total}\n")
        f.write(f"Identities: {len(id_images)}\n")
        if eval_only:
            f.write(f"  Test (all images): {len(test_images)} from {len(test_ids)} IDs\n\n")
        else:
            f.write(f"  Train: {len(train_images)} from {len(train_ids)} IDs\n")
            f.write(f"  Val:   {len(val_images)} from {len(val_ids)} IDs\n")
            f.write(f"  Test:  {len(test_images)} from {len(test_ids)} IDs\n\n")
        f.write("Split strategy: Identity-based (no leakage)\n")
        f.write("Synthesis: light reduction only (no noise/WB/blur by default)\n")

    print(f"\n  Statistics saved to: {stats_file}")
    print("\n" + "="*70)
    print("Dataset preparation complete!")
    print("="*70)
    print(f"\nDataset location: {output_dir}")
    print("\nDirectory structure (preserves identity):")
    print("  SS_Face_lowlight/")
    if eval_only:
        print("  └── test/")
        print("      ├── low/<ID>/*.png")
        print("      └── high/<ID>/*.png")
    else:
        print("  ├── train/")
        print("  │   ├── low/<ID>/*.png")
        print("  │   └── high/<ID>/*.png")
        print("  ├── val/ (same structure)")
        print("  └── test/ (same structure)")
    return True


def main():
    parser = argparse.ArgumentParser(description='Prepare SS_Face dataset (low-light synthesis)')
    parser.add_argument('--unzip', action='store_true', help='Unzip SS_Face.zip before processing')
    parser.add_argument('--download', action='store_true', help='Alias for --unzip (match LFW script)')
    parser.add_argument('--src_root', type=str, default='./datasets/SS_Face/SS_Face', help='Path to extracted SS_Face root')
    parser.add_argument('--zip_path', type=str, default='./datasets/SS_Face/SS_Face.zip', help='Path to SS_Face.zip')
    parser.add_argument('--output_dir', type=str, default='./datasets/SS_Face_lowlight', help='Output dataset directory')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--min_images', type=int, default=1, help='Min face images per identity')
    parser.add_argument('--max_images', type=int, default=None, help='Limit total images (debug)')
    parser.add_argument('--enable_blur', action='store_true', help='Enable blur during synthesis')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split', action='store_true', help='Create train/val/test splits (default: test-only)')

    args = parser.parse_args()

    if args.unzip or args.download:
        ok = unzip_ss_face(args.zip_path, os.path.dirname(args.zip_path))
        if not ok:
            sys.exit(1)

    ok = prepare_ss_face_lowlight(
        src_root=args.src_root,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_images_per_identity=args.min_images,
        max_images=args.max_images,
        enable_blur=args.enable_blur,
        seed=args.seed,
        eval_only=(not args.split),
    )
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
