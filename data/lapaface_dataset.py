"""
LaPa-Face Dataset Loader for Low-Light Enhancement

This module provides PyTorch Dataset classes for the LaPa-Face dataset with
underexposed (low-light) images, enabling face recognition-aware low-light
image enhancement training.

Dataset Structure:
    LaPa-Face/
    ├── train/
    │   ├── normal/         # Ground truth normal-exposure images
    │   ├── underexposed/   # Low-light/underexposed images
    │   └── seg/            # Face segmentation masks (optional, for future use)
    └── test/
        ├── normal/
        ├── underexposed/
        └── seg/

The dataset expects paired images with the same filename in normal/ and underexposed/ folders.

Usage:
    from data.lapaface_dataset import LaPaFaceDatasetFromFolder
    dataset = LaPaFaceDatasetFromFolder('./datasets/LaPa-Face/train', transform=transform)
"""

import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from data.util import *


class LaPaFaceDatasetFromFolder(data.Dataset):
    """LaPa-Face Dataset for low-light image enhancement training

    This dataset loader loads pairs of underexposed (low-light) and normal-exposure
    face images from the LaPa-Face dataset. It's designed for supervised training
    of low-light enhancement models with face recognition awareness.

    Args:
        data_dir (str): Root directory containing 'underexposed' and 'normal' subdirectories
        transform (callable, optional): Transformation pipeline for data augmentation
        use_seg (bool, optional): Whether to load segmentation masks (default: False)

    Directory structure:
        data_dir/
        ├── underexposed/  # Underexposed/low-light face images
        ├── normal/        # Ground truth normal-exposure face images
        └── seg/           # Face segmentation masks (optional)

    Returns:
        tuple: (underexposed_img, normal_img, underexposed_filename, normal_filename)
            - underexposed_img: Underexposed image tensor
            - normal_img: Normal-exposure ground truth image tensor
            - underexposed_filename: Filename of underexposed image
            - normal_filename: Filename of normal-exposure image
    """

    def __init__(self, data_dir, transform=None, use_seg=False):
        super(LaPaFaceDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.use_seg = use_seg

        # Get all image files
        self.underexposed_folder = join(data_dir, 'underexposed')
        self.normal_folder = join(data_dir, 'normal')
        self.seg_folder = join(data_dir, 'seg') if use_seg else None

        # Verify directories exist
        if not os.path.exists(self.underexposed_folder):
            raise FileNotFoundError(f"Underexposed directory not found: {self.underexposed_folder}")
        if not os.path.exists(self.normal_folder):
            raise FileNotFoundError(f"Normal-exposure directory not found: {self.normal_folder}")
        if use_seg and not os.path.exists(self.seg_folder):
            raise FileNotFoundError(f"Segmentation directory not found: {self.seg_folder}")

        # Load filenames
        self.underexposed_filenames = sorted([x for x in listdir(self.underexposed_folder) if is_image_file(x)])
        self.normal_filenames = sorted([x for x in listdir(self.normal_folder) if is_image_file(x)])

        # Verify paired data
        if len(self.underexposed_filenames) != len(self.normal_filenames):
            print(f"Warning: Number of underexposed images ({len(self.underexposed_filenames)}) "
                  f"!= number of normal images ({len(self.normal_filenames)})")
            print("Attempting to match by filename...")

            # Match by filename (handle potential naming differences)
            underexposed_set = set(self.underexposed_filenames)
            normal_set = set(self.normal_filenames)
            common_files = sorted(underexposed_set.intersection(normal_set))

            if len(common_files) == 0:
                raise ValueError("No matching filenames found between underexposed and normal folders!")

            print(f"Found {len(common_files)} matching pairs")
            self.underexposed_filenames = common_files
            self.normal_filenames = common_files

        self.num_images = len(self.underexposed_filenames)

        print(f"[LaPa-Face Dataset] Loaded {self.num_images} image pairs from {data_dir}")
        if use_seg:
            print(f"[LaPa-Face Dataset] Segmentation masks enabled")

    def __getitem__(self, index):
        """Get a pair of underexposed and normal-exposure images

        Args:
            index (int): Index of the image pair

        Returns:
            tuple: (underexposed_img, normal_img, underexposed_filename, normal_filename)
        """
        # Get file paths
        underexposed_path = join(self.underexposed_folder, self.underexposed_filenames[index])
        normal_path = join(self.normal_folder, self.normal_filenames[index])

        # Load images
        try:
            im_underexposed = load_img(underexposed_path)
            im_normal = load_img(normal_path)
        except Exception as e:
            print(f"Error loading images at index {index}: {e}")
            print(f"  Underexposed: {underexposed_path}")
            print(f"  Normal: {normal_path}")
            # Return first image as fallback
            return self.__getitem__(0)

        # Load segmentation mask if enabled
        if self.use_seg:
            seg_path = join(self.seg_folder, self.underexposed_filenames[index])
            try:
                seg_mask = load_img(seg_path)
            except Exception as e:
                print(f"Warning: Could not load segmentation mask at {seg_path}: {e}")
                seg_mask = None

        # Apply transforms with same random seed for consistency
        if self.transform:
            # Generate seed for consistent transforms
            seed = random.randint(1, 1000000)
            seed = np.random.randint(seed)

            # Apply to underexposed image
            random.seed(seed)
            torch.manual_seed(seed)
            im_underexposed = self.transform(im_underexposed)

            # Apply same transform to normal-exposure image
            random.seed(seed)
            torch.manual_seed(seed)
            im_normal = self.transform(im_normal)

            # Apply same transform to segmentation mask if enabled
            if self.use_seg and seg_mask is not None:
                random.seed(seed)
                torch.manual_seed(seed)
                seg_mask = self.transform(seg_mask)

        # Validate tensors - check for NaN, Inf, or extreme values
        if torch.isnan(im_underexposed).any() or torch.isinf(im_underexposed).any():
            print(f"Warning: Invalid values in underexposed image at index {index}: {underexposed_path}")
            im_underexposed = torch.clamp(im_underexposed, 0, 1)

        if torch.isnan(im_normal).any() or torch.isinf(im_normal).any():
            print(f"Warning: Invalid values in normal image at index {index}: {normal_path}")
            im_normal = torch.clamp(im_normal, 0, 1)

        # Check for extreme values that would cause NaN in loss
        if im_normal.max() > 1e10 or im_underexposed.max() > 1e10:
            print(f"Warning: Extreme values detected at index {index}")
            print(f"  Underexposed range: [{im_underexposed.min()}, {im_underexposed.max()}]")
            print(f"  Normal range: [{im_normal.min()}, {im_normal.max()}]")
            im_underexposed = torch.clamp(im_underexposed, 0, 1)
            im_normal = torch.clamp(im_normal, 0, 1)

        if self.use_seg and seg_mask is not None:
            return im_underexposed, im_normal, seg_mask, self.underexposed_filenames[index], self.normal_filenames[index]
        else:
            return im_underexposed, im_normal, self.underexposed_filenames[index], self.normal_filenames[index]

    def __len__(self):
        """Return the number of image pairs"""
        return self.num_images


class LaPaFaceDatasetFromFolderEval(data.Dataset):
    """LaPa-Face Dataset for evaluation (no data augmentation)

    This dataset loader is used for validation and testing. It loads images
    without data augmentation to ensure consistent evaluation.

    Args:
        data_dir (str): Root directory containing 'underexposed' and 'normal' subdirectories
        transform (callable, optional): Transformation pipeline (typically just ToTensor)

    Returns:
        tuple: (underexposed_img, filename)
    """

    def __init__(self, data_dir, transform=None):
        super(LaPaFaceDatasetFromFolderEval, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

        # Get all image files
        self.underexposed_folder = join(data_dir, 'underexposed')
        self.normal_folder = join(data_dir, 'normal')

        # Verify directories exist
        if not os.path.exists(self.underexposed_folder):
            raise FileNotFoundError(f"Underexposed directory not found: {self.underexposed_folder}")
        if not os.path.exists(self.normal_folder):
            raise FileNotFoundError(f"Normal-exposure directory not found: {self.normal_folder}")

        # Load filenames
        self.underexposed_filenames = sorted([x for x in listdir(self.underexposed_folder) if is_image_file(x)])
        self.normal_filenames = sorted([x for x in listdir(self.normal_folder) if is_image_file(x)])

        # Match filenames
        underexposed_set = set(self.underexposed_filenames)
        normal_set = set(self.normal_filenames)
        common_files = sorted(underexposed_set.intersection(normal_set))

        if len(common_files) == 0:
            raise ValueError("No matching filenames found between underexposed and normal folders!")

        self.underexposed_filenames = common_files
        self.normal_filenames = common_files
        self.num_images = len(common_files)

        print(f"[LaPa-Face Eval Dataset] Loaded {self.num_images} image pairs from {data_dir}")

    def __getitem__(self, index):
        """Get an underexposed image for evaluation

        For evaluation, we only need the underexposed image and its filename.
        Ground truth is stored separately and used for metrics computation.

        Args:
            index (int): Index of the image

        Returns:
            tuple: (underexposed_img, filename)
        """
        # Get file path
        underexposed_path = join(self.underexposed_folder, self.underexposed_filenames[index])

        # Load image
        im_underexposed = load_img(underexposed_path)

        # Apply transforms (no data augmentation, just resize + ToTensor)
        if self.transform:
            im_underexposed = self.transform(im_underexposed)

        # Return only input and filename (matching standard eval dataset format)
        return im_underexposed, self.underexposed_filenames[index]

    def __len__(self):
        """Return the number of image pairs"""
        return self.num_images


def get_lapaface_statistics(data_dir):
    """
    Get statistics about the LaPa-Face dataset

    Args:
        data_dir (str): Root directory of LaPa-Face dataset

    Returns:
        dict: Statistics including image count, mean brightness, etc.
    """
    underexposed_folder = join(data_dir, 'underexposed')
    normal_folder = join(data_dir, 'normal')

    underexposed_files = [x for x in listdir(underexposed_folder) if is_image_file(x)]
    normal_files = [x for x in listdir(normal_folder) if is_image_file(x)]

    stats = {
        'num_underexposed': len(underexposed_files),
        'num_normal': len(normal_files),
        'num_pairs': min(len(underexposed_files), len(normal_files))
    }

    return stats


# Test/example usage
if __name__ == '__main__':
    """Test the LaPa-Face dataset loader"""
    from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, Resize

    print("Testing LaPa-Face Dataset Loader...")
    print("="*70)

    # Example dataset path (assumes unzipped)
    dataset_path = './datasets/LaPa-Face/train'

    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset not found at {dataset_path}")
        print("Please unzip LaPa-Face-train.zip first")
        exit(1)

    # Define transforms
    train_transform = Compose([
        Resize((288, 288)),  # Resize to slightly larger than crop size
        RandomCrop((256, 256)),
        RandomHorizontalFlip(),
        ToTensor(),
    ])

    eval_transform = Compose([
        Resize((256, 256)),
        ToTensor()
    ])

    # Test training dataset
    print("\n[1] Testing Training Dataset...")
    try:
        train_dataset = LaPaFaceDatasetFromFolder(dataset_path, transform=train_transform)
        print(f"  ✓ Dataset loaded: {len(train_dataset)} image pairs")

        # Test loading a sample
        under, normal, under_name, normal_name = train_dataset[0]
        print(f"  ✓ Sample loaded:")
        print(f"    Underexposed shape: {under.shape}")
        print(f"    Normal-exposure shape: {normal.shape}")
        print(f"    Underexposed filename: {under_name}")
        print(f"    Normal filename: {normal_name}")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

    # Test evaluation dataset
    print("\n[2] Testing Evaluation Dataset...")
    test_path = './datasets/LaPa-Face/test'
    if os.path.exists(test_path):
        try:
            eval_dataset = LaPaFaceDatasetFromFolderEval(test_path, transform=eval_transform)
            print(f"  ✓ Eval dataset loaded: {len(eval_dataset)} image pairs")

            # Test loading a sample
            under, under_name = eval_dataset[0]
            print(f"  ✓ Sample loaded:")
            print(f"    Underexposed shape: {under.shape}")
            print(f"    Filename: {under_name}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  Warning: Test dataset not found at {test_path}")

    # Get statistics
    print("\n[3] Dataset Statistics...")
    try:
        stats = get_lapaface_statistics(dataset_path)
        print(f"  Underexposed images: {stats['num_underexposed']}")
        print(f"  Normal-exposure images: {stats['num_normal']}")
        print(f"  Valid pairs: {stats['num_pairs']}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    print("\n" + "="*70)
    print("Testing complete!")
