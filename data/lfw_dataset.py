"""
LFW (Labeled Faces in the Wild) Dataset Loader for Low-Light Enhancement

This module provides PyTorch Dataset classes for the LFW dataset with
synthetic low-light images, enabling face recognition-aware low-light
image enhancement training.

Dataset Structure (supports both flat and LFW subdirectory formats):

    Option 1: LFW person subdirectories (recommended)
    LFW_lowlight/
    ├── train/
    │   ├── low/
    │   │   ├── George_W_Bush/
    │   │   │   ├── George_W_Bush_0001.png
    │   │   │   └── George_W_Bush_0002.png
    │   │   └── Colin_Powell/
    │   │       └── Colin_Powell_0001.png
    │   └── high/ (same structure)
    ├── val/
    └── test/

    Option 2: Flat structure (legacy, backward compatible)
    LFW_lowlight/
    ├── train/
    │   ├── low/      # All low-light images in one folder
    │   └── high/     # All ground truth images in one folder
    ├── val/
    └── test/

Usage:
    from data.lfw_dataset import LFWDatasetFromFolder
    dataset = LFWDatasetFromFolder('./datasets/LFW_lowlight/train', transform=transform)
"""

import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from data.util import *


class LFWDatasetFromFolder(data.Dataset):
    """LFW Dataset for low-light image enhancement training

    This dataset loader loads pairs of low-light and ground truth face images
    from the LFW dataset. It's designed for supervised training of low-light
    enhancement models with face recognition awareness.

    Args:
        data_dir (str): Root directory containing 'low' and 'high' subdirectories
        transform (callable, optional): Transformation pipeline for data augmentation

    Directory structure:
        data_dir/
        ├── low/      # Low-light face images
        └── high/     # Ground truth face images

    Returns:
        tuple: (low_light_img, gt_img, low_filename, gt_filename)
            - low_light_img: Low-light image tensor
            - gt_img: Ground truth image tensor
            - low_filename: Filename of low-light image
            - gt_filename: Filename of ground truth image
    """

    def __init__(self, data_dir, transform=None):
        super(LFWDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

        # Get all image files
        self.low_folder = join(data_dir, 'low')
        self.high_folder = join(data_dir, 'high')

        # Verify directories exist
        if not os.path.exists(self.low_folder):
            raise FileNotFoundError(f"Low-light directory not found: {self.low_folder}")
        if not os.path.exists(self.high_folder):
            raise FileNotFoundError(f"Ground truth directory not found: {self.high_folder}")

        # Load filenames (handles both flat structure and LFW person subdirectories)
        def collect_image_files(directory):
            """Recursively collect all image files, preserving relative paths"""
            image_files = []
            # Check if using subdirectories (LFW structure)
            has_subdirs = any(os.path.isdir(join(directory, d)) for d in listdir(directory))

            if has_subdirs:
                # Scan subdirectories (e.g., George_W_Bush/)
                for person_name in sorted(listdir(directory)):
                    person_dir = join(directory, person_name)
                    if not os.path.isdir(person_dir):
                        continue
                    for filename in sorted(listdir(person_dir)):
                        if is_image_file(filename):
                            # Store relative path: person/filename
                            image_files.append(join(person_name, filename))
            else:
                # Flat structure (legacy)
                image_files = sorted([x for x in listdir(directory) if is_image_file(x)])

            return image_files

        self.low_filenames = collect_image_files(self.low_folder)
        self.high_filenames = collect_image_files(self.high_folder)

        # Verify paired data
        if len(self.low_filenames) != len(self.high_filenames):
            print(f"Warning: Number of low-light images ({len(self.low_filenames)}) "
                  f"!= number of ground truth images ({len(self.high_filenames)})")

        self.num_images = min(len(self.low_filenames), len(self.high_filenames))

        print(f"[LFW Dataset] Loaded {self.num_images} image pairs from {data_dir}")

    def __getitem__(self, index):
        """Get a pair of low-light and ground truth images

        Args:
            index (int): Index of the image pair

        Returns:
            tuple: (low_light_img, gt_img, low_filename, gt_filename)
        """
        # Get file paths
        low_path = join(self.low_folder, self.low_filenames[index])
        high_path = join(self.high_folder, self.high_filenames[index])

        # Load images
        im_low = load_img(low_path)
        im_high = load_img(high_path)

        # Apply transforms with same random seed for consistency
        if self.transform:
            # Generate seed for consistent transforms
            seed = random.randint(1, 1000000)
            seed = np.random.randint(seed)

            # Apply to low-light image
            random.seed(seed)
            torch.manual_seed(seed)
            im_low = self.transform(im_low)

            # Apply same transform to ground truth
            random.seed(seed)
            torch.manual_seed(seed)
            im_high = self.transform(im_high)

        return im_low, im_high, self.low_filenames[index], self.high_filenames[index]

    def __len__(self):
        """Return the number of image pairs"""
        return self.num_images


class LFWDatasetFromFolderEval(data.Dataset):
    """LFW Dataset for evaluation (no data augmentation)

    This dataset loader is used for validation and testing. It loads images
    without data augmentation to ensure consistent evaluation.

    Args:
        data_dir (str): Root directory containing 'low' and 'high' subdirectories
        transform (callable, optional): Transformation pipeline (typically just ToTensor)

    Returns:
        tuple: (low_light_img, gt_img, low_filename, gt_filename)
    """

    def __init__(self, data_dir, transform=None):
        super(LFWDatasetFromFolderEval, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

        # Get all image files
        self.low_folder = join(data_dir, 'low')
        self.high_folder = join(data_dir, 'high')

        # Verify directories exist
        if not os.path.exists(self.low_folder):
            raise FileNotFoundError(f"Low-light directory not found: {self.low_folder}")
        if not os.path.exists(self.high_folder):
            raise FileNotFoundError(f"Ground truth directory not found: {self.high_folder}")

        # Load filenames (handles both flat structure and LFW person subdirectories)
        def collect_image_files(directory):
            """Recursively collect all image files, preserving relative paths"""
            image_files = []
            # Check if using subdirectories (LFW structure)
            has_subdirs = any(os.path.isdir(join(directory, d)) for d in listdir(directory))

            if has_subdirs:
                # Scan subdirectories (e.g., George_W_Bush/)
                for person_name in sorted(listdir(directory)):
                    person_dir = join(directory, person_name)
                    if not os.path.isdir(person_dir):
                        continue
                    for filename in sorted(listdir(person_dir)):
                        if is_image_file(filename):
                            # Store relative path: person/filename
                            image_files.append(join(person_name, filename))
            else:
                # Flat structure (legacy)
                image_files = sorted([x for x in listdir(directory) if is_image_file(x)])

            return image_files

        self.low_filenames = collect_image_files(self.low_folder)
        self.high_filenames = collect_image_files(self.high_folder)

        self.num_images = min(len(self.low_filenames), len(self.high_filenames))

        print(f"[LFW Eval Dataset] Loaded {self.num_images} image pairs from {data_dir}")

    def __getitem__(self, index):
        """Get a low-light image for evaluation

        For evaluation, we only need the low-light image and its filename.
        Ground truth is stored separately and used for metrics computation.

        Args:
            index (int): Index of the image

        Returns:
            tuple: (low_light_img, filename)
        """
        # Get file path
        low_path = join(self.low_folder, self.low_filenames[index])

        # Load image
        im_low = load_img(low_path)

        # Apply transforms (no data augmentation, just resize + ToTensor)
        if self.transform:
            im_low = self.transform(im_low)

        # Return only input and filename (matching standard eval dataset format)
        return im_low, self.low_filenames[index]

    def __len__(self):
        """Return the number of image pairs"""
        return self.num_images


def get_lfw_statistics(data_dir):
    """
    Get statistics about the LFW dataset

    Args:
        data_dir (str): Root directory of LFW dataset

    Returns:
        dict: Statistics including image count, mean brightness, etc.
    """
    low_folder = join(data_dir, 'low')
    high_folder = join(data_dir, 'high')

    low_files = [x for x in listdir(low_folder) if is_image_file(x)]
    high_files = [x for x in listdir(high_folder) if is_image_file(x)]

    stats = {
        'num_low': len(low_files),
        'num_high': len(high_files),
        'num_pairs': min(len(low_files), len(high_files))
    }

    return stats


# Test/example usage
if __name__ == '__main__':
    """Test the LFW dataset loader"""
    from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip

    print("Testing LFW Dataset Loader...")
    print("="*70)

    # Example dataset path
    dataset_path = './datasets/LFW_lowlight/train'

    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset not found at {dataset_path}")
        print("Please run: python prepare_lfw_dataset.py --download")
        exit(1)

    # Define transforms
    train_transform = Compose([
        RandomCrop((256, 256)),
        RandomHorizontalFlip(),
        ToTensor(),
    ])

    eval_transform = Compose([ToTensor()])

    # Test training dataset
    print("\n[1] Testing Training Dataset...")
    try:
        train_dataset = LFWDatasetFromFolder(dataset_path, transform=train_transform)
        print(f"  ✓ Dataset loaded: {len(train_dataset)} image pairs")

        # Test loading a sample
        low, high, low_name, high_name = train_dataset[0]
        print(f"  ✓ Sample loaded:")
        print(f"    Low-light shape: {low.shape}")
        print(f"    Ground truth shape: {high.shape}")
        print(f"    Low-light filename: {low_name}")
        print(f"    Ground truth filename: {high_name}")

    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Test evaluation dataset
    print("\n[2] Testing Evaluation Dataset...")
    eval_path = './datasets/LFW_lowlight/val'
    if os.path.exists(eval_path):
        try:
            eval_dataset = LFWDatasetFromFolderEval(eval_path, transform=eval_transform)
            print(f"  ✓ Eval dataset loaded: {len(eval_dataset)} image pairs")

            # Test loading a sample
            low, high, low_name, high_name = eval_dataset[0]
            print(f"  ✓ Sample loaded:")
            print(f"    Low-light shape: {low.shape}")
            print(f"    Ground truth shape: {high.shape}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
    else:
        print(f"  Warning: Eval dataset not found at {eval_path}")

    # Get statistics
    print("\n[3] Dataset Statistics...")
    try:
        stats = get_lfw_statistics(dataset_path)
        print(f"  Low-light images: {stats['num_low']}")
        print(f"  Ground truth images: {stats['num_high']}")
        print(f"  Valid pairs: {stats['num_pairs']}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    print("\n" + "="*70)
    print("Testing complete!")
