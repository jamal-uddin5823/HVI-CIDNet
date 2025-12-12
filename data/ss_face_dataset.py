"""
SS_Face Dataset Loader (train/eval) for low-light enhancement training

Matches the structure produced by prepare_ss_face_dataset.py:
    SS_Face_lowlight/
    ├── train/
    │   ├── low/<ID>/*.png
    │   └── high/<ID>/*.png
    ├── val/
    └── test/
"""

import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from data.util import *


def _collect_recursive(directory):
    """Recursively collect image files from possible identity subfolders.
    Returns a list of relative paths from `directory`.
    """
    image_files = []
    has_subdirs = any(os.path.isdir(join(directory, d)) for d in listdir(directory)) if os.path.isdir(directory) else False
    if has_subdirs:
        for sub in sorted(listdir(directory)):
            subdir = join(directory, sub)
            if not os.path.isdir(subdir):
                continue
            for fn in sorted(listdir(subdir)):
                if is_image_file(fn):
                    image_files.append(join(sub, fn))
    else:
        if os.path.isdir(directory):
            image_files = sorted([x for x in listdir(directory) if is_image_file(x)])
    return image_files


class SSFaceDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform

        self.low_folder = join(data_dir, 'low')
        self.high_folder = join(data_dir, 'high')

        if not os.path.exists(self.low_folder):
            raise FileNotFoundError(f"Low-light directory not found: {self.low_folder}")
        if not os.path.exists(self.high_folder):
            raise FileNotFoundError(f"High (GT) directory not found: {self.high_folder}")

        self.low_filenames = _collect_recursive(self.low_folder)
        self.high_filenames = _collect_recursive(self.high_folder)

        if len(self.low_filenames) != len(self.high_filenames):
            print(f"Warning: #low ({len(self.low_filenames)}) != #high ({len(self.high_filenames)})")

        self.num_images = min(len(self.low_filenames), len(self.high_filenames))
        print(f"[SS_Face Dataset] Loaded {self.num_images} pairs from {data_dir}")

    def __getitem__(self, index):
        low_path = join(self.low_folder, self.low_filenames[index])
        high_path = join(self.high_folder, self.high_filenames[index])

        try:
            im_low = load_img(low_path)
            im_high = load_img(high_path)
        except Exception as e:
            print(f"Error loading index {index}: {e}\n  low={low_path}\n  high={high_path}")
            return self.__getitem__(0)

        if self.transform:
            seed = random.randint(1, 1_000_000)
            seed = np.random.randint(seed)
            random.seed(seed); torch.manual_seed(seed)
            im_low = self.transform(im_low)
            random.seed(seed); torch.manual_seed(seed)
            im_high = self.transform(im_high)

        if torch.isnan(im_low).any() or torch.isinf(im_low).any():
            im_low = torch.clamp(im_low, 0, 1)
        if torch.isnan(im_high).any() or torch.isinf(im_high).any():
            im_high = torch.clamp(im_high, 0, 1)

        return im_low, im_high, self.low_filenames[index], self.high_filenames[index]

    def __len__(self):
        return self.num_images


class SSFaceDatasetFromFolderEval(data.Dataset):
    """Eval dataset. Accepts either the split root (e.g., .../val) or the low folder (.../val/low).
    If a 'high' folder sibling exists, filenames are matched for metrics later.
    """
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.transform = transform

        # Accept both .../val and .../val/low forms.
        if os.path.basename(data_dir.rstrip('/')) == 'low' and os.path.isdir(data_dir):
            self.low_folder = data_dir
            self.high_folder = join(os.path.dirname(data_dir), 'high')
        else:
            self.low_folder = join(data_dir, 'low')
            self.high_folder = join(data_dir, 'high')

        if not os.path.exists(self.low_folder):
            raise FileNotFoundError(f"Low-light directory not found: {self.low_folder}")
        # High may be used by external metrics; enforce presence here for consistency
        if not os.path.exists(self.high_folder):
            raise FileNotFoundError(f"High (GT) directory not found: {self.high_folder}")

        self.low_filenames = _collect_recursive(self.low_folder)
        self.high_filenames = _collect_recursive(self.high_folder)
        self.num_images = min(len(self.low_filenames), len(self.high_filenames))
        print(f"[SS_Face Eval Dataset] Loaded {self.num_images} pairs from {data_dir}")

    def __getitem__(self, index):
        low_path = join(self.low_folder, self.low_filenames[index])
        im_low = load_img(low_path)
        if self.transform:
            im_low = self.transform(im_low)
        return im_low, self.low_filenames[index]

    def __len__(self):
        return self.num_images
