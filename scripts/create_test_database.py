#!/usr/bin/env python
"""
Create Test Face Database Script

This script creates a test face database from existing image datasets.
It extracts face images and organizes them into the folder-per-person
structure required by the face recognition app.

Database Structure:
    face_database/
    ├── person_1/
    │   ├── img_001.jpg
    │   ├── img_002.jpg
    │   └── ...
    ├── person_2/
    │   └── img_001.jpg
    └── ...

Usage:
    # Create database from LFW dataset
    python scripts/create_test_database.py \
        --source ./datasets/LFW_multilevel/val_mixed/high \
        --output ./face_database \
        --num_persons 20 \
        --images_per_person 3

    # Create from flat image directory
    python scripts/create_test_database.py \
        --source ./datasets/faces \
        --output ./face_database \
        --num_persons 10 \
        --images_per_person 5 \
        --flat

    # Copy random subset
    python scripts/create_test_database.py \
        --source ./datasets/LFW_multilevel/val_mixed/high \
        --output ./face_database \
        --num_persons 50 \
        --images_per_person 2 \
        --random
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def find_images(directory: Path, extensions: set = None) -> List[Path]:
    """Find all image files in directory recursively

    Args:
        directory: Directory to search
        extensions: Set of valid extensions (default: common image formats)

    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    images = []
    for ext in extensions:
        images.extend(directory.rglob(f'*{ext}'))
        images.extend(directory.rglob(f'*{ext.upper()}'))

    return sorted(images)


def group_images_by_person(
    images: List[Path],
    flat: bool = False
) -> dict:
    """Group images by person ID

    Args:
        images: List of image paths
        flat: If True, treat each image as separate person (for flat directories)

    Returns:
        Dict mapping person_id to list of image paths
    """
    person_groups = {}

    if flat:
        # Each image is a separate person
        for i, img in enumerate(images):
            person_id = f"person_{i+1:04d}"
            person_groups[person_id] = [img]
    else:
        # Group by parent directory name (assumes LFW-style structure)
        for img in images:
            # Get parent directory name as person ID
            parent = img.parent.name

            # Also handle nested structures like person_name/other/images
            if len(img.parts) > 2 and img.parts[-2] != img.parent.name:
                parent = img.parts[-2]

            if parent not in person_groups:
                person_groups[parent] = []
            person_groups[parent].append(img)

    return person_groups


def create_database(
    source_dir: str,
    output_dir: str,
    num_persons: int,
    images_per_person: int,
    flat: bool = False,
    random_select: bool = False,
    seed: int = 42
) -> Tuple[int, int]:
    """Create face database from source images

    Args:
        source_dir: Source directory containing images
        output_dir: Output directory for database
        num_persons: Number of people to include
        images_per_person: Number of images per person
        flat: If True, treat source as flat directory
        random_select: If True, randomly select persons and images
        seed: Random seed for reproducibility

    Returns:
        Tuple of (num_persons_created, total_images_copied)
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Set random seed
    if random_select:
        random.seed(seed)

    # Find all images
    print(f"Scanning source directory: {source_dir}")
    all_images = find_images(source_path)
    print(f"Found {len(all_images)} images")

    if not all_images:
        raise ValueError(f"No images found in {source_dir}")

    # Group by person
    person_groups = group_images_by_person(all_images, flat=flat)
    print(f"Found {len(person_groups)} persons/groups")

    # Select persons
    person_ids = list(person_groups.keys())

    if len(person_ids) < num_persons:
        print(f"Warning: Requested {num_persons} persons but only {len(person_ids)} available")
        num_persons = len(person_ids)

    if random_select:
        person_ids = random.sample(person_ids, num_persons)
        person_ids.sort()
    else:
        person_ids = person_ids[:num_persons]

    # Create database
    print(f"\nCreating database with {num_persons} persons, {images_per_person} images each...")

    total_copied = 0

    for i, person_id in enumerate(person_ids):
        person_dir = output_path / person_id
        person_dir.mkdir(exist_ok=True)

        # Get images for this person
        images = person_groups[person_id]

        if len(images) < images_per_person:
            print(f"  Warning: {person_id} has only {len(images)} images")
            images_to_copy = images
        else:
            if random_select:
                images_to_copy = random.sample(images, images_per_person)
            else:
                images_to_copy = images[:images_per_person]

        # Copy images
        for j, img_path in enumerate(images_to_copy):
            # Create new filename
            ext = img_path.suffix
            new_name = f"img_{j+1:03d}{ext}"
            dest_path = person_dir / new_name

            shutil.copy2(img_path, dest_path)
            total_copied += 1

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{num_persons} persons...")

    print(f"\nDatabase created successfully!")
    print(f"  Location: {output_path}")
    print(f"  Persons: {num_persons}")
    print(f"  Total images: {total_copied}")

    return num_persons, total_copied


def create_synthetic_database(
    output_dir: str,
    num_persons: int,
    images_per_person: int,
    image_size: Tuple[int, int] = (112, 112)
) -> Tuple[int, int]:
    """Create a synthetic face database with colored squares

    This is useful for testing when no real face images are available.
    Each person gets a unique color.

    Args:
        output_dir: Output directory for database
        num_persons: Number of people to create
        images_per_person: Number of images per person
        image_size: Size of each image (width, height)

    Returns:
        Tuple of (num_persons_created, total_images_created)
    """
    from PIL import Image

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Creating synthetic database with {num_persons} persons...")

    total_created = 0

    for i in range(num_persons):
        person_dir = output_path / f"person_{i+1:04d}"
        person_dir.mkdir(exist_ok=True)

        # Generate unique color for this person
        hue = (i * 137.508) % 360  # Golden angle approximation for distribution
        import colorsys
        rgb = colorsys.hsv_to_rgb(hue / 360, 0.7, 0.9)
        color = tuple(int(c * 255) for c in rgb)

        for j in range(images_per_person):
            # Add some variation to each image
            variation = random.randint(-20, 20)
            varied_color = tuple(
                max(0, min(255, c + variation)) for c in color
            )

            img = Image.new('RGB', image_size, color=varied_color)
            img.save(person_dir / f"img_{j+1:03d}.jpg")
            total_created += 1

    print(f"Synthetic database created: {output_path}")
    print(f"  Persons: {num_persons}, Images: {total_created}")

    return num_persons, total_created


def main():
    parser = argparse.ArgumentParser(
        description='Create test face database from existing images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create from LFW-style dataset (subdirectories = persons)
  python create_test_database.py \\
      --source ./datasets/LFW_multilevel/val_mixed/high \\
      --output ./face_database \\
      --num_persons 20 \\
      --images_per_person 3

  # Create from flat directory
  python create_test_database.py \\
      --source ./datasets/faces \\
      --output ./face_database \\
      --num_persons 10 \\
      --images_per_person 5 \\
      --flat

  # Create synthetic database for testing
  python create_test_database.py \\
      --output ./test_database \\
      --num_persons 50 \\
      --images_per_person 3 \\
      --synthetic
        """
    )

    parser.add_argument('--source', type=str,
                       help='Source directory containing face images')
    parser.add_argument('--output', type=str, default='./face_database',
                       help='Output directory for face database (default: ./face_database)')
    parser.add_argument('--num_persons', type=int, default=20,
                       help='Number of persons to include (default: 20)')
    parser.add_argument('--images_per_person', type=int, default=3,
                       help='Number of images per person (default: 3)')
    parser.add_argument('--flat', action='store_true',
                       help='Treat source as flat directory (each image = separate person)')
    parser.add_argument('--random', action='store_true',
                       help='Randomly select persons and images')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--synthetic', action='store_true',
                       help='Create synthetic database with colored squares (for testing)')

    args = parser.parse_args()

    if args.synthetic:
        # Create synthetic database
        create_synthetic_database(
            output_dir=args.output,
            num_persons=args.num_persons,
            images_per_person=args.images_per_person
        )
    elif args.source:
        # Create from source images
        if not os.path.exists(args.source):
            print(f"Error: Source directory not found: {args.source}")
            return 1

        create_database(
            source_dir=args.source,
            output_dir=args.output,
            num_persons=args.num_persons,
            images_per_person=args.images_per_person,
            flat=args.flat,
            random_select=args.random,
            seed=args.seed
        )
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
