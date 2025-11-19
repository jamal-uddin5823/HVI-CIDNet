"""
Generate LFW Pairs for Face Verification Evaluation

This script generates pairs.txt following the standard LFW verification protocol:
- Genuine pairs (same person): Enhanced low-light image vs. ground truth
- Impostor pairs (different people): Enhanced image of person A vs. GT of person B

This enables proper face verification evaluation to measure:
1. True Accept Rate (TAR): % of genuine pairs correctly verified
2. False Accept Rate (FAR): % of impostor pairs incorrectly accepted
3. Verification Accuracy at various thresholds
4. Equal Error Rate (EER): Where TAR = 1 - FAR

Usage:
    # Generate 1000 genuine + 1000 impostor pairs
    python generate_lfw_pairs.py --test_dir=./datasets/LFW_lowlight/test --num_pairs=1000

    # Custom output location
    python generate_lfw_pairs.py --test_dir=./datasets/LFW_small/test --output=./pairs_small.txt
"""

import os
import argparse
import random
from pathlib import Path


def generate_pairs(test_dir, num_pairs=1000, output_file='pairs.txt', seed=42):
    """
    Generate genuine and impostor pairs for face verification evaluation.

    Args:
        test_dir: Directory containing test/low and test/high subdirectories
        num_pairs: Number of pairs of each type (genuine and impostor)
        output_file: Output file path for pairs.txt
        seed: Random seed for reproducibility

    Returns:
        tuple: (num_genuine, num_impostor) pairs generated
    """
    random.seed(seed)

    low_dir = os.path.join(test_dir, 'low')
    high_dir = os.path.join(test_dir, 'high')

    # Check if directories exist
    if not os.path.exists(low_dir):
        raise FileNotFoundError(f"Low-light directory not found: {low_dir}")
    if not os.path.exists(high_dir):
        raise FileNotFoundError(f"High-quality directory not found: {high_dir}")

    # Scan for images in person subdirectories (follows LFW structure)
    # Structure: low/George_W_Bush/George_W_Bush_0001.png
    person_to_images = defaultdict(list)

    # Check if using directory structure or flat structure
    has_subdirs = any(os.path.isdir(os.path.join(low_dir, d)) for d in os.listdir(low_dir))

    if has_subdirs:
        print("Detected LFW directory structure (person subdirectories)")
        # Scan subdirectories
        for person_name in os.listdir(low_dir):
            person_low_dir = os.path.join(low_dir, person_name)
            if not os.path.isdir(person_low_dir):
                continue

            # Get images for this person
            images = sorted([f for f in os.listdir(person_low_dir)
                           if f.endswith(('.png', '.jpg', '.jpeg'))])

            for filename in images:
                image_basename = os.path.splitext(filename)[0]
                # Store as person/image_basename for proper path construction
                person_to_images[person_name].append(f"{person_name}/{image_basename}")

        total_images = sum(len(imgs) for imgs in person_to_images.values())
        print(f"Found {total_images} images from {len(person_to_images)} people")
    else:
        print("Detected flat directory structure (extracting person from filename)")
        # Flat structure: extract person name from filename
        low_files = sorted([f for f in os.listdir(low_dir)
                           if f.endswith(('.png', '.jpg', '.jpeg'))])

        def extract_person_name(filename):
            """Extract person identity from filename (e.g., George_W_Bush_0001.png -> George_W_Bush)"""
            basename = os.path.splitext(filename)[0]
            # Split by underscore and remove last part (image number)
            parts = basename.split('_')
            if len(parts) > 1 and parts[-1].isdigit():
                return '_'.join(parts[:-1])
            return basename  # Fallback if format doesn't match

        for filename in low_files:
            person_name = extract_person_name(filename)
            image_basename = os.path.splitext(filename)[0]
            person_to_images[person_name].append(image_basename)

        print(f"Found {len(low_files)} images from {len(person_to_images)} people")

    people = list(person_to_images.keys())
    print(f"Found {len(people)} unique identities")

    # Filter people with at least min_images_per_person images
    people = [p for p in people if len(person_to_images[p]) >= min_images_per_person]
    print(f"  {len(people)} people with at least {min_images_per_person} images")

    if len(people) < 2:
        raise ValueError(f"Need at least 2 people to generate pairs (found {len(people)})")

    # Adjust num_pairs based on available data
    total_images = sum(len(person_to_images[p]) for p in people)
    max_genuine_pairs = total_images  # Each image can form a genuine pair with itself
    max_impostor_pairs = len(people) * (len(people) - 1) * 10  # Conservative estimate

    if num_pairs > max_genuine_pairs:
        print(f"⚠ Warning: Requested {num_pairs} genuine pairs, but only {max_genuine_pairs} images available")
        num_pairs = max_genuine_pairs

    # Generate genuine pairs (same person: low-light vs. ground truth of same image)
    print(f"\nGenerating {num_pairs} genuine pairs (same person)...")
    genuine_pairs = []

    # Collect all images
    all_images = []
    for person in people:
        all_images.extend([(person, img) for img in person_to_images[person]])

    # Sample images for genuine pairs
    if len(all_images) < num_pairs:
        sampled = all_images
    else:
        sampled = random.sample(all_images, num_pairs)

    for person, img_name in sampled:
        # Genuine pair: same image, low vs high (label=1)
        genuine_pairs.append((img_name, img_name, 1))

    print(f"  ✓ Generated {len(genuine_pairs)} genuine pairs")

    # Generate impostor pairs (different people)
    print(f"Generating {num_pairs} impostor pairs (different people)...")
    impostor_pairs = []
    attempts = 0
    max_attempts = num_pairs * 20  # Avoid infinite loop

    while len(impostor_pairs) < num_pairs and attempts < max_attempts:
        # Pick two different people
        person1, person2 = random.sample(people, 2)

        # Pick random image from each person
        img1 = random.choice(person_to_images[person1])
        img2 = random.choice(person_to_images[person2])

        pair = (img1, img2, 0)

        # Avoid duplicates and reversed duplicates
        if pair not in impostor_pairs and (img2, img1, 0) not in impostor_pairs:
            impostor_pairs.append(pair)

        attempts += 1

    print(f"  ✓ Generated {len(impostor_pairs)} impostor pairs")

    # Combine and shuffle
    all_pairs = genuine_pairs + impostor_pairs
    random.shuffle(all_pairs)

    # Write to file
    print(f"\nWriting pairs to: {output_file}")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    with open(output_file, 'w') as f:
        # Write header
        f.write(f"# LFW Face Verification Pairs\n")
        f.write(f"# Format: low_image_name high_image_name label\n")
        f.write(f"#   label=1: genuine pair (same person)\n")
        f.write(f"#   label=0: impostor pair (different people)\n")
        f.write(f"# Total pairs: {len(all_pairs)} ({len(genuine_pairs)} genuine + {len(impostor_pairs)} impostor)\n")
        f.write(f"#\n")

        # Write pairs
        for low_name, high_name, label in all_pairs:
            f.write(f"{low_name} {high_name} {label}\n")

    print(f"  ✓ Wrote {len(all_pairs)} pairs")
    print(f"\nSummary:")
    print(f"  Genuine pairs (same person):     {len(genuine_pairs)}")
    print(f"  Impostor pairs (different people): {len(impostor_pairs)}")
    print(f"  Total pairs:                      {len(all_pairs)}")
    print(f"\nPairs file ready for evaluation!")

    return len(genuine_pairs), len(impostor_pairs)


def main():
    parser = argparse.ArgumentParser(description='Generate LFW pairs for face verification')
    parser.add_argument('--test_dir', type=str, default='./datasets/LFW_lowlight/test',
                       help='Test directory containing low/ and high/ subdirectories')
    parser.add_argument('--num_pairs', type=int, default=1000,
                       help='Number of pairs of each type (genuine and impostor)')
    parser.add_argument('--output', type=str, default='pairs.txt',
                       help='Output file path for pairs.txt')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    print("="*70)
    print("LFW Pairs Generation for Face Verification Evaluation")
    print("="*70)
    print(f"Test directory: {args.test_dir}")
    print(f"Pairs per type: {args.num_pairs}")
    print(f"Output file:    {args.output}")
    print(f"Random seed:    {args.seed}")
    print("="*70)
    print()

    try:
        generate_pairs(
            test_dir=args.test_dir,
            num_pairs=args.num_pairs,
            output_file=args.output,
            seed=args.seed
        )
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
