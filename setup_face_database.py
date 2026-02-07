#!/usr/bin/env python3
"""
Setup script to create face database from LFW dataset

This script:
1. Creates a face database from the LFW_original dataset
2. Extracts and caches face embeddings for efficient matching
3. Prepares the database for the face recognition app

Usage:
    python setup_face_database.py --lfw_path datasets/LFW_original/lfw --db_path face_database
"""

import argparse
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

from face_database import FaceDatabase
from recognizers import AdaFaceRecognizer


def setup_database(lfw_path, db_path, recognizer_type='AdaFace', device='cuda'):
    """
    Create face database from LFW dataset
    
    Args:
        lfw_path: Path to LFW dataset (contains person folders)
        db_path: Output path for face database
        recognizer_type: Type of recognizer to use ('AdaFace' or 'InsightFace')
        device: Device to use ('cuda' or 'cpu')
    """
    lfw_path = Path(lfw_path)
    db_path = Path(db_path)
    
    # Check CUDA availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = 'cpu'
    
    # Create database directory if it doesn't exist
    if db_path.exists():
        print(f"\nWarning: Database directory already exists: {db_path}")
        response = input("Do you want to overwrite it? (y/n): ").lower()
        if response == 'y':
            shutil.rmtree(db_path)
            print(f"Removed existing database at {db_path}")
        else:
            print("Aborting...")
            return
    
    db_path.mkdir(parents=True, exist_ok=True)
    
    # Load recognizer
    print(f"\nLoading {recognizer_type} recognizer...")
    if recognizer_type.lower() == 'adaface':
        recognizer = AdaFaceRecognizer(
            checkpoint_path='weights/adaface/adaface_ir50_webface4m.ckpt',
            device=device
        )
    else:
        from recognizers import InsightFaceRecognizer
        recognizer = InsightFaceRecognizer(device=device)
    
    print(f"Using device: {device}")
    
    # Find all person folders in LFW
    person_folders = sorted([d for d in lfw_path.iterdir() if d.is_dir()])
    print(f"\nFound {len(person_folders)} people in LFW dataset")
    
    # Create database structure
    print(f"\nCreating database at: {db_path}")
    for person_dir in tqdm(person_folders, desc="Copying LFW images to database"):
        person_name = person_dir.name
        target_dir = db_path / person_name
        target_dir.mkdir(exist_ok=True)
        
        # Copy all images
        for img_file in person_dir.glob('*.jpg'):
            shutil.copy2(img_file, target_dir / img_file.name)
    
    # Load database and extract embeddings
    print(f"\nLoading and caching embeddings...")
    # use_face_detection=False because LFW images are pre-cropped faces
    face_db = FaceDatabase(str(db_path), recognizer, device=device, use_face_detection=False)
    
    print(f"\n✓ Database successfully created!")
    print(f"  Location: {db_path}")
    print(f"  People: {len(face_db.person_ids)}")
    print(f"  Total images: {sum(len(v) for v in face_db.image_paths.values())}")
    print(f"  Cache file: {face_db.cache_file}")
    
    return face_db


def main():
    parser = argparse.ArgumentParser(
        description='Setup face database from LFW dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default setup (LFW_original -> face_database)
  python setup_face_database.py

  # Custom paths
  python setup_face_database.py --lfw_path datasets/LFW_original/lfw --db_path my_face_db

  # Use CPU
  python setup_face_database.py --device cpu
        """
    )
    
    parser.add_argument(
        '--lfw_path',
        type=str,
        default='datasets/LFW_original/lfw',
        help='Path to LFW dataset folder (default: datasets/LFW_original/lfw)'
    )
    parser.add_argument(
        '--db_path',
        type=str,
        default='face_database',
        help='Output path for face database (default: face_database)'
    )
    parser.add_argument(
        '--recognizer',
        type=str,
        default='AdaFace',
        choices=['AdaFace', 'InsightFace'],
        help='Face recognizer type (default: AdaFace)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )
    
    args = parser.parse_args()
    
    # Check if LFW path exists
    if not Path(args.lfw_path).exists():
        print(f"Error: LFW path does not exist: {args.lfw_path}")
        return
    
    # Setup database
    setup_database(args.lfw_path, args.db_path, args.recognizer, args.device)


if __name__ == '__main__':
    main()
