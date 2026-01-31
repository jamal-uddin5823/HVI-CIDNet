"""
Face Database Management Module

This module handles loading, preprocessing, and matching of face images
for face recognition applications. It supports a folder-per-person database
structure and caches embeddings for efficient matching.

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
    from face_database import FaceDatabase
    from recognizers import AdaFaceRecognizer

    recognizer = AdaFaceRecognizer(device='cuda')
    db = FaceDatabase('face_database', recognizer, device='cuda')

    # Match a query face
    query_embedding = recognizer.get_embedding(query_image)
    results = db.match(query_embedding, top_k=5)
    # Returns: [('person_1', 0.95, '/path/to/img.jpg'), ...]
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm


class FaceDatabase:
    """Face database for storing and matching face embeddings

    Attributes:
        db_path: Path to face database directory
        recognizer: Face recognizer model for extracting embeddings
        device: Device to use for computations
        embeddings: Dict mapping person_id to their embedding tensor
        image_paths: Dict mapping person_id to list of image paths
        person_ids: List of all person IDs in the database
    """

    def __init__(
        self,
        db_path: str,
        recognizer,
        device: str = 'cuda',
        cache_file: Optional[str] = None
    ):
        """Initialize face database

        Args:
            db_path: Path to face database directory
            recognizer: Face recognizer instance (AdaFaceRecognizer or InsightFaceRecognizer)
            device: Device to use ('cuda' or 'cpu')
            cache_file: Optional path to cache embeddings for faster loading
        """
        self.db_path = Path(db_path)
        self.recognizer = recognizer
        self.device = device
        self.cache_file = cache_file or str(self.db_path / 'embeddings_cache.pth')

        # Storage
        self.embeddings: Dict[str, torch.Tensor] = {}
        self.image_paths: Dict[str, List[str]] = {}
        self.person_ids: List[str] = []

        # Load database
        if self.db_path.exists():
            self.load_database()
        else:
            print(f"Warning: Database path does not exist: {self.db_path}")
            print(f"Creating empty database. Use load_database() after adding images.")

    def load_database(self, db_path: Optional[str] = None) -> None:
        """Load face database from directory structure

        Scans the database directory and extracts face embeddings for each person.
        Each subfolder represents one person.

        Args:
            db_path: Optional path to database (uses self.db_path if not provided)
        """
        if db_path:
            self.db_path = Path(db_path)

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database path not found: {self.db_path}")

        print(f"Loading face database from: {self.db_path}")

        # Try to load from cache first
        if self._load_from_cache():
            print(f"  Loaded {len(self.person_ids)} people from cache")
            return

        # Scan for person directories
        self.person_ids = []
        self.embeddings = {}
        self.image_paths = {}

        person_dirs = [d for d in self.db_path.iterdir() if d.is_dir()]

        if not person_dirs:
            print(f"  Warning: No person subdirectories found in {self.db_path}")
            return

        # Transform for loading images
        to_tensor = transforms.ToTensor()

        # Process each person directory
        for person_dir in tqdm(person_dirs, desc="Loading database"):
            person_id = person_dir.name

            # Find all images
            image_files = self._find_images(person_dir)

            if not image_files:
                continue

            # Extract embeddings for all images and average
            embeddings_list = []
            valid_paths = []

            for img_path in image_files:
                try:
                    # Load and preprocess image
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = to_tensor(img).unsqueeze(0).to(self.device)

                    # Extract embedding
                    embedding = self.recognizer.get_embedding(img_tensor)
                    embeddings_list.append(embedding.squeeze(0))
                    valid_paths.append(str(img_path))

                except Exception as e:
                    print(f"  Error loading {img_path}: {e}")
                    continue

            if embeddings_list:
                # Average embeddings from multiple images
                avg_embedding = torch.stack(embeddings_list).mean(dim=0)
                self.embeddings[person_id] = avg_embedding
                self.image_paths[person_id] = valid_paths
                self.person_ids.append(person_id)

        print(f"  Loaded {len(self.person_ids)} people with {sum(len(v) for v in self.image_paths.values())} images")

        # Save to cache
        self._save_to_cache()

    def match(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5
    ) -> List[Tuple[str, float, str]]:
        """Find top-K matching faces in database

        Computes cosine similarity between query embedding and all database
        embeddings, returning the top-K matches.

        Args:
            query_embedding: Query face embedding (512-dim tensor)
            top_k: Number of top matches to return

        Returns:
            List of tuples: [(person_id, confidence, image_path), ...]
            Sorted by confidence (descending)
        """
        if not self.embeddings:
            return []

        # Ensure query embedding is normalized
        if query_embedding.dim() > 1:
            query_embedding = query_embedding.squeeze(0)
        query_embedding = F.normalize(query_embedding.unsqueeze(0), p=2, dim=1)

        # Compute similarities with all database entries
        similarities = []
        for person_id in self.person_ids:
            db_embedding = self.embeddings[person_id].to(self.device)
            db_embedding = F.normalize(db_embedding.unsqueeze(0), p=2, dim=1)

            # Cosine similarity
            sim = (query_embedding * db_embedding).sum().item()

            # Get first image path for display
            img_path = self.image_paths[person_id][0] if self.image_paths[person_id] else ""

            similarities.append((person_id, sim, img_path))

        # Sort by similarity (descending) and return top-K
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def add_person(
        self,
        person_id: str,
        image_paths: List[str],
        recompute_embedding: bool = True
    ) -> None:
        """Add a new person to the database

        Args:
            person_id: Unique identifier for the person
            image_paths: List of image paths for this person
            recompute_embedding: Whether to recompute the average embedding
        """
        if person_id in self.embeddings:
            print(f"Warning: {person_id} already exists in database. Updating...")

        self.image_paths[person_id] = image_paths

        if recompute_embedding:
            self._recompute_person_embedding(person_id)

            if person_id not in self.person_ids:
                self.person_ids.append(person_id)

    def _recompute_person_embedding(self, person_id: str) -> None:
        """Recompute embedding for a specific person"""
        if person_id not in self.image_paths:
            return

        to_tensor = transforms.ToTensor()
        embeddings_list = []

        for img_path in self.image_paths[person_id]:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = to_tensor(img).unsqueeze(0).to(self.device)
                embedding = self.recognizer.get_embedding(img_tensor)
                embeddings_list.append(embedding.squeeze(0))
            except Exception as e:
                print(f"  Error loading {img_path}: {e}")
                continue

        if embeddings_list:
            avg_embedding = torch.stack(embeddings_list).mean(dim=0)
            self.embeddings[person_id] = avg_embedding

    def _find_images(self, directory: Path) -> List[Path]:
        """Find all image files in a directory"""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        return [f for f in directory.iterdir() if f.suffix.lower() in extensions]

    def _save_to_cache(self) -> None:
        """Save embeddings to cache file"""
        try:
            cache_data = {
                'embeddings': {k: v.cpu() for k, v in self.embeddings.items()},
                'image_paths': self.image_paths,
                'person_ids': self.person_ids
            }
            torch.save(cache_data, self.cache_file)
            print(f"  Cached embeddings to: {self.cache_file}")
        except Exception as e:
            print(f"  Warning: Could not save cache: {e}")

    def _load_from_cache(self) -> bool:
        """Load embeddings from cache file

        Returns:
            True if cache was loaded successfully, False otherwise
        """
        if not Path(self.cache_file).exists():
            return False

        try:
            cache_data = torch.load(self.cache_file, map_location=self.device)
            self.embeddings = {k: v.to(self.device) for k, v in cache_data['embeddings'].items()}
            self.image_paths = cache_data['image_paths']
            self.person_ids = cache_data['person_ids']
            return True
        except Exception as e:
            print(f"  Warning: Could not load cache: {e}")
            return False

    def clear_cache(self) -> None:
        """Remove cached embeddings file"""
        if Path(self.cache_file).exists():
            Path(self.cache_file).unlink()
            print(f"  Cache cleared: {self.cache_file}")

    def get_stats(self) -> Dict:
        """Get database statistics

        Returns:
            Dict with database statistics
        """
        return {
            'num_people': len(self.person_ids),
            'total_images': sum(len(v) for v in self.image_paths.values()),
            'embedding_dim': list(self.embeddings.values())[0].shape[0] if self.embeddings else 0
        }


def create_test_database(
    source_dir: str,
    output_dir: str,
    num_persons: int = 20,
    images_per_person: int = 3
) -> None:
    """Create a test face database from existing images

    Args:
        source_dir: Source directory containing images
        output_dir: Output directory for the database
        num_persons: Number of people to include
        images_per_person: Number of images per person
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    # Find all images
    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    all_images = [f for f in source_path.rglob('*') if f.suffix.lower() in extensions]

    if len(all_images) < num_persons * images_per_person:
        print(f"Warning: Not enough images. Found {len(all_images)}, need {num_persons * images_per_person}")

    # Group images by parent directory (assumes each dir = person)
    person_groups = {}
    for img in all_images:
        parent = img.parent.name
        if parent not in person_groups:
            person_groups[parent] = []
        person_groups[parent].append(img)

    # Select first N persons
    selected_persons = list(person_groups.keys())[:num_persons]

    for person_id in selected_persons:
        person_dir = output_path / person_id
        person_dir.mkdir(exist_ok=True)

        images = person_groups[person_id][:images_per_person]
        for i, img in enumerate(images):
            dest = person_dir / f"img_{i+1:03d}{img.suffix}"
            shutil.copy2(img, dest)

    print(f"Created test database with {len(selected_persons)} persons")
    print(f"Database location: {output_path}")


if __name__ == "__main__":
    # Test the face database
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from recognizers import AdaFaceRecognizer

    # Create recognizer
    recognizer = AdaFaceRecognizer(device='cpu')

    # Create test database if needed
    test_db_path = Path(__file__).parent / 'test_database'

    if not test_db_path.exists():
        print("Creating test database...")
        # Use a sample source directory
        create_test_database(
            source_dir='./datasets/LFW_multilevel/val_mixed/high',
            output_dir=str(test_db_path),
            num_persons=10,
            images_per_person=3
        )

    # Load database
    db = FaceDatabase(str(test_db_path), recognizer, device='cpu')
    print(f"Database stats: {db.get_stats()}")

    # Test matching
    if db.person_ids:
        test_person = db.person_ids[0]
        test_embedding = db.embeddings[test_person]
        results = db.match(test_embedding, top_k=5)

        print(f"\nTop matches for {test_person}:")
        for person_id, confidence, img_path in results:
            print(f"  {person_id}: {confidence:.4f}")
