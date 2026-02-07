"""
Face Database Management with Embedding Extraction and Matching

This module provides face database functionality with caching support
and similarity-based matching.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np


class FaceDatabase:
    """Manages face database with embedding extraction and similarity matching"""

    def __init__(self, db_path: str, recognizer, device: str = 'cuda',
                 use_face_detection: bool = False, cache_file: Optional[str] = None):
        """Initialize face database

        Args:
            db_path: Path to database directory (LFW format: person_name/image.jpg)
            recognizer: Face recognizer instance (AdaFace or InsightFace)
            device: Device to use ('cuda' or 'cpu')
            use_face_detection: Whether to use face detection (False for pre-cropped LFW)
            cache_file: Path to cache file for embeddings (auto-generated if None)
        """
        self.db_path = Path(db_path)
        self.recognizer = recognizer
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_face_detection = use_face_detection

        # Set cache file path
        if cache_file is None:
            recognizer_name = type(recognizer).__name__.replace('Recognizer', '').lower()
            self.cache_file = self.db_path / f'embeddings_cache_{recognizer_name}.pth'
        else:
            self.cache_file = Path(cache_file)

        # Database storage
        self.embeddings: Dict[str, List[torch.Tensor]] = {}  # person_id -> list of embeddings
        self.image_paths: Dict[str, List[str]] = {}  # person_id -> list of image paths
        self.person_ids: List[str] = []

        # Load database
        self.load_database()

    def load_database(self) -> Dict[str, List[torch.Tensor]]:
        """Load database from cache or compute embeddings

        Returns:
            Dictionary mapping person_id to list of embeddings
        """
        # Try to load from cache
        if self.cache_file.exists():
            print(f"Loading from cache: {self.cache_file}")
            cache_data = torch.load(self.cache_file, map_location=self.device)
            self.embeddings = cache_data['embeddings']
            self.image_paths = cache_data['image_paths']
            self.person_ids = list(self.embeddings.keys())
            print(f"✓ Loaded {len(self.person_ids)} people, {sum(len(v) for v in self.embeddings.values())} embeddings")
            return self.embeddings

        # Build database from scratch
        print(f"Building database from: {self.db_path}")
        if not self.db_path.exists():
            raise ValueError(f"Database path does not exist: {self.db_path}")

        # Scan directory structure
        person_dirs = [d for d in self.db_path.iterdir() if d.is_dir()]
        if len(person_dirs) == 0:
            raise ValueError(f"No person directories found in {self.db_path}")

        # Extract embeddings for each person
        for person_dir in tqdm(person_dirs, desc="Processing people"):
            person_id = person_dir.name
            image_files = list(person_dir.glob('*.jpg')) + list(person_dir.glob('*.png'))

            if len(image_files) == 0:
                continue

            person_embeddings = []
            person_image_paths = []

            for img_path in image_files:
                try:
                    # Load image
                    image = Image.open(img_path).convert('RGB')

                    # Extract embedding
                    embedding = self.recognizer.get_embedding(
                        image,
                        use_face_detection=self.use_face_detection
                    )

                    person_embeddings.append(embedding.cpu())
                    person_image_paths.append(str(img_path))

                except Exception as e:
                    print(f"Warning: Failed to process {img_path}: {e}")
                    continue

            if len(person_embeddings) > 0:
                self.embeddings[person_id] = person_embeddings
                self.image_paths[person_id] = person_image_paths

        self.person_ids = list(self.embeddings.keys())

        # Save cache
        print(f"Saving embeddings cache to: {self.cache_file}")
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'embeddings': self.embeddings,
            'image_paths': self.image_paths
        }, self.cache_file)

        print(f"Database built: {len(self.person_ids)} people with {sum(len(v) for v in self.embeddings.values())} total embeddings")

        return self.embeddings

    def add_person(self, person_id: str, image_paths: List[str]):
        """Add a person to the database

        Args:
            person_id: Person identifier
            image_paths: List of image file paths for this person
        """
        person_embeddings = []
        person_image_paths = []

        for img_path in tqdm(image_paths, desc=f"Adding {person_id}"):
            try:
                # Load image
                image = Image.open(img_path).convert('RGB')

                # Extract embedding
                embedding = self.recognizer.get_embedding(
                    image,
                    use_face_detection=self.use_face_detection
                )

                person_embeddings.append(embedding.cpu())
                person_image_paths.append(str(img_path))

            except Exception as e:
                print(f"Warning: Failed to process {img_path}: {e}")
                continue

        if len(person_embeddings) > 0:
            if person_id in self.embeddings:
                # Append to existing person
                self.embeddings[person_id].extend(person_embeddings)
                self.image_paths[person_id].extend(person_image_paths)
            else:
                # Add new person
                self.embeddings[person_id] = person_embeddings
                self.image_paths[person_id] = person_image_paths
                self.person_ids.append(person_id)

            # Update cache
            torch.save({
                'embeddings': self.embeddings,
                'image_paths': self.image_paths
            }, self.cache_file)

            print(f"Added {person_id} with {len(person_embeddings)} embeddings")

    def match(self, query_embedding: torch.Tensor, top_k: int = 5,
              threshold: float = 0.0) -> List[Tuple[str, float, str]]:
        """Find top-K matches for query embedding

        Args:
            query_embedding: Query embedding tensor of shape (1, 512)
            top_k: Number of top matches to return
            threshold: Minimum similarity threshold (0.0 = no threshold)

        Returns:
            List of tuples (person_id, similarity_score, image_path)
            sorted by similarity score (descending)
        """
        if len(self.embeddings) == 0:
            return []

        # Normalize query embedding
        query_embedding = F.normalize(query_embedding, p=2, dim=1)

        # Compute similarities for all database embeddings
        matches = []

        for person_id in self.person_ids:
            person_embeddings = self.embeddings[person_id]
            person_paths = self.image_paths[person_id]

            for embedding, img_path in zip(person_embeddings, person_paths):
                # Move to same device as query
                embedding = embedding.to(query_embedding.device)

                # Normalize embedding
                embedding = F.normalize(embedding, p=2, dim=1)

                # Compute cosine similarity
                similarity = F.cosine_similarity(query_embedding, embedding, dim=1)
                similarity_score = similarity.item()

                # Apply threshold
                if similarity_score >= threshold:
                    matches.append((person_id, similarity_score, img_path))

        # Sort by similarity score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)

        # Return top-K
        return matches[:top_k]

    def get_stats(self) -> Dict:
        """Get database statistics

        Returns:
            Dictionary with statistics
        """
        total_embeddings = sum(len(v) for v in self.embeddings.values())
        avg_embeddings_per_person = total_embeddings / len(self.person_ids) if len(self.person_ids) > 0 else 0

        return {
            'num_people': len(self.person_ids),
            'total_embeddings': total_embeddings,
            'avg_embeddings_per_person': avg_embeddings_per_person,
            'cache_file': str(self.cache_file),
            'cache_exists': self.cache_file.exists()
        }

    def clear_cache(self):
        """Clear the embeddings cache file"""
        if self.cache_file.exists():
            self.cache_file.unlink()
            print(f"Cache cleared: {self.cache_file}")
        else:
            print(f"No cache file to clear: {self.cache_file}")

    def reload(self):
        """Reload database from cache or recompute"""
        self.embeddings = {}
        self.image_paths = {}
        self.person_ids = []
        self.load_database()

    def get_person_embeddings(self, person_id: str) -> List[torch.Tensor]:
        """Get all embeddings for a specific person

        Args:
            person_id: Person identifier

        Returns:
            List of embedding tensors
        """
        return self.embeddings.get(person_id, [])

    def get_person_images(self, person_id: str) -> List[str]:
        """Get all image paths for a specific person

        Args:
            person_id: Person identifier

        Returns:
            List of image file paths
        """
        return self.image_paths.get(person_id, [])

    def __len__(self):
        """Return number of people in database"""
        return len(self.person_ids)

    def __repr__(self):
        stats = self.get_stats()
        return (f"FaceDatabase(people={stats['num_people']}, "
                f"embeddings={stats['total_embeddings']}, "
                f"avg_per_person={stats['avg_embeddings_per_person']:.1f})")
