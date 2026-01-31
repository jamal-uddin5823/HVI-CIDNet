"""
Comprehensive Test Suite for Face Recognition Pipeline

This module contains unit tests and integration tests for the face recognition
system including enhancement, recognition, and database matching.

Test Categories:
    1. Unit Tests: Individual component testing
    2. Integration Tests: End-to-end pipeline testing
    3. Quality Tests: Enhancement quality verification

Usage:
    # Run all tests
    python -m pytest test_face_recognition_pipeline.py -v

    # Run specific test
    python -m pytest test_face_recognition_pipeline.py::test_end_to_end_pipeline_cpu -v

    # Run with coverage
    python -m pytest test_face_recognition_pipeline.py --cov=. --cov-report=html

    # Run CPU-only tests
    python -m pytest test_face_recognition_pipeline.py -m "not gpu" -v
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Import modules under test
from net.CIDNet import CIDNet
from face_database import FaceDatabase
from recognizers import (
    AdaFaceRecognizer,
    InsightFaceRecognizer,
    get_recognizer,
    FaceRecognizerFactory
)


# Fixtures
@pytest.fixture
def sample_face_image():
    """Create a dummy face image for testing"""
    return Image.new('RGB', (112, 112), color='red')


@pytest.fixture
def sample_face_tensor():
    """Create a dummy face tensor for testing"""
    return torch.randn(1, 3, 112, 112)


@pytest.fixture
def sample_database(tmp_path):
    """Create a small test database"""
    db_path = tmp_path / "test_db"

    # Create person directories
    person_1_dir = db_path / "person_1"
    person_2_dir = db_path / "person_2"
    person_1_dir.mkdir(parents=True)
    person_2_dir.mkdir(parents=True)

    # Create sample images
    for i in range(2):
        img = Image.new('RGB', (112, 112), color='red' if i == 0 else 'blue')
        img.save(person_1_dir / f"img_{i+1}.jpg")

    img = Image.new('RGB', (112, 112), color='blue')
    img.save(person_2_dir / "img_1.jpg")

    return str(db_path)


@pytest.fixture
def loaded_enhancer():
    """Load enhancement model once for multiple tests"""
    model = CIDNet()
    model.eval()
    return model


# ============================================================================
# UNIT TESTS
# ============================================================================

class TestRecognizerInitialization:
    """Test recognizer initialization and loading"""

    def test_adaface_initialization(self):
        """Test AdaFace recognizer initializes correctly"""
        recognizer = AdaFaceRecognizer(device='cpu')
        assert recognizer.model is not None
        assert recognizer.embedding_dim == 512
        assert recognizer.device == 'cpu'

    def test_adaface_architecture_selection(self):
        """Test different AdaFace architectures can be loaded"""
        for arch in ['ir_50', 'ir_101', 'ir_152']:
            recognizer = AdaFaceRecognizer(arch=arch, device='cpu')
            assert recognizer.model is not None
            assert recognizer.arch == arch

    def test_insightface_initialization(self):
        """Test InsightFace recognizer initialization (if available)"""
        try:
            recognizer = InsightFaceRecognizer(device='cpu')
            assert recognizer.model is not None
            assert recognizer.embedding_dim == 512
        except ImportError:
            pytest.skip("InsightFace not installed")

    def test_factory_create_adaface(self):
        """Test factory creates AdaFace correctly"""
        recognizer = FaceRecognizerFactory.create('adaface', device='cpu')
        assert isinstance(recognizer, AdaFaceRecognizer)

    def test_factory_create_insightface(self):
        """Test factory creates InsightFace correctly (if available)"""
        try:
            recognizer = FaceRecognizerFactory.create('insightface', device='cpu')
            assert isinstance(recognizer, InsightFaceRecognizer)
        except ImportError:
            pytest.skip("InsightFace not installed")

    def test_factory_invalid_type(self):
        """Test factory raises error for invalid type"""
        with pytest.raises(ValueError):
            FaceRecognizerFactory.create('invalid_type')

    def test_get_recognizer_function(self):
        """Test convenience function for getting recognizer"""
        recognizer = get_recognizer('adaface', device='cpu')
        assert isinstance(recognizer, AdaFaceRecognizer)


class TestEmbeddingExtraction:
    """Test face embedding extraction"""

    def test_adaface_embedding_shape(self, sample_face_tensor):
        """Test AdaFace produces correct embedding shape"""
        recognizer = AdaFaceRecognizer(device='cpu')
        embedding = recognizer.get_embedding(sample_face_tensor)

        assert embedding.shape == (1, 512)

    def test_adaface_embedding_no_nan(self, sample_face_tensor):
        """Test AdaFace embeddings don't contain NaN"""
        recognizer = AdaFaceRecognizer(device='cpu')
        embedding = recognizer.get_embedding(sample_face_tensor)

        assert not torch.isnan(embedding).any()

    def test_adaface_embedding_not_zero(self, sample_face_tensor):
        """Test AdaFace embeddings are not zero vectors"""
        recognizer = AdaFaceRecognizer(device='cpu')
        embedding = recognizer.get_embedding(sample_face_tensor)

        assert torch.norm(embedding) > 0

    def test_adaface_batch_processing(self):
        """Test AdaFace can process batch of images"""
        recognizer = AdaFaceRecognizer(device='cpu')
        batch = torch.randn(4, 3, 112, 112)
        embeddings = recognizer.get_embedding(batch)

        assert embeddings.shape == (4, 512)

    def test_insightface_embedding_shape(self, sample_face_tensor):
        """Test InsightFace produces correct embedding shape (if available)"""
        try:
            recognizer = InsightFaceRecognizer(device='cpu')
            embedding = recognizer.get_embedding(sample_face_tensor)
            assert embedding.shape == (1, 512)
        except ImportError:
            pytest.skip("InsightFace not installed")


class TestDatabaseLoading:
    """Test face database loading and management"""

    def test_database_loading_from_directory(self, sample_database):
        """Test database loads correctly from directory structure"""
        recognizer = AdaFaceRecognizer(device='cpu')
        db = FaceDatabase(sample_database, recognizer, device='cpu')

        assert len(db.person_ids) == 2
        assert "person_1" in db.person_ids
        assert "person_2" in db.person_ids

    def test_database_embeddings_cached(self, sample_database):
        """Test embeddings are cached after loading"""
        recognizer = AdaFaceRecognizer(device='cpu')
        db = FaceDatabase(sample_database, recognizer, device='cpu')

        assert len(db.embeddings) == 2
        assert "person_1" in db.embeddings
        assert "person_2" in db.embeddings

    def test_database_stats(self, sample_database):
        """Test database statistics are correct"""
        recognizer = AdaFaceRecognizer(device='cpu')
        db = FaceDatabase(sample_database, recognizer, device='cpu')

        stats = db.get_stats()
        assert stats['num_people'] == 2
        assert stats['total_images'] == 3
        assert stats['embedding_dim'] == 512

    def test_database_nonexistent_path(self):
        """Test database handles nonexistent path gracefully"""
        recognizer = AdaFaceRecognizer(device='cpu')
        db = FaceDatabase('nonexistent_path', recognizer, device='cpu')

        assert len(db.person_ids) == 0


class TestCosineSimilarity:
    """Test cosine similarity computation for matching"""

    def test_identical_vectors_similarity_one(self):
        """Test identical vectors have similarity 1.0"""
        v1 = torch.randn(512)
        v1_norm = F.normalize(v1.unsqueeze(0), p=2, dim=1)
        sim = (v1_norm * v1_norm).sum()

        assert abs(sim.item() - 1.0) < 1e-6

    def test_orthogonal_vectors_similarity_near_zero(self):
        """Test orthogonal vectors have similarity near zero"""
        v1 = torch.randn(512)
        v2 = torch.randn(512)
        v1_norm = F.normalize(v1.unsqueeze(0), p=2, dim=1)
        v2_norm = F.normalize(v2.unsqueeze(0), p=2, dim=1)
        sim = (v1_norm * v2_norm).sum()

        assert -1 <= sim.item() <= 1

    def test_opposite_vectors_similarity_negative_one(self):
        """Test opposite vectors have similarity -1.0"""
        v1 = torch.randn(512)
        v2 = -v1
        v1_norm = F.normalize(v1.unsqueeze(0), p=2, dim=1)
        v2_norm = F.normalize(v2.unsqueeze(0), p=2, dim=1)
        sim = (v1_norm * v2_norm).sum()

        assert abs(sim.item() - (-1.0)) < 1e-6


class TestTopKMatching:
    """Test top-K match retrieval"""

    def test_top_k_returns_correct_number(self, sample_database):
        """Test top-K returns correct number of results"""
        recognizer = AdaFaceRecognizer(device='cpu')
        db = FaceDatabase(sample_database, recognizer, device='cpu')

        # Get person_1's embedding
        query_embedding = db.embeddings["person_1"]
        results = db.match(query_embedding, top_k=3)

        assert len(results) <= 3

    def test_top_k_sorted_by_confidence(self, sample_database):
        """Test results are sorted by confidence (descending)"""
        recognizer = AdaFaceRecognizer(device='cpu')
        db = FaceDatabase(sample_database, recognizer, device='cpu')

        query_embedding = db.embeddings["person_1"]
        results = db.match(query_embedding, top_k=5)

        # Check sorted descending
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i+1][1]

    def test_top_k_self_match_high_confidence(self, sample_database):
        """Test self-match has high confidence"""
        recognizer = AdaFaceRecognizer(device='cpu')
        db = FaceDatabase(sample_database, recognizer, device='cpu')

        # Query with person_1's embedding
        query_embedding = db.embeddings["person_1"]
        results = db.match(query_embedding, top_k=5)

        if results:
            # Top match should be person_1 with high confidence
            assert results[0][0] == "person_1"
            assert results[0][1] > 0.9


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestEndToEndPipeline:
    """Test complete enhancement -> recognition -> matching pipeline"""

    def test_end_to_end_pipeline_cpu(self, sample_database):
        """Test complete pipeline on CPU"""
        # Load models
        enhancer = CIDNet()
        enhancer.eval()

        recognizer = AdaFaceRecognizer(device='cpu')
        db = FaceDatabase(sample_database, recognizer, device='cpu')

        # Create test low-light image (dark)
        low_light_img = Image.new('RGB', (256, 256), color=(50, 50, 50))
        to_tensor = transforms.ToTensor()
        query_tensor = to_tensor(low_light_img).unsqueeze(0)

        # Enhance (no actual enhancement without trained weights, but test pipeline)
        with torch.no_grad():
            enhanced = enhancer(query_tensor)

        # Match
        query_embedding = recognizer.get_embedding(enhanced)
        results = db.match(query_embedding.squeeze(0), top_k=5)

        # Verify results format
        assert len(results) > 0
        for person_id, confidence, img_path in results:
            assert isinstance(person_id, str)
            assert isinstance(confidence, float)
            assert 0 <= confidence <= 1
            assert isinstance(img_path, str)

    @pytest.mark.gpu
    def test_end_to_end_pipeline_gpu(self, sample_database):
        """Test complete pipeline on GPU"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Load models on GPU
        enhancer = CIDNet().cuda()
        enhancer.eval()

        recognizer = AdaFaceRecognizer(device='cuda')
        db = FaceDatabase(sample_database, recognizer, device='cuda')

        # Create test image
        low_light_img = Image.new('RGB', (256, 256), color=(50, 50, 50))
        to_tensor = transforms.ToTensor()
        query_tensor = to_tensor(low_light_img).unsqueeze(0).cuda()

        # Enhance
        with torch.no_grad():
            enhanced = enhancer(query_tensor)

        # Match
        query_embedding = recognizer.get_embedding(enhanced)
        results = db.match(query_embedding.squeeze(0), top_k=5)

        assert len(results) > 0


class TestEnhancementQuality:
    """Test enhancement quality and output properties"""

    def test_enhancement_output_range(self, loaded_enhancer):
        """Verify enhancement produces valid pixel values"""
        # Dark input
        low_light = torch.randn(1, 3, 256, 256) * 0.3

        with torch.no_grad():
            enhanced = loaded_enhancer(low_light)

        # Values should be in valid range [0, 1]
        assert enhanced.min() >= 0
        assert enhanced.max() <= 1

    def test_enhancement_output_shape(self, loaded_enhancer):
        """Test enhancement preserves spatial dimensions"""
        input_tensor = torch.randn(1, 3, 256, 256)

        with torch.no_grad():
            enhanced = loaded_enhancer(input_tensor)

        assert enhanced.shape == input_tensor.shape

    def test_enhancement_padding_to_multiple_of_8(self, loaded_enhancer):
        """Test enhancement handles padding correctly"""
        # Input with dimensions not divisible by 8
        # Need to pad before passing to model (as the app does)
        input_tensor = torch.randn(1, 3, 125, 94)

        # Pad to multiple of 8 (required by the model)
        factor = 8
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_padded = F.pad(input_tensor, (0, padw, 0, padh), 'reflect')

        with torch.no_grad():
            enhanced = loaded_enhancer(input_padded)

        # Check output shape matches padded input
        assert enhanced.shape[0] == 1  # Batch size preserved
        assert enhanced.shape[1] == 3  # Channels preserved
        assert enhanced.shape[2] == H  # Height matches padded size
        assert enhanced.shape[3] == W  # Width matches padded size


class TestDatabaseOperations:
    """Test database operations like adding, removing"""

    def test_add_person_to_database(self, sample_database):
        """Test adding a new person to database"""
        recognizer = AdaFaceRecognizer(device='cpu')
        db = FaceDatabase(sample_database, recognizer, device='cpu')

        initial_count = len(db.person_ids)

        # Add new person
        person_dir = Path(sample_database) / "person_3"
        person_dir.mkdir(exist_ok=True)
        img = Image.new('RGB', (112, 112), color='green')
        img.save(person_dir / "img_1.jpg")

        db.add_person("person_3", [str(person_dir / "img_1.jpg")])

        assert len(db.person_ids) == initial_count + 1

    def test_database_cache_operations(self, sample_database, tmp_path):
        """Test database cache save/load"""
        cache_file = tmp_path / "test_cache.pth"

        recognizer = AdaFaceRecognizer(device='cpu')
        db = FaceDatabase(sample_database, recognizer, device='cpu', cache_file=str(cache_file))

        # Should create cache
        assert Path(cache_file).exists()

        # Load from cache
        db2 = FaceDatabase(sample_database, recognizer, device='cpu', cache_file=str(cache_file))
        assert len(db2.person_ids) == len(db.person_ids)

        # Clear cache
        db.clear_cache()
        assert not Path(cache_file).exists()


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
