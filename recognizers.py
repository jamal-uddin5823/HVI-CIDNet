"""
Face Recognizer Wrapper Module

This module provides a unified interface for different face recognition models
including AdaFace and InsightFace. All recognizers implement a common interface
for extracting face embeddings.

Supported Recognizers:
    - AdaFace: Quality-adaptive face recognition (uses existing loss.adaface_model)
    - InsightFace: State-of-the-art face recognition (requires insightface library)

Usage:
    from recognizers import AdaFaceRecognizer, InsightFaceRecognizer

    # Using AdaFace
    recognizer = AdaFaceRecognizer(device='cuda')
    embedding = recognizer.get_embedding(image_tensor)  # (B, 512)

    # Using InsightFace
    recognizer = InsightFaceRecognizer(device='cuda')
    embedding = recognizer.get_embedding(image_tensor)  # (B, 512)
"""

import os
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


class BaseRecognizer(ABC):
    """Abstract base class for face recognizers

    All face recognizers should implement this interface to ensure
    compatibility with the FaceDatabase class.
    """

    def __init__(self, device: str = 'cuda'):
        """Initialize recognizer

        Args:
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device
        self.embedding_dim = 512
        self.model = None

    @abstractmethod
    def get_embedding(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Extract face embedding from image

        Args:
            image_tensor: Input image tensor (B, C, H, W) in range [0, 1]

        Returns:
            Face embedding tensor (B, 512)
        """
        pass

    def preprocess(self, image_tensor: torch.Tensor, size: int = 112) -> torch.Tensor:
        """Preprocess image for face recognition

        Args:
            image_tensor: Input image tensor in range [0, 1]
            size: Target size for the face recognizer

        Returns:
            Preprocessed tensor normalized to [-1, 1]
        """
        # Ensure tensor has batch dimension
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Resize to target size
        if image_tensor.shape[-2:] != (size, size):
            image_tensor = F.interpolate(
                image_tensor,
                size=(size, size),
                mode='bilinear',
                align_corners=False
            )

        # Normalize to [-1, 1] (standard for face recognition models)
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(image_tensor.device)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(image_tensor.device)
        image_tensor = (image_tensor - mean) / std

        return image_tensor

    @abstractmethod
    def load_weights(self, weights_path: str) -> None:
        """Load model weights from file

        Args:
            weights_path: Path to weights file
        """
        pass

    def eval(self):
        """Set model to evaluation mode"""
        if self.model is not None:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        return self


class AdaFaceRecognizer(BaseRecognizer):
    """AdaFace Face Recognizer

    Uses the AdaFace model architecture from loss.adaface_model.
    AdaFace is quality-adaptive and robust to low-light conditions.

    Reference:
        Kim et al. (2022) "AdaFace: Quality Adaptive Margin for Face Recognition"
        https://arxiv.org/abs/2204.00964
    """

    def __init__(
        self,
        arch: str = 'ir_50',
        weights_path: Optional[str] = None,
        device: str = 'cuda'
    ):
        """Initialize AdaFace recognizer

        Args:
            arch: Model architecture ('ir_50', 'ir_101', 'ir_152')
            weights_path: Optional path to pretrained weights
            device: Device to use ('cuda' or 'cpu')
        """
        super().__init__(device=device)

        # Import AdaFace model builder
        from loss.adaface_model import build_model

        # Build model
        self.model = build_model(arch).to(device)
        self.arch = arch
        self.eval()

        # Load weights if provided
        if weights_path:
            self.load_weights(weights_path)

    def get_embedding(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Extract face embedding using AdaFace

        Args:
            image_tensor: Input image tensor (B, C, H, W) in range [0, 1]

        Returns:
            Face embedding tensor (B, 512)
        """
        # Preprocess
        image_tensor = self.preprocess(image_tensor, size=112)

        # Extract embedding
        with torch.no_grad():
            embedding = self.model(image_tensor)

        # Flatten if needed
        if embedding.dim() > 2:
            embedding = embedding.view(embedding.size(0), -1)

        return embedding

    def load_weights(self, weights_path: str) -> None:
        """Load AdaFace weights from file

        Args:
            weights_path: Path to weights file (.pth or .pt)
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        try:
            state_dict = torch.load(weights_path, map_location=self.device)

            # Handle different checkpoint formats
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            if 'model' in state_dict:
                state_dict = state_dict['model']

            # Load with strict=False to allow partial loading
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded AdaFace weights from: {weights_path}")

        except Exception as e:
            print(f"Warning: Could not load weights from {weights_path}: {e}")
            print("Using randomly initialized weights (for testing only)")


class InsightFaceRecognizer(BaseRecognizer):
    """InsightFace Face Recognizer

    Wrapper for InsightFace library. Requires insightface and onnxruntime.

    Installation:
        pip install insightface onnxruntime-gpu

    Note: InsightFace is optional - if not installed, this class will raise
    an ImportError during initialization.
    """

    def __init__(
        self,
        model_name: str = 'buffalo_l',
        device: str = 'cuda',
        providers: Optional[list] = None
    ):
        """Initialize InsightFace recognizer

        Args:
            model_name: InsightFace model name ('buffalo_l', 'buffalo_m', etc.)
            device: Device to use ('cuda' or 'cpu')
            providers: Optional ONNX providers list
        """
        super().__init__(device=device)

        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "InsightFace not installed. Install with:\n"
                "  pip install insightface onnxruntime-gpu\n"
                "Or use AdaFaceRecognizer instead."
            )

        # Set default providers based on device
        if providers is None:
            if device == 'cuda' and torch.cuda.is_available():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

        # Initialize FaceAnalysis
        self.model = FaceAnalysis(name=model_name, providers=providers)
        self.model.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(640, 640))

        # Get the face model
        self.face_model = self.model.models.get('recognition')

        if self.face_model is None:
            raise RuntimeError(f"Recognition model not found in InsightFace '{model_name}'")

        self.model_name = model_name
        self.eval()

    def eval(self):
        """Override eval for InsightFace - ONNX models don't have train/eval modes"""
        # ONNX models don't have train/eval mode, just return self for API compatibility
        return self

    def get_embedding(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Extract face embedding using InsightFace

        Args:
            image_tensor: Input image tensor (B, C, H, W) in range [0, 1]

        Returns:
            Face embedding tensor (B, 512)
        """
        import numpy as np

        # Convert tensor to numpy image
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)

        # Convert to numpy and denormalize
        img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        # Detect faces and extract embedding
        faces = self.model.get(img_np)

        if not faces:
            # No face detected, return zeros
            return torch.zeros(1, self.embedding_dim, device=self.device)

        # Use the first (largest) face
        embedding = faces[0].embedding

        # Convert to tensor
        embedding_tensor = torch.from_numpy(embedding).float().unsqueeze(0).to(self.device)

        return embedding_tensor

    def load_weights(self, weights_path: str) -> None:
        """Load InsightFace weights

        Note: InsightFace models are loaded automatically during initialization.
        This method is provided for API compatibility but does nothing.

        Args:
            weights_path: Path to weights file (ignored for InsightFace)
        """
        print("Note: InsightFace weights are loaded automatically during initialization")


class FaceRecognizerFactory:
    """Factory class for creating face recognizers

    Provides a convenient way to create recognizers with consistent configuration.
    """

    @staticmethod
    def create(
        recognizer_type: str = 'adaface',
        device: str = 'cuda',
        **kwargs
    ) -> BaseRecognizer:
        """Create a face recognizer

        Args:
            recognizer_type: Type of recognizer ('adaface' or 'insightface')
            device: Device to use ('cuda' or 'cpu')
            **kwargs: Additional arguments passed to recognizer constructor

        Returns:
            Face recognizer instance
        """
        recognizer_type = recognizer_type.lower()

        if recognizer_type == 'adaface':
            return AdaFaceRecognizer(device=device, **kwargs)
        elif recognizer_type in ('insightface', 'insight'):
            return InsightFaceRecognizer(device=device, **kwargs)
        else:
            raise ValueError(
                f"Unknown recognizer type: {recognizer_type}. "
                f"Choose from: 'adaface', 'insightface'"
            )


def get_recognizer(
    recognizer_type: str = 'adaface',
    device: str = 'cuda',
    weights_path: Optional[str] = None
) -> BaseRecognizer:
    """Convenience function to get a face recognizer

    Args:
        recognizer_type: Type of recognizer ('adaface' or 'insightface')
        device: Device to use ('cuda' or 'cpu')
        weights_path: Optional path to pretrained weights

    Returns:
        Face recognizer instance

    Example:
        recognizer = get_recognizer('adaface', device='cuda')
        embedding = recognizer.get_embedding(image)
    """
    return FaceRecognizerFactory.create(
        recognizer_type=recognizer_type,
        device=device,
        weights_path=weights_path
    )


if __name__ == "__main__":
    # Test recognizers
    print("Testing face recognizers...")

    # Create dummy face image
    dummy_face = torch.randn(1, 3, 112, 112)

    # Test AdaFace
    print("\n1. Testing AdaFaceRecognizer...")
    try:
        recognizer = AdaFaceRecognizer(device='cpu')
        embedding = recognizer.get_embedding(dummy_face)

        print(f"   Input shape: {dummy_face.shape}")
        print(f"   Output shape: {embedding.shape}")
        print(f"   Output range: [{embedding.min():.3f}, {embedding.max():.3f}]")
        print(f"   Norm: {torch.norm(embedding).item():.3f}")
        print("   AdaFace: PASSED")

    except Exception as e:
        print(f"   AdaFace: FAILED - {e}")

    # Test InsightFace (if available)
    print("\n2. Testing InsightFaceRecognizer...")
    try:
        recognizer = InsightFaceRecognizer(device='cpu')
        embedding = recognizer.get_embedding(dummy_face)

        print(f"   Input shape: {dummy_face.shape}")
        print(f"   Output shape: {embedding.shape}")
        print(f"   Output range: [{embedding.min():.3f}, {embedding.max():.3f}]")
        print(f"   Norm: {torch.norm(embedding).item():.3f}")
        print("   InsightFace: PASSED")

    except ImportError:
        print("   InsightFace: SKIPPED (not installed)")
    except Exception as e:
        print(f"   InsightFace: FAILED - {e}")

    print("\nAll tests complete!")
