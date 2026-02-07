"""
Face Recognition Model Wrappers with Proper Preprocessing

This module provides recognizer implementations for AdaFace and InsightFace with
proper color channel handling and normalization.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from abc import ABC, abstractmethod
from typing import Union, Optional
import os


class BaseFaceRecognizer(ABC):
    """Abstract base class for face recognizers"""

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.face_detector = None

    @abstractmethod
    def get_embedding(self, image: Union[np.ndarray, Image.Image],
                     use_face_detection: bool = False) -> torch.Tensor:
        """Extract face embedding from image

        Args:
            image: Input image (PIL Image or numpy array)
            use_face_detection: Whether to detect face first

        Returns:
            Embedding tensor of shape (1, 512)
        """
        pass

    @abstractmethod
    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """Preprocess image for model input

        Args:
            image: Input image

        Returns:
            Preprocessed tensor ready for model
        """
        pass

    def detect_face(self, image: Union[np.ndarray, Image.Image]) -> Optional[Image.Image]:
        """Detect and crop face from image

        Args:
            image: Input image

        Returns:
            Cropped face as PIL Image, or None if no face detected
        """
        if self.face_detector is None:
            try:
                from insightface.app import FaceAnalysis
                self.face_detector = FaceAnalysis(
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                ctx_id = 0 if self.device.type == 'cuda' else -1
                self.face_detector.prepare(ctx_id=ctx_id, det_size=(640, 640))
            except ImportError:
                print("Warning: InsightFace not available for face detection")
                return None

        # Convert to BGR numpy array for InsightFace
        if isinstance(image, Image.Image):
            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img_bgr = image if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        faces = self.face_detector.get(img_bgr)
        if len(faces) == 0:
            return None

        # Use largest face
        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        x1, y1, x2, y2 = map(int, face.bbox)

        # Crop and return as PIL Image
        if isinstance(image, Image.Image):
            return image.crop((x1, y1, x2, y2))
        else:
            cropped = img_bgr[y1:y2, x1:x2]
            return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))


class AdaFaceRecognizer(BaseFaceRecognizer):
    """AdaFace recognizer with BGR color channel handling"""

    def __init__(self, checkpoint_path: str = 'weights/adaface/adaface_ir50_webface4m.ckpt',
                 device: str = 'cuda'):
        super().__init__(device)
        self.checkpoint_path = checkpoint_path
        self._load_model()

    def _load_model(self):
        """Load AdaFace model from checkpoint"""
        from loss.adaface_model import build_model

        # Build model
        self.model = build_model('ir_50')

        # Load checkpoint
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Clean up state dict keys
            new_state_dict = {}
            for k, v in state_dict.items():
                # Skip AdaFace head parameters (we only need backbone)
                if k.startswith('head.'):
                    continue

                # Remove prefixes: 'module.' and 'model.'
                new_key = k
                if new_key.startswith('module.'):
                    new_key = new_key[7:]  # Remove 'module.'
                if new_key.startswith('model.'):
                    new_key = new_key[6:]  # Remove 'model.'

                new_state_dict[new_key] = v

            # Load with strict=False to ignore missing head parameters
            self.model.load_state_dict(new_state_dict, strict=False)
        else:
            print(f"Warning: AdaFace checkpoint not found at {self.checkpoint_path}")

        self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """Preprocess image for AdaFace (expects BGR)

        Args:
            image: Input image (PIL RGB or numpy BGR)

        Returns:
            Preprocessed tensor of shape (1, 3, 112, 112) in range [-1, 1]
        """
        # Convert PIL RGB to BGR numpy array
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Resize to 112x112
        if image.shape[:2] != (112, 112):
            image = cv2.resize(image, (112, 112))

        # Convert BGR to RGB for transforms.ToTensor (it expects RGB)
        # Then we'll maintain the channel order in the tensor
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to tensor [0, 1]
        img_tensor = transforms.ToTensor()(image_rgb)

        # Normalize to [-1, 1]
        img_tensor = (img_tensor - 0.5) / 0.5

        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor

    def get_embedding(self, image: Union[np.ndarray, Image.Image],
                     use_face_detection: bool = False) -> torch.Tensor:
        """Extract face embedding using AdaFace

        Args:
            image: Input image
            use_face_detection: Whether to detect face first

        Returns:
            Embedding tensor of shape (1, 512)
        """
        # Detect face if requested
        if use_face_detection:
            detected_face = self.detect_face(image)
            if detected_face is None:
                raise ValueError("No face detected in image")
            image = detected_face

        # Preprocess
        img_tensor = self.preprocess_image(image).to(self.device)

        # Extract embedding
        with torch.no_grad():
            embedding = self.model(img_tensor)
            # Normalize embedding
            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding


class InsightFaceRecognizer(BaseFaceRecognizer):
    """InsightFace (ArcFace) recognizer with RGB color channel handling"""

    def __init__(self, model_name: str = 'buffalo_l', device: str = 'cuda'):
        super().__init__(device)
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        """Load InsightFace model"""
        try:
            from insightface.app import FaceAnalysis

            self.face_app = FaceAnalysis(
                name=self.model_name,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            ctx_id = 0 if self.device.type == 'cuda' else -1
            self.face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

            # Extract recognition model
            self.model = self.face_app.models['recognition']

        except ImportError:
            raise ImportError("InsightFace library not installed. Install with: pip install insightface")

    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Preprocess image for InsightFace (expects RGB)

        Args:
            image: Input image (PIL RGB or numpy BGR)

        Returns:
            Preprocessed numpy array of shape (112, 112, 3) in RGB
        """
        # Convert BGR to RGB if numpy array
        if isinstance(image, np.ndarray) and image.shape[2] == 3:
            # Assume cv2 BGR, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            # PIL is already RGB
            image = np.array(image)

        # Resize to 112x112
        if image.shape[:2] != (112, 112):
            image = cv2.resize(image, (112, 112))

        return image

    def get_embedding(self, image: Union[np.ndarray, Image.Image],
                     use_face_detection: bool = False) -> torch.Tensor:
        """Extract face embedding using InsightFace

        Args:
            image: Input image
            use_face_detection: Whether to detect face first

        Returns:
            Embedding tensor of shape (1, 512)
        """
        # Convert to BGR numpy for InsightFace
        if isinstance(image, Image.Image):
            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img_bgr = image

        if use_face_detection:
            # Use face detection
            faces = self.face_app.get(img_bgr)
            if len(faces) == 0:
                raise ValueError("No face detected in image")
            # Use largest face
            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            embedding = face.embedding
        else:
            # Direct embedding extraction without detection
            # Preprocess image
            img_rgb = self.preprocess_image(img_bgr)

            # Convert back to BGR for InsightFace internal processing
            img_bgr_processed = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # Get embedding directly from recognition model
            # InsightFace expects BGR input
            embedding = self.model.get_feat(img_bgr_processed)

        # Convert to torch tensor and add batch dimension
        embedding = torch.from_numpy(embedding).unsqueeze(0).float()

        # Normalize
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding


class FaceRecognizerFactory:
    """Factory for creating face recognizer instances"""

    @staticmethod
    def create(recognizer_type: str, device: str = 'cuda', **kwargs):
        """Create face recognizer instance

        Args:
            recognizer_type: Type of recognizer ('AdaFace' or 'InsightFace')
            device: Device to use ('cuda' or 'cpu')
            **kwargs: Additional arguments for recognizer

        Returns:
            Face recognizer instance
        """
        recognizer_type = recognizer_type.lower()

        if recognizer_type in ['adaface', 'ada_face', 'ada']:
            checkpoint_path = kwargs.get('checkpoint_path', 'weights/adaface/adaface_ir50_webface4m.ckpt')
            return AdaFaceRecognizer(checkpoint_path=checkpoint_path, device=device)
        elif recognizer_type in ['insightface', 'insight_face', 'arcface', 'arc']:
            model_name = kwargs.get('model_name', 'buffalo_l')
            return InsightFaceRecognizer(model_name=model_name, device=device)
        else:
            raise ValueError(f"Unknown recognizer type: {recognizer_type}. "
                           f"Supported types: 'AdaFace', 'InsightFace'")
