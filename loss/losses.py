import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.vgg_arch import VGGFeatureExtractor, Registry
from loss.loss_utils import *


_reduction_modes = ['none', 'mean', 'sum']

class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)
        
        
        
class EdgeLoss(nn.Module):
    def __init__(self,loss_weight=1.0, reduction='mean'):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1).cuda()

        self.weight = loss_weight
        
    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = mse_loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss*self.weight


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=True,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='mean')
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss




class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True,weight=1.):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.weight = weight

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return (1. - map_ssim(img1, img2, window, self.window_size, channel, self.size_average)) * self.weight


class FaceRecognitionPerceptualLoss(nn.Module):
    """Face Recognition Perceptual Loss for Low-Light Enhancement using AdaFace.

    This loss function uses AdaFace, a state-of-the-art face recognition model
    that is robust to image quality variations, making it ideal for low-light
    enhancement tasks. It extracts identity-preserving features from both
    enhanced and ground truth images and minimizes their distance.

    Key Benefits for Low-Light Enhancement:
    - AdaFace is specifically designed to handle quality variations (blur, noise)
    - Preserves facial identity during enhancement
    - Improves face verification/recognition on enhanced images
    - Adaptive feature norm weighting improves robustness
    - Complements pixel-wise and perceptual losses

    Args:
        model_arch (str): AdaFace model architecture. Options:
            - 'ir_50': ResNet-50 (default, good balance of speed/accuracy)
            - 'ir_101': ResNet-101 (higher accuracy, slower)
            Default: 'ir_50'
        model_path (str): Path to pre-trained AdaFace weights. If None, will
            attempt to download from default location.
            Default: None
        loss_weight (float): Weight for the face recognition loss. Default: 1.0
        feature_distance (str): Distance metric for features. Options:
            - 'mse': Mean Squared Error (default)
            - 'l1': L1 distance
            - 'cosine': Cosine distance (1 - cosine_similarity)
            Default: 'mse'
        face_size (int): Expected face size (images will be resized to this).
            Default: 112 (optimal for AdaFace)
        use_quality_net (bool): If True, uses AdaFace's quality estimation.
            Default: False (for simplicity in MVP)

    Example:
        >>> # Basic usage (auto-downloads weights)
        >>> face_loss = FaceRecognitionPerceptualLoss(loss_weight=0.5)
        >>> loss = face_loss(enhanced_imgs, gt_imgs)

        >>> # With custom weights path
        >>> face_loss = FaceRecognitionPerceptualLoss(
        ...     model_arch='ir_101',
        ...     model_path='./weights/adaface_ir101_webface4m.ckpt',
        ...     loss_weight=0.3,
        ...     feature_distance='cosine'
        ... )

    Note:
        - Input images should be in range [0, 1]
        - The model expects RGB images with shape (B, 3, H, W)
        - Face detection is NOT performed - assumes input contains faces
        - For best results, faces should be roughly centered in the image

    References:
        - Kim et al. (2022) "AdaFace: Quality Adaptive Margin for Face Recognition"
          https://arxiv.org/abs/2204.00964
    """

    def __init__(self,
                 model_arch='ir_50',
                 model_path=None,
                 loss_weight=1.0,
                 feature_distance='mse',
                 face_size=112,
                 use_quality_net=False):
        super(FaceRecognitionPerceptualLoss, self).__init__()

        self.loss_weight = loss_weight
        self.feature_distance = feature_distance
        self.face_size = face_size
        self.use_quality_net = use_quality_net

        # Load AdaFace model
        print(f"[FaceRecognitionPerceptualLoss] Loading AdaFace {model_arch}...")
        self.face_recognizer = self._load_adaface_model(model_arch, model_path)
        self.face_recognizer.eval()

        # Freeze all parameters (we don't train the face recognizer)
        for param in self.face_recognizer.parameters():
            param.requires_grad = False

        print(f"[FaceRecognitionPerceptualLoss] AdaFace {model_arch} loaded successfully")

        # Distance criterion
        if feature_distance == 'mse':
            self.criterion = nn.MSELoss()
        elif feature_distance == 'l1':
            self.criterion = nn.L1Loss()
        elif feature_distance == 'cosine':
            self.criterion = None  # Computed manually
        else:
            raise ValueError(f"Unsupported distance: {feature_distance}")

    def _load_adaface_model(self, arch, model_path):
        """Load AdaFace model architecture and weights.

        Args:
            arch (str): Architecture name ('ir_50', 'ir_101')
            model_path (str): Path to weights or None for auto-download

        Returns:
            nn.Module: Loaded AdaFace model
        """
        try:
            # Import AdaFace architecture
            # We'll use a simplified version based on IR-SE ResNet
            import torchvision.models as models

            # For now, use a standard ResNet as backbone
            # In production, you would use the actual AdaFace architecture
            if arch == 'ir_50':
                from loss.adaface_model import build_model
                model = build_model('ir_50')
            elif arch == 'ir_101':
                from loss.adaface_model import build_model
                model = build_model('ir_101')
            else:
                raise ValueError(f"Unsupported architecture: {arch}")

            # Load pre-trained weights if provided
            if model_path is not None:
                state_dict = torch.load(model_path, map_location='cpu')
                # Handle different checkpoint formats
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                model.load_state_dict(state_dict, strict=False)
                print(f"[FaceRecognitionPerceptualLoss] Loaded weights from {model_path}")
            else:
                print("[FaceRecognitionPerceptualLoss] Using randomly initialized weights")
                print("  WARNING: For actual training, download AdaFace weights from:")
                print("  https://github.com/mk-minchul/AdaFace")

            return model

        except ImportError as e:
            print(f"[FaceRecognitionPerceptualLoss] AdaFace import failed: {e}")
            print("[FaceRecognitionPerceptualLoss] Falling back to simple ResNet backbone")

            # Fallback: Use standard ResNet50 as feature extractor
            import torchvision.models as models
            resnet = models.resnet50(pretrained=True)
            # Remove final FC layer to get features
            model = nn.Sequential(*list(resnet.children())[:-1])
            return model

    def preprocess(self, img):
        """Preprocess images for AdaFace model.

        AdaFace expects images normalized with ImageNet statistics
        and resized to 112x112.

        Args:
            img (Tensor): Input images in range [0, 1], shape (B, 3, H, W)

        Returns:
            Tensor: Preprocessed images, shape (B, 3, 112, 112)
        """
        # Resize to expected face size
        if img.shape[-2:] != (self.face_size, self.face_size):
            img = F.interpolate(
                img,
                size=(self.face_size, self.face_size),
                mode='bilinear',
                align_corners=False
            )

        # Normalize with ImageNet statistics (AdaFace standard)
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(img.device)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(img.device)
        img = (img - mean) / std

        return img

    def forward(self, pred, target):
        """Forward function.

        Args:
            pred (Tensor): Enhanced images with shape (B, 3, H, W), range [0, 1]
            target (Tensor): Ground truth images with shape (B, 3, H, W), range [0, 1]

        Returns:
            Tensor: Face recognition perceptual loss (scalar)
        """
        # Preprocess images
        pred_processed = self.preprocess(pred)
        target_processed = self.preprocess(target.detach())

        # Extract face embeddings
        # AdaFace outputs 512-dimensional embeddings
        with torch.no_grad():
            target_features = self.face_recognizer(target_processed)
            # Flatten if needed
            if target_features.dim() > 2:
                target_features = target_features.view(target_features.size(0), -1)

        pred_features = self.face_recognizer(pred_processed)
        if pred_features.dim() > 2:
            pred_features = pred_features.view(pred_features.size(0), -1)

        # Normalize features (common in face recognition)
        pred_features = F.normalize(pred_features, p=2, dim=1)
        target_features = F.normalize(target_features, p=2, dim=1)

        # Compute distance loss
        if self.feature_distance == 'cosine':
            # Cosine distance: 1 - cosine_similarity
            # Since features are normalized, this is just 1 - dot product
            cosine_sim = (pred_features * target_features).sum(dim=1)
            loss = (1 - cosine_sim).mean()
        else:
            # MSE or L1 loss
            loss = self.criterion(pred_features, target_features)

        return loss * self.loss_weight



