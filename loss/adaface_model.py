"""
AdaFace Model Architecture for Face Recognition Perceptual Loss

This module implements the AdaFace model architecture based on IR-SE (Improved ResNet
with Squeeze-and-Excitation) backbone. AdaFace is specifically designed to be robust
to image quality variations, making it ideal for low-light image enhancement.

Reference:
    Kim et al. (2022) "AdaFace: Quality Adaptive Margin for Face Recognition"
    https://arxiv.org/abs/2204.00964

Official Implementation:
    https://github.com/mk-minchul/AdaFace
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple


# Bottleneck with Squeeze-and-Excitation
class SEModule(nn.Module):
    """Squeeze-and-Excitation Module"""

    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class BottleneckIR(nn.Module):
    """Improved ResNet Bottleneck with optional SE module"""

    def __init__(self, in_channel, depth, stride, use_se=True):
        super(BottleneckIR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth)
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth)
        )
        self.use_se = use_se
        if self.use_se:
            self.se = SEModule(depth, reduction=16)

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        if self.use_se:
            res = self.se(res)
        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    """A named tuple describing a ResNet block"""
    pass


def get_block(in_channel, depth, num_units, stride=2):
    """Generate a list of blocks for a ResNet stage"""
    return [Bottleneck(in_channel, depth, stride)] + \
           [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    """Get block configuration for different ResNet depths"""
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    else:
        raise ValueError(f"Invalid number of layers: {num_layers}. Must be 50, 100, or 152.")
    return blocks


class BackboneIRSE(nn.Module):
    """Improved ResNet with Squeeze-and-Excitation (IR-SE) Backbone

    This is the backbone used in AdaFace for feature extraction.

    Args:
        num_layers (int): Number of layers (50, 100, or 152)
        drop_ratio (float): Dropout ratio
        mode (str): 'ir' for basic IR, 'ir_se' for IR with SE modules
    """

    def __init__(self, num_layers, drop_ratio=0.4, mode='ir_se'):
        super(BackboneIRSE, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50, 100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'

        self.use_se = (mode == 'ir_se')

        blocks = get_blocks(num_layers)
        unit_module = BottleneckIR

        # Input layer
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )

        # Body layers
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel,
                        bottleneck.depth,
                        bottleneck.stride,
                        use_se=self.use_se
                    )
                )
        self.body = nn.Sequential(*modules)

        # Output layer
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(drop_ratio),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return x


class AdaFaceHead(nn.Module):
    """AdaFace Head with Quality-Adaptive Margin

    This is the head that applies adaptive margins based on image quality.
    For the loss function, we only need the backbone features, so this
    is optional.
    """

    def __init__(self, embedding_size=512, classnum=70722, m=0.4, h=0.333, s=64.0):
        super(AdaFaceHead, self).__init__()
        self.classnum = classnum
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, classnum))
        nn.init.normal_(self.kernel, std=0.01)

        # AdaFace hyperparameters
        self.m = m  # margin
        self.h = h  # threshold
        self.s = s  # scale
        self.eps = 1e-3

    def forward(self, embbedings, norms, label):
        """
        Args:
            embbedings: (B, 512) normalized embeddings
            norms: (B,) feature norms (for quality estimation)
            label: (B,) ground truth labels
        """
        kernel_norm = F.normalize(self.kernel, dim=0)
        cosine = F.linear(F.normalize(embbedings), kernel_norm)
        cosine = cosine.clamp(-1 + self.eps, 1 - self.eps)

        # Adaptive margin based on feature norm (quality proxy)
        safe_norms = torch.clip(norms, min=0.001, max=100)
        safe_norms = safe_norms.clone().detach()
        margin_scaler = (safe_norms - self.h) / (self.m)
        margin_scaler = margin_scaler * self.m

        # Apply margin
        theta = cosine.acos()
        theta_m = torch.clip(theta + margin_scaler, min=self.eps, max=3.14159 - self.eps)
        cosine_m = theta_m.cos()

        # One-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)

        # Combine
        output = (one_hot * cosine_m) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        return output


def build_model(arch='ir_50', drop_ratio=0.4):
    """Build AdaFace model

    Args:
        arch (str): Architecture name. Options:
            - 'ir_50': IR-SE ResNet-50
            - 'ir_101': IR-SE ResNet-100
            - 'ir_152': IR-SE ResNet-152
        drop_ratio (float): Dropout ratio

    Returns:
        nn.Module: AdaFace backbone model (only backbone, no head)
    """
    if arch == 'ir_50':
        model = BackboneIRSE(num_layers=50, drop_ratio=drop_ratio, mode='ir_se')
    elif arch == 'ir_101':
        model = BackboneIRSE(num_layers=100, drop_ratio=drop_ratio, mode='ir_se')
    elif arch == 'ir_152':
        model = BackboneIRSE(num_layers=152, drop_ratio=drop_ratio, mode='ir_se')
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing AdaFace model architecture...")

    # Test IR-50
    model_50 = build_model('ir_50')
    print(f"IR-50 parameters: {sum(p.numel() for p in model_50.parameters()) / 1e6:.2f}M")

    # Test forward pass
    x = torch.randn(2, 3, 112, 112)
    features = model_50(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {features.shape}")
    assert features.shape == (2, 512), f"Expected (2, 512), got {features.shape}"

    print("All tests passed!")
