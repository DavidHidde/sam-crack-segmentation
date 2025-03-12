from typing import Optional

import torch
from torch.nn.modules.module import T

from sam2.modeling.sam2_utils import LayerNorm2d

from load_sam import SAMVariant, load_sam


class DecoderBlock(torch.nn.Module):
    """A decoder block of a specific size"""

    conv: torch.nn.ConvTranspose2d
    norm: LayerNorm2d
    dropout: Optional[torch.nn.Dropout] = None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dropout: float
    ):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = LayerNorm2d(out_channels)

        if dropout > 0:
            self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the transpose conv, normalization, ReLu and optionally dropout."""
        x = self.conv(x)
        x = self.norm(x)
        x = torch.nn.functional.relu(x)
        if self.dropout:
            x = self.dropout(x)

        return x


class CrackSAM(torch.nn.Module):
    """
    The crack segmentation model, based on Meta's SAM 2.1 model.
    """

    image_encoder: torch.nn.Module
    feature_decoder: torch.nn.Sequential
    image_size: int

    freeze_neck: bool
    freeze_trunk: bool

    def __init__(
        self,
        sam_variant: SAMVariant,
        freeze_trunk: bool = True,
        freeze_neck: bool = False,
        device: torch.device = torch.device('cpu'),
        **kwargs
    ):
        """Initialize the model."""
        super(CrackSAM, self).__init__(**kwargs)

        # Encoder, loaded from SAM - Input: 1024x1024x3, Output: 256x64x64
        sam_model = load_sam(sam_variant, device)
        self.image_size = sam_model.image_size
        self.image_encoder = sam_model.image_encoder
        self.image_encoder.trunk.train(not freeze_trunk)
        self.image_encoder.neck.train(not freeze_neck)

        self.freeze_trunk = freeze_trunk
        self.freeze_neck = freeze_neck

        # Decoder from U-Net - Input: 256x64x64, Output: 1024x1024x1
        self.feature_decoder = torch.nn.Sequential(
            DecoderBlock(256, 128, 2, 2, 0, 0.2),
            DecoderBlock(128, 64, 2, 2, 0, 0.),
            DecoderBlock(64, 32, 2, 2, 0, 0.2),
            DecoderBlock(32, 16, 2, 2, 0, 0.),
            torch.nn.ConvTranspose2d(16, 1, 1, 1, 0),
            torch.nn.Sigmoid()
        ).to(device)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        features = self.image_encoder(image)
        return self.feature_decoder(features['vision_features'])

    def train(self: T, mode: bool = True) -> T:
        """Override the training mode to only set the decoder."""
        super().train(mode)
        self.image_encoder.trunk.train(mode and not self.freeze_trunk)
        self.image_encoder.neck.train(mode and not self.freeze_neck)
