import torch

from load_sam import SAMVariant, load_sam
        

class SegmentationHead(torch.nn.Sequential):
    """
    Final layers that result in the binary segmentation map.
    """

    def __init__(self, in_channels: int):
        super().__init__(
            torch.nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels, 1, kernel_size=1, stride=1),
            torch.nn.Sigmoid()
        )

class DoubleConvReluNorm(torch.nn.Sequential):
    """Double Conv2d layer followed by a Relu and BatchNorm"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(out_channels)
        )

class DecoderBlock(torch.nn.Module):
    """U-Net decoder block."""

    upscale_conv: torch.nn.ConvTranspose2d
    conv: DoubleConvReluNorm

    def __init__(self, in_channels: int):
        super(DecoderBlock, self).__init__()
        out_channels = in_channels // 2
        self.upscale_conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConvReluNorm(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Upscale and apply convs"""
        x1 = self.upscale_conv(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetDecoder(torch.nn.Module):
    """Decoder of the U-Net model."""

    conv: DoubleConvReluNorm
    blocks: torch.nn.ModuleList
    segmentation_head: SegmentationHead

    def __init__(self, base_channels: int, device: torch.device = torch.device('cpu')):
        super(UNetDecoder, self).__init__()
        self.segmentation_head = SegmentationHead(base_channels).to(device)

        # Create decoder blocks top-down and reverse
        self.blocks = torch.nn.ModuleList()
        for idx in range(3):
            base_channels *= 2
            self.blocks.append(DecoderBlock(base_channels).to(device))
        self.blocks = self.blocks[::-1]

        self.conv = DoubleConvReluNorm(base_channels, base_channels).to(device)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Decode given features."""
        features = features[::-1]

        x = self.conv(features[0])
        for skip, block in zip(features[1:], self.blocks):
            x = block(x, skip)

        return self.segmentation_head(x)


class CrackSAM(torch.nn.Module):
    """The crack segmentation model, based on Meta's SAM 2.1 model."""

    image_encoder: torch.nn.Module
    feature_decoder: UNetDecoder

    image_size: int

    def __init__(
        self,
        sam_variant: SAMVariant,
        freeze_encoder: bool = True,
        device: torch.device = torch.device('cpu'),
        **kwargs
    ):
        """Initialize the model."""
        super(CrackSAM, self).__init__(**kwargs)

        # Encoder, loaded from SAM - Input: 1024 x 1024 x 3, Output: channels x 32 x 32
        sam_model = load_sam(sam_variant, device)
        self.image_encoder = sam_model.image_encoder.trunk
        self.image_encoder.train(not freeze_encoder)
        self.image_size = sam_model.image_size

        if freeze_encoder:
            for parameter in self.image_encoder.parameters():
                parameter.requires_grad = False

        # Decoder from U-Net - Input: channels x 32 x 32, Output: 1 x 1024 x 1024
        match(sam_variant):
            case SAMVariant.TINY | SAMVariant.SMALL:
                base_channels = 96
            case SAMVariant.BASE:
                base_channels = 112
            case SAMVariant.LARGE:
                base_channels = 144

        self.feature_decoder = UNetDecoder(base_channels, device=device)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        return self.feature_decoder(self.image_encoder(image))
