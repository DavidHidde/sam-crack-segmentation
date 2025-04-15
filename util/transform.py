from typing import Union

import numpy as np
import torch
from PIL.Image import Image

from torch import nn, Tensor
from torchvision.transforms import v2, InterpolationMode
from torchvision.transforms.v2.functional import to_pil_image

# Constants from Sam2Transforms
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Data augmentation constants
RANDOM_FLIP_PROBABILITY = 0.25
ROTATION_RANGE = 30
RESIZE_SCALE_RANGE = (0.6, 1.)


class SquarePad(nn.Module):
    """Module for padding images to be squares. Padding is added such that the original image is in the top left."""
    value: float

    def __init__(self, value: float):
        super(SquarePad, self).__init__()
        self.value = value

    def forward(self, x: Tensor) -> Tensor:
        """Pad the tensor."""
        _, h, w = x.size()
        max_size = max(h, w)
        padding = (0, max_size - w, 0, max_size - h)
        return nn.functional.pad(x, padding, "constant", value=self.value)


class InputImageTransform(nn.Sequential):
    """
    Preprocessing transforms for input images. These transforms match the SAM 2 image transforms.
    Accepts a PIL image or numpy array and returns a tensor.
    """

    def __init__(self, image_size: tuple[int, int], crop_image: bool):
        super().__init__(
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),  # toTensor
            SquarePad(0.),
            v2.Resize(image_size) if not crop_image else nn.Identity(),
            v2.Normalize(MEAN, STD)
        )


class InputLabelTransform(nn.Sequential):
    """
    Preprocessing transforms for input labels.
    Accepts a PIL image or numpy array and returns a tensor.
    """

    threshold: float

    def __init__(self, image_size: tuple[int, int], crop_image: bool, mask_threshold: float):
        super().__init__(
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),  # toTensor
            v2.Grayscale(),
            SquarePad(0.),
            v2.Resize(image_size, interpolation=InterpolationMode.NEAREST) if not crop_image else nn.Identity()
        )
        self.threshold = mask_threshold

    def __call__(self, image: Union[Image, np.ndarray]) -> Tensor:
        """Apply transformation and return thresholded tensor."""
        label_tensor = super().forward(image)
        return torch.where(label_tensor[[0], :, :] >= self.threshold, 1., 0.)


class DataAugmentationTransform(nn.Module):
    """Data augmentation transforms for training."""

    geometric_transforms: nn.Sequential

    threshold: float

    def __init__(self, image_size: tuple[int, int], crop_image: bool, mask_threshold: float):
        super(DataAugmentationTransform, self).__init__()
        self.threshold = mask_threshold

        self.geometric_transforms = nn.Sequential(
            v2.RandomHorizontalFlip(RANDOM_FLIP_PROBABILITY),
            v2.RandomVerticalFlip(RANDOM_FLIP_PROBABILITY),
            v2.RandomRotation(ROTATION_RANGE),
            v2.RandomResizedCrop(size=image_size, scale=RESIZE_SCALE_RANGE, ratio=(1., 1.)) if crop_image else nn.Identity()
        )
    
    def forward(self, img: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply a data augmentation on an image and label pair. Only geometric transformations are applied to the label.
        We keep track of the RNG state to ensure both the image and label apply the same transform.
        """
        rng_state = torch.get_rng_state()
        img = self.geometric_transforms(img)
        torch.set_rng_state(rng_state)
        label = self.geometric_transforms(label)
        label = torch.where(label >= self.threshold, 1., 0.)
        return img, label


class OutputLabelTransform:
    """Transform a label back to the original size. Turns a tensor back into a PIL image."""

    threshold: float

    def __init__(self, mask_threshold: float):
        self.threshold = mask_threshold

    def __call__(self, labels: Tensor, image_size: tuple[int, int]) -> list[Image]:
        """Transform a label tensor into a PIL image of a desired size."""
        label_copy = labels.clone()
        max_size = max(image_size[0], image_size[1])
        label_copy = nn.functional.interpolate(label_copy, [max_size, max_size], mode='bilinear', align_corners=False)
        label_copy = label_copy[:, :, :image_size[1], :image_size[0]]

        label_copy = nn.functional.sigmoid(torch.clamp(label_copy, -32., 32))
        label_copy = torch.where(label_copy >= self.threshold, 255, 0)
        label_arr = label_copy.detach().cpu().numpy()
        return [to_pil_image(arr.astype(np.uint8).squeeze()) for arr in label_arr]
