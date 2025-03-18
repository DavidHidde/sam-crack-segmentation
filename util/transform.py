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
RANDOM_ROTATION_PROBABILITY = 0.25
ROTATION_RANGE = (-30, 30)

class InputImageTransform(nn.Sequential):
    """
    Preprocessing transforms for input images. These transforms match the SAM 2 image transforms.
    Accepts a PIL image or numpy array and returns a tensor.
    """

    def __init__(self, image_size: tuple[int, int]):
        super().__init__(
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),  # toTensor
            v2.Resize(image_size),
            v2.Normalize(MEAN, STD)
        )


class InputLabelTransform(nn.Sequential):
    """
    Preprocessing transforms for input labels.
    Accepts a PIL image or numpy array and returns a tensor.
    """

    threshold: float

    def __init__(self, image_size: tuple[int, int], mask_threshold: float):
        super().__init__(
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),  # toTensor
            v2.Grayscale(),
            v2.Resize(image_size, interpolation=InterpolationMode.NEAREST)
        )
        self.threshold = mask_threshold

    def __call__(self, image: Union[Image, np.ndarray]) -> Tensor:
        """Apply transformation and return thresholded tensor."""
        label_tensor = super().forward(image)
        return torch.where(label_tensor[[0], :, :] >= self.threshold, 1, 0)


class DataAugmentationTransform(nn.Sequential):
    """Data augmentation transforms for training."""

    def __init__(self):
        super().__init__(
            v2.RandomHorizontalFlip(RANDOM_FLIP_PROBABILITY),
            v2.RandomVerticalFlip(RANDOM_FLIP_PROBABILITY),
            v2.RandomRotation(ROTATION_RANGE)
        )


class OutputLabelTransform:
    """Transform a label back to the original size. Turns a tensor back into a PIL image."""

    threshold: float

    def __init__(self, mask_threshold: float):
        self.threshold = mask_threshold

    def __call__(self, label: Tensor, image_size: tuple[int, int]) -> Image:
        """Transform a label tensor into a PIL image of a desired size."""
        label_copy = label.clone()
        label_copy = torch.where(label_copy >= self.threshold, 255, 0)
        label_copy = v2.functional.resize(label_copy, image_size, interpolation=InterpolationMode.NEAREST)
        return to_pil_image(label_copy.detach().cpu(), mode='L')
