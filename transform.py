import torch
from torchvision.transforms import Normalize, Resize, ToTensor

MASK_THRESHOLD = 0.5

# Constants from Sam2Transforms
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class ImagePreprocessTransform(torch.nn.Module):
    """Preprocessing transforms for images."""

    to_tensor: torch.nn.Module
    resize: torch.nn.Module
    normalize: torch.nn.Module

    def __init__(self, image_size: int):
        super().__init__()
        self.to_tensor = ToTensor()
        self.resize = Resize((image_size, image_size))
        self.normalize = Normalize(MEAN, STD)

    def __call__(self, image):
        """Apply transformation"""
        return self.normalize(self.resize(self.to_tensor(image)))

class LabelPreprocessTransform(torch.nn.Module):
    """Preprocessing transforms for labels."""

    to_tensor: torch.nn.Module
    resize: torch.nn.Module
    normalize: torch.nn.Module

    def __init__(self, image_size: int):
        super().__init__()
        self.to_tensor = ToTensor()
        self.resize = Resize((image_size, image_size))

    def __call__(self, image):
        """Apply transformation"""
        label_tensor = self.resize(self.to_tensor(image))
        return torch.where(label_tensor[[0], :, :] >= MASK_THRESHOLD, 1, 0)  # Threshold
