import os
from typing import Callable, Optional

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from transform import ImagePreprocessTransform, LabelPreprocessTransform

IMAGE_DIR = 'images'
LABEL_DIR = 'labels'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'JPG'}


def scan_dataset_dir(dataset_dir: str) -> list[tuple[str, str]]:
    """Scan a dataset directory for image - label pairs."""
    file_name_buffer = set()
    pairs = []

    # First pass over image dir
    for filename in os.listdir(os.path.join(dataset_dir, IMAGE_DIR)):
        if filename.split('.')[-1] in ALLOWED_EXTENSIONS:
            file_name_buffer.add(filename)

    # Second filtering pass over label dir
    for filename in os.listdir(os.path.join(dataset_dir, IMAGE_DIR)):
        if filename in file_name_buffer:
            pairs.append(
                (
                    os.path.join(dataset_dir, IMAGE_DIR, filename),
                    os.path.join(dataset_dir, LABEL_DIR, filename)
                )
            )

    return pairs


def gather_datasets(
    dataset_dir: str,
    test_split: float,
    image_size: int,
    device: torch.device,
    transform: Optional[Callable] = None
) -> tuple[Dataset, Dataset]:
    """Get a test and training dataset from a directory. Datasets will load on the fly."""
    dataset_pairs = scan_dataset_dir(dataset_dir)
    train_split, test_split = train_test_split(dataset_pairs, test_size=test_split)
    return (
        CrackSamDataset(train_split, image_size, device, transform=transform),
        CrackSamDataset(test_split, image_size, device, transform=None) # Do not apply transforms to validation datasets
    )


class CrackSamDataset(Dataset):
    """A simple custom dataset which applies a transform to preloaded data."""

    sample_paths: list[tuple[str, str]]
    transform: Optional[Callable]
    device: torch.device
    
    image_preprocess: ImagePreprocessTransform
    label_preprocess: LabelPreprocessTransform

    def __init__(self, sample_paths: list[tuple[str, str]], image_size: int, device: torch.device, transform: Optional[Callable] = None):
        self.sample_paths = sample_paths
        self.transform = transform
        self.image_size = image_size
        self.device = device
        self.image_preprocess = ImagePreprocessTransform(image_size)
        self.label_preprocess = LabelPreprocessTransform(image_size)

    def __len__(self) -> int:
        """Return the number of samples in this dataset."""
        return len(self.sample_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the pair of image and label."""
        image_path, label_path = self.sample_paths[index]
        image = cv2.imread(image_path)
        label = cv2.imread(label_path, flags=cv2.IMREAD_GRAYSCALE)

        # Apply SAM transform and threshold mask
        image_tensor = self.image_preprocess(image)
        label_tensor = self.label_preprocess(label)

        # Apply optional transform
        if self.transform:
            image_tensor = self.transform(image_tensor)
            label_tensor = self.transform(label_tensor)

        return image_tensor.to(self.device, non_blocking=True), \
            label_tensor.to(self.device, non_blocking=True)
