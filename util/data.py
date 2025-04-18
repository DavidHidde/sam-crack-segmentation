import os
from typing import Callable, Optional

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from util.transform import InputImageTransform, InputLabelTransform, DataAugmentationTransform

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'JPG'}


def scan_dataset_dir(image_dir: str, label_dir: str) -> list[tuple[str, str]]:
    """Scan dataset directories for image - label pairs."""
    file_name_buffer = set()
    pairs = []

    # First pass over image dir
    for filename in os.listdir(image_dir):
        if filename.split('.')[-1] in ALLOWED_EXTENSIONS:
            file_name_buffer.add(filename)

    # Second filtering pass over label dir
    for filename in os.listdir(label_dir):
        if filename in file_name_buffer:
            pairs.append(
                (
                    os.path.join(image_dir, filename),
                    os.path.join(label_dir, filename)
                )
            )

    return pairs


def gather_datasets(
    image_dir: str,
    label_dir: str,
    test_split: float,
    image_size: tuple[int, int],
    mask_threshold: float,
    data_augmentations: bool,
    crop_image: bool
) -> tuple[Dataset, Dataset]:
    """Get a test and training dataset from a directory. Datasets will load on the fly."""
    dataset_pairs = scan_dataset_dir(image_dir, label_dir)

    if test_split == 1.:
        train_split, test_split = [], dataset_pairs
    elif test_split == 0.:
        train_split, test_split = dataset_pairs, []
    else:
        train_split, test_split = train_test_split(dataset_pairs, test_size=test_split)
    
    transform = DataAugmentationTransform(image_size, crop_image, mask_threshold) if data_augmentations else None
    return (
        SimpleDataset(train_split, image_size, mask_threshold, crop_image, transform=transform),
        SimpleDataset(test_split, image_size, mask_threshold, False, transform=None)  # No transforms on the validation set.
    )


class SimpleDataset(Dataset):
    """A simple custom dataset which applies a transform to preloaded data."""

    image_size: tuple[int, int]
    sample_paths: list[tuple[str, str]]
    transform: Optional[Callable]

    image_preprocess: InputImageTransform
    label_preprocess: InputLabelTransform

    def __init__(
        self,
        sample_paths: list[tuple[str, str]],
        image_size: tuple[int, int],
        mask_threshold: float,
        crop_image: bool = False,
        transform: Optional[Callable] = None
    ):
        self.sample_paths = sample_paths
        self.transform = transform
        self.image_size = image_size
        self.image_preprocess = InputImageTransform(image_size, crop_image)
        self.label_preprocess = InputLabelTransform((image_size[0], image_size[1]), crop_image, mask_threshold)

    def __len__(self) -> int:
        """Return the number of samples in this dataset."""
        return len(self.sample_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the pair of image and label."""
        image_path, label_path = self.sample_paths[index]
        image, label = cv2.imread(image_path), cv2.imread(label_path, flags=cv2.IMREAD_GRAYSCALE)

        # Apply SAM transform and threshold mask - Do not cache, as this might crash otherwise
        image_tensor = self.image_preprocess(image)
        label_tensor = self.label_preprocess(label)

        # Apply optional transform
        if self.transform:
            image_tensor, label_tensor = self.transform(image_tensor, label_tensor)

        return image_tensor, label_tensor
