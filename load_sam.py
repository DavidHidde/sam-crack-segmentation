import os
from enum import Enum

import torch

from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base

CHECKPOINT_DIR = 'sam-checkpoints'
CONFIG_DIR = os.path.join('configs', 'sam2.1')  # Relative to sam2/sam2 repo


class SAMVariant(Enum):
    """Available variants of the SAM 2.1 model."""
    TINY = 'tiny'
    SMALL = 'small'
    BASE = 'base_plus'
    LARGE = 'large'


def load_sam(variant: SAMVariant, device: torch.device) -> SAM2Base:
    """Load a SAM variant based on the model name."""
    match variant:
        case SAMVariant.TINY:
            yaml_variant = 't'
        case SAMVariant.SMALL:
            yaml_variant = 's'
        case SAMVariant.BASE:
            yaml_variant = 'b+'
        case SAMVariant.LARGE:
            yaml_variant = 'l'
        case _:
            yaml_variant = 't'

    return build_sam2(
        os.path.join(CONFIG_DIR, f'sam2.1_hiera_{yaml_variant}.yaml'),
        os.path.join(CHECKPOINT_DIR, f'sam2.1_hiera_{variant.value}.pt'),
        device=device
    )
