from enum import Enum

import torch

from sam2.build_sam import build_sam2_hf
from sam2.modeling.sam2_base import SAM2Base


class SAMVariant(Enum):
    """Available variants of the SAM 2.1 model."""
    TINY = 'tiny'
    SMALL = 'small'
    BASE = 'base_plus'
    LARGE = 'large'


def load_sam(variant: SAMVariant, device: torch.device) -> SAM2Base:
    """Load a SAM variant from HuggingFace based on the model name."""
    model_name = 'facebook/sam2.1-hiera'
    match variant:
        case SAMVariant.TINY:
            model_name = f'{model_name}-tiny'
        case SAMVariant.SMALL:
            model_name = f'{model_name}-small'
        case SAMVariant.BASE:
            model_name = f'{model_name}-base-plus'
        case SAMVariant.LARGE:
            model_name = f'{model_name}-large'

    return build_sam2_hf(model_name, device=device)
