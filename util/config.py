from dataclasses import dataclass

import yaml

from model.functional import SAMVariant


@dataclass
class TrainConfig:
    """Configuration detailing how a model should be trained."""
    id: str

    # Training hyperparams
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    scheduler_step_size: int
    scheduler_gamma: float

    # Model/fine-tuning params
    sam_variant: SAMVariant
    freeze_trunk: bool
    freeze_neck: bool
    freeze_prompt_encoder: bool
    freeze_mask_decoder: bool
    apply_lora: bool
    lora_rank: int

    # Data params
    image_directory: str
    label_directory: str
    test_split: float
    mask_threshold: float
    shuffle: bool
    augment: bool
    crop_image: bool

    # Output params
    output_dir: str
    epochs_per_checkpoint: int


def load_config(config_path: str) -> TrainConfig:
    """Load and instantiate config from a YAML file."""
    with open(config_path, 'r') as config_file:
        config_dict = yaml.load(config_file, Loader=yaml.SafeLoader)

    config_dict['sam_variant'] = SAMVariant(config_dict['sam_variant'])
    return TrainConfig(**config_dict)
