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

    # Model params
    sam_variant: SAMVariant
    freeze_encoder: bool

    # Data params
    image_directory: str
    label_directory: str
    test_split: float
    mask_threshold: float
    shuffle: bool
    augment: bool

    # Output params
    output_dir: str
    epochs_per_checkpoint: int


def load_config(config_path: str) -> TrainConfig:
    """Load and instantiate config from a YAML file."""
    with open(config_path, 'r') as config_file:
        config_dict = yaml.load(config_file, Loader=yaml.SafeLoader)

    config_dict['sam_variant'] = SAMVariant(config_dict['sam_variant'])
    return TrainConfig(**config_dict)
