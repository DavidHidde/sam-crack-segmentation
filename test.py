from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex, BinaryRecall

from model import SAM2Wrapper
from train import validate_epoch
from util.config import TrainConfig, load_config
from util.data import gather_datasets


def main(
    config: TrainConfig,
    weights_path: str,
    override_image_dir: str = None,
    override_label_dir: str = None
) -> None:
    """Main entrypoint"""
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')
    model = SAM2Wrapper(
        config.sam_variant,
        freeze_encoder_trunk=config.freeze_trunk,
        freeze_encoder_neck=config.freeze_neck,
        freeze_prompt=config.freeze_prompt_encoder,
        freeze_decoder=config.freeze_mask_decoder,
        apply_lora=config.apply_lora,
        lora_rank=config.lora_rank,
        device=device
    )
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
    model.eval()

    _, test_dataset = gather_datasets(
        config.image_directory if not override_image_dir else override_image_dir,
        config.label_directory if not override_label_dir else override_label_dir,
        config.test_split,
        (model.model.image_size, model.model.image_size),
        config.mask_threshold,
        config.augment,
        config.crop_image
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True
    )

    metrics = {
        'validation_f1': BinaryF1Score(threshold=config.mask_threshold).to(device),
        'validation_iou': BinaryJaccardIndex(threshold=config.mask_threshold).to(device),
        'validation_recall': BinaryRecall(threshold=config.mask_threshold).to(device),
    }
    metric_vals = validate_epoch(model, test_dataloader, metrics.values(), device)
    pretty_vals = [f"{metric_name}: {(value * 100):.2f}" for metric_name, value in zip(metrics.keys(), metric_vals)]
    print(' | '.join(pretty_vals))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str, dest='config_path', required=True, help='Path to config file')
    parser.add_argument(
        '-w',
        '--weights',
        type=str,
        dest='weights_path',
        required=True,
        help='Path to model weights file. Should fit the set configuration.'
    )
    parser.add_argument(
        '--dataset',
        '-d',
        type=str,
        nargs=2,
        dest='dataset',
        required=False,
        help='Path to dataset images (first) and labels (second)',
        default=None
    )
    args = parser.parse_args()
    main(load_config(args.config_path), args.weights_path, *args.dataset)
