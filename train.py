import os
import time
from argparse import ArgumentParser

import torch
import segmentation_models_pytorch as sm

from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score
from tqdm import tqdm

from model import SAM2Wrapper
from util.config import TrainConfig, load_config
from util.data import gather_datasets
from util.log import CSVLogger, MetricItem


def train_epoch(
    model: SAM2Wrapper,
    dataloader: DataLoader,
    scaler: torch.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    gradient_accumulation_steps: int,
) -> float:
    """Train the model for a single epoch and return the average loss."""
    total_loss = 0.
    resize_dims = [model.model.image_size, model.model.image_size]
    for idx, (inputs, labels) in enumerate(tqdm(dataloader)):
        inputs, labels = inputs.to(device, non_blocking=True, memory_format=torch.channels_last), labels.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last
        )
        # Get output and apply gradient
        with torch.amp.autocast(device.type, dtype=torch.float16):
            outputs = model(inputs)
            outputs = F.interpolate(outputs, resize_dims, mode='bilinear', align_corners=False)
            loss = loss_fn(outputs, labels)
            loss = loss / gradient_accumulation_steps

        total_loss += loss.item()
        scaler.scale(loss).backward()

        # Apply gradient accumulation
        if (idx + 1) % gradient_accumulation_steps == 0 or (idx + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    return total_loss / len(dataloader)


def validate_epoch(
    model: SAM2Wrapper,
    dataloader: DataLoader,
    metric: torch.nn.Module,
    device: torch.device
) -> float:
    """Validate the epoch and return the average loss and F1-score."""
    metric.reset()
    resize_dims = [model.model.image_size, model.model.image_size]
    with torch.no_grad():
        for (inputs, labels) in tqdm(dataloader):
            inputs, labels = inputs.to(device, non_blocking=True, memory_format=torch.channels_last), labels.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last
            )
            with torch.amp.autocast(device.type, dtype=torch.float16):
                outputs = model(inputs)
                outputs = F.interpolate(outputs, resize_dims, mode='bilinear', align_corners=False)
                metric(outputs, labels)

    return metric.compute().item()


def main(config: TrainConfig) -> None:
    """Main entrypoint"""
    torch.manual_seed(0) # Fix randomness
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

    train_dataset, test_dataset = gather_datasets(
        config.image_directory,
        config.label_directory,
        config.test_split,
        (model.model.image_size, model.model.image_size),
        config.mask_threshold,
        config.augment,
        config.crop_image
    )
    loader_args = {
        'batch_size': config.batch_size,
        'shuffle': config.shuffle,
        'num_workers': 12,
        'pin_memory': True,
        'persistent_workers': True
    }
    train_dataloader = DataLoader(train_dataset, **loader_args)
    test_dataloader = DataLoader(test_dataset, **loader_args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.scheduler_step_size,
        gamma=config.scheduler_gamma
    )
    loss_fn = sm.losses.DiceLoss(mode='binary', eps=1e-7)
    metric = BinaryF1Score(threshold=config.mask_threshold).to(device)

    output_dir = os.path.join(config.output_dir, config.id)
    tracked_values = {
        'epoch': MetricItem(name='Epoch', value=0),
        'time': MetricItem(name='Time (s)', value=0),
        'training_loss': MetricItem(name='Training loss', value=0),
        'validation_f1': MetricItem(name='Validation F1-score', value=0),
    }
    logger = CSVLogger(os.path.join(output_dir, 'log.csv'), items=tracked_values.values())

    best_score = 0
    start_time = time.time()

    # Start actual training
    for epoch in range(1, config.epochs + 1):
        print(f'Epoch {epoch}/{config.epochs}')
        tracked_values['epoch'].value = epoch

        model.train()
        tracked_values['training_loss'].value = train_epoch(
            model,
            train_dataloader,
            scaler,
            optimizer,
            loss_fn,
            device,
            config.gradient_accumulation_steps
        )
        model.eval()
        average_f1 = validate_epoch(model, test_dataloader, metric, device)
        tracked_values['validation_f1'].value = average_f1
        tracked_values['time'].value = time.time() - start_time

        pretty_vals = [f"{item.name}: {round(item.value, 4) if '.' in str(item.value) else item.value}" for item in
            tracked_values.values()]
        print(' | '.join(pretty_vals))
        logger.write_line()

        if average_f1 > best_score:
            filename = f'{epoch}-f1-{average_f1:.2f}.pt'
            print(f'Model improved from {best_score:.4f} to {average_f1:.4f}, saving to {filename}')
            torch.save(model.state_dict(), os.path.join(output_dir, filename))
            best_score = average_f1
        elif epoch % config.epochs_per_checkpoint == 0:
            filename = f'{epoch}-checkpoint.pt'
            print(f'Score did not improve from {best_score:.4f}, saving to checkpoint {filename}')
            torch.save(model.state_dict(), os.path.join(output_dir, filename))
        else:
            print(f'Score did not improve from {best_score:.4f}')

        epoch += 1
        scheduler.step()

    logger.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str, dest='config_path', required=True, help='Path to config file')
    args = parser.parse_args()
    main(load_config(args.config_path))
