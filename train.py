import os
from argparse import ArgumentParser

import torch
import segmentation_models_pytorch as sm

from torch.utils.data import DataLoader
from torchrl.record import CSVLogger
from torchmetrics.classification import BinaryF1Score
from tqdm import tqdm

from model import CrackSAM
from util.config import TrainConfig, load_config
from util.data import gather_datasets


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    scaler: torch.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    gradient_accumulation_steps: int,
) -> float:
    """Train the model for a single epoch and return the average loss."""
    total_loss = 0.
    for idx, (inputs, labels) in enumerate(tqdm(dataloader)):
        inputs, labels = inputs.to(device, non_blocking=True, memory_format=torch.channels_last), labels.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last
        )
        # Get output and apply gradient
        with torch.amp.autocast(device.type, dtype=torch.float16):
            output = model(inputs)
            loss = loss_fn(output, labels)
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
    model: torch.nn.Module,
    dataloader: DataLoader,
    metric: torch.nn.Module,
    device: torch.device
) -> tuple[float, float]:
    """Validate the epoch and return the average loss and F1-score."""
    metric.reset()
    with torch.no_grad():
        for (inputs, labels) in tqdm(dataloader):
            inputs, labels = inputs.to(device, non_blocking=True, memory_format=torch.channels_last), labels.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last
            )
            with torch.amp.autocast(device.type, dtype=torch.float16):
                outputs = model(inputs)
                metric(outputs, labels)

    return metric.compute()


def main(config: TrainConfig) -> None:
    """Main entrypoint"""
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')
    model = CrackSAM(config.sam_variant, device=device, freeze_encoder=config.freeze_encoder)

    train_dataset, test_dataset = gather_datasets(
        config.image_directory,
        config.label_directory,
        config.test_split,
        (model.image_size, model.image_size),
        config.mask_threshold,
        config.augment
    )
    loader_args = {
        'batch_size': config.batch_size,
        'shuffle': config.shuffle,
        'num_workers': 6,
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
    metric = BinaryF1Score(threshold=0.5).to(device)

    output_dir = os.path.join(config.output_dir, config.id)
    logger = CSVLogger(exp_name='logs', log_dir=output_dir)
    best_score = 0

    # Start actual training
    for epoch in range(1, config.epochs + 1):
        print(f'Epoch {epoch}/{config.epochs}')

        model.train()
        average_train_loss = train_epoch(
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

        print(f'Train loss: {average_train_loss} | Validation F1-score: {average_f1}')
        logger.log_scalar('Training loss', average_train_loss, step=epoch)
        logger.log_scalar('Validation F1-score', average_f1, step=epoch)

        if average_f1 > best_score:
            best_score = average_f1
            filename = f'{epoch}-f1-{average_f1:.2f}.pt'
            print(f'Model improved to {average_f1}, saving to {filename}')
            torch.save(model.state_dict(), os.path.join(output_dir, filename))
        elif epoch % config.epochs_per_checkpoint == 0:
            filename = f'{epoch}-checkpoint.pt'
            print(f'Score did not improve from {best_score}, saving to checkpoint {filename}')
            torch.save(model.state_dict(), os.path.join(output_dir, filename))
        else:
            print(f'Score did not improve from {best_score}')

        epoch += 1
        scheduler.step()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str, dest='config_path', required=True, help='Path to config file')
    args = parser.parse_args()
    main(load_config(args.config_path))
