import os
import torch
import segmentation_models_pytorch as sm

from torch.utils.data import DataLoader
from torchrl.record import CSVLogger
from torchmetrics.classification import BinaryF1Score
from tqdm import tqdm

from data import gather_datasets
from load_sam import SAMVariant
from model import CrackSAM

SAM_VARIANT = SAMVariant.BASE

DATASET = '...'
BATCH_SIZE = 8
TEST_SPLIT = 0.4
SHUFFLE_DATA = True

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 50
SCHEDULER_STEP_SIZE = 10
SCHEDULER_GAMMA = 0.2
GRADIENT_ACCUMULATION_STEPS = 4

EPOCHS_PER_CHECKPOINT = 10
RUN_ID = '...'


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    scaler: torch.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device
) -> float:
    """Train the model for a single epoch and return the average loss."""
    total_loss = 0.
    for idx, (inputs, labels) in enumerate(tqdm(dataloader)):
        inputs, labels = inputs.to(device, non_blocking=True, memory_format=torch.channels_last), labels.to(device, non_blocking=True, memory_format=torch.channels_last)
        # Get output and apply gradient
        with torch.amp.autocast(device.type, dtype=torch.float16):
            output = model(inputs)
            loss = loss_fn(output, labels)
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            
        total_loss += loss.item()
        scaler.scale(loss).backward()
        
        # Apply gradient accumulation
        if (idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (idx + 1) == len(dataloader):
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
            inputs, labels = inputs.to(device, non_blocking=True, memory_format=torch.channels_last), labels.to(device, non_blocking=True, memory_format=torch.channels_last)
            with torch.amp.autocast(device.type, dtype=torch.float16):
                outputs = model(inputs)
                metric(outputs, labels)

    return metric.compute()


def main():
    """Main entrypoint"""
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')
    model = CrackSAM(SAM_VARIANT, device=device, freeze_encoder=True)

    train_dataset, test_dataset = gather_datasets(os.path.join('data', DATASET), TEST_SPLIT, model.image_size)
    loader_args = {
        'batch_size': BATCH_SIZE,
        'shuffle': SHUFFLE_DATA,
        'num_workers': 6,
        'pin_memory': True,
        'persistent_workers': True
    }
    train_dataloader = DataLoader(train_dataset, **loader_args)
    test_dataloader = DataLoader(test_dataset, **loader_args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
    loss_fn = sm.losses.DiceLoss(mode='binary', eps=1e-7)
    metric = BinaryF1Score(threshold=0.5).to(device)

    output_dir = os.path.join('output', RUN_ID)
    logger = CSVLogger(exp_name='logs', log_dir=output_dir)
    best_score = 0

    # Start actual training
    for epoch in range(1, EPOCHS + 1):
        print(f'Epoch {epoch}/{EPOCHS}')

        model.train()
        average_train_loss = train_epoch(model, train_dataloader, scaler, optimizer, loss_fn, device)
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
        elif epoch % EPOCHS_PER_CHECKPOINT == 0:
            filename = f'{epoch}-checkpoint.pt'
            print(f'Score did not improve from {best_score}, saving to checkpoint {filename}')
            torch.save(model.state_dict(), os.path.join(output_dir, filename))
        else:
            print(f'Score did not improve from {best_score}')

        epoch += 1
        scheduler.step()


if __name__ == "__main__":
    main()
