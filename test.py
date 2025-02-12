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

SAM_VARIANT = SAMVariant.TINY

DATASET = '...'
BATCH_SIZE = 2
TEST_SPLIT = 0.2
SHUFFLE_DATA = True

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 10
SCHEDULER_STEP_SIZE = 20
SCHEDULER_GAMMA = 0.2
GRADIENT_ACCUMULATION_STEPS = 4

EPOCHS_PER_CHECKPOINT = 10
EPOCHS_PER_LOG = 5
RUN_ID = 'crack_sam_test'


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module
) -> float:
    """Train the model for a single epoch and return the average loss."""
    total_loss = 0.

    for data in tqdm(dataloader):
        inputs, labels = data

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    metric: torch.nn.Module
) -> tuple[float, float]:
    """Validate the epoch and return the average loss and F1-score."""
    total_loss = 0.
    metric.reset()
    with torch.no_grad():
        for data in tqdm(dataloader):
            inputs, labels = data
            outputs = model(inputs)
            total_loss += loss_fn(outputs, labels)
            metric(outputs.type(torch.float16) / 255, labels.type(torch.float16) / 255)

    return total_loss / len(dataloader), metric.compute()


def main():
    """Main entrypoint"""
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')
    model = CrackSAM(SAM_VARIANT, device=device)

    train_dataset, test_dataset = gather_datasets(os.path.join('data', DATASET), TEST_SPLIT, device)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # scaler = torch.amp.GradScaler()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
    loss_fn = sm.losses.DiceLoss(mode='binary', eps=1e-7)
    metric = BinaryF1Score().to(device)

    output_dir = os.path.join('output', RUN_ID)
    logger = CSVLogger(exp_name='logs', log_dir=output_dir)
    best_score = 0

    # Start actual training
    for epoch in range(1, EPOCHS + 1):
        print(f'Epoch {epoch}/{EPOCHS}')

        model.train(True)
        average_train_loss = train_epoch(model, train_dataloader, optimizer, loss_fn)
        model.eval()
        average_val_loss, average_f1 = validate_epoch(model, test_dataloader, loss_fn, metric)

        print(f'Train loss: {average_train_loss} | Validation loss: {average_val_loss} | F1-score: {average_f1}')
        logger.log_scalar('Training loss', average_train_loss, step=epoch)
        logger.log_scalar('Validation loss', average_val_loss, step=epoch)
        logger.log_scalar('Validation F1-score', average_f1, step=epoch)

        if average_f1 > best_score:
            filename = f'{epoch}-f1-{average_f1:.2f}'
            print(f'Model improved, saving to {filename}')
            torch.save(model.state_dict(), os.path.join(output_dir, filename))

        epoch += 1


if __name__ == "__main__":
    main()
