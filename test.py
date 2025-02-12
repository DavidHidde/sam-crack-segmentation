import os
import torch

from torch.utils.data import DataLoader

from data import gather_datasets
from load_sam import SAMVariant
from model import CrackSAM

SAM_VARIANT = SAMVariant.SMALL

DATASET = '...'
BATCH_SIZE = 64
TEST_SPLIT = 0.2
SHUFFLE_DATA = True


def get_torch_device() -> torch.device:
    """Select the best device for torch to run on."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    """Main entrypoint"""
    device = get_torch_device()
    model = CrackSAM(SAM_VARIANT, device=device)

    train_dataset, test_dataset = gather_datasets(os.path.join('data', DATASET), TEST_SPLIT)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA)


if __name__ == "__main__":
    main()
