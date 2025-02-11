import os

import torch

from load_sam import SAMVariant

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from model import CrackSAM


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
    model = CrackSAM(SAMVariant.SMALL, device=device)


if __name__ == "__main__":
    main()
