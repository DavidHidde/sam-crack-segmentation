import os
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as sm

from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score
from tqdm import tqdm

from data import gather_datasets
from load_sam import SAMVariant
from model import CrackSAM

SAM_VARIANT = SAMVariant.BASE

DATASET = '/home/dboerema/my-scratch/dataset/original'
BATCH_SIZE = 8
TEST_SPLIT = 0.4
SHUFFLE_DATA = True
RUN_ID = 'crack_sam_test_unet_frozen'

LABEL_COLOUR = np.flip((129, 66, 255)).astype(np.float32)
OPACITY = 0.6


def overlay_label(image_tensor, label_tensor):
    image = np.transpose(image_tensor.detach().cpu().numpy().astype(np.uint8) * 255, [1, 2, 0])
    prediction_img = (label_tensor.detach().cpu().numpy() > 0.5).astype(np.uint8).squeeze() * 255

    mask = prediction_img > 0
    overlayed = np.copy(image)
    overlayed[mask, :] = (OPACITY * LABEL_COLOUR + (1 - OPACITY) * image[mask, :].astype(float)).astype(image.dtype)
    return np.flip(overlayed, axis=2)

def main():
    """Main entrypoint"""
    output_dir = os.path.join('output', RUN_ID)
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')
    model = CrackSAM(SAM_VARIANT, device=device, freeze_encoder=True)
    model.load_state_dict(torch.load(os.path.join(output_dir, '34-f1-0.68.pt'), weights_only=True, map_location=device))

    train_dataset, test_dataset = gather_datasets(os.path.join('data', DATASET), TEST_SPLIT, model.image_size)
    loader_args = {
        'batch_size': BATCH_SIZE,
        'shuffle': SHUFFLE_DATA,
        'num_workers': 6,
        'pin_memory': True,
        'persistent_workers': True
    }
    test_dataloader = DataLoader(test_dataset, **loader_args)
    metric = BinaryF1Score(threshold=0.5).to(device)
    output_dir = os.path.join(output_dir, 'predictions')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    

    model.eval()
    idx = 0
    with torch.no_grad():
        for (inputs, labels) in tqdm(test_dataloader):
            inputs, labels = inputs.to(device, non_blocking=True, memory_format=torch.channels_last), labels.to(device, non_blocking=True, memory_format=torch.channels_last)
            with torch.amp.autocast(device.type, dtype=torch.float16):
                outputs = model(inputs)
                metric(outputs, labels)
                for (input_tensor, output_tensor) in zip(inputs, outputs):
                    cv2.imwrite(os.path.join(output_dir, f'{idx}.png'), overlay_label(input_tensor, output_tensor))
                    idx += 1

    print('F1-score:', metric.compute())

if __name__ == "__main__":
    main()
