import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from PIL import Image

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

MODEL_CHECKPOINT = os.path.join('sam-checkpoints', 'sam2.1_hiera_small.pt')
MODEL_CONFIG = os.path.join('configs', 'sam2.1', 'sam2.1_hiera_s.yaml') # Relative to sam2/sam2 repo

IMAGE = '...'

# Load model and set image
model = build_sam2(MODEL_CONFIG, MODEL_CHECKPOINT, device=torch.device('mps'))
predictor = SAM2ImagePredictor(model)

image = Image.open(IMAGE)
image = np.array(image.convert('RGB'))
predictor.set_image(image)

# Plot image and get marker points
points = []
colors = []
fig = plt.figure()

def on_click(event):
    if event.button is MouseButton.LEFT:
        point = int(event.xdata), int(event.ydata)
        points.append([point])
        sc = plt.scatter(point[0], [point[1]])
        colors.append(sc.get_facecolor()[0, :3])
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event', on_click)
plt.imshow(image)
plt.axis('off')
plt.show()

# Run prediction
print('Running prediction...')
masks, scores, _ = predictor.predict(
    point_coords=points,
    point_labels=np.ones((len(points), 1)),
    multimask_output=True,
)
masks = masks[range(len(masks)), np.argmax(scores, axis=-1)]

# Show masks
plt.gca()
plt.imshow(image)
for idx, mask in enumerate(masks):
    color = np.concatenate([colors[idx], np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    plt.imshow(mask_image)

plt.axis('off')
plt.show()
