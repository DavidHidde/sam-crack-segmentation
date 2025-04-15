# Crack segmentation using SAM 2, LoRA and transfer learning

Simple repository aimed at binary crack segmentation using LoRA and
the [Segment Anything Model 2](https://github.com/facebookresearch/sam2).
Uses a simple config file to control training.

## Installation

All requirements, including the SAM 2 repo, can be installed through the provided `requirements.txt`. No special extra
steps are applicable.

## Usage

The repository provides 3 basic scripts:

1. [`train.py`](train.py): Train models.
2. [`test.py`](test.py): Test a model's performance.
3. [`export.py`](export.py): Export the model to ONNX format.

All scripts rely on a `TrainConfig` (an example of which is provided in [`example_config.yaml`](example_config.yaml)) to
provide info on how the model should be configured.
Future improvements include adding a simple export config to avoid the large number of parameters needed in the confiig.
Both `test.py` and `export.py` include extra CLI commands to control which trained model gets used, where the output is
stored (`export.py`) and which dataset is used (`test.py`).
