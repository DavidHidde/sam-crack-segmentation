import torch
from argparse import ArgumentParser

from model import SAM2Wrapper
from util.config import TrainConfig, load_config


def main(
    config: TrainConfig,
    weights_path: str,
    output_path: str = None,
) -> None:
    """Main entrypoint"""
    device = torch.device('cpu')
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

    example_inputs = (torch.randn(1, 3, 1024, 1024),)
    export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
    onnx_program = torch.onnx.export(
        model,
        example_inputs,
        dynamo=True,
        export_options=export_options,
    )
    onnx_program.optimize()

    if output_path:
        onnx_program.save(output_path)
    else:
        parts = weights_path.split('.')[:-1]
        onnx_program.save(f'{'.'.join(parts)}.onnx')


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
        '-o',
        '--output',
        type=str,
        dest='output_path',
        required=False,
        default=None,
        help='Path of the output file. If not provided, will be the same as the original model with the .onnx extension.'
    )
    args = parser.parse_args()
    main(load_config(args.config_path), args.weights_path, output_path=args.output_path)
