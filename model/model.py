import loralib as lora
import torch
from torch import nn, Tensor

from sam2.modeling.sam2_base import SAM2Base
from .functional.load_sam import SAMVariant, load_sam

FEAT_SIZES = [
    (64, 64),
    (128, 128),
    (256, 256),
]


class SAM2Wrapper(nn.Module):
    """A wrapper for training around Meta's SAM 2.1 model."""

    model: SAM2Base

    def __init__(
        self,
        sam_variant: SAMVariant,
        freeze_encoder_trunk: bool = True,
        freeze_encoder_neck: bool = True,
        freeze_prompt: bool = False,
        freeze_decoder: bool = False,
        apply_lora: bool = False,
        lora_rank: int = 4,
        device: torch.device = torch.device('cpu'),
        **kwargs
    ):
        """Initialize the model."""
        super(SAM2Wrapper, self).__init__(**kwargs)
        self.model = load_sam(sam_variant, device)
        self.freeze_components(freeze_encoder_trunk, freeze_encoder_neck, freeze_prompt, freeze_decoder)

        if apply_lora:
            self.inject_lora(lora_rank, device)

    def freeze_components(
        self,
        freeze_encoder_trunk: bool,
        freeze_encoder_neck: bool,
        freeze_prompt: bool,
        freeze_decoder: bool
    ) -> None:
        """Freeze specific components of the model based on the requirements."""
        if freeze_encoder_trunk:
            for param in self.model.image_encoder.trunk.parameters():
                param.requires_grad = False

        if freeze_encoder_neck:
            for param in self.model.image_encoder.neck.parameters():
                param.requires_grad = False

        if freeze_prompt:
            for param in self.model.sam_prompt_encoder.parameters():
                param.requires_grad = False

        if freeze_decoder:
            for param in self.model.sam_mask_decoder.parameters():
                param.requires_grad = False

    def inject_lora(self, lora_rank: int, device: torch.device) -> None:
        """Inject LoRA matrices into the encoder. Will always freeze the trunk."""
        for block in self.model.image_encoder.trunk.blocks:
            current_qkv = block.attn.qkv
            new_qkv = lora.Linear(
                current_qkv.in_features,
                current_qkv.out_features,
                bias=current_qkv.bias is not None,
                device=device,
                r=lora_rank,
            )

            # Temporarily remove requires_grad for copy; This is reset in the last expression.
            for parameter in new_qkv.parameters():
                parameter.requires_grad = False

            new_qkv.bias.copy_(current_qkv.bias)
            new_qkv.weight.copy_(current_qkv.weight)

            # Finally, replace the layer.
            block.attn.qkv = new_qkv

        lora.mark_only_lora_as_trainable(self.model.image_encoder.trunk)

    def forward(self, image: Tensor) -> Tensor:
        """Forward pass of the model. Heavy inspiration taken from the SAM2ImagePredictor."""
        batch_size = image.shape[0]
        backbone_out = self.model.forward_image(image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], FEAT_SIZES)
        ][::-1]
        high_res_features = [feat_level for feat_level in feats[:-1]]

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
        sparse_embeddings = sparse_embeddings.tile((batch_size, 1, 1))
        dense_embeddings = dense_embeddings.tile((batch_size, 1, 1, 1))

        masks, _, _, _ = self.model.sam_mask_decoder(
            image_embeddings=feats[-1],
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        return masks.float()
