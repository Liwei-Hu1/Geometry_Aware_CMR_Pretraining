import os
import sys
from pathlib import Path
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from timm.models import register_model

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream_stage.seg.modeling_segementation import (
    SegMAEUNETR,
    SegMAEVol,
    SegmentationModelConfig,
)
from pretrain_stage.modeling_pretrain import MODEL_REGISTRY as PRETRAIN_REGISTRY


class SegMAEWrapper(nn.Module):
    """Wrapper around SegMAE to adapt its output format."""

    def __init__(self, seg_mae_instance: nn.Module) -> None:
        super().__init__()
        self.output_fmt = "BSCTHW"
        self.seg_mae = seg_mae_instance
        self.backbone = self.seg_mae.shared_encoder

    def forward(self, videos: torch.Tensor, patch_coords_3d: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.seg_mae(videos, patch_coords_3d=patch_coords_3d)
        out = out.permute(0, 2, 1, 3, 4, 5)
        return out


def create_finetune_model(
    backbone_name: str,
    checkpoint_path: Optional[str] = None,
    num_classes: int = 4,
    img_size: int = 112,
    patch_size: int = 4,
    num_frames: int = 32,
    tubelet_size: int = 8,
    num_slices: int = 6,
    decoder_feature: int = 32,
) -> nn.Module:
    is_lite = "lite" in backbone_name.lower()
    pretrain_backbone_name = backbone_name
    if is_lite:
        pretrain_backbone_name = backbone_name.replace("seg_lite_", "pretrain_multi")

    print(f"Creating {'SegMAELite' if is_lite else 'SegMAE'} model with backbone: {pretrain_backbone_name}")

    backbone_ctor = PRETRAIN_REGISTRY.get(pretrain_backbone_name)
    if backbone_ctor is None:
        raise ValueError(f"Backbone {pretrain_backbone_name} not found in registry")

    temp_backbone = backbone_ctor()

    if checkpoint_path:
        print(f"Loading pretrained weights from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        msg = temp_backbone.load_state_dict(state_dict, strict=True)
        print(f"Backbone load result: {msg}")

    shared_encoder = temp_backbone.shared_encoder
    volume_head = getattr(temp_backbone, "volume_head", None)

    embed_dim = shared_encoder.embed_dim
    depth = len(shared_encoder.blocks)
    num_heads = shared_encoder.blocks[0].attn.num_heads

    frame_latent = num_frames // tubelet_size
    size_latent = img_size // patch_size

    if is_lite:
        config = SegmentationModelConfig(
            img_size=img_size,
            num_frames=num_frames,
            num_slices=num_slices,
            encoder_embed_dim=embed_dim,
            encoder_depth=depth,
            encoder_num_heads=num_heads,
            decoder_num_classes=num_classes,
            decoder_feature=decoder_feature,
            decoder_embed_dim=embed_dim // 2,
            tubelet_size=tubelet_size,
            decoder_grid_size=(num_slices, frame_latent, size_latent, size_latent),
        )
        model = SegMAEVol(config, shared_encoder=shared_encoder, volume_head=volume_head, inner_dim=decoder_feature)
    else:
        config = SegmentationModelConfig(
            img_size=img_size,
            num_frames=num_frames,
            num_slices=num_slices,
            encoder_embed_dim=embed_dim,
            encoder_depth=depth,
            encoder_num_heads=num_heads,
            decoder_num_classes=num_classes,
            decoder_feature=decoder_feature,
            decoder_embed_dim=embed_dim // 2,
            tubelet_size=tubelet_size,
            decoder_grid_size=(num_slices, frame_latent, size_latent, size_latent),
        )
        model = SegMAEUNETR(config, shared_encoder=shared_encoder, volume_head=volume_head)

    return model
