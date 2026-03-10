"""Segmentation-specific models reusing the pretrained encoder."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pretrain_stage.modeling_pretrain import PretrainVisionTransformerEncoder
from downstream_stage.seg.decoder.unetr_decoder import (
    DecoderConfig,
    SegmentationUNETRDecoder,
)
from downstream_stage.seg.decoder.vol_seg_decoder import VolSegDecoder
from monai.losses import DiceLoss, DiceCELoss

ModelConstructor = Callable[..., nn.Module]
MODEL_REGISTRY: Dict[str, ModelConstructor] = {}


def register_model(name: str) -> Callable[[ModelConstructor], ModelConstructor]:
    """Decorator to register a segmentation model architecture."""
    def decorator(fn: ModelConstructor) -> ModelConstructor:
        MODEL_REGISTRY[name] = fn
        return fn

    return decorator


# =========================================================
# Loss
# =========================================================
class SegmentationCriterion(nn.Module):
    """Dice + CE loss for segmentation, plus per-class Dice reporting."""

    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()

        ce_weight = torch.ones(num_classes)
        if num_classes > 4:
            ce_weight[4] = 2.0
        if num_classes > 5:
            ce_weight[5] = 3.0

        self.register_buffer("_ce_weight", ce_weight)

        self.loss_fct = DiceCELoss(
            softmax=True,
            include_background=True,
            batch=True,
            reduction="mean",
            lambda_dice=1.0,
            lambda_ce=1.0,
            ce_weight=ce_weight,
        )

        self.dice_only = DiceLoss(
            reduction="none",
            softmax=True,
            include_background=True,
            batch=True,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pred:   [B, C, S, T, H, W]
            target: [B, C, S, T, H, W]

        Returns:
            loss: scalar
            fg_dice: [C-1]
        """
        assert pred.shape == target.shape, (
            f"pred.shape={pred.shape} != target.shape={target.shape}"
        )

        B, C, S, T, H, W = pred.shape
        pred_ = pred.permute(0, 2, 1, 3, 4, 5).reshape(B * S, C, T, H, W)
        target_ = target.permute(0, 2, 1, 3, 4, 5).reshape(B * S, C, T, H, W)

        loss = self.loss_fct(pred_, target_)

        dice_per_class = 1.0 - self.dice_only(pred_, target_)  # [C]
        fg_dice = dice_per_class[1:]  # skip background
        return loss, fg_dice


# =========================================================
# Config
# =========================================================
@dataclass
class SegmentationModelConfig:
    """Configuration for segmentation models."""

    img_size: int = 112
    num_frames: int = 32
    num_slices: int = 6
    in_chans: int = 1

    encoder_embed_dim: int = 192
    encoder_depth: int = 9
    encoder_num_heads: int = 3

    decoder_embed_dim: int = 96
    decoder_feature: int = 32
    decoder_num_classes: int = 3

    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    norm_layer: Callable[[int], nn.Module] = partial(nn.LayerNorm, eps=1e-6)
    init_values: float = 0.0
    tubelet_size: int = 8
    use_checkpoint: bool = False

    decoder_grid_size: Sequence[int] = (9, 4, 28, 28)
    decoder_upsample_kernels: Sequence[Sequence[int]] = (
        (1, 2, 2),
        (2, 2, 2),
        (1, 1, 1),
    )

    def to_decoder_config(self) -> DecoderConfig:
        return DecoderConfig(
            in_channels=1,
            out_channels=self.decoder_num_classes,
            img_size=(self.num_slices, self.img_size, self.img_size),
            feature_size=self.decoder_feature,
            hidden_size=self.decoder_embed_dim,
            grid_size=self.decoder_grid_size,
            upsample_kernel_sizes=self.decoder_upsample_kernels,
        )


_default_tiny_patch8 = SegmentationModelConfig(
    encoder_embed_dim=192,
    encoder_depth=9,
    encoder_num_heads=3,
    decoder_embed_dim=96,
    decoder_feature=14,
    decoder_num_classes=3,
)


def build_segmentation_config(
    num_classes: int = 6,
    img_size: int = 112,
    num_frames: int = 32,
    tubelet_size: int = 8,
    num_slices: int = 6,
    decoder_feature: int = 32,
) -> SegmentationModelConfig:
    cfg = replace(_default_tiny_patch8)
    cfg.img_size = img_size
    cfg.num_frames = num_frames
    cfg.num_slices = num_slices
    cfg.tubelet_size = tubelet_size
    cfg.decoder_feature = decoder_feature
    cfg.decoder_num_classes = num_classes
    return cfg


# =========================================================
# Encoder helpers
# =========================================================
def build_or_use_shared_encoder(
    config: SegmentationModelConfig,
    shared_encoder: Optional[PretrainVisionTransformerEncoder] = None,
) -> PretrainVisionTransformerEncoder:
    if shared_encoder is not None:
        return shared_encoder

    return PretrainVisionTransformerEncoder(
        img_size=config.img_size,
        num_frames=config.num_frames,
        num_slices=config.num_slices,
        in_chans=config.in_chans,
        embed_dim=config.encoder_embed_dim,
        depth=config.encoder_depth,
        num_heads=config.encoder_num_heads,
        mlp_ratio=config.mlp_ratio,
        qkv_bias=config.qkv_bias,
        qk_scale=config.qk_scale,
        drop_rate=config.drop_rate,
        attn_drop_rate=config.attn_drop_rate,
        drop_path_rate=config.drop_path_rate,
        norm_layer=config.norm_layer,
        init_values=config.init_values,
        tubelet_size=config.tubelet_size,
        use_checkpoint=config.use_checkpoint,
    )


# =========================================================
# Base model
# =========================================================
class BaseSegmentationModel(nn.Module):
    """Base class for segmentation models with shared encoder logic."""

    def __init__(
        self,
        config: SegmentationModelConfig,
        shared_encoder: Optional[PretrainVisionTransformerEncoder] = None,
        volume_head: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.volume_head = volume_head

        self.shared_encoder = build_or_use_shared_encoder(config, shared_encoder)

        self.decoder_num_classes = config.decoder_num_classes
        self.segmentation_criterion = SegmentationCriterion(config.decoder_num_classes)

    def _encode_batch(
        self,
        videos: torch.Tensor,
        view_ids: torch.Tensor,
        patch_coords_3d: Optional[torch.Tensor] = None,
        skip_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Args:
            videos: [B_tot, 1, T, H, W]
            view_ids: [B_tot]
            patch_coords_3d: [B_tot, N_sp, 3] or None
            skip_indices: layers to save skip tokens from

        Returns:
            x: final tokens [B_tot, N, D]
            hidden_states: dict[layer_idx -> tokens]
        """
        x = self.shared_encoder.patch_embed(videos)
        B_tot, N_tot, _ = x.shape

        x = x + self.shared_encoder.pos_embed[:, 1:, :].type_as(x).to(x.device)

        if hasattr(self.shared_encoder, "view_embed"):
            x = x + self.shared_encoder.view_embed(view_ids).unsqueeze(1)

        if patch_coords_3d is not None and hasattr(self.shared_encoder, "patch_spatial_proj"):
            N_sp = patch_coords_3d.shape[1]
            T_p = N_tot // N_sp
            coords_expanded = (
                patch_coords_3d.unsqueeze(1)
                .expand(-1, T_p, -1, -1)
                .reshape(B_tot, N_tot, 3)
            )
            x = x + self.shared_encoder.patch_spatial_proj(coords_expanded.type_as(x))

        hidden_states: Dict[int, torch.Tensor] = {}
        for i, blk in enumerate(self.shared_encoder.blocks):
            if skip_indices is not None and i in skip_indices:
                hidden_states[i] = x
            x = blk(x)

        x = self.shared_encoder.norm(x)
        return x, hidden_states

    def _encode_context(self):
        is_frozen = not any(p.requires_grad for p in self.shared_encoder.parameters())
        return torch.no_grad() if is_frozen else torch.enable_grad()

    def _prepare_common_inputs(
        self,
        videos: torch.Tensor,
        patch_coords_3d: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int, int, int, int, int]:
        """
        Flatten [B, S, T, H, W] to encoder-ready inputs.

        Returns:
            videos_flat: [B*S, 1, T, H, W]
            view_ids: [B*S]
            patch_coords_flat: [B*S, N_sp, 3] or None
            B, S, T, H, W
        """
        B, S, T, H, W = videos.shape
        videos_flat = videos.view(B * S, 1, T, H, W)
        view_ids = (
            torch.arange(S, device=videos.device).unsqueeze(0).expand(B, -1).flatten()
        )
        patch_coords_flat = (
            patch_coords_3d.reshape(B * S, -1, 3)
            if patch_coords_3d is not None
            else None
        )
        return videos_flat, view_ids, patch_coords_flat, B, S, T, H, W

    def _labels_to_onehot(self, segs: torch.Tensor) -> torch.Tensor:
        """
        segs: [B, S, T, H, W]
        return: [B, C, S, T, H, W]
        """
        gt_onehot = F.one_hot(segs.long(), num_classes=self.decoder_num_classes)
        return gt_onehot.permute(0, 5, 1, 2, 3, 4).float()

    def prepare_targets(
        self,
        segs: torch.Tensor,
        preds: torch.Tensor,
    ) -> torch.Tensor:
        """Override in subclasses if needed."""
        return self._labels_to_onehot(segs)

    def compute_loss(
        self,
        videos: torch.Tensor,
        segs: torch.Tensor,
        patch_coords_3d: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        preds = self.forward(videos, patch_coords_3d=patch_coords_3d)
        with torch.no_grad():
            gt_onehot = self.prepare_targets(segs, preds)
        return self.segmentation_criterion(preds, gt_onehot)


# =========================================================
# UNETR model
# =========================================================
class SegMAEUNETR(BaseSegmentationModel):
    """VideoMAE-based segmentation model with UNETR decoder."""

    SKIP_INDICES = (2, 5)

    def __init__(
        self,
        config: SegmentationModelConfig,
        shared_encoder: Optional[PretrainVisionTransformerEncoder] = None,
        volume_head: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(
            config=config,
            shared_encoder=shared_encoder,
            volume_head=volume_head,
        )

        self.encoder_to_decoder = nn.Linear(
            config.encoder_embed_dim,
            config.decoder_embed_dim,
            bias=False,
        )
        self.decoder = SegmentationUNETRDecoder(config.to_decoder_config())

    def forward(
        self,
        videos: torch.Tensor,
        patch_coords_3d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            videos: [B, S, T, H, W]
            patch_coords_3d: [B, S, N_sp, 3] or None

        Returns:
            logits: [B, C, S, T_p, H, W]  (depends on decoder implementation)
        """
        (
            videos_flat,
            view_ids,
            patch_coords_flat,
            B,
            S,
            T,
            H,
            W,
        ) = self._prepare_common_inputs(videos, patch_coords_3d)

        with self._encode_context():
            encoded_flat, hidden_states = self._encode_batch(
                videos_flat,
                view_ids,
                patch_coords_flat,
                skip_indices=self.SKIP_INDICES,
            )

        encoded_reshaped = encoded_flat.view(B, S, -1, encoded_flat.shape[-1])
        slice_outputs = [encoded_reshaped[:, i, ...] for i in range(S)]

        skip_features: List[List[torch.Tensor]] = [[] for _ in range(len(self.SKIP_INDICES))]
        for level_i, idx in enumerate(self.SKIP_INDICES):
            h_state = hidden_states[idx]  # [B*S, N, D]
            h_state = h_state.view(B, S, -1, h_state.shape[-1])
            for s_idx in range(S):
                skip_features[level_i].append(h_state[:, s_idx, ...])

        if self.volume_head is not None:
            enc_patch = torch.stack(slice_outputs, dim=1).flatten(0, 1)  # [B*S, N, D]
            current_slices = videos.shape[1]

            if current_slices != self.volume_head.num_slices:
                _, rec_features = self.volume_head.forward_partial(
                    enc_patch,
                    active_slices=current_slices,
                )
            else:
                _, rec_features = self.volume_head(
                    enc_patch,
                    patch_coords_3d=patch_coords_3d,
                )

            B_v, S_v, N_sp, C_v = rec_features.shape
            N_total_per_slice = slice_outputs[0].shape[1]
            T_patches = N_total_per_slice // N_sp

            rec_features = rec_features.unsqueeze(2).expand(-1, -1, T_patches, -1, -1)
            proj_input = rec_features.reshape(B_v, -1, C_v)
        else:
            proj_input = torch.cat(slice_outputs, dim=1)  # [B, S*N, D]

        proj = self.encoder_to_decoder(proj_input)
        decoder_skips = [
            self.encoder_to_decoder(torch.cat(feats, dim=1)) for feats in skip_features
        ]

        return self.decoder(videos, proj, decoder_skips)


# =========================================================
# VolSeg model
# =========================================================
class SegMAEVol(BaseSegmentationModel):
    """
    Segmentation model with lightweight VolSegDecoder.

    Output:
        [B, num_classes, S, T_p, H, W]
    """

    def __init__(
        self,
        config: SegmentationModelConfig,
        shared_encoder: Optional[PretrainVisionTransformerEncoder] = None,
        volume_head: Optional[nn.Module] = None,
        inner_dim: int = 128,
    ) -> None:
        super().__init__(
            config=config,
            shared_encoder=shared_encoder,
            volume_head=volume_head,
        )

        self.tubelet_size = config.tubelet_size
        img_patch_size = self.shared_encoder.patch_embed.img_patch_size[0]
        self.patch_size = img_patch_size
        self.H_p = config.img_size // img_patch_size
        self.W_p = config.img_size // img_patch_size

        self.decoder = VolSegDecoder(
            encoder_dim=config.encoder_embed_dim,
            inner_dim=inner_dim,
            num_classes=config.decoder_num_classes,
            patch_size=img_patch_size,
        )

    def forward(
        self,
        videos: torch.Tensor,
        patch_coords_3d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            videos: [B, S, T, H, W]
            patch_coords_3d: [B, S, N_sp, 3] or None

        Returns:
            seg: [B, num_classes, S, T_p, H, W]
        """
        (
            videos_flat,
            view_ids,
            patch_coords_flat,
            B,
            S,
            T,
            H,
            W,
        ) = self._prepare_common_inputs(videos, patch_coords_3d)

        T_p = T // self.tubelet_size

        with self._encode_context():
            enc_flat, _ = self._encode_batch(
                videos_flat,
                view_ids,
                patch_coords_flat,
                skip_indices=None,
            )

        if self.volume_head is not None:
            N_expected = T_p * self.H_p * self.W_p
            if enc_flat.shape[1] == N_expected + 1:
                enc_patch_only = enc_flat[:, 1:, :]
            else:
                enc_patch_only = enc_flat

            with self._encode_context():
                _, rec_features = self.volume_head(
                    enc_patch_only,
                    patch_coords_3d=patch_coords_3d,
                )
        else:
            N_sp = self.H_p * self.W_p
            rec_features = enc_flat.view(B, S, T_p, N_sp, -1).mean(dim=2)

        N_expected = T_p * self.H_p * self.W_p
        if enc_flat.shape[1] == N_expected + 1:
            enc_for_dec = enc_flat[:, 1:, :]
        else:
            enc_for_dec = enc_flat

        seg = self.decoder(
            enc_for_dec,
            rec_features,
            B=B,
            S=S,
            T_p=T_p,
            H_p=self.H_p,
            W_p=self.W_p,
        )
        return seg

    def prepare_targets(
        self,
        segs: torch.Tensor,
        preds: torch.Tensor,
    ) -> torch.Tensor:
        """
        segs:  [B, S, T, H, W]
        preds: [B, C, S, T_p, H, W]
        """
        T_p = preds.shape[3]
        B, S, T, H, W = segs.shape

        if T % T_p != 0:
            raise ValueError(
                f"Cannot evenly downsample GT in time: T={T}, T_p={T_p}"
            )

        # Dataset logic for downsampling
        segs_ds = segs[:, :, :: T // T_p, :, :].long()  # [B, S, T_p, H, W]
        gt_onehot = F.one_hot(segs_ds, num_classes=self.decoder_num_classes)
        return gt_onehot.permute(0, 5, 1, 2, 3, 4).float()


# =========================================================
# Factory
# =========================================================
def parse_segmentation_module_name(module_name: str) -> Tuple[str, str]:
    """
    Returns:
        decoder_type: "unetr" or "vol_seg"
        pretrain_name: matching name in pretrain MODEL_REGISTRY
    """
    if module_name.startswith("seg_unetr_"):
        decoder_type = "unetr"
        suffix = module_name[len("seg_unetr_"):]
    elif module_name.startswith("seg_vol_seg_"):
        decoder_type = "vol_seg"
        suffix = module_name[len("seg_vol_seg_"):]
    else:
        raise ValueError(
            f"Unsupported module_name: {module_name}. "
            "Expected prefix 'seg_unetr_' or 'seg_vol_seg_'."
        )

    pretrain_name = f"pretrain_multi{suffix}"
    return decoder_type, pretrain_name


def create_segmentation_model(
    module_name: str,
    checkpoint_path: Optional[str] = None,
    num_classes: int = 6,
    img_size: int = 112,
    patch_size: int = 4,
    num_frames: int = 32,
    tubelet_size: int = 8,
    num_slices: int = 6,
    decoder_feature: int = 32,
    volume_head: Optional[nn.Module] = None,
    inner_dim: int = 128,
) -> nn.Module:
    """
    Factory for segmentation models.
    """
    from pretrain_stage.modeling_pretrain import MODEL_REGISTRY as PRETRAIN_REGISTRY

    decoder_type, pretrain_name = parse_segmentation_module_name(module_name)

    backbone_ctor = PRETRAIN_REGISTRY.get(pretrain_name)
    if backbone_ctor is None:
        raise ValueError(
            f"Backbone '{pretrain_name}' not found in PRETRAIN_REGISTRY. "
            f"Derived from module_name='{module_name}'."
        )

    pretrain_model = backbone_ctor()

    if hasattr(pretrain_model, "shared_encoder"):
        shared_encoder = pretrain_model.shared_encoder
    else:
        shared_encoder = pretrain_model

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        msg = shared_encoder.load_state_dict(state_dict, strict=False)
        print(f"[Segmentation] Loaded checkpoint from {checkpoint_path}")

    config = build_segmentation_config(
        num_classes=num_classes,
        img_size=img_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        num_slices=num_slices,
        decoder_feature=decoder_feature,
    )

    if decoder_type == "unetr":
        return SegMAEUNETR(
            config=config,
            shared_encoder=shared_encoder,
            volume_head=volume_head,
        )

    if decoder_type == "vol_seg":
        return SegMAEVol(
            config=config,
            shared_encoder=shared_encoder,
            volume_head=volume_head,
            inner_dim=inner_dim,
        )

    raise RuntimeError(f"Unexpected decoder_type: {decoder_type}")
