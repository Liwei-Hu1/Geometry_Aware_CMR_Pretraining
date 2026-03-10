from __future__ import annotations

from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

ModelConstructor = Callable[..., nn.Module]
MODEL_REGISTRY: Dict[str, ModelConstructor] = {}


def register_model(name: str) -> Callable[[ModelConstructor], ModelConstructor]:
    def decorator(fn: ModelConstructor) -> ModelConstructor:
        MODEL_REGISTRY[name] = fn
        return fn

    return decorator


__all__ = [
    "ModelConstructor",
    "MODEL_REGISTRY",
    "register_model",
    "pretrain_multivideomae_tiny_patch8_112",
    "pretrain_multivideomae_tiny_patch4_112",
]


def get_sinusoid_encoding_table(n_position: int, d_hid: int) -> torch.Tensor:
    def get_angle_vec(position: int) -> List[float]:
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.tensor(sinusoid_table, dtype=torch.float32, requires_grad=False).unsqueeze(0)


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> None:
    __call_trunc_normal_(tensor, mean=mean, std=std, a=0, b=std)


class DropPath(nn.Module):
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        qkv_bias = None
        if self.q_bias is not None and self.v_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        init_values: Optional[float] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        attn_head_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_head_dim=attn_head_dim,
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden, act_layer=act_layer, drop=drop)

        if init_values and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gamma_1 is None or self.gamma_2 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            attn_out = self.attn(self.norm1(x))
            mlp_out = self.mlp(self.norm2(x))
            x = x + self.drop_path(self.gamma_1 * attn_out)
            x = x + self.drop_path(self.gamma_2 * mlp_out)
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 112,
        tubelet_size: int = 4,
        img_patch_size: int = 8,
        in_chans: int = 3,
        embed_dim: int = 768,
        num_frames: int = 26,
    ) -> None:
        super().__init__()
        img_size = to_2tuple(img_size)
        img_patch_size = to_2tuple(img_patch_size)
        self.tubelet_size = int(tubelet_size)
        self.grid_size = (
            1,
            num_frames // self.tubelet_size,
            img_size[0] // img_patch_size[0],
            img_size[1] // img_patch_size[1],
        )
        self.num_patches = int(np.prod(self.grid_size))
        self.img_size = img_size
        self.img_patch_size = img_patch_size
        assert in_chans == 1, "Only supporting single-channel input"
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(self.tubelet_size, img_patch_size[0], img_patch_size[1]),
            stride=(self.tubelet_size, img_patch_size[0], img_patch_size[1]),
        )

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        B, S, T, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input size ({H}*{W}) mismatched with model ({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PretrainVisionTransformerEncoder(nn.Module):
    """Encoder used in the pretraining loop."""

    def __init__(
        self,
        img_size: int = 112,
        img_patch_size: int = 8,
        in_chans: int = 1,
        num_classes: int = 0,
        embed_dim: int = 192,
        num_frames: int = 32,
        depth: int = 9,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        init_values: Optional[float] = None,
        tubelet_size: int = 4,
        use_checkpoint: bool = False,
        use_learnable_pos_emb: bool = True,
        num_views: int = 9,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            tubelet_size=tubelet_size,
            img_patch_size=img_patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_frames=num_frames,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            trunc_normal_(self.pos_embed, std=0.02)
        else:
            pe = get_sinusoid_encoding_table(num_patches + 1, embed_dim)
            self.register_buffer("pos_embed", pe, persistent=False)

        self.view_embed = nn.Embedding(num_views, embed_dim)
        # Spatial coordinate projection: 9D (3D position + 6D orientation) -> embed_dim (per-slice)
        self.spatial_proj = nn.Sequential(
            nn.Linear(9, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Per-patch 3D coordinate projection: 3D (X, Y, Z) -> embed_dim
        self.patch_spatial_proj = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        trunc_normal_(self.view_embed.weight, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path_rate=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        return {"pos_embed", "cls_token"}

    def forward_features(self, x: torch.Tensor, mask: torch.Tensor, view_id: torch.Tensor,
                         spatial_coords: Optional[torch.Tensor] = None,
                         patch_coords_3d: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.patch_embed(x)
        N, L, D = x.shape

        # Per-patch 3D spatial embedding (preferred) or per-slice 9D broadcast
        if patch_coords_3d is not None:
            # patch_coords_3d: [B*S, N_spatial_patches, 3]
            # We need to expand to [B*S, N_total_patches, 3] by repeating across temporal tubelets
            # N_total = T_patches * H_patches * W_patches, N_spatial = H_patches * W_patches
            N_spatial = patch_coords_3d.shape[1]  # 784
            T_patches = L // N_spatial  # temporal patches (e.g., 4)
            # Repeat each spatial coord for all temporal positions: [B*S, T*H*W, 3]
            coords_expanded = patch_coords_3d.unsqueeze(1).expand(-1, T_patches, -1, -1)  # [B*S, T, N_sp, 3]
            
            coords_expanded = coords_expanded.reshape(N, L, 3)  # [B*S, N_total, 3]
            v_patch = self.patch_spatial_proj(coords_expanded.type_as(x))  # [B*S, N_total, D]
            x = x + self.pos_embed[:, 1:, :].type_as(x) + v_patch
            # CLS token: use mean of patch coords
            v_cls = v_patch.mean(dim=1, keepdim=True)  # [B*S, 1, D]
        elif spatial_coords is not None:
            # Fallback: per-slice spatial embedding (broadcast to all patches)
            v = self.spatial_proj(spatial_coords.to(x.device).type_as(x)).unsqueeze(1)  # [B*S, 1, D]
            x = x + self.pos_embed[:, 1:, :].type_as(x) + v
            v_cls = v
        else:
            v = self.view_embed(view_id.to(x.device).long()).type_as(x).unsqueeze(1)
            x = x + self.pos_embed[:, 1:, :].type_as(x) + v
            v_cls = v

        mask = mask.to(x.device).bool()
        
        # Robust handling for non-uniform mask ratios (e.g. Zero-Shot 100% masking)
        bool_vis_mask = ~mask
        num_vis = bool_vis_mask.sum(dim=1) # [N]
        max_vis = num_vis.max().item()
        
        if (num_vis == max_vis).all():
            vis = x[bool_vis_mask].reshape(N, max_vis, D)
        else:
            # Padding-based robust handling
            vis = torch.zeros(N, max_vis, D, device=x.device, dtype=x.dtype)
            for i in range(N):
                row_vis = x[i][bool_vis_mask[i]]
                if row_vis.size(0) > 0:
                    vis[i, :row_vis.size(0)] = row_vis

        cls = (self.cls_token + self.pos_embed[:, :1, :]).type_as(x) + v_cls
        cls = cls.expand(N, -1, -1)
        x_vis = torch.cat([cls, vis], dim=1)

        if self.use_checkpoint:
            for blk in self.blocks:
                x_vis = checkpoint.checkpoint(blk, x_vis)
        else:
            for blk in self.blocks:
                x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x: torch.Tensor, mask: torch.Tensor, view_id: torch.Tensor,
                spatial_coords: Optional[torch.Tensor] = None,
                         patch_coords_3d: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.forward_features(x, mask, view_id, spatial_coords, patch_coords_3d)
        x = self.head(x)
        return x


class PretrainVisionTransformerDecoder(nn.Module):
    """Decoder module used to reconstruct masked patches (shared decoder, 2 heads)."""

    def __init__(
        self,
        img_patch_size: int = 8,
        num_classes: int = 256,
        embed_dim: int = 96,
        depth: int = 3,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        init_values: Optional[float] = None,
        tubelet_size: int = 4,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == 1 * tubelet_size * img_patch_size ** 2
        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path_rate=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, return_token_num: int) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:]))
        else:
            x = self.head(self.norm(x))
        return x


class ThreeDVolumeHead(nn.Module):
    """
    Estimates an implicit 3D volume representation from multi-slice encoder features.
    
    Architecture:
    1. Patch → Volume: Cross-attention aggregates 2D patch features into 3D voxels
    2. Volume → Slice: F.grid_sample (trilinear interpolation) at per-patch physical
       3D coordinates, enforcing geometric structure in the volume.
    """
    def __init__(
        self,
        embed_dim: int = 192,
        volume_dim: int = 192,
        volume_size: tuple = (16, 16, 16),
        num_slices: int = 9,
        num_patches_per_slice: int = 196, # 14x14
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.volume_dim = volume_dim
        self.volume_size = volume_size
        self.num_slices = num_slices
        
        # 1. Volume Queries (The "Canvas")
        num_voxels = volume_size[0] * volume_size[1] * volume_size[2]
        self.volume_queries = nn.Parameter(torch.zeros(1, num_voxels, volume_dim))
        trunc_normal_(self.volume_queries, std=0.02)

        # 3D sinusoidal positional encoding for volume voxels (fixed, not learned)
        self.register_buffer(
            'volume_pos_embed',
            self._build_3d_sincos_pos(volume_size, volume_dim)
        )

        # 2. Patch-to-Volume Attention (The "Painter")
        self.patch_to_vol_attn = nn.MultiheadAttention(
            embed_dim=volume_dim,
            num_heads=4,
            batch_first=True
        )

        # 3. Geometric Volume-to-Slice Projection
        # Maps normalized patch 3D coordinates to grid coords in [-1, 1]
        # Fixed Scaling: Replace learnable MLP with fixed factor to solve "Cold Start"
        self.grid_scale_factor = 4.0
        
        # Adapter if encoder dim != volume dim
        if embed_dim != volume_dim:
            self.enc_to_vol = nn.Linear(embed_dim, volume_dim)
            self.vol_to_enc = nn.Linear(volume_dim, embed_dim)
        else:
            self.enc_to_vol = nn.Identity()
            self.vol_to_enc = nn.Identity()

        self.norm_vol = nn.LayerNorm(volume_dim)
        self.norm_slice = nn.LayerNorm(volume_dim)

    @staticmethod
    def _build_3d_sincos_pos(volume_size, dim):
        """Build 3D sinusoidal positional encoding. Returns [1, H*W*D, dim]."""
        H, W, D = volume_size
        assert dim % 6 == 0, f'volume_dim={dim} must be divisible by 6 for 3D sincos'
        dim_per_axis = dim // 3  # each axis gets dim/3 channels

        def sincos_1d(length, d):
            pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)  # [L, 1]
            omega = 1.0 / (10000 ** (torch.arange(0, d, 2, dtype=torch.float32) / d))  # [d/2]
            out = torch.zeros(length, d)
            out[:, 0::2] = torch.sin(pos * omega)
            out[:, 1::2] = torch.cos(pos * omega)
            return out  # [L, d]

        pe_h = sincos_1d(H, dim_per_axis)  # [H, d/3]
        pe_w = sincos_1d(W, dim_per_axis)  # [W, d/3]
        pe_d = sincos_1d(D, dim_per_axis)  # [D, d/3]

        # Broadcast to [H, W, D, dim]
        pe = torch.cat([
            pe_h[:, None, None, :].expand(H, W, D, -1),
            pe_w[None, :, None, :].expand(H, W, D, -1),
            pe_d[None, None, :, :].expand(H, W, D, -1),
        ], dim=-1)  # [H, W, D, dim]

        return pe.reshape(1, H * W * D, dim)

    def forward(self, enc_patch, patch_coords_3d, spatial_coords=None):
        """
        Args:
            enc_patch: [B*S, N_vis, D] - Visible patch tokens
            patch_coords_3d: [B, S, N_spatial_patches, 3] - Per-patch 3D physical coordinates
            spatial_coords: [B, S, 9] - (unused, kept for signature compat)
    
        Returns:
            volume: [B, C, H, W, D] - 3D feature volume
            rec_features: [B, S, N_spatial_patches, D] - Reconstructed features per patch
        """
        BS, N_vis, D = enc_patch.shape
        B = BS // self.num_slices
        S = self.num_slices
        N_sp = patch_coords_3d.shape[2]  # spatial patches per slice (e.g., 784)
        
        # 1. Prepare Keys/Values: All visible patches in the batch
        patch_feats = enc_patch.reshape(B, S*N_vis, D)
        patch_feats = self.enc_to_vol(patch_feats)
        
        # 2. Build Volume (Cross-Attention) — same as before
        queries = (self.volume_queries + self.volume_pos_embed).expand(B, -1, -1)
        volume_flat, _ = self.patch_to_vol_attn(
            query=queries,
            key=patch_feats,
            value=patch_feats
        )
        volume_flat = self.norm_vol(volume_flat)  # [B, N_voxels, C]

        # Reshape to 3D layout
        # Dynamically calculate D_vol to handle checkpoint mismatch (e.g. 16 vs 32)
        N_voxels = volume_flat.shape[1]
        D_vol = int(round(N_voxels ** (1/3)))
        if D_vol ** 3 != N_voxels:
             # Fallback to configured if not cubic (unlikely for now)
             H, W, D_vol = self.volume_size
        else:
             H = W = D_vol
             
        volume = volume_flat.permute(0, 2, 1).reshape(B, self.volume_dim, H, W, D_vol)

        # 3. Geometric Projection: sample from volume at per-patch 3D locations
        # Map normalized coords to grid coords in [-1, 1]
        coords_flat = patch_coords_3d.reshape(B, S * N_sp, 3)  # [B, S*N_sp, 3]
        
        # Fixed Scaling + Clamp
        grid = (coords_flat / self.grid_scale_factor).clamp(-1, 1)
        
        # F.grid_sample expects grid shape [B, D_out, H_out, W_out, 3] for 5D input
        # We treat each sample point as a separate "voxel": [B, S*N_sp, 1, 1, 3]
        grid_5d = grid.unsqueeze(2).unsqueeze(3)  # [B, S*N_sp, 1, 1, 3]
        
        # Sample: volume [B, C, H, W, D] -> sampled [B, C, S*N_sp, 1, 1]
        sampled = F.grid_sample(
            volume, grid_5d,
            mode='bilinear',
            align_corners=True,
            padding_mode='zeros'
        )
        # sampled: [B, C, S*N_sp, 1, 1] -> [B, S*N_sp, C]
        rec_features_flat = sampled.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        rec_features_flat = self.norm_slice(rec_features_flat)
        
        # Back to encoder dim and reshape
        rec_features = self.vol_to_enc(rec_features_flat)
        rec_features = rec_features.reshape(B, S, N_sp, D)
        
        return volume, rec_features

    def forward_partial(self, enc_patch, active_slices: int):
        """
        Forward pass for partial slices (e.g. only SAX).
        Args:
            enc_patch: [B*active_slices, N_vis, D]
            active_slices: number of slices present in input
        """
        BS, N_vis, D = enc_patch.shape
        B = BS // active_slices
        S_total = self.num_slices
        
        # 1. Prepare Keys/Values (Visible patches from active slices)
        patch_feats = enc_patch.reshape(B, active_slices*N_vis, D)
        patch_feats = self.enc_to_vol(patch_feats)
        
        # 2. Build Volume (Cross-Attention)
        queries = self.volume_queries.expand(B, -1, -1)
        # We only attend to available patches
        volume_flat, _ = self.patch_to_vol_attn(
            query=queries,
            key=patch_feats,
            value=patch_feats
        )
        volume_flat = self.norm_vol(volume_flat)

        # 3. Project back to Active Slice Grids
        # We only use grid queries for the active slices (assuming 0..active_slices)
        # slice_grid_queries: [1, S_total, N_patches, D]
        active_grid_queries = self.slice_grid_queries[:, :active_slices, :, :]
        
        slice_queries_flat = active_grid_queries.reshape(1, active_slices * active_grid_queries.shape[2], -1).expand(B, -1, -1)
        
        rec_features_flat, _ = self.vol_to_slice_attn(
            query=slice_queries_flat,
            key=volume_flat,
            value=volume_flat
        )
        rec_features_flat = self.norm_slice(rec_features_flat)
        
        # Reshape back to [B, active_slices, N_patches, D]
        rec_features = self.vol_to_enc(rec_features_flat)
        rec_features = rec_features.reshape(B, active_slices, -1, D)
        
        # We don't return volume here as we consume rec_features
        return None, rec_features

    def predict_features_from_volume(self, volume):
        """
        Args:
            volume: [B, C, H, W, D]
        Returns:
            rec_features: [B, S, N_patches, D]
        """
        B, C, H, W, D_vol = volume.shape
        S = self.num_slices
        
        # Flatten volume back to [B, N_voxels, C]
        # Previous forward: volume_flat.permute(0, 2, 1).reshape(B, C, H, W, D)
        # So reverse: 
        volume_flat = volume.reshape(B, C, -1).permute(0, 2, 1) # [B, N_voxels, C]
        
        # 3. Project back to 2D Slice Grids (Cross-Attention)
        # Flatten slice queries: [B, S*N_patches, C]
        slice_queries_flat = self.slice_grid_queries.reshape(1, S * self.slice_grid_queries.shape[2], -1).expand(B, -1, -1)
        
        rec_features_flat, _ = self.vol_to_slice_attn(
            query=slice_queries_flat,
            key=volume_flat,
            value=volume_flat
        )
        rec_features_flat = self.norm_slice(rec_features_flat)
        
        # Reshape back to [B, S, N_patches, D]
        rec_features = self.vol_to_enc(rec_features_flat)
        rec_features = rec_features.reshape(B, S, -1, D_vol)
        
        return rec_features


class PretrainVisionTransformer(nn.Module):
    """Combined encoder-decoder architecture used for pretraining (vectorized over S)."""

    def __init__(
        self,
        img_size: int = 112,
        tubelet_size: int = 4,
        img_patch_size: int = 8,
        num_frames: int = 32,
        encoder_in_chans: int = 1,
        encoder_embed_dim: int = 192,
        encoder_depth: int = 9,
        encoder_num_heads: int = 3,
        decoder_num_classes: int = 256,
        decoder_embed_dim: int = 96,
        decoder_depth: int = 3,
        decoder_num_heads: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        init_values: float = 0.0,
        use_learnable_pos_emb: bool = False,
        use_checkpoint: bool = False,
        recon_types: str = "sax+lax",
        sax_slices: int = 6,
        lax_slices: int = 3,
    ) -> None:
        super().__init__()
        self.num_views = sax_slices + lax_slices
        self.num_types = 2 if lax_slices > 0 else 1
        self.decoder_num_classes = decoder_num_classes
        self.sax_slices = sax_slices
        self.lax_slices = lax_slices
        self.encoder_embed_dim = encoder_embed_dim

        self.shared_encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            img_patch_size=img_patch_size,
            in_chans=encoder_in_chans,
            num_classes=0,
            embed_dim=encoder_embed_dim,
            num_frames=num_frames,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            use_learnable_pos_emb=use_learnable_pos_emb,
            num_views=self.num_views,
        )

        num_patches = self.shared_encoder.patch_embed.num_patches
        
        # 3D Volume Head for learning implicit 3D structure
        # Replaces the explicit decoders
        self.volume_head = ThreeDVolumeHead(
            embed_dim=encoder_embed_dim,
            volume_dim=encoder_embed_dim,
            volume_size=(16, 16, 16),
            num_slices=sax_slices + lax_slices,
            num_patches_per_slice=num_patches
        )
        
        # Decoder (to reconstruct details from 3D features)
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False
        )
        dec_pos_embed = get_sinusoid_encoding_table(num_patches, decoder_embed_dim)
        self.decoder_pos_embed.data.copy_(dec_pos_embed)

        self.decoder = PretrainVisionTransformerDecoder(
            img_patch_size=img_patch_size,
            num_classes=tubelet_size * img_patch_size * img_patch_size * encoder_in_chans,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, # Reuse drop_path_rate from encoder? Or 0.0?
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
        )


    def _build_ids(self, B: int, S: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.sax_slices + self.lax_slices != S:
            raise ValueError(f"S={S} must equal sax_slices+lax_slices={self.sax_slices}+{self.lax_slices}")

        view_type_s = torch.zeros(S, device=device, dtype=torch.long)
        view_type_s[self.sax_slices:] = 1

        if self.num_views < S:
            raise ValueError(f"num_views={self.num_views} must be >= S={S} for per-slice view_id")
        view_id_s = torch.arange(S, device=device, dtype=torch.long)

        view_type = view_type_s.unsqueeze(0).expand(B, S).reshape(B * S)
        view_id = view_id_s.unsqueeze(0).expand(B, S).reshape(B * S)
        return view_type, view_id


    def forward(self, videos: torch.Tensor, masks: List[torch.Tensor],
                spatial_coords: Optional[torch.Tensor] = None,
                patch_coords_3d: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            videos: [B, S, T, H, W]
            masks: List of [B, T*H*W*tubelet]
            spatial_coords: [B, S, 9] (optional)
            patch_coords_3d: [B, S, N_patches, 3] (optional, precision coords)
        """
        B, S, T, H, W = videos.shape
        device = videos.device

        # Flatten inputs for shared encoder
        # videos: [B*S, T, H, W]
        x = videos.reshape(B * S, T, H, W).unsqueeze(1).contiguous()
        
        # Flatten masks: [B*S, N_patches]
        # masks is a list of S tensors, each [B, N_patches]
        mask = torch.stack(masks, dim=1).reshape(B * S, -1).to(device).bool()

        # View IDs for embedding (0..S-1 per batch)
        # [B, S] -> [B*S]
        view_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1).reshape(-1)
        
        spatial_coords_flat = None
        if spatial_coords is not None:
             spatial_coords_flat = spatial_coords.reshape(B * S, -1).to(device)

        patch_coords_3d_flat = None
        if patch_coords_3d is not None:
             # [B, S, N, 3] -> [B*S, N, 3]
             patch_coords_3d_flat = patch_coords_3d.reshape(B * S, -1, 3).to(device)

        # 1. Encode (Shared Encoder)
        # Returns [B*S, N_vis, D] (only visible patches) + CLS token
        x_vis = self.shared_encoder(x, mask, view_id=view_ids,
                                    spatial_coords=spatial_coords_flat,
                                    patch_coords_3d=patch_coords_3d_flat)
        
        enc_cls = x_vis[:, :1, :]
        enc_patch = x_vis[:, 1:, :] # [B*S, N_vis, D]

        # 2. Volume Head (Cross-Attention + Geometric Projection)
        # Constructs 3D volume from visible patches
        volume, rec_from_vol = self.volume_head(enc_patch, patch_coords_3d)

        # rec_from_vol: [B, S, N_sp, volume_dim] (only spatial!)
        # Decoder expects: [B*S, N_total, decoder_dim]
        # We need to expand spatial-only features to temporal tubelets
        B_dec, S_dec, N_sp, C_vol = rec_from_vol.shape
        # Total patches = T * N_sp (but T here is tubelets)
        # We can infer T_patches from self.decoder_pos_embed.shape[1] // N_sp
        N_total = self.decoder_pos_embed.shape[1]
        T_patches = N_total // N_sp
        
        # Expand: [B, S, N_sp, C] -> [B, S, 1, N_sp, C] -> [B, S, T_p, N_sp, C]
        rec_spatial = rec_from_vol.unsqueeze(2).expand(-1, -1, T_patches, -1, -1)
        # Reshape to [B, S, N_total, C]
        rec_expanded = rec_spatial.reshape(B_dec, S_dec, N_total, C_vol)

        # 3. Decode
        # Reproject volume features to decoder dim
        x_rec = self.encoder_to_decoder(rec_expanded) # [B, S, N, D_dec]
        
        # Reshape to expected decoder input [B*S, N, D_dec]
        x_rec = x_rec.reshape(B*S, -1, self.decoder.embed_dim)
        
        # Add decoder pos embed
        x_rec = x_rec + self.decoder_pos_embed.type_as(x_rec)

        # 4. Final Reconstruction
        # [B*S, N, D_dec] -> [B*S, N, pixel_dim]
        logits = self.decoder(x_rec, return_token_num=0)
        
        # 3D Volume Estimation & Geometric Reconstruction
        vol_patch_coords = patch_coords_3d.to(device) if patch_coords_3d is not None else None
        vol_spatial = spatial_coords.to(device) if spatial_coords is not None else None
        volume_3d, rec_features_spatial = self.volume_head(enc_patch, patch_coords_3d=vol_patch_coords, spatial_coords=vol_spatial)
        # rec_features_spatial: [B, S, N_sp, D] where N_sp = spatial patches (784)
        
        intermidiate_value = rec_features_spatial.clone()
        
        # Expand from spatial-only to spatio-temporal patches
        # N_total = T_patches * N_sp, each spatial patch maps to T_patches temporal positions
        N_sp = rec_features_spatial.shape[2]  # 784
        N_total = mask.shape[1]  # 3136 = 4 * 784
        T_patches = N_total // N_sp
        
        # Repeat spatial features across temporal dimension: [B, S, N_sp, D] -> [B, S, T*N_sp, D]
        rec_features = rec_features_spatial.unsqueeze(3).expand(-1, -1, -1, T_patches, -1)  # [B, S, N_sp, T, D]
        rec_features = rec_features.reshape(B, S, N_total, -1)  # [B, S, T*N_sp, D] — but need (t, h, w) ordering
        
        # Actually patch ordering is (t, h, w): patch[k] = t*(H_p*W_p) + h*W_p + w
        # So we need: for each (h,w), repeat across all t positions
        # rec_features_spatial is [B, S, H_p*W_p, D], need to expand to [B, S, T_p*H_p*W_p, D]
        # with patch ordering (t=0,h,w), (t=1,h,w), ...
        rec_features = rec_features_spatial.unsqueeze(2).expand(-1, -1, T_patches, -1, -1)  # [B, S, T, N_sp, D]
        rec_features = rec_features.reshape(B, S, N_total, rec_features_spatial.shape[-1])  # [B, S, T*N_sp, D]
        
        # SKIP CONNECTION: Replace Visible Tokens with Encoder Features
        # rec_features: [B, S, N, D] -> [B*S, N, D]
        rec_features = rec_features.reshape(B*S, -1, self.volume_head.embed_dim)
        
        # mask is [B*S, N] (True=Masked)
        # We want to replace ~mask (Visible) with enc_patch
        # enc_patch: [B*S, N_vis, D]
        visible_mask = ~mask
        # Robust assignment for non-uniform mask ratios (padding handled row-by-row)
        for i in range(B * S):
            num_v = visible_mask[i].sum()
            if num_v > 0:
                rec_features[i, visible_mask[i]] = enc_patch[i, :num_v]
        
        # Project to decoder dim
        rec_features = self.encoder_to_decoder(rec_features) # [B*S, N, D_dec]
        
        # Add Pos Embed
        # pos_embed: [1, N, D_dec] -> expand to [B*S, N, D_dec]
        # Note: num_patches in pos_embed matches N here because we reconstruct full N patches
        pos_embed = self.decoder_pos_embed.type_as(rec_features).to(rec_features.device).clone().detach()
        rec_features = rec_features + pos_embed
        
        # Apply Transformer Decoder
        decoder_pred = self.decoder(rec_features, return_token_num=0) # [B*S, N, Pixels]
        
        # Separate SAX and LAX for compatibility with engine (though mostly we use all)
        C = decoder_pred.shape[-1]
        
        # Reshape to [B, S, N_patches, C]
        decoder_pred = decoder_pred.reshape(B, S, -1, C)
        
        sax_slice_idx = self.sax_slices
        pred_sax = decoder_pred[:, :sax_slice_idx, :, :]
        pred_lax = decoder_pred[:, sax_slice_idx:, :, :]

        # Reshape enc_cls to [B, S, D] for visualization
        enc_cls_out = enc_cls.reshape(B, S, -1)

        return {
            "sax": pred_sax,
            "lax": pred_lax,
            "full_recon": decoder_pred,
            "volume_3d": volume_3d,
            "enc_cls": enc_cls_out,
            "intermidiate_value": intermidiate_value, 
            "enc_patch": enc_patch.reshape(B, S, -1, self.shared_encoder.embed_dim),
        }

    def decode_volume(self, volume_3d):
        """
        Generate full reconstruction from latent volume.
        Args:
            volume_3d: [B, C, H, W, D]
        """
        B, C, H, W, D = volume_3d.shape
        S = self.num_views
        
        # 1. Project Volume -> Features
        rec_features = self.volume_head.predict_features_from_volume(volume_3d) # [B, S, N, D]
        rec_features = rec_features.reshape(B*S, -1, self.volume_head.embed_dim)
        
        # 2. Project to Decoder Dim
        rec_features = self.encoder_to_decoder(rec_features)
        
        # 3. Add Pos Embed
        pos_embed = self.decoder_pos_embed.type_as(rec_features).to(rec_features.device).clone().detach()
        rec_features = rec_features + pos_embed
        
        # 4. Decoder
        decoder_pred = self.decoder(rec_features, return_token_num=0) # [B*S, N, Pixels]
        
        # 5. Reshape and Split Views
        C_out = decoder_pred.shape[-1]
        decoder_pred = decoder_pred.reshape(B, S, -1, C_out)
        
        sax_slice_idx = self.sax_slices
        pred_sax = decoder_pred[:, :sax_slice_idx, :, :]
        pred_lax = decoder_pred[:, sax_slice_idx:, :, :]
        
        return {
            "sax": pred_sax,
            "lax": pred_lax,
            "full_recon": decoder_pred
        }


@register_model("pretrain_multivideomae_tiny_patch4_112")
def pretrain_multivideomae_tiny_patch4_112(pretrained: bool = False, **kwargs: object) -> PretrainVisionTransformer:
    model = PretrainVisionTransformer(
        img_size=112,
        sax_slices=6,
        tubelet_size=8,
        img_patch_size=4,
        num_frames=32,
        encoder_in_chans=1,
        encoder_embed_dim=192,
        encoder_depth=9,
        encoder_num_heads=3,
        decoder_num_classes=128,
        decoder_embed_dim=96,
        decoder_depth=3,
        decoder_num_heads=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    if pretrained and "init_ckpt" in kwargs:
        checkpoint_data = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint_data["model"])
    return model


@register_model("pretrain_multivideomae_tiny_patch8_112")
def pretrain_multivideomae_tiny_patch8_112(pretrained: bool = False, **kwargs: object) -> PretrainVisionTransformer:
    model = PretrainVisionTransformer(
        img_size=112,
        tubelet_size=8,
        img_patch_size=8,
        num_frames=32,
        encoder_in_chans=1,
        encoder_embed_dim=192,
        encoder_depth=9,
        encoder_num_heads=3,
        decoder_num_classes=256,
        decoder_embed_dim=96,
        decoder_depth=3,
        decoder_num_heads=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    if pretrained and "init_ckpt" in kwargs:
        checkpoint_data = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint_data["model"])
    return model

@register_model("pretrain_multivideomae_small_patch8_112")
def pretrain_multivideomae_small_patch8_112(pretrained: bool = False, **kwargs: object) -> PretrainVisionTransformer:
    model = PretrainVisionTransformer(
        img_size=112,
        tubelet_size=4,
        img_patch_size=8,
        num_frames=32,
        encoder_in_chans=1,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        decoder_num_classes=256,
        decoder_embed_dim=192,
        decoder_depth=3,
        decoder_num_heads=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    if pretrained and "init_ckpt" in kwargs:
        checkpoint_data = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint_data["model"])
    return model