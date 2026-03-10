"""
VolSegDecoder — Lightweight volume-aware segmentation decoder.

Design:
  - Fuses encoder patch features (temporal dynamics) with VolumeHead rec_features
    (view-consistent anatomical prior) entirely in 2D space.
  - Avoids 3D convolutions on large volumes.
  - ~350K decoder params vs UNETR ~1.5M.
  - Output at tubelet temporal resolution T_p = T // tubelet_size.
    GT is downsampled to T_p during training; predictions are upsampled at eval.

Tensor shapes (config: S=9, T=32, H=W=112, tubelet=8, patch=4):
  enc_flat:     [B*S, T_p*N_sp, D]  = [9, 3136, 192]  (T_p=4, N_sp=784)
  rec_features: [B, S, N_sp, D]     = [1, 9, 784, 192]
  output:       [B, num_classes, S, T_p, H, W]
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _UpBlock(nn.Module):
    """ConvTranspose2d upsample ×2 followed by a residual refinement conv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """Initialize the _UpBlock.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act = nn.GELU()
        self.refine = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_ch, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm(self.up(x)))
        x = self.act(self.norm2(self.refine(x)))
        return x


class VolSegDecoder(nn.Module):
    """
    Lightweight segmentation decoder exploiting the 3D Volume Head representation.

    The decoder fuses two complementary feature sources:
      - ``enc_flat``:     per-slice spatio-temporal encoder patch tokens — captures
                         fine-grained local texture and temporal cardiac dynamics.
      - ``rec_features``: VolumeHead geometrically-reconstructed spatial features —
                         provides cross-view, anatomically consistent 3D priors.

    These are concatenated along the channel dimension, linearly projected, reshaped
    to 2D spatial grids, and progressively upsampled with lightweight transposed
    convolutions to full spatial resolution.

    Args:
        encoder_dim:  Dimension of encoder / VolumeHead output features (default 192).
        inner_dim:    Projection dimension after fusion (default 128).
        num_classes:  Number of segmentation classes (default 6).
        patch_size:   Spatial patch size used in the encoder (4 or 8).
    """

    def __init__(
        self,
        encoder_dim: int = 192,
        inner_dim: int = 128,
        num_classes: int = 6,
        patch_size: int = 4,
    ) -> None:
        super().__init__()
        if patch_size not in (4, 8):
            raise ValueError(f"patch_size must be 4 or 8, got {patch_size}")
        self.patch_size = patch_size

        # Fusion projection: cat([enc, vol]) → inner_dim
        self.fuse_proj = nn.Linear(encoder_dim * 2, inner_dim, bias=False)
        self.fuse_norm = nn.LayerNorm(inner_dim)
        self.fuse_act = nn.GELU()

        # Temporal smoothing: 1D depthwise conv along T_p to encourage consistency
        self.temp_conv = nn.Conv1d(
            inner_dim, inner_dim, kernel_size=3, padding=1, groups=inner_dim, bias=False
        )

        # 2D spatial upsampling: patch_size=4 needs 2 ×2 stages (28→56→112)
        #                        patch_size=8 needs 3 ×2 stages (14→28→56→112)
        n_ups = {4: 2, 8: 3}[patch_size]
        up_blocks = []
        ch = inner_dim
        for _ in range(n_ups):
            out_ch = max(ch // 2, 32)
            up_blocks.append(_UpBlock(ch, out_ch))
            ch = out_ch
        self.upsample = nn.ModuleList(up_blocks)

        # Segmentation head
        self.head = nn.Conv2d(ch, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fuse_proj.weight)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(
        self,
        enc_flat: torch.Tensor,
        rec_features: torch.Tensor,
        B: int,
        S: int,
        T_p: int,
        H_p: int,
        W_p: int,
    ) -> torch.Tensor:
        """Forward pass of the volume segmentation decoder.

        Args:
            enc_flat (torch.Tensor): Encoder output [B*S, T_p*N_sp, D].
            rec_features (torch.Tensor): Reconstructed features [B, S, N_sp, D].
            B (int): Batch size.
            S (int): Number of slices.
            T_p (int): Unrolled temporal patches.
            H_p (int): Unrolled spatial height patches.
            W_p (int): Unrolled spatial width patches.

        Returns:
            torch.Tensor: seg matching [B, num_classes, S, T_p, H, W]
        """
        D = enc_flat.shape[-1]
        N_sp = H_p * W_p

        # ── 1. Reshape encoder tokens to spatio-temporal grid ──────────────────
        # [B*S, T_p*N_sp, D]  →  [B, S, T_p, N_sp, D]
        enc_st = enc_flat.view(B, S, T_p, N_sp, D)

        # ── 2. Expand volume features over temporal dimension ──────────────────
        # [B, S, N_sp, D]  →  [B, S, T_p, N_sp, D]
        vol_feat = rec_features.unsqueeze(2).expand(-1, -1, T_p, -1, -1)

        # ── 3. Fuse: cat + project ──────────────────────────────────────────────
        # [B, S, T_p, N_sp, 2D]  →  [B, S, T_p, N_sp, inner_dim]
        fused = torch.cat([enc_st, vol_feat], dim=-1)
        fused = self.fuse_act(self.fuse_norm(self.fuse_proj(fused)))

        # ── 4. Temporal smoothing (depthwise conv along T_p) ───────────────────
        # [B, S, T_p, N_sp, C]  →  [B*S*N_sp, C, T_p]  →  apply  →  back
        C = fused.shape[-1]
        fused_t = fused.permute(0, 1, 3, 4, 2)  # [B, S, N_sp, C, T_p]
        fused_t = fused_t.reshape(B * S * N_sp, C, T_p)
        fused_t = self.temp_conv(fused_t)  # [B*S*N_sp, C, T_p]
        fused_t = fused_t.reshape(B, S, N_sp, C, T_p)
        fused = fused_t.permute(0, 1, 4, 2, 3)  # [B, S, T_p, N_sp, C]

        # ── 5. Reshape for 2D spatial CNN ──────────────────────────────────────
        # [B, S, T_p, N_sp, C]  →  [B*S*T_p, C, H_p, W_p]
        BST = B * S * T_p
        fused = fused.reshape(BST, N_sp, C)  # [BST, N_sp, C]
        fused = fused.permute(0, 2, 1)  # [BST, C, N_sp]
        fused = fused.reshape(BST, C, H_p, W_p)  # [BST, C, H_p, W_p]

        # ── 6. Progressive 2D spatial upsampling ───────────────────────────────
        x = fused
        for up_block in self.upsample:
            x = up_block(x)  # [BST, C', H', W']

        # ── 7. Segmentation head ───────────────────────────────────────────────
        seg = self.head(x)  # [BST, num_classes, H, W]

        # ── 8. Reshape to output format ────────────────────────────────────────
        H, W = seg.shape[2], seg.shape[3]
        seg = seg.view(B, S, T_p, -1, H, W)  # [B, S, T_p, C_cls, H, W]
        seg = seg.permute(0, 3, 1, 2, 4, 5).contiguous()  # [B, C_cls, S, T_p, H, W]
        return seg
