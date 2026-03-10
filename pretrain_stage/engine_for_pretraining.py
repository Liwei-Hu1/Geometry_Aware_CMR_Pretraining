import logging
import math
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from numpy import ndarray as NDArray
from torch import Tensor
from torch.optim import Optimizer

from utils.misc import (
    MetricLogger,
    NativeScalerWithGradNormCount,
    SmoothedValue,
    TensorboardLogger,
)

logger = logging.getLogger(__name__)


NDArrayLike = NDArray


def expand_patch_mask_to_pixel_mask(
    mask_bn: torch.Tensor, patch_size: Tuple[int, int, int], window_size: Tuple[int, int, int]
) -> torch.Tensor:
    """Expand a patch-level boolean mask to a pixel-level boolean mask.

    Args:
        mask_bn (torch.Tensor): Binary mask for patches, shape [B, L], where True=masked.
        patch_size (Tuple[int, int, int]): Dimensions of each patch (time, height, width).
        window_size (Tuple[int, int, int]): Dimensions of the window in number of patches.

    Returns:
        torch.Tensor: Pixel-level mask of shape [B, T, H, W], where True=masked pixels.
    """
    p0, p1, p2 = patch_size  # (tubelet, ph, pw)
    t, h, w = window_size  # (T', H', W')
    B, L = mask_bn.shape
    assert L == t * h * w, (L, t * h * w)

    m = mask_bn.view(B, t, h, w)  # [B, T', H', W']
    m = m.repeat_interleave(p0, dim=1)  # -> [B, T, H', W']
    m = m.repeat_interleave(p1, dim=2)  # -> [B, T, H,  W']
    m = m.repeat_interleave(p2, dim=3)  # -> [B, T, H,  W]
    return m


def masked_only_psnr(
    x: torch.Tensor,
    y: torch.Tensor,
    masks_list: List[torch.Tensor],
    patch_size: Tuple[int, int, int],
    window_size: Tuple[int, int, int],
    max_val: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute PSNR scalar metric only on masked pixels.

    Args:
        x (torch.Tensor): Predicted tensor, shape [B, S, T, H, W].
        y (torch.Tensor): Target tensor, shape [B, S, T, H, W].
        masks_list (List[torch.Tensor]): List of boolean masks length S, each [B, L]. (True=masked).
        patch_size (Tuple[int, int, int]): Spatiotemporal patch size (pt, ph, pw).
        window_size (Tuple[int, int, int]): Array dimensions in terms of patches.
        max_val (float, optional): Maximum possible pixel value. Defaults to 1.0.
        eps (float, optional): Epsilon to prevent division by zero. Defaults to 1e-12.

    Returns:
        torch.Tensor: Scalar PSNR over masked pixels only.
    """
    assert x.shape == y.shape
    B, S, T, H, W = x.shape
    device = x.device

    mse_views = []
    for s in range(S):
        m_bn = masks_list[s].to(device).bool()  # [B,L]
        m_pix = expand_patch_mask_to_pixel_mask(
            m_bn, patch_size, window_size
        )  # [B,T,H,W]
        w = m_pix.float()

        sum_w = w.sum()
        if sum_w > 0:
            diff2 = (x[:, s] - y[:, s]).pow(2)  # [B,T,H,W]
            mse = (diff2 * w).sum() / sum_w  # 只在 masked 像素上
            mse_views.append(mse)

    if len(mse_views) == 0:
        return torch.tensor(0.0, device=device)

    mse_all = torch.stack(mse_views).mean()
    psnr = 10 * torch.log10((max_val**2) / (mse_all + eps))
    return psnr


def masked_only_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    masks_list: List[torch.Tensor],
    patch_size: Tuple[int, int, int],
    window_size: Tuple[int, int, int],
    data_range: float = 1.0,
    C1: float = 0.01**2,
    C2: float = 0.03**2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute structural similarity index metric (SSIM) exclusively on masked regions.

    Args:
        x (torch.Tensor): Predicted tensor, shape [B, S, T, H, W] in [0, 1] range.
        y (torch.Tensor): Output tensor, shape [B, S, T, H, W] in [0, 1] range.
        masks_list (List[torch.Tensor]): List of bool masks, length S, each [B, L].
        patch_size (Tuple[int, int, int]): Spatiotemporal patch size.
        window_size (Tuple[int, int, int]): Sequence dimensions in terms of patches.
        data_range (float, optional): Dynamic range of the images. Defaults to 1.0.
        C1 (float, optional): SSIM stability constant 1. Defaults to 0.01**2.
        C2 (float, optional): SSIM stability constant 2. Defaults to 0.03**2.
        eps (float, optional): Small constant to avoid div by zero. Defaults to 1e-8.

    Returns:
        torch.Tensor: scalar SSIM over masked pixels only.
    """
    x = x.float() / data_range
    y = y.float() / data_range
    B, S, T, H, W = x.shape
    device = x.device

    ssim_views = []
    for s in range(S):
        m_bn = masks_list[s].to(device).bool()  # [B,L]
        m_pix = expand_patch_mask_to_pixel_mask(
            m_bn, patch_size, window_size
        )  # [B,T,H,W]
        w = m_pix.float()

        if w.sum() == 0:
            continue

        xs = x[:, s]  # [B,T,H,W]
        ys = y[:, s]

        # 每帧的 masked 像素数: [B,T,1,1]
        w_sum = w.sum(dim=(2, 3), keepdim=True).clamp_min(eps)

        mu_x = (xs * w).sum(dim=(2, 3), keepdim=True) / w_sum
        mu_y = (ys * w).sum(dim=(2, 3), keepdim=True) / w_sum

        var_x = ((xs - mu_x).pow(2) * w).sum(dim=(2, 3), keepdim=True) / w_sum
        var_y = ((ys - mu_y).pow(2) * w).sum(dim=(2, 3), keepdim=True) / w_sum
        cov_xy = (((xs - mu_x) * (ys - mu_y)) * w).sum(dim=(2, 3), keepdim=True) / w_sum

        ssim_n = (2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)
        ssim_d = (mu_x.pow(2) + mu_y.pow(2) + C1) * (var_x + var_y + C2)
        ssim_frame = ssim_n / (ssim_d + eps)  # [B,T,1,1]

        ssim_views.append(ssim_frame.mean())

    if len(ssim_views) == 0:
        return torch.tensor(0.0, device=device)

    return torch.stack(ssim_views).mean()


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScalerWithGradNormCount,
    max_norm: Optional[float] = None,
    log_writer: Optional[TensorboardLogger] = None,
    start_steps: int = 0,
    lr_schedule_values: Optional[List[float]] = None,
    wd_schedule_values: Optional[List[float]] = None,
    patch_size: Tuple[int, int, int] = (16, 16, 16),
    normalize_target: bool = True,
) -> Dict[str, float]:
    """Execute one training epoch for pre-training models.

    Args:
        model (torch.nn.Module): The model to train.
        data_loader (Iterable): Data loader for providing batches.
        optimizer (Optimizer): Model optimizer.
        device (torch.device): Compute device for inputs and targets.
        epoch (int): Current epoch number.
        loss_scaler (NativeScalerWithGradNormCount): Handler for scaling and backward pass.
        max_norm (Optional[float], optional): Max grad norm for clipping. Defaults to None.
        log_writer (Optional[TensorboardLogger], optional): Optional Tensorboard writer. Defaults to None.
        start_steps (int, optional): Initial global step count. Defaults to 0.
        lr_schedule_values (Optional[List[float]], optional): Scheduled learning rates. Defaults to None.
        wd_schedule_values (Optional[List[float]], optional): Scheduled weight decay values. Defaults to None.
        patch_size (Tuple[int, int, int], optional): Size of patches. Defaults to (16, 16, 16).
        normalize_target (bool, optional): Unused legacy norm param. Defaults to True.

    Returns:
        Dict[str, float]: Aggregated metrics over the epoch.
    """
    model.train()
    metric_logger = MetricLogger(delimiter="")
    metric_logger.add_meter("lr", SmoothedValue(fmt="{value:.6f}"))
    metric_logger.add_meter("min_lr", SmoothedValue(fmt="{value:.6f}"))

    header = f"Epoch: [{epoch}]"
    print_freq = 10
    loss_func = nn.MSELoss()

    for step, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        iteration = start_steps + step
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for group in optimizer.param_groups:
                if lr_schedule_values is not None:
                    group["lr"] = lr_schedule_values[iteration] * group["lr_scale"]
                if wd_schedule_values is not None and group["weight_decay"] > 0:
                    group["weight_decay"] = wd_schedule_values[iteration]

        videos, masks, spatial_coords, patch_coords_3d, _ = batch
        videos = videos.to(device, non_blocking=True)
        spatial_coords = spatial_coords.to(device, non_blocking=True)  # [B, S, 9]
        patch_coords_3d = patch_coords_3d.to(device, non_blocking=True)

        # --- Hybrid Cross-View Masking Strategy ---
        B, S, T, H, W = videos.shape
        window_size = (T // patch_size[0], H // patch_size[1], W // patch_size[2])
        bool_masked_pos = [
            mask.to(device, non_blocking=True).flatten(1).to(torch.bool)
            for mask in masks
        ]

        # [New] View-Drop Logic: Always drop 2 entire views per sample
        total_dropped_views = 0
        is_dropped = torch.zeros(B, S, dtype=torch.bool, device=device)
        for b in range(B):
            # Randomly pick 2 distinct views to drop (100% mask)
            v_indices = torch.randperm(S, device=device)[:2]
            for v_idx in v_indices:
                v_idx = v_idx.item()
                bool_masked_pos[v_idx][b, :] = True
                is_dropped[b, v_idx] = True
                total_dropped_views += 1

        # construct labels for reconstruction loss
        with torch.no_grad():
            videos_patches = rearrange(
                videos,
                "b s (t p0) (h p1) (w p2) -> b s (t h w) (p0 p1 p2)",
                p0=patch_size[0],
                p1=patch_size[1],
                p2=patch_size[2],
            )

            B, _, _, C = videos_patches.shape

            labels = [
                videos_patches[:, idx, ...][mask].reshape(B, -1, C)
                for idx, mask in enumerate(bool_masked_pos)
            ]

            videos_patches_target = videos_patches

        outputs = model(
            videos,
            bool_masked_pos,
            spatial_coords=spatial_coords,
            patch_coords_3d=patch_coords_3d,
        )

        n_lax = 3
        assert S >= n_lax, f"S={S} must be >= {n_lax}"

        sax_idx = torch.arange(0, S - n_lax, device=videos.device)
        lax_idx = torch.arange(S - n_lax, S, device=videos.device)

        pred_sax = outputs["sax"]  # (B, S-3, ...)
        pred_lax = outputs["lax"]  # (B, 3, ...)

        pred_all = torch.cat([pred_sax, pred_lax], dim=1)
        target_all = videos_patches_target

        # Compute MSE per patch
        loss_pixel = (pred_all - target_all) ** 2
        loss_pixel = loss_pixel.mean(dim=-1)  # [B, S, N]

        # Apply mask: Only compute loss on MASKED patches
        # bool_masked_pos is list of [B, N], stack to [B, S, N]
        mask_flat = torch.stack(bool_masked_pos, dim=1)

        loss_recon = (loss_pixel * mask_flat).sum() / (mask_flat.sum() + 1e-6)

        pred_all_denorm = pred_all

        single_view_outputs = [pred_all_denorm[:, s, ...] for s in range(S)]

        single_view_recon = reconstruct_video(
            videos_patches,
            single_view_outputs,
            bool_masked_pos,
            patch_size,
            window_size,
        )

        recon_videos_fp32 = single_view_recon.float()

        # Differentiate metrics for VISIBLE vs DROPPED views
        vis_mask_list = [
            bool_masked_pos[s] & (~is_dropped[:, s].unsqueeze(-1)) for s in range(S)
        ]
        drop_mask_list = [
            bool_masked_pos[s] & (is_dropped[:, s].unsqueeze(-1)) for s in range(S)
        ]

        vis_v_psnr = masked_only_psnr(
            recon_videos_fp32,
            videos.float(),
            vis_mask_list,
            patch_size=patch_size,
            window_size=window_size,
            max_val=1.0,
        )
        vis_v_ssim = masked_only_ssim(
            recon_videos_fp32,
            videos.float(),
            vis_mask_list,
            patch_size=patch_size,
            window_size=window_size,
            data_range=1.0,
        )

        mask_v_psnr = masked_only_psnr(
            recon_videos_fp32,
            videos.float(),
            drop_mask_list,
            patch_size=patch_size,
            window_size=window_size,
            max_val=1.0,
        )
        mask_v_ssim = masked_only_ssim(
            recon_videos_fp32,
            videos.float(),
            drop_mask_list,
            patch_size=patch_size,
            window_size=window_size,
            data_range=1.0,
        )

        loss = loss_recon

        if not math.isfinite(loss.item()):
            logger.error("Loss is %f, stopping training", loss.item())
            raise RuntimeError("Non-finite loss")

        optimizer.zero_grad()
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        grad_norm = loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )
        loss_scale_value = loss_scaler.state_dict()["scale"]
        torch.cuda.synchronize()

        loss_value = loss.item()
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_recon=loss_recon.item())
        metric_logger.update(loss_scale=loss_scale_value)
        metric_logger.update(vis_v_psnr=vis_v_psnr)
        metric_logger.update(vis_v_ssim=vis_v_ssim)
        if total_dropped_views > 0:
            metric_logger.update(mask_v_psnr=mask_v_psnr)
            metric_logger.update(mask_v_ssim=mask_v_ssim)
        metric_logger.update(mask_count=total_dropped_views)

        lr_values = [group["lr"] for group in optimizer.param_groups]
        metric_logger.update(lr=max(lr_values))
        metric_logger.update(min_lr=min(lr_values))

        wd_values = [
            group["weight_decay"]
            for group in optimizer.param_groups
            if group["weight_decay"] > 0
        ]
        metric_logger.update(weight_decay=wd_values[-1] if wd_values else None)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_recon=loss_recon.item(), head="loss")
            log_writer.update(vis_v_sism=vis_v_ssim, head="opt")
            log_writer.update(vis_v_psnr=vis_v_psnr, head="opt")
            if total_dropped_views > 0:
                log_writer.update(mask_v_psnr=mask_v_psnr, head="opt")
                log_writer.update(mask_v_ssim=mask_v_ssim, head="opt")
            log_writer.update(mask_count=float(total_dropped_views), head="opt")

            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max(lr_values), head="opt")
            log_writer.update(min_lr=min(lr_values), head="opt")
            if wd_values:
                log_writer.update(weight_decay=wd_values[-1], head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: %s", metric_logger)
    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}


def reconstruct_video(
    combined_videos_patch: torch.Tensor,
    single_view_outputs: List[torch.Tensor],
    bool_masked_pos: List[torch.Tensor],
    patch_size: Tuple[int, int, int],
    window_size: Tuple[int, int, int],
) -> torch.Tensor:
    """Reconstruct sequence of frames replacing masked inputs with predictions.

    Args:
        combined_videos_patch (torch.Tensor): Original flattened patches, shape [B, S, N, C].
        single_view_outputs (List[torch.Tensor]): Sequence of output predictions per view.
        bool_masked_pos (List[torch.Tensor]): List indicating dropouts per view.
        patch_size (Tuple[int, int, int]): Temporal and spatial sizes of patches.
        window_size (Tuple[int, int, int]): Spatiotemporal sizes in number of patches.

    Returns:
        torch.Tensor: De-patched reconstructed videos, shape [B, S, T, H, W].
    """
    B, S, N, C = combined_videos_patch.shape
    recon_patches = combined_videos_patch.clone()  # [B, S, N, C]
    recon_videos = []

    for idx, mask in enumerate(bool_masked_pos):
        # mask: [B, N] (boolean)
        patch = recon_patches[:, idx, ...].clone()  # (B, N, C)
        out = single_view_outputs[idx].float()  # (B, N_out, C)

        mask_flat = mask.flatten().bool()  # [B*N]

        # Determine if out is full reconstruction or masked only
        if out.shape[1] == N:
            # Full reconstruction: extract masked parts
            # We want to show: Visible=Original, Masked=Reconstructed
            # out: [B, N, C] -> flatten -> [B*N, C]
            out_masked = out.reshape(-1, C)[mask_flat]
            patch.view(-1, C)[mask_flat] = out_masked
        else:
            # Masked only reconstruction (Old behavior)
            patch.view(-1, C)[mask_flat] = out.reshape(-1, C)

        recon_video = rearrange(
            patch,
            "b (t h w) (p0 p1 p2) -> b (t p0) (h p1) (w p2)",
            p0=patch_size[0],
            p1=patch_size[1],
            p2=patch_size[2],
            t=window_size[0],
            h=window_size[1],
            w=window_size[2],
        )
        recon_videos.append(recon_video)

    return torch.stack(recon_videos, dim=1)  # (B, S, T, H, W)
