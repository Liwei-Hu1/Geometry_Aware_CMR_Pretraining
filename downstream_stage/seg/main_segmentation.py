"""Segmentation fine-tuning entry-point.

Mirrors the structure of ``downstream_stage/main_finetune.py`` from the
Multimodal_Cardiac_Cycle_Generation repo, adapted for the new Geometry-Aware
pre-training codebase.

Usage::

    python downstream_stage/seg/main_segmentation.py \
        --config downstream_stage/seg/config/segmentation.yaml
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so local modules are found first.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")  # headless – must be set before importing pyplot
import matplotlib.pyplot as plt

try:
    import imageio
    _HAS_IMAGEIO = True
except ImportError:
    _HAS_IMAGEIO = False

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.dataloader import CMRDataModule
from modeling_segementation import create_segmentation_model
from pretrain_stage.modeling_pretrain import MODEL_REGISTRY as PRETRAIN_REGISTRY
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
import utils.misc as misc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Segmentation colour palette
# ---------------------------------------------------------------------------
# 0=background  1=LVBP(red)  2=LVMYO(green)  3=RVBP(blue)
# 4=LABP(yellow)  5=RABP(magenta)
_SEG_COLORS: List[Tuple[int, int, int]] = [
    (0,   0,   0),
    (255,  50,  50),
    ( 50, 200,  50),
    ( 50,  50, 255),
    (255, 220,   0),
    (220,  50, 220),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Segmentation Fine-tuning", add_help=True)
    parser.add_argument("--config", default="downstream_stage/seg/config/segmentation.yaml", type=str)
    # distributed
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", type=str)
    return parser.parse_args()





def compute_loss_safe(
    model: torch.nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    patch_coords_3d: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass + loss; unwraps DDP if needed."""
    m = model.module if hasattr(model, "module") else model
    return m.compute_loss(images, targets, patch_coords_3d=patch_coords_3d)


def save_training_logs(directory: str, train_stats: Dict[str, float], epoch: int) -> None:
    if not directory:
        return
    Path(directory).mkdir(parents=True, exist_ok=True)
    payload = {f"train_{k}": v for k, v in train_stats.items()}
    payload["epoch"] = epoch
    with open(Path(directory) / "log.txt", "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScaler,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch [{epoch}]"
    print_freq = 10

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # batch: (im_data, seg_data, masks, spatial_coords, patch_coords_3d, index)
        images  = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True).long()
        patch_coords_3d = batch[4].to(device, non_blocking=True) if isinstance(batch[4], torch.Tensor) else None

        with torch.cuda.amp.autocast(enabled=True):
            loss, dice_scores = compute_loss_safe(model, images, targets, patch_coords_3d=patch_coords_3d)

        if not math.isfinite(loss.item()):
            logger.error("Loss is %s, stopping training", loss.item())
            sys.exit(1)

        optimizer.zero_grad()
        loss_scaler(loss, optimizer, parameters=model.parameters())

        dice_list = dice_scores.flatten().tolist()
        mean_dice = sum(dice_list) / len(dice_list) if dice_list else 0.0

        metric_logger.update(loss=loss.item())
        metric_logger.update(dice=mean_dice)
        # Per-class Dice (foreground only: LVBP, LVMYO, RVBP, LABP, RABP)
        labels = ["lvbp", "lvmyo", "rvbp", "labp", "rabp"]
        for k, d in enumerate(dice_list):
            if k < len(labels):
                metric_logger.update(**{f"dice_{labels[k]}": d})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: %s", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def visualize_seg_batch(
    model: torch.nn.Module,
    batch: Any,
    device: torch.device,
    save_dir: Path,
    num_samples: int = 2,
    viz_freq_frames: int = 4,
) -> None:
    """Produce colour-overlay visualisations and save GIFs / PNGs.

    For each sample writes ``[raw | GT overlay | Pred overlay]`` horizontally
    across all slices, animated over time.  A summary PNG of the middle frame
    is always written.

    Args:
        model: Unwrapped (non-DDP) segmentation model.
        batch: DataLoader output tuple (images at index 0, targets at index 1).
        device: Compute device.
        save_dir: Root directory for this epoch's visualisations.
        num_samples: Number of batch elements to visualise.
        viz_freq_frames: Temporal stride for the animation.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    images  = batch[0].to(device)            # [B, S, T, H, W]
    targets = batch[1].to(device).long()     # [B, S, T, H, W]
    patch_coords_3d = batch[4].to(device, non_blocking=True) if isinstance(batch[4], torch.Tensor) else None
    B, S, T, H, W = images.shape
    num_samples = min(num_samples, B)

    model.eval()
    with torch.cuda.amp.autocast(enabled=True):
        logits = model(images, patch_coords_3d=patch_coords_3d)               # [B, C, S, T_out, H_out, W_out]
    _, C, S_out, T_out, H_out, W_out = logits.shape
    if T_out != T:
        logits = F.interpolate(
            logits.view(B, C, -1, H_out, W_out),
            size=(S * T, H_out, W_out), mode="trilinear", align_corners=False,
        ).view(B, C, S, T, H_out, W_out)

    preds   = logits.argmax(dim=1).cpu()    # [B, S, T, H, W]
    images  = images.cpu()
    targets = targets.cpu()
    max_label = len(_SEG_COLORS) - 1

    for b in range(num_samples):
        frames_viz: List[np.ndarray] = []
        for t in range(0, T, viz_freq_frames):
            raw_row, gt_row, pr_row = [], [], []
            for s in range(S):
                raw = images[b, s, t].numpy().astype(float)
                lo, hi = raw.min(), raw.max()
                raw_norm = ((raw - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
                raw_rgb  = np.stack([raw_norm] * 3, axis=-1)

                gt_rgb, pr_rgb = raw_rgb.copy(), raw_rgb.copy()
                for cls_idx in range(1, max_label + 1):
                    m_gt = targets[b, s, t].numpy() == cls_idx
                    m_pr = preds[b, s, t].numpy()   == cls_idx
                    if m_gt.any():
                        gt_rgb[m_gt] = _SEG_COLORS[cls_idx]
                    if m_pr.any():
                        pr_rgb[m_pr] = _SEG_COLORS[cls_idx]

                raw_row.append(raw_rgb)
                gt_row.append(gt_rgb)
                pr_row.append(pr_rgb)

            frame = np.concatenate([
                np.concatenate(raw_row, axis=1),
                np.concatenate(gt_row,  axis=1),
                np.concatenate(pr_row,  axis=1),
            ], axis=0)
            frames_viz.append(frame.astype(np.uint8))

        sample_dir = save_dir / f"sample_{b}"
        sample_dir.mkdir(exist_ok=True)

        if _HAS_IMAGEIO:
            imageio.mimsave(str(sample_dir / "overlay.gif"), frames_viz, fps=4, loop=0)
        else:
            for ti, frame in enumerate(frames_viz):
                fig, ax = plt.subplots(figsize=(frame.shape[1] / 72, frame.shape[0] / 72))
                ax.imshow(frame); ax.axis("off")
                plt.tight_layout(pad=0)
                plt.savefig(sample_dir / f"frame_{ti:03d}.png", dpi=72, bbox_inches="tight")
                plt.close()

        mid = frames_viz[len(frames_viz) // 2]
        fig, ax = plt.subplots(figsize=(mid.shape[1] / 72, mid.shape[0] / 72))
        ax.imshow(mid); ax.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(sample_dir / "summary.png", dpi=100, bbox_inches="tight")
        plt.close()

    logger.info("[Viz] Saved %d samples → %s", num_samples, save_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    misc.init_distributed_mode(args)
    cfg = OmegaConf.load(args.config)

    output_dir = Path(cfg.general.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.general.device)
    cudnn.benchmark = True

    # ── Reproducibility ──────────────────────────────────────────────────────
    seed = int(cfg.general.get("seed", 0)) + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Data ─────────────────────────────────────────────────────────────────
    data_module = CMRDataModule.from_config(cfg)
    dataset_train = data_module.get_train_dataloader()
    dataset_val   = data_module.get_val_dataloader()
    logger.info("Train samples: %d  |  Val samples: %d", len(dataset_train), len(dataset_val))

    if misc.get_world_size() > 1:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=misc.get_world_size(),
            rank=misc.get_rank(), shuffle=True,
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.get("pin_mem", False)),
        drop_last=True,
        persistent_workers=True
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model_name = cfg.module.module_name
    dataset_cls = cfg.data.get("dataset_cls", "")
    
    if "AllAX" in dataset_cls:
        model_slices = int(cfg.data.sax_slice_num) + 3
    elif "LAX" in dataset_cls and "SAX" not in dataset_cls:
        model_slices = 3
    else:
        model_slices = int(cfg.data.sax_slice_num)

    # Handle OmegaConf ListConfig properly
    patch_size_cfg = cfg.data.get("img_patch_size", [4, 4])
    import omegaconf
    if omegaconf.OmegaConf.is_list(patch_size_cfg):
        patch_size_cfg = omegaconf.OmegaConf.to_container(patch_size_cfg, resolve=True)

    patch_size = patch_size_cfg[0] if isinstance(patch_size_cfg, (list, tuple)) else int(patch_size_cfg)
    pretrained_path = cfg.general.get("pretrained_path", "")

    logger.info("Building dynamic segmentation model (backbone: %s) for %d classes, %d slices", 
                model_name, int(cfg.data.get("num_classes", 6)), model_slices)
    
    model = create_segmentation_model(
        module_name=model_name,
        checkpoint_path=pretrained_path if pretrained_path else None,
        num_classes=int(cfg.data.get("num_classes", 6)),
        img_size=int(cfg.data.get("image_size", 112)),
        patch_size=patch_size,
        num_frames=int(cfg.data.get("time_frame", 32)),
        tubelet_size=int(cfg.data.get("tubelet_size", 8)),
        num_slices=model_slices,
        decoder_feature=int(cfg.data.get("decoder_feature", 32)),
    )
    model.to(device)

    # Optionally freeze the encoder
    if cfg.general.get("freeze_encoder", False):
        m = model.module if hasattr(model, "module") else model
        if hasattr(m, "shared_encoder"):
            for p in m.shared_encoder.parameters():
                p.requires_grad = False
            logger.info("Frozen shared_encoder (%d params)",
                        sum(p.numel() for p in m.shared_encoder.parameters()))

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total params: %.2f M  |  Trainable: %.2f M",
                n_total / 1e6, n_trainable / 1e6)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    opt_cfg = cfg.module.get("optimizer_params", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 1e-4)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.05)),
    )
    loss_scaler = NativeScaler()

    # ── Optional checkpoint resume ────────────────────────────────────────────
    ckpt_path = cfg.general.get("ckpt_path", "")
    start_epoch = 0
    if ckpt_path and Path(ckpt_path).exists():
        logger.info("Resuming from %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info("Resumed at epoch %d", start_epoch)

    # ── Pre-load one fixed val batch for per-epoch visualisation ──────────────
    # Use num_workers=0 to avoid shared-memory conflicts on NAS storage.
    viz_batch: Optional[Any] = None
    if misc.is_main_process():
        _viz_loader = DataLoader(
            dataset_val,
            batch_size=min(2, len(dataset_val)),
            num_workers=0,
            pin_memory=False,
            shuffle=False,
        )
        for _b in _viz_loader:
            viz_batch = [x.cpu() if isinstance(x, torch.Tensor) else x for x in _b]
            break
        del _viz_loader
        if viz_batch:
            logger.info("[Viz] Pre-loaded viz batch: %s", viz_batch[0].shape)

    # ── Training loop ─────────────────────────────────────────────────────────
    sched_cfg = cfg.module.get("scheduler_params", cfg.module.get("secheduler_params", {}))
    num_epochs = int(sched_cfg.get("epochs", 50))
    save_freq  = int(cfg.general.get("save_ckpt_freq", 5))

    logger.info("Starting training for %d epochs", num_epochs)
    start_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        if misc.get_world_size() > 1:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler)

        # ── Save logs ─────────────────────────────────────────────────────────
        if misc.is_main_process():
            save_training_logs(str(output_dir), train_stats, epoch)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if misc.is_main_process():
            if (epoch + 1) % save_freq == 0 or (epoch + 1) == num_epochs:
                save_path = output_dir / f"checkpoint-{epoch}.pth"
                torch.save({
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch":     epoch,
                }, save_path)
                logger.info("Saved checkpoint → %s", save_path)

        # ── Per-epoch visualisation ────────────────────────────────────────────
        if misc.is_main_process() and viz_batch is not None:
            viz_save_dir = output_dir / "viz" / f"epoch_{epoch:04d}"
            visualize_seg_batch(
                model,
                viz_batch,
                device,
                save_dir=viz_save_dir,
                num_samples=min(2, viz_batch[0].shape[0]),
                viz_freq_frames=4,
            )

        elapsed = time.time() - start_time
        logger.info(
            "Epoch %d finished | loss=%.4f | dice=%.4f | elapsed=%s",
            epoch,
            train_stats.get("loss", 0.0),
            train_stats.get("dice", 0.0),
            str(datetime.timedelta(seconds=int(elapsed))),
        )

    total_time = time.time() - start_time
    logger.info("Training complete in %s", str(datetime.timedelta(seconds=int(total_time))))


if __name__ == "__main__":
    main(get_args())
