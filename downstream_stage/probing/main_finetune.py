import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

try:
    import imageio

    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

import matplotlib

matplotlib.use("Agg")  # headless — must be set before any other matplotlib import
import matplotlib.pyplot as plt

# Add project root to path (3 levels up from downstream_stage/probing/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Any, Optional, Tuple

import utils.misc as misc
from data import unified_datasets
from modeling_finetune import create_finetune_model
from utils.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser for fine-tuning.

    Returns:
        argparse.ArgumentParser: Parser with defined CLI arguments.
    """
    parser = argparse.ArgumentParser("Finetune Segmentation", add_help=False)
    parser.add_argument(
        "--config", default="downstream_stage/probing/config/finetune.yaml", type=str
    )
    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )

    # Distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    return parser


def load_data(config: OmegaConf) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Load train and validation datasets based on the unified_datasets specification.

    Args:
        config (OmegaConf): Configuration settings.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]: A tuple containing the (train, validation) datasets.
    """
    dataset_cls = getattr(unified_datasets, config.data.dataset_cls)

    # Common args
    args = dict(
        load_seg=config.data.load_seg,
        mask_ratio=config.data.mask_ratio,  # Should be 0.0
        window_size=(
            (config.data.tubelet_size, *config.data.img_patch_size)
            if hasattr(config.data, "img_patch_size")
            else None
        ),
    )

    print(f"Loading Train Data from {config.data.processed_dir}")
    # Assuming file structure: [processed_dir]/train/*.npz
    # unified_datasets usually takes list of paths.

    # We need to list files.
    data_root = Path(config.data.processed_dir)
    # The AbstractDataset expects subject_paths.

    all_files = sorted(list(data_root.glob("*.npz")))
    if len(all_files) == 0:
        # Maybe recursion?
        all_files = sorted(list(data_root.rglob("*.npz")))

    # Split
    num_train = config.data.num_train
    num_val = config.data.num_val
    train_files = all_files[:num_train]
    val_files = all_files[num_train : num_train + num_val]

    dataset_train = dataset_cls(train_files, **args)
    dataset_val = dataset_cls(val_files, **args)

    print(f"Train samples: {len(dataset_train)}, Val samples: {len(dataset_val)}")

    return dataset_train, dataset_val


def compute_loss_safe(
    model: nn.Module, 
    images: torch.Tensor, 
    targets: torch.Tensor, 
    patch_coords_3d: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, dict]:
    """Compute loss safely handling DDP or serial models.

    Args:
        model (nn.Module): Training model.
        images (torch.Tensor): Input image tensor.
        targets (torch.Tensor): Target segmentation maps.
        patch_coords_3d (Optional[torch.Tensor], optional): Physical volume coords. Defaults to None.

    Returns:
        Tuple[torch.Tensor, dict]: Computed loss value, and loss metric dictionary.
    """
    if hasattr(model, "module"):
        return model.module.compute_loss(
            images, targets, patch_coords_3d=patch_coords_3d
        )
    return model.compute_loss(images, targets, patch_coords_3d=patch_coords_3d)


def save_training_logs(
    directory: str, train_stats: dict, val_stats: dict, epoch: int
) -> None:
    """Save epoch performance metrics incrementally out to a JSON file.

    Args:
        directory (str): Metric log export directory.
        train_stats (dict): Dictionary mapping training names to float stats.
        val_stats (dict): Dictionary mapping validation names to float stats.
        epoch (int): Training timeline marker.
    """
    if not directory:
        return
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_stats = {
        **{f"train_{k}": v for k, v in train_stats.items()},
        **{f"val_{k}": v for k, v in val_stats.items()},
        "epoch": epoch,
    }

    with (output_dir / "log.txt").open("a") as f:
        f.write(json.dumps(log_stats) + "\n")


def train_one_epoch(
    model: nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device, 
    epoch: int, 
    loss_scaler: Any, 
    config: OmegaConf
) -> dict:
    """Perform one training epoch over the dataloader.

    Args:
        model (nn.Module): Segmentation model mapping volume.
        data_loader (torch.utils.data.DataLoader): Dataset dataloader iterator.
        optimizer (torch.optim.Optimizer): Model parameter updater.
        device (torch.device): Compute target (e.g. cuda).
        epoch (int): Current epoch number.
        loss_scaler (Any): Mixed precision gradient scaler.
        config (OmegaConf): Complete OmegaConf system parameters.

    Returns:
        dict: Aggregated running step metrics mapping metric names to scalar averages.
    """
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    for step, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        images = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True).long()
        patch_coords_3d = batch[4].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=True):
            loss, dice_scores = compute_loss_safe(
                model, images, targets, patch_coords_3d=patch_coords_3d
            )

        optimizer.zero_grad()
        loss_scaler(loss, optimizer, parameters=model.parameters())

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            print("Loss is infinite, stopping")
            sys.exit(1)

        dice_list = dice_scores.flatten().tolist()
        dice = sum(dice_list) / len(dice_list) if len(dice_list) > 0 else 0.0

        metric_logger.update(loss=loss_val)
        metric_logger.update(dice=dice)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # Log Per-Class Dice
        labels = ["LVBP", "LVMYO", "RVBP", "LABP", "RABP"]
        for k, d_val in enumerate(dice_list):
            if k < len(labels):
                metric_logger.update(**{f"dice_{labels[k]}": d_val})

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    config: OmegaConf
) -> dict:
    """Evaluate deep neural network performance via evaluation epoch.

    Args:
        model (nn.Module): Network segment mapping payload properties.
        data_loader (torch.utils.data.DataLoader): Sourced system datasets iterator.
        device (torch.device): Distributed computation target.
        config (OmegaConf): Evaluation scope configs payload mapping.

    Returns:
        dict: Final calculated evaluation epoch performance scalar metrics.
    """
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    num_classes_eval = 5
    per_class_dice_sum = torch.zeros(num_classes_eval, device=device)
    total_samples = torch.zeros(1, device=device)

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True).long()
        patch_coords_3d = batch[4].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=True):
            loss, dice_scores = compute_loss_safe(
                model, images, targets, patch_coords_3d=patch_coords_3d
            )

        dice_list = dice_scores.flatten().tolist()
        dice = sum(dice_list) / len(dice_list) if len(dice_list) > 0 else 0.0

        metric_logger.update(loss=loss.item())
        metric_logger.update(dice=dice)

        # Log Per-Class Dice
        labels = ["LVBP", "LVMYO", "RVBP", "LABP", "RABP"]
        for k, d_val in enumerate(dice_list):
            if k < len(labels):
                metric_logger.update(**{f"dice_{labels[k]}": d_val})

        # Accumulate per-class dice
        for k, d_val in enumerate(dice_list):
            if k < num_classes_eval:
                per_class_dice_sum[k] += d_val
        total_samples += 1

    # Synchronize
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if misc.is_dist_avail_and_initialized():
        torch.distributed.all_reduce(per_class_dice_sum)
        torch.distributed.all_reduce(total_samples)

    avg_per_class = per_class_dice_sum / (total_samples + 1e-6)
    avg_per_class = avg_per_class.tolist()

    labels = ["LVBP", "LVMYO", "RVBP", "LABP", "RABP"]
    log_str = " | ".join(
        [f"{labels[i]}: {avg_per_class[i]:.4f}" for i in range(len(labels))]
    )

    print(f"Val Per-Class Dice:\n{log_str}")
    print(f"Val Mean Dice: {metric_logger.dice.global_avg:.4f}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ---------------------------------------------------------------------------
# Segmentation Visualisation
# ---------------------------------------------------------------------------
# Class 0=background(black), 1=LVBP(red), 2=LVMYO(green), 3=RVBP(blue),
#         4=LABP(yellow),     5=RABP(magenta)
_SEG_COLORS = [
    (0, 0, 0),  # 0 background
    (255, 50, 50),  # 1 LVBP    – red
    (50, 200, 50),  # 2 LVMYO   – green
    (50, 50, 255),  # 3 RVBP    – blue
    (255, 220, 0),  # 4 LABP    – yellow
    (220, 50, 220),  # 5 RABP    – magenta
]


@torch.no_grad()
def visualize_seg_batch(
    model,
    batch,
    device,
    save_dir: Path,
    num_samples: int = 2,
    viz_freq_frames: int = 4,  # show every N-th time frame to reduce file size
):
    """
    Produce a colour-overlay visualisation for the first `num_samples` items
    of `batch` and write them to `save_dir`.

    For each sample a GIF (or falling back to per-frame PNGs) is saved showing:
        [raw_image | GT_overlay | Pred_overlay]
    stacked horizontally, one row per slice, animated over time.

    Args:
        batch:  output from DataLoader (list).
                batch[0] = images   [B, S, T, H, W]
                batch[1] = targets  [B, S, T, H, W]  (integer labels)
                batch[4] = patch_coords_3d
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    images = batch[0].to(device)  # [B, S, T, H, W]
    targets = batch[1].to(device).long()  # [B, S, T, H, W]
    patch_coords = batch[4].to(device)
    B, S, T, H, W = images.shape
    num_samples = min(num_samples, B)

    # Run forward pass
    model.eval()
    with torch.cuda.amp.autocast(enabled=True):
        logits = model(
            images, patch_coords_3d=patch_coords
        )  # [B, C, S, T, H, W] or [B, C, S, T_p, H, W]
    # Upsample temporal dim if model output is at tubelet resolution
    _, C, S_out, T_out, H_out, W_out = logits.shape
    if T_out != T:
        logits = F.interpolate(
            logits.view(B, C, -1, H_out, W_out),  # treat S as depth for 3D interp
            size=(S * T, H_out, W_out),
            mode="trilinear",
            align_corners=False,
        ).view(B, C, S, T, H_out, W_out)
    preds = logits.argmax(dim=1).cpu()  # [B, S, T, H, W]
    images = images.cpu()
    targets = targets.cpu()

    max_label = len(_SEG_COLORS) - 1

    for b in range(num_samples):
        # ------------------------------------------------------------------
        # Build colour overlay for every frame
        # frame_list: list of (H_strip, W_strip, 3) uint8 arrays
        # ------------------------------------------------------------------
        frames_viz = []
        for t in range(0, T, viz_freq_frames):
            raw_row, gt_row, pr_row = [], [], []
            for s in range(S):
                # ── raw image (normalised to 0-255) ──
                raw = images[b, s, t].numpy().astype(float)
                lo, hi = raw.min(), raw.max()
                raw_norm = ((raw - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
                raw_rgb = np.stack([raw_norm] * 3, axis=-1)  # (H, W, 3) gray

                # ── GT overlay ──
                gt_rgb = raw_rgb.copy()
                gt_lbl = targets[b, s, t].numpy()
                for cls_idx in range(1, max_label + 1):
                    mask = gt_lbl == cls_idx
                    if mask.any():
                        gt_rgb[mask] = _SEG_COLORS[cls_idx]

                # ── Pred overlay ──
                pr_rgb = raw_rgb.copy()
                pr_lbl = preds[b, s, t].numpy()
                for cls_idx in range(1, max_label + 1):
                    mask = pr_lbl == cls_idx
                    if mask.any():
                        pr_rgb[mask] = _SEG_COLORS[cls_idx]

                raw_row.append(raw_rgb)
                gt_row.append(gt_rgb)
                pr_row.append(pr_rgb)

            # Each row: all slices side-by-side horizontally → (H, S*W, 3)
            # Then stack 3 rows vertically → (3H, S*W, 3)
            frame = np.concatenate(
                [
                    np.concatenate(raw_row, axis=1),  # row 1: raw
                    np.concatenate(gt_row, axis=1),  # row 2: GT
                    np.concatenate(pr_row, axis=1),  # row 3: pred
                ],
                axis=0,
            )
            frames_viz.append(frame.astype(np.uint8))

        sample_dir = save_dir / f"sample_{b}"
        sample_dir.mkdir(exist_ok=True)

        if HAS_IMAGEIO:
            gif_path = sample_dir / "overlay.gif"
            imageio.mimsave(str(gif_path), frames_viz, fps=4, loop=0)
        else:
            # Fallback: save individual PNG frames with matplotlib
            for ti, frame in enumerate(frames_viz):
                fig, ax = plt.subplots(
                    figsize=(frame.shape[1] / 72, frame.shape[0] / 72)
                )
                ax.imshow(frame)
                ax.axis("off")
                plt.tight_layout(pad=0)
                plt.savefig(
                    sample_dir / f"frame_{ti:03d}.png", dpi=72, bbox_inches="tight"
                )
                plt.close()

        # Always save a middle-frame summary PNG for quick inspection
        mid_frame = frames_viz[len(frames_viz) // 2]
        fig, ax = plt.subplots(
            figsize=(mid_frame.shape[1] / 72, mid_frame.shape[0] / 72)
        )
        ax.imshow(mid_frame)
        ax.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(sample_dir / "summary.png", dpi=100, bbox_inches="tight")
        plt.close()

    print(f"[Viz] Saved {num_samples} samples → {save_dir}")


def main(args: argparse.Namespace) -> None:
    """Launch fine-tuning application and run epochs.

    Args:
        args (argparse.Namespace): System defined argument mapped to options.
    """
    misc.init_distributed_mode(args)

    cfg = OmegaConf.load(args.config)

    # Override with relevant args
    if args.output_dir:
        cfg.general.output_dir = args.output_dir

    output_dir = Path(cfg.general.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    cudnn.benchmark = True

    # Dataset
    dataset_train, dataset_val = load_data(cfg)

    # Dataloader
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if len(dataset_val) % num_tasks != 0:
            print(
                "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                "This will slightly alter validation results as extra duplicate entries are added to achieve "
                "equal num of samples per-process."
            )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_mem,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_mem,
        drop_last=False,
    )

    # Calculate num_slices for model creation based on dataset class
    if "AllAX" in cfg.data.dataset_cls:
        model_slices = cfg.data.sax_slice_num + 3
    elif (
        "LAX" in cfg.data.dataset_cls and "SAX" not in cfg.data.dataset_cls
    ):  # Pure LAX
        model_slices = 3
    else:  # SAX or default
        model_slices = cfg.data.sax_slice_num

    # Model
    model = create_finetune_model(
        backbone_name=cfg.module.module_name,
        checkpoint_path=cfg.general.pretrained_path,
        num_classes=cfg.data.num_classes,
        img_size=cfg.data.image_size,
        num_frames=cfg.data.time_frame,
        patch_size=4,  # Based on config name
        tubelet_size=cfg.data.tubelet_size,
        num_slices=model_slices,
        decoder_feature=(
            cfg.data.decoder_feature if hasattr(cfg.data, "decoder_feature") else 32
        ),
    )
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # Freeze Encoder if requested
    if cfg.general.freeze_encoder:
        print("Freezing Backbone Encoder and Volume Head")
        if hasattr(model_without_ddp, "shared_encoder"):
            for param in model_without_ddp.shared_encoder.parameters():
                param.requires_grad = False
            print("Frozen shared_encoder")
        if (
            hasattr(model_without_ddp, "volume_head")
            and model_without_ddp.volume_head is not None
        ):
            for param in model_without_ddp.volume_head.parameters():
                param.requires_grad = False
            print("Frozen volume_head")

        # Fallback for other models
        if hasattr(model_without_ddp, "backbone"):
            for param in model_without_ddp.backbone.parameters():
                param.requires_grad = False

    print(
        f"Model created. Param count: {sum(p.numel() for p in model.parameters())/1e6:.2f}M"
    )
    print(
        f"Trainable Param count: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.module.optimizer_params.lr),
        weight_decay=cfg.module.optimizer_params.weight_decay,
    )
    loss_scaler = NativeScaler()

    print("Start training")
    start_time = time.time()
    max_accuracy = 0.0

    num_epochs = cfg.module.secheduler_params.epochs
    print(f"Total Epochs: {num_epochs}")

    # Pre-load ONE fixed val batch for visualisation (num_workers=0 to avoid deadlock
    # with the main val dataloader which uses multi-process workers)
    viz_batch = None
    if misc.is_main_process():
        _viz_loader = DataLoader(
            dataset_val,
            batch_size=min(2, len(dataset_val)),
            num_workers=0,  # IMPORTANT: 0 to avoid worker conflict
            pin_memory=False,
            shuffle=False,
        )
        for _b in _viz_loader:
            viz_batch = [x.cpu() if isinstance(x, torch.Tensor) else x for x in _b]
            break
        del _viz_loader
        print(
            f"[Viz] Pre-loaded viz batch: {viz_batch[0].shape if viz_batch else None}"
        )

    for epoch in range(num_epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler, cfg
        )
        val_stats = evaluate(model, data_loader_val, device, cfg)

        print(
            f"Epoch {epoch} Train Loss: {train_stats['loss']:.4f} Dice: {train_stats['dice']:.4f}  Val Loss: {val_stats['loss']:.4f} Dice: {val_stats['dice']:.4f}"
        )

        # Save Logs
        if misc.is_main_process():
            save_training_logs(cfg.general.output_dir, train_stats, val_stats, epoch)

        # Segmentation Visualisation — every epoch
        # Use model_without_ddp (unwrapped) so rank-0 can run forward independently
        # without triggering DDP's all-reduce which requires ALL ranks to participate.
        if misc.is_main_process() and viz_batch is not None:
            viz_save_dir = output_dir / "viz" / f"epoch_{epoch:04d}"
            visualize_seg_batch(
                model_without_ddp,  # ← unwrapped model, no DDP sync needed
                viz_batch,
                device,
                save_dir=viz_save_dir,
                num_samples=min(2, viz_batch[0].shape[0]),
                viz_freq_frames=4,
            )

        # Save Checkpoint
        if (
            epoch % cfg.general.save_ckpt_freq == 0
            or epoch == cfg.module.secheduler_params.epochs - 1
        ) and misc.is_main_process():
            save_path = output_dir / f"checkpoint-{epoch}.pth"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                },
                save_path,
            )

            if val_stats["dice"] > max_accuracy:
                max_accuracy = val_stats["dice"]
                torch.save(
                    model_without_ddp.state_dict(), output_dir / "best_checkpoint.pth"
                )

    total_time = time.time() - start_time
    print(f"Training time {str(datetime.timedelta(seconds=int(total_time)))}")


if __name__ == "__main__":
    # import debugpy
    # try:
    #     # 9501 is the default attach port in the VS Code debug configuration
    #     debugpy.listen(("localhost", 9501))
    #     print("Waiting for debugger attach")
    #     debugpy.wait_for_client()
    # except Exception as e:
    #     pass
    args = get_args_parser().parse_args()
    main(args)
