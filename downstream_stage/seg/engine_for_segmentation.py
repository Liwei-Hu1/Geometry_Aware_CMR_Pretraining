import math
import sys
from typing import Any, Dict, Iterable, Tuple, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from modeling_segementation import SegmentationCriterion, to_1hot

from utils.misc import MetricLogger, SmoothedValue


def _prepare_batch(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract videos and segmentation masks from the batch dictionary.

    Args:
        batch (Dict[str, torch.Tensor]): Data batch from dataloader.
        device (torch.device): Target device for tensors.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (videos, segs) tensors on the target device.
    """
    videos = batch["sax_im_data"].to(device, non_blocking=True)
    segs = batch["seg_sax_data"].to(device, non_blocking=True)
    return videos, segs


def _update_optimizer(
    optimizer: torch.optim.Optimizer,
    it: int,
    lr_schedule_values: Optional[np.ndarray],
    wd_schedule_values: Optional[np.ndarray],
) -> None:
    """Update learning rate and weight decay per step according to schedules.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to update.
        it (int): Current training step iteration.
        lr_schedule_values (Optional[np.ndarray]): Learning rate schedule array.
        wd_schedule_values (Optional[np.ndarray]): Weight decay schedule array.
    """
    if lr_schedule_values is not None or wd_schedule_values is not None:
        for i, param_group in enumerate(optimizer.param_groups):
            if lr_schedule_values is not None:
                param_group["lr"] = lr_schedule_values[it] * param_group.get(
                    "lr_scale", 1.0
                )
            if (
                wd_schedule_values is not None
                and param_group.get("weight_decay", 0.0) > 0
            ):
                param_group["weight_decay"] = wd_schedule_values[it]


def _compute_segmentation_loss(
    model: torch.nn.Module,
    videos: torch.Tensor,
    segs: torch.Tensor,
    loss_func: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the segmentation loss and Dice scores for a batch.

    Args:
        model (torch.nn.Module): The segmentation model.
        videos (torch.Tensor): Input video tensor.
        segs (torch.Tensor): Ground truth segmentation masks.
        loss_func (nn.Module): The criterion to use if the model doesn't compute loss internally.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the loss scalar and Dice scores.
    """
    if hasattr(model, "compute_loss"):
        return model.compute_loss(videos, segs)
    outputs = model(videos)
    gt_segs = to_1hot(segs, num_class=3).permute(0, 5, 1, 2, 3, 4)
    return loss_func(pred=outputs, target=gt_segs)


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler: Any,
    max_norm: float = 0.0,
    log_writer: Any = None,
    lr_schedule_values: Optional[np.ndarray] = None,
    wd_schedule_values: Optional[np.ndarray] = None,
    start_steps: int = 0,
    normlize_target: bool = False,
) -> Dict[str, Any]:
    """Train the segmentation model for one epoch.

    Args:
        model (torch.nn.Module): The segmentation model.
        data_loader (Iterable): Training DataLoader.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Compute device.
        epoch (int): Current epoch number.
        loss_scaler (Any): AMP loss scaler.
        max_norm (float, optional): Max gradient norm for clipping. Defaults to 0.0.
        log_writer (Any, optional): Logger for TensorBoard/WandB. Defaults to None.
        lr_schedule_values (Optional[np.ndarray], optional): LR schedule array. Defaults to None.
        wd_schedule_values (Optional[np.ndarray], optional): Weight decay schedule array. Defaults to None.
        start_steps (int, optional): Global step count at epoch start. Defaults to 0.
        normlize_target (bool, optional): Unused legacy flag. Defaults to False.

    Returns:
        Dict[str, Any]: Dictionary of averaged training statistics.
    """
    model.train()
    metric_logger = MetricLogger(delimiter="")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("min_lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 10

    loss_func = SegmentationCriterion()

    for step, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        it = start_steps + step
        _update_optimizer(optimizer, it, lr_schedule_values, wd_schedule_values)

        # Batch: (im_data, seg_data, masks, spatial_coords, patch_coords_3d, index)
        videos, segs = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, dice_scores = _compute_segmentation_loss(
                model, videos, segs, loss_func
            )

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

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
        loss_scale_value = loss_scaler.state_dict().get("scale", 1.0)

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        metric_logger.update(dice_lvmyo=dice_scores[0])
        metric_logger.update(dice_lvbp=dice_scores[1])

        min_lr = min(group["lr"] for group in optimizer.param_groups)
        max_lr = max(group["lr"] for group in optimizer.param_groups)
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        weight_decay_value = next(
            (
                group["weight_decay"]
                for group in optimizer.param_groups
                if group.get("weight_decay", 0.0) > 0
            ),
            None,
        )
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(dice_lvmyo=dice_scores[0], head="metrics")
            log_writer.update(dice_lvbp=dice_scores[1], head="metrics")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def overlay_seg_on_image(
    img: Union[torch.Tensor, np.ndarray],
    mask: Union[torch.Tensor, np.ndarray],
    num_classes: int = 3,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a segmentation mask on a grayscale image.

    Args:
        img (Union[torch.Tensor, np.ndarray]): Grayscale image (2D or 3D).
        mask (Union[torch.Tensor, np.ndarray]): Integer mask of the same spatial shape.
        num_classes (int, optional): Expected number of classes. Defaults to 3.
        alpha (float, optional): Transparency of the mask overlay. Defaults to 0.4.

    Returns:
        np.ndarray: RGB image with the overlaid mask as uint8.
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    if img.ndim == 3:
        img = img[0]
    H, W = img.shape
    img_norm = np.clip(img, 0.0, 1.0)
    img_rgb = (np.stack([img_norm] * 3, axis=-1) * 255.0).astype(np.uint8)
    color_map = {
        0: np.array([0, 0, 0], dtype=np.uint8),
        1: np.array([255, 0, 0], dtype=np.uint8),
        2: np.array([0, 255, 0], dtype=np.uint8),
    }
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id, color in color_map.items():
        color_mask[mask == cls_id] = color
    overlay = (1 - alpha) * img_rgb.astype(np.float32) + alpha * color_mask.astype(
        np.float32
    )
    return overlay.astype(np.uint8)


def simple_inference_and_vis(
    model: torch.nn.Module,
    specific_cmr_around_seg: torch.Tensor,
    specific_cmr_around_seg_label: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> None:
    """Run simplistic inference on a specific CMR frame and plot the results using matplotlib.

    Args:
        model (torch.nn.Module): The segmentation model.
        specific_cmr_around_seg (torch.Tensor): Single CMR sequence snippet.
        specific_cmr_around_seg_label (Optional[torch.Tensor], optional): Optional ground truth labels. Defaults to None.
        device (Optional[torch.device], optional): Device for tensor computation. Defaults to None.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x = specific_cmr_around_seg.to(device)
        outputs = model(x)
        preds = outputs.argmax(dim=1)
    b, s, t = 0, 3, 5
    img = specific_cmr_around_seg[b, s, t]
    mask = preds[b, s, t]
    overlay_pred = overlay_seg_on_image(img, mask, alpha=0.5)

    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1)
    plt.title("Image")
    plt.axis("off")
    plt.imshow(img.cpu(), cmap="gray")
    plt.subplot(1, 4, 2)
    plt.title("Pred Mask")
    plt.axis("off")
    plt.imshow(mask.cpu(), cmap="jet")
    plt.subplot(1, 4, 3)
    plt.title("Pred Overlay")
    plt.axis("off")
    plt.imshow(overlay_pred)

    if specific_cmr_around_seg_label is not None:
        gt_mask = specific_cmr_around_seg_label[b, s, t]
        overlay_gt = overlay_seg_on_image(img, gt_mask, alpha=0.5)
        plt.subplot(1, 4, 4)
        plt.title("GT Overlay")
        plt.axis("off")
        plt.imshow(overlay_gt)

    plt.tight_layout()
    plt.show()
