import argparse
import datetime
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typing import Optional, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from engine_for_pretraining import train_one_epoch
from modeling_pretrain import MODEL_REGISTRY
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from data.dataloader import CMRDataModule
from optim.optim_factory import create_optimizer
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.misc import (
    TensorboardLogger,
    auto_load_model,
    cosine_scheduler,
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_main_process,
    save_model,
    seed_worker,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class OptimizerArguments:
    """Arguments for optimizer configuration."""
    lr: float
    min_lr: float
    warmup_lr: float
    use_checkpoint: bool = False
    opt: str = "adamw"
    weight_decay: float = 0.0
    weight_decay_end: Optional[float] = None
    opt_eps: float = 1e-8
    opt_betas: Optional[Tuple[float, float]] = None
    drop_path: float = 0.0
    start_epoch: int = 0
    clip_grad: Optional[float] = None
    momentum: float = 0.9
    output_dir: str = ""
    resume: str = ""
    auto_resume: bool = False
    distributed: bool = False
    gpu: Optional[int] = None
    epochs: int = 0


@dataclass
class SchedulerArguments:
    """Arguments for learning rate scheduling."""
    epochs: int
    warmup_epochs: int
    warmup_steps: int
    normalize_target: bool = False


@dataclass
class GeneralArguments:
    """General arguments for the training script."""
    seed: int = 0
    wandb_disabled: bool = False
    freeze_encoder: bool = False
    resume_training: bool = False
    load_encoder: bool = False
    log_dir: str = ""
    output_dir: str = ""
    ckpt_path: str = ""
    save_ckpt_freq: int = 50
    device: str = "cuda"
    resume: str = ""
    auto_resume: bool = False


def get_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser("VideoMAE CMR pre-training", add_help=False)
    parser.add_argument(
        "--config",
        default="pretrain_stage/config/pretrain.yaml",
        type=str,
        help="Path to YAML config",
    )

    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", type=str)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def load_config(path: str) -> DictConfig:
    return OmegaConf.load(path)


def build_optimizer_arguments(
    cfg: DictConfig,
    general_cfg: GeneralArguments,
    batch_size: int,
    total_batch_size: int,
) -> Tuple[OptimizerArguments, SchedulerArguments]:
    """Build optimizer and scheduler argument dataclasses from config.

    Args:
        cfg (DictConfig): Hydar/OmegaConf configuration object.
        general_cfg (GeneralArguments): General training arguments.
        batch_size (int): Per-GPU batch size.
        total_batch_size (int): Total effective batch size across all GPUs.

    Returns:
        Tuple[OptimizerArguments, SchedulerArguments]: Configured arguments.
    """
    optim_cfg = cfg.module.optimizer_params
    scaled_lr = float(optim_cfg.lr) * total_batch_size / batch_size
    scaled_min_lr = float(optim_cfg.min_lr) * total_batch_size / batch_size
    scaled_warmup_lr = float(optim_cfg.warmup_lr) * total_batch_size / batch_size

    weight_decay_end_value = (
        float(optim_cfg.weight_decay_end)
        if optim_cfg.weight_decay_end is not None
        else float(optim_cfg.weight_decay)
    )

    opt_betas = tuple(optim_cfg.opt_betas) if optim_cfg.opt_betas is not None else None

    optimizer_args = OptimizerArguments(
        lr=scaled_lr,
        min_lr=scaled_min_lr,
        warmup_lr=scaled_warmup_lr,
        opt=optim_cfg.opt,
        weight_decay=float(optim_cfg.weight_decay),
        weight_decay_end=weight_decay_end_value,
        opt_eps=float(optim_cfg.opt_eps),
        opt_betas=opt_betas,
        output_dir=general_cfg.output_dir,
        resume=general_cfg.resume,
        auto_resume=general_cfg.auto_resume,
        distributed=get_world_size() > 1,
        gpu=None,
        start_epoch=int(optim_cfg.start_epoch),
        clip_grad=(
            float(optim_cfg.clip_grad) if optim_cfg.clip_grad is not None else None
        ),
    )

    sched_args = cfg.module.secheduler_params
    scheduler_args = SchedulerArguments(
        epochs=int(sched_args.epochs),
        warmup_epochs=int(sched_args.warmup_epochs),
        warmup_steps=int(sched_args.warmup_steps),
    )

    return optimizer_args, scheduler_args


def create_data_loader(
    cfg: DictConfig,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Create a distributed DataLoader for the dataset.

    Args:
        cfg (DictConfig): Global configuration.
        dataset (Dataset): Dataset instance.
        batch_size (int): Batch size per process.
        num_workers (int): Number of dataloader workers.
        pin_memory (bool): Whether to use pinned memory.

    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    num_tasks = get_world_size()
    global_rank = get_rank()
    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True,
    )
    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=seed_worker,
    )


def prepare_log_writer(general_cfg: GeneralArguments) -> Optional[TensorboardLogger]:
    if is_main_process() and general_cfg.log_dir:
        path = Path(general_cfg.log_dir)
        path.mkdir(parents=True, exist_ok=True)
        logger.info("TensorBoard log dir: %s", path)
        return TensorboardLogger(log_dir=str(path))
    return None


def build_model(module_cfg: DictConfig) -> torch.nn.Module:
    model_name = module_cfg.module_name
    constructor = MODEL_REGISTRY.get(model_name)
    if constructor is None:
        raise ValueError(
            f"Model {model_name} not registered in modeling_pretrain.MODEL_REGISTRY."
        )
    logger.info("Creating model %s", model_name)
    model_kwargs = {}
    for key in (
        "volume_head_type",
        "geo_window_radius",
        "geo_distance_alpha",
        "geo_q_chunk_size",
    ):
        if key in module_cfg:
            model_kwargs[key] = module_cfg[key]
    return constructor(**model_kwargs)


def save_training_logs(
    directory: str, train_stats: dict, epoch: int, n_parameters: int
) -> None:
    """Save training statistics to a JSON log file.

    Args:
        directory (str): The output directory.
        train_stats (dict): Dictionary of training metrics.
        epoch (int): Current epoch number.
        n_parameters (int): Total number of trainable parameters.
    """
    if not directory:
        return
    os.makedirs(directory, exist_ok=True)
    payload = {
        **{f"train_{key}": float(value) for key, value in train_stats.items()},
        "epoch": epoch,
        "n_parameters": n_parameters,
    }
    log_path = Path(directory) / "log.txt"
    with open(log_path, "a", encoding="utf-8") as writer:
        writer.write(json.dumps(payload) + "\n")


def _resolve_window_size(data_cfg: DictConfig) -> Tuple[int, int, int]:
    window_size = getattr(data_cfg, "window_size", None)
    if window_size:
        return tuple(window_size)

    time_frame = int(data_cfg.time_frame)
    tubelet_size = int(data_cfg.tubelet_size)
    if time_frame % tubelet_size != 0:
        raise ValueError("time_frame must be divisible by tubelet_size")
    img_patch_size = tuple(int(x) for x in data_cfg.img_patch_size)
    if len(img_patch_size) != 2:
        raise ValueError("img_patch_size must contain H and W sizes")
    image_size = int(data_cfg.image_size)
    if image_size % img_patch_size[0] != 0 or image_size % img_patch_size[1] != 0:
        raise ValueError("image_size must be divisible by img_patch_size")
    window_h = image_size // img_patch_size[0]
    window_w = image_size // img_patch_size[1]
    return (time_frame // tubelet_size, window_h, window_w)


def main(args: argparse.Namespace) -> None:
    """Main function to setup and run the pre-training loop.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
    """
    cfg = load_config(args.config)
    init_distributed_mode(args)
    general_cfg = GeneralArguments(**OmegaConf.to_container(cfg.general, resolve=True))
    device = torch.device(general_cfg.device)

    if general_cfg.output_dir:
        Path(general_cfg.output_dir).mkdir(parents=True, exist_ok=True)
    if general_cfg.log_dir:
        Path(general_cfg.log_dir).mkdir(parents=True, exist_ok=True)

    seed = general_cfg.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    model = build_model(cfg.module)
    img_patch_size = model.shared_encoder.patch_embed.img_patch_size
    tubelet_size = model.shared_encoder.patch_embed.tubelet_size

    assert cfg.data.img_patch_size == img_patch_size, (
        f"Config img_patch_size {cfg.data.img_patch_size} does not match "
        f"model img_patch_size {img_patch_size}."
    )

    assert cfg.data.tubelet_size == tubelet_size, (
        f"Config tubelet_size {cfg.data.tubelet_size} does not match "
        f"model tubelet_size {tubelet_size}."
    )

    patch_size = (tubelet_size, img_patch_size[0], img_patch_size[1])
    logger.info("Patch size = %s", patch_size)

    cfg.data.window_size = _resolve_window_size(cfg.data)
    data_module = CMRDataModule.from_config(cfg)
    dataset_train = data_module.get_train_dataloader()

    batch_size = int(cfg.data.batch_size)
    total_batch_size = batch_size * get_world_size()
    data_loader_train = create_data_loader(
        cfg,
        dataset_train,
        batch_size,
        int(cfg.data.num_workers),
        bool(cfg.data.pin_mem),
    )

    logger.info(
        "Global batch size = %d (%d per GPU x %d GPUs)",
        total_batch_size,
        batch_size,
        get_world_size(),
    )

    log_writer = prepare_log_writer(general_cfg)
    model.to(device)
    optimizer_args, scheduler_args = build_optimizer_arguments(
        cfg, general_cfg, batch_size, total_batch_size
    )
    optimizer = create_optimizer(optimizer_args, model)
    loss_scaler = NativeScaler()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process():
        logger.info("Number of trainable params: %.3f M", n_parameters / 1e6)
        logger.info(
            "Base LR = %.6e, Scaled LR = %.6e",
            float(cfg.module.optimizer_params.lr),
            optimizer_args.lr,
        )

    num_training_steps_per_epoch = max(len(dataset_train) // total_batch_size, 1)

    lr_schedule_values = cosine_scheduler(
        optimizer_args.lr,
        optimizer_args.min_lr,
        scheduler_args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=scheduler_args.warmup_epochs,
        warmup_steps=scheduler_args.warmup_steps,
    )

    wd_end = optimizer_args.weight_decay_end or optimizer_args.weight_decay
    wd_schedule_values = cosine_scheduler(
        optimizer_args.weight_decay,
        wd_end,
        scheduler_args.epochs,
        num_training_steps_per_epoch,
    )

    if is_main_process():
        logger.info(
            "Use step-level LR & WD scheduler. WD in [%.7f, %.7f]",
            float(min(wd_schedule_values)),
            float(max(wd_schedule_values)),
        )

    if general_cfg.resume_training:
        auto_load_model(
            args=optimizer_args,
            model=model,
            model_without_ddp=model,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
        )

    start_time = time.time()
    epochs = scheduler_args.epochs

    for epoch in range(optimizer_args.start_epoch, epochs):
        if get_world_size() > 1:
            data_loader_train.sampler.set_epoch(epoch)

        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            max_norm=optimizer_args.clip_grad,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            patch_size=patch_size,
            normalize_target=scheduler_args.normalize_target,
        )

        if general_cfg.output_dir and is_main_process():
            if (epoch + 1) % general_cfg.save_ckpt_freq == 0 or (epoch + 1) == epochs:
                save_model(
                    args=optimizer_args,
                    model=model,
                    model_without_ddp=model,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )

        if is_main_process():
            save_training_logs(general_cfg.output_dir, train_stats, epoch, n_parameters)
            if log_writer is not None:
                log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time: %s", total_time_str)


if __name__ == "__main__":
    args = get_args()
    cfg_tmp = OmegaConf.load(args.config)
    out_dir = cfg_tmp.general.output_dir
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    main(args)
