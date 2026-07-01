import datetime
import io
import json
import math
import os
import random
import subprocess
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter
from timm.utils import get_state_dict
from torch import inf
from torch.utils.data._utils.collate import default_collate


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size: int = 20, fmt: Optional[str] = None) -> None:
        """Initialize the SmoothedValue object.

        Args:
            window_size (int, optional): Size of the tracking window. Defaults to 20.
            fmt (str, optional): Format string for printing. Defaults to None.
        """
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque: deque = deque(maxlen=window_size)
        self.total: float = 0.0
        self.count: int = 0
        self.fmt: str = fmt

    def update(self, value: float, n: int = 1) -> None:
        """Update the tracker with a new value.

        Args:
            value (float): The value to add.
            n (int, optional): The weight/count of the value. Defaults to 1.
        """
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self) -> None:
        """Synchronize the count and total values across all distributed processes.
        
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist() # type: ignore
        self.count = int(t[0])
        self.total = float(t[1])

    @property
    def median(self) -> float:
        """Return the median of the current window."""
        d = torch.tensor(list(self.deque))
        return d.median().item() # type: ignore

    @property
    def avg(self) -> float:
        """Return the mean of the current window."""
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item() # type: ignore

    @property
    def global_avg(self) -> float:
        """Return the global average over all updates."""
        if self.count == 0:
            return 0.0
        return self.total / self.count

    @property
    def max(self) -> float:
        """Return the maximum value in the current window."""
        return max(self.deque)

    @property
    def value(self) -> float:
        """Return the most recently added value."""
        return self.deque[-1]

    def __str__(self) -> str:
        """String representation using the format string."""
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    """Logger for tracking multiple metrics using SmoothedValue."""

    def __init__(self, delimiter: str = "\t") -> None:
        """Initialize MetricLogger.

        Args:
            delimiter (str, optional): String to separate metrics during printing. Defaults to "\t".
        """
        self.meters: defaultdict = defaultdict(SmoothedValue)
        self.delimiter: str = delimiter

    def update(self, **kwargs: Any) -> None:
        """Update metrics given in kwargs.

        Args:
            **kwargs: Metric names and their corresponding float/Tensor values.
        """
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            if hasattr(v, "item"):
                v = v.item()
            if not isinstance(v, (float, int)):
                try:
                    v = float(v)
                except (ValueError, TypeError):
                    pass
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr: str) -> Any:
        """Allow dot notation access to meters."""
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self) -> str:
        """Return formatted string of all tracked metrics."""
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self) -> None:
        """Synchronize all tracked metrics across distributed processes."""
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name: str, meter: SmoothedValue) -> None:
        """Explicitly add a custom meter.

        Args:
            name (str): Name of the metric.
            meter (SmoothedValue): The meter object.
        """
        self.meters[name] = meter

    def log_every(self, iterable: Iterable, print_freq: int, header: Optional[str] = None) -> Iterable:
        """Generator that logs progress through an iterable.

        Args:
            iterable (Iterable): The collection to iterate over.
            print_freq (int): The frequency (in steps) to print logs.
            header (str, optional): A prefix for the log lines. Defaults to None.

        Yields:
            The items from the iterable.
        """
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        
        length = len(iterable) if hasattr(iterable, "__len__") else 0
        space_fmt = ":" + str(len(str(length))) + "d" if length > 0 else ":d"
        
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]" if length > 0 else "[{0" + space_fmt + "}]",
            "eta: {eta}" if length > 0 else "",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        log_msg = [m for m in log_msg if m] # Remove empty strings
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        
        log_msg_str = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or (length > 0 and i == length - 1):
                if length > 0:
                    eta_seconds = iter_time.global_avg * (length - i)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                else:
                    eta_string = "N/A"
                    
                metrics = {
                    "i": i,
                    "length": length if length > 0 else i,
                    "eta": eta_string,
                    "meters": str(self),
                    "time": str(iter_time),
                    "data": str(data_time),
                }
                
                if torch.cuda.is_available():
                    metrics["memory"] = torch.cuda.max_memory_allocated() / MB
                    
                # We format manually to avoid KeyError for missing kwargs depending on structure
                print(log_msg_str.format(*[metrics.get("i", i), metrics.get("length", length)], **metrics))
                
            i += 1
            end = time.time()
            
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / (i if i > 0 else 1)
            )
        )


class TensorboardLogger(object):
    """Wrapper for Tensorboard SummaryWriter."""

    def __init__(self, log_dir: str) -> None:
        """Initialize the TensorboardLogger.

        Args:
            log_dir (str): Directory where the logs will be saved.
        """
        self.writer = SummaryWriter(logdir=log_dir)
        self.step: int = 0

    def set_step(self, step: Optional[int] = None) -> None:
        """Set the current step.

        Args:
            step (int, optional): The step to set. If None, increments the current step.
        """
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head: str = "scalar", step: Optional[int] = None, **kwargs: Any) -> None:
        """Log scalars to Tensorboard.

        Args:
            head (str, optional): The prefix for the logged metric. Defaults to "scalar".
            step (int, optional): Specific step to log at. Defaults to None.
            **kwargs: Dictionary of metrics.
        """
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(
                head + "/" + k, v, self.step if step is None else step
            )

    def flush(self) -> None:
        """Flush the summary writer."""
        self.writer.flush()


def seed_worker(worker_id: int) -> None:
    """Set random seed for a DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _load_checkpoint_for_ema(model_ema: Any, checkpoint: Dict[str, Any]) -> None:
    """Workaround for ModelEma._load_checkpoint to accept an already-loaded object.

    Args:
        model_ema (Any): The Exponential Moving Average model instance.
        checkpoint (dict): The loaded checkpoint dictionary.
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master: bool) -> None:
    """Disable printing when not in the master process.

    Args:
        is_master (bool): Whether the current process is the master process.
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args: Any, **kwargs: Any) -> None:
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized() -> bool:
    """Check if distributed mode is completely initialized."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    """Get the current world size (number of distributed processes)."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get the rank of the current process."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    """Check if the current process is the main (rank 0) process."""
    return get_rank() == 0


def save_on_master(*args: Any, **kwargs: Any) -> None:
    """Save objects via torch.save only on the master process."""
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args: Any) -> None:
    """Initialize distributed training mode based on environment variables.

    Args:
        args (argparse.Namespace): Arguments parsed from command line containing 
            fields that will be populated with distributed properties.
    """
    if getattr(args, "dist_on_itp", False):
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.gpu = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        args.dist_url = "tcp://%s:%s" % (
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
        )
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
    # Prefer torchrun-provided env vars when both torchrun and Slurm vars exist.
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = int(os.environ["SLURM_LOCALID"])
        args.world_size = int(os.environ["SLURM_NTASKS"])
        os.environ["RANK"] = str(args.rank)
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["WORLD_SIZE"] = str(args.world_size)

        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl" if torch.cuda.is_available() else "gloo"
    print(
        "| distributed init (rank {}): {}, gpu {}".format(
            args.rank, args.dist_url, args.gpu
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def load_state_dict(
    model: torch.nn.Module, 
    state_dict: Dict[str, Any], 
    prefix: str = "", 
    ignore_missing: str = "relative_position_index"
) -> None:
    """Load state_dict into a model with custom prefix and ignore filters for missing keys.

    Args:
        model (nn.Module): The model to load weights into.
        state_dict (Dict[str, Any]): The state dictionary with weights.
        prefix (str, optional): A prefix to add/expect for the state_dict keys. Defaults to "".
        ignore_missing (str, optional): Pipe-separated keywords to ignore if missing. 
            Defaults to "relative_position_index".
    """
    missing_keys: List[str] = []
    unexpected_keys: List[str] = []
    error_msgs: List[str] = []
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module: torch.nn.Module, prefix: str = "") -> None:
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split("|"):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print(
            "Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys
            )
        )
    if len(unexpected_keys) > 0:
        print(
            "Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys
            )
        )
    if len(ignore_missing_keys) > 0:
        print(
            "Ignored weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, ignore_missing_keys
            )
        )
    if len(error_msgs) > 0:
        print("\n".join(error_msgs))


class NativeScalerWithGradNormCount:
    """Automatic Mixed Precision (AMP) gradient scaler tracking gradient norms."""
    state_dict_key: str = "amp_scaler"

    def __init__(self) -> None:
        """Initialize the AMP GradScaler."""
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        clip_grad: Optional[float] = None,
        parameters: Optional[Union[Iterable[torch.nn.Parameter], torch.Tensor]] = None,
        create_graph: bool = False,
        update_grad: bool = True,
    ) -> Optional[torch.Tensor]:
        """Scale loss, run backward pass, and step optimizer.

        Args:
            loss (Tensor): The loss to scale and backpropagate.
            optimizer (Optimizer): The optimizer to step.
            clip_grad (float, optional): Maximum allowed gradient norm. Defaults to None.
            parameters (Iterable, optional): Parameters for gradient clipping/norm calculation. Defaults to None.
            create_graph (bool, optional): Create graph for higher-order derivatives. Defaults to False.
            update_grad (bool, optional): If True, step the optimizer. Defaults to True.

        Returns:
            Optional[Tensor]: The calculated parameter gradient norm.
        """
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self) -> Dict[str, Any]:
        """Get the state dict of the underlying scaler."""
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the state dict into the underlying scaler."""
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(
    parameters: Union[Iterable[torch.nn.Parameter], torch.Tensor], 
    norm_type: float = 2.0
) -> torch.Tensor:
    """Compute the gradient norm of an iterable of parameters.

    Args:
        parameters (Iterable[Tensor] or Tensor): An iterable of Tensors or a single Tensor.
        norm_type (float, optional): The type of the used p-norm. Defaults to 2.0.

    Returns:
        Tensor: Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    niter_per_ep: int,
    warmup_epochs: int = 0,
    start_warmup_value: float = 0,
    warmup_steps: int = -1,
) -> np.ndarray:
    """Create a cosine annealing scheduler with optional linear warmup.

    Args:
        base_value (float): Initial value after warmup.
        final_value (float): Final value at the end of the schedule.
        epochs (int): Total number of epochs.
        niter_per_ep (int): Number of iterations per epoch.
        warmup_epochs (int, optional): Number of warmup epochs. Defaults to 0.
        start_warmup_value (float, optional): Initial warmup value. Defaults to 0.
        warmup_steps (int, optional): Absolute number of warmup steps (overrides warmup_epochs). Defaults to -1.

    Returns:
        np.ndarray: Scheduled values over all iterations.
    """
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [
            final_value
            + 0.5
            * (base_value - final_value)
            * (1 + math.cos(math.pi * i / (len(iters))))
            for i in iters
        ]
    )

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_model(
    args: Any,
    epoch: int,
    model: torch.nn.Module,
    model_without_ddp: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_scaler: Optional[NativeScalerWithGradNormCount],
    model_ema: Optional[Any] = None,
) -> None:
    """Save the model checkpoint, optimizer state, and scaler.

    Args:
        args (argparse.Namespace): Arguments containing output_dir and other config.
        epoch (int): Current epoch number.
        model (nn.Module): Distributed model.
        model_without_ddp (nn.Module): Underlying model without DDP wrapper.
        optimizer (Optimizer): The optimizer object.
        loss_scaler (NativeScalerWithGradNormCount, optional): The mixed-precision scaler.
        model_ema (Any, optional): Exponential Moving Average model representation. Defaults to None.
    """
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ("checkpoint-%s.pth" % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "scaler": loss_scaler.state_dict(),
            }

            if model_ema is not None:
                to_save["model_ema"] = get_state_dict(model_ema)

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {"epoch": epoch}
        if model_ema is not None:
            client_state["model_ema"] = get_state_dict(model_ema)
        model.save_checkpoint( # type: ignore
            save_dir=args.output_dir,
            tag="checkpoint-%s" % epoch_name,
            client_state=client_state,
        )


def auto_load_model(
    args: Any,
    model: torch.nn.Module,
    model_without_ddp: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_scaler: Optional[NativeScalerWithGradNormCount],
    model_ema: Optional[Any] = None,
) -> None:
    """Automatically discover and load the latest checkpoint from args.output_dir.

    Args:
        args (argparse.Namespace): Arguments containing tracking config.
        model (nn.Module): The DDP-wrapped model.
        model_without_ddp (nn.Module): The underlying core base model.
        optimizer (Optimizer): The active optimizer.
        loss_scaler (NativeScalerWithGradNormCount, optional): The mixed precision scaler.
        model_ema (Any, optional): The exponential moving average model track. Defaults to None.
    """
    output_dir = Path(args.output_dir)
    if loss_scaler is not None:
        # torch.amp
        if getattr(args, "auto_resume", False) and len(args.resume) == 0:
            import glob

            all_checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*.pth"))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split("-")[-1].split(".")[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(
                    output_dir, "checkpoint-%d.pth" % latest_ckpt
                )
            print("Auto resume checkpoint: %s" % args.resume)

        if hasattr(args, "resume") and args.resume:
            if args.resume.startswith("https"):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location="cpu", check_hash=True
                )
            else:
                checkpoint = torch.load(
                    args.resume, weights_only=False, map_location="cpu"
                )
            model_without_ddp.load_state_dict(checkpoint["model"])
            print("Resume checkpoint %s" % args.resume)
            if "optimizer" in checkpoint and "epoch" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
                args.start_epoch = checkpoint["epoch"] + 1
                if hasattr(args, "model_ema") and args.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint["model_ema"])
                if "scaler" in checkpoint:
                    loss_scaler.load_state_dict(checkpoint["scaler"])
                print("With optim & sched!")
    else:
        # deepspeed, only support '--auto_resume'.
        if getattr(args, "auto_resume", False):
            import glob

            all_checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split("-")[-1].split(".")[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, "checkpoint-%d" % latest_ckpt)
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint( # type: ignore
                    args.output_dir, tag="checkpoint-%d" % latest_ckpt
                )
                args.start_epoch = client_states["epoch"] + 1
                if model_ema is not None:
                    if getattr(args, "model_ema", False):
                        _load_checkpoint_for_ema(model_ema, client_states["model_ema"])


def create_ds_config(args: Any) -> None:
    """Create a deepspeed configuration JSON dynamically from parsed args.

    Args:
        args (argparse.Namespace): The parsed runtime definitions.
    """
    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                },
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128,
            },
        }

        writer.write(json.dumps(ds_config, indent=2))


def multiple_samples_collate(batch: Iterable, fold: bool = False) -> Union[Tuple[List[torch.Tensor], List[torch.Tensor], List[int], List[Any]], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Collate function for repeated augmentation handling multiple samples.

    Args:
        batch (tuple or list): data batch to collate. Each item is a batch from repeated augmentation.
        fold (bool, optional): Whether to wrap inputs in a list. Defaults to False.

    Returns:
        tuple | list: collated data batch (inputs, labels, video_idxs, extra_datas).
    """
    inputs, labels, video_idx, extra_data = zip(*batch) # type: ignore
    
    # Flatten inner chunks
    inputs_flat = [item for sublist in inputs for item in sublist]
    labels_flat = [item for sublist in labels for item in sublist]
    video_idx_flat = [item for sublist in video_idx for item in sublist]
    
    # Collate flats
    inputs_col: torch.Tensor = default_collate(inputs_flat)
    labels_col: torch.Tensor = default_collate(labels_flat)
    video_idx_col: torch.Tensor = default_collate(video_idx_flat)
    extra_data_col = default_collate(extra_data)

    if fold:
        return [inputs_col], labels_col, video_idx_col, extra_data_col # type: ignore
    else:
        return inputs_col, labels_col, video_idx_col, extra_data_col
