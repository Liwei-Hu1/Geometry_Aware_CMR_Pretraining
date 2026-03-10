import json
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP
from torch import optim as optim
from torch.optim import Optimizer

try:
    from apex.optimizers import FusedAdam, FusedLAMB, FusedNovoGrad, FusedSGD

    has_apex = True
except ImportError:
    has_apex = False


def get_num_layer_for_vit(var_name: str, num_max_layer: int) -> int:
    """Get the layer ID for a Vision Transformer variable.

    Args:
        var_name (str): Name of the variable.
        num_max_layer (int): Total number of layers.

    Returns:
        int: The computed layer ID.
    """
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split(".")[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    """Assigns layer-wise learning rate decay values."""

    def __init__(self, values: Sequence[float]) -> None:
        """Initialize the assigner with decay values.

        Args:
            values (Sequence[float]): List of decay values.
        """
        self.values = values

    def get_scale(self, layer_id: int) -> float:
        """Get the scale for a specific layer.

        Args:
            layer_id (int): The ID of the layer.

        Returns:
            float: The scale value.
        """
        return self.values[layer_id]

    def get_layer_id(self, var_name: str) -> int:
        """Get the layer ID corresponding to a variable name.

        Args:
            var_name (str): Name of the variable.

        Returns:
            int: The computed layer ID.
        """
        return get_num_layer_for_vit(var_name, len(self.values))


def get_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float = 1e-5,
    skip_list: Iterable[str] = (),
    get_num_layer: Optional[Callable[[str], int]] = None,
    get_layer_scale: Optional[Callable[[int], float]] = None,
) -> List[Dict[str, Any]]:
    """Group model parameters for optimization, applying weight decay and layer-wise learning rate scales.

    Args:
        model (torch.nn.Module): The model containing parameters to optimize.
        weight_decay (float, optional): Weight decay value. Defaults to 1e-5.
        skip_list (Iterable[str], optional): List of parameter names to skip decay. Defaults to ().
        get_num_layer (Optional[Callable[[str], int]], optional): Function to get layer ID. Defaults to None.
        get_layer_scale (Optional[Callable[[int], float]], optional): Function to get layer scale. Defaults to None.

    Returns:
        List[Dict[str, Any]]: List of parameter groups configured for the optimizer.
    """
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.0
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.0

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(
    args: Any,
    model: torch.nn.Module,
    get_num_layer: Optional[Callable[[str], int]] = None,
    get_layer_scale: Optional[Callable[[int], float]] = None,
    filter_bias_and_bn: bool = True,
    skip_list: Optional[Iterable[str]] = None,
) -> Optimizer:
    """Create an optimizer based on specified arguments.

    Args:
        args (Any): Parsed arguments containing optimizer configuration (e.g., args.opt, args.lr, args.weight_decay).
        model (torch.nn.Module): Model to be optimized.
        get_num_layer (Optional[Callable[[str], int]], optional): Function mapping variable name to layer ID. Defaults to None.
        get_layer_scale (Optional[Callable[[int], float]], optional): Function mapping layer ID to learning rate scale. Defaults to None.
        filter_bias_and_bn (bool, optional): Whether to skip weight decay for biases and batch norm weights. Defaults to True.
        skip_list (Optional[Iterable[str]], optional): Explicit list of variables to skip weight decay. Defaults to None.

    Returns:
        Optimizer: The instantiated PyTorch optimizer.
    """
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(
            model, weight_decay, skip, get_num_layer, get_layer_scale
        )
        weight_decay = 0.0
    else:
        parameters = model.parameters()

    if "fused" in opt_lower:
        assert (
            has_apex and torch.cuda.is_available()
        ), "APEX and CUDA required for fused optimizers"

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, "opt_eps") and args.opt_eps is not None:
        opt_args["eps"] = args.opt_eps
    if hasattr(args, "opt_betas") and args.opt_betas is not None:
        opt_args["betas"] = args.opt_betas

    print("optimizer settings:", opt_args)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "sgd" or opt_lower == "nesterov":
        opt_args.pop("eps", None)
        optimizer = optim.SGD(
            parameters, momentum=args.momentum, nesterov=True, **opt_args
        )
    elif opt_lower == "momentum":
        opt_args.pop("eps", None)
        optimizer = optim.SGD(
            parameters, momentum=args.momentum, nesterov=False, **opt_args
        )
    elif opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adamp":
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == "sgdp":
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "adafactor":
        if not args.lr:
            opt_args["lr"] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == "adahessian":
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == "rmsprop":
        optimizer = optim.RMSprop(
            parameters, alpha=0.9, momentum=args.momentum, **opt_args
        )
    elif opt_lower == "rmsproptf":
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == "nvnovograd":
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == "fusedsgd":
        opt_args.pop("eps", None)
        optimizer = FusedSGD(
            parameters, momentum=args.momentum, nesterov=True, **opt_args
        )
    elif opt_lower == "fusedmomentum":
        opt_args.pop("eps", None)
        optimizer = FusedSGD(
            parameters, momentum=args.momentum, nesterov=False, **opt_args
        )
    elif opt_lower == "fusedadam":
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == "fusedadamw":
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == "fusedlamb":
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == "fusednovograd":
        opt_args.setdefault("betas", (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == "lookahead":
            optimizer = Lookahead(optimizer)

    return optimizer
