import argparse
import os
import sys

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
from einops import rearrange
from omegaconf import OmegaConf

# Add project root to path (3 levels up from downstream_stage/interplane_recon/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from data import unified_datasets
from pretrain_stage.modeling_pretrain import (
    pretrain_multivideomae_tiny_patch4_112,
    pretrain_multivideomae_tiny_patch8_112,
)
from typing import Any, Dict, List, Optional, Tuple


class TubeMaskingGenerator:
    """Generate tube-shaped patch masks for VideoMAE pretraining."""

    def __init__(self, input_size: Tuple[int, int, int], mask_ratio: float) -> None:
        """Initialize the mask generator.

        Args:
            input_size (Tuple[int, int, int]): Volume input geometry sequence (frames, height, width).
            mask_ratio (float): Ratio maps float masking parameter variables limits metrics scalar map.
        """
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __call__(self) -> np.ndarray:
        """Produce the random patch mask arrays masking parameters mapping frames format.

        Returns:
            np.ndarray: Evaluated vector of boolean mapped array components items masking indices elements logic.
        """
        mask_frame = np.hstack(
            [
                np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
                np.ones(self.num_masks_per_frame),
            ]
        )
        np.random.shuffle(mask_frame)
        mask = np.tile(mask_frame, (self.frames, 1)).flatten()
        return mask.astype(np.int64)


def get_args_parser() -> argparse.ArgumentParser:
    """Instantiate command line interface limits mapping string parameter items bounds format values mapping format array values variable options property logic space offset.

    Returns:
        argparse.ArgumentParser: Active console map variable bounds arguments mapping properties target elements parameters.
    """
    parser = argparse.ArgumentParser(
        "Probe Inter-Plane Reconstruction with Pretrain MAE Model", add_help=False
    )
    parser.add_argument(
        "--config",
        default="downstream_stage/interplane_recon/config/interplane_recon.yaml",
        type=str,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to pretrain checkpoint (e.g. checkpoint-299.pth)",
    )
    parser.add_argument(
        "--output_dir", default="downstream_stage/reconstruction_results", type=str
    )
    parser.add_argument(
        "--num_samples",
        default=5,
        type=int,
        help="How many validation samples to save/process",
    )
    parser.add_argument(
        "--viz_freq_frames", default=1, type=int, help="Save every N-th time frame"
    )
    parser.add_argument(
        "--drop_pattern",
        default="alternate",
        type=str,
        help='How to drop slices: "alternate" or a comma-separated list of indices e.g., "1,3,5"',
    )
    parser.add_argument(
        "--mask_type",
        default="slice",
        choices=["slice", "random"],
        help="Type of masking: slice (100% drop) or random (MAE style)",
    )
    parser.add_argument(
        "--mask_ratio", default=0.75, type=float, help="Masking ratio for random type"
    )
    parser.add_argument("--device", default="cuda", type=str)
    return parser


def save_image(img_arr: np.ndarray, save_path: str) -> None:
    """Save given numpy array to an image.

    Args:
        img_arr (np.ndarray): Tensor bounds element targets format values map array.
        save_path (str): File systems metrics offset items maps location limits values variable format element string values location index values limit array points.
    """
    from PIL import Image

    Image.fromarray(img_arr).save(save_path)


def load_weights(model: nn.Module, checkpoint_path: str) -> None:
    """Load neural network parameter definitions and load checkpoint mapped maps variable array metric scale maps limits variable states.

    Args:
        model (nn.Module): Target state logic representation component metric limits bounds space index values mapping values graph map scalar sequence sequence.
        checkpoint_path (str): Location sequence limits bounds scalar scalar index index mapping array elements offset offset mapping location parameters points parameter files mapping value range format limits elements string.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded checkpoint {checkpoint_path}: {msg}")


def expand_patch_mask_to_pixel_mask(
    mask_bn: torch.Tensor, 
    patch_size: Tuple[int, int, int], 
    window_size: Tuple[int, int, int]
) -> torch.Tensor:
    """Reconstruct logical masking indices element sequences back into the initial image parameter resolution logic format metrics.

    Args:
        mask_bn (torch.Tensor): Binary encoded parameter vectors mapping mask arrays metric logical sequences limits bounds space indices tensor.
        patch_size (Tuple[int, int, int]): Size map property target arrays bounds sequences dimension element maps size formatting limit mapping arrays index items tuple parameter array components formats values targets vectors indices offsets tuple.
        window_size (Tuple[int, int, int]): Resolution values targets logic sizes map limits index values properties size values string values mapped formats offset properties maps sizes values properties logic target dimensions targets.

    Returns:
        torch.Tensor: Evaluated value parameters limits points value offset arrays value arrays bounds index values boolean formats logic element maps.
    """
    p0, p1, p2 = patch_size  # (tubelet, ph, pw)
    t, h, w = window_size  # (T', H', W')
    L = mask_bn.shape[0]
    assert L == t * h * w, (L, t * h * w)

    m = mask_bn.view(t, h, w)  # [T', H', W']
    m = m.repeat_interleave(p0, dim=0)  # -> [T, H', W']
    m = m.repeat_interleave(p1, dim=1)  # -> [T, H,  W']
    m = m.repeat_interleave(p2, dim=2)  # -> [T, H,  W]
    return m


def unpatchify(
    x: torch.Tensor, 
    tubelet_size: int, 
    p: int, 
    T: int, 
    H: int, 
    W: int
) -> torch.Tensor:
    """Remap logic linear embedding variables property mapping limits map ranges dimensions bounds sequences index targets scale mappings.

    Args:
        x (torch.Tensor): Sequence value mapping tensors value scale matrix.
        tubelet_size (int): Segment sequence index parameter mapping spacing metrics variable limit lengths bounds sequence offsets values integer.
        p (int): Limit offset target maps formats spacing size string metric format spatial properties properties mapping vectors lengths.
        T (int): Dimension property temporal bounds metric string properties variables size sizes element map formats values scale elements vector index offset dimensions offset limit indices length values sequence values map values parameter items dimensions arrays vectors parameter array logic dimensions integer integer dimension limit targets spatial limit parameters space size mapped sizes parameter values index mapped indices lengths limits matrix sequences sizes offsets scalar format mapping length offset targets mapping metrics formats vector metrics value length limit parameter string length size spacing elements size length scale limits parameter limits scalar vectors vector property targets range limits target index length metrics limit space vectors limit spacing limits scale array index index variable arrays lengths index offsets index dimension.
        H (int): Size lengths dimension bounds value map.
        W (int): Value mapping format length vector width values scale.

    Returns:
        torch.Tensor: Target spatial dimensions scale logic length offset sequence mapping limit dimension metric targets mapped scalar items logic.
    """
    T_p = T // tubelet_size
    H_p = H // p
    W_p = W // p

    # Matching main_visualization.py logic
    x = rearrange(
        x,
        "bs (t h w) (p0 p1 p2 c) -> bs c (t p0) (h p1) (w p2)",
        p0=tubelet_size,
        p1=p,
        p2=p,
        t=T_p,
        h=H_p,
        w=W_p,
    )
    return x.squeeze(1)  # [BS, T, H, W]


def calc_image_metrics(
    pred: np.ndarray, 
    target: np.ndarray
) -> Tuple[float, float]:
    """Calculate and log model mapping image space similarity and structure quality metric metrics metrics index property elements mapping parameters target arrays. Uses SSIM targets logic string limit index values.

    Args:
        pred (np.ndarray): The map array of the index components mapped properties dimension scalar limits predicted string properties elements.
        target (np.ndarray): Values reference bounds value offset target arrays value parameter metric index limits targets targets matrix strings offsets values arrays properties formats vector target limit dimensions scale logic metric limits offset limits index space strings map.

    Returns:
        Tuple[float, float]: Offset logic property mappings index spacing arrays sizes vector elements format parameters mapping range limits target mapped value components limits sequences format map sizes lengths length variable mapping variable value arrays logic properties logic. (PSNR, SSIM).
    """
    assert pred.shape == target.shape

    p_val = psnr(target, pred, data_range=1.0)
    # Use win_size=3 since image is 112x112 and could throw errors for large defaults
    s_val = ssim(target, pred, data_range=1.0, win_size=3)

    return p_val, s_val


def main(args: argparse.Namespace) -> None:
    """Execute main interplane reconstruction pipeline logic maps parameters evaluation.

    Args:
        args (argparse.Namespace): System mapping parameters evaluation properties limit objects configuration variables index format strings parameters index variables items mapping bounds property logic.
    """
    config = OmegaConf.load(args.config)
    device = torch.device(args.device)

    # Load Data (No Segments, just raw images for reconstruction)
    print(f"Loading Val Data from {config.data.processed_dir}")
    dataset_cls_name = config.data.dataset_cls
    if not dataset_cls_name.endswith("_Test"):
        dataset_cls_name = dataset_cls_name + "_Test"
    dataset_cls = getattr(unified_datasets, dataset_cls_name)
    data_args = dict(
        load_seg=False,
        mask_ratio=0.0,
        window_size=(
            (config.data.tubelet_size, *config.data.img_patch_size)
            if hasattr(config.data, "img_patch_size")
            else None
        ),
    )

    data_root = Path(config.data.processed_dir)
    all_files = sorted(list(data_root.glob("*.npz")))
    if len(all_files) == 0:
        all_files = sorted(list(data_root.rglob("*.npz")))

    num_train = config.data.num_train
    num_val = config.data.num_val
    val_files = all_files[num_train : num_train + num_val]

    if len(val_files) > args.num_samples:
        val_files = val_files[: args.num_samples]

    dataset_val = dataset_cls(val_files, **data_args)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False
    )

    # Calculate num_slices for model creation
    if "AllAX" in config.data.dataset_cls:
        model_slices = config.data.sax_slice_num + 3
        sax_slices = config.data.sax_slice_num
        lax_slices = 3
    elif "LAX" in config.data.dataset_cls and "SAX" not in config.data.dataset_cls:
        model_slices = 3
        sax_slices = 0
        lax_slices = 3
    else:
        model_slices = config.data.sax_slice_num
        sax_slices = config.data.sax_slice_num
        lax_slices = 0

    # Determine which slices to drop based on args
    if args.drop_pattern == "alternate":
        drop_indices = [i for i in range(sax_slices) if i % 2 == 0]
    else:
        drop_indices = [
            int(x.strip()) for x in args.drop_pattern.split(",") if x.strip().isdigit()
        ]

    print(
        f"Targeting image reconstruction... Dropped SAX Slices will be: {drop_indices}"
    )

    print(f"Creating model: {config.module.module_name}")
    if "patch4" in config.module.module_name:
        model_builder = pretrain_multivideomae_tiny_patch4_112
    else:
        model_builder = pretrain_multivideomae_tiny_patch8_112

    model = model_builder()
    model.to(device)
    model.eval()

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "input_gt"
    masked_dir = output_dir / "input_masked"
    pred_dir = output_dir / "recon_pred"

    for d in [raw_dir, masked_dir, pred_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load FINETUNED or PRETRAINED checkpoints
    load_weights(model, args.checkpoint)

    total_psnr = []
    total_ssim = []

    # Masking Setup
    T_p = config.data.time_frame // config.data.tubelet_size
    img_p = 4 if "patch4" in config.module.module_name else 8
    H_p = config.data.image_size // img_p
    W_p = config.data.image_size // img_p

    mask_gen = TubeMaskingGenerator((T_p, H_p, W_p), args.mask_ratio)

    print(f"Running Inference & Image Metrics extraction (Mode: {args.mask_type})...")
    with torch.no_grad():
        for i, batch in enumerate(data_loader_val):
            images_orig = batch[0]  # [1, S, T, H, W]

            # When load_seg=False and using Cardiac3DplusTAllAX, batch has 5 items:
            # im_data, masks, spatial_coords, patch_coords_3d, index
            spatial_coords = None
            patch_coords = None
            if len(batch) >= 5:
                spatial_coords = batch[2]
                patch_coords = batch[3]

            S, T, H, W = images_orig[0].shape

            N_patches = T_p * H_p * W_p
            if args.mask_type == "slice":
                # Slice dropping logic (100% mask for specific slices)
                bool_masked_pos = torch.zeros(
                    1, S, N_patches, dtype=torch.bool, device=device
                )
                images_gpu = images_orig.clone().to(device)
                for s_idx in drop_indices:
                    if s_idx < S:
                        bool_masked_pos[0, s_idx, :] = True
                        images_gpu[0, s_idx, ...] = 0.0  # Zero out pixels
            else:
                # Random patch masking logic (75% random mask)
                bool_masked_pos = torch.zeros(
                    1, S, N_patches, dtype=torch.bool, device=device
                )
                images_gpu = images_orig.clone().to(device)
                for s_idx in range(S):
                    m = torch.from_numpy(mask_gen()).to(device)
                    bool_masked_pos[0, s_idx] = m.bool()

                    # Also mask the input pixels to mimic VideoMAE pretraining
                    p_mask_pix = expand_patch_mask_to_pixel_mask(
                        m, (config.data.tubelet_size, img_p, img_p), (T_p, H_p, W_p)
                    )
                    images_gpu[0, s_idx] *= 1.0 - p_mask_pix.to(device)

            mask_list = [bool_masked_pos[:, s_idx, :] for s_idx in range(S)]
            pc_gpu = patch_coords.to(device) if patch_coords is not None else None
            sc_gpu = spatial_coords.to(device) if spatial_coords is not None else None

            import time

            t0 = time.time()

            print(f"Sample {i}: Forwarding Pretrain Model...", end=" ")
            # Forward returns: dict with 'full_recon'
            preds_out = model(
                images_gpu, mask_list, spatial_coords=sc_gpu, patch_coords_3d=pc_gpu
            )
            if isinstance(preds_out, dict):
                # full_recon is [B, S, N_patches, C]
                logits_bs = preds_out["full_recon"]
                logits = logits_bs.reshape(
                    -1, logits_bs.shape[2], logits_bs.shape[3]
                )  # [B*S, N, C]
            else:
                logits = preds_out

            # Decoder returns: [B*S, N_patches, pixel_dim]
            # Convert back to [B, S, T, H, W]
            recon_flat = unpatchify(logits, config.data.tubelet_size, img_p, T, H, W)
            recon = recon_flat.view(1, S, T, H, W).cpu()

            print(f"Elapsed: {time.time()-t0:.2f}s")

            # Calculate mean statistics from the original image for normalization standard
            mean = images_orig.mean(dim=(2, 3, 4), keepdim=True)
            var = images_orig.var(dim=(2, 3, 4), keepdim=True)

            # Reverse normalization (if images_orig were normalized in dataset)
            # Assuming standard zero-mean unit-variance in dataset [0, 1] usually
            # We clip values to [0, 1] for metrics
            targets = torch.clamp(images_orig, 0, 1).cpu()
            preds = torch.clamp(recon, 0, 1).cpu()

            # --- Metrics ONLY on the dropped slices ---
            sample_psnrs = []
            sample_ssims = []

            for s_idx in drop_indices:
                if s_idx >= S:
                    continue
                for t in range(T):
                    p = preds[0, s_idx, t].numpy()
                    tg = targets[0, s_idx, t].numpy()

                    p_val, s_val = calc_image_metrics(p, tg)

                    sample_psnrs.append(p_val)
                    sample_ssims.append(s_val)

            if len(sample_psnrs) > 0:
                mean_p = np.mean(sample_psnrs)
                mean_s = np.mean(sample_ssims)

                total_psnr.append(mean_p)
                total_ssim.append(mean_s)

                print(
                    f"Sample {i}: PSNR={mean_p:.2f}, SSIM={mean_s:.4f} (on dropped slices {drop_indices})"
                )

            # --- Visualization ---
            sample_raw_dir = raw_dir / f"sample_{i}"
            sample_masked_dir = masked_dir / f"sample_{i}"
            sample_pred_dir = pred_dir / f"sample_{i}"

            sample_raw_dir.mkdir(exist_ok=True)
            sample_masked_dir.mkdir(exist_ok=True)
            sample_pred_dir.mkdir(exist_ok=True)

            # Composition: Final Recon = Mask * Prediction + (1-Mask) * Original
            # Create pixel-level mask from boolean mask_list
            # bool_masked_pos is [1, S, N_patches]
            window_size = (T_p, H_p, W_p)
            patch_size_3d = (config.data.tubelet_size, img_p, img_p)

            pixel_mask_list = []
            for s in range(S):
                # expand_patch_mask_to_pixel_mask returns [T, H, W]
                m_pix = expand_patch_mask_to_pixel_mask(
                    bool_masked_pos[0, s], patch_size_3d, window_size
                )
                pixel_mask_list.append(m_pix)

            # [S, T, H, W]
            pixel_mask = torch.stack(pixel_mask_list, dim=0).cpu().float()

            # Reconstruction Composition
            # recon: [1, S, T, H, W], targets: [1, S, T, H, W]
            recon_final = preds[0] * pixel_mask + targets[0] * (1.0 - pixel_mask)

            for t in range(0, T, args.viz_freq_frames):
                for s in range(S):
                    # Default orientation (aligned with main_visualization.py)
                    gt_np = targets[0, s, t].numpy()
                    pr_np = recon_final[s, t].numpy()
                    ms_np = targets[0, s, t].numpy() * (1.0 - pixel_mask[s, t].numpy())

                    # Convert [0, 1] float to [0, 255] uint8 grayscale
                    gt_img = (gt_np * 255).clip(0, 255).astype(np.uint8)
                    pr_img = (pr_np * 255).clip(0, 255).astype(np.uint8)
                    ms_img = (ms_np * 255).clip(0, 255).astype(np.uint8)

                    # Save GT
                    save_image(
                        np.stack([gt_img] * 3, axis=-1),
                        sample_raw_dir / f"t{t:02d}_s{s:02d}.png",
                    )
                    # Save Masked Input
                    save_image(
                        np.stack([ms_img] * 3, axis=-1),
                        sample_masked_dir / f"t{t:02d}_s{s:02d}.png",
                    )
                    # Save Recon (Composited)
                    save_image(
                        np.stack([pr_img] * 3, axis=-1),
                        sample_pred_dir / f"t{t:02d}_s{s:02d}.png",
                    )

    print("\n" + "=" * 50)
    print("--- Inter-Plane Image Reconstruction Results (Over Dropped Slices) ---")
    print(f"Mean PSNR: {np.mean(total_psnr):.2f}")
    print(f"Mean SSIM: {np.mean(total_ssim):.4f}")
    print("=" * 50)
    print(f"[Finished] All visual results saved to {output_dir}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
