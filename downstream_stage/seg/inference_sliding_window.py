import os
import sys
import torch
import numpy as np
import nibabel as nib
import datetime
from pathlib import Path
from tqdm import tqdm
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.unified_datasets import image_normalization
from downstream_stage.probing.modeling_finetune import create_finetune_model


def calculate_dice(pred, target, num_classes=6):
    """
    pred, target: [H, W, S, T] or other compatible shapes
    Returns: list of dice scores for each foreground class.
    """
    dice_scores = []
    for i in range(1, num_classes):
        p = (pred == i).astype(np.float32)
        t = (target == i).astype(np.float32)
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        if union == 0:
            dice_scores.append(1.0)
        else:
            dice_scores.append(2.0 * intersection / union)
    return dice_scores


def get_patch_coords_3d(spatial_coords, pixel_spacing, cropped_size=112, patch_hw=4):
    # spatial_coords: [S, 9], pixel_spacing: [S, 2]
    # Per-subject normalization (for the current window)
    pos = spatial_coords[:, :3].copy()
    pos_mean = pos.mean(axis=0, keepdims=True)
    pos_std = pos.std(axis=0, keepdims=True).clip(min=1e-6)
    spatial_coords_norm = spatial_coords.copy()
    spatial_coords_norm[:, :3] = (pos - pos_mean) / pos_std

    n_patches_h = cropped_size // patch_hw
    n_patches_w = cropped_size // patch_hw
    N_spatial_patches = n_patches_h * n_patches_w

    S = spatial_coords.shape[0]
    patch_coords_3d = np.zeros((S, N_spatial_patches, 3), dtype=np.float32)

    for s in range(S):
        ipp = spatial_coords_norm[s, :3]
        row_dir = spatial_coords_norm[s, 3:6]
        col_dir = spatial_coords_norm[s, 6:9]
        ps_row = pixel_spacing[s, 0]
        ps_col = pixel_spacing[s, 1]

        pi = np.arange(n_patches_h) * patch_hw + patch_hw / 2
        pj = np.arange(n_patches_w) * patch_hw + patch_hw / 2
        pj_grid, pi_grid = np.meshgrid(pj, pi)
        pj_flat = pj_grid.ravel()
        pi_flat = pi_grid.ravel()

        offset_x = pj_flat * ps_col * row_dir[0] + pi_flat * ps_row * col_dir[0]
        offset_y = pj_flat * ps_col * row_dir[1] + pi_flat * ps_row * col_dir[1]
        offset_z = pj_flat * ps_col * row_dir[2] + pi_flat * ps_row * col_dir[2]

        patch_coords_3d[s, :, 0] = ipp[0] + offset_x / pos_std[0, 0]
        patch_coords_3d[s, :, 1] = ipp[1] + offset_y / pos_std[0, 1]
        patch_coords_3d[s, :, 2] = ipp[2] + offset_z / pos_std[0, 2]

    return patch_coords_3d


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = args.data_path
    print(f"Loading data from {data_path} ...")
    data = np.load(data_path)

    sax_im_data = data["sax"].astype(np.float32)
    lax_im_data = data["lax"].astype(np.float32)
    print(f"Original SAX shape: {sax_im_data.shape}")
    print(f"Original LAX shape: {lax_im_data.shape}")

    H, W, S_sax_total, T_total = sax_im_data.shape

    # Load Ground Truth if available
    seg_sax_gt = None
    seg_lax_gt = None
    if "seg_sax" in data:
        seg_sax_gt = data["seg_sax"].astype(np.uint8)
    if "seg_lax" in data:
        seg_lax_gt = data["seg_lax"].astype(np.uint8)

        # Apply relabeling logic to LAX GT (following training logic)
        # 2ch: label 3 -> 4
        seg_lax_gt[..., 0, :][seg_lax_gt[..., 0, :] == 3] = 4
        # 3ch: label 5 -> 0
        seg_lax_gt[..., 1, :][seg_lax_gt[..., 1, :] == 5] = 0

    # spatial_coords and pixel_spacing
    if "spatial_coords_cropped" in data:
        all_spatial_coords = data["spatial_coords_cropped"].astype(np.float32)
    elif "spatial_coords" in data:
        all_spatial_coords = data["spatial_coords"].astype(np.float32)
    else:
        all_spatial_coords = np.zeros((S_sax_total + 3, 9), dtype=np.float32)

    if "pixel_spacing" in data:
        all_pixel_spacing = data["pixel_spacing"].astype(np.float32)
    else:
        all_pixel_spacing = np.ones((S_sax_total + 3, 2), dtype=np.float32)

    all_sax_spatial = all_spatial_coords[:S_sax_total]
    lax_spatial = all_spatial_coords[S_sax_total:]

    all_sax_pixel_spacing = all_pixel_spacing[:S_sax_total]
    lax_pixel_spacing = all_pixel_spacing[S_sax_total:]

    # 2. Setup Model
    print(f"Initializing model with decoder: {args.decoder} ...")
    if args.decoder == "unetr":
        backbone_name = "pretrain_multivideomae_tiny_patch4_112"
    elif args.decoder == "volseg":
        backbone_name = "seg_lite_pretrain_multivideomae_tiny_patch4_112"
    else:
        raise ValueError(f"Unknown decoder: {args.decoder}. Choose 'unetr' or 'volseg'.")

    model = create_finetune_model(
        backbone_name=backbone_name,
        checkpoint_path=None,
        num_classes=6,
        img_size=112,
        patch_size=4,
        num_frames=32,
        tubelet_size=8,
        num_slices=9  # 6 SAX + 3 LAX
    )
    model.to(device)

    ckpt_path = args.checkpoint
    print(f"Loading checkpoint from {ckpt_path} ...")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    # Handle module. prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 3. Sliding Window Inference
    S_window_sax = 6
    T_window = 32
    T_stride = 16
    S_stride = 3

    probs_sum = np.zeros((6, S_sax_total + 3, T_total, H, W), dtype=np.float32)
    counts = np.zeros((S_sax_total + 3, T_total, H, W), dtype=np.float32)

    t_starts = list(range(0, max(1, T_total - T_window + 1), T_stride))
    if t_starts[-1] + T_window < T_total:
        t_starts.append(max(0, T_total - T_window))

    s_starts = list(range(0, max(1, S_sax_total - S_window_sax + 1), S_stride))
    if s_starts[-1] + S_window_sax < S_sax_total:
        s_starts.append(max(0, S_sax_total - S_window_sax))

    print(f"Time window starts: {t_starts}")
    print(f"SAX slice window starts: {s_starts}")

    with torch.no_grad():
        for t_start in tqdm(t_starts, desc="Time loop"):
            t_end = min(t_start + T_window, T_total)

            for s_start in s_starts:
                s_end = min(s_start + S_window_sax, S_sax_total)

                # Crop SAX
                sax_crop = sax_im_data[:, :, s_start:s_end, t_start:t_end]
                lax_crop = lax_im_data[:, :, :, t_start:t_end]

                # mirror image on x=y (Dataset does np.moveaxis(1, 0))
                sax_crop = np.moveaxis(sax_crop, 1, 0)
                lax_crop = np.moveaxis(lax_crop, 1, 0)

                im_crop = np.concatenate([sax_crop, lax_crop], axis=2)
                im_crop = np.transpose(im_crop, (2, 3, 0, 1))  # [S, T, H, W]
                im_crop = image_normalization(im_crop)
                im_tensor = torch.from_numpy(im_crop).unsqueeze(0).to(device)  # [1, 9, 32, 112, 112]

                # patch_coords_3d
                win_sax_spatial = all_sax_spatial[s_start:s_end]
                win_sax_pixel_spacing = all_sax_pixel_spacing[s_start:s_end]

                win_spatial_coords = np.concatenate([win_sax_spatial, lax_spatial], axis=0)
                win_pixel_spacing = np.concatenate([win_sax_pixel_spacing, lax_pixel_spacing], axis=0)

                patch_coords_3d = get_patch_coords_3d(win_spatial_coords, win_pixel_spacing)
                patch_coords_3d_tensor = torch.from_numpy(patch_coords_3d).unsqueeze(0).to(device)  # [1, 9, N, 3]

                # forward
                logits = model(im_tensor, patch_coords_3d=patch_coords_3d_tensor)  # [1, 6, 9, 32, 112, 112]
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # [6, 9, 32, 112, 112]

                probs_sum[:, s_start:s_end, t_start:t_end, :, :] += probs[:, :S_window_sax, :, :, :]
                counts[s_start:s_end, t_start:t_end, :, :] += 1

                probs_sum[:, S_sax_total:, t_start:t_end, :, :] += probs[:, S_window_sax:, :, :, :]
                counts[S_sax_total:, t_start:t_end, :, :] += 1

    final_probs = probs_sum / np.clip(counts, 1, None)[np.newaxis, ...]
    final_preds = np.argmax(final_probs, axis=0)  # [S, T, H, W]

    # Recover to original shape [H, W, S, T]
    final_preds_unmirrored = np.moveaxis(final_preds, 2, 3)
    final_preds_original_shape = np.transpose(final_preds_unmirrored, (2, 3, 0, 1))

    # Save output as sequence of NIfTI files by time frame
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # --- SAX Affine ---
    dy, dx = all_sax_pixel_spacing[0]
    pos = all_sax_spatial[0, :3]
    row = all_sax_spatial[0, 3:6]
    col = all_sax_spatial[0, 6:9]
    nrm = np.cross(row, col)
    nrm = nrm / (np.linalg.norm(nrm) + 1e-12)

    if S_sax_total >= 2:
        pos1 = all_sax_spatial[1, :3]
        proj0 = np.dot(pos, nrm)
        proj1 = np.dot(pos1, nrm)
        dz = abs(proj1 - proj0)
        if dz < 1e-3:
            dz = 10.0
    else:
        dz = 10.0

    affine_sax = np.eye(4, dtype=np.float64)
    affine_sax[:3, 0] = row * dx
    affine_sax[:3, 1] = col * dy
    affine_sax[:3, 2] = nrm * dz
    affine_sax[:3, 3] = pos

    # Build distinct output folders for SAX and LAX
    sax_dir = os.path.join(out_dir, "SAX")
    os.makedirs(sax_dir, exist_ok=True)

    lax_dirs = [os.path.join(out_dir, f"LAX_{ch}Ch") for ch in ["2", "3", "4"]]
    for d in lax_dirs:
        os.makedirs(d, exist_ok=True)

    for t in range(T_total):
        frame_data = final_preds_original_shape[:, :, :, t]  # [H, W, S_Total]

        # 1. Save SAX
        sax_data = frame_data[:, :, :S_sax_total]
        nii_sax = nib.Nifti1Image(sax_data.astype(np.uint8), affine_sax)
        try:
            nii_sax.header.set_zooms((dx, dy, dz))
        except Exception:
            pass
        nib.save(nii_sax, os.path.join(sax_dir, f"time_{t:02d}.nii.gz"))

        # 2. Save LAX views (2Ch, 3Ch, 4Ch)
        for idx in range(3):
            lax_idx = S_sax_total + idx
            lax_data = frame_data[:, :, lax_idx:lax_idx + 1]  # [H, W, 1]

            ldy, ldx = lax_pixel_spacing[idx]
            lpos = lax_spatial[idx, :3]
            lrow = lax_spatial[idx, 3:6]
            lcol = lax_spatial[idx, 6:9]
            lnrm = np.cross(lrow, lcol)
            lnrm = lnrm / (np.linalg.norm(lnrm) + 1e-12)

            affine_lax = np.eye(4, dtype=np.float64)
            affine_lax[:3, 0] = lrow * ldx
            affine_lax[:3, 1] = lcol * ldy
            affine_lax[:3, 2] = lnrm * 1.0  # 2D slice
            affine_lax[:3, 3] = lpos

            nii_lax = nib.Nifti1Image(lax_data.astype(np.uint8), affine_lax)
            try:
                nii_lax.header.set_zooms((ldx, ldy, 1.0))
            except Exception:
                pass

            nib.save(nii_lax, os.path.join(lax_dirs[idx], f"time_{t:02d}.nii.gz"))

    print(f"Inference completed! Saved SAX and LAX sequences to {out_dir}")
    print(f"Total time frames {T_total} generated for each view.")

    # --- Performance Comparison ---
    if seg_sax_gt is not None and seg_lax_gt is not None:
        print("Calculating performance metrics...")

        # Prepare GT in the same format as final_preds_original_shape: [H, W, S, T]
        gt_combined = np.concatenate([seg_sax_gt, seg_lax_gt], axis=2)

        dice_scores = calculate_dice(final_preds_original_shape, gt_combined, num_classes=6)
        mean_dice = np.mean(dice_scores)

        labels = ["LVBP", "LVMYO", "RVBP", "LABP", "RABP"]
        metrics_file = os.path.join(out_dir, "metrics.txt")

        with open(metrics_file, "w") as f:
            f.write(f"Evaluation results for {data_path}\n")
            f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 30 + "\n")
            for label, score in zip(labels, dice_scores):
                f.write(f"Dice {label}: {score:.4f}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Mean Dice: {mean_dice:.4f}\n")

        print(f"Metrics saved to {metrics_file}")
        print(f"Mean Dice: {mean_dice:.4f}")
    else:
        print("Ground truth not found in data. Skipping performance comparison.")


def get_args():
    parser = argparse.ArgumentParser(description="Sliding Window Inference for CMR Segmentation")
    parser.add_argument("--decoder", type=str, default="unetr", choices=["unetr", "volseg"], help="Decoder type")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input .npz file")
    parser.add_argument("--out_dir", type=str, default="./vis_results", help="Output directory")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)