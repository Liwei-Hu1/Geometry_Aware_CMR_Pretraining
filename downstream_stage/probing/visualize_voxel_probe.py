"""
visualize_voxel_probe.py
========================
Sample a dense 3D grid in PHYSICAL cardiac coordinate space,
feed through the volume head's trilinear interpolation pipeline,
apply a linear probe, scatter-plot the results, save NIfTI volumes,
and visualize prediction errors.

No domain gap: the feature extraction path is identical to training.

Usage
-----
python downstream_stage/visualize_voxel_probe.py \
    --checkpoint save_ckpt/run_3/checkpoint-299.pth \
    --config    downstream_stage/config/probing.yaml \
    --output_dir viz_outputs \
    --probe_batches 20 \
    --probe_epochs  20 \
    --voxel_mm 0.2
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from downstream_stage.modeling_finetune import create_finetune_model

from data import unified_datasets

# ── style ─────────────────────────────────────────────────────────────────────
CLASS_COLORS = {
    0: "#333333",
    1: "#D62728",
    2: "#2CA02C",
    3: "#1F77B4",
    4: "#FFBF00",
    5: "#9467BD",
}
CLASS_NAMES = {0: "BG", 1: "LVBP", 2: "LVMYO", 3: "RVBP", 4: "LABP", 5: "RABP"}
DARK_BG = "#0d0d0d"
PANEL_BG = "#111111"
FONT_CLR = "#e0e0e0"
TICK_CLR = "#555555"
SPINE_CLR = "#333333"
matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "text.color": FONT_CLR,
        "axes.labelcolor": FONT_CLR,
        "xtick.color": TICK_CLR,
        "ytick.color": TICK_CLR,
    }
)
PATCH_SIZE = 4


# ── CLI ───────────────────────────────────────────────────────────────────────
def get_args() -> argparse.Namespace:
    """Parse runtime arguments for visualization.

    Returns:
        argparse.Namespace: CLI parsed arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="downstream_stage/config/probing.yaml")
    p.add_argument("--output_dir", default="viz_outputs")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--probe_batches", default=20, type=int)
    p.add_argument("--probe_epochs", default=20, type=int)
    p.add_argument(
        "--voxel_mm",
        default=0.2,
        type=float,
        help="Isotropic voxel size in mm for query grid",
    )
    p.add_argument("--num_val", default=3, type=int)
    p.add_argument(
        "--padding",
        default=0.05,
        type=float,
        help="Padding fraction added to bounding box of GT coords",
    )
    p.add_argument(
        "--save_nifti",
        action="store_true",
        default=True,
        help="Save pred and GT as NIfTI (.nii.gz) volumes",
    )
    p.add_argument(
        "--no_error_vis",
        action="store_true",
        default=False,
        help="Skip error visualization",
    )
    p.add_argument(
        "--gt_nn_radius",
        default=0.5,
        type=float,
        help="Max physical distance (mm) to assign a GT label to query point",
    )
    return p.parse_args()


# ── model ─────────────────────────────────────────────────────────────────────
def build_model(cfg: OmegaConf, args: argparse.Namespace) -> nn.Module:
    """Build the base segmentation model acting as a structural feature backbone.

    Args:
        cfg (OmegaConf): Configuration map logic values.
        args (argparse.Namespace): Terminal parameter constraints.

    Returns:
        nn.Module: Network configured object holding loaded checkpoint topology.
    """
    n_slices = cfg.data.sax_slice_num + (3 if "AllAX" in cfg.data.dataset_cls else 0)
    model = create_finetune_model(
        backbone_name=cfg.module.module_name,
        checkpoint_path=args.checkpoint,
        num_classes=cfg.data.num_classes,
        img_size=cfg.data.image_size,
        num_frames=cfg.data.time_frame,
        patch_size=PATCH_SIZE,
        tubelet_size=cfg.data.tubelet_size,
        num_slices=n_slices,
    )
    model.to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def forward_one(
    model: nn.Module, 
    images: torch.Tensor, 
    coords: torch.Tensor, 
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate single inference step outputs locally storing spatial features.

    Args:
        model (nn.Module): Forward computation mapping segment pipeline graph.
        images (torch.Tensor): Visual sequences matrix.
        coords (torch.Tensor): Dimensional indices arrays mapping slice locations.
        device (torch.device): Processor location.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Computed output volume and reconstructed component items.
    """
    B, S, T, H, W = images.shape
    images = images.to(device)
    coords = coords.to(device)
    vf = images.view(B * S, 1, T, H, W)
    vids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1).flatten()
    cf = coords.reshape(B * S, -1, 3)
    enc = model._encode_batch(vf, vids, cf)
    enc = enc[0] if isinstance(enc, tuple) else enc
    vol, rec = model.volume_head(enc, patch_coords_3d=coords)
    return vol, rec  # [B,C,H,W,D], [B,S,N_sp,D]


def patchify_labels(
    targets: torch.Tensor, 
    B: int, 
    S: int, 
    H: int, 
    W: int, 
    ps: int = PATCH_SIZE
) -> torch.Tensor:
    """Contract mask mappings down to match latent dimensions structurally format.

    Args:
        targets (torch.Tensor): Input dense segmentation matrix.
        B (int): Batch magnitude items.
        S (int): Slice element depth count.
        H (int): Vertical extent measure.
        W (int): Base horizontal pixels span.
        ps (int, optional): Extent of side size patching. Defaults to PATCH_SIZE.

    Returns:
        torch.Tensor: Contracted and mapped patch array format elements.
    """
    H_p, W_p = H // ps, W // ps
    t_max, _ = targets.max(2)
    return (
        F.interpolate(
            t_max.float().flatten(0, 1).unsqueeze(1), size=(H_p, W_p), mode="nearest"
        )
        .squeeze(1)
        .view(B, S, H_p * W_p)
        .long()
    )


# ── probe ─────────────────────────────────────────────────────────────────────
class LinearProbe(nn.Module):
    """Linear projection layer converting volume features to class predictions."""
    def __init__(self, D: int, C: int) -> None:
        """Initialize the probe.
        
        Args:
            D (int): Input feature depth.
            C (int): Number of output classes.
        """
        super().__init__()
        self.fc = nn.Linear(D, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x (torch.Tensor): Dense features.
            
        Returns:
            torch.Tensor: Logits.
        """
        return self.fc(x)


def train_probe(
    model: nn.Module, 
    loader: DataLoader, 
    device: torch.device, 
    num_classes: int, 
    batches: int, 
    epochs: int
) -> LinearProbe:
    """Train a linear probing head on densely extracted patches.

    Args:
        model (nn.Module): Pre-trained foundation framework.
        loader (DataLoader): Sequence loading iterator.
        device (torch.device): Execution mapping device.
        num_classes (int): Category distribution magnitude limit.
        batches (int): Restrict data processing bounds count.
        epochs (int): Processing timeline parameter limits maximum runtime.

    Returns:
        LinearProbe: Converged and fitted linear boundary class mapping model.
    """
    all_f, all_l = [], []
    for i, batch in enumerate(tqdm(loader, desc="  collect", leave=False)):
        if i >= batches:
            break
        imgs, tgt, coords = batch[0], batch[1], batch[4]
        B, S, T, H, W = imgs.shape
        with torch.no_grad():
            _, rec = forward_one(model, imgs, coords, device)
        lbl = patchify_labels(tgt, B, S, H, W)
        all_f.append(rec.cpu())
        all_l.append(lbl.cpu())
    feats = torch.cat(all_f, 0).reshape(-1, all_f[0].shape[-1])
    labels = torch.cat(all_l, 0).reshape(-1)
    D = feats.shape[-1]
    probe = LinearProbe(D, num_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    bs = 8192
    print(f"  Probe: {epochs} ep × {len(feats):,} patches (D={D}) …")
    for ep in range(epochs):
        probe.train()
        perm = torch.randperm(len(feats))
        ep_loss = 0.0
        for i in range(0, len(feats), bs):
            idx = perm[i : i + bs]
            loss = crit(probe(feats[idx].to(device)), labels[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        if (ep + 1) % 5 == 0:
            print(f"    ep {ep+1}/{epochs}  loss={ep_loss/len(feats):.4f}")
    probe.eval()
    return probe


# ── dense physical-space query ────────────────────────────────────────────────
@torch.no_grad()
def query_volume_at_physical_coords(
    model: nn.Module,
    volume: torch.Tensor,
    probe: LinearProbe,
    device: torch.device,
    query_coords: np.ndarray,
    grid_scale_factor: float = 4.0,
) -> np.ndarray:
    """Evaluate physical metrics bounds at continuous query coordinates.

    1. Convert physical coords → normalised grid coords (same as model training)
    2. F.grid_sample from volume → rec_features_flat
    3. norm_slice → vol_to_enc → probe → class predictions
    Returns: np.ndarray [N] int

    Args:
        model (nn.Module): Root volume mapping definition interface object backbone.
        volume (torch.Tensor): Continuous source representation field block map.
        probe (LinearProbe): Learned classifier projection tensor interface module.
        device (torch.device): GPU hardware accelerator variable map reference index target.
        query_coords (np.ndarray): Physical positions mapping space bounds limit values metrics.
        grid_scale_factor (float, optional): Resolution step increment bounds element property offset value. Defaults to 4.0.

    Returns:
        np.ndarray: Predicted mapped label vector integers index matching coordinates index lengths sequence.
    """
    N = len(query_coords)
    vol = volume.unsqueeze(0).to(device)  # [1, C, H, W, D]
    vh = model.volume_head

    # Normalise
    qc = torch.from_numpy(query_coords).float().to(device)  # [N, 3]
    grid = (qc / grid_scale_factor).clamp(-1, 1)  # [N, 3]
    grid_5d = grid.view(1, N, 1, 1, 3)  # [1,N,1,1,3]

    sampled = F.grid_sample(
        vol, grid_5d, mode="bilinear", align_corners=True, padding_mode="zeros"
    )  # [1, C, N, 1, 1]
    feats = sampled.squeeze(0).squeeze(-1).squeeze(-1).T  # [N, C]

    # Exact same post-processing as in ThreeDVolumeHead.forward
    feats = vh.norm_slice(feats)  # LayerNorm
    feats = vh.vol_to_enc(feats)  # Linear or Identity

    logits = probe(feats)  # [N, num_cls]
    preds = logits.argmax(-1).cpu().numpy()
    return preds  # [N]


# ── plotting helpers ──────────────────────────────────────────────────────────
def setup_ax(ax: plt.Axes, title: str) -> None:
    """Configure axis aesthetics.
    
    Args:
        ax (plt.Axes): Plot axis element.
        title (str): String header text.
    """
    ax.set_facecolor(PANEL_BG)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_edgecolor(SPINE_CLR)
    ax.tick_params(colors=TICK_CLR, labelsize=5)
    ax.set_xlabel("X", color="#666", fontsize=6, labelpad=1)
    ax.set_ylabel("Y", color="#666", fontsize=6, labelpad=1)
    ax.set_zlabel("Z", color="#666", fontsize=6, labelpad=1)
    ax.set_title(title, color=FONT_CLR, fontsize=9, pad=4)


def scatter3d(
    ax: plt.Axes, 
    xyz: np.ndarray, 
    cls: np.ndarray, 
    elev: int = 25, 
    azim: int = -60, 
    s: int = 12, 
    alpha: float = 0.8
) -> None:
    """Render a 3D scatter plot colored by class label.
    
    Args:
        ax (plt.Axes): Subplot instance mapping target.
        xyz (np.ndarray): Coordinate tensor locations points.
        cls (np.ndarray): Class assignment scalar indices array.
        elev (int, optional): Viewpoint elevation height map. Defaults to 25.
        azim (int, optional): Viewpoint azimuth angle maps. Defaults to -60.
        s (int, optional): Points scale parameter indices size. Defaults to 12.
        alpha (float, optional): Opacity parameter scale vector array elements float. Defaults to 0.8.
    """
    fg = cls > 0
    for c in sorted(np.unique(cls[fg]).astype(int)):
        m = fg & (cls == c)
        ax.scatter(
            xyz[m, 0],
            xyz[m, 1],
            xyz[m, 2],
            c=CLASS_COLORS.get(c, "#aaa"),
            s=s,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )
    ax.view_init(elev=elev, azim=azim)


# ── NIfTI helpers ────────────────────────────────────────────────────────────
def save_nifti(
    class_grid: np.ndarray, 
    lo: np.ndarray, 
    hi: np.ndarray, 
    path: Path, 
    voxel_mm: float
) -> None:
    """Store the segmentation field locally as standard compressed NIFTI 3D file metric payload wrapper element wrapper.

    Args:
        class_grid (np.ndarray): Target category field 3D integers mapped format bounds limits volume map array.
        lo (np.ndarray): Cartesian properties floor bounds limits limits index parameter space range offsets array points array.
        hi (np.ndarray): Absolute boundary value parameters offset coordinate bounding index offset bounds values points maximum location vector elements.
        path (Path): Formatted target filesystem write mapping index name index label space string string file.
        voxel_mm (float): Structural spacing limits offset properties metric length units floating scalar component dimension scale vector units spatial range scalar property interval metrics elements variables spacing elements variable range vector spacing array units units range parameters elements.
    """
    import nibabel as nib

    # Build affine: voxel → physical coord (RAS)
    affine = np.eye(4, dtype=np.float32)
    affine[0, 0] = voxel_mm
    affine[1, 1] = voxel_mm
    affine[2, 2] = voxel_mm
    affine[:3, 3] = lo
    img = nib.Nifti1Image(class_grid.astype(np.int16), affine)
    img.header.set_zooms((voxel_mm, voxel_mm, voxel_mm))
    nib.save(img, str(path))
    print(f"    NIfTI saved → {path}")


def gt_to_grid(
    gt_coords: np.ndarray, 
    gt_labels: np.ndarray, 
    query_coords: np.ndarray, 
    nn_radius: float = 0.5
) -> np.ndarray:
    """Assign GT labels to each query grid point by nearest-neighbour.
    Points further than nn_radius mm from any GT patch are left as 0 (BG).

    Args:
        gt_coords (np.ndarray): physical coords of GT patches
        gt_labels (np.ndarray): int labels
        query_coords (np.ndarray): physical coords of query grid
        nn_radius (float, optional): Maximum radius bound metrics range offset map element property scalar space parameters target interval index length variable bounds array. Defaults to 0.5.

    Returns:  
        np.ndarray: [Q] int labels 
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(gt_coords)
    dists, idx = tree.query(query_coords, k=1, workers=-1)
    assigned = gt_labels[idx].copy()
    assigned[dists > nn_radius] = 0  # too far → BG
    return assigned.astype(int)


# ── Error visualisation ───────────────────────────────────────────────────────
ERROR_COLORS = {
    "correct": "#2CA02C",  # green  - correctly predicted fg
    "wrong": "#D62728",  # red    - fg predicted wrong class
    "fp": "#FF7F0E",  # orange - BG predicted as fg
    "fn": "#9467BD",  # purple - fg predicted as BG
}


def visualize_error(
    pred: np.ndarray, 
    gt_grid: np.ndarray, 
    query_coords: np.ndarray, 
    out: Path, 
    sample_i: int, 
    grid_shape: Tuple[int, int, int]
) -> None:
    """Compare pred [Q] with gt_grid [Q] on the query grid.
    Only meaningful where gt_grid > 0 (GT has coverage).

    Categories (showing in 3D scatter):
      correct  - gt > 0 & pred == gt
      wrong    - gt > 0 & pred > 0 & pred != gt   (mis-classified)
      fn       - gt > 0 & pred == 0               (false negative)
      fp       - gt == 0 & pred > 0               (false positive)
      
    Args:
        pred (np.ndarray): Predictions output.
        gt_grid (np.ndarray): Ground truth map.
        query_coords (np.ndarray): Input points positions map.
        out (Path): Directory scalar output locus element targets.
        sample_i (int): Ordinal count maps scalars identifier metrics.
        grid_shape (Tuple[int, int, int]): Tensor bounds map limits index spacing metric offset properties values indices property values tuple variables variables logic variables bounds vector arrays limits sizes variables property elements variables elements offset metrics properties maps sizes elements offset index variable vectors vector spacing map logic elements offsets index spacing formats size bounds offset variable array formats offset offsets index arrays offset element parameter tuple map scalar arrays arrays elements limit format formats tuple array target vector index arrays spacing variable offset length format size parameters size lengths parameter sizes offset element values variables mapping limit map limits value limits values limit value property string element arrays size string parameters maps metrics value formats elements element target parameters values values vector units vector offsets length metrics parameter properties metric scalar index index elements value formats format target map index index dimensions offset sizes offsets scalar format parameters variable array target variable parameter metrics formatting dimensions limits properties element values.
    """
    gt = gt_grid
    has_gt = gt > 0
    mask_correct = has_gt & (pred == gt)
    mask_wrong = has_gt & (pred > 0) & (pred != gt)
    mask_fn = has_gt & (pred == 0)
    mask_fp = (gt == 0) & (pred > 0)

    n_covered = int(has_gt.sum())
    n_correct = int(mask_correct.sum())
    n_wrong = int(mask_wrong.sum())
    n_fn = int(mask_fn.sum())
    n_fp = int(mask_fp.sum())
    accuracy = n_correct / n_covered * 100 if n_covered > 0 else 0
    print(
        f"  Error stats: covered={n_covered} | "
        f"correct={n_correct} ({accuracy:.1f}%) | "
        f"wrong={n_wrong} | FN={n_fn} | FP={n_fp}"
    )

    views = [(25, -60), (0, 0), (0, 90)]
    vtitles = ["Isometric", "Coronal", "Sagittal"]

    fig = plt.figure(figsize=(16, 5))
    fig.patch.set_facecolor(DARK_BG)
    gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.05, wspace=0.04)

    cat_colors = [
        (mask_correct, ERROR_COLORS["correct"], "Correct", 8, 0.7),
        (mask_wrong, ERROR_COLORS["wrong"], "Wrong cls", 12, 0.9),
        (mask_fn, ERROR_COLORS["fn"], "FN (BG)", 12, 0.8),
        (mask_fp, ERROR_COLORS["fp"], "FP", 6, 0.5),
    ]
    legend_patches = [
        mpatches.Patch(color=col, label=lbl) for _, col, lbl, _, _ in cat_colors
    ]

    for ci, ((elev, azim), vt) in enumerate(zip(views, vtitles)):
        ax = fig.add_subplot(gs[0, ci], projection="3d")
        setup_ax(ax, vt)
        for mask, col, _, s, alpha in cat_colors:
            if mask.any():
                xyz = query_coords[mask]
                ax.scatter(
                    xyz[:, 0],
                    xyz[:, 1],
                    xyz[:, 2],
                    c=col,
                    s=s,
                    alpha=alpha,
                    linewidths=0,
                    rasterized=True,
                )
        ax.view_init(elev=elev, azim=azim)

    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=len(legend_patches),
        bbox_to_anchor=(0.5, -0.03),
        framealpha=0.2,
        facecolor=PANEL_BG,
        edgecolor=SPINE_CLR,
        labelcolor=FONT_CLR,
        fontsize=9,
        markerscale=1.8,
    )
    fig.suptitle(
        f"Sample {sample_i+1} Error Map  |  "
        f"Acc={accuracy:.1f}%  Correct={n_correct}  "
        f"Wrong={n_wrong}  FN={n_fn}  FP={n_fp}  (GT coverage={n_covered})",
        color=FONT_CLR,
        fontsize=10,
        y=1.02,
    )
    fname = out / f"error_map_sample{sample_i}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    print(f"  Error map saved → {fname}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    """Execute main visualization parameters pipeline mapping coordinate space limits limits metric logic property target logic mapping offset elements logic."""
    args = get_args()
    device = torch.device(args.device)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.config)

    print("\n[1/4] Building model …")
    model = build_model(cfg, args)
    gsf = model.volume_head.grid_scale_factor  # typically 4.0
    print(f"  grid_scale_factor = {gsf}")

    print("[2/4] Building data loaders …")
    dataset_cls = getattr(unified_datasets, cfg.data.dataset_cls)
    kwargs = dict(
        load_seg=True,
        mask_ratio=0.0,
        window_size=(
            (cfg.data.tubelet_size, *cfg.data.img_patch_size)
            if hasattr(cfg.data, "img_patch_size")
            else None
        ),
    )
    root = Path(cfg.data.processed_dir)
    files = sorted(root.rglob("*.npz"))
    n_tr = cfg.data.num_train
    n_val = cfg.data.num_val
    tr_ld = DataLoader(
        dataset_cls(files[:n_tr], **kwargs), batch_size=2, shuffle=True, num_workers=0
    )
    val_ld = DataLoader(
        dataset_cls(files[n_tr : n_tr + n_val], **kwargs),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print("[3/4] Training linear probe …")
    probe = train_probe(
        model,
        tr_ld,
        device,
        cfg.data.num_classes,
        args.probe_batches,
        args.probe_epochs,
    )

    print(f"[4/4] Visualising {args.num_val} val sample(s) …")
    views = [(25, -60), (0, 0), (0, 90)]
    vtitles = ["Isometric", "Coronal", "Sagittal"]
    legend_patches = [
        mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c])
        for c in sorted(CLASS_COLORS)
        if c > 0
    ]
    for si, batch in enumerate(val_ld):
        if si >= args.num_val:
            break
        imgs, tgt, coords = batch[0], batch[1], batch[4]
        B, S, T, H, W = imgs.shape

        with torch.no_grad():
            vol_raw, _ = forward_one(model, imgs, coords, device)

        vol_np = vol_raw[0].cpu().float()  # [C, h, w, d]
        lbl_np = patchify_labels(tgt, B, S, H, W)[0].numpy()  # [S, N_sp]
        coords_np = coords[0].cpu().numpy()  # [S, N_sp, 3]

        # ── Bounding box from GT foreground patches ───────────────────────
        cf = coords_np.reshape(-1, 3)  # [S*N_sp, 3]
        lf = lbl_np.reshape(-1)  # [S*N_sp]
        fg_mask = lf > 0
        fg_cfull = cf[fg_mask] if fg_mask.any() else cf
        lo = fg_cfull.min(0)
        hi = fg_cfull.max(0)
        pad = (hi - lo) * args.padding
        lo -= pad
        hi += pad
        print(
            f"  sample {si}: cardiac bbox X[{lo[0]:.2f},{hi[0]:.2f}] "
            f"Y[{lo[1]:.2f},{hi[1]:.2f}] Z[{lo[2]:.2f},{hi[2]:.2f}]"
        )

        # ── Dense Isotropic Query Grid ────────────────────────────────────────
        Nx = max(int(np.ceil((hi[0] - lo[0]) / args.voxel_mm)), 1)
        Ny = max(int(np.ceil((hi[1] - lo[1]) / args.voxel_mm)), 1)
        Nz = max(int(np.ceil((hi[2] - lo[2]) / args.voxel_mm)), 1)
        Q_total = Nx * Ny * Nz

        xs = np.linspace(lo[0], hi[0], Nx)
        ys = np.linspace(lo[1], hi[1], Ny)
        zs = np.linspace(lo[2], hi[2], Nz)
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        query_coords_flat = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], 1)  # [Q,3]

        # ── F.grid_sample → probe → pred ─────────────────────────────────
        preds = query_volume_at_physical_coords(
            model, vol_np, probe, device, query_coords_flat, grid_scale_factor=gsf
        )
        n_fg = int((preds > 0).sum())
        print(
            f"  sample {si}: grid {Nx}x{Ny}x{Nz} [{args.voxel_mm}mm] | fg preds {n_fg}/{Q_total} ({100*n_fg/Q_total:.1f}%)"
        )

        # ── GT grid (nearest-neighbour interpolation) ─────────────────────
        gt_grid = gt_to_grid(cf, lf, query_coords_flat, nn_radius=args.gt_nn_radius)
        preds_grid = preds.reshape(Nx, Ny, Nz)
        gt_vol = gt_grid.reshape(Nx, Ny, Nz)

        # ── NIfTI export ──────────────────────────────────────────────────
        if args.save_nifti:
            sample_dir = out / f"sample{si}_nifti"
            sample_dir.mkdir(exist_ok=True)
            save_nifti(preds_grid, lo, hi, sample_dir / "pred.nii.gz", args.voxel_mm)
            save_nifti(gt_vol, lo, hi, sample_dir / "gt.nii.gz", args.voxel_mm)

        # ── Main scatter figure ───────────────────────────────────────────
        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor(DARK_BG)
        gs2 = gridspec.GridSpec(2, 3, figure=fig, hspace=0.05, wspace=0.04)

        box_aspect = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])

        for ci, ((elev, azim), vt) in enumerate(zip(views, vtitles)):
            ax = fig.add_subplot(gs2[0, ci], projection="3d")
            setup_ax(ax, (f"Pred — {Nx}x{Ny}x{Nz} | {vt}") if ci == 0 else vt)
            ax.set_box_aspect(box_aspect)
            scatter3d(
                ax, query_coords_flat, preds, elev=elev, azim=azim, s=35, alpha=0.8
            )

        for ci, ((elev, azim), vt) in enumerate(zip(views, vtitles)):
            ax = fig.add_subplot(gs2[1, ci], projection="3d")
            setup_ax(ax, (f"GT (patches) | {vt}") if ci == 0 else vt)
            ax.set_box_aspect(box_aspect)
            scatter3d(ax, cf, lf, elev=elev, azim=azim, s=25, alpha=0.85)

        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=len(legend_patches),
            bbox_to_anchor=(0.5, -0.01),
            framealpha=0.2,
            facecolor=PANEL_BG,
            edgecolor=SPINE_CLR,
            labelcolor=FONT_CLR,
            fontsize=9,
            markerscale=1.8,
        )
        fig.suptitle(
            f"Latent Volume Probe — sample {si+1}  "
            f"| Isotropic grid ({args.voxel_mm}mm)  "
            f"| fg: {n_fg}/{Q_total} ({100*n_fg/Q_total:.0f}%)",
            color=FONT_CLR,
            fontsize=11,
            y=1.01,
        )
        fname = out / f"voxel_probe_sample{si}.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
        print(f"  Scatter saved → {fname}")
        plt.close(fig)

        # ── Error visualisation ───────────────────────────────────────────
        if not args.no_error_vis:
            visualize_error(preds, gt_grid, query_coords_flat, out, si, (Nx, Ny, Nz))

    print(f"\n✓ Done → {out}")


if __name__ == "__main__":
    main()
