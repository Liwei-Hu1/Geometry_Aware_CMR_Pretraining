import dataclasses
import os
import socket
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import tqdm


@dataclasses.dataclass
class PathHolder:
    dataset_folder: str
    all_feature_tabular_dir: str
    biomarker_tabular_dir: str
    processed_folder: str
    log_folder: str
    dataloader_file_folder: str
    cmr_path_pickle_name: str
    biomarker_table_pickle_name: str
    processed_table_pickle_name: str
    extra_tabular_dir: Optional[str] = None


def get_computer_id() -> str:
    """Get a known identifier based on the current host's name.

    Returns:
        str: A formatted identifier for the host (e.g. 'yundi-wks' or 'username-gpu').

    Raises:
        Exception: If the hostname is unrecognized.
    """
    hostname = socket.gethostname()
    if hostname == "unicorn":
        return "yundi-wks"
    elif hostname in [
        "atlas",
        "chameleon",
        "helios",
        "prometheus",
        "leto",
        "hercules",
        "apollo",
    ]:  # GPU server
        logname = os.environ["LOGNAME"]
        return f"{logname}-gpu"
    else:
        raise Exception(f"Unknown hostname: {hostname}.")


def get_data_paths() -> PathHolder:
    """Get the standard dataset paths for WholeHeartRL.

    Returns:
        PathHolder: Data class holding absolute paths.
    """
    return PathHolder(
        dataset_folder=os.path.join(
            "./data/raw_data"
        ),
        all_feature_tabular_dir=os.path.join(
            "./data"
        ),
        biomarker_tabular_dir=os.path.join(
            "./data"
        ),
        processed_folder=os.path.join(
            "./data/train_data"
        ),
        log_folder=os.path.join(
            "./data/log"
        ),
        dataloader_file_folder=os.path.join(
            "./data"
        ),
        cmr_path_pickle_name=os.path.join(
            "./data/cmr_subject_paths.pkl"
        ),
        biomarker_table_pickle_name=os.path.join(
            "./data/biomarker_table.pkl"
        ),
        processed_table_pickle_name=os.path.join(
            "./data/processed_table.pkl"
        ),
        extra_tabular_dir=os.getenv("EXTRA_TABULAR_DIR"),
    )


def get_biggest_2D_slice_bbox(
    segmentation: np.ndarray, padding: int = 10, **kwargs: Any
) -> np.ndarray:
    """Calculate the tightest 2D bounding box encompassing the foreground across all slices with padding.

    Collapse all non-height/width dimensions into a 2D mask using the max operation
    such that the largest slice/frame is taken into account.
    Then find max and min indices of foreground mask to create the tightest possible bbox.
    Add padding to bbox by taking into account image's edges.

    Args:
        segmentation (np.ndarray): The full segmentation array.
        padding (int, optional): Spatial padding around bounds. Defaults to 10.
        **kwargs: Additional keyword arguments.

    Returns:
        np.ndarray: Bounding box coordinates in format [Y1, X1, Y2, X2].
    """
    if len(segmentation.shape) > 2:
        non_slice_dims = tuple(range(2, len(segmentation.shape)))
        segmentation = segmentation.max(axis=non_slice_dims)
    indices = np.argwhere(segmentation > 0)
    min_indices = indices.min(0) - padding
    min_indices = np.maximum(min_indices, 0)
    assert len(min_indices) == 2
    max_indices = indices.max(0) + padding
    max_indices = np.minimum(max_indices, np.array(segmentation.shape[:2]) - 1)
    assert len(max_indices) == 2
    bbox = np.concatenate((min_indices, max_indices), axis=0)
    return bbox


def get_2D_slice_bbox_with_fixed_size(
    segmentation: np.ndarray, bbox_size: Union[List[int], Tuple[int, int]] = [128, 128]
) -> np.ndarray:
    """Find the biggest 2D slice center and crop/pad it into the desired fixed bbox_size.

    Args:
        segmentation (np.ndarray): The spatial segmentation array.
        bbox_size (List[int], optional): The target dimensions [H, W]. Defaults to [128, 128].

    Returns:
        np.ndarray: Bounding box coordinates [Y1, X1, Y2, X2].
    """
    if len(segmentation.shape) > 2:
        non_slice_dims = tuple(range(2, len(segmentation.shape)))
        segmentation = segmentation.max(axis=non_slice_dims)
    indices = np.argwhere(segmentation > 0)
    center_indices = (indices.min(0) + indices.max(0)) // 2
    min_indices = center_indices - np.array(bbox_size) // 2
    min_indices = np.maximum(min_indices, 0)
    max_indices = min_indices + np.array(bbox_size)
    bbox = np.concatenate((min_indices, max_indices), axis=0)
    return bbox


def get_2D_central_bbox(
    im_size: Union[Tuple[int, ...], List[int]], bbox_size: Union[List[int], Tuple[int, int]] = [128, 128]
) -> np.ndarray:
    """Calculate the central bounding box of the image using a fixed bbox_size.

    Args:
        im_size (tuple or list): Original spatial dimensions of the image.
        bbox_size (List[int], optional): The target block size [H, W]. Defaults to [128, 128].

    Returns:
        np.ndarray: Bounding box coordinates [Y1, X1, Y2, X2].
    """
    center = np.array(im_size[:2]) // 2 # Only consider spatial dims H,W
    min_indices = center - np.array(bbox_size) // 2
    min_indices = np.maximum(min_indices, 0)
    max_indices = min_indices + np.array(bbox_size)
    bbox = np.concatenate((min_indices, max_indices), axis=0)
    return bbox


def find_healthy_subjects(path: Union[str, Path]) -> pd.Series:
    """Filter out non-healthy subjects from a dataset manifest CSV.

    Args:
        path (str or Path): Path to the CSV file holding patient data fields.

    Returns:
        pd.Series: A pandas Series containing patient IDs (eid) that met the healthy criteria.
    """
    df = pd.read_csv(path)
    conditions = {
        "977:981": [1, 2],  # Overall health rating: Excellent
        # "3325:3337": [-7], # Vascular/heart problems diagnosed by doctor: None of the above
        # "1033:1037": [0], # Diabetes diagnosed by doctor: No
        # "4218:4410": np.nan, # Treatment/medication code
        # "6603": [113, 114], # Tobacco smoking: Never smoked
        # "6359": [-600], # Degree bothered by feeling heart pound/race in the last 3 months: Not bothered at all
        "10362": np.nan,  # Date E66 first reported (obesity)
        "10229": np.nan,  # Date of myocardial infarction
        "10377": np.nan,  # Date Date I21 first reported (acute myocardial infarction)
        # "10357": np.nan, # Date E10 first reported (insulin-dependent diabetes mellitus)
    }
    cond = np.ones(df.shape[0], dtype=bool)
    for key, val in conditions.items():
        if key in ["977:981", "3325:3337", "1033:1037"]:
            # select rows where the column is either nan or the value
            cond1 = np.all(
                eval(f"df.iloc[:, {key}].isna() | df.iloc[:, {key}].isin({val})"),
                axis=1,
            )
            cond2 = np.all(eval(f"df.iloc[:, {key}].isna()"), axis=1)
            # select rows where the column is either nan or the value provided but not all nan
            cond_tmp = cond1 & ~cond2
        elif key == "4218:4410":
            cond_tmp = np.all(eval(f"df.iloc[:, {key}].isna()"), axis=1)
        else:
            if np.isnan(val).all():
                cond_tmp = eval(f"df.iloc[:, {key}].isna()")
            else:
                cond_tmp = eval(f"df.iloc[:, {key}].isin({val})")
        cond &= cond_tmp
    indices = np.where(cond)
    eid = df.iloc[indices]["eid"]
    return eid


def find_indices_of_images(
    load_dir: Union[str, Path],
    sax_file_name: str = "SAX.nii.gz",
    seg_file_name: str = "SAX_nnUNetSeg.nii.gz",
    lax_file_name: List[str] = ["LAX_2Ch.nii.gz", "LAX_3Ch.nii.gz", "LAX_4Ch.nii.gz"],
    sax_bbox_func: Optional[Callable] = get_2D_slice_bbox_with_fixed_size,
    lax_bbox_func: Optional[Callable] = get_2D_central_bbox,
    sax_bbox_size: Union[List[int], Tuple[int, int]] = [128, 128],
    lax_bbox_size: Union[List[int], Tuple[int, int]] = [128, 128],
    id_list: Optional[List[int]] = None,
) -> List[int]:
    """Find the indices of the SAX and LAX images in the given path that have enough sax slices and match given spatial dimension. If id_list is not None, then return the union of the indices.

    :param load_dir: Data directory where subjects are layed out.
    :param img_file_name: Naming convention of the short axis image files you are intending to extract.
    :param seg_file_name: Naming convention of the segmentation files you are intending to extract.
    :param lax_file_name: Naming convention of the long axis image files you are intending to extract.
    :param sax_bbox_func: Function to be used to extract bounding boxes for short axis images.
                          Should follow format [Y1, X1, ..., Y2, X2, ...]
    :param lax_bbox_func: Function to be used to extract bounding boxes for long axis images.
    :param sax_bbox_size: the size of sax bounding box
    :param lax_bbox_size: the size of lax bounding box
    :param id_list: List of subject ids that we search among.
    """
    indices = []
    load_dir = Path(load_dir)
    # Used to collect dataset statistics if necessary
    for i, parent in enumerate(sorted(os.listdir(str(load_dir)))):
        # If id_list is not None, then only extract the images with the indices in the id_list.
        if id_list is not None:
            if int(parent) not in id_list:
                continue

        parent = Path(parent) / "Instance_2"

        im_path = load_dir / parent / sax_file_name
        seg_path = load_dir / parent / "nnUNet_segs" / seg_file_name
        lax_path = [load_dir / parent / i for i in lax_file_name]

        # Make sure both image and segmentation files exist
        if not os.path.exists(im_path):
            continue
        if not os.path.exists(seg_path):
            continue
        if not all(os.path.exists(path) for path in lax_path):
            continue
        try:
            # Short axis image should have at least 9 slices and 50 frames
            im = nib.load(im_path)
            if im.shape[2] < 6:
                print(
                    f"Found an image with suspiciously low number of SAX slices: {im.shape[2]} slices. {im_path.parent.name}"
                )
                continue
            if im.shape[3] != 50:
                print(
                    f"Found an image with suspicious number of SAX frames: {im.shape[3]} frames. {im_path.parent.name}"
                )
                continue
            if sax_bbox_func is not None:
                if im.shape[0] < sax_bbox_size[0] or im.shape[1] < sax_bbox_size[1]:
                    print(
                        f"Found an SAX image with suspiciously low resolution: {im.shape[0:2]} pixels. {im_path.parent.name}"
                    )
                    continue
            # Long axis image should have 50 frames
            lax_ims = [nib.load(path) for path in lax_path]
            if not all(lax_im.shape[3] == 50 for lax_im in lax_ims):
                print(
                    f"Found an image with suspicious number of LAX frames: {lax_ims[0].shape[3], lax_ims[1].shape[3], lax_ims[3].shape[2]} frames. {im_path.parent.name}"
                )
                continue
            if lax_bbox_func is not None:
                if not all(
                    lax_im.shape[0] >= lax_bbox_size[0]
                    and lax_im.shape[1] >= lax_bbox_size[1]
                    for lax_im in lax_ims
                ):
                    print(
                        f"Found an LAX image with suspiciously low resolution: {lax_ims[0].shape[0:2], lax_ims[1].shape[0:2], lax_ims[2].shape[0:2]} pixels. {im_path.parent.name}"
                    )
                    continue
            indices.append(int(parent.parent.name.strip()))
        except Exception as e:
            continue
    return indices


def process_cmr_images(
    load_dir: Union[str, Path],
    prep_dir: Union[str, Path],
    num_cases: int = -1,
    case_start_idx: int = 0,
    sax_file_name: str = "SAX.nii.gz",
    lax_file_name: List[str] = ["LAX_2Ch.nii.gz", "LAX_3Ch.nii.gz", "LAX_4Ch.nii.gz"],
    seg_sax_file_name: str = "SAX_nnUNetSeg.nii.gz",
    seg_lax_file_name: List[Optional[str]] = [
        "LAX_2Ch_nnUNetSeg.nii.gz",
        "LAX_3Ch_nnUNetSeg.nii.gz",
        "LAX_4Ch_nnUNetSeg.nii.gz",
    ],
    file_name: str = "processed_data.npy",
    sax_bbox_func: Optional[Callable] = get_2D_slice_bbox_with_fixed_size,
    lax_bbox_func: Optional[Callable] = get_2D_central_bbox,
    sax_bbox_size: Union[List[int], Tuple[int, int]] = [128, 128],
    lax_bbox_size: Union[List[int], Tuple[int, int]] = [128, 128],
    id_list: Optional[List[int]] = None,
    replace_processed: Optional[bool] = False,
) -> List[int]:
    """Crop and process the CMR raw files, converting them into single .npy/.npz outputs.

    :param load_dir: Data directory where subjects are layed out.
    :param prep_dir: Directory where processed data will be saved.
    :param num_cases: Number of subjects to extract before breaking out of the data gathering loop.
                      If value is negative, collect all subjects.
    :param case_start_idx: Index of subject at which to start. Used to avoid loading same subjects
                            into train/val/test datasets. Ex: If you want to use 1000 subjects for
                            training and 100 for validation -> Train_start_idx = 0, Val_start_idx = 1000
    :param bbox_func: Function to be used to extract bounding boxes.
                      Should follow format [Y1, X1, ..., Y2, X2, ...]
    :param sax_bbox_size: the size of sax bounding box
    :param lax_bbox_size: the size of lax bounding box
    :param img_file_name: Naming convention of the image files you are intending to extract.
    :param seg_file_name: Naming convention of the segmentation files you are intending to extract.
    :param id_list: The id list from target dataframe.

    :return processed_npy_paths: List of paths to the processed npy files.
    """
    assert num_cases != 0
    assert os.path.exists(prep_dir), f"Processed directory {prep_dir} does not exist."
    count = 0
    # processed_npy_paths = []
    processed_case_ids = []
    start_time = time.time()
    load_dir = Path(load_dir)
    prep_dir = Path(prep_dir)

    dir_id_list = sorted(os.listdir(str(load_dir)))
    for i, parent in tqdm.tqdm(
        enumerate(dir_id_list), desc="index list", unit=" iter", position=1, leave=False
    ):
        if i < case_start_idx:
            continue  # Skip all subjects not belonging to this dataset
        if count >= num_cases > 0:
            break  # If we have collected enough subjects, break
        if int(parent) not in id_list:
            continue  # Skip all subjects not in the image dataset

        processed_npy_path = prep_dir / parent / file_name
        if not replace_processed and os.path.exists(processed_npy_path):
            # processed_npy_paths.append(processed_npy_path)
            processed_case_ids.append(int(parent))
            count += 1
            continue  # Skip all subjects that have already been processed
        else:
            sax_path = load_dir / parent / "Instance_2" / sax_file_name
            lax_path = [load_dir / parent / "Instance_2" / i for i in lax_file_name]
            sax_seg_path = (
                load_dir / parent / "Instance_2" / "nnUNet_segs" / seg_sax_file_name
            )
            lax_seg_path = [
                (
                    load_dir / parent / "Instance_2" / "nnUNet_segs" / i
                    if i is not None
                    else None
                )
                for i in seg_lax_file_name
            ]

            if sax_bbox_func is not None:
                # Get bounding box of where foreground mask is present
                seg = nib.load(sax_seg_path).get_fdata()
                sax_bbox = sax_bbox_func(seg, sax_bbox_size)
            if lax_bbox_func is not None:
                lax_bbox = np.zeros((len(lax_path), 4), dtype=np.int32)
                for i in range(len(lax_path)):
                    lax_im = nib.load(lax_path[i]).get_fdata()
                    lax_bbox[i] = lax_bbox_func(lax_im.shape[:2], lax_bbox_size)

            # Load cropped sax images and segmentations into arrays
            nii_sax = nib.load(sax_path).get_fdata().astype(np.float32)
            nii_seg_sax = nib.load(sax_seg_path).get_fdata().astype(np.int32)
            raw_shape = nii_sax.shape
            assert len(sax_bbox) % 2 == 0
            if len(sax_bbox) // 2 == 2:
                sax_bbox = (*sax_bbox[:2], 0, *sax_bbox[-2:], raw_shape[2])
            if len(sax_bbox) // 2 == 3:
                sax_bbox = (*sax_bbox[:3], 0, *sax_bbox[-3:], sax_bbox[3])
            idx_slices = (
                slice(sax_bbox[0], sax_bbox[0 + len(sax_bbox) // 2]),
                slice(sax_bbox[1], sax_bbox[1 + len(sax_bbox) // 2]),
            )
            sax_arrs = nii_sax[idx_slices]
            seg_sax_arrs = nii_seg_sax[idx_slices]

            # Load cropped long images into an array
            laxs = []
            seg_laxs = []
            for i in range(len(lax_path)):
                nii_lax = nib.load(lax_path[i]).get_fdata().astype(np.float32)
                raw_shape = nii_lax.shape
                assert len(lax_bbox[i]) % 2 == 0
                if len(lax_bbox[i]) // 2 == 2:
                    lax_bbox_ = (*lax_bbox[i][:2], 0, *lax_bbox[i][-2:], raw_shape[2])
                if len(lax_bbox[i]) // 2 == 3:
                    lax_bbox_ = (*lax_bbox[i][:3], 0, *lax_bbox[i][-3:], lax_bbox[i][3])
                idx_slices = (
                    slice(lax_bbox_[0], lax_bbox_[0 + len(lax_bbox_) // 2]),
                    slice(lax_bbox_[1], lax_bbox_[1 + len(lax_bbox_) // 2]),
                )
                lax = nii_lax[idx_slices]
                laxs.append(lax)

                if lax_seg_path[i] is not None:
                    nii_seg_lax = nib.load(lax_seg_path[i]).get_fdata().astype(np.int32)
                    seg_lax = nii_seg_lax[idx_slices]
                    seg_laxs.append(seg_lax)
                else:
                    pad_seg_lax = np.zeros_like(lax).astype(np.int32)
                    seg_laxs.append(pad_seg_lax)

            lax_arrs = np.stack(laxs, axis=2).squeeze(-2)
            seg_lax_arrs = np.stack(seg_laxs, axis=2).squeeze(-2)

            # Save to a numpy file
            if not os.path.exists(processed_npy_path.parent):
                os.makedirs(processed_npy_path.parent)
            if replace_processed and os.path.exists(processed_npy_path):
                os.remove(processed_npy_path)
            processed_npy = {
                "sax": sax_arrs,
                "lax": lax_arrs,
                "seg_sax": seg_sax_arrs,
                "seg_lax": seg_lax_arrs,
            }
            if file_name[-4:] == ".npy":
                np.save(processed_npy_path, processed_npy)
            elif file_name[-4:] == ".npz":
                np.savez(
                    processed_npy_path,
                    sax=sax_arrs,
                    lax=lax_arrs,
                    seg_sax=seg_sax_arrs,
                    seg_lax=seg_lax_arrs,
                )
            else:
                raise NotImplementedError
            # processed_npy_paths.append(processed_npy_path)
            processed_case_ids.append(int(parent))

            count += 1

    if num_cases > 0 and count != num_cases:
        raise ValueError(
            f"Did not find required amount of cases ({num_cases}) in directory: {load_dir}"
        )

    elapsed = time.time() - start_time
    print(
        f"Processed {count} cases in out of {len(dir_id_list)} in {elapsed//60}m {int(elapsed%60)}s."
    )
    print(f"The data searching range is from {case_start_idx} to {i}.")
    return processed_case_ids


def get_subject_sax_range(
    npy_path: Union[str, Path],
    z_num: Optional[int] = None,
) -> Tuple[int, int]:
    """Return the start and end z slice index for the SAX images bounding box.
    
    Args:
        npy_path (str or Path): File to the saved numpy zip.
        z_num (int, optional): Number of slices in the final SAX dimension.

    Returns:
        Tuple[int, int]: The minimum sequence index, maximum sequence index.
    """

    assert os.path.exists(npy_path), f"File not found: {npy_path}"
    if npy_path.name[-4:] == ".npy":
        process_npy = np.load(npy_path, allow_pickle=True).item()
    elif npy_path.name[-4:] == ".npz":
        process_npy = np.load(npy_path)

    sax_im_data = process_npy["sax"]  # [H, W, S, T]
    seg_sax_data = process_npy["seg_sax"]  # [H, W, S, T]

    # Select only the from 3rd to 3+z_num slices with segmentation map
    if z_num is not None:
        z_seg_start = (seg_sax_data[..., 0] == 1).any((0, 1)).argmax() + 2
        z_max = min(z_seg_start + z_num, seg_sax_data.shape[-2])
        z_min = z_max - z_num
        seg_sax_data = seg_sax_data[..., z_min:z_max, :]  # [H, W, z_num, T]
        sax_im_data = sax_im_data[..., z_min:z_max, :]
        assert (
            sax_im_data.shape[-2] == z_num
        ), f"Img path: {npy_path}, shape: {sax_im_data.shape}"
    return z_min, z_max
