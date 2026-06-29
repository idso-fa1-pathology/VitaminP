"""Shared helpers for the public-dataset preprocessing command-line tools.

The notebooks use Zarr v2 arrays compressed with Blosc.  Keeping the small
amount of shared I/O here makes the individual scripts easier to audit while
preserving that on-disk format.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numcodecs
import numpy as np
import pandas as pd
import scipy.io
import zarr


PATCH_SIZE = 512


def base_parser(description: str, *, default_seed: int | None = None) -> argparse.ArgumentParser:
    """Create the path arguments shared by every converter."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Raw dataset root, for example /path/to/raw_dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination root, for example /path/to/output_zarr.",
    )
    if default_seed is not None:
        parser.add_argument(
            "--seed",
            type=int,
            default=default_seed,
            help="Random seed used for the dataset split.",
        )
    return parser


def require_directory(path: Path, label: str = "directory") -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def require_file(path: Path, label: str = "file") -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def prepare_directory(path: Path) -> Path:
    """Replace one generated output directory, leaving its siblings intact."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_instance_tiff(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"OpenCV could not read mask: {path}")
    return mask.astype(np.int32, copy=False)


def read_mat_instance(path: Path, keys: Sequence[str] = ("inst_map", "instance_map", "map", "mask")) -> np.ndarray:
    """Read the first named, or otherwise first 2-D, MATLAB mask array."""
    data = scipy.io.loadmat(str(path))
    for key in keys:
        value = data.get(key)
        if isinstance(value, np.ndarray) and value.ndim == 2:
            return value.astype(np.int32, copy=False)
    for key, value in data.items():
        if not key.startswith("__") and isinstance(value, np.ndarray) and value.ndim == 2:
            return value.astype(np.int32, copy=False)
    raise KeyError(f"No 2-D instance mask found in {path}; keys={sorted(data)}")


def pad_to_multiple(array: np.ndarray, multiple: int = PATCH_SIZE) -> np.ndarray:
    """Zero-pad the bottom and right edges to a positive multiple."""
    height, width = array.shape[:2]
    target_h = max(multiple, int(np.ceil(height / multiple)) * multiple)
    target_w = max(multiple, int(np.ceil(width / multiple)) * multiple)
    return pad_to_shape(array, target_h, target_w)


def pad_to_minimum(array: np.ndarray, minimum: int = PATCH_SIZE) -> np.ndarray:
    height, width = array.shape[:2]
    return pad_to_shape(array, max(height, minimum), max(width, minimum))


def pad_to_shape(array: np.ndarray, height: int, width: int) -> np.ndarray:
    pad_h = height - array.shape[0]
    pad_w = width - array.shape[1]
    if pad_h < 0 or pad_w < 0:
        raise ValueError("pad_to_shape cannot crop an array")
    if pad_h == 0 and pad_w == 0:
        return array
    padding = ((0, pad_h), (0, pad_w))
    if array.ndim == 3:
        padding += ((0, 0),)
    return np.pad(array, padding, mode="constant", constant_values=0)


def edge_aligned_starts(length: int, patch_size: int = PATCH_SIZE, stride: int | None = None) -> list[int]:
    """Return starts that include a final crop aligned with the far edge."""
    if length < patch_size:
        raise ValueError(f"length {length} is smaller than patch size {patch_size}; pad first")
    if length == patch_size:
        return [0]
    step = stride or patch_size
    starts = list(range(0, length - patch_size + 1, step))
    starts.append(length - patch_size)
    return sorted(set(starts))


def tile_pair(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    patch_size: int = PATCH_SIZE,
    stride: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Image/mask shape mismatch: {image.shape[:2]} vs {mask.shape[:2]}")
    y_starts = edge_aligned_starts(image.shape[0], patch_size, stride)
    x_starts = edge_aligned_starts(image.shape[1], patch_size, stride)
    return [
        (
            image[y : y + patch_size, x : x + patch_size],
            mask[y : y + patch_size, x : x + patch_size],
            y,
            x,
        )
        for y in y_starts
        for x in x_starts
    ]


def shuffled_train_val(items: Iterable[Any], val_ratio: float, seed: int) -> tuple[list[Any], list[Any]]:
    values = sorted(items, key=lambda value: str(value))
    random.Random(seed).shuffle(values)
    n_val = int(len(values) * val_ratio)
    if n_val == 0 and len(values) > 1 and val_ratio > 0:
        n_val = 1
    return values[n_val:], values[:n_val]


def split_70_15_15(items: Sequence[Any], seed: int) -> tuple[list[Any], list[Any], list[Any]]:
    """Reproduce the notebooks' two-stage sklearn 70/15/15 split."""
    from sklearn.model_selection import train_test_split

    if len(items) < 3:
        raise ValueError("At least three image/mask pairs are required for a 70/15/15 split")
    train_val, test = train_test_split(list(items), test_size=0.15, random_state=seed, shuffle=True)
    train, val = train_test_split(
        train_val,
        test_size=0.15 / 0.85,
        random_state=seed,
        shuffle=True,
    )
    return train, val, test


def write_image_mask_zarr(
    output_dir: Path,
    images: np.ndarray,
    masks: np.ndarray,
    metadata: Sequence[dict[str, Any]],
) -> None:
    """Write the flat image/mask/metadata layout consumed by dataset.py."""
    if len(images) != len(masks) or len(images) != len(metadata):
        raise ValueError("Images, masks, and metadata must contain the same number of records")
    output_dir = prepare_directory(output_dir)
    compressor = numcodecs.Blosc(cname="zstd", clevel=3)
    image_chunks = (1,) + tuple(images.shape[1:])
    mask_chunks = (1,) + tuple(masks.shape[1:])
    z_images = zarr.open_array(
        str(output_dir / "images.zarr"),
        mode="w",
        shape=images.shape,
        chunks=image_chunks,
        dtype=images.dtype,
        compressor=compressor,
    )
    z_masks = zarr.open_array(
        str(output_dir / "nuclei_masks.zarr"),
        mode="w",
        shape=masks.shape,
        chunks=mask_chunks,
        dtype=np.int32,
        compressor=compressor,
    )
    z_images[:] = images
    z_masks[:] = masks.astype(np.int32, copy=False)
    pd.DataFrame(metadata).to_csv(output_dir / "metadata.csv", index=False)


def write_he_sample(
    sample_dir: Path,
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    metadata: Sequence[dict[str, Any]],
) -> None:
    write_image_mask_zarr(
        sample_dir,
        np.stack(images).astype(np.uint8, copy=False),
        np.stack(masks).astype(np.int32, copy=False),
        metadata,
    )


def write_mif_sample(
    sample_dir: Path,
    image: np.ndarray,
    nuclei_mask: np.ndarray,
    cell_mask: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Write TissueNet's per-sample MIF layout."""
    mif_dir = prepare_directory(sample_dir / "mif")
    compressor = numcodecs.Blosc(cname="zstd", clevel=3)
    arrays = {
        "images.zarr": (image[np.newaxis], image.dtype),
        "nuclei_masks.zarr": (nuclei_mask[np.newaxis], np.int32),
        "cell_masks.zarr": (cell_mask[np.newaxis], np.int32),
    }
    for name, (values, dtype) in arrays.items():
        values = values.astype(dtype, copy=False)
        target = zarr.open_array(
            str(mif_dir / name),
            mode="w",
            shape=values.shape,
            chunks=values.shape,
            dtype=dtype,
            compressor=compressor,
        )
        target[:] = values
    pd.DataFrame([metadata]).to_csv(mif_dir / "metadata.csv", index=False)


def stack_or_raise(values: Sequence[np.ndarray], label: str) -> np.ndarray:
    if not values:
        raise RuntimeError(f"No {label} were produced; check the input layout and annotations")
    return np.stack(values)


def count_instances(mask: np.ndarray) -> int:
    return int(np.count_nonzero(np.unique(mask) > 0))

