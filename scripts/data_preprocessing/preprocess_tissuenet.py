#!/usr/bin/env python3
"""Convert TissueNet NPZ splits to per-sample MIF Zarr data.

Expected input: ``tissuenet_v1.1_{train,val,test}.npz`` containing ``X``
two-channel images and ``y`` two-channel instance masks.  Notebook semantics
are preserved: mask channel 1 is nuclei and channel 0 is cells; image channels
are independently percentile-normalized to uint16.

Split: official NPZ train/val/test files are preserved.  Native 512 images are
saved directly; four consecutive native 256 images are stitched into a 512
mosaic with instance-ID offsets.  ``tissuenet_*`` directories contain one
sample subdirectory with ``mif/`` Zarr arrays, matching dataset.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from _common import base_parser, prepare_directory, require_directory, require_file, write_mif_sample


def normalize_uint16(image: np.ndarray) -> np.ndarray:
    output = np.zeros_like(image, dtype=np.uint16)
    for channel in range(2):
        values = image[:, :, channel]
        if values.max() > values.min():
            low, high = np.percentile(values, (1, 99))
            if high > low:
                values = np.clip((values - low) / (high - low), 0, 1)
                output[:, :, channel] = (values * 65535).astype(np.uint16)
            else:
                output[:, :, channel] = values.astype(np.uint16)
        else:
            output[:, :, channel] = values.astype(np.uint16)
    return output


def offset_instances(source: np.ndarray, offset: int) -> tuple[np.ndarray, int]:
    source = source.astype(np.int32, copy=False)
    shifted = np.zeros_like(source)
    foreground = source > 0
    shifted[foreground] = source[foreground] + offset
    return shifted, offset + (int(source.max()) if np.any(foreground) else 0)


def write_direct(
    images: np.ndarray,
    masks: np.ndarray,
    split_dir: Path,
    split_name: str,
) -> None:
    for index in tqdm(range(len(images)), desc=split_name):
        sample = f"tissuenet_sample_{index:06d}"
        write_mif_sample(
            split_dir / sample,
            normalize_uint16(images[index]),
            masks[index, :, :, 1].astype(np.int32),
            masks[index, :, :, 0].astype(np.int32),
            {"patch_idx": 0, "sample": sample, "split": split_name, "type": "direct_512"},
        )


def write_mosaics(
    images: np.ndarray,
    masks: np.ndarray,
    split_dir: Path,
    split_name: str,
) -> None:
    quadrants = ((0, 0), (0, 256), (256, 0), (256, 256))
    starts = list(range(0, len(images), 4))
    for mosaic_index, start in enumerate(tqdm(starts, desc=split_name)):
        image = np.zeros((512, 512, 2), dtype=np.uint16)
        nuclei = np.zeros((512, 512), dtype=np.int32)
        cells = np.zeros((512, 512), dtype=np.int32)
        nuclei_offset = 0
        cell_offset = 0
        source_indices: list[int] = []
        for quadrant, index in enumerate(range(start, min(start + 4, len(images)))):
            y, x = quadrants[quadrant]
            source_indices.append(index)
            image[y : y + 256, x : x + 256] = normalize_uint16(images[index])
            shifted, nuclei_offset = offset_instances(masks[index, :, :, 1], nuclei_offset)
            nuclei[y : y + 256, x : x + 256] = shifted
            shifted, cell_offset = offset_instances(masks[index, :, :, 0], cell_offset)
            cells[y : y + 256, x : x + 256] = shifted
        sample = f"tissuenet_mosaic_{mosaic_index:06d}"
        write_mif_sample(
            split_dir / sample,
            image,
            nuclei,
            cells,
            {
                "patch_idx": 0,
                "sample": sample,
                "split": split_name,
                "type": "mosaic_4x_256",
                "source_indices": str(source_indices),
            },
        )


def convert_split(npz_path: Path, output: Path, split: str) -> None:
    data = np.load(npz_path, allow_pickle=True)
    images, masks = data["X"], data["y"]
    if images.ndim != 4 or images.shape[-1] != 2 or masks.shape[-1] != 2:
        raise ValueError(f"Unexpected TissueNet shapes: X={images.shape}, y={masks.shape}")
    split_name = f"tissuenet_{split}"
    split_dir = prepare_directory(output / split_name)
    if images.shape[1:3] == (256, 256):
        write_mosaics(images, masks, split_dir, split_name)
    elif images.shape[1:3] == (512, 512):
        write_direct(images, masks, split_dir, split_name)
    else:
        raise ValueError(f"TissueNet images must be 256 or 512 square, got {images.shape[1:3]}")


def main() -> None:
    parser = base_parser("Convert TissueNet NPZ files to VitaminP MIF Zarr data.")
    args = parser.parse_args()
    root = require_directory(args.input_dir, "TissueNet dataset root")
    output = args.output_dir.expanduser().resolve()
    for split in ("train", "val", "test"):
        npz = require_file(root / f"tissuenet_v1.1_{split}.npz", f"TissueNet {split} NPZ")
        convert_split(npz, output, split)


if __name__ == "__main__":
    main()

