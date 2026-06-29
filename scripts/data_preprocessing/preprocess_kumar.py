#!/usr/bin/env python3
"""Convert Kumar images and MATLAB instance maps to split-level Zarr.

Expected input: each of ``train``, ``val``, and ``test`` contains ``Images``
and ``Labels`` directories; TIFF images and MAT labels share a stem.

Split: source folders are preserved exactly (no random split).  Each image is
tiled into 512x512 patches with stride 450 and a final edge-aligned tile.
Outputs are ``kumar_train``, ``kumar_val``, and ``kumar_test`` with flat Zarr
arrays and per-patch metadata.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from _common import (
    base_parser,
    pad_to_minimum,
    read_mat_instance,
    read_rgb,
    require_directory,
    stack_or_raise,
    tile_pair,
    write_image_mask_zarr,
)


def convert_source(root: Path, output: Path, source_name: str) -> None:
    image_dir = require_directory(root / source_name / "Images", f"Kumar {source_name} images")
    label_dir = require_directory(root / source_name / "Labels", f"Kumar {source_name} labels")
    output_name = f"kumar_{source_name}"
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    for image_path in tqdm(sorted(image_dir.glob("*.tif")), desc=output_name):
        label_path = label_dir / f"{image_path.stem}.mat"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing Kumar label: {label_path}")
        image = read_rgb(image_path)
        mask = read_mat_instance(label_path)
        original_h, original_w = image.shape[:2]
        image = pad_to_minimum(image)
        mask = pad_to_minimum(mask)
        for image_tile, mask_tile, y, x in tile_pair(image, mask, stride=450):
            images.append(image_tile)
            masks.append(mask_tile)
            metadata.append(
                {
                    "original_file": image_path.name,
                    "patch_y": y,
                    "patch_x": x,
                    "original_height": original_h,
                    "original_width": original_w,
                }
            )
    write_image_mask_zarr(
        output / output_name,
        stack_or_raise(images, f"tiles for {output_name}").astype(np.uint8),
        stack_or_raise(masks, f"masks for {output_name}").astype(np.int32),
        metadata,
    )


def main() -> None:
    parser = base_parser("Convert Kumar nuclei data to VitaminP Zarr data.")
    args = parser.parse_args()
    root = require_directory(args.input_dir, "Kumar dataset root")
    output = args.output_dir.expanduser().resolve()
    for split in ("train", "val", "test"):
        convert_source(root, output, split)


if __name__ == "__main__":
    main()

