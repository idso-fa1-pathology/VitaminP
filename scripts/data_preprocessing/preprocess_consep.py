#!/usr/bin/env python3
"""Convert CoNSeP PNG images and MATLAB instance labels to Zarr.

Expected input: ``Train/Images``, ``Train/Labels``, ``Test/Images``, and
``Test/Labels``.  Label files share an image stem and contain ``inst_map``.

Split: the source test set remains test; source training images are shuffled
and split 90/10 before tiling (default seed 42).  Edge-aligned 512x512 tiles
are written to flat ``consep_train``, ``consep_val``, and ``consep_test``
directories with ``images.zarr``, ``nuclei_masks.zarr``, and ``metadata.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from _common import (
    base_parser,
    count_instances,
    pad_to_minimum,
    read_mat_instance,
    read_rgb,
    require_directory,
    shuffled_train_val,
    stack_or_raise,
    tile_pair,
    write_image_mask_zarr,
)


def convert_split(files: list[Path], labels: Path, output_dir: Path, name: str) -> None:
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    for image_path in tqdm(files, desc=name):
        label_path = labels / f"{image_path.stem}.mat"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing CoNSeP label: {label_path}")
        image = read_rgb(image_path)
        mask = read_mat_instance(label_path, keys=("inst_map",))
        original_h, original_w = image.shape[:2]
        image = pad_to_minimum(image)
        mask = pad_to_minimum(mask)
        for tile_y, (image_tile, mask_tile, y, x) in enumerate(tile_pair(image, mask)):
            images.append(image_tile)
            masks.append(mask_tile)
            metadata.append(
                {
                    "original_file": image_path.name,
                    "split": name,
                    "tile_index": tile_y,
                    "y_start": y,
                    "y_end": y + 512,
                    "x_start": x,
                    "x_end": x + 512,
                    "original_height": original_h,
                    "original_width": original_w,
                    "num_cells": count_instances(mask_tile),
                }
            )
    write_image_mask_zarr(
        output_dir / name,
        stack_or_raise(images, f"tiles for {name}").astype(np.uint8),
        stack_or_raise(masks, f"masks for {name}").astype(np.int32),
        metadata,
    )


def main() -> None:
    parser = base_parser("Convert CoNSeP to VitaminP Zarr data.", default_seed=42)
    args = parser.parse_args()
    root = require_directory(args.input_dir, "CoNSeP dataset root")
    output = args.output_dir.expanduser().resolve()

    train_images = require_directory(root / "Train" / "Images", "CoNSeP train images")
    train_labels = require_directory(root / "Train" / "Labels", "CoNSeP train labels")
    test_images = require_directory(root / "Test" / "Images", "CoNSeP test images")
    test_labels = require_directory(root / "Test" / "Labels", "CoNSeP test labels")

    train, val = shuffled_train_val(train_images.glob("*.png"), 0.10, args.seed)
    convert_split(train, train_labels, output, "consep_train")
    convert_split(val, train_labels, output, "consep_val")
    convert_split(sorted(test_images.glob("*.png")), test_labels, output, "consep_test")


if __name__ == "__main__":
    main()

