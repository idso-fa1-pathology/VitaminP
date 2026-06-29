#!/usr/bin/env python3
"""Convert CPM-17 PNG/MAT pairs to flat split-level Zarr arrays.

Expected input: ``train/Images``, ``train/Labels``, ``test/Images``, and
``test/Labels`` with same-stem PNG/MAT pairs.

Split: source test remains test; source train is shuffled and split 95/5 at
image level before tiling (default seed 42).  Images smaller than 512 are
bottom/right padded; larger images use 512x512 edge-aligned tiles with stride
400.  Outputs are ``cpm17_train``, ``cpm17_val``, and ``cpm17_test``.
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


def convert_split(files: list[Path], label_dir: Path, output: Path, name: str) -> None:
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    for image_path in tqdm(files, desc=name):
        label_path = label_dir / f"{image_path.stem}.mat"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing CPM-17 label: {label_path}")
        image = read_rgb(image_path)
        mask = read_mat_instance(label_path, keys=("inst_map", "instance_map", "map", "mask", "scmap"))
        original_h, original_w = image.shape[:2]
        was_padded = original_h < 512 or original_w < 512
        image = pad_to_minimum(image)
        mask = pad_to_minimum(mask)
        for image_tile, mask_tile, y, x in tile_pair(image, mask, stride=400):
            images.append(image_tile)
            masks.append(mask_tile)
            metadata.append(
                {
                    "original_file": image_path.name,
                    "original_h": original_h,
                    "original_w": original_w,
                    "patch_y": y,
                    "patch_x": x,
                    "was_padded": was_padded,
                }
            )
    write_image_mask_zarr(
        output / name,
        stack_or_raise(images, f"tiles for {name}").astype(np.uint8),
        stack_or_raise(masks, f"masks for {name}").astype(np.int32),
        metadata,
    )


def main() -> None:
    parser = base_parser("Convert CPM-17 to VitaminP Zarr data.", default_seed=42)
    args = parser.parse_args()
    root = require_directory(args.input_dir, "CPM-17 dataset root")
    output = args.output_dir.expanduser().resolve()

    train_images = require_directory(root / "train" / "Images", "CPM-17 train images")
    train_labels = require_directory(root / "train" / "Labels", "CPM-17 train labels")
    test_images = require_directory(root / "test" / "Images", "CPM-17 test images")
    test_labels = require_directory(root / "test" / "Labels", "CPM-17 test labels")

    all_train = sorted(train_images.glob("*.png"))
    permutation = np.random.RandomState(args.seed).permutation(len(all_train))
    n_val = int(len(all_train) * 0.05)
    if n_val == 0 and len(all_train) > 1:
        n_val = 1
    val = [all_train[index] for index in permutation[:n_val]]
    train = [all_train[index] for index in permutation[n_val:]]
    convert_split(train, train_labels, output, "cpm17_train")
    convert_split(val, train_labels, output, "cpm17_val")
    convert_split(sorted(test_images.glob("*.png")), test_labels, output, "cpm17_test")


if __name__ == "__main__":
    main()

