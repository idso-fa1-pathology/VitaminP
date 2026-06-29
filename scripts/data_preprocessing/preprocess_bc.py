#!/usr/bin/env python3
"""Convert BC/DeepLIIF strips to split-level Zarr arrays.

Expected input: ``train/*.png`` plus ``val/*.png`` (``validation`` is also
accepted).  Every 512x3072 strip contains six horizontal 512-pixel panels.
Panel 1 is saved as RGB input and panel 6 is converted from red/blue semantic
labels to instances with the notebook's watershed procedure.

Split: source validation -> test; source train -> 90% train / 10% validation
at image level before conversion (default seed 42).  Output directories are
``bc_train``, ``bc_val``, and ``bc_test``, each with two Zarr arrays and CSV.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from _common import (
    base_parser,
    count_instances,
    read_rgb,
    require_directory,
    shuffled_train_val,
    stack_or_raise,
    write_image_mask_zarr,
)


def semantic_panel_to_instances(panel: np.ndarray, image: np.ndarray) -> np.ndarray:
    positive = (panel[:, :, 0] > 50).astype(np.uint8)
    negative = (panel[:, :, 2] > 50).astype(np.uint8)
    num_positive, positive_markers = cv2.connectedComponents(positive)
    _, negative_markers = cv2.connectedComponents(negative)

    markers = positive_markers.copy()
    negative_pixels = negative_markers > 0
    markers[negative_pixels] = negative_markers[negative_pixels] + num_positive - 1
    markers += 1

    all_cells = cv2.bitwise_or(positive, negative)
    dilated = cv2.dilate(all_cells, np.ones((5, 5), np.uint8), iterations=1)
    unknown = (dilated > 0) & (all_cells == 0)
    markers[unknown] = 0

    cv2.watershed(image, markers)
    markers[markers == -1] = 1
    return (markers - 1).astype(np.int32)


def convert_files(files: list[Path], output_dir: Path, split_name: str) -> None:
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    for path in tqdm(files, desc=split_name):
        strip = read_rgb(path)
        if strip.shape[:2] != (512, 3072):
            raise ValueError(f"Expected a 512x3072 DeepLIIF strip, got {strip.shape}: {path}")
        image = strip[:, :512]
        mask = semantic_panel_to_instances(strip[:, 2560:3072], image)
        images.append(image)
        masks.append(mask)
        metadata.append(
            {
                "original_file": path.name,
                "split": split_name,
                "original_height": 512,
                "original_width": 512,
                "num_cells": count_instances(mask),
            }
        )
    write_image_mask_zarr(
        output_dir / split_name,
        stack_or_raise(images, f"images for {split_name}").astype(np.uint8),
        stack_or_raise(masks, f"masks for {split_name}").astype(np.int32),
        metadata,
    )


def main() -> None:
    parser = base_parser("Convert raw BC/DeepLIIF strips to VitaminP Zarr data.", default_seed=42)
    args = parser.parse_args()
    input_dir = require_directory(args.input_dir, "BC dataset root")
    output_dir = args.output_dir.expanduser().resolve()

    train_dir = require_directory(input_dir / "train", "BC train directory")
    source_test_dir = input_dir / "val"
    if not source_test_dir.is_dir():
        source_test_dir = input_dir / "validation"
    source_test_dir = require_directory(source_test_dir, "BC validation/test directory")

    train_files, val_files = shuffled_train_val(train_dir.glob("*.png"), 0.10, args.seed)
    test_files = sorted(source_test_dir.glob("*.png"))
    convert_files(train_files, output_dir, "bc_train")
    convert_files(val_files, output_dir, "bc_val")
    convert_files(test_files, output_dir, "bc_test")


if __name__ == "__main__":
    main()

