#!/usr/bin/env python3
"""Convert the TNBC nuclei dataset to patient-disjoint Zarr stacks.

Expected input: ``Slide_01`` ... ``Slide_11`` image directories and matching
``GT_01`` ... ``GT_11`` binary-mask directories.  Same-named PNG files form a
pair.  Connected components convert each binary mask to instance IDs.

Split: patient slides 1-7 train, 8-9 validation, and 10-11 test.  Assignment
is performed before padding to 512 multiples and tiling.  Each ``tnbc_*``
split contains one Zarr subdirectory per slide, as expected by dataset.py.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from tqdm import tqdm

from _common import (
    base_parser,
    count_instances,
    pad_to_multiple,
    prepare_directory,
    read_rgb,
    require_directory,
    tile_pair,
    write_he_sample,
)


SPLITS = {
    "tnbc_train": tuple(range(1, 8)),
    "tnbc_val": (8, 9),
    "tnbc_test": (10, 11),
}


def convert_slide(root: Path, split_dir: Path, slide_number: int) -> None:
    slide_name = f"Slide_{slide_number:02d}"
    image_dir = require_directory(root / slide_name, f"TNBC {slide_name}")
    mask_dir = require_directory(root / f"GT_{slide_number:02d}", f"TNBC GT_{slide_number:02d}")
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    for image_path in tqdm(sorted(image_dir.glob("*.png")), desc=slide_name, leave=False):
        mask_path = mask_dir / image_path.name
        binary = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if binary is None:
            raise FileNotFoundError(f"Missing or unreadable TNBC mask: {mask_path}")
        image = read_rgb(image_path)
        original_h, original_w = image.shape[:2]
        instance_mask, _ = ndimage.label(binary > 127)
        image = pad_to_multiple(image)
        instance_mask = pad_to_multiple(instance_mask.astype(np.int32))
        for image_tile, mask_tile, y, x in tile_pair(image, instance_mask):
            images.append(image_tile)
            masks.append(mask_tile)
            metadata.append(
                {
                    "slide": slide_name,
                    "original_file": image_path.name,
                    "x": x,
                    "y": y,
                    "original_height": original_h,
                    "original_width": original_w,
                    "num_nuclei": count_instances(mask_tile),
                }
            )
    if not images:
        raise RuntimeError(f"No TNBC PNG files found in {image_dir}")
    write_he_sample(split_dir / slide_name, images, masks, metadata)


def main() -> None:
    parser = base_parser("Convert TNBC to patient-disjoint VitaminP Zarr data.")
    args = parser.parse_args()
    root = require_directory(args.input_dir, "TNBC dataset root")
    output = args.output_dir.expanduser().resolve()
    for split_name, slide_numbers in SPLITS.items():
        split_dir = prepare_directory(output / split_name)
        for slide_number in slide_numbers:
            convert_slide(root, split_dir, slide_number)


if __name__ == "__main__":
    main()

