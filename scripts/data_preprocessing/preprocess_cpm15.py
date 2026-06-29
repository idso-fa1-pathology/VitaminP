#!/usr/bin/env python3
"""Convert CPM-15 PNG/MAT pairs to per-image Zarr stacks.

Expected input: ``Images/*.png`` and same-stem ``Labels/*.mat`` files whose
instance map is stored in ``inst_map`` (with a generic 2-D-array fallback).

The source notebook defines no train/validation/test partition.  All 15 images
are therefore preserved in ``cpm15_full`` and can be assigned by the training
configuration without inventing a new split.  Images are padded to at least
512 and tiled into non-overlapping, edge-aligned 512x512 patches.  Each image
has its own Zarr subdirectory, which is the layout consumed by dataset.py.
"""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from _common import (
    base_parser,
    pad_to_minimum,
    prepare_directory,
    read_mat_instance,
    read_rgb,
    require_directory,
    tile_pair,
    write_he_sample,
)


def main() -> None:
    parser = base_parser("Convert CPM-15 to VitaminP Zarr data.")
    args = parser.parse_args()
    root = require_directory(args.input_dir, "CPM-15 dataset root")
    image_dir = require_directory(root / "Images", "CPM-15 images")
    label_dir = require_directory(root / "Labels", "CPM-15 labels")
    split_dir = prepare_directory(args.output_dir.expanduser().resolve() / "cpm15_full")

    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No CPM-15 PNG files found in {image_dir}")
    for image_path in tqdm(image_paths, desc="cpm15_full"):
        label_path = label_dir / f"{image_path.stem}.mat"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing CPM-15 label: {label_path}")
        image = read_rgb(image_path)
        mask = read_mat_instance(label_path, keys=("inst_map", "label"))
        image = pad_to_minimum(image)
        mask = pad_to_minimum(mask)
        tiles = tile_pair(image, mask)
        write_he_sample(
            split_dir / image_path.stem,
            [tile[0] for tile in tiles],
            [tile[1] for tile in tiles],
            [
                {"original_file": image_path.name, "x": x, "y": y}
                for _, _, y, x in tiles
            ],
        )


if __name__ == "__main__":
    main()

