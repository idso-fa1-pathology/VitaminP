#!/usr/bin/env python3
"""Convert PanopTILs RGB/CSV polygons to per-image Zarr stacks.

Expected input: a ``tcga`` directory (or a parent containing it) with
``rgbs/*.png`` and same-stem ``csv/*.csv`` files.  Only rows whose ``type`` is
``polyline`` are rasterized, with one unique nucleus ID per row.

The source notebook defines no random split, so all samples are preserved in
``panoptils_full`` for assignment by the training configuration.  Images are
padded to at least 512 and tiled into non-overlapping edge-aligned 512x512
patches; each source image has a loader-compatible Zarr subdirectory.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from _common import (
    base_parser,
    pad_to_minimum,
    prepare_directory,
    read_rgb,
    require_directory,
    tile_pair,
    write_he_sample,
)


def polygon_mask(csv_path: Path, shape: tuple[int, int]) -> np.ndarray:
    frame = pd.read_csv(csv_path)
    required = {"type", "coords_x", "coords_y"}
    if not required.issubset(frame.columns):
        raise ValueError(f"PanopTILs CSV {csv_path} lacks columns {sorted(required)}")
    mask = np.zeros(shape, dtype=np.int32)
    polylines = frame[frame["type"].astype(str).str.lower() == "polyline"]
    for instance_id, (_, row) in enumerate(polylines.iterrows(), start=1):
        x_values = [int(float(value)) for value in str(row["coords_x"]).split(",")]
        y_values = [int(float(value)) for value in str(row["coords_y"]).split(",")]
        if len(x_values) != len(y_values) or len(x_values) < 3:
            continue
        points = np.asarray(list(zip(x_values, y_values)), dtype=np.int32)
        cv2.fillPoly(mask, [points], instance_id)
    return mask


def main() -> None:
    parser = base_parser("Convert PanopTILs polygons to VitaminP Zarr data.")
    args = parser.parse_args()
    root = require_directory(args.input_dir, "PanopTILs dataset root")
    if (root / "tcga").is_dir():
        root = root / "tcga"
    image_dir = require_directory(root / "rgbs", "PanopTILs RGB images")
    csv_dir = require_directory(root / "csv", "PanopTILs CSV annotations")
    split_dir = prepare_directory(args.output_dir.expanduser().resolve() / "panoptils_full")

    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No PanopTILs PNG files found in {image_dir}")
    for image_path in tqdm(image_paths, desc="panoptils_full"):
        csv_path = csv_dir / f"{image_path.stem}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing PanopTILs CSV: {csv_path}")
        image = read_rgb(image_path)
        mask = polygon_mask(csv_path, image.shape[:2])
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

