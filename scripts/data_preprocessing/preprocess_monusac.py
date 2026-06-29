#!/usr/bin/env python3
"""Convert MoNuSAC TIFF/XML images to per-image Zarr stacks.

Expected input: ``train`` (or ``Training Data``) and ``test`` (or ``Test``),
with TIFF/XML pairs either directly inside or one patient directory below.
All annotation regions are merged into a single nuclei-instance target, as in
the source notebook.

Split: source test remains test; training images are shuffled and split 90/10
before edge-aligned 512x512 tiling (default seed 42).  Each ``monusac_*`` split
contains per-image Zarr subdirectories for compatibility with dataset.py.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from _common import (
    base_parser,
    count_instances,
    pad_to_minimum,
    prepare_directory,
    read_rgb,
    require_directory,
    shuffled_train_val,
    tile_pair,
    write_he_sample,
)


def xml_to_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing MoNuSAC XML annotation: {path}")
    mask = np.zeros(shape, dtype=np.int32)
    root = ET.parse(path).getroot()
    instance_id = 1
    for region in root.findall(".//Region"):
        vertices = [
            [int(float(vertex.get("X", "0"))), int(float(vertex.get("Y", "0")))]
            for vertex in region.findall(".//Vertex")
        ]
        if len(vertices) > 2:
            cv2.fillPoly(mask, [np.asarray(vertices, dtype=np.int32)], instance_id)
            instance_id += 1
    return mask


def find_split(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"None of the expected directories exist below {root}: {names}")


def collect_images(folder: Path) -> list[Path]:
    return sorted({*folder.glob("*.tif"), *folder.glob("*/*.tif")})


def convert_split(files: list[Path], output: Path, name: str) -> None:
    split_dir = prepare_directory(output / name)
    for image_path in tqdm(files, desc=name):
        image = read_rgb(image_path)
        original_h, original_w = image.shape[:2]
        mask = xml_to_mask(image_path.with_suffix(".xml"), (original_h, original_w))
        image = pad_to_minimum(image)
        mask = pad_to_minimum(mask)
        tiles = tile_pair(image, mask)
        patient = image_path.parent.name
        sample_name = f"{patient}__{image_path.stem}"
        write_he_sample(
            split_dir / sample_name,
            [tile[0] for tile in tiles],
            [tile[1] for tile in tiles],
            [
                {
                    "original_file": image_path.name,
                    "patient_id": patient,
                    "split": name,
                    "tile_y": y,
                    "tile_x": x,
                    "y_start": y,
                    "x_start": x,
                    "num_nuclei": count_instances(mask_tile),
                }
                for _, mask_tile, y, x in tiles
            ],
        )


def main() -> None:
    parser = base_parser("Convert MoNuSAC to VitaminP Zarr data.", default_seed=42)
    args = parser.parse_args()
    root = require_directory(args.input_dir, "MoNuSAC dataset root")
    output = args.output_dir.expanduser().resolve()
    train_root = find_split(root, ("train", "Training Data"))
    test_root = find_split(root, ("test", "Test"))
    train, val = shuffled_train_val(collect_images(train_root), 0.10, args.seed)
    convert_split(train, output, "monusac_train")
    convert_split(val, output, "monusac_val")
    convert_split(collect_images(test_root), output, "monusac_test")


if __name__ == "__main__":
    main()

