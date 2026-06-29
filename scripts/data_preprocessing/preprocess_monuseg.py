#!/usr/bin/env python3
"""Convert MoNuSeg TIFF/XML slides to per-slide Zarr stacks.

Expected input: source training data under ``train/Tissue Images`` and
``train/Annotations`` (a flat ``train`` fallback is accepted), and a flat
``test`` directory containing same-stem TIFF/XML pairs.  Aperio ``Region``
polygons are rasterized with one integer ID per nucleus.

Split: source test remains test; training slides are shuffled and split 90/10
at slide level before bottom/right padding and non-overlapping 512x512 tiling
(default seed 42).  Each ``monuseg_*`` split contains per-slide Zarr folders,
as expected by dataset.py.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from _common import (
    base_parser,
    pad_to_multiple,
    prepare_directory,
    read_rgb,
    require_directory,
    shuffled_train_val,
    tile_pair,
    write_he_sample,
)


Pair = tuple[Path, Path]


def xml_to_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.int32)
    root = ET.parse(path).getroot()
    instance_id = 1
    for region in root.iter("Region"):
        vertices = [
            [int(float(vertex.get("X", "0"))), int(float(vertex.get("Y", "0")))]
            for vertex in region.iter("Vertex")
        ]
        if len(vertices) > 2:
            cv2.fillPoly(mask, [np.asarray(vertices, dtype=np.int32)], instance_id)
            instance_id += 1
    return mask


def collect_pairs(folder: Path, structured: bool) -> list[Pair]:
    image_dir = folder / "Tissue Images" if structured else folder
    xml_dir = folder / "Annotations" if structured else folder
    if not image_dir.is_dir() or not xml_dir.is_dir():
        return []
    images = sorted([*image_dir.glob("*.tif"), *image_dir.glob("*.png")])
    pairs = [(image, xml_dir / f"{image.stem}.xml") for image in images]
    missing = [xml for _, xml in pairs if not xml.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing MoNuSeg XML annotation: {missing[0]}")
    return pairs


def convert_split(pairs: list[Pair], output: Path, name: str) -> None:
    split_dir = prepare_directory(output / name)
    for image_path, xml_path in tqdm(pairs, desc=name):
        image = read_rgb(image_path)
        original_h, original_w = image.shape[:2]
        mask = xml_to_mask(xml_path, (original_h, original_w))
        image = pad_to_multiple(image)
        mask = pad_to_multiple(mask)
        tiles = tile_pair(image, mask)
        write_he_sample(
            split_dir / image_path.stem,
            [tile[0] for tile in tiles],
            [tile[1] for tile in tiles],
            [
                {
                    "original_file": image_path.name,
                    "x": x,
                    "y": y,
                    "original_height": original_h,
                    "original_width": original_w,
                }
                for _, _, y, x in tiles
            ],
        )


def main() -> None:
    parser = base_parser("Convert MoNuSeg to VitaminP Zarr data.", default_seed=42)
    args = parser.parse_args()
    root = require_directory(args.input_dir, "MoNuSeg dataset root")
    output = args.output_dir.expanduser().resolve()
    train_root = require_directory(root / "train", "MoNuSeg train directory")
    test_root = require_directory(root / "test", "MoNuSeg test directory")
    train_pairs = collect_pairs(train_root, structured=True) or collect_pairs(train_root, structured=False)
    train, val = shuffled_train_val(train_pairs, 0.10, args.seed)
    convert_split(train, output, "monuseg_train")
    convert_split(val, output, "monuseg_val")
    convert_split(collect_pairs(test_root, structured=False), output, "monuseg_test")


if __name__ == "__main__":
    main()

