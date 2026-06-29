#!/usr/bin/env python3
"""Convert Lizard RGB slides and MAT instance maps to merged split Zarrs.

Expected input is the public release layout with images below
``lizard_images1/Lizard_Images1`` and ``lizard_images2/Lizard_Images2``, and
MAT labels below ``lizard_labels/Lizard_Labels/Labels``.  A recursive fallback
is used for equivalent extracted layouts.

Split: sorted slide/image pairs are shuffled with seed 42, then assigned 85%
train, 5% validation, and the remainder test before tiling.  Slides are padded
to at least 512 and edge-aligned 512x512 patches are merged into flat
``lizard_train``, ``lizard_val``, and ``lizard_test`` archives.
"""

from __future__ import annotations

import random
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


def collect_pairs(root: Path) -> list[tuple[Path, Path]]:
    preferred_image_dirs = [
        root / "lizard_images1" / "Lizard_Images1",
        root / "lizard_images2" / "Lizard_Images2",
    ]
    images = [path for directory in preferred_image_dirs if directory.is_dir() for path in directory.glob("*.png")]
    label_root = root / "lizard_labels" / "Lizard_Labels" / "Labels"
    labels = list(label_root.rglob("*.mat")) if label_root.is_dir() else list(root.rglob("*.mat"))
    if not images:
        images = [path for path in root.rglob("*.png") if "label" not in str(path).lower()]
    label_by_stem = {path.stem: path for path in labels}
    missing = [path for path in images if path.stem not in label_by_stem]
    if missing:
        raise FileNotFoundError(f"Missing Lizard MAT labels for {len(missing)} images; first: {missing[0]}")
    return sorted(((path, label_by_stem[path.stem]) for path in images), key=lambda pair: pair[0].stem)


def convert_split(pairs: list[tuple[Path, Path]], output: Path, name: str) -> None:
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    for image_path, label_path in tqdm(pairs, desc=name):
        image = read_rgb(image_path)
        mask = read_mat_instance(label_path, keys=("inst_map",))
        original_h, original_w = image.shape[:2]
        image = pad_to_minimum(image)
        mask = pad_to_minimum(mask)
        for image_tile, mask_tile, y, x in tile_pair(image, mask):
            images.append(image_tile)
            masks.append(mask_tile)
            metadata.append(
                {
                    "slide": image_path.stem,
                    "original_file": image_path.name,
                    "x": x,
                    "y": y,
                    "original_height": original_h,
                    "original_width": original_w,
                    "split": name,
                }
            )
    write_image_mask_zarr(
        output / name,
        stack_or_raise(images, f"tiles for {name}").astype(np.uint8),
        stack_or_raise(masks, f"masks for {name}").astype(np.int32),
        metadata,
    )


def main() -> None:
    parser = base_parser("Convert Lizard to slide-disjoint VitaminP Zarr splits.", default_seed=42)
    args = parser.parse_args()
    root = require_directory(args.input_dir, "Lizard dataset root")
    output = args.output_dir.expanduser().resolve()
    pairs = collect_pairs(root)
    random.Random(args.seed).shuffle(pairs)
    n_train = int(len(pairs) * 0.85)
    n_val = int(len(pairs) * 0.05)
    assignments = {
        "lizard_train": pairs[:n_train],
        "lizard_val": pairs[n_train : n_train + n_val],
        "lizard_test": pairs[n_train + n_val :],
    }
    for name, split_pairs in assignments.items():
        convert_split(split_pairs, output, name)


if __name__ == "__main__":
    main()

