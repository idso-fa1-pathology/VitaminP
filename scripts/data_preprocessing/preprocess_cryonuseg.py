#!/usr/bin/env python3
"""Convert CryoNuSeg 512x512 image/mask pairs to Zarr splits.

Expected input: ``tissue images/*.tif`` and
``Annotator 1 (biologist)/label masks/*.tif`` with matching stems.  The organ
name is read from ``Human_<Organ>_<Index>.tif``.

Split: pooled images are split 70/15/15 at image level with the notebook's
two-stage sklearn split (default seed 42).  Images are already 512x512 and are
not tiled.  Outputs are flat ``cryonuseg_{train,val,test}`` archives.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from _common import (
    base_parser,
    count_instances,
    read_instance_tiff,
    read_rgb,
    require_directory,
    split_70_15_15,
    stack_or_raise,
    write_image_mask_zarr,
)


Pair = tuple[Path, Path, str]


def collect_pairs(root: Path) -> list[Pair]:
    image_dir = require_directory(root / "tissue images", "CryoNuSeg tissue images")
    mask_dir = require_directory(
        root / "Annotator 1 (biologist)" / "label masks",
        "CryoNuSeg annotator-1 label masks",
    )
    pairs: list[Pair] = []
    for image_path in sorted(image_dir.glob("*.tif")):
        mask_path = mask_dir / image_path.name
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing CryoNuSeg mask: {mask_path}")
        parts = image_path.stem.split("_")
        organ = parts[1] if len(parts) >= 2 else "Unknown"
        pairs.append((image_path, mask_path, organ))
    return pairs


def convert_split(pairs: list[Pair], output: Path, name: str) -> None:
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    for image_path, mask_path, organ in tqdm(pairs, desc=name):
        image = read_rgb(image_path)
        mask = read_instance_tiff(mask_path)
        if image.shape != (512, 512, 3) or mask.shape != (512, 512):
            raise ValueError(f"CryoNuSeg expects 512x512 pairs: {image_path} / {mask_path}")
        images.append(image)
        masks.append(mask)
        metadata.append(
            {
                "organ_type": organ,
                "original_file": image_path.name,
                "original_height": 512,
                "original_width": 512,
                "num_nuclei": count_instances(mask),
            }
        )
    write_image_mask_zarr(
        output / name,
        stack_or_raise(images, f"images for {name}").astype(np.uint8),
        stack_or_raise(masks, f"masks for {name}").astype(np.int32),
        metadata,
    )


def main() -> None:
    parser = base_parser("Convert CryoNuSeg to VitaminP Zarr data.", default_seed=42)
    args = parser.parse_args()
    root = require_directory(args.input_dir, "CryoNuSeg dataset root")
    output = args.output_dir.expanduser().resolve()
    train, val, test = split_70_15_15(collect_pairs(root), args.seed)
    for name, pairs in (
        ("cryonuseg_train", train),
        ("cryonuseg_val", val),
        ("cryonuseg_test", test),
    ):
        convert_split(pairs, output, name)


if __name__ == "__main__":
    main()

