#!/usr/bin/env python3
"""Convert NuInsSeg tissue folders to image-disjoint Zarr splits.

Expected input: one directory per tissue, each containing ``tissue images``
(PNG) and ``label masks`` (same-stem TIF instance masks).  Tissue folders are
discovered rather than hard-coded.

Split: all image/mask pairs are pooled and split 70/15/15 at image level with
the notebook's two-stage sklearn procedure (default seed 19).  Splitting is
done before bottom/right zero-padding to 512 multiples and non-overlapping
512x512 tiling.  Outputs are flat ``nuinsseg_{train,val,test}`` archives.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from _common import (
    base_parser,
    count_instances,
    pad_to_multiple,
    read_instance_tiff,
    read_rgb,
    require_directory,
    split_70_15_15,
    stack_or_raise,
    tile_pair,
    write_image_mask_zarr,
)


Pair = tuple[Path, Path, str]


def collect_pairs(root: Path) -> list[Pair]:
    pairs: list[Pair] = []
    for image_dir in sorted(root.rglob("tissue images")):
        if not image_dir.is_dir():
            continue
        tissue_dir = image_dir.parent
        mask_dir = tissue_dir / "label masks"
        if not mask_dir.is_dir():
            continue
        tissue = str(tissue_dir.relative_to(root))
        for image_path in sorted(image_dir.glob("*.png")):
            mask_path = mask_dir / f"{image_path.stem}.tif"
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing NuInsSeg mask: {mask_path}")
            pairs.append((image_path, mask_path, tissue))
    if not pairs:
        raise RuntimeError("No NuInsSeg image/mask pairs found")
    return pairs


def convert_split(pairs: list[Pair], output: Path, name: str) -> None:
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    for image_path, mask_path, tissue in tqdm(pairs, desc=name):
        image = read_rgb(image_path)
        mask = read_instance_tiff(mask_path)
        original_h, original_w = image.shape[:2]
        image = pad_to_multiple(image)
        mask = pad_to_multiple(mask)
        for image_tile, mask_tile, y, x in tile_pair(image, mask):
            images.append(image_tile)
            masks.append(mask_tile)
            metadata.append(
                {
                    "tissue_type": tissue,
                    "original_file": image_path.name,
                    "x": x,
                    "y": y,
                    "original_height": original_h,
                    "original_width": original_w,
                    "num_nuclei": count_instances(mask_tile),
                }
            )
    write_image_mask_zarr(
        output / name,
        stack_or_raise(images, f"tiles for {name}").astype(np.uint8),
        stack_or_raise(masks, f"masks for {name}").astype(np.int32),
        metadata,
    )


def main() -> None:
    parser = base_parser("Convert NuInsSeg to VitaminP Zarr data.", default_seed=19)
    args = parser.parse_args()
    root = require_directory(args.input_dir, "NuInsSeg dataset root")
    output = args.output_dir.expanduser().resolve()
    train, val, test = split_70_15_15(collect_pairs(root), args.seed)
    for name, pairs in (
        ("nuinsseg_train", train),
        ("nuinsseg_val", val),
        ("nuinsseg_test", test),
    ):
        convert_split(pairs, output, name)


if __name__ == "__main__":
    main()

