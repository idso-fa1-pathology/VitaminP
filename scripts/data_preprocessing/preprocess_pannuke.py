#!/usr/bin/env python3
"""Convert PanNuke folds to 512x512 per-sample Zarr mosaics.

Expected input: ``Fold N/images/foldN/images.npy`` and
``Fold N/masks/foldN/masks.npy`` for folds 1, 2, and 3.  Each source item is
256x256; four consecutive items from the same fold are placed in a 2x2
mosaic.  The five foreground type channels are collapsed to one instance map
and IDs are offset between quadrants.

Split: mosaics from folds 1+2 are shuffled and split 90/10 train/validation
(default seed 42); fold 3 remains test.  Each ``pannuke_*`` split contains
one subdirectory per mosaic because that is the layout indexed by dataset.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from _common import base_parser, prepare_directory, require_directory, require_file, write_he_sample


Task = tuple[int, int]


def load_folds(root: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    folds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for fold in (1, 2, 3):
        fold_root = root / f"Fold {fold}"
        image_path = require_file(
            fold_root / "images" / f"fold{fold}" / "images.npy",
            f"PanNuke fold {fold} images",
        )
        mask_path = require_file(
            fold_root / "masks" / f"fold{fold}" / "masks.npy",
            f"PanNuke fold {fold} masks",
        )
        folds[fold] = (np.load(image_path, mmap_mode="r"), np.load(mask_path, mmap_mode="r"))
    return folds


def make_mosaic(images: np.ndarray, masks: np.ndarray, start: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    mosaic_image = np.zeros((512, 512, 3), dtype=np.uint8)
    mosaic_mask = np.zeros((512, 512), dtype=np.int32)
    quadrants = ((0, 0), (0, 256), (256, 0), (256, 256))
    instance_offset = 0
    source_indices: list[int] = []
    for quadrant, index in enumerate(range(start, min(start + 4, len(images)))):
        y, x = quadrants[quadrant]
        source_indices.append(index)
        mosaic_image[y : y + 256, x : x + 256] = images[index].astype(np.uint8)
        source_mask = np.max(masks[index, :, :, :5], axis=-1).astype(np.int32)
        shifted = np.zeros_like(source_mask)
        foreground = source_mask > 0
        shifted[foreground] = source_mask[foreground] + instance_offset
        mosaic_mask[y : y + 256, x : x + 256] = shifted
        if np.any(foreground):
            instance_offset += int(source_mask.max())
    return mosaic_image, mosaic_mask, source_indices


def convert_tasks(
    tasks: list[Task],
    folds: dict[int, tuple[np.ndarray, np.ndarray]],
    output: Path,
    split_name: str,
) -> None:
    split_dir = prepare_directory(output / split_name)
    for mosaic_index, (fold, start) in enumerate(tqdm(tasks, desc=split_name)):
        image, mask, source_indices = make_mosaic(*folds[fold], start)
        sample_name = f"pannuke_mosaic_{mosaic_index:05d}"
        write_he_sample(
            split_dir / sample_name,
            [image],
            [mask],
            [
                {
                    "patch_idx": 0,
                    "sample": sample_name,
                    "original_fold": f"Fold_{fold}",
                    "original_indices": str(source_indices),
                }
            ],
        )


def main() -> None:
    parser = base_parser("Convert PanNuke folds to VitaminP Zarr mosaics.", default_seed=42)
    args = parser.parse_args()
    root = require_directory(args.input_dir, "PanNuke dataset root")
    output = args.output_dir.expanduser().resolve()
    folds = load_folds(root)

    train_pool = [
        (fold, start)
        for fold in (1, 2)
        for start in range(0, len(folds[fold][0]), 4)
    ]
    np.random.RandomState(args.seed).shuffle(train_pool)
    n_val = int(len(train_pool) * 0.10)
    train_tasks = train_pool[: len(train_pool) - n_val]
    val_tasks = train_pool[len(train_pool) - n_val :]
    test_tasks = [(3, start) for start in range(0, len(folds[3][0]), 4)]

    convert_tasks(train_tasks, folds, output, "pannuke_train")
    convert_tasks(val_tasks, folds, output, "pannuke_val")
    convert_tasks(test_tasks, folds, output, "pannuke_test")


if __name__ == "__main__":
    main()

