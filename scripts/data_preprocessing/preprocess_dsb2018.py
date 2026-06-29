#!/usr/bin/env python3
"""Decode DSB2018 Kaggle RLE masks and write modality-specific Zarr data.

Expected input: ``stage1-test/<ImageId>/images/<ImageId>.png`` and
``stage1-test-mask/stage1_solution.csv`` with ``ImageId``/``EncodedPixels``.
Each RLE row becomes one instance.  Images are copied into the top-left of a
512x512 canvas, matching the notebook (larger inputs are top-left cropped).

The notebook defines no data split, but dataset.py distinguishes brightfield
and fluorescence containers.  ``--modality auto`` deterministically assigns
dark-background images to ``fluorescence_dsb2018`` and the rest to
``he_dsb2018``; use a modality-map CSV for audited assignments.  Each
container holds one per-image Zarr subdirectory and is not randomly split.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from _common import base_parser, prepare_directory, read_rgb, require_directory, require_file, write_he_sample


def rle_decode(value: str, shape: tuple[int, int]) -> np.ndarray:
    fields = value.split()
    starts = np.asarray(fields[::2], dtype=np.int64) - 1
    lengths = np.asarray(fields[1::2], dtype=np.int64)
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, length in zip(starts, lengths):
        flat[start : start + length] = 1
    return flat.reshape(shape, order="F")


def load_modality_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    frame = pd.read_csv(require_file(path, "DSB2018 modality map"))
    required = {"ImageId", "modality"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Modality map must contain columns {sorted(required)}")
    values = dict(zip(frame["ImageId"].astype(str), frame["modality"].astype(str).str.lower()))
    invalid = {value for value in values.values() if value not in {"he", "fluorescence"}}
    if invalid:
        raise ValueError(f"Invalid modalities in map: {sorted(invalid)}")
    return values


def classify(image: np.ndarray, image_id: str, args, modality_map: dict[str, str]) -> str:
    if image_id in modality_map:
        return modality_map[image_id]
    if args.modality != "auto":
        return args.modality
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return "fluorescence" if float(np.median(gray)) < args.dark_background_threshold else "he"


def main() -> None:
    parser = base_parser("Convert DSB2018 RLE annotations to VitaminP Zarr data.")
    parser.add_argument(
        "--modality",
        choices=("auto", "he", "fluorescence"),
        default="auto",
        help="Assign every image explicitly or infer dark-background fluorescence images.",
    )
    parser.add_argument(
        "--modality-map",
        type=Path,
        help="Optional CSV with ImageId and modality (he/fluorescence), overriding inference.",
    )
    parser.add_argument(
        "--dark-background-threshold",
        type=float,
        default=127.0,
        help="Median grayscale threshold used only by --modality auto.",
    )
    args = parser.parse_args()
    root = require_directory(args.input_dir, "DSB2018 dataset root")
    image_root = require_directory(root / "stage1-test", "DSB2018 stage1-test")
    csv_path = require_file(root / "stage1-test-mask" / "stage1_solution.csv", "DSB2018 solution CSV")
    output = args.output_dir.expanduser().resolve()
    modality_map = load_modality_map(args.modality_map)
    frame = pd.read_csv(csv_path)
    if not {"ImageId", "EncodedPixels"}.issubset(frame.columns):
        raise ValueError("DSB2018 CSV must contain ImageId and EncodedPixels")

    split_dirs = {
        "he": prepare_directory(output / "he_dsb2018"),
        "fluorescence": prepare_directory(output / "fluorescence_dsb2018"),
    }
    for image_id, rows in tqdm(frame.groupby("ImageId", sort=True), desc="dsb2018"):
        image_path = image_root / str(image_id) / "images" / f"{image_id}.png"
        image = read_rgb(image_path)
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.int32)
        for instance_id, encoded in enumerate(rows["EncodedPixels"].dropna(), start=1):
            mask[rle_decode(str(encoded), (height, width)) > 0] = instance_id

        final_image = np.zeros((512, 512, 3), dtype=np.uint8)
        final_mask = np.zeros((512, 512), dtype=np.int32)
        copy_h, copy_w = min(height, 512), min(width, 512)
        final_image[:copy_h, :copy_w] = image[:copy_h, :copy_w]
        final_mask[:copy_h, :copy_w] = mask[:copy_h, :copy_w]
        modality = classify(image, str(image_id), args, modality_map)
        write_he_sample(
            split_dirs[modality] / str(image_id),
            [final_image],
            [final_mask],
            [
                {
                    "original_file": image_id,
                    "x": 0,
                    "y": 0,
                    "orig_h": height,
                    "orig_w": width,
                    "modality": modality,
                }
            ],
        )


if __name__ == "__main__":
    main()

