#!/usr/bin/env python3
"""Reproduce the Xenium/ORION-CRC mask synchronization notebook safely.

Expected input is an already tiled Zarr root, one sample directory at a time:
``<sample>/he`` and ``<sample>/mif`` must exist and MIF must contain
``nuclei_masks.zarr``.  This notebook did not contain raw-image tiling logic;
it replaced each H&E nuclei mask with the registered MIF nuclei mask.

There is no random split: sample-level train/validation/test assignment stays
in the VitaminP configuration.  By default the complete input tree is copied
to ``--output-dir`` and masks are replaced there.  An explicit ``--in-place``
flag is required to reproduce the notebook's destructive in-place operation.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from tqdm import tqdm

from _common import base_parser, require_directory


def valid_samples(root: Path) -> list[Path]:
    samples = sorted(path for path in root.iterdir() if path.is_dir())
    if not samples:
        raise RuntimeError(f"No Xenium/CRC sample directories found in {root}")
    for sample in samples:
        if not (sample / "he").is_dir() or not (sample / "mif").is_dir():
            raise FileNotFoundError(f"Sample lacks he/ and mif/ directories: {sample}")
        if not (sample / "mif" / "nuclei_masks.zarr").exists():
            raise FileNotFoundError(f"Sample lacks MIF nuclei mask: {sample}")
    return samples


def main() -> None:
    parser = base_parser("Copy Xenium/CRC Zarr data and synchronize H&E nuclei masks from MIF.")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Allow --input-dir and --output-dir to be identical and modify that tree.",
    )
    args = parser.parse_args()
    source = require_directory(args.input_dir, "Xenium/CRC Zarr root")
    destination = args.output_dir.expanduser().resolve()
    same_path = source == destination
    if same_path and not args.in_place:
        parser.error("refusing an in-place replacement without --in-place")
    if not same_path:
        if destination.is_relative_to(source) or source.is_relative_to(destination):
            parser.error("input and output directories must not contain one another")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)

    for sample in tqdm(valid_samples(destination), desc="synchronizing masks"):
        source_mask = sample / "mif" / "nuclei_masks.zarr"
        destination_mask = sample / "he" / "nuclei_masks.zarr"
        if destination_mask.exists():
            shutil.rmtree(destination_mask)
        shutil.copytree(source_mask, destination_mask)


if __name__ == "__main__":
    main()
