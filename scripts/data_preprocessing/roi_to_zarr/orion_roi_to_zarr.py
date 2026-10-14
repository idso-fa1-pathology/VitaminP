#!/usr/bin/env python3
"""Orion ROI pipeline; see README.md for source limitations and exact rules."""

from __future__ import annotations

import csv
from pathlib import Path

from shapely.geometry import shape, box

from _reporting import (
    LOG, cli_main, compare_metadata, discover_samples, finalize_report,
    load_expected_counts, load_features, metadata_rows, parser_for, save_report,
    write_metadata,
)

PATCH_SIZE = 512

# Transcribed verbatim from the supplied Orion cell. Do not unify the two algorithms.
def generate_patches_from_roi(roi_geometry, patch_size=512, overlap=0):
    """
    Generate 512x512 patch coordinates from an ROI.
    """
    minx, miny, maxx, maxy = roi_geometry.bounds

    roi_width = maxx - minx
    roi_height = maxy - miny

    stride = patch_size - overlap

    # If ROI is smaller than patch_size, create one patch centered on ROI
    if roi_width < patch_size and roi_height < patch_size:
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2

        x_start = int(center_x - patch_size / 2)
        y_start = int(center_y - patch_size / 2)
        x_end = x_start + patch_size
        y_end = y_start + patch_size

        x_start = max(0, x_start)
        y_start = max(0, y_start)

        return [(y_start, y_end, x_start, x_end)]

    # Otherwise, tile the ROI with patches
    patches = []

    x_start_pos = int(minx)
    y_start_pos = int(miny)

    y = y_start_pos
    while y < maxy:
        x = x_start_pos
        while x < maxx:
            x_end = min(x + patch_size, int(maxx) + patch_size)
            y_end = min(y + patch_size, int(maxy) + patch_size)

            if x_end - x < patch_size:
                x = x_end - patch_size
            if y_end - y < patch_size:
                y = y_end - patch_size

            x = max(0, x)
            y = max(0, y)
            x_end = x + patch_size
            y_end = y + patch_size

            from shapely.geometry import box
            patch_box = box(x, y, x_end, y_end)

            if patch_box.intersects(roi_geometry):
                intersection_area = patch_box.intersection(roi_geometry).area
                patch_area = patch_box.area
                overlap_ratio = intersection_area / patch_area

                if overlap_ratio > 0.1:
                    patches.append((y, y_end, x, x_end))

            x += stride
            if x >= maxx:
                break

        y += stride
        if y >= maxy:
            break

    return patches


def first_image(sample_dir: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        candidate = sample_dir / "images" / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing image in {sample_dir / 'images'}: {' or '.join(names)}")

def tiff_shape(path: Path) -> tuple[int, ...]:
    from tifffile import TiffFile

    with TiffFile(path) as image:
        return image.series[0].shape

def inspect_inputs(sample_dir: Path, he_path: Path, he_shape: tuple) -> tuple[dict, list[str]]:
    """Check storage shapes without loading entire images; do not filter tiles."""
    import numpy as np

    paths = {"he_image": he_path}
    issues = []
    if len(he_shape) != 3 or he_shape[2] != 3:
        issues.append(f"H&E must have the notebook's (H,W,3) shape, got {he_shape}")
    try:
        paths["mif_image"] = first_image(sample_dir, ("mif_cells.tif", "mif_cells.tiff"))
        mif_shape = tiff_shape(paths["mif_image"])
        if len(mif_shape) == 3 and mif_shape[0] == 2:
            mif_shape = (mif_shape[1], mif_shape[2], mif_shape[0])
        if mif_shape != (*he_shape[:2], 2):
            issues.append(f"MIF shape after original transpose is {mif_shape}; expected {(*he_shape[:2], 2)}")
    except (OSError, ValueError) as exc:
        issues.append(str(exc))
    for key, filename in (("he_nuclei", "he_nuclei.npy"),
                          ("mif_nuclei", "mif_nuclei.npy"), ("cells", "mif_cells.npy")):
        path = sample_dir / "masks" / filename
        paths[key] = path
        try:
            mask = np.load(path, mmap_mode="r", allow_pickle=False)
            if mask.shape != he_shape[:2]:
                issues.append(f"{filename} shape {mask.shape} does not match H&E {he_shape[:2]}")
        except (OSError, ValueError) as exc:
            issues.append(str(exc))
    return paths, issues

def main(argv=None) -> int:
    parser = parser_for("orion", "Reproduce the supplied ORION ROI coordinate scan (dry-run only).")
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error("The supplied ORION notebook cell only scans/counts ROIs. Use --dry-run; "
                     "Zarr writing requires the original image/mask extraction and writing cell.")
    samples = discover_samples(args, orion=True)
    expected = load_expected_counts(args.expected_counts)
    args.report_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for sample in samples:
        try:
            features = load_features(args.roi_dir, sample)
            rows = metadata_rows(sample, (
                (index, generate_patches_from_roi(shape(feature["geometry"]),
                                                 patch_size=PATCH_SIZE, overlap=0))
                for index, feature in enumerate(features)))
            write_metadata(args.report_dir / f"{sample}.csv", rows)
            result = {"sample": sample, "status": "planned", "num_rois": len(features),
                      "num_patches": len(rows),
                      "metadata_comparison": compare_metadata(rows, args.reference_dir, sample)}
            LOG.info("%s: %d ROIs -> %d candidate tiles", sample, len(features), len(rows))
        except Exception as exc:
            result = {"sample": sample, "status": "error", "error": str(exc)}
            LOG.error("%s: %s", sample, exc)
        results.append(result)
    report, failed = finalize_report(args, "orion", results, expected)
    report["extraction_verified"] = False
    with (args.report_dir / "orion_crc_patch_summary.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample", "has_annotations", "num_rois", "num_patches"))
        writer.writeheader()
        for result in results:
            writer.writerow({"sample": result["sample"], "has_annotations": result["status"] == "planned",
                             "num_rois": result.get("num_rois", 0), "num_patches": result.get("num_patches", 0)})
    save_report(args, report)
    LOG.warning("ORION counts are ROI candidates, not verified extractable tiles. "
                "The supplied source has no image-bound checks or Zarr conversion.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(cli_main(main))
