#!/usr/bin/env python3
"""Xenium ROI pipeline; see README.md for source limitations and exact rules."""

from __future__ import annotations

from pathlib import Path

from shapely.geometry import shape, box

from _reporting import (
    LOG, cli_main, compare_metadata, discover_samples, finalize_report,
    load_expected_counts, load_features, metadata_rows, parser_for, save_report,
    write_metadata,
)

PATCH_SIZE = 512

# Transcribed verbatim from the supplied Xenium cell. Do not unify the two algorithms.
def generate_patches_from_roi(roi_geometry, patch_size=512, overlap=0, img_height=None, img_width=None):
    """Generate 512x512 patch coordinates from an ROI. Only include patches that fit exactly."""
    minx, miny, maxx, maxy = roi_geometry.bounds

    roi_width = maxx - minx
    roi_height = maxy - miny

    stride = patch_size - overlap

    if roi_width < patch_size and roi_height < patch_size:
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2

        x_start = int(center_x - patch_size / 2)
        y_start = int(center_y - patch_size / 2)
        x_end = x_start + patch_size
        y_end = y_start + patch_size

        x_start = max(0, x_start)
        y_start = max(0, y_start)

        # Check if patch fits within image bounds
        if img_height is not None and img_width is not None:
            if x_end > img_width or y_end > img_height:
                return []

        return [(y_start, y_end, x_start, x_end)]

    patches = []
    x_start_pos = int(minx)
    y_start_pos = int(miny)

    y = y_start_pos
    while y < maxy:
        x = x_start_pos
        while x < maxx:
            x_end = x + patch_size
            y_end = y + patch_size

            # Skip if patch extends beyond image bounds
            if img_height is not None and img_width is not None:
                if x_end > img_width or y_end > img_height:
                    x += stride
                    if x >= maxx:
                        break
                    continue

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


def plan_sample(args, sample: str) -> tuple[dict, list[dict], dict]:
    features = load_features(args.roi_dir, sample)
    he_path = first_image(args.input_dir / sample, ("he.tiff", "he.tif"))
    image_shape = tiff_shape(he_path)
    if len(image_shape) < 2:
        raise ValueError(f"Cannot obtain H&E dimensions from {he_path}: {image_shape}")
    height, width = image_shape[:2]
    rows = metadata_rows(sample, (
        (index, generate_patches_from_roi(shape(feature["geometry"]),
                                         patch_size=PATCH_SIZE, overlap=0,
                                         img_height=height, img_width=width))
        for index, feature in enumerate(features)))
    paths, issues = inspect_inputs(args.input_dir / sample, he_path, image_shape)
    invalid_tiles = []
    for row in rows:
        # Use Python slice semantics, including the original negative-start behavior.
        actual_height = len(range(*slice(row["y_start"], row["y_end"]).indices(height)))
        actual_width = len(range(*slice(row["x_start"], row["x_end"]).indices(width)))
        if (actual_height, actual_width) != (PATCH_SIZE, PATCH_SIZE):
            invalid_tiles.append(row["patch_idx"])
    if invalid_tiles:
        issues.append(f"Original coordinates yield undersized/empty slices at patch indices {invalid_tiles}; "
                      "coordinates are retained, but Zarr writing is refused.")
    if not rows:
        issues.append("No patches generated; original converter skips this sample.")
    result = {"sample": sample, "status": "planned" if rows else "skipped",
              "num_rois": len(features), "num_patches": len(rows),
              "he_shape": list(image_shape), "extraction_issues": issues,
              "metadata_comparison": compare_metadata(rows, args.reference_dir, sample)}
    return result, rows, paths


def write_sample(destination: Path, rows: list[dict], paths: dict) -> None:
    """Write the source notebook's six arrays, preserving dtype, chunks and order."""
    import numpy as np
    import zarr
    from numcodecs import Blosc
    from tifffile import imread

    if int(zarr.__version__.split(".")[0]) != 2:
        raise RuntimeError("This workflow requires Zarr v2 (install zarr>=2.16,<3).")
    he_image = imread(paths["he_image"])
    mif_image = imread(paths["mif_image"])
    if mif_image.shape[0] == 2 and len(mif_image.shape) == 3:
        mif_image = np.transpose(mif_image, (1, 2, 0))
    he_nuclei = np.load(paths["he_nuclei"], allow_pickle=False)
    mif_nuclei = np.load(paths["mif_nuclei"], allow_pickle=False)
    cells = np.load(paths["cells"], allow_pickle=False)
    sources = {
        "he/images": (he_image, "uint8"),
        "he/nuclei_masks": (he_nuclei, he_nuclei.dtype),
        "he/cell_masks": (cells, cells.dtype),
        "mif/images": (mif_image, mif_image.dtype),
        "mif/nuclei_masks": (mif_nuclei, mif_nuclei.dtype),
        "mif/cell_masks": (cells, cells.dtype),
    }
    destination.mkdir(exist_ok=False)
    for modality in ("he", "mif"):
        (destination / modality).mkdir()
    arrays = {}
    for name, (source, dtype) in sources.items():
        tile_shape = (PATCH_SIZE, PATCH_SIZE, *source.shape[2:])
        arrays[name] = zarr.open_array(
            str(destination / f"{name}.zarr"), mode="w-",
            shape=(len(rows), *tile_shape), chunks=(1, *tile_shape), dtype=dtype,
            compressor=Blosc(cname="zstd", clevel=3))
    for row in rows:
        ys, ye, xs, xe = (row[key] for key in ("y_start", "y_end", "x_start", "x_end"))
        for name, (source, _) in sources.items():
            tile = source[ys:ye, xs:xe]
            if tile.shape != arrays[name].shape[1:]:
                raise ValueError(f"{name}, patch {row['patch_idx']}: unexpected slice shape {tile.shape}")
            arrays[name][row["patch_idx"]] = tile
    for modality in ("he", "mif"):
        write_metadata(destination / modality / "metadata.csv", rows)


def main(argv=None) -> int:
    parser = parser_for("xenium", "Reproduce the original Xenium paired ROI-to-Zarr conversion.")
    args = parser.parse_args(argv)
    if not args.dry_run and args.output_dir is None:
        parser.error("--output-dir is required unless --dry-run is used")
    samples = discover_samples(args)
    expected = load_expected_counts(args.expected_counts)
    args.report_dir.mkdir(parents=True, exist_ok=False)
    results, plans = [], {}
    for sample in samples:
        try:
            result, rows, paths = plan_sample(args, sample)
            write_metadata(args.report_dir / f"{sample}.csv", rows)
            plans[sample] = (rows, paths)
            LOG.info("%s: %d ROIs -> %d candidate tiles", sample, result["num_rois"], len(rows))
            for issue in result["extraction_issues"]:
                LOG.warning("%s: %s", sample, issue)
        except Exception as exc:
            result = {"sample": sample, "status": "error", "error": str(exc)}
            LOG.error("%s: %s", sample, exc)
        results.append(result)
    report, failed = finalize_report(args, "xenium", results, expected)
    extraction_failed = any(s.get("extraction_issues") for s in results)
    if extraction_failed:
        failed = True
        report["status"] = "failed"
    report["historical_target"]["source_arrays_checked"] = not extraction_failed and not any(
        s["status"] == "error" for s in results)
    save_report(args, report)
    LOG.info("Xenium historical target: %s", report["historical_target"]["status"])
    if failed:
        LOG.error("Source checks or requested validations failed. Coordinates retained in report; no Zarr written.")
        return 1
    if args.dry_run:
        return 0
    # Never reuse an existing root, even if empty. Do not add an overwrite switch.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    marker = args.output_dir / "INCOMPLETE"
    marker.write_text("Conversion in progress. Do not use until this marker is removed.\n")
    completed = []
    try:
        for sample in samples:
            rows, paths = plans[sample]
            write_sample(args.output_dir / sample, rows, paths)
            completed.append(sample)
            LOG.info("%s: wrote %d paired tiles", sample, len(rows))
    except Exception as exc:
        report.update(status="failed", write_error=str(exc), completed_samples=completed)
        save_report(args, report)
        raise
    marker.unlink()
    report.update(status="written", zarr_written=True, completed_samples=completed)
    save_report(args, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main(main))
