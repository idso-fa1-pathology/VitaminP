"""CLI and validation utilities only; ROI selection lives in each pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

LOG = logging.getLogger("roi_to_zarr")
FIELDS = ("patch_idx", "sample", "roi_idx", "y_start", "y_end", "x_start", "x_end")
COORD_FIELDS = ("y_start", "y_end", "x_start", "x_end")


def parser_for(dataset: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input-dir", required=True, type=Path, help="Root of original sample directories.")
    parser.add_argument("--roi-dir", type=Path,
                        default=Path(__file__).parent / "pathologist_geojson" / dataset,
                        help="Contains <sample>/pathologist_annotations.geojson; no segmentation files.")
    parser.add_argument("--samples", nargs="+", help="Optional exact sample directory names.")
    parser.add_argument("--output-dir", type=Path, help="New Zarr root (must not exist).")
    parser.add_argument("--report-dir", required=True, type=Path, help="New coordinate/report directory (must not exist).")
    parser.add_argument("--reference-dir", type=Path,
                        help="Released Zarr root containing <sample>/{he,mif}/metadata.csv.")
    parser.add_argument("--expected-counts", type=Path,
                        help="Original count CSV with sample,num_patches columns (ORION summary also accepted).")
    parser.add_argument("--dry-run", action="store_true", help="Write coordinates/reports only, never Zarr.")
    return parser


def discover_samples(args, *, orion: bool = False) -> list[str]:
    args.input_dir = args.input_dir.expanduser().resolve()
    args.roi_dir = args.roi_dir.expanduser().resolve()
    args.report_dir = args.report_dir.expanduser().absolute()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().absolute()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not args.roi_dir.is_dir():
        raise FileNotFoundError(f"Pathologist ROI directory not found: {args.roi_dir}")
    for path in (args.report_dir, args.output_dir if not args.dry_run else None):
        if path is not None and (path.exists() or path.is_symlink()):
            raise FileExistsError(f"Refusing to overwrite {path}; choose a new destination.")
    if args.output_dir is not None:
        output, report = args.output_dir.resolve(), args.report_dir.resolve()
        if output == report or output in report.parents or report in output.parents:
            raise ValueError("Output and report directories must be separate, non-nested destinations.")
    # Preserve original discovery, excluding explicitly configured auxiliary roots.
    excluded = {p.resolve() for p in (args.output_dir, args.report_dir, args.roi_dir, args.reference_dir)
                if p is not None}
    if args.samples:
        if len(set(args.samples)) != len(args.samples):
            raise ValueError("--samples contains duplicate names")
        samples = sorted(args.samples)
    else:
        samples = sorted(p.name for p in args.input_dir.iterdir()
                         if p.is_dir() and p.resolve() not in excluded
                         and (p.name.startswith("CRC") if orion else p.name != "zarr_data"))
    if not samples:
        raise ValueError("No sample directories found; source data are required for reproduction.")
    for sample in samples:
        if Path(sample).name != sample or sample in (".", ".."):
            raise ValueError(f"Expected a sample directory name, got {sample!r}")
        if orion and not sample.startswith("CRC"):
            raise ValueError(f"ORION source scans only CRC-prefixed samples: {sample}")
        if not (args.input_dir / sample).is_dir():
            raise FileNotFoundError(f"Sample directory not found: {sample}")
    return samples


def load_features(roi_dir: Path, sample: str) -> list:
    path = roi_dir / sample / "pathologist_annotations.geojson"
    with path.open() as handle:
        data = json.load(handle)
    features = data.get("features", [])
    if not isinstance(features, list):
        raise ValueError(f"Expected a GeoJSON features list: {path}")
    return features


def metadata_rows(sample: str, roi_patches) -> list[dict]:
    rows = []
    for roi_idx, patches in roi_patches:
        for coords in patches:
            rows.append(dict(zip(FIELDS, (len(rows), sample, roi_idx, *coords))))
    return rows


def write_metadata(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def compare_metadata(rows: list[dict], root: Path | None, sample: str) -> dict:
    """Compare order and multiplicity; never reduce coordinates to a set."""
    if root is None:
        return {"status": "unverified", "reason": "No released metadata supplied."}
    results = {}
    for modality in ("he", "mif"):
        path = root.expanduser() / sample / modality / "metadata.csv"
        if not path.is_file():
            results[modality] = {"status": "unverified", "reason": f"Missing {path}"}
            continue
        try:
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                columns = reader.fieldnames or []
                missing = set(COORD_FIELDS) - set(columns)
                if missing:
                    raise ValueError(f"Missing coordinate columns: {sorted(missing)}")
                reference = list(reader)
            compared = [f for f in FIELDS if f in columns]
            mismatch = None
            for index, (actual, expected) in enumerate(zip(rows, reference)):
                for field in compared:
                    value = expected[field] if field == "sample" else int(expected[field])
                    if actual[field] != value:
                        mismatch = {"row": index, "field": field,
                                    "generated": actual[field], "reference": value}
                        break
                if mismatch:
                    break
            matches = mismatch is None and len(rows) == len(reference)
            results[modality] = {"status": "match" if matches else "mismatch",
                                 "generated_count": len(rows), "reference_count": len(reference),
                                 "compared_columns": compared, "first_difference": mismatch}
        except (OSError, ValueError, TypeError, csv.Error) as exc:
            results[modality] = {"status": "error", "reason": str(exc)}
    statuses = [r["status"] for r in results.values()]
    status = "match" if all(s == "match" for s in statuses) else (
        "mismatch" if any(s in ("mismatch", "error") for s in statuses) else "unverified")
    return {"status": status, "modalities": results}


def load_expected_counts(path: Path | None) -> dict[str, int] | None:
    if path is None:
        return None
    with path.expanduser().open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"sample", "num_patches"}.issubset(reader.fieldnames or []):
            raise ValueError("Expected-count CSV needs sample,num_patches columns.")
        counts = {}
        for row in reader:
            sample, count = row["sample"], int(row["num_patches"])
            if sample in counts or count < 0:
                raise ValueError(f"Duplicate sample or negative count in count CSV: {sample}")
            counts[sample] = count
    if not counts:
        raise ValueError("Expected-count CSV is empty.")
    return counts


def finalize_report(args, dataset: str, samples: list[dict], expected: dict | None) -> tuple[dict, bool]:
    total = sum(s.get("num_patches", 0) for s in samples)
    failed = any(s["status"] != "planned" for s in samples)
    validation_failed = False
    for sample in samples:
        comparison = sample.get("metadata_comparison", {})
        if args.reference_dir is not None and comparison.get("status") != "match":
            validation_failed = True
        name = sample["sample"]
        if expected is not None:
            count = expected.get(name)
            sample["count_comparison"] = {"expected": count,
                "status": "unverified" if count is None else (
                    "match" if sample.get("num_patches") == count and sample["status"] == "planned" else "mismatch")}
            validation_failed |= sample["count_comparison"]["status"] != "match"
    missing_samples = sorted(set(expected or {}) - {s["sample"] for s in samples})
    # A count manifest covers the full cohort unless a subset was explicitly selected.
    if missing_samples and not args.samples:
        validation_failed = True
    report = {"dataset": dataset, "dry_run": args.dry_run,
              "patch_size": 512, "stride": 512,
              "input_dir": str(args.input_dir), "roi_dir": str(args.roi_dir),
              "num_samples": len(samples), "total_patches": total, "samples": samples,
              "count_manifest": "provided" if expected is not None else "unverified: not supplied",
              "unprocessed_manifest_samples": missing_samples,
              "released_metadata": "provided" if args.reference_dir else "unverified: not supplied",
              "zarr_written": False}
    report["software_versions"] = {"python": platform.python_version()}
    for package in ("numpy", "shapely", "tifffile", "zarr", "numcodecs"):
        try:
            report["software_versions"][package] = version(package)
        except PackageNotFoundError:
            report["software_versions"][package] = "not installed"
    if dataset == "xenium":
        report["historical_target"] = {
            "expected_total": 1909, "expected_samples": 9,
            "scope": "selected subset" if args.samples else "discovered cohort",
            "status": "unverified" if failed or args.samples else (
                "count_match_only" if total == 1909 and len(samples) == 9 else "mismatch"),
            "note": "Aggregate counts do not establish coordinate or individual-sample reproduction."}
        if not args.samples and report["historical_target"]["status"] == "mismatch":
            validation_failed = True
    report["status"] = "failed" if failed or validation_failed else "planned"
    report["validation_failed"] = validation_failed
    return report, failed or validation_failed


def save_report(args, report: dict) -> None:
    # Rewriting our own report after conversion is safe; report-dir was exclusively created.
    (args.report_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    LOG.info("%s: %d candidate tiles across %d samples; report: %s",
             report["dataset"], report["total_patches"], report["num_samples"], args.report_dir)
    LOG.info("Reference metadata: %s; count manifest: %s",
             report["released_metadata"], report["count_manifest"])


def cli_main(main) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        return main()
    except Exception as exc:
        LOG.error("%s: %s", type(exc).__name__, exc)
        return 1
