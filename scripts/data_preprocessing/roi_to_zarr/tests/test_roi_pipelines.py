"""Boundary, correspondence, reference-validation and no-overwrite regression tests.

All image fixtures are temporary. ROI geometries stay in memory; no artificial
pathologist annotation files are created or installed in the annotation folders.
"""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from shapely.geometry import box, mapping, Polygon
from tifffile import imwrite
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import orion_roi_to_zarr as orion
import xenium_roi_to_zarr as xenium
from _reporting import compare_metadata, metadata_rows, write_metadata


class CoordinateTests(unittest.TestCase):
    def test_small_roi_does_not_require_ten_percent_coverage(self):
        roi = box(300, 300, 301, 301)
        expected = [(44, 556, 44, 556)]
        self.assertEqual(xenium.generate_patches_from_roi(roi, img_height=1024, img_width=1024), expected)
        self.assertEqual(orion.generate_patches_from_roi(roi), expected)

    def test_clipped_small_roi_preserves_original_undersized_coordinates(self):
        expected = [(0, 306, 0, 306)]
        self.assertEqual(xenium.generate_patches_from_roi(box(0, 0, 100, 100)), expected)
        self.assertEqual(orion.generate_patches_from_roi(box(0, 0, 100, 100)), expected)

    def test_ten_percent_is_strict_for_general_grid(self):
        for generate in (xenium.generate_patches_from_roi, orion.generate_patches_from_roi):
            self.assertEqual(generate(box(0, 0, 512, 51.2)), [])
            self.assertEqual(generate(box(0, 0, 512, 51.201)), [(0, 512, 0, 512)])

    def test_xenium_checks_image_bounds_orion_does_not(self):
        roi = box(0, 0, 1024, 1024)
        self.assertEqual(xenium.generate_patches_from_roi(roi, img_height=700, img_width=700),
                         [(0, 512, 0, 512)])
        self.assertEqual(orion.generate_patches_from_roi(roi),
                         [(0, 512, 0, 512), (0, 512, 512, 1024),
                          (512, 1024, 0, 512), (512, 1024, 512, 1024)])

    def test_small_roi_upper_bound_check_is_xenium_only(self):
        roi = box(900, 900, 1000, 1000)
        self.assertEqual(xenium.generate_patches_from_roi(roi, img_height=1024, img_width=1024), [])
        self.assertEqual(orion.generate_patches_from_roi(roi), [(694, 1206, 694, 1206)])

    def test_negative_general_grid_differs_between_pipelines(self):
        roi = box(-10.8, 0, 600, 512)
        self.assertEqual(xenium.generate_patches_from_roi(roi)[0], (0, 512, -10, 502))
        self.assertEqual(orion.generate_patches_from_roi(roi)[0], (0, 512, 0, 512))

    def test_polygon_hole_is_not_filled(self):
        roi = Polygon([(0, 0), (1536, 0), (1536, 1536), (0, 1536)],
                      holes=[[(512, 512), (1024, 512), (1024, 1024), (512, 1024)]])
        for generate in (xenium.generate_patches_from_roi, orion.generate_patches_from_roi):
            coords = generate(roi)
            self.assertEqual(len(coords), 8)
            self.assertNotIn((512, 1024, 512, 1024), coords)

    def test_duplicate_rois_retain_order_and_multiplicity(self):
        coords = [(0, 512, 0, 512)]
        rows = metadata_rows("sample", [(0, coords), (1, coords)])
        self.assertEqual([r["patch_idx"] for r in rows], [0, 1])
        self.assertEqual([r["roi_idx"] for r in rows], [0, 1])


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input_dir = self.root / "input"
        self.rois = self.root / "rois"
        self.rois.mkdir()
        self.sample = "sample"
        self.sample_dir = self.input_dir / self.sample
        (self.sample_dir / "images").mkdir(parents=True)
        (self.sample_dir / "masks").mkdir()
        self.he = np.zeros((1024, 1024, 3), dtype=np.uint8)
        self.he[:, 512:, :] = 73
        self.mif = np.zeros((2, 1024, 1024), dtype=np.uint16)
        self.mif[0] = 1000
        self.mif[1, :, 512:] = 2300
        self.he_nuclei = np.full((1024, 1024), 17, dtype=np.int32)
        self.mif_nuclei = np.full((1024, 1024), 91, dtype=np.uint32)
        self.cells = np.full((1024, 1024), 123456, dtype=np.uint32)
        imwrite(self.sample_dir / "images/he.tiff", self.he, photometric="rgb")
        imwrite(self.sample_dir / "images/mif_cells.tif", self.mif, photometric="minisblack")
        for name, array in (("he_nuclei", self.he_nuclei), ("mif_nuclei", self.mif_nuclei), ("mif_cells", self.cells)):
            np.save(self.sample_dir / f"masks/{name}.npy", array)
        self.features = [{"geometry": mapping(box(0, 0, 1024, 512))}]

    def args(self, report="report", output="output", dry=False):
        args = ["--input-dir", str(self.input_dir), "--roi-dir", str(self.rois),
                "--samples", self.sample, "--report-dir", str(self.root / report),
                "--output-dir", str(self.root / output)]
        return args + (["--dry-run"] if dry else [])

    def run_xenium(self, **kwargs):
        with patch.object(xenium, "load_features", return_value=self.features):
            return xenium.main(self.args(**kwargs))

    def test_dry_run_writes_coordinates_but_no_zarr(self):
        self.assertEqual(self.run_xenium(dry=True), 0)
        self.assertFalse((self.root / "output").exists())
        self.assertFalse(list(self.root.rglob("*.zarr")))
        with (self.root / "report/sample.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([int(row["x_start"]) for row in rows], [0, 512])
        report = json.loads((self.root / "report/summary.json").read_text())
        self.assertEqual(report["historical_target"]["status"], "unverified")
        self.assertEqual(report["samples"][0]["metadata_comparison"]["status"], "unverified")

    def test_zarr_pixels_masks_dtypes_chunks_and_metadata_correspond(self):
        self.assertEqual(self.run_xenium(), 0)
        base = self.root / "output/sample"
        sources = {"he/images": (self.he, "uint8"), "mif/images": (self.mif.transpose(1, 2, 0), "uint16"),
                   "he/nuclei_masks": (self.he_nuclei, "int32"), "mif/nuclei_masks": (self.mif_nuclei, "uint32"),
                   "he/cell_masks": (self.cells, "uint32"), "mif/cell_masks": (self.cells, "uint32")}
        for name, (source, dtype) in sources.items():
            array = zarr.open(str(base / f"{name}.zarr"), mode="r")
            self.assertEqual(array.shape, (2, 512, 512, *source.shape[2:]))
            self.assertEqual(array.chunks, (1, 512, 512, *source.shape[2:]))
            self.assertEqual(array.dtype, np.dtype(dtype))
            self.assertEqual((array.compressor.cname, array.compressor.clevel), ("zstd", 3))
            np.testing.assert_array_equal(array[0], source[:512, :512])
            np.testing.assert_array_equal(array[1], source[:512, 512:])
        self.assertEqual((base / "he/metadata.csv").read_bytes(), (base / "mif/metadata.csv").read_bytes())
        self.assertFalse((self.root / "output/INCOMPLETE").exists())

    def test_existing_output_and_reports_are_never_overwritten(self):
        self.assertEqual(self.run_xenium(), 0)
        sentinel = self.root / "output/sentinel"
        sentinel.write_text("untouched")
        with self.assertRaises(FileExistsError):
            self.run_xenium(report="report2")
        self.assertEqual(sentinel.read_text(), "untouched")
        self.assertFalse((self.root / "report2").exists())
        with self.assertRaises(FileExistsError):
            self.run_xenium(output="output2", dry=True)

    def test_short_edge_tile_reported_without_silent_filtering(self):
        self.features = [{"geometry": mapping(box(0, 0, 100, 100))}]
        self.assertEqual(self.run_xenium(dry=True), 1)
        report = json.loads((self.root / "report/summary.json").read_text())
        self.assertEqual(report["total_patches"], 1)
        self.assertIn("undersized", report["samples"][0]["extraction_issues"][0])
        self.assertTrue((self.root / "report/sample.csv").is_file())

    def test_missing_masks_still_report_coordinates_and_fail(self):
        (self.sample_dir / "masks/he_nuclei.npy").unlink()
        self.assertEqual(self.run_xenium(dry=True), 1)
        report = json.loads((self.root / "report/summary.json").read_text())
        self.assertEqual(report["total_patches"], 2)
        self.assertFalse(report["historical_target"]["source_arrays_checked"])

    def test_missing_annotations_report_error_without_output(self):
        self.assertEqual(xenium.main(self.args(dry=True)), 1)
        report = json.loads((self.root / "report/summary.json").read_text())
        self.assertEqual(report["samples"][0]["status"], "error")
        self.assertFalse((self.root / "output").exists())

    def test_metadata_detects_order_duplicates_and_modality_mismatch(self):
        rows = metadata_rows(self.sample, [(0, [(0, 512, 0, 512), (0, 512, 512, 1024)])])
        reference = self.root / "reference"
        for modality in ("he", "mif"):
            folder = reference / self.sample / modality
            folder.mkdir(parents=True)
            write_metadata(folder / "metadata.csv", rows)
        self.assertEqual(compare_metadata(rows, reference, self.sample)["status"], "match")
        self.assertEqual(compare_metadata(rows[::-1], reference, self.sample)["status"], "mismatch")
        self.assertEqual(compare_metadata(rows + rows, reference, self.sample)["status"], "mismatch")
        (reference / self.sample / "mif/metadata.csv").unlink()
        self.assertEqual(compare_metadata(rows, reference, self.sample)["status"], "unverified")

    def test_requested_reference_failure_blocks_writing(self):
        with patch.object(xenium, "load_features", return_value=self.features):
            result = xenium.main(self.args() + ["--reference-dir", str(self.root / "missing")])
        self.assertEqual(result, 1)
        self.assertFalse((self.root / "output").exists())

    def test_original_count_manifest_comparison(self):
        counts = self.root / "counts.csv"
        counts.write_text("sample,num_patches\nsample,3\n")
        with patch.object(xenium, "load_features", return_value=self.features):
            result = xenium.main(self.args() + ["--expected-counts", str(counts)])
        self.assertEqual(result, 1)
        report = json.loads((self.root / "report/summary.json").read_text())
        self.assertEqual(report["samples"][0]["count_comparison"]["status"], "mismatch")
        self.assertFalse((self.root / "output").exists())

    def test_full_cohort_mismatch_blocks_writing(self):
        args = self.args()
        index = args.index("--samples")
        del args[index:index + 2]
        with patch.object(xenium, "load_features", return_value=self.features):
            self.assertEqual(xenium.main(args), 1)
        report = json.loads((self.root / "report/summary.json").read_text())
        self.assertEqual(report["historical_target"]["status"], "mismatch")
        self.assertEqual(report["historical_target"]["expected_total"], 1909)
        self.assertEqual(report["historical_target"]["expected_samples"], 9)
        self.assertFalse((self.root / "output").exists())

    def test_write_failure_keeps_incomplete_marker_and_failure_report(self):
        with patch.object(xenium, "write_sample", side_effect=OSError("test write failure")):
            with self.assertRaises(OSError):
                self.run_xenium()
        self.assertTrue((self.root / "output/INCOMPLETE").exists())
        report = json.loads((self.root / "report/summary.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["zarr_written"])
        self.assertIn("test write failure", report["write_error"])


class OrionWorkflowTests(unittest.TestCase):
    args = WorkflowTests.args

    def setUp(self):
        WorkflowTests.setUp(self)
        renamed = self.input_dir / "CRC01"
        self.sample_dir.rename(renamed)
        self.sample_dir, self.sample = renamed, "CRC01"

    def run_orion(self, **kwargs):
        with patch.object(orion, "load_features", return_value=self.features):
            return orion.main(self.args(**kwargs))

    def assert_arrays(self, output, coords):
        # Independent expected values: original channel 1 becomes output 0;
        # original channel 0 becomes output 1. Masks never change identities.
        swapped = np.stack((self.mif[1], self.mif[0]), axis=-1)
        sources = {"he/images": self.he, "mif/images": swapped,
                   "he/nuclei_masks": self.he_nuclei, "mif/nuclei_masks": self.mif_nuclei,
                   "he/cell_masks": self.cells, "mif/cell_masks": self.cells}
        for name, source in sources.items():
            array = zarr.open(str(output / self.sample / f"{name}.zarr"), mode="r")
            self.assertEqual(array.dtype, source.dtype)
            self.assertEqual(array.shape, (len(coords), 512, 512, *source.shape[2:]))
            self.assertEqual(array.chunks, (1, 512, 512, *source.shape[2:]))
            self.assertEqual((array.compressor.cname, array.compressor.clevel), ("zstd", 3))
            for index, (ys, ye, xs, xe) in enumerate(coords):
                cropped = source[ys:ye, xs:xe]
                expected = np.zeros((512, 512, *source.shape[2:]), dtype=source.dtype)
                expected[:cropped.shape[0], :cropped.shape[1]] = cropped
                np.testing.assert_array_equal(array[index], expected)
        he_csv = output / self.sample / "he/metadata.csv"
        self.assertEqual(he_csv.read_bytes(), (output / self.sample / "mif/metadata.csv").read_bytes())
        with he_csv.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([tuple(int(row[f]) for f in ("y_start", "y_end", "x_start", "x_end"))
                          for row in rows], coords)
        self.assertEqual([int(r["patch_idx"]) for r in rows], list(range(len(coords))))

    def test_channel_swap_for_both_source_layouts_and_six_array_contract(self):
        coords = [(0, 512, 0, 512), (0, 512, 512, 1024)]
        for layout in ("chw", "hwc"):
            with self.subTest(layout=layout):
                values = self.mif if layout == "chw" else self.mif.transpose(1, 2, 0)
                imwrite(self.sample_dir / "images/mif_cells.tif", values, photometric="minisblack")
                self.assertEqual(self.run_orion(report=f"report-{layout}", output=f"output-{layout}"), 0)
                self.assert_arrays(self.root / f"output-{layout}", coords)
                self.assertFalse((self.root / f"output-{layout}/INCOMPLETE").exists())

    def test_padding_retains_edge_and_fully_outside_candidates(self):
        self.features = [{"geometry": mapping(roi)} for roi in (
            box(900, 900, 1000, 1000), box(0, 0, 100, 100),
            box(1600, 1600, 1700, 1700), box(0, 0, 512, 512))]
        coords = [(694, 1206, 694, 1206), (0, 306, 0, 306),
                  (1394, 1906, 1394, 1906), (0, 512, 0, 512)]
        self.assertEqual(self.run_orion(), 0)
        self.assert_arrays(self.root / "output", coords)
        report = json.loads((self.root / "report/summary.json").read_text())
        self.assertEqual(report["samples"][0]["padded_patch_indices"], [0, 1, 2])
        self.assertEqual(report["samples"][0]["empty_patch_indices"], [2])
        self.assertTrue(report["zarr_written"])

    def test_dry_run_only_reports_coordinates_and_padding(self):
        self.features = [{"geometry": mapping(box(900, 900, 1000, 1000))}]
        self.assertEqual(self.run_orion(dry=True), 0)
        self.assertFalse((self.root / "output").exists())
        self.assertFalse(list(self.root.rglob("*.zarr")))
        report = json.loads((self.root / "report/summary.json").read_text())
        self.assertEqual(report["samples"][0]["padded_patch_indices"], [0])
        self.assertTrue(report["source_arrays_checked"])
        self.assertEqual(report["samples"][0]["metadata_comparison"]["status"], "unverified")

    def test_missing_source_keeps_coordinates_but_fails(self):
        (self.sample_dir / "images/he.tiff").unlink()
        self.assertEqual(self.run_orion(dry=True), 1)
        report = json.loads((self.root / "report/summary.json").read_text())
        self.assertEqual(report["total_patches"], 2)
        self.assertFalse(report["source_arrays_checked"])
        self.assertTrue((self.root / "report/CRC01.csv").exists())

    def test_existing_output_and_reports_refused(self):
        self.assertEqual(self.run_orion(), 0)
        sentinel = self.root / "output/sentinel"
        sentinel.write_text("untouched")
        with self.assertRaises(FileExistsError):
            self.run_orion(report="report2")
        self.assertEqual(sentinel.read_text(), "untouched")
        self.assertFalse((self.root / "report2").exists())
        with self.assertRaises(FileExistsError):
            self.run_orion(output="output2", dry=True)

    def test_reference_and_count_failures_block_conversion(self):
        counts = self.root / "counts.csv"
        counts.write_text("sample,num_patches\nCRC01,3\n")
        for index, extra in enumerate((["--expected-counts", str(counts)],
                                       ["--reference-dir", str(self.root / "missing")])):
            with patch.object(orion, "load_features", return_value=self.features):
                result = orion.main(self.args(report=f"report{index}") + extra)
            self.assertEqual(result, 1)
            self.assertFalse((self.root / "output").exists())

    def test_write_failure_keeps_incomplete_marker(self):
        with patch.object(orion, "write_sample", side_effect=OSError("test write failure")):
            with self.assertRaises(OSError):
                self.run_orion()
        self.assertTrue((self.root / "output/INCOMPLETE").exists())
        report = json.loads((self.root / "report/summary.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["zarr_written"])

    def test_output_required_and_empty_rois_reported_as_skipped(self):
        args = self.args()
        index = args.index("--output-dir")
        del args[index:index + 2]
        with self.assertRaises(SystemExit) as exc:
            orion.main(args)
        self.assertEqual(exc.exception.code, 2)
        self.assertFalse((self.root / "report").exists())
        with patch.object(orion, "load_features", return_value=[]):
            self.assertEqual(orion.main(args + ["--dry-run"]), 1)
        report = json.loads((self.root / "report/summary.json").read_text())
        self.assertEqual(report["total_patches"], 0)
        self.assertEqual(report["samples"][0]["status"], "skipped")
        self.assertTrue((self.root / "report/orion_crc_patch_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
