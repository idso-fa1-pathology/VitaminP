# Pathologist ROI to paired Zarr datasets

## Source limitations and preserved behavior

These command-line tools transcribe the original Xenium and ORION conversion
cells supplied for the VitaminP manuscript revision. They keep the ROI
coordinate and extraction algorithms independent. The following behaviors
and ambiguities were documented before implementing the conversions:

- **The Xenium source is a complete converter.** Its intended reference result
  is 1,909 paired tiles across nine samples. The original images, masks, ROI
  files, individual sample counts, and released metadata are not present in
  this checkout, so this result cannot yet be verified.
- **The ORION source includes the complete converter and an ROI count scan.**
  The converter swaps mIF channels with `[..., [1, 0]]` after any channel-axis
  transpose. It keeps out-of-image candidates and zero-pads all six arrays
  when the H&E slice is not 512×512. Padding uses the H&E slice's height and
  width for both modalities, assuming registered arrays of matching sizes.
  These are ORION-specific operations; they are not applied to Xenium.
- ORION's original padding places the available slice at the tile's top-left
  and fills the remainder with zeros. Even an entirely out-of-image candidate
  can become an all-zero tile; there is no occupancy filter. Metadata retains
  the requested coordinates rather than clipped or padded bounds. These
  potentially surprising behaviors are preserved, not corrected.
- Both algorithms center a tile on an ROI when **both** bounding-box dimensions
  are smaller than 512. They compute endpoints before clamping negative start
  coordinates to zero. This can yield undersized tiles. The behavior is
  preserved; Xenium reports an extraction error rather than moving, padding,
  or silently dropping such a tile.
- Xenium's general grid checks upper image bounds but does not clamp negative
  starts. ORION's general grid clamps starts but has no image-bound check.
  These differences are retained, including the original integer truncation.
- Neither cell deduplicates overlapping ROIs, merges polygons, repairs invalid
  geometry, filters by ROI label, or excludes tiles based on tissue or masks.
  No such operations are added.
- Xenium uses `he_nuclei.npy` for H&E nuclei and `mif_cells.npy` for **both**
  modalities' cell masks. The later nuclei-mask replacement in
  `notebooks/data_preprocessing/Xenium_CRC.ipynb` is a separate operation and
  is **not** applied by this converter.
- ORION's original summary writer can fail on error rows because they contain
  an extra `error` field; its mean-per-ROI calculation can divide by zero for
  empty annotations. Structured reporting handles these reporting errors
  without changing ROI selection.

No original annotation is generated, edited, copied, committed, or published by
these tools. Before any public release, verify institutional permissions and
confirm that annotations, filenames, metadata, and source images contain no
patient-identifying information.

## Installation and scope

Run from the repository root with Python 3.10 or later. The required packages
are already declared in the repository's `requirements.txt`. For a dedicated
preprocessing environment without training dependencies:

```bash
python -m venv .venv-roi
source .venv-roi/bin/activate
python -m pip install 'numpy>=1.20' 'zarr>=2.16,<3' 'numcodecs>=0.11' \
  'shapely>=1.8' 'tifffile>=2022'
```

Zarr **v2** is required. `numcodecs.Blosc(cname="zstd", clevel=3)` is used
instead of the notebook's `zarr.Blosc` alias, preserving its codec settings.
Each report records the installed Python and package versions; the original
notebook environment was not supplied. Training, inference, and dataloader
code are not modified or run by these tools.

## Required source files and annotation naming

Xenium requires already registered images and instance-label masks on the
same pixel grid:

```text
/data/Xenium/
└── <sample>/
    ├── images/
    │   ├── he.tiff          # H,W,3 RGB; he.tif is the fallback
    │   └── mif_cells.tif    # H,W,2 or 2,H,W; .tiff is the fallback
    └── masks/
        ├── he_nuclei.npy
        ├── mif_nuclei.npy
        └── mif_cells.npy
```

The source labels mIF channel 0 as DAPI/nuclei and channel 1 as combined
cytoplasm. This tool does not reconstruct channels from raw instrument data,
register images, rescale coordinates, normalize intensities, rasterize
segmentation GeoJSONs, or relabel instance IDs. The upstream registration,
channel construction, and mask-generation workflows must be supplied
separately to reproduce these inputs from raw acquisitions.

Place the **original, unmodified pathologist ROI** files manually as follows:

```text
scripts/data_preprocessing/roi_to_zarr/
├── xenium_roi_to_zarr.py
├── orion_roi_to_zarr.py
├── _reporting.py
├── pathologist_geojson/
│   ├── xenium/
│   │   └── <sample>/pathologist_annotations.geojson
│   └── orion/
│       └── CRC<sample-id>/pathologist_annotations.geojson
├── tests/test_roi_pipelines.py
└── README.md
```

Sample directory names must exactly match the source data. Use `--roi-dir`
to point at another root with the same `<sample>/pathologist_annotations.geojson`
layout. To retain the notebook's original colocated layout, pass
`--roi-dir /data/Xenium` or `--roi-dir /data/ORION-CRC`. The scripts resolve
only that exact filename and never search for generic segmentation files.
Whole-cell and nuclear segmentation GeoJSONs are **not** pathologist ROIs and
must not be put under this name. The file format alone cannot establish its
provenance; the user must supply the correct original ROI annotations.

The annotation directories contain `.gitignore` files that exclude their
contents by default. No annotations are bundled here. Neither tool writes to
the source or ROI directories unless the user explicitly chooses a separate,
new report/output subdirectory there.

ORION's supplied scanner requires only sample directories whose names start
with `CRC`, plus their ROI annotations. The exact image files, mask files,
channels, and extraction-stage boundary handling needed for ORION Zarr
construction remain **unspecified by the supplied cell**.

## Exact tile-selection rules

GeoJSON features are processed in their original order; coordinates are
interpreted directly as image pixels (`x` = column, `y` = row). Coordinates
are not rounded to a global grid. Both functions use Python `int()`
truncation, a 512-pixel tile size, zero overlap, and a 512-pixel stride.
The CLI intentionally does not offer parameters to change these scientific
settings. Neither pipeline uses adaptive ROI expansion.

| Behavior | Xenium | ORION supplied scanner |
| --- | --- | --- |
| Both ROI bounding-box dimensions <512 | One centered candidate; clamp starts after computing ends | Same centered-candidate logic |
| Small ROI coverage threshold | No >10% test | No >10% test |
| Larger ROI | Grid from integer-truncated bounding-box minimum; row-major order | Its original grid with endpoint adjustment and nonnegative-start clamping |
| General grid coverage | Intersection area / **full tile area** must be strictly **>0.1** | Same strict threshold |
| Right/bottom image boundaries | Reject candidates whose endpoints exceed H&E dimensions, including centered tiles | No image-dimension check |
| Negative general-grid starts | Retained by original function | Clamped to zero by original function |
| Overlapping or repeated ROIs | Keep repeated tiles with distinct ROI indices | Keep repeated tiles with distinct ROI indices |
| Tissue, mask occupancy, ROI class filters | None | None |
| Padding, polygon repair, spatial transforms | None | None |

The Xenium CLI checks input shapes and the resulting Python slices before
writing. An incompatible array or undersized slice is a reported failure,
not a new exclusion criterion: candidate coordinates remain in the report,
and no tiles are shifted, padded, or silently dropped. A dry run needs the H&E
TIFF dimensions to apply the original boundary checks; missing mIF/mask files
are reported as extraction failures while available coordinates are retained.
It uses TIFF headers and memory-mapped NPY masks for these checks, without
loading whole images. Matching dimensions do not prove correct registration.

## Reproducing Xenium

First inspect the complete cohort without writing Zarr:

```bash
python scripts/data_preprocessing/roi_to_zarr/xenium_roi_to_zarr.py \
  --input-dir /data/Xenium \
  --report-dir /data/reproduction-reports/xenium-dry-run \
  --dry-run
```

Once the original released metadata and per-sample count table are available,
validate and convert using new destinations:

```bash
python scripts/data_preprocessing/roi_to_zarr/xenium_roi_to_zarr.py \
  --input-dir /data/Xenium \
  --output-dir /data/reproduced/Xenium-zarr \
  --report-dir /data/reproduction-reports/xenium-conversion \
  --reference-dir /data/released/Xenium-zarr \
  --expected-counts /data/references/xenium_original_counts.csv
```

The reference arguments are optional, but their absence is explicitly marked
as unverified. Do not populate the count file with guessed values. It must
contain the original `sample,num_patches` columns, one row per sample. The
ORION notebook's original summary CSV can also be used as a count manifest.
Paths above are placeholders, not downloadable data or known release paths.

By default, Xenium discovers sorted source directories, excludes `zarr_data`
and explicitly configured auxiliary roots, and checks for **nine samples and
1,909 total candidate tiles**. A mismatch returns a nonzero exit status and
prevents Zarr writing. For a deliberate subset, pass `--samples <sample> ...`;
the historical full-cohort target is then reported as unverified, while
per-sample reference checks still apply. Exact original sample names and their
individual counts have not yet been supplied.

### Xenium output contract

```text
<output-dir>/<sample>/
├── he/
│   ├── images.zarr         # N,512,512,3; uint8
│   ├── nuclei_masks.zarr   # N,512,512; he_nuclei.npy dtype
│   ├── cell_masks.zarr     # N,512,512; mif_cells.npy dtype
│   └── metadata.csv
└── mif/
    ├── images.zarr         # N,512,512,2; source mIF dtype
    ├── nuclei_masks.zarr   # N,512,512; mif_nuclei.npy dtype
    ├── cell_masks.zarr     # N,512,512; mif_cells.npy dtype
    └── metadata.csv
```

Every Zarr chunk contains one complete tile. Image values and label IDs are
copied without normalization or relabeling; H&E storage casts to `uint8`, as
in the original notebook. Only `(2,H,W)` mIF is transposed to `(H,W,2)`.
Both modalities use identical slices and patch indices. Both metadata files
have the original column order:

```text
patch_idx,sample,roi_idx,y_start,y_end,x_start,x_end
```

`patch_idx` starts at zero within each sample, `roi_idx` is the zero-based
GeoJSON feature index, and end coordinates are exclusive NumPy slice bounds.
This preserves the paths and shapes read by `dataset/dataset.py` for Xenium
and CRC. It does not claim equivalence to any later mask replacement or
unprovided preprocessing step. Actual conversion loads one sample's full
images and masks into memory, like the source notebook; plan RAM accordingly.

## Reproducing the supplied ORION scan

```bash
python scripts/data_preprocessing/roi_to_zarr/orion_roi_to_zarr.py \
  --input-dir /data/ORION-CRC \
  --report-dir /data/reproduction-reports/orion-dry-run \
  --reference-dir /data/released/ORION-CRC-zarr \
  --expected-counts /data/references/orion_crc_patch_summary.csv \
  --dry-run
```

Omit either reference option if that original artifact is unavailable; the
report will say so. The tool also writes `orion_crc_patch_summary.csv` with
the notebook's `sample,has_annotations,num_rois,num_patches` columns. Empty
feature collections produce zero candidates without division by zero;
missing/malformed annotations produce an error and a nonzero exit status.
Detailed errors are kept in `summary.json` rather than added to the summary
CSV's fixed column set.

Running ORION without `--dry-run` fails explicitly **before any writes**.
Candidate-coordinate CSVs use the same seven reporting columns as Xenium;
these are new audit reports, not evidence of the missing ORION writer's
metadata format. No ORION training dataset is generated until its original
extraction/writing implementation is provided.

## Validation, failure handling, and overwrite protection

Each run writes `<report-dir>/<sample>.csv` containing every candidate in
original order and a `summary.json` containing counts, source issues,
reference comparisons, and software versions. A dry run writes these audit
files only; it never creates a Zarr output directory.

`--reference-dir` is the released dataset root. For **each** sample, both
`he/metadata.csv` and `mif/metadata.csv` are compared read-only. Comparisons
check row counts and the ordered coordinate sequence, retaining duplicate
coordinates. If present, `patch_idx`, `sample`, and `roi_idx` are also
compared. Missing coordinate columns, reordered rows, count differences,
or different coordinates fail validation. Compared columns and the first
differing field are recorded. Missing reference files are marked unverified
and cause a nonzero exit status when reference validation was requested.

`--expected-counts` checks each processed sample. Missing count entries or
unprocessed manifest samples fail validation for a full-cohort run. With
explicit `--samples`, unprocessed manifest samples are listed without failing
the selected subset. An aggregate match to 1,909 is labeled
`count_match_only`; it does **not** establish sample-by-sample, coordinate,
pixel, or biological equivalence. ORION has no hardcoded expected total
because none was supplied.

Both report and dataset destinations must be new. Even existing empty
directories are rejected, and there is no overwrite flag. All sample plans
and requested reference checks finish before Xenium creates its output
root. Source or validation failures prevent Zarr writing. During writing,
an `INCOMPLETE` marker remains until every sample succeeds. A write failure
leaves the marker, records completed samples and the error in the report,
and returns nonzero; partial output must not be used for training. Inspect
it manually and use a new destination for a subsequent attempt.

Exit status 0 means the requested operation completed, **not** that the
published dataset has been reproduced. Status 1 indicates a source,
validation, or writing failure; status 2 indicates invalid CLI usage
(including an ORION conversion request). Review the verification fields.

## Regression checks and remaining evidence

```bash
python -m unittest discover \
  -s scripts/data_preprocessing/roi_to_zarr/tests -v
```

The tests cover strict coverage thresholds, small-ROI edge behavior, image
bounds, negative coordinates, polygon holes, duplicate ROI ordering,
paired array values/dtypes/chunks, metadata comparison, missing inputs,
and overwrite protection. Test images are temporary; test ROI geometries
stay in memory and no artificial annotation files are installed.

Before claiming reproduction of the released datasets, obtain:

1. Original pathologist ROI files and all registered Xenium images/masks.
2. The nine Xenium sample names, original individual counts, and both
   modalities' released metadata files (target total: 1,909).
3. The **ORION extraction and Zarr-writing cell**, corresponding source
   images/masks, original count outputs, and published metadata.
4. Any upstream image-registration/channel/mask methodology and downstream
   mask substitutions needed to connect these exact notebook inputs and
   outputs to the manuscript's final released training datasets.

No source images, original ROI files, released metadata, or original count
tables were available in this checkout during implementation. Code-level
regression checks cannot substitute for validation on those data.
