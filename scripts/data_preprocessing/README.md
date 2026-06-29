# Dataset preprocessing

These command-line scripts convert raw public datasets into the Zarr archives
loaded by [`dataset/dataset.py`](../../dataset/dataset.py). The exploratory
notebooks remain unchanged under [`notebooks/data_preprocessing`](../../notebooks/data_preprocessing)
for transparency; these scripts provide the reproducible path-based workflows.

Install the project requirements before running a converter (Zarr v2,
NumCodecs, and headless OpenCV are declared there):

```bash
python -m pip install -r requirements.txt
```

All image-, slide-, patient-, case-, or fold-level split decisions happen
before tiling. This prevents patches from one source item leaking between
train, validation, and test sets. Generated Zarr directories can be large and
must not be committed.

## Output layouts

Flat datasets use this split-level layout:

```text
output_dir/
├── <dataset>_train/
│   ├── images.zarr
│   ├── nuclei_masks.zarr
│   └── metadata.csv
├── <dataset>_val/
│   └── ...
└── <dataset>_test/
    └── ...
```

`dataset.py` explicitly indexes PanNuke and TissueNet per sample, and MoNuSeg,
MoNuSAC, TNBC, CPM-15, DSB2018, and PanopTILs per image/slide. Their split or
full-set directory therefore contains sample subdirectories, each with the
same Zarr/CSV files. TissueNet additionally stores those files below `mif/`.
CPM-15 and PanopTILs have no split in their source notebooks and are emitted as
`cpm15_full` and `panoptils_full`; assign those full-set sample names in the
training configuration. DSB2018 is similarly retained as full modality
containers. This preserves both notebook behavior and the current loader
contract without changing `dataset.py`.

## Dataset commands

BC / DeepLIIF — source validation becomes test; source training is split
90/10 at image level. Panel 1 is the image and panel 6 is converted with the
notebook watershed logic.

```bash
python scripts/data_preprocessing/preprocess_bc.py \
  --input-dir /path/to/DeepLIIF \
  --output-dir /path/to/output_zarr/bc \
  --seed 42
```

CoNSeP — official test remains test; official training is split 90/10 before
edge-aligned 512x512 tiling.

```bash
python scripts/data_preprocessing/preprocess_consep.py \
  --input-dir /path/to/CoNSeP \
  --output-dir /path/to/output_zarr/consep \
  --seed 42
```

Kumar — official train/val/test directories are preserved and tiled into
512x512 patches with stride 450.

```bash
python scripts/data_preprocessing/preprocess_kumar.py \
  --input-dir /path/to/Kumar \
  --output-dir /path/to/output_zarr/kumar
```

Lizard — slides are shuffled and split 85/5/10 before 512x512 tiling; patches
are merged by their slide assignment.

```bash
python scripts/data_preprocessing/preprocess_lizard.py \
  --input-dir /path/to/Lizard \
  --output-dir /path/to/output_zarr/lizard \
  --seed 42
```

NuInsSeg — all tissues are pooled, split 70/15/15 at image level, padded to
512 multiples, then tiled.

```bash
python scripts/data_preprocessing/preprocess_nuinsseg.py \
  --input-dir /path/to/NuInsSeg \
  --output-dir /path/to/output_zarr/nuinsseg \
  --seed 19
```

CryoNuSeg — images are split 70/15/15 and saved directly because they are
already 512x512.

```bash
python scripts/data_preprocessing/preprocess_cryonuseg.py \
  --input-dir /path/to/CryoNuSeg \
  --output-dir /path/to/output_zarr/cryonuseg \
  --seed 42
```

PanNuke — four consecutive 256x256 tiles are mosaicked; folds 1+2 are split
90/10 and fold 3 is test.

```bash
python scripts/data_preprocessing/preprocess_pannuke.py \
  --input-dir /path/to/PanNuke \
  --output-dir /path/to/output_zarr/pannuke \
  --seed 42
```

MoNuSeg — source test remains test; source training slides are split 90/10
before XML rasterization, padding, and tiling.

```bash
python scripts/data_preprocessing/preprocess_monuseg.py \
  --input-dir /path/to/MoNuSeg \
  --output-dir /path/to/output_zarr/monuseg \
  --seed 42
```

MoNuSAC — source test remains test; source training images are split 90/10
before XML rasterization and tiling.

```bash
python scripts/data_preprocessing/preprocess_monusac.py \
  --input-dir /path/to/MoNuSAC \
  --output-dir /path/to/output_zarr/monusac \
  --seed 42
```

TNBC — patient slides 1-7 are train, 8-9 validation, and 10-11 test; binary
masks become connected-component instances.

```bash
python scripts/data_preprocessing/preprocess_tnbc.py \
  --input-dir /path/to/TNBC \
  --output-dir /path/to/output_zarr/tnbc
```

TissueNet — official NPZ splits are preserved. Native 256x256 samples are
stitched four at a time; native 512x512 samples are saved directly.

```bash
python scripts/data_preprocessing/preprocess_tissuenet.py \
  --input-dir /path/to/tissuenet_v1.1 \
  --output-dir /path/to/output_zarr/tissuenet
```

CPM-15 — the notebook's unsplit full dataset is tiled and written beneath
`cpm15_full`.

```bash
python scripts/data_preprocessing/preprocess_cpm15.py \
  --input-dir /path/to/CPM-15 \
  --output-dir /path/to/output_zarr/cpm15
```

CPM-17 — official test remains test; official training images are split 95/5
before hybrid padding/tiling with stride 400.

```bash
python scripts/data_preprocessing/preprocess_cpm17.py \
  --input-dir /path/to/CPM-17 \
  --output-dir /path/to/output_zarr/cpm17 \
  --seed 42
```

DSB2018 — Kaggle RLE annotations are decoded to instances. For fully audited
modality assignment, pass a CSV with `ImageId,modality` (`he` or
`fluorescence`) using `--modality-map`; otherwise dark-background modality is
classified deterministically.

```bash
python scripts/data_preprocessing/preprocess_dsb2018.py \
  --input-dir /path/to/DSB2018 \
  --output-dir /path/to/output_zarr/dsb2018 \
  --modality-map /path/to/dsb2018_modalities.csv
```

PanopTILs — polyline CSV annotations are rasterized and the notebook's unsplit
full dataset is written beneath `panoptils_full`.

```bash
python scripts/data_preprocessing/preprocess_panoptils.py \
  --input-dir /path/to/PanopTILs \
  --output-dir /path/to/output_zarr/panoptils
```

Xenium / ORION-CRC — the source notebook starts from existing registered Zarr
samples rather than raw public images. The script copies that tree and replaces
each H&E nuclei mask with its registered MIF nuclei mask. Passing the same path
for input and output additionally requires `--in-place`.

```bash
python scripts/data_preprocessing/preprocess_xenium_crc.py \
  --input-dir /path/to/existing_xenium_or_crc_zarr \
  --output-dir /path/to/output_zarr/xenium_or_crc
```
