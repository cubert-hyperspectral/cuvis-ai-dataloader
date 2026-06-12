# Changelog

## [Unreleased]

- Initial release. Pluggable hyperspectral **DataModules** on the SDK-free
  `cuvis_ai_core.data.datamodule.BaseHyperspectralDataModule`:
  - **`Cu3sDataModule`** (`cu3s`, extras `[cu3s, coco]`): reads `.cu3s` cubes via
    the `cuvis` SDK and attaches COCO-derived masks. Refactor of the former core
    `SingleCu3sDataModule`; the public surface (`cu3s_file_path`,
    `annotation_json_path`, `processing_mode`, `measurement_indices`, sibling
    `<stem>.json` auto-discovery) is preserved. `SingleCu3sDataModule` /
    `SingleCu3sDataset` back-compat aliases ship so old call sites migrate with
    only an import-path change.
  - **`TiffPairedDataModule`** (`tiff_paired`, extra `[tiff]`): reads a directory
    of TIFF cubes (axes SYX / YXS / YX) via `tifffile`, parses wavelengths from
    the GDAL_METADATA ENVI tag, and pairs stem-keyed PNG labels (default
    `label_rgb`).
  - **`MultiCu3sDataModule`** (`cu3s_multi`, extras `[cu3s, coco]`): multi-file
    cu3s driven by an external `splits.csv` (`split, cu3s_path, annotation_json,
    image_id`), per-day COCO JSONs, module-owned splits (leaves
    `DataConfig.splits = None` and overrides `build_stage_dataset`); one cached
    `CocoLabeler` per unique annotation JSON.
- **Richer split selectors.** `Cu3sDataModule` gains a **folder source**: a `data_dir` (directory)
  without `dataset_name` globs `*.{glob}` (default `cu3s`) into one ordered universe, and split
  selectors index into it by int position or filename stem. `measurement_indices` accepts inclusive
  range strings (`"0-100"`, `"0-10:2"`) alongside the comma list. `MultiCu3sDataModule` accepts a
  ranged `image_id` cell (`0-5`) that fans a CSV row out into one sample per measurement (reading
  measurement *m*, COCO `image_id` *m*); a scalar `image_id` keeps the legacy single-frame `read(0)`
  behavior. Inclusive range strings in `DataConfig.splits` id-lists are expanded by the core base.
- Internal `data/readers` (`Cu3sCubeReader`, `TiffCubeReader`) and `data/labelers`
  (`CocoLabeler` + `create_mask`, `PairedPngLabeler`) reused across modules; not a
  plugin contract.
- Lazy-import pattern (`data/_extras.py`): heavy deps load only on first use, so
  the manifest registers cleanly with any subset of extras installed.
- Manifest `configs/plugins/cuvis_ai_dataloader.yaml` declares each DataModule as
  a `kind: data_module` entry with its `data_module_name` + `extras`.
