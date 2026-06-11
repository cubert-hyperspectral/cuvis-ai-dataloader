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
- Internal `data/readers` (`Cu3sCubeReader`, `TiffCubeReader`) and `data/labelers`
  (`CocoLabeler` + `create_mask`, `PairedPngLabeler`) reused across modules; not a
  plugin contract.
- Lazy-import pattern (`data/_extras.py`): heavy deps load only on first use, so
  the manifest registers cleanly with any subset of extras installed.
- Manifest `configs/plugins/cuvis_ai_dataloader.yaml` declares each DataModule as
  a `kind: data_module` entry with its `data_module_name` + `extras`.
