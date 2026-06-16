# Changelog

## [Unreleased]

- Initial release. Pluggable hyperspectral **DataModules** on the SDK-free
  `cuvis_ai_core.data.datamodule.BaseCuvisAIDataModule`:
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
    image_id`), per-day COCO JSONs. Runs **module-owned** (the CSV `split` column,
    `DataConfig.splits = None`) **or selector-driven** (the CSV rows are the
    `enumerate()` universe); `read_index < total_measurements` is checked at build.
- **Selector model + universe enumeration.** Each module implements
  `enumerate(required_attrs)` (attributed `SampleRef`s with a content-derived `uid`, canonical order,
  attributes only when a `tag`/`categories` selector needs them) and `build_dataset_from_refs(refs)`;
  the cu3s/tiff readers are cached per source, so single-file mode opens one SDK session and folder
  mode stays a glob at setup. `CocoLabeler` gains `is_annotated`/`categories_for`; `PairedPngLabeler`
  derives `category_ids` from PNG mask values, so `tag`/`categories`/AD-aware work for tiff too.
- **Split resolvers + `resolve-splits` CLI.** New `data/resolvers.py` (`resolve_random` /
  `resolve_stratified`, seeded, AD-aware train-on-normals, opt-in `group_by` to keep a file whole;
  `import_csv_splits` to fold a cu3s_multi CSV into selectors) and a `resolve-splits` CLI that writes
  a committable `splits.json` (incl. `--from-csv`).
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
