# Changelog

All notable changes are documented here. The format follows Keep a Changelog and the project
uses semantic versioning.

## [Unreleased]

- **DataModule constructors reject unknown keyword arguments.** `Cu3sDataModule`,
  `MultiCu3sDataModule`, and `TiffPairedDataModule` no longer end in a `**_` catch-all that
  silently dropped unrecognized kwargs. A typo or a removed option (e.g. an old `train_ids` /
  `predict_ids`) now raises `TypeError` at construction instead of being ignored. The nested
  `cls(**cfg.data)` shape still works: the one config-carried passthrough key, `data_module`, is
  accepted explicitly and ignored.

## 0.1.0 - 2026-06-22

- **Initial release.** Pluggable hyperspectral DataModules on cuvis-ai-core's SDK-free
  `BaseCuvisAIDataModule`, each declared in `configs/plugins/cuvis_ai_dataloader.yaml` as a
  `kind: data_module` entry (`data_module_name` + pip `extras`). The `cuvis` SDK lives only here,
  behind the `[cu3s]` extra.
- **`Cu3sDataModule`** (`cu3s`, `[cu3s, coco]`): reads `.cu3s` cubes via the `cuvis` SDK with
  COCO-derived masks, preserving core's former `SingleCu3sDataModule` surface (`cu3s_file_path`,
  `annotation_json_path`, `processing_mode`, `measurement_indices`, sibling `<stem>.json`
  auto-discovery). For single-frame access, call `.setup("predict")` then read `predict_ds`.
- **`TiffPairedDataModule`** (`tiff_paired`, `[tiff]`): reads a directory of TIFF cubes (SYX / YXS /
  YX) via `tifffile`, parses wavelengths from the `GDAL_METADATA` ENVI tag as `int32` nm for parity
  with the cu3s reader and channel selectors, and pairs stem-keyed PNG labels (default `label_rgb`).
- **`MultiCu3sDataModule`** (`cu3s_multi`, `[cu3s, coco]`): multi-file cu3s driven by a CSV split
  column (`split, cu3s_path, annotation_json, image_id`) with per-day COCO JSONs; runs module-owned
  or selector-driven, with a `read_index < total_measurements` bounds check at build.
- **Selector split model.** Each module implements `enumerate(required_attrs)` (attributed
  `SampleRef`s with source/read-index `uid`s, attributes materialized only when a selector needs
  them) and `build_dataset_from_refs(refs)`; readers are cached per source.
- **Attribute labelers.** `CocoLabeler` gains `is_annotated` / `categories_for`; `PairedPngLabeler`
  derives `category_ids` from PNG mask values, so `tag` / `categories` / AD-aware splits work for
  TIFF too.
- **Split resolvers + `resolve-splits` CLI.** `data/resolvers.py` (`resolve_random` /
  `resolve_stratified`, seeded, AD-aware train-on-normals, opt-in `group_by`, `import_csv_splits`)
  writes a committable `splits.json` (incl. `--from-csv`).
- **Range selectors.** A `data_dir` without `dataset_name` globs `*.cu3s` into one ordered universe;
  `measurement_indices` and split id-lists accept range strings (`"0-100"`, `"0-10:2"`); a ranged
  `image_id` fans a CSV row into one sample per measurement.
- **Lazy heavy-dep imports** (`data/_extras.py`): `cuvis` / `tifffile` / `pycocotools` load on first
  use, so the manifest registers with any subset of extras installed.
- **Dependencies.** Requires `cuvis-ai-core>=0.8.0` and `cuvis-ai-schemas>=0.6.0` from PyPI.
- **Packaging + CI.** Apache-2.0 metadata; a tag-triggered `pypi-release` workflow (build, validate,
  TestPyPI then PyPI via trusted publishing, GitHub release with SBOM + license report); a `ci`
  workflow (pytest+coverage, mypy, ruff, pip-audit / detect-secrets / bandit) and a compatibility
  workflow auditing dependency floors against core's lock; Dependabot for pip and Actions.

