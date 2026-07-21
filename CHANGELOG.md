# Changelog

All notable changes are documented here. The format follows Keep a Changelog and the project
uses semantic versioning.

## [Unreleased]

- **DataModule constructors de-bloated: nested `DataConfig` handling centralized, three dead
  `cu3s` args removed.** A shared `accepts_data_config` decorator (`data/_extras.py`) now owns the
  `DataModule(**cfg.data)` nested-shape normalization (drops the redundant `data_module`, splices
  `params` onto the flat signature), replacing the duplicated `params`/`data_module`/`if params:`
  block in all four modules (`cu3s`, `cu3s_multi`, `npz_multi`, `tiff_paired`); behavior is
  unchanged for every real call shape. `Cu3sDataModule` drops three unused constructor args:
  `normalize_to_unit` (was accepted but inert), `dataset_name` (the `data_dir` + `dataset_name`
  single-file composition), and `glob` (cu3s folder mode is always `*.cu3s`). An unrecognized key
  inside `params` now raises `TypeError` instead of being silently dropped, matching the flat-path
  loud-rejection. Breaking only for callers that passed one of the three removed args (they now
  fail loudly at construction); no shipped config used them.
- **`cu3s` folder mode gained per-measurement enumeration (`frames: measurements`) + `recursive`.**
  Folder sources can now enumerate one sample per measurement per file (canonical absolute
  `Path.resolve().as_posix()` sources, sibling `<stem>.json` COCO attached, uid = `source#index`),
  which is the contract for externally authored `splits.json` (the CuvisNEXT split designer);
  `recursive: true` walks per-day subfolders. Default `frames: file` keeps the legacy
  one-ref-per-file-at-measurement-0 behavior. Documented in the README ("GUI-authored splits over
  a cu3s folder"), pinned by the committed golden fixture
  `tests/cuvis_ai_dataloader/fixtures/gui_authored_splits.json` (shared byte-for-byte with the
  CuvisNEXT test suite; its `universe_hash` is the shared sha256 test vector).
- **`cu3s` now refuses split-less training stages.** `setup("fit"/"validate"/"test")` with no
  `DataConfig.splits` raises instead of silently iterating the whole universe, which let
  statistical initialization (e.g. MinMax) ingest anomalous frames with no error; split-less
  `setup("predict")` (and `setup()` building only the predict dataset) keeps serving the whole
  universe. Breaking for pipelines that trained a cu3s source without splits — add a `splits`
  block (e.g. a frozen `splits.json` via `splits_path`).

## 0.4.0 - 2026-07-15

- Added `samples_per_frame: int = 1` to `MultiNpzDataModule`: index-level duplication of the train rows so each frame yields N independent samples per epoch (downstream per-sample transforms such as random crops draw fresh for every occurrence; the shuffled loader interleaves duplicates across the epoch). Val/test/predict are never expanded.
- **Renamed the `npz_multi` universe input `index_csv` → `universe_csv`, generalized its columns.** The
  argument is now `universe_csv` (a path to a `universe.csv`), one name across modules; its columns are
  `source, index, path` (was `npz_path, source, image_id`) plus optional `annotation, format, group`
  (`group` is reserved, carried onto `SampleRef.group` but not yet enforced by the leakage check).
  The converter, `cu3s-to-npz` CLI (`--universe-csv`), and `convert_split_manifest` emit the new columns and write
  `universe.csv` (`SplitManifestOutputs.universe_csv`). The reader now also rejects a duplicate `path`
  and any `..` path escape (on top of the existing duplicate-`(source, index)` guard). Regenerate
  npz `universe.csv` files; hand-authored splits.json are unaffected (identity is unchanged).
- **`MultiNpzDataModule` surfaces an optional `class_mask`.** Each `.npz` may now carry a
  `class_mask [H,W] uint8` (per-pixel COCO category id, 0 = background); it is emitted in the batch
  (zeros when absent) for per-class evaluation (e.g. per-class pixel AUROC). Backward-compatible:
  the extra key is additive and frames without it get a zero plane.
- **cu3s → per-frame NPZ converter (`data/npz_converter.py`) + `cu3s-to-npz` CLI.** Converts each
  measurement of a `.cu3s` (Preview → Reflectance via the cu3s reader) into one `.npz` for
  `npz_multi`, baking the frame's COCO annotations into `mask` (binary int32) + `class_mask` (uint8
  category id) via the COCO labeler, with optional edge crop. Emits a universe
  (`source, index, path`). **No train/val/test split is assigned** — splitting is a
  separate concern.

## 0.3.0 - 2026-07-01

- **Added `MultiNpzDataModule` (`data_module_name: npz_multi`).** A generic one-frame-per-file NPZ
  loader driven by a splits CSV (`split, npz_path, image_id`; extra columns ignored). Each `.npz`
  carries `cube [H,W,C] f32` + `wavelengths [C]`, and an optional baked `mask [H,W] int32` (zeros
  when absent); samples are `{cube, mask, wavelengths, mesu_index, frame_id}`. Needs no extras
  (numpy/torch are core) and no Cuvis SDK. Module-owned splits only (CSV `split` column). Unlike the
  cu3s modules it honors `pin_memory` / `persistent_workers` / `worker_multiprocessing_context`,
  since pure-CPU numpy loads benefit from them. Migrated from the cuvis-ai-dinomaly plugin so any
  pipeline can use it.
- Added a `no-local-sources` CI workflow that fails if `pyproject.toml` declares a local `[tool.uv.sources]` path entry (a machine-specific path must not ship in a release).

## 0.2.0 - 2026-06-23

- **DataModule constructors reject unknown keyword arguments.** `Cu3sDataModule`,
  `MultiCu3sDataModule`, and `TiffPairedDataModule` no longer end in a `**_` catch-all that
  silently dropped unrecognized kwargs. A typo or a removed option (e.g. an old `train_ids` /
  `predict_ids`) now raises `TypeError` at construction instead of being ignored. The nested
  `cls(**cfg.data)` shape still works: the one config-carried passthrough key, `data_module`, is
  accepted explicitly and ignored.
- **`Cu3sDataModule` datasets expose the wavelength axis.** `dm.<split>_ds.wavelengths_nm`
  (with a `wavelengths` alias matching the former dataset API) returns the per-channel
  wavelengths read once from the first sample's source, so consumers no longer have to pull a
  full cube via `ds[0]["wavelengths"]` just for the axis.

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

