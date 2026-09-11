# Changelog

All notable changes are documented here. The format follows Keep a Changelog and the project
uses semantic versioning.

## 0.9.0 - 2026-09-11

- **`cuda_cubes` hands out device-resident cubes**, off by default. A cube processed on the GPU
  was copied into host memory and its device copy freed, only for torch to copy it straight back
  for training; with this on, `batch["cube"]` is a zero-copy CUDA tensor (`cuvis.cuda` plus
  DLPack) and neither copy happens. Measured to a GPU-resident cube on one 940-frame `Raw`
  session: **1.55x** at one reader thread rising to **2.22x** at eight, because the copy is a
  shared resource that reader threads queue behind. Evidence: `benchmarks/cuda_cubes/report.md`.
- **The device cube is the same cube**, verified per element against the host path on real data:
  it is the buffer the SDK already produced, so this skips a copy rather than recomputing.
- Requires `sdk_cuda` and `num_workers=0`; the module raises rather than demoting either, since a
  CUDA tensor is not sent across the DataLoader worker queue and host processing leaves no device
  buffer to hand out. Turns itself off with a warning when the SDK, the device or the binding
  cannot support it.
- **Needs `cuvis` 3.6.0.0rc2**, which the `cu3s` extra requires. The rc1 wrapper could not hand
  out a device cube at all -- `CudaImageData._view` called a `cuvis_il` symbol no build exports,
  and its frees passed the handle by value where the C API declares a pointer, so no buffer was
  ever returned to the SDK's pool. Both are fixed upstream in cuvis.python#98 and verified here
  against rc2, so this package carries no workaround for them. Note `cuvis.cuda.capabilities()`
  could not have gated on it: it probes the native symbols and reports the path as available
  regardless.
- `Cu3sCubeReader` reads its first cube through `_read_with` rather than reaching into the
  measurement, so the rule about when a processing mode is applied lives in one place. This also
  fixes `processing_mode=None` opening: it no longer applies a mode while probing channel count.
  `Cu3sCubeReader.wavelengths` is now `int32`, matching `wavelengths_nm`.

## 0.8.0 - 2026-09-11

- **`sdk_cuda` selects the device the cuvis SDK processes on**, defaulting to the GPU. Available
  on both cu3s DataModules, on `convert_cu3s_file` / `convert_cu3s` and as `--no-sdk-cuda` on
  `cu3s-to-npz`. This closes the known limitation noted in 0.7.0: SDK 3.6.0 processes on the host
  unless a process calls `cuvis.init`, and nothing here called it, so 0.7.0 on its own moved every
  cu3s read onto the CPU. Measured on one 940-frame `Raw` session: the GPU is **5.7x** faster
  single-threaded (16.1 against 2.8 cubes/s) and **15.5x** at eight reader threads. Evidence:
  `benchmarks/sdk_device/report.md`.
- **`read_threads` is a GPU-only lever.** On the host, eight handles reach 3.34 cubes/s against a
  2.81 single-handle baseline (1.19x, against 3.2x on the GPU), so the two parameters are not
  independent. Documented rather than enforced, since a host-only machine is a legitimate
  configuration.
- **The device does not meaningfully change the cube.** Verified per element against the real SDK:
  the two implementations disagree by one LSB on 0.0001% of elements and by no more than one LSB
  anywhere, which is rounding in the cubalize interpolation.
- `require_cuvis` now performs the SDK init, once per process, because it is the one call every
  SDK entry in this package already goes through -- including inside a DataLoader worker, which is
  a fresh process that has initialized nothing. `Cu3sReaderCache` carries the choice across the
  pickle boundary so a worker reopens on the right device.
- **The first `cuvis.init` of a process wins.** The SDK fixes its device there and silently ignores
  every later one, returning success either way, so a host application that already initialized the
  SDK keeps its own choice and a second, conflicting `sdk_cuda` warns instead of pretending to
  switch. A machine without CUDA is unaffected: the SDK falls back to the host by itself.

## 0.7.0 - 2026-09-11

- **Threaded cu3s reading (`read_threads`).** A batch is read on several `cuvis.SessionFile`
  handles at once, all sharing one `ProcessingContext` -- the only topology measured to produce
  correct cubes, and the only one that fits on an 8 GB card, since a private context per handle
  costs both a ~13 s build and its own GPU buffers. Off by default. On one 940-frame session it
  takes reading from 14.8 to 54.5 cubes/s at six threads (3.7x), with zero wrong cubes verified
  per frame at every thread count from 1 to 16. Concurrency is bounded by `batch_size`, not by
  `read_threads`, because torch hands a map-style dataset a whole batch of indices and nothing
  earlier; `num_workers` must be 0. `Cu3sReaderCache` treats `read_threads` as a budget for the
  cache as a whole rather than a per-file count, so the open-handle total stays flat however many
  recordings an epoch touches, and the opt-in `source_coherent_batches` keeps a batch inside one
  recording so the cache stops evicting mid-batch. The `cu3s-to-npz` converter takes the same
  parameter. Evidence: `benchmarks/threaded_reading/report.md`.
- **`Cu3sCubeReader` shares its `ProcessingContext` with the SDK's lazy cube path.** The SDK's
  `Measurement.cube` property builds a second context when `session._pc` is unset, and that
  second build cost ~1.3 s per file open and held duplicate GPU and host buffers for the reader's
  lifetime while never being used.
- **Requires cuvis 3.6.0.0rc2** and the matching system-wide C++ Cuvis SDK. The floor is rc2
  rather than rc1 because rc1's device-buffer path is broken end to end (fixed upstream in
  cuvis.python#98); the binding is unchanged and stays at its own rc1. The Windows
  `cuvis-il<3.5.4` cap is gone: 3.6.0 publishes `win_amd64`, `manylinux_2_35_x86_64` and
  `manylinux_2_35_aarch64` wheels. `cuvis-il` is now named directly in the `cu3s` extra
  alongside `cuvis`, because uv only enables pre-releases for a package a direct requirement
  mentions one for.
- **The published `cuvis-il` 3.6.0 wheels release the GIL**, so `read_threads` is reachable from
  a plain install for the first time; every earlier wheel held it. The runtime probe and its
  single-threaded fallback stay, since they are what makes one config safe on both bindings.
- **Known limitation, removed in the next release:** on SDK 3.6.0 a process that never calls
  `cuvis.init` processes on the host rather than the GPU, at roughly 260 ms per cube against
  67 ms, and threading then buys almost nothing. This package does not yet initialize the SDK,
  so a host application wanting the GPU path must call
  `cuvis.init(cuvis.SdkSettings(force_gpu_mode="cuda"))` before constructing a DataModule.
- Added a `bench` extra carrying the plotting and process-memory dependencies the scripts under
  `benchmarks/` need; they previously relied on an undeclared ad-hoc environment.
## 0.6.3 - 2026-09-10

- `cu3s` folder mode opens only the recordings a run actually uses. A new `files` param
  takes the list outright (what the CuvisNEXT training wizard sends, since it already
  knows every recording its split assigns; a list or a comma string, and an empty list
  means "not given" so a preset can ship `files: []` as a placeholder). Without it, a
  split whose every selector names its sources (`files` / `file_indices`, or set
  operations over those) narrows the folder by itself, which fixes the same cost for a
  `restore-trainrun` over a folder. A positional or attribute-driven selector keeps the
  walk, since only the full universe can answer it. Enumeration previously built a full
  reader for every `*.cu3s` under `data_dir` before any selector was applied: an SDK
  session, a `ProcessingContext` and a read of measurement 0 each, plus every sibling
  COCO file when a constraint needed labels.
- Enumeration counts measurements with a bare session (`count_measurements`), no
  processing context, and names the recording when one cannot be opened; the SDK raises
  without the path in it. `cu3s_multi` uses the same probe for its read-index check.
- `validate_params` validates a `files` list and then leaves the folder alone.
  `data_dir` can be a filesystem root, and the folder check walks until it meets its
  first recording.
- A source a split names but cannot be read now fails by name, before enumeration,
  instead of surfacing later as core's anonymous "selector matched 0 samples".
- Floor: `cuvis-ai-core>=0.12.0`, the release that introduced the shared path-spelling
  rule this module now compares with. The lock moves to core 0.17.1 / schemas 0.12.0,
  which retired the top-level `leakage_check` flag in favour of typed split
  constraints; the GUI-authored-splits fixture and the two tests that asserted the old
  flag are updated to the contract the GUI actually writes.

## 0.6.2 - 2026-08-31

- Scoped the torch/torchvision cu128 index pin to a `cuda` dependency group (installed by
  default in this checkout): uv reads a git dependency's `[tool.uv.sources]`, so the previous
  unscoped pin leaked into every composed child environment pulling this plugin from git and
  collided with the host-mirrored torch index there (cu130 on a Jetson Thor host). Consumers
  never install a dependency's groups, so the scoped pin binds nothing outside this checkout;
  the committed lock now resolves torch from the cu128 index. On an aarch64 checkout, sync
  without the pin: `uv sync --no-default-groups`.

## 0.6.1 - 2026-08-27

- Fix RLE-object `segmentation` payloads silently rasterizing to 0 px under dataclass-wizard
  1.x: `Annotation.segmentation` is typed `Any` so polygon lists and RLE dicts pass through
  verbatim on every wizard version and entry point.
- `create_mask` raises `ValueError` on a present but unrecognized `segmentation` payload (flat
  polygon lists, RLE dicts without `counts`) instead of skipping it silently.
- Cap `dataclass-wizard<1.0` in the `coco` extra until the 1.x dump path is validated.

## 0.6.0 - 2026-08-21

- **`CocoLabeler` reads both COCO label dialects.** Track-dialect files (top-level
  `videos`, one annotation per track with per-frame parallel arrays — the shape the
  mask-tracking writer historically emitted and CuvisNEXT saves for tracked sessions) are
  now converted to standard image-keyed COCO in memory before pycocotools indexes them,
  with object identity preserved as an additive `track_id` per annotation. Previously such
  files failed at construction with a bare `KeyError: 'image_id'`, so datasets annotated
  with the tracking tools could not be used for training. Malformed or ambiguous inputs
  (hybrid `videos`+`images` files, empty or multi-entry `videos`, duplicate
  `frame_indices`, parallel-array length mismatches, non-RLE segmentation entries) are
  rejected with a `ValueError` naming the file instead of being silently mis-parsed.
- **Standard RLE-object `segmentation` rasterizes.** `create_mask`, `load_for`, and
  `Annotation.to_torchvision` now decode the standard COCO RLE dict form
  (`{"size": [H, W], "counts": str | list}`) — the form image-dialect mask exports carry —
  alongside polygons and the legacy non-standard `mask` key. Compressed string counts
  decode at their declared size and are padded/cropped to the label canvas on mismatch
  (with a warning); list counts keep the existing canvas-authoritative decode.

## 0.5.1 - 2026-08-20

- Documented the torch cu128 index tables as local-development-only: installs of this package
  as a git or registry dependency never read them, and composed child environments mirror the
  host's torch build (cuvis-ai-core >= 0.12.1).
- **Custom white/dark reference override for the cu3s reflectance path.** `Cu3sCubeReader` now
  accepts `white_ref` / `dark_ref` (paths to cu3s reference recordings) so an application can
  supply its own references at load time — reusing a shared calibration across sessions,
  non-destructively re-processing with updated references without re-exporting, or reading
  sessions that carry no usable baked references. Each reference is given as `path` or `path:frame`
  — `path`/`path:0` uses the reference session's measurement 0, `path:N` uses measurement N (for a
  session holding several references), and `path:-1` uses that session's embedded/baked reference
  (matching `cuvis_batch_exporter`'s `:frame_no` with `-1` = embedded). References are loaded via
  `get_measurement` — deliberately not `get_reference`, which on some sessions can return an
  unintended baked reference, except the explicit `-1` embedded case — and installed with
  `ProcessingContext.set_reference` before the processing mode is applied. A supplied reference also satisfies the Reflectance / SpectralRadiance
  reference validation. Threaded through `convert_cu3s_file` / `convert_cu3s` (`white_ref=` /
  `dark_ref=`) and the `cu3s-to-npz` CLI (`--white-ref` / `--dark-ref`). No references supplied →
  behaviour unchanged (baked references, bit-for-bit). This supplies references; it does not repair
  wrong ones — if a session's baked references are incorrect, correct them at the source with the
  exporter (`cuvis_batch_exporter --force_white/--force_dark`), byte-identical to this override.
  Use references matching the measurement's capture conditions; with `resume=True`
  previously-converted npz are reused as-is, so clear the output dir when re-converting with
  different references.

## 0.5.0 - 2026-07-27

- **cu3s reader cache is now a bounded LRU with close-on-evict.** `_Cu3sRefDataset` keeps at most `max_open_sessions` (default 4) open SDK sessions in an LRU, closing the least-recently-used on eviction (and via `close()` / `__del__`); each open Reflectance session holds native SDK GPU pools, and too many open at once crash the SDK's CUDA allocator ("illegal memory access") when torch shares the GPU. Keeps the per-dataset session footprint flat across a shuffled multi-file epoch.
- **`npz_multi` opts out of constraint sample-attrs (`supported_attrs() == frozenset()`).** The NPZ pool carries no per-frame tag/category metadata, so core's split-constraint evaluation reports `no_train_anomalous` as `unavailable` (soft-skipped or raised per severity) instead of crashing `enumerate` with `NotImplementedError`. Deriving `category_ids` from the baked `class_mask` is recorded as a follow-up in `TODOS.md`.
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
- **Unified `cu3s_multi` + `npz_multi` onto one `universe.csv` vocabulary via a shared parser (`data/_universe.py`).** Both modules now read `source, index` (required) plus optional `materialized_path, split, annotation, format, group`. `cu3s_multi`'s `splits_csv` argument is renamed `universe_csv` and its old columns (`split, cu3s_path, annotation_json, image_id`) are gone; `npz_multi`'s `path` column is renamed `materialized_path`. `materialized_path` defaults to `source` for cu3s (a raw `.cu3s` is its own file) and is required for npz (the physical file is the derived `.npz`). An inline `split` column is honored only by `cu3s_multi` (present → module-owned, absent → a training stage needs a splits.json; a splits.json always wins), and rejected by `npz_multi`. `source` is posix-normalized in both modules, fixing a cross-module `(source, index)` selector-key mismatch on Windows so one splits.json resolves against both the raw cu3s data and the converted npz. `cu3s_multi` no longer decouples a scalar `image_id` from the read index (`index` is now both). Regenerate every `universe.csv` / split CSV to the new columns; the converter, `cu3s-to-npz`, and `resolve-splits --from-csv` emit/consume them.

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

