# cuvis-ai-dataloader

Pluggable hyperspectral DataModules for the cuvis-ai ecosystem.

[![PyPI version](https://img.shields.io/pypi/v/cuvis-ai-dataloader.svg)](https://pypi.org/project/cuvis-ai-dataloader/)
[![CI Status](https://github.com/cubert-hyperspectral/cuvis-ai-dataloader/actions/workflows/ci.yml/badge.svg)](https://github.com/cubert-hyperspectral/cuvis-ai-dataloader/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/cubert-hyperspectral/cuvis-ai-dataloader/branch/main/graph/badge.svg)](https://codecov.io/gh/cubert-hyperspectral/cuvis-ai-dataloader)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

## Overview

`cuvis-ai-dataloader` ships the concrete hyperspectral **DataModules** for the
[Cuvis.AI](https://github.com/cubert-hyperspectral/cuvis-ai) ecosystem. A
DataModule is the unit the framework uses for both training and inference: it
bundles the data, the labels, the splits, and the `train` / `val` / `test` /
`predict` dataloaders.

This single plugin holds every concrete loader, with per-format heavy deps gated
behind optional extras. The `cuvis` SDK lives **only** here, behind `[cu3s]`; no
other Cuvis.AI repo pins it.

| Module (`data_module_name`) | Reads | Labels | Extra |
|---|---|---|---|
| `cu3s` | one `.cu3s` session (or a folder of them) via `cuvis` | COCO JSON | `[cu3s, coco]` |
| `cu3s_multi` | many `.cu3s`, one frame per CSV row | per-day COCO JSON | `[cu3s, coco]` |
| `npz_multi` | many `.npz`, one frame per CSV row | baked `mask` array | none |
| `tiff_paired` | a folder of `*.tif` / `*.tiff` cubes via `tifffile` | paired PNG | `[tiff]` |

Key points:
- One plugin, three DataModules, per-format heavy deps behind extras
- The `cuvis` SDK is isolated here, behind `[cu3s]`
- Composable split selectors over an attributed sample universe
- `cuvis` / `tifffile` / `pycocotools` import lazily, so the plugin registers with any subset of extras

### Splits

Splits are defined in one of two ways:

- **Selectors (`splits.json`)** is the general mechanism, shared by every module.
  Composable selectors over an attributed sample universe are resolved into a
  committable `splits.json` by the `resolve-splits` CLI, then referenced from a
  `DataConfig.splits`.
- **One `universe.csv` vocabulary** (`source, index` + optional `materialized_path, split,
  annotation, format, group`) is read by both `cu3s_multi` and `npz_multi` through a shared
  parser; each module keeps its own reader. `cu3s_multi` may carry an inline `split` column
  (present → module-owned; absent → needs a `splits.json`), and `resolve-splits --from-csv`
  turns that column into a committable `splits.json`. `npz_multi` is selector-only (it rejects a
  `split` column) and requires `materialized_path` (the `.npz`); for `cu3s_multi`,
  `materialized_path` defaults to `source` (a raw `.cu3s` is its own file). `source` is the posix
  identity a `splits.json` selector keys on, so one split resolves against both the raw cu3s data
  and the converted npz.

## Installation

```bash
uv pip install "cuvis-ai-dataloader[cu3s,coco]"   # cu3s + COCO
uv pip install "cuvis-ai-dataloader[tiff]"         # TIFF + paired PNG
uv pip install "cuvis-ai-dataloader[all]"          # every format
```

Extras:
- `cu3s`: `.cu3s` session reading via the `cuvis` SDK binding
- `coco`: COCO-JSON mask labels (`pycocotools`, `scikit-image`)
- `tiff`: TIFF cube reading (`tifffile`)
- `all`: All formats
- `bench`: Plotting and process-memory deps for the scripts under `benchmarks/`
- `dev`: Development dependencies

The `cu3s` extra requires `cuvis` 3.6.0.0, which pulls the matching `cuvis-il`.

### Cuvis SDK (system install, required for `cu3s`)

The `[cu3s]` extra installs the `cuvis` **binding**, but that binding needs the system-wide
**C++ Cuvis SDK** at a matching version (3.6.0) too, or any `.cu3s` read fails at import with
`DLL load failed while importing _cuvis_pyil`. See the
[Cuvis.AI installation guide](https://docs.cuvis.ai/latest/get-started/installation/) for OS
support (Windows / Linux; not macOS), the SDK download, and verification. Quick check once
installed:

```bash
uv run python -c "import cuvis; print(cuvis.version())"
```

## Usage

**Inference** (`restore-pipeline`) selects a module and its params on the CLI:

```bash
restore-pipeline \
  --pipeline-path X.yaml \
  --plugins-dir   <this-repo>/configs/plugins \
  --data-module cu3s \
  --data-arg    cu3s_file_path=X.cu3s \
  --data-arg    annotation_json_path=Y.json
```

**Training** (`Train` / `RestoreTrainRun`) selects the same module via the yaml
`DataConfig`:

```yaml
data:
  data_module: cu3s
  splits:
    train:
      - { kind: file_indices, source: X.cu3s, ids: [0, 2, 3] }
    val:
      - { kind: file_indices, source: X.cu3s, ids: [1, 5] }
  batch_size: 4
  params:
    cu3s_file_path: X.cu3s
    annotation_json_path: Y.json
    processing_mode: Reflectance
```

**In-process / notebooks** construct the DataModule directly and run it through
the `Predictor`:

```python
from cuvis_ai_dataloader.data import Cu3sDataModule
from cuvis_ai_core.training import Predictor
from cuvis_ai_core.utils.restore import restore_pipeline

pipeline = restore_pipeline("X.yaml", plugins_dirs=[...])
dm = Cu3sDataModule(cu3s_file_path="X.cu3s", batch_size=1)
Predictor(pipeline, dm).predict()
```

### Threaded cu3s reading (`read_threads`)

Reading a cu3s frame is dominated by the SDK, not by Python: about 78 ms per cube on a warm
file cache, almost none of it under the interpreter.
`read_threads` reads a batch on several `SessionFile` handles at once, all sharing one
`ProcessingContext`, which is the only topology measured to produce correct cubes.
It is off by default.

```yaml
data:
  data_module: cu3s
  batch_size: 8          # the lever: concurrency is bounded by the batch
  num_workers: 0         # required; reader threads replace worker processes
  params:
    cu3s_file_path: X.cu3s
    read_threads: 8
```

Measured on one 940-frame session, CUDA Raw mode, 20 cores and an RTX 4070, SDK 3.6.0, median of
five passes per cell ([full evidence](benchmarks/threaded_reading/report.md)):

| read_threads | fps | RSS GB |
| --- | --- | --- |
| 1 | 12.9 | 2.7 |
| 2 | 25.6 | 4.5 |
| 4 | 40.7 | 5.2 |
| 6 | 50.7 | 5.7 |
| 8 | 52.6 | 6.3 |
| 16 | 50.1 | 8.5 |

Eight is the knee and, on this SDK build, also the peak: twelve and sixteen are slower, not
faster. Six reaches 96% of eight for 91% of its memory. Budget roughly 0.38 GB of RSS per
handle, and re-measure the thread count on your own machine rather than copying this table.

Things worth knowing before turning it on:

- **It only helps on the GPU.** On the SDK's host device, eight handles buy almost nothing.
  Leave `sdk_cuda` on (its default); see
  [SDK processing device](#sdk-processing-device-sdk_cuda) below.
- **It needs a cuvis binding that releases the GIL.** The published `cuvis-il` 3.6.0.0 wheels
  do; every wheel before them held it. On a binding that holds it, extra threads are a small loss
  rather than a gain, so the reader probes the binding once per process and falls back to
  single-threaded reads with a warning.
- **Concurrency equals `batch_size`.** torch hands a dataset a whole batch of indices at once and
  nothing earlier, so `batch_size: 1` gets no speedup from `read_threads` alone;
  [`read_ahead`](#read-ahead-at-batch-1-read_ahead) is the lever there, and the module warns
  when threads are configured at batch 1 without it.
- **`num_workers` must be 0.** Worker processes each build their own sessions and their own
  processing context, so combining the two multiplies both; the module raises instead.
- **`samples_per_frame > 1` disables it for the train loader only.** The repeat wrapper in
  `cuvis-ai-core` does not forward the batched fetch. Validation, test and predict are unaffected.
- **Host processing mode barely benefits.** There the lever is `processing_thread_count` in
  the SDK settings, not this parameter.
- **Multi-file divides the budget across recordings unless batches are source-coherent.** Without
  `source_coherent_batches`, `read_threads` is divided by the sessions the cache holds open
  (`max_open_sessions`, default 4), so the handle count stays flat; `read_threads: 4` over four or
  more recordings then leaves one thread per recording, only batches spanning several recordings
  overlap, and the reader warns about the split. `source_coherent_batches: true` keeps each batch
  inside one recording, so the cache stops evicting mid-batch and every recording gets the whole
  `read_threads` inside itself, at a cost of up to `max_open_sessions x read_threads` open handles
  (about 0.38 GB of RSS each): pair it with `max_open_sessions: 2`. It changes which samples share
  a batch and replaces the loader's sampler, so it cannot be used under DDP.

The npz converter takes the same parameter and needs no batch size, since it already knows every
index it will read:

```bash
cu3s-to-npz --cu3s X.cu3s --out-dir out --annotations sibling --read-threads 8
```

### Read-ahead at batch 1 (`read_ahead`)

`read_threads` overlaps the reads inside one batch, so a `batch_size: 1` loader (the CuvisNEXT
training wizard's) gains nothing from it: torch asks the dataset for one frame at a time and
Lightning never prefetches for a sized loader, so the SDK read of frame i+1 waits for the model
step on frame i. `read_ahead` closes that gap. The module owns the loader's batch sampler,
learns the epoch's order the moment torch draws it, and keeps up to `read_ahead` frames of that
order in flight on reader threads while the model works on the current one. Off by default.

```yaml
data:
  data_module: cu3s
  batch_size: 1
  num_workers: 0         # required; the read-ahead runs on reader threads in this process
  params:
    cu3s_file_path: X.cu3s
    read_ahead: 2        # frames in flight ahead of the model step
```

What it costs and what to know:

- **One whole cube per frame in flight.** A 1000x1080x61 Reflectance cube is about 264 MB, in
  host memory, or in device memory with `cuda_cubes`. Two is the sensible depth on an 8 GB card;
  the cap is 8.
- **Handles follow the depth, not `read_threads`.** A read-ahead of depth k runs on a pool k
  threads wide per recording (`read_threads` still wins when it is larger), so depth 1 opens no
  extra handle and depth 2 one more (about 0.38 GB of RSS each).
- **The order is exactly torch's.** With `shuffle=True` the sampler draws the same permutation a
  plain loader would for the same seed, so a run reproduces frame for frame with `read_ahead` on
  or off.
- **A consumer that leaves the announced order is served synchronously.** A `__getitem__`
  caller or an out-of-order fetch makes the plan stop looking ahead for that epoch and say so
  once; no frame is ever handed out for the wrong index. An iterator dropped mid-epoch
  (Lightning's two-batch sanity check, an early stop, a statistical pass that stops after its
  first frames) releases the frames read ahead of it the moment torch lets go of the iterator,
  so at most `read_ahead` reads are wasted and nothing lingers in memory until the next epoch.
  A reader with reads in flight is never evicted from the cache, and the cache shrinks back to
  `max_open_sessions` on its next access once those reads land.
- **With `cuda_cubes`, the reading thread synchronizes the device before handing a cube over.**
  The SDK's DLPack export carries no stream, so a device-wide `torch.cuda.synchronize` on the
  reader thread is what orders the SDK's writes against the model's stream; depth 2 hides its
  latency. Provisional until the SDK offers an event for its buffers.
- **`num_workers` must be 0**, for the same reason as `read_threads`; the module raises.
- **`samples_per_frame > 1` disables it for the train loader only** (the repeat wrapper in
  `cuvis-ai-core` forwards neither the batched fetch nor the order); the loader warns.
- **Not under DDP.** It replaces the loader's sampler, like `source_coherent_batches`, with which
  it composes.
- **Where the time goes.** On the CuvisNEXT wizard trainrun (RTX 5070 Ti, 1000x1080x61
  Reflectance, batch 1) a training step took 0.24 s, of which the SDK read was about 0.10 s and
  the cube's host round trip 0.06 s; the read-ahead overlaps the first, `cuda_cubes` removes the
  second. The measured effect is in the changelog entry of the release that shipped it.

### SDK processing device (`sdk_cuda`)

The cuvis SDK can process cubes on the GPU or on the host CPU. This package asks for the GPU,
which is what `sdk_cuda: true`, the default, means. Turn it off to run on the host.

```yaml
data:
  data_module: cu3s
  params:
    cu3s_file_path: X.cu3s
    sdk_cuda: false      # default true
```

`cu3s-to-npz` takes `--no-sdk-cuda` for the same thing.

On one 940-frame session, `Raw` mode, RTX 4070
([full evidence](benchmarks/sdk_device/report.md)):

| read_threads | GPU | host | ratio |
| --- | --- | --- | --- |
| 1 | 16.8 | 3.1 | 5.4x |
| 8 | 49.3 | 3.7 | 13.5x |

- **The flag exists because SDK 3.6.0 changed the default.** A process that never calls
  `cuvis.init` now processes on the host. Nothing in this package called it, so the upgrade
  would otherwise have moved every cu3s read onto the CPU silently.
- **`read_threads` is a GPU-only lever.** On the host the 1-to-8 sweep stays flat at about
  3 cubes/s, against 2.9x on the GPU. The two parameters are not independent.
- **A machine without CUDA needs no change.** The SDK falls back to the host on its own, so
  the default costs nothing there.
- **The device barely changes the cube**, by one LSB on 0.0001% of elements and no more than
  one LSB anywhere on the measured session: rounding in the cubalize interpolation, well under
  sensor noise.
- **The first `cuvis.init` in a process wins.** The SDK fixes its device there and silently
  ignores every later one, returning success. So a host application that already initialized
  the SDK keeps whatever device it chose, and two DataModules disagreeing in one process is a
  warning rather than a second switch. This package initializes at DataModule construction and
  again in each DataLoader worker, which is a fresh process that has initialized nothing.

### Device-resident cubes (`cuda_cubes`)

A cube processed on the GPU is copied into host memory and its device copy freed, only for
torch to copy it straight back for training. `cuda_cubes` keeps it where it already is and
hands the batch a zero-copy CUDA tensor. Off by default, because it changes what a batch
contains.

```yaml
data:
  data_module: cu3s
  batch_size: 8
  num_workers: 0         # required
  params:
    cu3s_file_path: X.cu3s
    read_threads: 8
    cuda_cubes: true     # default false
```

Cubes per second delivered **onto the GPU**, one 940-frame session, `Raw`, RTX 4070
([full evidence](benchmarks/cuda_cubes/report.md)):

| read_threads | `cuda_cubes: true` | `cuda_cubes: false` | ratio |
| --- | --- | --- | --- |
| 1 | 24.1 | 15.5 | 1.55x |
| 4 | 74.4 | 34.3 | 2.17x |
| 8 | 84.1 | 37.9 | 2.22x |

- **`batch["cube"]` becomes a CUDA tensor** instead of a host one. It is the same cube, bit for
  bit, on the same device torch would have put it on; what changes is that nothing downstream
  should call `.cuda()` on it, and anything calling `.numpy()` on it will now fail.
- **The gain grows with `read_threads`**, because the copy is a shared resource that reader
  threads queue behind. The two parameters compound.
- **`num_workers` must be 0** and `sdk_cuda` must be on; the module raises rather than
  demoting either.
- **It turns itself off** when the SDK or the device cannot support it, with a warning, and
  reads through host memory instead.
- **It needs `cuvis` 3.6.0.0**, which the `cu3s` extra requires. The 3.6.0.0rc1 wrapper's
  device-buffer path raised on first use (fixed upstream in cuvis.python#98), and
  `cuvis.cuda.capabilities()` does not catch that, since it probes the native symbols rather
  than the Python layer over them.
- **A reader opened with `cuda_cubes: false` in a process that already switched to device
  cubes** copies each cube back to host memory and warns once, since the SDK's switch is
  process-wide and one-way; its callers keep getting numpy.

### NPZ (`npz_multi`)

`npz_multi` loads one frame per compressed `.npz`, selected by a `splits.json` over a
`universe_csv` (a `universe.csv`). It needs no extras (numpy is a core dep) and no Cuvis SDK. Each `.npz` carries:

- `cube`: `[H, W, C]` float32
- `wavelengths`: `[C]` (cast to int32)
- `mask` (optional): `[H, W]` int32 ground truth (zeros are emitted when absent)
- `class_mask` (optional): `[H, W]` uint8 per-pixel COCO category id (0 = background)

The `universe_csv` requires `source, index` plus `materialized_path` (the `.npz`, required for npz;
optional `annotation, format, group`; extra columns are ignored); `materialized_path` is relative
to the CSV and must not escape it via `..`. A `split` column is rejected here (npz is
selector-only). Each sample is
`{cube, mask, class_mask, wavelengths, mesu_index, frame_id}`. Unlike the cu3s modules, `npz_multi`
honors `pin_memory` / `persistent_workers` / `worker_multiprocessing_context` (pure-CPU numpy loads
benefit from them).

```python
from cuvis_ai_dataloader.data import MultiNpzDataModule
from cuvis_ai_schemas.training.data import DataSplitConfig, Selector, SelectorKind

splits = DataSplitConfig(
    train=[Selector(kind=SelectorKind.FILE_INDICES, source="X.cu3s", ids=[0, 2, 3])],
    val=[Selector(kind=SelectorKind.FILE_INDICES, source="X.cu3s", ids=[1, 5])],
)
dm = MultiNpzDataModule(splits=splits, universe_csv="universe.csv", batch_size=4, num_workers=4)
dm.setup("fit")
batch = next(iter(dm.train_dataloader()))  # cube [B,H,W,C], mask [B,H,W], ...
```

In a `DataConfig` (training / `restore-trainrun`):

```yaml
data:
  data_module: npz_multi
  batch_size: 4
  splits:
    train:
      - { kind: file_indices, source: X.cu3s, ids: [0, 2, 3] }
    val:
      - { kind: file_indices, source: X.cu3s, ids: [1, 5] }
  params:
    universe_csv: universe.csv
```

### GUI-authored splits over a cu3s folder (contract)

External split authors (e.g. the CuvisNEXT split designer) write a frozen `splits.json`
(a serialized `DataSplitConfig` with `file_indices` selectors) against a **folder of cu3s
files with per-measurement granularity**. That contract is `cu3s` folder mode with
`frames: measurements`:

```yaml
data:
  data_module: cu3s
  batch_size: 1
  num_workers: 0
  splits:
    splits_path: <absolute path to the frozen splits.json>
  params:
    files:                   # the recordings to use, when the author knows them
      - <absolute path to a .cu3s>
    data_dir: <folder holding the .cu3s files>
    frames: measurements
    recursive: true          # walk per-day subfolders (fallback only)
    processing_mode: Reflectance
```

The frozen rules both sides implement:

- **Universe** = the recordings the run actually uses, one sample per measurement
  `0..N-1`, ordered by `(source, index)`. Which recordings those are is answered in this
  order: the `files` list when given (nothing is walked, and the list may point outside
  `data_dir`, e.g. a split spanning two drives); otherwise the sources the split's
  selectors name, when every selector names its sources (`files` / `file_indices`, or set
  operations over those); otherwise every `*.cu3s` under `data_dir`, recursive when
  `recursive: true`. A positional or attribute-driven selector (`dir_indices`, `stems`,
  `glob`, `tag`, `categories`, `all`) can only be answered by the full universe, so it
  keeps the walk. Enumeration opens each recording once for its measurement count only
  (no processing context), so a folder holding recordings the split does not name costs
  nothing.
- **Source identity is canonical**: the absolute path with forward slashes and
  filesystem-true case — Python `Path(p).resolve().as_posix()`, C++/Qt
  `QFileInfo::canonicalFilePath()`. Selectors in the authored `splits.json` must carry
  exactly this form; a moved or renamed member file fails loud, naming the recording it
  could not read, rather than silently shrinking a split. Sources reached only through a
  set operation may be absent (`except(files[a], files[gone])` is legitimate, and core
  resolves a set operation's operands without its zero-match check).
- **An empty `predict` stage** serves the whole universe, which with an explicit `files`
  list or a fully source-naming split means the recordings that split uses, not the
  whole folder.
- **`uid` = `<source>#<index>`** (the sibling COCO image id equals the read position, so
  it never extends the uid). `universe_hash` = sha256 over the ordered uids, each
  followed by `\n` (`cuvis_ai_core.data.splits_io.universe_hash`). For `file_indices`
  splits the server treats the hash as informational (only positional `dir_indices`
  splits are hash-verified); staleness detection is the author's concern.
- **Annotations** are the sibling `<stem>.json` COCO next to each cu3s (attached
  automatically); an empty `predict` stage means the whole universe.
- **Training stages require splits.** `cu3s` does not own split semantics: `fit` /
  `validate` / `test` with no `DataConfig.splits` raise instead of silently iterating
  the whole universe (which would contaminate statistical initialization with anomalous
  frames). Split-less `predict` over the whole universe stays valid.

The golden fixture `tests/cuvis_ai_dataloader/fixtures/gui_authored_splits.json` is the
byte-level reference of the authored shape (the `{DATA_DIR}` token stands in for the
machine-specific folder); the same file is committed in the CuvisNEXT test suite and its
`universe_hash` doubles as the shared sha256 test vector. Changing it is a cross-repo
contract change.

## Architecture

Concrete DataModules subclass `cuvis_ai_core.data.datamodule.BaseCuvisAIDataModule`
and implement `validate_params` plus the selector hooks `enumerate(required_attrs)`
(the module's attributed sample universe) and `build_dataset_from_refs(refs)`;
a module that owns its own splits also implements `build_stage_dataset(stage)`.
Per-format cube readers and labelers are **internal helpers** (`data/readers/`,
`data/labelers/`), reused but not a plugin contract. Module-top imports stay free
of heavy deps; `cuvis` / `tifffile` / `pycocotools` / `scikit-image` load lazily
on first use (`data/_extras.py`).

## Development

```bash
uv sync --extra dev
uv run pytest tests/ -v
uv run ruff check cuvis_ai_dataloader/ tests/
uv run ruff format cuvis_ai_dataloader/ tests/
uv run mypy cuvis_ai_dataloader/
```

### Git hooks

Enable the repo's hooks once per clone:

```bash
git config core.hooksPath .githooks
```

- **pre-commit**: `ruff format` + `ruff check --fix` on staged Python, then re-stages.
- **pre-push**: `ruff format --check`, `ruff check`, docstring coverage
  (`uvx interrogate`, ≥95%, configured in `[tool.interrogate]`), and
  `pytest -m "not slow and not gpu"`.

Skip a hook for one command with `--no-verify`.

## Contributing

Contributions are welcome. Please:
1. Ensure tests pass
2. Run ruff format and ruff check
3. Keep type hints and update docs as needed

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
