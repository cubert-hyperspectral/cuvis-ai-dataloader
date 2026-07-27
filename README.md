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
- `dev`: Development dependencies

The `cu3s` extra carries the Windows `cuvis-il<3.5.3` pin (the last build with a
`win_amd64` wheel).

### Cuvis SDK (system install, required for `cu3s`)

The `[cu3s]` extra installs the `cuvis` **binding** (with the Windows `cuvis-il<3.5.3` pin noted
above), but that binding needs the system-wide **C++ Cuvis SDK** too, or any `.cu3s` read fails at
runtime. See the
[Cuvis.AI installation guide](https://docs.cuvis.ai/latest/get-started/installation/) for OS
support (Windows / Linux; not macOS), the SDK download, and verification. Quick check once
installed:

```bash
uv run python -c "import cuvis; print(cuvis.__version__)"
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
