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
- **CSV `split` column** is specific to `cu3s_multi`, which is driven by a
  required CSV manifest (`split, cu3s_path, annotation_json, image_id`). With no
  `DataConfig.splits`, each stage maps straight to that `split` column; otherwise
  the CSV rows become the selector universe (and `resolve-splits --from-csv`
  turns the CSV into a `splits.json`).

## Installation

```bash
uv pip install "cuvis-ai-dataloader[cu3s,coco]"   # cu3s + COCO
uv pip install "cuvis-ai-dataloader[tiff]"         # TIFF + paired PNG
uv pip install "cuvis-ai-dataloader[all]"          # every format
```

Extras:
- `cu3s`: `.cu3s` / `.cu3` reading via the `cuvis` SDK binding
- `coco`: COCO-JSON mask labels (`pycocotools`, `scikit-image`)
- `tiff`: TIFF cube reading (`tifffile`)
- `all`: All formats
- `dev`: Development dependencies

The `cu3s` extra carries the Windows `cuvis-il<3.5.3` pin (the last build with a
`win_amd64` wheel).

### Cuvis SDK (system install, required for `cu3s`)

The `[cu3s]` extra installs the `cuvis` Python package, but that is only a
binding: the **C++ Cuvis SDK** must also be installed system-wide, or any
`.cu3s` / `.cu3` read fails at runtime.

> **macOS not supported.** Cuvis SDK ships for Windows and Linux only. On macOS,
> `.cu3s` / `.cu3` reads fail at runtime; the `tiff_paired` module and any
> numpy / video input still work.

Obtain a build matching the `cuvis>=3.5.0` pin for your OS from the
[Cuvis SDK download page](https://cubert-hyperspectral.github.io/cuvis.sdk/installation/),
then verify:

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

## Contributing

Contributions are welcome. Please:
1. Ensure tests pass
2. Run ruff format and ruff check
3. Keep type hints and update docs as needed

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
