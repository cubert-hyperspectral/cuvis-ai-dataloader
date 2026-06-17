# cuvis-ai-dataloader

Pluggable hyperspectral **DataModules** for the [Cuvis.AI](https://github.com/cubert-hyperspectral/cuvis-ai)
ecosystem. A DataModule is the unit the framework uses for both training and
inference: it bundles the data, the labels, the splits, and the
`train` / `val` / `test` / `predict` dataloaders.

This single plugin ships the concrete loaders, with per-format heavy deps gated
behind optional extras. The `cuvis` SDK lives **only** here, behind `[cu3s]`; no
other Cuvis.AI repo pins it.

| Module (`data_module_name`) | Reads | Labels | Extra |
|---|---|---|---|
| `cu3s` | `.cu3s` via `cuvis` | COCO JSON | `[cu3s, coco]` |
| `tiff_paired` | `*.tif` / `*.tiff` via `tifffile` | paired PNG | `[tiff]` |

## Install

```bash
# cu3s + COCO
uv pip install 'cuvis-ai-dataloader[cu3s,coco]'
# TIFF + paired PNG
uv pip install 'cuvis-ai-dataloader[tiff]'
# everything, for development
uv sync --extra dev
```

The `cu3s` extra carries the Windows `cuvis-il<3.5.3` pin (the last build with a
`win_amd64` wheel).

## Cuvis SDK (system install, required for `cu3s`)

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

## Use

**Inference** (`restore-pipeline`) selects a module + its params on the CLI:

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
  splits: { train_ids: [0, 2, 3], val_ids: [1, 5], test_ids: [1, 5] }
  batch_size: 4
  params:
    cu3s_file_path: X.cu3s
    annotation_json_path: Y.json
    processing_mode: Reflectance
```

**In-process / notebooks**: construct the DataModule directly and run it through
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

Concrete DataModules subclass `cuvis_ai_core.data.datamodule.BaseHyperspectralDataModule`
and implement only `validate_params` plus `build_dataset` (id-list splits) or
`build_stage_dataset` (module-owned splits). Per-format cube readers and labelers
are **internal helpers** (`data/readers/`, `data/labelers/`), reused but not a
plugin contract. Module-top imports stay free of heavy deps; `cuvis` / `tifffile`
/ `pycocotools` / `scikit-image` load lazily on first use (`data/_extras.py`).
