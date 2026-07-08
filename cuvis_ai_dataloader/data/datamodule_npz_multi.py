"""npz_multi DataModule: one-frame-per-file compressed NPZ with a CSV universe.

``DATA_MODULE_NAME = "npz_multi"`` (no extras: numpy/torch are core deps). One ``.npz`` per
frame, an externally-supplied ``splits.csv`` whose rows are ``(split, npz_path, image_id)``.

Each ``.npz`` carries:

* ``cube``: ``[H, W, C]`` float32
* ``wavelengths``: ``[C]`` (cast to int32 for node compatibility)
* ``mask`` (optional): ``[H, W]`` int32 ground-truth (zeros are emitted when absent)

Split model: **module-owned only** (``DataConfig.splits is None``). Each Lightning stage maps
to the CSV ``split`` column via ``build_stage_dataset``. The selector-driven path is not
implemented; wire it through the base ``enumerate`` / ``build_dataset_from_refs`` contract if a
trainrun ever needs it.

Unlike the cu3s modules, this module honors the DataLoader options ``pin_memory`` /
``persistent_workers`` / ``worker_multiprocessing_context``: NPZ reads are pure-CPU numpy loads
where those measurably help throughput.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from cuvis_ai_core.data.datamodule import BaseCuvisAIDataModule

_REQUIRED_COLUMNS = ("split", "npz_path", "image_id")


class _MultiNpzDataset(Dataset):
    """Holds the rows for one subset; reads each frame's cube + baked mask from its ``.npz``."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        # Expose the wavelength axis (cu3s parity: consumers read ``dm.<split>_ds.wavelengths_nm``).
        if self._rows:
            with np.load(self._rows[0]["npz_path"]) as z:
                wl = np.asarray(z["wavelengths"]).ravel()
            self.wavelengths_nm = wl.astype(np.int32, copy=False)
            self.num_channels = int(self.wavelengths_nm.shape[0])
        else:
            self.wavelengths_nm = np.array([], dtype=np.int32)
            self.num_channels = 0

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | int]:
        rec = self._rows[idx]
        with np.load(rec["npz_path"]) as z:
            cube = np.asarray(z["cube"], dtype=np.float32)
            wavelengths = np.asarray(z["wavelengths"]).ravel().astype(np.int32, copy=False)
            mask = (
                np.asarray(z["mask"], dtype=np.int32)
                if "mask" in z.files
                else np.zeros((cube.shape[0], cube.shape[1]), dtype=np.int32)
            )
        return {
            "cube": cube,
            "mask": mask,
            "wavelengths": wavelengths,
            "mesu_index": int(rec["image_id"]),
            "frame_id": int(rec["frame_id"]),
        }


class MultiNpzDataModule(BaseCuvisAIDataModule):
    """Multi-file NPZ DataModule driven by an external ``splits.csv``."""

    DATA_MODULE_NAME: ClassVar[str] = "npz_multi"

    def __init__(
        self,
        *,
        splits=None,
        batch_size: int = 1,
        num_workers: int = 0,
        splits_csv: str | None = None,
        split: str | None = None,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        worker_multiprocessing_context: str = "spawn",
        samples_per_frame: int = 1,
        params: dict | None = None,
        # Carried by the nested `cls(**cfg.data)` shape; accepted and ignored (the class
        # identity fixes the module). Any other unknown kwarg raises.
        data_module: str | None = None,
    ) -> None:
        if params:
            splits_csv = splits_csv or params.get("splits_csv")
            split = split if split is not None else params.get("split")
            pin_memory = params.get("pin_memory", pin_memory)
            persistent_workers = params.get("persistent_workers", persistent_workers)
            worker_multiprocessing_context = params.get(
                "worker_multiprocessing_context", worker_multiprocessing_context
            )
            samples_per_frame = params.get("samples_per_frame", samples_per_frame)
        super().__init__(splits=splits, batch_size=batch_size, num_workers=num_workers)
        if not splits_csv:
            raise ValueError("npz_multi requires 'splits_csv'.")
        if int(samples_per_frame) < 1:
            raise ValueError(f"samples_per_frame must be >= 1, got {samples_per_frame}")
        self._splits_csv = Path(splits_csv).resolve()
        self._csv_dir = self._splits_csv.parent
        self._predict_split = split  # which CSV split predict_dataloader iterates (module-owned)
        self._pin_memory = bool(pin_memory)
        self._persistent_workers = bool(persistent_workers)
        self._worker_multiprocessing_context = worker_multiprocessing_context
        self._samples_per_frame = int(samples_per_frame)
        self._rows = self._parse_csv(self._splits_csv)

    @staticmethod
    def validate_params(params: dict[str, Any]) -> None:
        """Validate that a ``splits_csv`` path is given, ends in ``.csv``, and exists."""
        csv_path = params.get("splits_csv")
        if not csv_path:
            raise ValueError("npz_multi requires 'splits_csv' in params.")
        if not str(csv_path).endswith(".csv"):
            raise ValueError(f"splits_csv must end with .csv: {csv_path!r}")
        if not Path(csv_path).is_file():
            raise ValueError(f"splits_csv does not exist: {csv_path}")

    def build_stage_dataset(self, stage: str) -> Dataset:
        """Module-owned path: map the Lightning stage to the matching CSV ``split`` rows."""
        # DataConfig.splits is None: predict honors --data-arg split (default test).
        split = (self._predict_split or "test") if stage == "predict" else stage
        rows = [r for r in self._rows if split == "all" or r["split"] == split]
        if stage == "train" and self._samples_per_frame > 1:
            # Index-level duplication: each occurrence is one independent sample per
            # epoch (downstream per-sample transforms, e.g. random crops, draw fresh
            # for every occurrence), and the shuffled loader interleaves duplicates
            # across the epoch. Val/test/predict are never expanded.
            rows = [r for r in rows for _ in range(self._samples_per_frame)]
            ds = _MultiNpzDataset(rows)
            logger.info(
                "npz_multi {} dataset: {} samples ({} frames x {})",
                stage,
                len(ds),
                len(ds) // self._samples_per_frame,
                self._samples_per_frame,
            )
            return ds
        ds = _MultiNpzDataset(rows)
        logger.info("npz_multi {} dataset: {} frames", stage, len(ds))
        return ds

    def _loader(self, dataset: Dataset | None, *, shuffle: bool, name: str) -> DataLoader:
        """Like the base loader, but honor pin_memory / persistent_workers / mp-context."""
        if dataset is None:
            raise RuntimeError(
                f"{type(self).__name__}: {name} dataset is not built; "
                f"call setup(stage={name!r}) (or setup()) first."
            )
        kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self._pin_memory,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self._persistent_workers
            if self._worker_multiprocessing_context:
                kwargs["multiprocessing_context"] = self._worker_multiprocessing_context
        return DataLoader(dataset, **kwargs)

    def _parse_csv(self, csv_path: Path) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = [c for c in _REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(
                    f"{csv_path}: missing required column(s) {missing}. "
                    f"Required: {list(_REQUIRED_COLUMNS)}. Extra columns are allowed and ignored."
                )
            for frame_counter, row in enumerate(reader):
                out.append(
                    {
                        "frame_id": frame_counter,  # stable global row identity
                        "split": row["split"],
                        "npz_path": str(self._resolve(row["npz_path"])),
                        "image_id": int(str(row["image_id"]).strip()),
                    }
                )
        if not out:
            raise ValueError(f"{csv_path}: no rows.")
        return out

    def _resolve(self, raw: str) -> Path:
        """Resolve relative paths against the CSV's parent dir; pass absolutes through."""
        p = Path(raw)
        return p if p.is_absolute() else (self._csv_dir / p).resolve()
