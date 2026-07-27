"""npz_multi DataModule: one-frame-per-file compressed NPZ, selector-driven splits.

``DATA_MODULE_NAME = "npz_multi"`` (no extras: numpy/torch are core deps). One ``.npz`` per
frame, selected by a core ``splits.json`` over a ``universe_csv`` (the shared ``universe.csv``):

* ``universe_csv`` (``source, index, materialized_path`` + optional ``annotation, format,
  group``): the sample universe, one row per frame. ``source`` is the opaque logical identity (a
  cu3s-derived npz carries its posix cu3s path); ``index`` is the read position (== COCO
  image_id); ``materialized_path`` is the ``.npz`` for that frame, relative to the CSV. npz has no
  physical frame at ``source``, so ``materialized_path`` is required (unlike cu3s, which defaults
  it to ``source``). A ``split`` column is rejected here: npz is selector-driven.
* ``splits.json`` (a core ``DataSplitConfig`` passed as ``DataConfig.splits``): ``file_indices``
  selectors pick each split's subset by ``(source, index)``. Because ``source`` is the cu3s
  identity, one ``splits.json`` resolves against both the raw cu3s data (``cu3s_multi``) and the
  converted npz, and round-trips through the CuvisNEXT split designer.

Each ``.npz`` carries:

* ``cube``: ``[H, W, C]`` float32
* ``wavelengths``: ``[C]`` (cast to int32 for node compatibility)
* ``mask`` (optional): ``[H, W]`` int32 binary ground-truth (zeros are emitted when absent)
* ``class_mask`` (optional): ``[H, W]`` uint8 per-pixel COCO category id (0 = background);
  zeros are emitted when absent. Consumed by per-class evaluation (e.g. per-class AUROC).

Unlike the cu3s modules, this module honors the DataLoader options ``pin_memory`` /
``persistent_workers`` / ``worker_multiprocessing_context``: NPZ reads are pure-CPU numpy loads
where those measurably help throughput.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from torch.utils.data import DataLoader, Dataset

from cuvis_ai_core.data.datamodule import BaseCuvisAIDataModule

from ._universe import parse_universe, validate_universe_csv_param

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cuvis_ai_schemas.training.data import SampleRef


class _MultiNpzDataset(Dataset):
    """Holds the rows for one subset; reads each frame's cube + baked mask from its ``.npz``."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        # Expose the wavelength axis (cu3s parity: consumers read ``dm.<split>_ds.wavelengths_nm``).
        if self._rows:
            with np.load(self._rows[0]["materialized_path"]) as z:
                wl = np.asarray(z["wavelengths"]).ravel()
            self.wavelengths_nm = wl.astype(np.int32, copy=False)
            self.num_channels = int(self.wavelengths_nm.shape[0])
        else:
            self.wavelengths_nm = np.array([], dtype=np.int32)
            self.num_channels = 0

    @property
    def rows(self) -> list[dict]:
        """Public read-only view of the per-frame rows (``materialized_path, index, frame_id``)."""
        return self._rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | int]:
        rec = self._rows[idx]
        with np.load(rec["materialized_path"]) as z:
            cube = np.asarray(z["cube"], dtype=np.float32)
            wavelengths = np.asarray(z["wavelengths"]).ravel().astype(np.int32, copy=False)
            mask = (
                np.asarray(z["mask"], dtype=np.int32)
                if "mask" in z.files
                else np.zeros((cube.shape[0], cube.shape[1]), dtype=np.int32)
            )
            # Optional multi-class GT (pixel = COCO category id, 0 = background). Emitted for
            # every frame (zeros when absent) so batch keys stay uniform for default_collate;
            # consumed by per-class evaluation (e.g. per-class AUROC).
            class_mask = (
                np.asarray(z["class_mask"], dtype=np.uint8)
                if "class_mask" in z.files
                else np.zeros((cube.shape[0], cube.shape[1]), dtype=np.uint8)
            )
        return {
            "cube": cube,
            "mask": mask,
            "class_mask": class_mask,
            "wavelengths": wavelengths,
            "mesu_index": int(rec["index"]),
            "frame_id": int(rec["frame_id"]),
        }


class MultiNpzDataModule(BaseCuvisAIDataModule):
    """Multi-file NPZ DataModule driven by a core ``splits.json`` over a ``universe_csv`` (universe.csv)."""

    DATA_MODULE_NAME: ClassVar[str] = "npz_multi"

    def __init__(
        self,
        *,
        splits=None,
        batch_size: int = 1,
        num_workers: int = 0,
        universe_csv: str | None = None,
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
            universe_csv = universe_csv or params.get("universe_csv")
            pin_memory = params.get("pin_memory", pin_memory)
            persistent_workers = params.get("persistent_workers", persistent_workers)
            worker_multiprocessing_context = params.get(
                "worker_multiprocessing_context", worker_multiprocessing_context
            )
            samples_per_frame = params.get("samples_per_frame", samples_per_frame)
        super().__init__(
            splits=splits,
            batch_size=batch_size,
            num_workers=num_workers,
            samples_per_frame=samples_per_frame,  # multiplicity handled by the base
        )
        self._pin_memory = bool(pin_memory)
        self._persistent_workers = bool(persistent_workers)
        self._worker_multiprocessing_context = worker_multiprocessing_context

        if not universe_csv:
            raise ValueError("npz_multi requires 'universe_csv' (the universe.csv lookup).")
        if self.splits is None:
            raise ValueError(
                "npz_multi requires a 'splits' selector config (a DataSplitConfig or splits.json)."
            )
        self._universe_csv = Path(universe_csv).resolve()
        self._universe = parse_universe(
            self._universe_csv,
            require_materialized_path=True,  # npz has no frame at `source`; the .npz is the file
            accept_split=False,  # npz is selector-driven; a split column is rejected
            unique_materialized_path=True,  # one .npz per row
            allow_index_ranges=False,  # one row per frame; no ranges
        )

    @staticmethod
    def validate_params(params: dict[str, Any]) -> None:
        """Validate that ``universe_csv`` is given, ends in ``.csv``, and exists."""
        validate_universe_csv_param(params, "npz_multi")

    # -- selector path ---------------------------------------------------------
    def enumerate(self, required_attrs: frozenset[str] = frozenset()) -> list[SampleRef]:
        """List the universe rows as the attributed sample universe (one ref per row).

        Identity is ``(source, index)`` with ``index`` the read position (the framework enforces
        read_index == COCO image_id for merged sessions). NPZ frames carry no COCO name->id
        map, so a ``tags`` / ``category_ids`` selector is unsupported here.
        """
        from cuvis_ai_schemas.training.data import SampleRef

        if required_attrs & {"tags", "category_ids"}:
            raise NotImplementedError(
                "npz_multi has no category name->id map; tag / categories selectors are "
                "unsupported. Use file_indices / files / stems / dir_indices selectors."
            )
        assert self._universe is not None
        refs = [
            SampleRef(
                source=rec["source"],
                index=int(rec["index"]),
                label_id=int(rec["index"]),
                stem=Path(rec["source"]).stem,
                group=rec.get("group") or rec["source"],
            )
            for rec in self._universe
        ]
        refs.sort(key=lambda r: (r.source, -1 if r.index is None else r.index))
        return refs

    def build_dataset_from_refs(self, refs: list[SampleRef]) -> Dataset:
        """Build the dataset for a resolved subset, mapping ``(source, index)`` -> ``.npz``."""
        assert self._universe is not None
        by_identity = {
            (rec["source"], int(rec["index"])): rec["materialized_path"] for rec in self._universe
        }
        rows = []
        for i, ref in enumerate(refs):
            key = (ref.source, int(ref.index if ref.index is not None else 0))
            path = by_identity.get(key)
            if path is None:
                raise ValueError(f"ref {key} has no matching npz in the universe")
            rows.append(
                {
                    "frame_id": i,
                    "materialized_path": path,
                    "index": int(ref.label_id if ref.label_id is not None else key[1]),
                }
            )
        return _MultiNpzDataset(rows)

    def category_name_to_id(self) -> dict[str, int] | None:
        """NPZ frames bake masks in; no COCO category name->id map is available."""
        return None

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
