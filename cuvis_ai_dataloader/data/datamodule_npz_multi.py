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

Set ``crop_size=(h, w)`` to crop each TRAIN sample to a foreground-biased patch inside the dataset
(``__getitem__``), shipping ~patch-sized samples instead of whole frames — a large I/O win when the
model trains on crops. ``crop_fg_percent`` / ``crop_fg_labels`` tune the oversampling, and
``crop_pad_mode`` (``"constant"`` = 0, the default, or ``"reflect"``) fills the out-of-frame region
when a foreground-centered window sits against an edge. Off by default (whole frames, unchanged);
val/test/predict always see whole frames for tiled inference.

.. note::
   The crop is applied in the dataset, i.e. **upstream of the consuming pipeline**. Any downstream
   transform that *fits statistics* over the training data — e.g. a z-score normalizer with
   dataset-level running stats — will therefore fit them on the **cropped patches, not the full
   frames**, and eval (full frames) is then normalized with crop-fit stats. With global z-score
   this only shifts the fitted mean/std (arguably more representative of what the model trains on),
   but if full-frame statistics matter, fit them before enabling ``crop_size``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cuvis_ai_core.data.datamodule import BaseCuvisAIDataModule
from cuvis_ai_dataloader.data._crop import PAD_MODES, crop_with_pad, fg_crop_window

from ._extras import accepts_data_config
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

    def _read(self, idx: int):
        """Read one frame's arrays; the cube is kept in its STORED dtype (NOT expanded to f32).

        Returns ``(rec, cube, wavelengths, mask, class_mask)``. Keeping the cube un-expanded lets a
        caller crop first and cast only the patch to float32 (see ``_CropDataset``), skipping the
        full-frame f32 expansion. The whole-frame path (``__getitem__``) expands the cube itself.
        """
        rec = self._rows[idx]
        with np.load(rec["materialized_path"]) as z:
            cube = np.asarray(z["cube"])  # stored dtype (e.g. float16); NOT expanded to f32 here
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
        return rec, cube, wavelengths, mask, class_mask

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | int]:
        rec, cube, wavelengths, mask, class_mask = self._read(idx)
        return {
            "cube": np.asarray(cube, dtype=np.float32),  # whole-frame path: expand the full cube
            "mask": mask,
            "class_mask": class_mask,
            "wavelengths": wavelengths,
            "mesu_index": int(rec["index"]),
            "frame_id": int(rec["frame_id"]),
        }


class _CropDataset(Dataset):
    """Wrap a frame dataset to return a foreground-biased crop per access (train split only).

    Each ``__getitem__`` crops the underlying frame's ``cube`` / ``mask`` / ``class_mask`` to
    ``size`` with a fresh window (see :func:`fg_crop_window`), so with ``samples_per_frame=N`` the
    N visits to a frame yield N *independent* patches. A foreground-centered window may extend past
    the border; :func:`crop_with_pad` fills the out-of-frame region with ``pad_mode``. The cube is
    cropped in its stored dtype and only the patch is cast to float32 (the whole-frame f32 expansion
    is skipped), and only that patch crosses the collation boundary. The RNG is seeded from
    ``torch.initial_seed()`` (distinct per DataLoader worker) so workers don't draw correlated crops.

    NOTE: this reads the whole frame from disk (npz members can't be partially read); it saves the
    f32 expansion + the transfer, not the read. Removing the redundant per-``samples_per_frame``
    reads needs a decoded-frame cache or mmap-able raw ``.npy`` (see ALL-5905 follow-ups).
    """

    def __init__(
        self,
        base: Dataset,
        size: tuple[int, int],
        fg_percent: float,
        fg_labels: list[int] | None,
        pad_mode: str,
    ) -> None:
        self._base = base
        self._size = size
        self._fg_percent = fg_percent
        self._fg_labels = fg_labels
        self._pad_mode = pad_mode
        self._rng: np.random.Generator | None = None
        # cu3s parity: forward the wavelength axis consumers read off the dataset.
        self.wavelengths_nm = getattr(base, "wavelengths_nm", np.array([], dtype=np.int32))
        self.num_channels = getattr(base, "num_channels", 0)

    @property
    def rows(self) -> list[dict]:
        """Per-frame rows of the wrapped dataset (unchanged by cropping)."""
        return self._base.rows

    def __len__(self) -> int:
        return len(self._base)

    def _generator(self) -> np.random.Generator:
        """Lazily build a per-worker RNG (distinct seed per DataLoader worker)."""
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed())
        return self._rng

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | int]:
        # Read the frame with the cube in its STORED dtype (e.g. float16), pick the window from the
        # mask, then crop and cast ONLY the patch to float32. This skips the whole-frame f32
        # expansion (~30x less to allocate/convert per sample) and ships only the patch downstream.
        rec, cube, wavelengths, mask, class_mask = self._base._read(idx)
        top, left = fg_crop_window(
            mask,
            self._size,
            fg_percent=self._fg_percent,
            fg_labels=self._fg_labels,
            rng=self._generator(),
        )
        # A foreground-centered window may poke past the border; crop_with_pad slices the in-frame
        # overlap and fills the rest with pad_mode. Cast only the (small) cube patch to float32.
        return {
            "cube": crop_with_pad(cube, top, left, self._size, self._pad_mode).astype(np.float32),
            "mask": crop_with_pad(mask, top, left, self._size, self._pad_mode),
            "class_mask": crop_with_pad(class_mask, top, left, self._size, self._pad_mode),
            "wavelengths": wavelengths,
            "mesu_index": int(rec["index"]),
            "frame_id": int(rec["frame_id"]),
        }


class MultiNpzDataModule(BaseCuvisAIDataModule):
    """Multi-file NPZ DataModule driven by a core ``splits.json`` over a ``universe_csv`` (universe.csv)."""

    DATA_MODULE_NAME: ClassVar[str] = "npz_multi"

    @accepts_data_config
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
        crop_size: tuple[int, int] | None = None,
        crop_fg_percent: float = 0.33,
        crop_fg_labels: list[int] | None = None,
        crop_pad_mode: str = "constant",
    ) -> None:
        super().__init__(
            splits=splits,
            batch_size=batch_size,
            num_workers=num_workers,
            samples_per_frame=samples_per_frame,  # multiplicity handled by the base
        )
        self._pin_memory = bool(pin_memory)
        self._persistent_workers = bool(persistent_workers)
        self._worker_multiprocessing_context = worker_multiprocessing_context

        # Optional foreground-biased crop applied inside the dataset on the TRAIN split only
        # (see train_dataloader). Default (crop_size=None) ships whole frames, unchanged.
        if crop_size is not None:
            try:
                crop_size = tuple(int(x) for x in crop_size)
            except TypeError:
                raise ValueError(f"crop_size must be an (h, w) pair, got {crop_size!r}") from None
            if len(crop_size) != 2 or any(s <= 0 for s in crop_size):
                raise ValueError(f"crop_size must be a pair of positive ints, got {crop_size!r}")
        if not 0.0 <= float(crop_fg_percent) <= 1.0:
            raise ValueError(f"crop_fg_percent must be in [0, 1], got {crop_fg_percent!r}")
        if crop_pad_mode not in PAD_MODES:
            raise ValueError(f"crop_pad_mode must be one of {PAD_MODES}, got {crop_pad_mode!r}")
        self._crop_size = crop_size
        self._crop_fg_percent = float(crop_fg_percent)
        self._crop_fg_labels = None if crop_fg_labels is None else [int(x) for x in crop_fg_labels]
        self._crop_pad_mode = crop_pad_mode

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

    def supported_attrs(self) -> frozenset[str]:
        """NPZ frames carry no COCO category map, so no metadata attrs can be supplied.

        Returning an empty set lets constraint evaluation treat an anomaly-based constraint
        as ``unavailable`` (soft-skip / warn) rather than forcing ``enumerate`` to raise.
        A ``tags`` / ``categories`` *selector* still hard-raises in ``enumerate`` (that is a
        genuine authoring error), but an opportunistic constraint attr is skipped cleanly.
        """
        return frozenset()

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

    def train_dataloader(self) -> DataLoader:
        """Train loader; when ``crop_size`` is set, ship foreground-biased patches (train only).

        Wraps the train dataset in :class:`_CropDataset` *before* the base applies
        ``samples_per_frame`` multiplicity, so ``samples_per_frame=N`` yields N independent patches
        per frame. Val/test/predict loaders are inherited unchanged (whole frames), so tiled
        full-frame evaluation is unaffected. ``crop_size=None`` (default) is a plain passthrough.
        """
        if self._crop_size is None or self._train_ds is None:
            return super().train_dataloader()
        base_train = self._train_ds
        self._train_ds = _CropDataset(
            base_train,
            self._crop_size,
            self._crop_fg_percent,
            self._crop_fg_labels,
            self._crop_pad_mode,
        )
        try:
            return super().train_dataloader()
        finally:
            self._train_ds = base_train

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
