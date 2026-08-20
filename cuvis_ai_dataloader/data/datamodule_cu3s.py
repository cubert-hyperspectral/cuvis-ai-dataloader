"""cu3s DataModule: ``.cu3s`` cubes via the cuvis SDK + optional COCO masks.

``DATA_MODULE_NAME = "cu3s"`` (manifest extras ``[cu3s, coco]``). The split/dataloader
plumbing lives in ``BaseCuvisAIDataModule``; cube reading is the internal
``Cu3sCubeReader`` and COCO labeling the internal ``CocoLabeler``.

Selector path: ``enumerate()`` lists the attributed measurement universe (single-file mode:
one ref per measurement; folder mode: one ref per file at measurement 0 by default, or one
ref per measurement with ``frames="measurements"``), and ``build_dataset_from_refs`` reads
exactly the resolved subset.

Folder mode with ``frames="measurements"`` is the contract for externally authored splits
(e.g. the CuvisNEXT split designer): sources are canonical absolute paths
(``Path.resolve().as_posix()``), one sample per measurement ``0..N-1``, sibling
``<stem>.json`` COCO attached; see ``README.md`` ("GUI-authored splits over a cu3s folder").
The module does not own split semantics: training stages without ``DataConfig.splits``
are refused (see ``setup``), so statistical initialization can never silently ingest the
whole universe.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from torch.utils.data import DataLoader, Dataset

from cuvis_ai_core.data.datamodule import BaseCuvisAIDataModule, DataStage
from cuvis_ai_schemas.training.data import DataSplitConfig, SampleRef

from ._extras import accepts_data_config, parse_bool, parse_int_list
from .readers.cu3s_pool import Cu3sReaderCache, SourceCoherentBatchSampler
from .readers.cu3s_reader import Cu3sCubeReader, total_measurements_of


def _sibling_json(annotation_json_path, cu3s_file_path) -> str | None:
    """Resolve the annotation path, falling back to a sibling ``<stem>.json``."""
    if annotation_json_path:
        return str(annotation_json_path)
    if cu3s_file_path:
        sib = Path(cu3s_file_path).with_suffix(".json")
        if sib.exists():
            return str(sib)
    return None


class _Cu3sRefDataset(Dataset):
    """Torch Dataset over resolved ``SampleRef``s (single-file or folder).

    Readers and labelers are cached per source, so single-file mode reuses one SDK session
    and folder mode opens a session only for the files actually selected (lazily, in
    ``__getitem__``, never at setup).

    The reader cache is a bounded LRU with close-on-evict: every open session holds
    native SDK resources including GPU processing pools, and past a handful of
    concurrently open Reflectance sessions the SDK's CUDA allocator fails hard
    ("illegal memory access", killing the process) — especially when torch shares
    the GPU during training. Bounding + closing keeps the total footprint flat no
    matter how many sources a shuffled multi-file epoch touches.
    """

    def __init__(
        self,
        refs: list[SampleRef],
        processing_mode: str,
        *,
        max_open_sessions: int = 4,
        read_threads: int = 0,
    ) -> None:
        self._refs = refs
        self._processing_mode = processing_mode
        self._cache = Cu3sReaderCache(
            processing_mode=processing_mode,
            max_open_sessions=max_open_sessions,
            read_threads=read_threads,
            sources=len({ref.source for ref in refs}) or 1,
        )
        self._labelers: dict[str, Any] = {}

    def __getstate__(self) -> dict:
        # Drop cached labelers before pickling to DataLoader workers; the reader cache drops
        # its own native handles, which do not pickle.
        state = self.__dict__.copy()
        state["_labelers"] = {}
        return state

    def close(self) -> None:
        """Release every cached SDK session (safe to call repeatedly)."""
        self._cache.close()

    def __del__(self) -> None:  # best-effort: replaced datasets free their sessions
        try:
            self.close()
        except Exception:
            pass

    def _labeler_for(self, annotation: str):
        if annotation not in self._labelers:
            from .labelers.coco_labeler import CocoLabeler

            self._labelers[annotation] = CocoLabeler(annotation)
        return self._labelers[annotation]

    def __len__(self) -> int:
        return len(self._refs)

    @property
    def sample_sources(self) -> list[str]:
        """The cu3s each sample reads from, positionally, for source-coherent batching."""
        return [ref.source for ref in self._refs]

    @property
    def wavelengths_nm(self) -> np.ndarray:
        """Per-channel wavelengths (nm, int32) read from the first sample's source.

        Cubes in one cu3s share a wavelength axis, so consumers can read this once
        without iterating the dataset.
        """
        if not self._refs:
            raise ValueError("dataset is empty; no wavelengths available")
        return self._cache.get(self._refs[0].source).wavelengths_nm

    @property
    def wavelengths(self) -> np.ndarray:
        """Alias of :attr:`wavelengths_nm` (the accessor the former dataset exposed)."""
        return self.wavelengths_nm

    def _decorate(self, ref: SampleRef, read_pos: int, item: dict) -> dict:
        """Attach the ref's identity, and its COCO labels when it has an annotation.

        Labeling stays on the calling thread: CocoLabeler holds the GIL for its whole
        duration and keeps mutable index state, so a pool would add risk and no speed.
        """
        item["stem"] = ref.stem
        # COCO image id (defaults to the read position); kept distinct from read_index.
        image_id = ref.label_id if ref.label_id is not None else read_pos
        item["read_index"] = int(read_pos)
        item["mesu_index"] = int(image_id)
        if ref.annotation:
            item.update(self._labeler_for(ref.annotation).load_for(int(image_id), item))
        return item

    def __getitem__(self, idx: int) -> dict:
        ref = self._refs[idx]
        read_pos = ref.index if ref.index is not None else 0
        return self._decorate(ref, read_pos, self._cache.get(ref.source).read(read_pos))

    def __getitems__(self, indices: list[int]) -> list[dict]:
        """Fetch a whole batch at once, so the reader cache can overlap its reads.

        torch calls this in place of per-index ``__getitem__`` when a dataset defines it. It is
        the only point at which several indices are known together, so it is the only place
        the readers' threads can be used; parallelism is therefore bounded by ``batch_size``.
        """
        refs = [self._refs[i] for i in indices]
        positions = [(ref.source, ref.index if ref.index is not None else 0) for ref in refs]
        return [
            self._decorate(ref, position, item)
            for ref, (_, position), item in zip(refs, positions, self._cache.read_many(positions))
        ]


class Cu3sDataModule(BaseCuvisAIDataModule):
    """cu3s + COCO DataModule on the shared base."""

    DATA_MODULE_NAME: ClassVar[str] = "cu3s"

    @accepts_data_config
    def __init__(
        self,
        *,
        splits: DataSplitConfig | None = None,
        batch_size: int = 1,
        num_workers: int = 0,
        cu3s_file_path: str | None = None,
        annotation_json_path: str | None = None,
        processing_mode: str = "Reflectance",
        measurement_indices: Any = None,
        # Folder source: a data_dir (no single file) lists *.cu3s into one ordered universe;
        # selectors then index into it.
        data_dir: str | None = None,
        # Folder-mode granularity: "file" = one sample per file at measurement 0 (legacy
        # default); "measurements" = one sample per measurement with canonical absolute
        # sources (the GUI-authored-splits contract). Single-file mode is always
        # per-measurement and ignores this.
        frames: str = "file",
        recursive: Any = False,
        samples_per_frame: int = 1,
        # Concurrently open SDK sessions per dataset (LRU, close-on-evict). Each open
        # Reflectance session holds SDK GPU processing pools; keep this small when
        # torch shares the GPU.
        max_open_sessions: int = 4,
        # SDK reader threads for the whole dataset, divided across the sessions it holds open.
        # 0 disables. Needs a cuvis binding that releases the GIL; on one that does not, the
        # reader warns and falls back, because extra threads there are a measured loss.
        read_threads: int = 0,
        # Keep each batch inside as few recordings as possible, so a shuffled multi-file
        # epoch stops evicting readers mid-batch. Changes which samples share a batch, and
        # replaces the loader's sampler, so it is off by default and unusable under DDP.
        source_coherent_batches: bool = False,
    ) -> None:
        super().__init__(
            splits=splits,
            batch_size=batch_size,
            num_workers=num_workers,
            samples_per_frame=samples_per_frame,
        )

        self.cu3s_file_path = str(cu3s_file_path) if cu3s_file_path else None
        self.data_dir = Path(data_dir) if (self.cu3s_file_path is None and data_dir) else None
        # Folder mode reads *.cu3s; kept as a list so _list_folder_files stays generic.
        self.cu3s_globs: list[str] | None = ["cu3s"] if self.data_dir is not None else None
        frames = str(frames or "file")
        if frames not in ("file", "measurements"):
            raise ValueError(f"frames must be 'file' or 'measurements', got {frames!r}")
        self.frames = frames
        self.recursive = (
            parse_bool(recursive, key="recursive")
            if isinstance(recursive, str)
            else bool(recursive)
        )
        self.annotation_json_path = _sibling_json(annotation_json_path, self.cu3s_file_path)
        self.processing_mode = processing_mode
        self.measurement_indices = (
            parse_int_list(measurement_indices, key="measurement_indices")
            if isinstance(measurement_indices, str)
            else measurement_indices
        )
        self.max_open_sessions = int(max_open_sessions)
        if self.max_open_sessions < 1:
            raise ValueError(f"max_open_sessions must be >= 1, got {max_open_sessions}")
        self.read_threads = int(read_threads)
        if self.read_threads < 0:
            raise ValueError(f"read_threads must be >= 0, got {read_threads}")
        self.source_coherent_batches = bool(source_coherent_batches)
        # Process workers each build their own sessions and their own ProcessingContext, so
        # combining them multiplies both the handle count and the ~9 s context build. The
        # failure mode is an OOM or a killed CUDA process, not a slowdown, so refuse instead
        # of silently overriding either knob.
        if self.read_threads > 1 and int(num_workers) > 0:
            raise ValueError(
                f"read_threads={read_threads} cannot be combined with num_workers="
                f"{num_workers}; reader threads replace DataLoader worker processes, so set "
                "num_workers=0 to use them."
            )
        self._enum_labelers: dict[str, Any] = {}

    def _loader(self, dataset, *, shuffle: bool, name: str) -> DataLoader:
        """The base loader, or one whose batches stay within a recording when asked for."""
        sources = getattr(getattr(dataset, "_base", dataset), "sample_sources", None)
        if not self.source_coherent_batches or not sources:
            return super()._loader(dataset, shuffle=shuffle, name=name)
        # samples_per_frame wraps the dataset in a repeat whose index i reads base i % len,
        # so repeating the base's source list reproduces that mapping exactly.
        sources = list(sources) * max(1, len(dataset) // len(sources))
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_sampler=SourceCoherentBatchSampler(sources, self.batch_size, shuffle=shuffle),
        )

    @staticmethod
    def validate_params(params: dict[str, Any]) -> None:
        """Validate cu3s params: a file (or folder) source exists and any annotation is JSON."""
        cu3s = params.get("cu3s_file_path")
        data_dir = params.get("data_dir")
        frames = params.get("frames", "file")
        if frames not in ("file", "measurements"):
            raise ValueError(f"frames must be 'file' or 'measurements', got {frames!r}")
        if not cu3s and not data_dir:
            raise ValueError(
                "cu3s requires 'cu3s_file_path', or 'data_dir' (a folder of .cu3s files), "
                "in params."
            )
        if cu3s:
            if not str(cu3s).endswith(".cu3s"):
                raise ValueError(f"cu3s_file_path must end with .cu3s: {cu3s!r}")
            if not os.path.exists(cu3s):
                raise ValueError(f"cu3s_file_path does not exist: {cu3s}")
        else:
            folder = Path(data_dir)
            if not folder.is_dir():
                raise ValueError(f"data_dir does not exist or is not a directory: {data_dir}")
            recursive = params.get("recursive", False)
            recursive = (
                parse_bool(recursive, key="recursive")
                if isinstance(recursive, str)
                else bool(recursive)
            )
            find = folder.rglob if recursive else folder.glob
            if not any(find("*.cu3s")):
                raise ValueError(f"data_dir holds no *.cu3s files: {data_dir}")
        ann = params.get("annotation_json_path")
        if ann:
            if not str(ann).endswith(".json"):
                raise ValueError(f"annotation_json_path must end with .json: {ann!r}")
            if not os.path.exists(ann):
                raise ValueError(f"annotation_json_path does not exist: {ann}")

    # -- split-less training guard ----------------------------------------------
    def setup(self, stage: str | None = None) -> None:
        """Refuse split-less training stages; predict over the whole universe stays valid.

        cu3s does not own split semantics: without ``DataConfig.splits``, ``fit`` /
        ``validate`` / ``test`` would silently iterate the whole configured universe, and
        statistical initialization (e.g. MinMax) would ingest anomalous frames with no
        error. ``setup(None)`` builds only the predict dataset (the whole universe), which
        is the one meaningful split-less stage.
        """
        if self.splits is None and stage != DataStage.PREDICT:
            if stage is not None:
                raise ValueError(
                    f"cu3s does not own split semantics: setup({str(stage)!r}) without "
                    "DataConfig.splits would train/evaluate on the whole universe "
                    "(statistical initialization would silently ingest anomalous frames). "
                    "Provide splits (e.g. a frozen splits.json via splits_path); "
                    "predict over the whole universe stays valid."
                )
            self._predict_ds = self.build_stage_dataset("predict")
            return
        super().setup(stage)

    # -- selector contract -----------------------------------------------------
    def _list_folder_files(self) -> list[Path]:
        """Sorted, de-duplicated list of ``.cu3s`` files in the source folder.

        ``recursive=True`` walks subfolders (``rglob``), e.g. a dataset root holding
        per-day session folders.
        """
        files: list[Path] = []
        find = self.data_dir.rglob if self.recursive else self.data_dir.glob
        for ext in self.cu3s_globs:
            files.extend(find(f"*.{ext.lstrip('.')}"))
        files = sorted(set(files))
        if not files:
            raise FileNotFoundError(f"No {self.cu3s_globs} files in {self.data_dir}")
        return files

    def _enum_labeler_for(self, annotation: str):
        if annotation not in self._enum_labelers:
            from .labelers.coco_labeler import CocoLabeler

            self._enum_labelers[annotation] = CocoLabeler(annotation)
        return self._enum_labelers[annotation]

    def _attrs_for(
        self, annotation: str | None, image_id: int, required: frozenset[str]
    ) -> tuple[list[str], list[int]]:
        """Populate (tags, category_ids) for a ref only when a stage needs them."""
        if not annotation or not (required & {"tags", "category_ids"}):
            return [], []
        labeler = self._enum_labeler_for(annotation)
        cats = labeler.categories_for(image_id)
        tags = (["anomalous"] if cats else ["normal"]) if "tags" in required else []
        return tags, (cats if "category_ids" in required else [])

    def enumerate(self, required_attrs: frozenset[str] = frozenset()) -> list[SampleRef]:
        """List the attributed sample universe (one ref per measurement, or per folder file).

        Folder mode with ``frames="measurements"`` emits one ref per measurement per file
        with a **canonical** absolute source (``Path.resolve().as_posix()``: forward
        slashes, filesystem-true case), so ``SampleRef.uid`` matches what an external
        split author (Qt ``QFileInfo::canonicalFilePath()``) writes into ``splits.json``.
        The count probe opens each session without a processing mode, so enumeration
        never requires references in the file.
        """
        refs: list[SampleRef] = []
        if self.data_dir is not None:
            for path in self._list_folder_files():
                if self.frames == "measurements":
                    source = path.resolve().as_posix()
                    annotation = _sibling_json(None, source)
                    total = total_measurements_of(source)
                    for m in range(total):
                        tags, cats = self._attrs_for(annotation, m, required_attrs)
                        refs.append(
                            SampleRef(
                                source=source,
                                index=m,
                                label_id=m,
                                stem=path.stem,
                                annotation=annotation,
                                tags=tags,
                                category_ids=cats,
                            )
                        )
                    continue
                source = str(path)
                annotation = _sibling_json(None, source)
                tags, cats = self._attrs_for(annotation, 0, required_attrs)
                refs.append(
                    SampleRef(
                        source=source,
                        index=0,
                        label_id=0,
                        stem=path.stem,
                        annotation=annotation,
                        tags=tags,
                        category_ids=cats,
                    )
                )
        else:
            source = self.cu3s_file_path
            indices = self.measurement_indices
            if indices is None:
                reader = Cu3sCubeReader(source, processing_mode=self.processing_mode)
                try:
                    indices = range(reader.total_measurements)
                finally:
                    reader.close()
            annotation = self.annotation_json_path
            stem = Path(source).stem
            for m in indices:
                m = int(m)
                tags, cats = self._attrs_for(annotation, m, required_attrs)
                refs.append(
                    SampleRef(
                        source=source,
                        index=m,
                        label_id=m,
                        stem=stem,
                        annotation=annotation,
                        tags=tags,
                        category_ids=cats,
                    )
                )
        refs.sort(key=lambda r: (r.source, -1 if r.index is None else r.index))
        return refs

    def build_dataset_from_refs(self, refs: list[SampleRef]) -> Dataset:
        """Build the torch Dataset reading exactly the resolved ``SampleRef`` subset."""
        return _Cu3sRefDataset(
            refs,
            self.processing_mode,
            max_open_sessions=self.max_open_sessions,
            read_threads=self.read_threads,
        )

    def category_name_to_id(self) -> dict[str, int] | None:
        """Map COCO category names to ids (from the annotation), or None when unlabeled."""
        annotation = self.annotation_json_path
        if annotation is None and self.data_dir is not None:
            files = self._list_folder_files()
            annotation = _sibling_json(None, str(files[0])) if files else None
        if not annotation:
            return None
        labeler = self._enum_labeler_for(annotation)
        return {name: cid for cid, name in labeler.category_id_to_name.items()}

    def build_stage_dataset(self, stage: str) -> Dataset:
        """Module-owned path (no splits): the whole configured universe.

        Only the predict stage reaches this (``setup`` refuses split-less training
        stages); it serves every measurement (single-file / measurements mode) or every
        file (folder ``frames="file"`` mode).
        """
        return self.build_dataset_from_refs(self.enumerate())
