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

Which recordings folder mode looks at is decided once, by ``_folder_files()``: an explicit
``files`` list if the caller passed one, else the sources the split's selectors name if
every selector names them, else the folder walk. So a folder holding recordings the split
does not use costs nothing, and ``recursive`` governs the walk alone. Two consequences
worth knowing:

* ``SampleRef.source`` keeps the spelling each mode has always emitted -- canonical posix
  for ``frames="measurements"``, the path as given for ``frames="file"`` -- because
  ``resolve-splits`` writes its selectors from these strings, and a relative ``data_dir``
  must keep yielding relative sources.
* an empty ``predict`` stage still means "the whole universe", which now means the
  recordings the split names rather than everything under ``data_dir``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

import numpy as np
from torch.utils.data import DataLoader, Dataset

from cuvis_ai_core.data.datamodule import BaseCuvisAIDataModule, DataStage

# The one path-spelling rule the ecosystem shares: the GUI's split designer, core's
# selector resolver and this module must agree on when two spellings name one file
# (separator style, drive-letter case), or a Windows-authored splits.json selects
# nothing. Private in core 0.17.1 and read from there deliberately rather than copied,
# so a comparison can never drift from the resolver's; it moves to a public
# ``norm_source`` on core's next release and this import follows it.
from cuvis_ai_core.data.selectors import _norm_source
from cuvis_ai_schemas.training.data import DataSplitConfig, SampleRef, SelectorKind

from ._extras import (
    accepts_data_config,
    configure_cuvis_sdk,
    parse_bool,
    parse_int_list,
    parse_str_list,
)
from .readers.cu3s_pool import Cu3sReaderCache, SourceCoherentBatchSampler
from .readers.cu3s_reader import Cu3sCubeReader, count_measurements

#: Selector kinds that name their sources outright. Everything else (``all``,
#: ``dir_indices``, ``stems``, ``glob``, ``tag``, ``categories``) is answered only by
#: the full universe, so a split using one of them keeps the folder walk.
_SOURCE_NAMING_KINDS = frozenset({SelectorKind.FILES, SelectorKind.FILE_INDICES})
_SET_OP_KINDS = frozenset({SelectorKind.UNION, SelectorKind.EXCEPT, SelectorKind.INTERSECT})


class ExplicitSources(NamedTuple):
    """The sources a split names, split by whether they have to exist.

    ``required`` comes from a stage's own selectors: core raises "matched 0 samples"
    when one of those resolves to nothing, so a missing file is a hard error and is
    better reported by name, up front. ``optional`` comes from inside a set operation,
    where core resolves each child without that check on purpose, so
    ``except(files[a], files[gone])`` is a legitimate split that must keep working.
    """

    required: frozenset[str]
    optional: frozenset[str]

    @property
    def named(self) -> frozenset[str]:
        """Every source the split mentions, required or not."""
        return self.required | self.optional


def explicit_sources(splits: DataSplitConfig | None) -> ExplicitSources | None:
    """Which recordings a split names, or ``None`` when it needs the whole universe.

    This is what lets a folder-sourced module open only the recordings a split actually
    uses. It is a pure read of the selector tree: ``file_indices`` contributes its
    ``source``, ``files`` contributes its ``paths``, the set operations contribute their
    children's, and any positional or attribute-driven selector (``dir_indices``,
    ``stems``, ``glob``, ``tag``, ``categories``, ``all``) means the answer can only come
    from enumerating everything, so the walk stays.
    """
    if splits is None:
        return None
    required: set[str] = set()
    optional: set[str] = set()

    def visit(sel: Any, *, into: set[str]) -> bool:
        """Collect one selector's sources; ``False`` means the whole universe is needed."""
        if sel.kind in _SOURCE_NAMING_KINDS:
            if sel.kind == SelectorKind.FILE_INDICES:
                if not sel.source:
                    return False
                into.add(sel.source)
            else:
                if not sel.paths:
                    return False
                into.update(sel.paths)
            return True
        if sel.kind in _SET_OP_KINDS:
            # A set operation's children resolve without the zero-match check, so
            # nothing they name is required to exist.
            return all(visit(child, into=optional) for child in sel.of)
        return False

    stages = (splits.train, splits.val, splits.test, splits.predict)
    for stage in stages:
        for sel in stage:
            if not visit(sel, into=required):
                return None
    result = ExplicitSources(frozenset(required), frozenset(optional - required))
    return result if result.named else None


def _dedupe_by_spelling(paths: Any) -> list[Path]:
    """One entry per file, keeping the first spelling seen, ordered by that spelling.

    Two selectors can name one recording differently (a hand-typed row and a scanned one
    differ in drive-letter case on Windows; a legacy row differs in separator), and
    opening it twice would put the same measurements in the universe twice. Comparison
    goes through the shared rule; the surviving path keeps the spelling it arrived with.
    """
    seen: dict[str, Path] = {}
    for path in paths:
        key = _norm_source(str(Path(path).resolve()))
        seen.setdefault(key, Path(path))
    return [seen[key] for key in sorted(seen)]


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
        sdk_cuda: bool = True,
    ) -> None:
        self._refs = refs
        self._processing_mode = processing_mode
        self._cache = Cu3sReaderCache(
            processing_mode=processing_mode,
            max_open_sessions=max_open_sessions,
            read_threads=read_threads,
            sources=len({ref.source for ref in refs}) or 1,
            sdk_cuda=sdk_cuda,
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
        # Explicit source list: the recordings to enumerate, in place of walking
        # ``data_dir``. The caller that already knows which files its split uses (the
        # CuvisNEXT training wizard does) passes them here, so the universe never
        # contains a recording nobody asked for. Accepts a list or a comma string
        # (``--data-arg files=a.cu3s,b.cu3s``); an empty list counts as "not given", so a
        # trainrun preset can ship ``files: []`` as a placeholder and still fall back.
        files: Any = None,
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
        # Process cubes on the GPU. Off means the SDK's 'host' mode, roughly 4x slower per
        # cube. On a machine without CUDA the SDK falls back to the host on its own.
        sdk_cuda: Any = True,
    ) -> None:
        super().__init__(
            splits=splits,
            batch_size=batch_size,
            num_workers=num_workers,
            samples_per_frame=samples_per_frame,
        )

        self.cu3s_file_path = str(cu3s_file_path) if cu3s_file_path else None
        self.data_dir = Path(data_dir) if (self.cu3s_file_path is None and data_dir) else None
        self.files: list[str] | None = (
            parse_str_list(files, key="files") if (self.cu3s_file_path is None and files) else None
        )
        # Folder mode reads *.cu3s; kept as a list so _list_folder_files stays generic.
        self.cu3s_globs: list[str] | None = ["cu3s"] if self.data_dir is not None else None
        # Answered once per instance by _folder_files(); every later caller reuses it, so a
        # constructed universe never depends on when it was asked for.
        self._folder_files_cache: list[Path] | None = None
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
        self.sdk_cuda = parse_bool(sdk_cuda, key="sdk_cuda")
        # Recorded before anything can open a session, since the SDK fixes its device at the
        # first init of a process and ignores every later one.
        configure_cuvis_sdk(cuda=self.sdk_cuda)
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
        """Validate cu3s params: a file, file list or folder source exists; annotations are JSON.

        With an explicit ``files`` list the folder is not consulted at all. That is not
        only cheaper: ``data_dir`` may legitimately be a filesystem root (CuvisNEXT sends
        the deepest folder containing every assigned recording, which for a split spanning
        two drives is a drive root), and the folder check walks until it meets its first
        ``*.cu3s``, recursively when asked to. Validating the list the caller actually
        named avoids walking a whole volume to prove a file exists that we were handed.
        """
        cu3s = params.get("cu3s_file_path")
        data_dir = params.get("data_dir")
        raw_files = params.get("files")
        files = parse_str_list(raw_files, key="files") if raw_files else None
        frames = params.get("frames", "file")
        if frames not in ("file", "measurements"):
            raise ValueError(f"frames must be 'file' or 'measurements', got {frames!r}")
        if not cu3s and not data_dir and not files:
            raise ValueError(
                "cu3s requires 'cu3s_file_path', 'files' (explicit .cu3s paths), or "
                "'data_dir' (a folder of .cu3s files), in params."
            )
        if cu3s:
            if not str(cu3s).endswith(".cu3s"):
                raise ValueError(f"cu3s_file_path must end with .cu3s: {cu3s!r}")
            if not os.path.exists(cu3s):
                raise ValueError(f"cu3s_file_path does not exist: {cu3s}")
        elif files:
            for entry in files:
                if not str(entry).endswith(".cu3s"):
                    raise ValueError(f"files entries must end with .cu3s: {entry!r}")
                if not os.path.isfile(entry):
                    raise ValueError(f"files entry does not exist: {entry}")
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

    # -- which recordings this module looks at ---------------------------------
    @property
    def _folder_mode(self) -> bool:
        """True when the universe comes from a file list or a folder, not one recording."""
        return self.files is not None or self.data_dir is not None

    def _folder_files(self) -> list[Path]:
        """The recordings this module enumerates, in one place, computed once.

        Three answers, in order of how much the caller told us:

        1. ``files`` given: exactly those, and nothing is walked. The caller knows its
           split; a containment check against ``data_dir`` would only get in the way,
           since a split may legitimately span drives.
        2. No ``files``, but every selector in the split names its sources: those, so a
           ``restore-trainrun`` over a folder opens the split's recordings and no others.
           Sources from a stage's own selectors must exist and must sit under
           ``data_dir``; sources reached through a set operation may be absent, because
           core resolves those without its zero-match check.
        3. Otherwise the folder walk, unchanged: a positional or attribute-driven
           selector (``dir_indices``, ``stems``, ``glob``, ``tag``, ``categories``) can
           only be answered against the whole universe.

        Paths come back spelled the way they were given. ``enumerate()`` derives
        ``SampleRef.source`` from them per mode (canonical posix for
        ``frames="measurements"``, verbatim for ``frames="file"``), and resolving here
        would rewrite a relative source into an absolute one, which is exactly what
        ``resolve-splits`` wrote its selectors against.
        """
        if self._folder_files_cache is None:
            self._folder_files_cache = self._resolve_folder_files()
        return self._folder_files_cache

    def _resolve_folder_files(self) -> list[Path]:
        if self.files is not None:
            return _dedupe_by_spelling(Path(f) for f in self.files)
        named = explicit_sources(self._effective_splits() if self.splits else None)
        if named is not None:
            return self._named_source_files(named)
        return self._list_folder_files()

    def _named_source_files(self, named: ExplicitSources) -> list[Path]:
        """Turn the sources a split names into a file list, reporting what is missing."""
        root = self.data_dir.resolve() if self.data_dir is not None else None
        kept: list[Path] = []
        for source in sorted(named.named):
            path = Path(source)
            required = source in named.required
            if not path.is_file():
                if required:
                    raise ValueError(f"the split names a recording that is not a file: {source}")
                continue  # a set operation may name a file that is gone
            if root is not None and not path.resolve().is_relative_to(root):
                if required:
                    raise ValueError(
                        f"the split names {source}, which is outside data_dir {self.data_dir}"
                    )
                continue
            kept.append(path)
        if not kept:
            raise FileNotFoundError(
                f"none of the {len(named.named)} recordings the split names could be read"
            )
        return _dedupe_by_spelling(kept)

    def _list_folder_files(self) -> list[Path]:
        """Sorted, de-duplicated list of ``.cu3s`` files in the source folder.

        ``recursive=True`` walks subfolders (``rglob``), e.g. a dataset root holding
        per-day session folders. Reached only when neither an explicit ``files`` list nor
        a fully source-naming split narrowed the universe first (see ``_folder_files``),
        so ``recursive`` governs this fallback alone.
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
        if self._folder_mode:
            for path in self._folder_files():
                if self.frames == "measurements":
                    source = path.resolve().as_posix()
                    annotation = _sibling_json(None, source)
                    total = count_measurements(source)
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
            sdk_cuda=self.sdk_cuda,
        )

    def category_name_to_id(self) -> dict[str, int] | None:
        """Map COCO category names to ids (from the annotation), or None when unlabeled."""
        annotation = self.annotation_json_path
        if annotation is None and self._folder_mode:
            files = self._folder_files()
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
