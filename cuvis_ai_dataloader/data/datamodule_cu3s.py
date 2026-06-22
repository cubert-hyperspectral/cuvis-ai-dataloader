"""cu3s DataModule: ``.cu3s`` cubes via the cuvis SDK + optional COCO masks.

``DATA_MODULE_NAME = "cu3s"`` (manifest extras ``[cu3s, coco]``). The split/dataloader
plumbing lives in ``BaseCuvisAIDataModule``; cube reading is the internal
``Cu3sCubeReader`` and COCO labeling the internal ``CocoLabeler``.

Selector path: ``enumerate()`` lists the attributed measurement universe (single-file mode:
one ref per measurement; folder mode: one ref per file at measurement 0), and
``build_dataset_from_refs`` reads exactly the resolved subset.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

from torch.utils.data import Dataset

from cuvis_ai_core.data.datamodule import BaseCuvisAIDataModule
from cuvis_ai_schemas.training.data import DataSplitConfig, SampleRef

from ._extras import parse_bool, parse_int_list, parse_str_list
from .readers.cu3s_reader import Cu3sCubeReader


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
    """

    def __init__(self, refs: list[SampleRef], processing_mode: str) -> None:
        self._refs = refs
        self._processing_mode = processing_mode
        self._readers: dict[str, Cu3sCubeReader] = {}
        self._labelers: dict[str, Any] = {}

    def __getstate__(self) -> dict:
        # Drop cached SDK readers/labelers before pickling to DataLoader workers; each
        # worker reopens its own session lazily in __getitem__ (native handles don't pickle).
        state = self.__dict__.copy()
        state["_readers"] = {}
        state["_labelers"] = {}
        return state

    def _reader_for(self, source: str) -> Cu3sCubeReader:
        if source not in self._readers:
            self._readers[source] = Cu3sCubeReader(source, processing_mode=self._processing_mode)
        return self._readers[source]

    def _labeler_for(self, annotation: str):
        if annotation not in self._labelers:
            from .labelers.coco_labeler import CocoLabeler

            self._labelers[annotation] = CocoLabeler(annotation)
        return self._labelers[annotation]

    def __len__(self) -> int:
        return len(self._refs)

    def __getitem__(self, idx: int) -> dict:
        ref = self._refs[idx]
        read_pos = ref.index if ref.index is not None else 0
        item = self._reader_for(ref.source).read(read_pos)
        item["stem"] = ref.stem
        # COCO image id (defaults to the read position); kept distinct from read_index.
        image_id = ref.label_id if ref.label_id is not None else read_pos
        item["read_index"] = int(read_pos)
        item["mesu_index"] = int(image_id)
        if ref.annotation:
            item.update(self._labeler_for(ref.annotation).load_for(int(image_id), item))
        return item


class Cu3sDataModule(BaseCuvisAIDataModule):
    """cu3s + COCO DataModule on the shared base."""

    DATA_MODULE_NAME: ClassVar[str] = "cu3s"

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
        normalize_to_unit: Any = False,
        # Back-compat asset-resolution convenience.
        data_dir: str | None = None,
        dataset_name: str | None = None,
        # Folder source: a data_dir without dataset_name globs *.{glob} into one ordered
        # universe; selectors then index into it.
        glob: Any = None,
        params: dict | None = None,
        **_: Any,
    ) -> None:
        # Support `Cu3sDataModule(**cfg.data)` where cfg.data is the nested DataConfig shape
        # {data_module, splits, params, batch_size}: pull module values out of params.
        if params:
            cu3s_file_path = cu3s_file_path or params.get("cu3s_file_path")
            annotation_json_path = annotation_json_path or params.get("annotation_json_path")
            processing_mode = params.get("processing_mode", processing_mode)
            if measurement_indices is None:
                measurement_indices = params.get("measurement_indices")
            normalize_to_unit = params.get("normalize_to_unit", normalize_to_unit)
            data_dir = data_dir or params.get("data_dir")
            dataset_name = dataset_name or params.get("dataset_name")
            glob = glob if glob is not None else params.get("glob")
        super().__init__(splits=splits, batch_size=batch_size, num_workers=num_workers)

        if cu3s_file_path is None and data_dir and dataset_name:
            cu3s_file_path = str(Path(data_dir) / f"{dataset_name}.cu3s")
        self.cu3s_file_path = str(cu3s_file_path) if cu3s_file_path else None
        self.data_dir = Path(data_dir) if (self.cu3s_file_path is None and data_dir) else None
        self.cu3s_globs: list[str] | None = None
        if self.data_dir is not None:
            self.cu3s_globs = (
                parse_str_list(glob, key="glob")
                if isinstance(glob, str)
                else (list(glob) if glob else ["cu3s"])
            )
        self.annotation_json_path = _sibling_json(annotation_json_path, self.cu3s_file_path)
        self.processing_mode = processing_mode
        self.measurement_indices = (
            parse_int_list(measurement_indices, key="measurement_indices")
            if isinstance(measurement_indices, str)
            else measurement_indices
        )
        # Accepted for compatibility; currently inert (never applied to the cube).
        self.normalize_to_unit = (
            parse_bool(normalize_to_unit, key="normalize_to_unit")
            if isinstance(normalize_to_unit, str)
            else bool(normalize_to_unit)
        )
        self._enum_labelers: dict[str, Any] = {}

    @staticmethod
    def validate_params(params: dict[str, Any]) -> None:
        """Validate cu3s params: a file (or folder) source exists and any annotation is JSON."""
        cu3s = params.get("cu3s_file_path")
        data_dir = params.get("data_dir")
        dataset_name = params.get("dataset_name")
        if not cu3s and not data_dir:
            raise ValueError(
                "cu3s requires 'cu3s_file_path', or 'data_dir' (a folder of .cu3s files, "
                "optionally with 'dataset_name' for a single named file), in params."
            )
        if cu3s:
            if not str(cu3s).endswith(".cu3s"):
                raise ValueError(f"cu3s_file_path must end with .cu3s: {cu3s!r}")
            if not os.path.exists(cu3s):
                raise ValueError(f"cu3s_file_path does not exist: {cu3s}")
        elif dataset_name:
            named = Path(data_dir) / f"{dataset_name}.cu3s"
            if not named.exists():
                raise ValueError(f"cu3s file does not exist: {named}")
        else:
            folder = Path(data_dir)
            if not folder.is_dir():
                raise ValueError(f"data_dir does not exist or is not a directory: {data_dir}")
            glob = params.get("glob") or "cu3s"
            exts = (
                [e.strip().lstrip(".") for e in str(glob).split(",") if e.strip()]
                if isinstance(glob, str)
                else [str(e).lstrip(".") for e in glob]
            )
            if not any(any(folder.glob(f"*.{e}")) for e in exts):
                raise ValueError(f"data_dir holds no *.{exts} files: {data_dir}")
        ann = params.get("annotation_json_path")
        if ann:
            if not str(ann).endswith(".json"):
                raise ValueError(f"annotation_json_path must end with .json: {ann!r}")
            if not os.path.exists(ann):
                raise ValueError(f"annotation_json_path does not exist: {ann}")

    # -- selector contract -----------------------------------------------------
    def _list_folder_files(self) -> list[Path]:
        """Sorted, de-duplicated list of ``.cu3s`` files in the source folder."""
        files: list[Path] = []
        for ext in self.cu3s_globs:
            files.extend(self.data_dir.glob(f"*.{ext.lstrip('.')}"))
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
        """List the attributed sample universe (one ref per measurement, or per folder file)."""
        refs: list[SampleRef] = []
        if self.data_dir is not None:
            for path in self._list_folder_files():
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
        return _Cu3sRefDataset(refs, self.processing_mode)

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
        """Module-owned path (no splits): every stage iterates the whole configured universe."""
        # All measurements (single-file) or every file (folder mode).
        return self.build_dataset_from_refs(self.enumerate())
