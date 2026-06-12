"""cu3s DataModule: ``.cu3s`` cubes via the cuvis SDK + optional COCO masks.

``DATA_MODULE_NAME = "cu3s"`` (manifest extras ``[cu3s, coco]``). Refactor of the
former core ``SingleCu3sDataModule``: the split/dataloader plumbing moved up into
``BaseHyperspectralDataModule``, cube reading into the internal ``Cu3sCubeReader``,
and COCO labeling into the internal ``CocoLabeler``.

A back-compat ``SingleCu3sDataModule`` alias and a ``SingleCu3sDataset`` shim keep
the old call sites working with only an import-path change.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, Sequence

from torch.utils.data import Dataset

from cuvis_ai_core.data.datamodule import BaseHyperspectralDataModule
from cuvis_ai_schemas.training.data import DataSplitConfig

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


class _Cu3sDataset(Dataset):
    """Torch Dataset over a list of cu3s measurement indices (+ optional masks)."""

    def __init__(
        self,
        cu3s_file_path: str,
        mesu_indices: Sequence[int] | None,
        *,
        processing_mode: str = "Reflectance",
        annotation_json_path: str | None = None,
    ) -> None:
        self._reader = Cu3sCubeReader(cu3s_file_path, processing_mode=processing_mode)
        if mesu_indices is None:
            mesu_indices = range(self._reader.total_measurements)
        self._mesu_indices = [int(i) for i in mesu_indices]
        self._labeler = None
        if annotation_json_path:
            # Imported lazily: coco_labeler pulls the [coco] extra (pycocotools /
            # scikit-image), which a cu3s-without-labels or tiff-only env lacks.
            from .labelers.coco_labeler import CocoLabeler

            self._labeler = CocoLabeler(annotation_json_path)

    def __len__(self) -> int:
        return len(self._mesu_indices)

    def __getitem__(self, idx: int) -> dict:
        mesu_index = self._mesu_indices[idx]
        item = self._reader.read(mesu_index)
        if self._labeler is not None:
            item.update(self._labeler.load_for(mesu_index, item))
        return item


class _Cu3sFolderDataset(Dataset):
    """Torch Dataset over a folder of ``.cu3s`` files (measurement 0 of each).

    One sample per file. Multi-measurement files are the domain of the single-file
    module (``measurement_indices``) or ``cu3s_multi``; folder mode keeps the
    universe one-entry-per-file so discovery stays a cheap glob (no SDK session per
    file at setup). A sibling ``<stem>.json`` is loaded as COCO labels when present.
    """

    def __init__(
        self,
        files: Sequence[str],
        *,
        processing_mode: str = "Reflectance",
    ) -> None:
        self._files = [str(f) for f in files]
        self._processing_mode = processing_mode

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> dict:
        path = self._files[idx]
        reader = Cu3sCubeReader(path, processing_mode=self._processing_mode)
        item = reader.read(0)
        item["stem"] = Path(path).stem
        item["mesu_index"] = int(idx)
        sibling = _sibling_json(None, path)
        if sibling:
            from .labelers.coco_labeler import CocoLabeler

            item.update(CocoLabeler(sibling).load_for(0, item))
        return item


class Cu3sDataModule(BaseHyperspectralDataModule):
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
        # Back-compat flat selectors (folded into splits when splits is None).
        train_ids: list | None = None,
        val_ids: list | None = None,
        test_ids: list | None = None,
        predict_ids: list | None = None,
        # Back-compat asset-resolution convenience.
        data_dir: str | None = None,
        dataset_name: str | None = None,
        # Folder source: a data_dir without dataset_name globs *.{glob} into one
        # ordered universe; split selectors then index into it.
        glob: Any = None,
        params: dict | None = None,
        **_: Any,
    ) -> None:
        # Support `Cu3sDataModule(**cfg.data)` where cfg.data is the nested
        # DataConfig shape {data_module, splits, params, batch_size}: pull the
        # module-specific values out of params (explicit kwargs win).
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
        if splits is None and any(
            x is not None for x in (train_ids, val_ids, test_ids, predict_ids)
        ):
            splits = DataSplitConfig(
                train_ids=list(train_ids or []),
                val_ids=list(val_ids or []),
                test_ids=list(test_ids or []),
                predict_ids=list(predict_ids or []),
            )
        super().__init__(splits=splits, batch_size=batch_size, num_workers=num_workers)

        if cu3s_file_path is None and data_dir and dataset_name:
            cu3s_file_path = str(Path(data_dir) / f"{dataset_name}.cu3s")
        self.cu3s_file_path = str(cu3s_file_path) if cu3s_file_path else None
        # Folder mode: a data_dir with no single-file target. Glob defaults to *.cu3s.
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

    @staticmethod
    def validate_params(params: dict[str, Any]) -> None:
        cu3s = params.get("cu3s_file_path")
        data_dir = params.get("data_dir")
        dataset_name = params.get("dataset_name")
        if not cu3s and not data_dir:
            raise ValueError(
                "cu3s requires 'cu3s_file_path', or 'data_dir' (a folder of .cu3s "
                "files, optionally with 'dataset_name' for a single named file), in params."
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

    def build_dataset(self, ids: Sequence[int | str] | None) -> Dataset:
        if self.data_dir is not None:
            return self._build_folder_dataset(ids)
        effective = ids if ids is not None else self.measurement_indices
        return _Cu3sDataset(
            self.cu3s_file_path,
            effective,
            processing_mode=self.processing_mode,
            annotation_json_path=self.annotation_json_path,
        )

    def _list_folder_files(self) -> list[Path]:
        """Sorted, de-duplicated list of ``.cu3s`` files in the source folder."""
        files: list[Path] = []
        for ext in self.cu3s_globs:
            files.extend(self.data_dir.glob(f"*.{ext.lstrip('.')}"))
        files = sorted(set(files))
        if not files:
            raise FileNotFoundError(f"No {self.cu3s_globs} files in {self.data_dir}")
        return files

    def _build_folder_dataset(self, ids: Sequence[int | str] | None) -> Dataset:
        """Resolve int positions / stem keys against the folder universe."""
        files = self._list_folder_files()
        if ids:
            by_stem = {p.stem: p for p in files}
            selected: list[Path] = []
            for sel in ids:
                if isinstance(sel, int):
                    selected.append(files[sel])
                elif sel in by_stem:
                    selected.append(by_stem[sel])
                else:
                    raise ValueError(
                        f"cu3s folder selector {sel!r} is neither an int position nor "
                        f"a known .cu3s stem (have {len(files)} files)."
                    )
            files = selected
        return _Cu3sFolderDataset(files, processing_mode=self.processing_mode)

    def build_stage_dataset(self, stage: str) -> Dataset:
        # cu3s has no module-owned splits; with DataConfig.splits=None (the
        # inference case) every stage iterates the configured measurements (all
        # measurements unless measurement_indices narrows it), or every file in
        # folder mode.
        return self.build_dataset(None)


class SingleCu3sDataset(_Cu3sDataset):
    """Back-compat shim matching the former core ``SingleCu3sDataset`` signature."""

    def __init__(
        self,
        cu3s_file_path: str,
        annotation_json_path: str | None = None,
        processing_mode: str | None = "Raw",
        measurement_indices: Sequence[int] | None = None,
        normalize_to_unit: bool = False,
    ) -> None:
        super().__init__(
            cu3s_file_path,
            measurement_indices,
            processing_mode=processing_mode or "Raw",
            annotation_json_path=_sibling_json(annotation_json_path, cu3s_file_path),
        )


# Back-compat alias: the former core class name maps onto the plugin module.
SingleCu3sDataModule = Cu3sDataModule
