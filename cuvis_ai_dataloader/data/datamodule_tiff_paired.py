"""tiff_paired DataModule: a directory of TIFF cubes + stem-keyed PNG labels.

``DATA_MODULE_NAME = "tiff_paired"`` (manifest extras ``[tiff]``). Promoted from
the HSIMetalScrap experiment's ``tiff_dataset.py``. Emits per file
``{"cube", "wavelengths", "stem", "mesu_index", label_output_key?}``; the default
label key ``label_rgb`` is what the HSIMetalScrap viz nodes consume.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Sequence

from torch.utils.data import Dataset

from cuvis_ai_core.data.datamodule import BaseHyperspectralDataModule
from cuvis_ai_schemas.training.data import DataSplitConfig

from ._extras import parse_float_list, parse_str_list
from .labelers.paired_png_labeler import PairedPngLabeler
from .readers.tiff_reader import TiffCubeReader

_DEFAULT_GLOBS = ("tif", "tiff")


class _TiffPairedDataset(Dataset):
    """Torch Dataset over a list of TIFF files (+ optional paired PNG labels)."""

    def __init__(self, files: list[Path], *, reader: TiffCubeReader, labeler) -> None:
        self._files = list(files)
        self._reader = reader
        self._labeler = labeler

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> dict:
        path = self._files[idx]
        item = self._reader.read(path)
        item["stem"] = path.stem
        item["mesu_index"] = int(idx)
        if self._labeler is not None:
            cube = item["cube"]
            item.update(self._labeler.load_for(path.stem, (cube.shape[0], cube.shape[1])))
        return item


class TiffPairedDataModule(BaseHyperspectralDataModule):
    """TIFF cubes + paired-PNG labels DataModule on the shared base."""

    DATA_MODULE_NAME: ClassVar[str] = "tiff_paired"

    def __init__(
        self,
        *,
        splits: DataSplitConfig | None = None,
        batch_size: int = 1,
        num_workers: int = 0,
        images_dir: str | None = None,
        labels_dir: str | None = None,
        glob: Any = "tif,tiff",
        wavelengths: Any = None,
        label_output_key: str = "label_rgb",
        label_mode: str = "rgb",
        params: dict | None = None,
        **_: Any,
    ) -> None:
        if params:
            images_dir = images_dir or params.get("images_dir")
            labels_dir = labels_dir or params.get("labels_dir")
            glob = params.get("glob", glob)
            wavelengths = wavelengths if wavelengths is not None else params.get("wavelengths")
            label_output_key = params.get("label_output_key", label_output_key)
            label_mode = params.get("label_mode", label_mode)
        super().__init__(splits=splits, batch_size=batch_size, num_workers=num_workers)
        self.images_dir = Path(images_dir) if images_dir else None
        self.labels_dir = Path(labels_dir) if labels_dir else None
        self.globs = (
            parse_str_list(glob, key="glob") if isinstance(glob, str) else list(glob)
        ) or list(_DEFAULT_GLOBS)
        self.wavelengths_override = (
            parse_float_list(wavelengths, key="wavelengths") if wavelengths else None
        )
        self.label_output_key = label_output_key
        self.label_mode = label_mode

    @staticmethod
    def validate_params(params: dict[str, Any]) -> None:
        images_dir = params.get("images_dir")
        if not images_dir:
            raise ValueError("tiff_paired requires 'images_dir' in params.")
        images = Path(images_dir)
        if not images.is_dir():
            raise ValueError(f"images_dir does not exist or is not a directory: {images}")
        has_tif = any(images.glob("*.tif")) or any(images.glob("*.tiff"))
        if not has_tif:
            raise ValueError(f"images_dir holds no *.tif / *.tiff files: {images}")
        labels_dir = params.get("labels_dir")
        if labels_dir:
            labels = Path(labels_dir)
            if not labels.is_dir():
                raise ValueError(f"labels_dir does not exist: {labels}")
            if not any(labels.glob("*.png")):
                raise ValueError(f"labels_dir holds no *.png files: {labels}")

    def _list_files(self) -> list[Path]:
        files: list[Path] = []
        for ext in self.globs:
            files.extend(self.images_dir.glob(f"*.{ext.lstrip('.')}"))
        files = sorted(set(files))
        if not files:
            raise FileNotFoundError(f"No {self.globs} files in {self.images_dir}")
        return files

    def _make_dataset(self, files: list[Path]) -> Dataset:
        reader = TiffCubeReader(wavelengths_override=self.wavelengths_override)
        labeler = (
            PairedPngLabeler(
                self.labels_dir,
                label_output_key=self.label_output_key,
                label_mode=self.label_mode,
            )
            if self.labels_dir
            else None
        )
        return _TiffPairedDataset(files, reader=reader, labeler=labeler)

    def build_dataset(self, ids: Sequence[int | str] | None) -> Dataset:
        files = self._list_files()
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
                        f"tiff_paired selector {sel!r} is neither an int position nor a "
                        f"known TIFF stem (have {len(files)} files)."
                    )
            files = selected
        return self._make_dataset(files)

    def build_stage_dataset(self, stage: str) -> Dataset:
        return self.build_dataset(None)
