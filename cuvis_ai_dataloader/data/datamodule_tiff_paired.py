"""tiff_paired DataModule: a directory of TIFF cubes + stem-keyed PNG labels.

``DATA_MODULE_NAME = "tiff_paired"`` (manifest extras ``[tiff]``). Promoted from the
HSIMetalScrap experiment's ``tiff_dataset.py``. Emits per file
``{"cube", "wavelengths", "stem", "mesu_index", label_output_key?}``; the default label key
``label_rgb`` is what the HSIMetalScrap viz nodes consume.

Selector path: ``enumerate()`` lists one ref per TIFF file (``dir_indices`` selects by file
position, ``stems`` by name); ``build_dataset_from_refs`` reads the resolved files. Paired
PNGs drive the attributes (``tag`` / ``categories`` / AD-aware) for tiff too.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from torch.utils.data import Dataset

from cuvis_ai_core.data.datamodule import BaseCuvisAIDataModule
from cuvis_ai_schemas.training.data import DataSplitConfig, SampleRef

from ._extras import parse_float_list, parse_str_list
from .labelers.paired_png_labeler import PairedPngLabeler
from .readers.tiff_reader import TiffCubeReader

_DEFAULT_GLOBS = ("tif", "tiff")


class _TiffPairedRefDataset(Dataset):
    """Torch Dataset over resolved ``SampleRef``s (one TIFF file each, + optional PNG)."""

    def __init__(self, refs: list[SampleRef], *, reader: TiffCubeReader, labeler) -> None:
        self._refs = list(refs)
        self._reader = reader
        self._labeler = labeler

    def __len__(self) -> int:
        return len(self._refs)

    def __getitem__(self, idx: int) -> dict:
        ref = self._refs[idx]
        path = Path(ref.source)
        item = self._reader.read(path)
        item["stem"] = ref.stem
        item["mesu_index"] = int(idx)
        # Load the paired label only when this ref is annotated. enumerate() sets
        # ref.annotation to the PNG path when it exists, so an unlabeled file is a valid
        # unannotated sample (no label key) rather than a crash, which AD-aware splits
        # (normals carry no label) rely on.
        if self._labeler is not None and ref.annotation:
            cube = item["cube"]
            item.update(self._labeler.load_for(ref.stem, (cube.shape[0], cube.shape[1])))
        return item


class TiffPairedDataModule(BaseCuvisAIDataModule):
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
        """Validate that ``images_dir`` exists with TIFFs, and any ``labels_dir`` holds PNGs."""
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
        stems = [p.stem for p in files]
        if len(set(stems)) != len(stems):
            raise ValueError(
                f"duplicate TIFF stems in {self.images_dir} (e.g. both .tif and .tiff); "
                "stems must be unique to key paired labels and STEMS selectors."
            )
        return files

    def _labeler(self) -> PairedPngLabeler | None:
        if not self.labels_dir:
            return None
        return PairedPngLabeler(
            self.labels_dir,
            label_output_key=self.label_output_key,
            label_mode=self.label_mode,
        )

    def enumerate(self, required_attrs: frozenset[str] = frozenset()) -> list[SampleRef]:
        """List one attributed ref per TIFF file (paired PNG sets the annotation when present)."""
        labeler = self._labeler() if (required_attrs & {"tags", "category_ids"}) else None
        refs: list[SampleRef] = []
        for path in self._list_files():
            source = str(path)
            annotation = None
            tags: list[str] = []
            cats: list[int] = []
            if self.labels_dir is not None:
                png = self.labels_dir / f"{path.stem}.png"
                annotation = str(png) if png.exists() else None
            if labeler is not None:
                cats = labeler.categories_for(path.stem)
                if "tags" in required_attrs:
                    tags = ["anomalous"] if cats else ["normal"]
                if "category_ids" not in required_attrs:
                    cats = []
            refs.append(
                SampleRef(
                    source=source,
                    index=None,
                    stem=path.stem,
                    annotation=annotation,
                    tags=tags,
                    category_ids=cats,
                )
            )
        refs.sort(key=lambda r: r.source)
        return refs

    def build_dataset_from_refs(self, refs: list[SampleRef]) -> Dataset:
        """Build the torch Dataset reading the resolved TIFF files (+ paired PNGs)."""
        reader = TiffCubeReader(wavelengths_override=self.wavelengths_override)
        return _TiffPairedRefDataset(refs, reader=reader, labeler=self._labeler())

    def category_name_to_id(self) -> dict[str, int] | None:
        """Map label category names to ids (binary normal/anomalous, or label-map values)."""
        if not self.labels_dir:
            return None
        if self.label_mode != "label_map":
            return {"anomalous": 1, "normal": 0}
        labeler = self._labeler()
        names: dict[str, int] = {}
        for path in self._list_files():
            for cid in labeler.categories_for(path.stem):
                names[str(cid)] = cid
        return names or None

    def build_stage_dataset(self, stage: str) -> Dataset:
        """Module-owned path (no splits): every stage iterates every TIFF file."""
        return self.build_dataset_from_refs(self.enumerate())
