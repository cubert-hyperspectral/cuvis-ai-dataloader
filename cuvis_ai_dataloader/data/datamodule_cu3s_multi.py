"""cu3s_multi DataModule: multi-file cu3s with a CSV universe + per-day COCO.

``DATA_MODULE_NAME = "cu3s_multi"`` (manifest extras ``[cu3s, coco]``). One ``.cu3s`` per
frame, per-day COCO JSONs, an externally-supplied ``splits.csv`` whose rows are
``(split, cu3s_path, annotation_json, image_id)``.

Two ways to run:

* **Module-owned** (``DataConfig.splits is None``): each Lightning stage maps to the CSV
  ``split`` column via ``build_stage_dataset``.
* **Selector-driven** (``DataConfig.splits`` set): the CSV rows are the ``enumerate()``
  universe and selectors (or a ``splits.json`` produced by ``resolve-splits --from-csv``)
  pick subsets. Each row is a first-class ``SampleRef`` with a ``uid`` derived from its source
  and read index.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, ClassVar

from torch.utils.data import Dataset

from cuvis_ai_core.data.datamodule import BaseCuvisAIDataModule
from cuvis_ai_core.utils.general import expand_range_selectors
from cuvis_ai_schemas.training.data import SampleRef

from .readers.cu3s_reader import Cu3sCubeReader

_REQUIRED_COLUMNS = ("split", "cu3s_path", "annotation_json", "image_id")


class _MultiCu3sDataset(Dataset):
    """Holds the rows for one subset; reads each frame's cube + per-day mask."""

    def __init__(self, rows: list[dict], processing_mode: str) -> None:
        self._rows = rows
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

    def _labeler_for(self, ann: str):
        if ann not in self._labelers:
            from .labelers.coco_labeler import CocoLabeler

            self._labelers[ann] = CocoLabeler(annotation_json_path=ann)
        return self._labelers[ann]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        rec = self._rows[idx]
        item = self._reader_for(rec["cu3s_path"]).read(rec["read_index"])
        item.update(
            {
                "read_index": int(rec["read_index"]),
                "mesu_index": int(rec["image_id"]),
                "frame_id": int(rec["frame_id"]),
                "annotation_json": rec["annotation_json"],
            }
        )
        ann = rec["annotation_json"]
        if ann:
            item.update(self._labeler_for(ann).load_for(int(rec["image_id"]), item))
        return item


class MultiCu3sDataModule(BaseCuvisAIDataModule):
    """Multi-file cu3s DataModule driven by an external ``splits.csv``."""

    DATA_MODULE_NAME: ClassVar[str] = "cu3s_multi"

    def __init__(
        self,
        *,
        splits=None,
        batch_size: int = 1,
        num_workers: int = 0,
        splits_csv: str | None = None,
        processing_mode: str = "Reflectance",
        split: str | None = None,
        samples_per_frame: int = 1,
        params: dict | None = None,
        # Carried by the nested `cls(**cfg.data)` shape; accepted and ignored (the class
        # identity fixes the module). Any other unknown kwarg raises.
        data_module: str | None = None,
    ) -> None:
        if params:
            splits_csv = splits_csv or params.get("splits_csv")
            processing_mode = params.get("processing_mode", processing_mode)
            split = split if split is not None else params.get("split")
            samples_per_frame = params.get("samples_per_frame", samples_per_frame)
        super().__init__(
            splits=splits,
            batch_size=batch_size,
            num_workers=num_workers,
            samples_per_frame=samples_per_frame,
        )
        if not splits_csv:
            raise ValueError("cu3s_multi requires 'splits_csv'.")
        self._splits_csv = Path(splits_csv).resolve()
        self._csv_dir = self._splits_csv.parent
        self._processing_mode = processing_mode
        self._predict_split = split  # which CSV split predict_dataloader iterates (module-owned)
        self._rows = self._parse_csv(self._splits_csv)

    @property
    def rows(self) -> list[dict]:
        """Public read-only view of the parsed CSV rows (``split, cu3s_path, read_index, ...``)."""
        return self._rows

    @staticmethod
    def validate_params(params: dict[str, Any]) -> None:
        """Validate that a ``splits_csv`` path is given, ends in ``.csv``, and exists."""
        csv_path = params.get("splits_csv")
        if not csv_path:
            raise ValueError("cu3s_multi requires 'splits_csv' in params.")
        if not str(csv_path).endswith(".csv"):
            raise ValueError(f"splits_csv must end with .csv: {csv_path!r}")
        if not Path(csv_path).is_file():
            raise ValueError(f"splits_csv does not exist: {csv_path}")

    # -- module-owned path -----------------------------------------------------
    def build_stage_dataset(self, stage: str) -> Dataset:
        """Module-owned path: map the Lightning stage to the matching CSV ``split`` rows."""
        # DataConfig.splits is None: predict honors --data-arg split (default test).
        split = (self._predict_split or "test") if stage == "predict" else stage
        rows = [r for r in self._rows if split == "all" or r["split"] == split]
        return self._make_dataset(rows)

    # -- selector path ---------------------------------------------------------
    def enumerate(self, required_attrs: frozenset[str] = frozenset()) -> list[SampleRef]:
        """List the CSV rows as the attributed sample universe (one ref per row)."""
        labelers: dict[str, Any] = {}

        def attrs(ann: str | None, image_id: int) -> tuple[list[str], list[int]]:
            if not ann or not (required_attrs & {"tags", "category_ids"}):
                return [], []
            if ann not in labelers:
                from .labelers.coco_labeler import CocoLabeler

                labelers[ann] = CocoLabeler(annotation_json_path=ann)
            cats = labelers[ann].categories_for(image_id)
            tags = (["anomalous"] if cats else ["normal"]) if "tags" in required_attrs else []
            return tags, (cats if "category_ids" in required_attrs else [])

        refs: list[SampleRef] = []
        for rec in self._rows:
            ann = rec["annotation_json"] or None
            tags, cats = attrs(ann, int(rec["image_id"]))
            refs.append(
                SampleRef(
                    source=rec["cu3s_path"],
                    index=int(rec["read_index"]),
                    label_id=int(rec["image_id"]),
                    stem=Path(rec["cu3s_path"]).stem,
                    annotation=ann,
                    group=rec["cu3s_path"],
                    tags=tags,
                    category_ids=cats,
                )
            )
        refs.sort(
            key=lambda r: (
                r.source,
                -1 if r.index is None else r.index,
                -1 if r.label_id is None else r.label_id,
            )
        )
        return refs

    def build_dataset_from_refs(self, refs: list[SampleRef]) -> Dataset:
        """Build the dataset for the resolved subset, one row per ``SampleRef``."""
        rows = []
        for i, ref in enumerate(refs):
            read_index = int(ref.index if ref.index is not None else 0)
            image_id = int(ref.label_id if ref.label_id is not None else read_index)
            rows.append(
                {
                    "frame_id": i,
                    "cu3s_path": ref.source,
                    "annotation_json": ref.annotation or "",
                    "image_id": image_id,
                    "read_index": read_index,
                }
            )
        return self._make_dataset(rows)

    def category_name_to_id(self) -> dict[str, int] | None:
        """Map COCO category names to ids from the first annotated row, or None if unlabeled."""
        for rec in self._rows:
            ann = rec["annotation_json"]
            if ann:
                from .labelers.coco_labeler import CocoLabeler

                labeler = CocoLabeler(annotation_json_path=ann)
                return {name: cid for cid, name in labeler.category_id_to_name.items()}
        return None

    # -- shared dataset construction -------------------------------------------
    def _make_dataset(self, rows: list[dict[str, Any]]) -> Dataset:
        from ._extras import require_cuvis

        require_cuvis()
        # COCO deps are only needed when at least one row carries an annotation.
        if any(rec.get("annotation_json") for rec in rows):
            from ._extras import require_pycocotools, require_skimage_polygon2mask

            require_pycocotools()
            require_skimage_polygon2mask()
        self._validate_read_indices(rows)
        return _MultiCu3sDataset(rows, self._processing_mode)

    def _validate_read_indices(self, rows: list[dict[str, Any]]) -> None:
        """Fail loud at build if any row's read_index is out of [0, total_measurements)."""
        max_by_source: dict[str, int] = {}
        for rec in rows:
            read_index = int(rec["read_index"])
            if read_index < 0:
                raise ValueError(f"negative read_index {read_index} for {rec['cu3s_path']}")
            src = rec["cu3s_path"]
            max_by_source[src] = max(max_by_source.get(src, -1), read_index)
        for src, max_ri in max_by_source.items():
            reader = Cu3sCubeReader(src, processing_mode=self._processing_mode)
            try:
                total = reader.total_measurements
            finally:
                reader.close()
            if max_ri >= total:
                raise ValueError(f"row read_index {max_ri} >= {total} measurements in {src}")

    def _parse_csv(self, csv_path: Path) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        frame_counter = 0
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = [c for c in _REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(
                    f"{csv_path}: missing required column(s) {missing}. "
                    f"Required: {list(_REQUIRED_COLUMNS)}. Extra columns are allowed and ignored."
                )
            for row in reader:
                cu3s_path = str(self._resolve(row["cu3s_path"]))
                annotation_json = (
                    str(self._resolve(row["annotation_json"])) if row["annotation_json"] else ""
                )
                # A ranged image_id ("0-5") fans the row out into one sample per measurement m
                # (read measurement m, COCO image_id m); a scalar keeps the legacy single-frame
                # behavior (read measurement 0).
                cell = str(row["image_id"]).strip()
                if "-" in cell:
                    measurements = [int(x) for x in expand_range_selectors([cell])]
                    ranged = True
                else:
                    measurements = [int(cell)]
                    ranged = False
                for m in measurements:
                    out.append(
                        {
                            "frame_id": frame_counter,  # stable global row identity
                            "split": row["split"],
                            "cu3s_path": cu3s_path,
                            "annotation_json": annotation_json,
                            "image_id": m if ranged else int(cell),
                            "read_index": m if ranged else 0,
                        }
                    )
                    frame_counter += 1
        if not out:
            raise ValueError(f"{csv_path}: no rows.")
        return out

    def _resolve(self, raw: str) -> Path:
        """Resolve relative paths against the CSV's parent dir; pass absolutes through."""
        p = Path(raw)
        return p if p.is_absolute() else (self._csv_dir / p).resolve()
