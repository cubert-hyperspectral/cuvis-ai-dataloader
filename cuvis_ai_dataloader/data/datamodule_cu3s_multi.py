"""cu3s_multi DataModule: multi-file cu3s over a shared ``universe.csv`` + per-day COCO.

``DATA_MODULE_NAME = "cu3s_multi"`` (manifest extras ``[cu3s, coco]``). Reads the shared
``universe.csv`` vocabulary (``source, index [, materialized_path, split, annotation, format,
group]``) via :mod:`cuvis_ai_dataloader.data._universe`; each frame is a measurement ``index`` of
a ``.cu3s`` recording (``materialized_path``, defaulting to ``source``), optionally labeled by a
per-day COCO ``annotation``.

Two ways to run:

* **Module-owned** (``DataConfig.splits is None``): each Lightning stage maps to the CSV
  ``split`` column via ``build_stage_dataset``.
* **Selector-driven** (``DataConfig.splits`` set): the CSV rows are the ``enumerate()`` universe
  and selectors (or a ``splits.json`` produced by ``resolve-splits --from-csv``) pick subsets.
  Each row is a first-class ``SampleRef`` whose ``uid`` derives from its posix ``source`` and
  read ``index`` — so one ``splits.json`` resolves against both the raw cu3s data and a converted
  npz universe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from torch.utils.data import Dataset

from cuvis_ai_core.data.datamodule import BaseCuvisAIDataModule
from cuvis_ai_schemas.training.data import SampleRef

from ._universe import parse_universe, validate_universe_csv_param
from .readers.cu3s_reader import Cu3sCubeReader


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

    def _reader_for(self, path: str) -> Cu3sCubeReader:
        if path not in self._readers:
            self._readers[path] = Cu3sCubeReader(path, processing_mode=self._processing_mode)
        return self._readers[path]

    def _labeler_for(self, ann: str):
        if ann not in self._labelers:
            from .labelers.coco_labeler import CocoLabeler

            self._labelers[ann] = CocoLabeler(annotation_json_path=ann)
        return self._labelers[ann]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        rec = self._rows[idx]
        item = self._reader_for(rec["materialized_path"]).read(rec["index"])
        item.update(
            {
                "read_index": int(rec["index"]),
                "mesu_index": int(rec["index"]),
                "frame_id": int(rec["frame_id"]),
                "annotation_json": rec["annotation"],
            }
        )
        ann = rec["annotation"]
        if ann:
            item.update(self._labeler_for(ann).load_for(int(rec["index"]), item))
        return item


class MultiCu3sDataModule(BaseCuvisAIDataModule):
    """Multi-file cu3s DataModule driven by a shared ``universe.csv``."""

    DATA_MODULE_NAME: ClassVar[str] = "cu3s_multi"

    def __init__(
        self,
        *,
        splits=None,
        batch_size: int = 1,
        num_workers: int = 0,
        universe_csv: str | None = None,
        processing_mode: str = "Reflectance",
        split: str | None = None,
        samples_per_frame: int = 1,
        params: dict | None = None,
        # Carried by the nested `cls(**cfg.data)` shape; accepted and ignored (the class
        # identity fixes the module). Any other unknown kwarg raises.
        data_module: str | None = None,
    ) -> None:
        if params:
            universe_csv = universe_csv or params.get("universe_csv")
            processing_mode = params.get("processing_mode", processing_mode)
            split = split if split is not None else params.get("split")
            samples_per_frame = params.get("samples_per_frame", samples_per_frame)
        super().__init__(
            splits=splits,
            batch_size=batch_size,
            num_workers=num_workers,
            samples_per_frame=samples_per_frame,
        )
        if not universe_csv:
            raise ValueError("cu3s_multi requires 'universe_csv'.")
        self._universe_csv = Path(universe_csv).resolve()
        self._processing_mode = processing_mode
        self._predict_split = split  # which CSV split predict_dataloader iterates (module-owned)
        self._rows = parse_universe(
            self._universe_csv,
            require_materialized_path=False,  # a raw .cu3s IS its own file; default to source
            accept_split=True,
            unique_materialized_path=False,  # one recording holds many frames
            allow_index_ranges=True,  # `index=0-49` fans out into one row per measurement
        )
        # A `split` column makes the module own its splits; without it a training stage needs an
        # explicit splits.json (the base raises), while predict stays valid either way.
        self.OWNS_SPLITS = any(r["split"] for r in self._rows)

    @property
    def rows(self) -> list[dict]:
        """Public read-only view of the parsed universe rows (``source, index, split, ...``)."""
        return self._rows

    @staticmethod
    def validate_params(params: dict[str, Any]) -> None:
        """Validate that a ``universe_csv`` path is given, ends in ``.csv``, and exists."""
        validate_universe_csv_param(params, "cu3s_multi")

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
            ann = rec["annotation"] or None
            tags, cats = attrs(ann, int(rec["index"]))
            refs.append(
                SampleRef(
                    source=rec["source"],
                    index=int(rec["index"]),
                    label_id=int(rec["index"]),
                    stem=Path(rec["source"]).stem,
                    annotation=ann,
                    group=rec["group"] or rec["source"],
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
        """Build the dataset for the resolved subset, mapping each ref to its universe row."""
        by_identity = {(rec["source"], int(rec["index"])): rec for rec in self._rows}
        rows = []
        for i, ref in enumerate(refs):
            index = int(ref.index if ref.index is not None else 0)
            rec = by_identity.get((ref.source, index))
            if rec is None:
                raise ValueError(f"ref ({ref.source}, {index}) has no matching row in the universe")
            rows.append(
                {
                    "frame_id": i,
                    "source": rec["source"],
                    "materialized_path": rec["materialized_path"],
                    "annotation": rec["annotation"],
                    "index": index,
                }
            )
        return self._make_dataset(rows)

    def category_name_to_id(self) -> dict[str, int] | None:
        """Map COCO category names to ids from the first annotated row, or None if unlabeled."""
        for rec in self._rows:
            ann = rec["annotation"]
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
        if any(rec.get("annotation") for rec in rows):
            from ._extras import require_pycocotools, require_skimage_polygon2mask

            require_pycocotools()
            require_skimage_polygon2mask()
        self._validate_read_indices(rows)
        return _MultiCu3sDataset(rows, self._processing_mode)

    def _validate_read_indices(self, rows: list[dict[str, Any]]) -> None:
        """Fail loud at build if any row's read index is out of ``[0, total_measurements)``."""
        max_by_path: dict[str, int] = {}
        for rec in rows:
            index = int(rec["index"])
            if index < 0:
                raise ValueError(f"negative read index {index} for {rec['materialized_path']}")
            path = rec["materialized_path"]
            max_by_path[path] = max(max_by_path.get(path, -1), index)
        for path, max_idx in max_by_path.items():
            reader = Cu3sCubeReader(path, processing_mode=self._processing_mode)
            try:
                total = reader.total_measurements
            finally:
                reader.close()
            if max_idx >= total:
                raise ValueError(f"row read index {max_idx} >= {total} measurements in {path}")
