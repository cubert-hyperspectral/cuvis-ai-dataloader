"""cu3s_multi DataModule: multi-file cu3s with CSV-encoded splits + per-day COCO.

``DATA_MODULE_NAME = "cu3s_multi"`` (manifest extras ``[cu3s, coco]``). One ``.cu3s``
per frame, per-day COCO JSONs, an externally-supplied ``splits.csv`` whose rows are
``(split, cu3s_path, annotation_json, image_id)``. The split assignment is
module-owned (it lives in the CSV, not a flat id-list), so this module leaves
``DataConfig.splits = None`` and overrides ``build_stage_dataset``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, ClassVar

from torch.utils.data import Dataset

from cuvis_ai_core.data.datamodule import BaseHyperspectralDataModule
from cuvis_ai_core.utils.general import expand_range_selectors

from .readers.cu3s_reader import Cu3sCubeReader

_REQUIRED_COLUMNS = ("split", "cu3s_path", "annotation_json", "image_id")


class _MultiCu3sDataset(Dataset):
    """Holds the rows for one split; reads each frame's cube + per-day mask."""

    def __init__(self, rows: list[dict], processing_mode: str) -> None:
        self._rows = rows
        self._processing_mode = processing_mode
        self._labelers: dict[str, Any] = {}

    def _labeler_for(self, ann: str):
        if ann not in self._labelers:
            from .labelers.coco_labeler import CocoLabeler

            self._labelers[ann] = CocoLabeler(annotation_json_path=ann)
        return self._labelers[ann]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        rec = self._rows[idx]
        reader = Cu3sCubeReader(rec["cu3s_path"], processing_mode=self._processing_mode)
        item = reader.read(rec["read_index"])
        item.update(
            {
                "mesu_index": int(rec["image_id"]),
                "frame_id": int(rec["frame_id"]),
                "annotation_json": rec["annotation_json"],
            }
        )
        ann = rec["annotation_json"]
        if ann:
            item.update(self._labeler_for(ann).load_for(int(rec["image_id"]), item))
        return item


class MultiCu3sDataModule(BaseHyperspectralDataModule):
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
        params: dict | None = None,
        **_: Any,
    ) -> None:
        if params:
            splits_csv = splits_csv or params.get("splits_csv")
            processing_mode = params.get("processing_mode", processing_mode)
            split = split if split is not None else params.get("split")
        super().__init__(splits=None, batch_size=batch_size, num_workers=num_workers)
        if not splits_csv:
            raise ValueError("cu3s_multi requires 'splits_csv'.")
        self._splits_csv = Path(splits_csv).resolve()
        self._csv_dir = self._splits_csv.parent
        self._processing_mode = processing_mode
        self._predict_split = split  # which CSV split predict_dataloader iterates
        self._rows = self._parse_csv(self._splits_csv)

    @staticmethod
    def validate_params(params: dict[str, Any]) -> None:
        csv_path = params.get("splits_csv")
        if not csv_path:
            raise ValueError("cu3s_multi requires 'splits_csv' in params.")
        if not str(csv_path).endswith(".csv"):
            raise ValueError(f"splits_csv must end with .csv: {csv_path!r}")
        if not Path(csv_path).is_file():
            raise ValueError(f"splits_csv does not exist: {csv_path}")

    def build_stage_dataset(self, stage: str) -> Dataset:
        # DataConfig.splits is None, so the base routes every stage here. Map the
        # lightning stage to a CSV split (predict honors --data-arg split, default test).
        split = (self._predict_split or "test") if stage == "predict" else stage
        rows = [r for r in self._rows if split == "all" or r["split"] == split]
        # Lazy heavy imports land here, never at module top.
        from ._extras import require_cuvis, require_pycocotools, require_skimage_polygon2mask

        require_cuvis()
        require_pycocotools()
        require_skimage_polygon2mask()
        return _MultiCu3sDataset(rows, self._processing_mode)

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
                # A ranged image_id ("0-5") fans the row out into one sample per
                # measurement m (read measurement m, COCO image_id m); a scalar keeps
                # the legacy single-frame-per-file behavior (read measurement 0).
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
                            "frame_id": frame_counter,  # stable global identity
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
