"""cu3s_workspace DataModule: workspace.json membership + splits.json sidecar.

``DATA_MODULE_NAME = "cu3s_workspace"`` (manifest extras ``[cu3s, coco]``). The
GUI-driven sibling of ``cu3s_multi``: instead of a CSV, the split assignment lives
in the workspace folder — ``workspace.json`` lists the member ``.cu3s`` files
(absolute paths, anywhere on disk) and ``splits.json`` (written by the split
resolver) carries per-file ``train/val/test/predict`` frame-id lists. Split
assignment is module-owned, so this module leaves ``DataConfig.splits = None``
and overrides ``build_stage_dataset`` — the pattern the base documents for
non-flat splits. Frames are served through the same ``_MultiCu3sDataset`` rows as
``cu3s_multi`` (one row per frame; per-file COCO sibling auto-discovered).
"""

from __future__ import annotations

from typing import Any, ClassVar

from torch.utils.data import Dataset

from cuvis_ai_core.data.datamodule import BaseCuvisAIDataModule
from cuvis_ai_core.utils.general import expand_range_selectors

from .datamodule_cu3s_multi import _MultiCu3sDataset
from .workspace import SplitsFile, Workspace, coco_sibling

_STAGES = ("train", "val", "test", "predict")


class WorkspaceCu3sDataModule(BaseCuvisAIDataModule):
    """Multi-file cu3s DataModule driven by a workspace folder."""

    DATA_MODULE_NAME: ClassVar[str] = "cu3s_workspace"

    def __init__(
        self,
        *,
        splits=None,
        batch_size: int = 1,
        num_workers: int = 0,
        workspace_path: str | None = None,
        processing_mode: str | None = None,
        params: dict | None = None,
        **_: Any,
    ) -> None:
        if params:
            workspace_path = workspace_path or params.get("workspace_path")
            processing_mode = processing_mode or params.get("processing_mode")
        super().__init__(splits=None, batch_size=batch_size, num_workers=num_workers)
        if not workspace_path:
            raise ValueError("cu3s_workspace requires 'workspace_path'.")
        self._workspace = Workspace.load(workspace_path)
        missing = self._workspace.missing_paths()
        if missing:
            raise ValueError(
                "workspace has missing measurement file(s) (locate or remove them):\n  "
                + "\n  ".join(missing)
            )
        self._splits_file = SplitsFile.load(self._workspace.root)
        self._splits_file.validate_against(self._workspace)
        # Training/inference mode comes from the workspace unless explicitly
        # overridden — regardless of what mode any GUI tab is being VIEWED in.
        self._processing_mode = processing_mode or self._workspace.default_processing_mode
        self._rows_by_stage = self._build_rows()

    @classmethod
    def resolve_splits(cls, config: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        """Registry hook for the ``ResolveSplits`` RPC.

        ``config`` is a ``SplitsResolveConfig``-shaped mapping. Returns the
        resolved splits payload (the ``splits.json`` content) plus the path it
        was written to (``None`` when ``write=False``). Core dispatches here via
        ``registry.data_modules[config["data_module"]]`` so the strategy
        semantics stay plugin-owned.
        """
        from .split_resolver import resolve_splits as _resolve

        workspace = Workspace.load(config["workspace_path"])
        result = _resolve(
            workspace,
            strategy=config.get("strategy", "random"),
            train_ratio=config.get("train_ratio", 0.70),
            val_ratio=config.get("val_ratio", 0.15),
            seed=config.get("seed"),
            selected_files=config.get("selected_files"),
            write=config.get("write", True),
        )
        payload = {
            "version": result.version,
            "coco_hash_per_file": result.coco_hash_per_file,
            "files": result.files,
        }
        written = str(workspace.root / "splits.json") if config.get("write", True) else None
        return payload, written

    @staticmethod
    def validate_params(params: dict[str, Any]) -> None:
        from pathlib import Path

        from .workspace import SPLITS_FILENAME, WORKSPACE_FILENAME

        workspace_path = params.get("workspace_path")
        if not workspace_path:
            raise ValueError("cu3s_workspace requires 'workspace_path' in params.")
        root = Path(workspace_path)
        root = root.parent if root.is_file() else root
        if not (root / WORKSPACE_FILENAME).is_file():
            raise ValueError(f"not a workspace (no {WORKSPACE_FILENAME}): {workspace_path}")
        if not (root / SPLITS_FILENAME).is_file():
            raise ValueError(
                f"workspace has no {SPLITS_FILENAME} yet: run the split resolver "
                f"(ResolveSplits / Create splits) before training."
            )

    def build_stage_dataset(self, stage: str) -> Dataset:
        rows = self._rows_by_stage.get(stage, [])
        if stage == "predict" and not rows:
            # DataSplitConfig convention: an empty predict set means ALL frames.
            rows = self._all_frames_rows()
        # Lazy heavy imports land here, never at module top.
        from ._extras import require_cuvis, require_pycocotools, require_skimage_polygon2mask

        require_cuvis()
        require_pycocotools()
        require_skimage_polygon2mask()
        return _MultiCu3sDataset(rows, self._processing_mode)

    # -- row construction --------------------------------------------------------
    def _build_rows(self) -> dict[str, list[dict[str, Any]]]:
        """One row per (member file, frame, stage); same shape as the CSV rows of
        ``cu3s_multi`` so ``_MultiCu3sDataset`` is reused unchanged. ``frame_id``
        is a module-owned, stable global counter across members in order."""
        rows: dict[str, list[dict[str, Any]]] = {s: [] for s in _STAGES}
        frame_counter = 0
        for m in self._workspace.member_measurements():
            lists = self._splits_file.files.get(m.path)
            if lists is None:
                continue  # member without splits: not part of this run
            sibling = coco_sibling(m.path)
            annotation_json = str(sibling) if sibling else ""
            for stage in _STAGES:
                for frame in expand_range_selectors(lists.get(stage, [])):
                    rows[stage].append(
                        {
                            "frame_id": frame_counter,
                            "split": stage,
                            "cu3s_path": m.path,
                            "annotation_json": annotation_json,
                            "image_id": int(frame),
                            "read_index": int(frame),
                        }
                    )
                    frame_counter += 1
        return rows

    def _all_frames_rows(self) -> list[dict[str, Any]]:
        """Every frame of every member (the empty-predict case). Needs the cached
        ``frames`` metadata; members without it are named rather than skipped."""
        rows: list[dict[str, Any]] = []
        uncounted = [m.path for m in self._workspace.member_measurements() if m.frames is None]
        if uncounted:
            raise ValueError(
                "predict-on-all needs cached 'frames' for every measurement; missing for:\n  "
                + "\n  ".join(uncounted)
            )
        frame_counter = 0
        for m in self._workspace.member_measurements():
            sibling = coco_sibling(m.path)
            for frame in range(int(m.frames or 0)):
                rows.append(
                    {
                        "frame_id": frame_counter,
                        "split": "predict",
                        "cu3s_path": m.path,
                        "annotation_json": str(sibling) if sibling else "",
                        "image_id": frame,
                        "read_index": frame,
                    }
                )
                frame_counter += 1
        return rows


__all__ = ["WorkspaceCu3sDataModule"]
