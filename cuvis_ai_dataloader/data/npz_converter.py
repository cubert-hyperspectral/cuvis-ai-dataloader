"""Convert ``.cu3s`` sessions into per-frame ``.npz`` files (for the ``npz_multi`` loader).

For each measurement of a ``.cu3s`` this reads the cube (Preview -> Reflectance via the cu3s
reader), optionally rasterizes that frame's COCO annotations into a binary ``mask`` +
multi-class ``class_mask`` (via the COCO labeler), optionally crops, and writes one compressed
``.npz`` per frame:

* ``cube``        : ``[H, W, C]`` float32
* ``wavelengths`` : ``[C]`` int32
* ``mask``        : ``[H, W]`` int32 binary GT      (only when annotations are given)
* ``class_mask``  : ``[H, W]`` uint8 category id    (only when annotations are given; 0 = bg)
* ``source_cu3s`` : originating ``.cu3s`` path (traceability)

These load directly via :class:`~cuvis_ai_dataloader.data.datamodule_npz_multi.MultiNpzDataModule`.

**No train/val/test split is assigned here** — splitting is a separate concern. The converter
only emits a small index (``npz_path, source_cu3s, image_id``) so a frame can be traced back to
its source session; a split CSV is produced elsewhere and joined on that.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

# (top, bottom, left, right) pixel margins removed from the first two axes.
CropMargins = tuple[int, int, int, int]


def derive_masks(category_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a per-pixel category-id mask into ``(mask, class_mask)``.

    ``mask``       : ``[H, W]`` int32 binary (1 where any category, 0 background).
    ``class_mask`` : ``[H, W]`` uint8 category id (0 = background), as produced by the labeler.
    """
    cat = np.asarray(category_mask)
    class_mask = cat.astype(np.uint8)
    mask = (cat > 0).astype(np.int32)
    return mask, class_mask


def apply_crop(arr: np.ndarray, crop: CropMargins | None) -> np.ndarray:
    """Remove ``(top, bottom, left, right)`` margins from the first two axes. ``None`` -> as-is.

    Applied identically to the cube and the masks so they stay pixel-aligned.
    """
    if crop is None:
        return arr
    top, bottom, left, right = (int(v) for v in crop)
    if min(top, bottom, left, right) < 0:
        raise ValueError(f"crop margins must be non-negative, got {crop}")
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if top + bottom >= h or left + right >= w:
        raise ValueError(f"crop margins {crop} too large for spatial shape {(h, w)}")
    return arr[top : h - bottom, left : w - right]


def convert_cu3s_file(
    cu3s_path: str | Path,
    out_dir: str | Path,
    *,
    annotation_json: str | Path | None = None,
    crop: CropMargins | None = None,
    processing_mode: str | None = "Reflectance",
    frame_indices: list[int] | None = None,
    compress: bool = True,
) -> list[dict[str, Any]]:
    """Convert one ``.cu3s`` to per-frame ``.npz`` files; return index records (no split).

    Masks are rasterized at the full cube size, then the same ``crop`` is applied to cube and
    masks together (so polygon coordinates stay aligned). Frames whose ``image_id`` is absent
    from / unannotated in the COCO get an all-zero mask (i.e. a normal frame). When
    ``annotation_json`` is ``None``, no masks are written (the loader emits zeros).
    """
    from .labelers.coco_labeler import CocoLabeler
    from .readers.cu3s_reader import Cu3sCubeReader

    cu3s_path = Path(cu3s_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labeler = CocoLabeler(annotation_json) if annotation_json is not None else None

    records: list[dict[str, Any]] = []
    with Cu3sCubeReader(str(cu3s_path), processing_mode=processing_mode) as reader:
        total = int(reader.total_measurements)
        indices = list(frame_indices) if frame_indices is not None else list(range(total))
        for i in indices:
            i = int(i)
            item = reader.read(i)
            cube_full = np.asarray(item["cube"], dtype=np.float32)  # [H, W, C]
            wavelengths = np.asarray(item["wavelengths"]).ravel().astype(np.int32, copy=False)
            payload: dict[str, Any] = {
                "cube": apply_crop(cube_full, crop),
                "wavelengths": wavelengths,
                "source_cu3s": str(cu3s_path),
            }
            if labeler is not None:
                # Rasterize at full cube size, then crop to match the cube.
                cat_full = np.asarray(labeler.load_for(i, {"cube": cube_full})["mask"])
                mask, class_mask = derive_masks(apply_crop(cat_full, crop))
                payload["mask"] = mask
                payload["class_mask"] = class_mask
            out = out_dir / f"{cu3s_path.stem}_{i:06d}.npz"
            (np.savez_compressed if compress else np.savez)(out, **payload)
            records.append({"npz_path": str(out), "source_cu3s": str(cu3s_path), "image_id": i})
    logger.info("converted {} -> {} frame(s) into {}", cu3s_path.name, len(records), out_dir)
    return records


def _resolve_annotation(
    cu3s_path: Path, annotations: str | Path | dict[str, str | Path] | None
) -> str | Path | None:
    """Resolve the COCO json for one cu3s.

    ``annotations`` may be: ``None`` (no masks); a str/Path (one shared COCO for every cu3s);
    a mapping ``{cu3s_path: json}``; or the literal ``"sibling"`` (``<stem>.json`` next to the
    cu3s, if it exists).
    """
    if annotations is None:
        return None
    if annotations == "sibling":
        sib = cu3s_path.with_suffix(".json")
        return sib if sib.exists() else None
    if isinstance(annotations, dict):
        return annotations.get(str(cu3s_path)) or annotations.get(cu3s_path.name)
    return annotations  # a single shared COCO path


def convert_cu3s(
    cu3s_paths: Sequence[str | Path],
    out_dir: str | Path,
    *,
    annotations: Any = "sibling",
    crop: CropMargins | None = None,
    processing_mode: str | None = "Reflectance",
    index_csv: str | Path | None = None,
    compress: bool = True,
    frame_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Convert many ``.cu3s`` files; optionally write a combined index CSV. Returns all records.

    ``frame_limit`` (if set) converts only the first N frames of each file (for smoke runs).
    """
    frame_indices = list(range(frame_limit)) if frame_limit else None
    all_records: list[dict[str, Any]] = []
    for p in cu3s_paths:
        p = Path(p)
        all_records.extend(
            convert_cu3s_file(
                p,
                out_dir,
                annotation_json=_resolve_annotation(p, annotations),
                crop=crop,
                processing_mode=processing_mode,
                frame_indices=frame_indices,
                compress=compress,
            )
        )
    if index_csv is not None:
        write_index_csv(all_records, index_csv)
    return all_records


def write_index_csv(records: list[dict[str, Any]], path: str | Path) -> None:
    """Write the traceability index (``npz_path, source_cu3s, image_id``). No split column."""
    fields = ["npz_path", "source_cu3s", "image_id"]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in records)
