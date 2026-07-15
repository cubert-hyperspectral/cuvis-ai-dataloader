"""Convert ``.cu3s`` sessions into per-frame ``.npz`` files (for the ``npz_multi`` loader).

For each measurement of a ``.cu3s`` this reads the cube (Preview -> Reflectance via the cu3s
reader), optionally rasterizes that frame's COCO annotations into a binary ``mask`` +
multi-class ``class_mask`` (via the COCO labeler), optionally crops, and writes one compressed
``.npz`` per frame:

* ``cube``        : ``[H, W, C]`` float32
* ``wavelengths`` : ``[C]`` int32
* ``mask``        : ``[H, W]`` int32 binary GT      (only when annotations are given)
* ``class_mask``  : ``[H, W]`` uint8 category id    (only when annotations are given; 0 = bg)
* ``source``      : originating ``.cu3s`` path (identity / traceability)

These load directly via :class:`~cuvis_ai_dataloader.data.datamodule_npz_multi.MultiNpzDataModule`.

**No train/val/test split is assigned here** — splitting is a separate concern. The converter
only emits a small universe (``source, index, path``) so a frame can be traced back to its
source session; a split CSV is produced elsewhere and joined on that. A frame's COCO
``image_id`` is its cu3s measurement index (use one COCO per cu3s, e.g. the per-session json).
"""

from __future__ import annotations

import csv
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from loguru import logger


class SplitManifestOutputs(NamedTuple):
    """The two artifacts :func:`convert_split_manifest` writes.

    ``splits_json`` is the selector-based assignment (core ``DataSplitConfig``); ``universe_csv``
    is the universe lookup (``source, index, path``) the npz_multi selector path reads.
    """

    splits_json: Path
    universe_csv: Path


def _posix(path: str | Path) -> str:
    """Normalize a source-identity string to posix so cross-platform selectors match."""
    return str(path).replace("\\", "/")


def _npz_is_valid(path: Path, *, need_masks: bool) -> bool:
    """True if ``path`` is a readable npz carrying the keys a converted frame must have.

    Used by the resume path: a bare ``exists()`` would keep a truncated or half-written npz,
    so we open it and confirm ``cube`` + ``wavelengths`` (and the masks when annotations were
    provided) are present before skipping reconversion.
    """
    if not path.is_file():
        return False
    required = ("cube", "wavelengths", *(("mask", "class_mask") if need_masks else ()))
    try:
        with np.load(path) as z:
            return all(key in z.files for key in required)
    except Exception:
        return False


# (top, bottom, left, right) pixel margins removed from the first two axes.
CropMargins = tuple[int, int, int, int]


def derive_masks(category_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a per-pixel category-id mask into ``(mask, class_mask)``.

    ``mask``       : ``[H, W]`` int32 binary (1 where any category, 0 background).
    ``class_mask`` : ``[H, W]`` uint8 category id (0 = background), as produced by the labeler.
    """
    cat = np.asarray(category_mask)
    if cat.size and int(cat.max()) > 255:
        raise ValueError(
            f"category id {int(cat.max())} exceeds uint8 range (0-255); class_mask cannot hold it"
        )
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
    frame_limit: int | None = None,
    compress: bool = True,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Convert one ``.cu3s`` to per-frame ``.npz`` files; return index records (no split).

    Masks are rasterized at the full cube size, then the same ``crop`` is applied to cube and
    masks together (so polygon coordinates stay aligned). A frame's COCO ``image_id`` is its cu3s
    measurement index; frames absent/unannotated in the COCO get an all-zero mask (normal frame).
    When ``annotation_json`` is ``None`` no masks are written (the loader emits zeros).

    ``frame_indices`` selects specific measurements (validated against the cu3s length);
    ``frame_limit`` converts only the first N (clamped to the length). Output names are prefixed
    with the cu3s's parent-folder name, so the same session name recurring across day folders
    does not overwrite (the caller's flat ``out_dir`` stays collision-free).

    ``resume=True`` skips a frame whose ``.npz`` already exists AND carries the expected keys
    (a cheap integrity check, not bare existence), so regenerating an index/splits over an
    already-converted set costs seconds instead of a full reconversion. When ``frame_indices``
    is given and every target is already valid, the cu3s reader is never opened, so regenerating
    the artifacts needs neither the SDK nor the raw cu3s present.
    """
    cu3s_path = Path(cu3s_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    need_masks = annotation_json is not None
    prefix = (
        f"{cu3s_path.parent.name}_{cu3s_path.stem}" if cu3s_path.parent.name else cu3s_path.stem
    )

    def _record(i: int, out: Path) -> dict[str, Any]:
        return {"path": str(out), "source": str(cu3s_path), "index": i}

    def _out(i: int) -> Path:
        return out_dir / f"{prefix}_{i:06d}.npz"

    # Fast path: an explicit frame set that is already fully converted needs no cu3s read at all.
    if resume and frame_indices is not None:
        targets = [int(i) for i in frame_indices]
        if all(_npz_is_valid(_out(i), need_masks=need_masks) for i in targets):
            logger.info(
                "converted {} -> {} frame(s) into {} (all reused, cu3s not opened)",
                cu3s_path.name,
                len(targets),
                out_dir,
            )
            return [_record(i, _out(i)) for i in targets]

    from .labelers.coco_labeler import CocoLabeler
    from .readers.cu3s_reader import Cu3sCubeReader

    labeler = CocoLabeler(annotation_json) if annotation_json is not None else None
    records: list[dict[str, Any]] = []
    with Cu3sCubeReader(str(cu3s_path), processing_mode=processing_mode) as reader:
        total = int(reader.total_measurements)
        if frame_indices is not None:
            bad = [int(i) for i in frame_indices if not (0 <= int(i) < total)]
            if bad:
                raise ValueError(
                    f"frame_indices {bad} out of range [0, {total}) for {cu3s_path.name}"
                )
            indices = [int(i) for i in frame_indices]
        elif frame_limit:
            indices = list(range(min(int(frame_limit), total)))
        else:
            indices = list(range(total))
        skipped = 0
        for i in indices:
            out = _out(i)
            if resume and _npz_is_valid(out, need_masks=need_masks):
                records.append(_record(i, out))
                skipped += 1
                continue
            item = reader.read(i)
            cube_full = np.asarray(item["cube"], dtype=np.float32)  # [H, W, C]
            wavelengths = np.asarray(item["wavelengths"]).ravel().astype(np.int32, copy=False)
            payload: dict[str, Any] = {
                "cube": apply_crop(cube_full, crop),
                "wavelengths": wavelengths,
                "source": str(cu3s_path),
            }
            if labeler is not None:
                # Rasterize at full cube size (looked up by COCO image_id == measurement index), then crop.
                cat_full = np.asarray(labeler.load_for(i, {"cube": cube_full})["mask"])
                mask, class_mask = derive_masks(apply_crop(cat_full, crop))
                payload["mask"] = mask
                payload["class_mask"] = class_mask
            (np.savez_compressed if compress else np.savez)(out, **payload)
            records.append(_record(i, out))
    logger.info(
        "converted {} -> {} frame(s) into {} ({} reused)",
        cu3s_path.name,
        len(records),
        out_dir,
        skipped,
    )
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
    universe_csv: str | Path | None = None,
    compress: bool = True,
    frame_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Convert many ``.cu3s`` files; optionally write a combined universe CSV. Returns all records.

    ``frame_limit`` (if set) converts only the first N frames of each file (for smoke runs).
    """
    # Output names are {parent}_{stem}_{i}.npz in one flat out_dir; two inputs sharing both
    # parent-folder and stem would overwrite each other, so reject that up front.
    keys = [(Path(p).parent.name, Path(p).stem) for p in cu3s_paths]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise ValueError(
            f"cu3s inputs collide on (parent, stem) and would overwrite each other in {out_dir}: {dupes}"
        )
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
                frame_limit=frame_limit,
                compress=compress,
            )
        )
    if universe_csv is not None:
        write_universe_csv(all_records, universe_csv)
    return all_records


def write_universe_csv(records: list[dict[str, Any]], path: str | Path) -> None:
    """Write the universe (``source, index, path``). No split column."""
    fields = ["source", "index", "path"]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in fields} for r in records)


def convert_split_manifest(
    manifest_csv: str | Path,
    source_root: str | Path,
    out_dir: str | Path,
    *,
    universe_csv: str | Path | None = None,
    splits_json: str | Path | None = None,
    crop: CropMargins | None = None,
    processing_mode: str | None = "Reflectance",
    compress: bool = True,
    resume: bool = True,
    limit: int = 0,
    splits: Sequence[str] = ("train", "val", "test"),
    predict_from: str = "test",
) -> SplitManifestOutputs:
    """Convert a split manifest's frames; emit a ``universe.csv`` + a ``splits.json``.

    The manifest is a CSV carrying at least ``cu3s_path``, ``local_image_id`` and ``split``
    (plus an optional per-session ``json_path``), with paths relative to ``source_root`` — the
    shape datasets publish next to their cu3s sessions (e.g. the lentils foreign-object
    dataset's ``splits_dinomaly.csv``). Rows whose ``split`` is not in ``splits`` are ignored;
    ``limit`` (when > 0) keeps at most that many rows per split for smoke runs. Each cu3s is
    opened once, with its group's ``local_image_id`` values as ``frame_indices``.

    Two artifacts are written (returned as a :class:`SplitManifestOutputs`):

    * ``universe.csv`` (``<out_dir>/universe.csv`` by default): the universe lookup
      ``source, index, path`` with ``source`` the manifest-relative posix cu3s path.
    * ``splits.json`` (``<out_dir>/splits.json`` by default): a core ``DataSplitConfig`` with,
      per split per source, one ``file_indices`` selector over the read indices. ``predict``
      copies the ``predict_from`` split's selectors (empty ``predict`` would otherwise resolve
      to the whole universe). The same ``splits.json`` resolves against the raw cu3s data.

    ``resume=True`` (default) skips already-converted, still-valid npz so regenerating the
    artifacts over an existing set is fast.
    """
    from cuvis_ai_core.data.splits_io import save_splits
    from cuvis_ai_schemas.training.data import DataSplitConfig, Selector, SelectorKind

    source_root = Path(source_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    universe_path = Path(universe_csv) if universe_csv is not None else out_dir / "universe.csv"
    splits_path = Path(splits_json) if splits_json is not None else out_dir / "splits.json"
    universe_dir = universe_path.resolve().parent  # path is stored relative to this

    with Path(manifest_csv).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"cu3s_path", "local_image_id", "split"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"split manifest {manifest_csv} lacks column(s): {sorted(missing)}")
        rows = [r for r in reader if r.get("split") in splits]

    if limit > 0:
        counts: dict[str, int] = {}
        kept: list[dict[str, str]] = []
        for r in rows:
            split = r["split"]
            if counts.get(split, 0) < limit:
                kept.append(r)
                counts[split] = counts.get(split, 0) + 1
        rows = kept

    groups: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        groups.setdefault(r["cu3s_path"], []).append(r)

    universe_records: list[dict[str, Any]] = []
    # split -> source(posix) -> sorted read indices, for the file_indices selectors.
    per_split: dict[str, dict[str, list[int]]] = {}
    for cu3s_rel, grp in groups.items():
        cu3s_path = source_root / cu3s_rel
        if not cu3s_path.is_file():
            raise FileNotFoundError(f"manifest references missing cu3s: {cu3s_path}")
        json_rel = (grp[0].get("json_path") or "").strip()
        annotation = source_root / json_rel if json_rel else None
        if annotation is not None and not annotation.is_file():
            raise FileNotFoundError(f"manifest references missing COCO json: {annotation}")
        source_id = _posix(cu3s_rel)  # identity used by the splits.json selectors
        records = convert_cu3s_file(
            cu3s_path,
            out_dir,
            annotation_json=annotation,
            crop=crop,
            processing_mode=processing_mode,
            frame_indices=[int(r["local_image_id"]) for r in grp],
            compress=compress,
            resume=resume,
        )
        for rec, row in zip(records, grp, strict=True):
            index = int(rec["index"])
            # Store path relative to the universe.csv dir (posix) so the artifact is
            # cwd-independent and survives moving the whole output folder.
            npz_rel = _posix(os.path.relpath(Path(rec["path"]).resolve(), universe_dir))
            universe_records.append({"source": source_id, "index": index, "path": npz_rel})
            per_split.setdefault(row["split"], {}).setdefault(source_id, []).append(index)

    def _selectors(split: str) -> list[Selector]:
        return [
            Selector(kind=SelectorKind.FILE_INDICES, source=src, ids=sorted(ids))
            for src, ids in sorted(per_split.get(split, {}).items())
        ]

    split_cfg = DataSplitConfig(
        train=_selectors("train"),
        val=_selectors("val"),
        test=_selectors("test"),
        predict=_selectors(predict_from),  # empty predict -> whole universe, so mirror a split
    )

    write_universe_csv(universe_records, universe_path)
    save_splits(split_cfg, splits_path)

    _validate_universe_and_splits(universe_records, split_cfg)
    logger.info(
        "convert_split_manifest: {} frame(s), {} session(s) -> {} + {}",
        len(universe_records),
        len(groups),
        universe_path,
        splits_path,
    )
    return SplitManifestOutputs(splits_json=splits_path, universe_csv=universe_path)


def _validate_universe_and_splits(universe_records: list[dict[str, Any]], split_cfg: Any) -> None:
    """Fail loud if the emitted splits.json cannot resolve cleanly against the universe.

    Guards the two consistency failures a downstream run would otherwise hit late: a
    duplicate ``(source, index)`` identity in the universe, and a selector id that no
    universe frame provides.
    """
    universe: set[tuple[str, int]] = set()
    for rec in universe_records:
        identity = (rec["source"], int(rec["index"]))
        if identity in universe:
            raise ValueError(f"universe has duplicate identity {identity}")
        universe.add(identity)
    for stage in (split_cfg.train, split_cfg.val, split_cfg.test, split_cfg.predict):
        for sel in stage:
            for i in sel.ids:
                if (sel.source, int(i)) not in universe:
                    raise ValueError(
                        f"splits selector references ({sel.source}, {i}) absent from the universe"
                    )
