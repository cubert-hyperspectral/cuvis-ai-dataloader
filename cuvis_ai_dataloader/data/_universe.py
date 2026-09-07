"""Shared parser for the ``universe.csv`` manifest read by ``cu3s_multi`` and ``npz_multi``.

One column vocabulary, two readers. A ``universe.csv`` row is:

* ``source`` (required): the stable logical identity and the ``splits.json`` selector key.
  Posix-normalized at parse so a selector authored on one platform resolves a row authored on
  another. For raw cu3s it is the recording; for a converted npz it is the *original* cu3s path.
* ``index`` (required): read position within ``source`` (== COCO image_id / measurement index).
* ``materialized_path`` (optional): the physical file to open. Defaults to ``source`` (a raw
  ``.cu3s`` is its own file); a converted format (npz) supplies the derived ``.npz`` here.
  Resolved relative to the CSV directory; a ``..`` escape is rejected.
* ``split`` (optional): train/val/test, honored only by the module-owned path (``cu3s_multi``).
  A selector-driven module (``npz_multi``) rejects the column.
* ``annotation`` (optional): paired label path (a per-day COCO json), resolved like a path.
* ``format`` (optional): provenance/advisory only; each module has a fixed reader.
* ``group`` (optional): reserved leakage-grouping key, carried onto the sample.

This module imports nothing heavy (no cuvis SDK, no reader), so the core-only npz path can import
it without pulling the SDK.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from cuvis_ai_core.utils.general import expand_range_selectors

#: Columns every ``universe.csv`` must carry.
REQUIRED_COLUMNS = ("source", "index")
#: Columns that are parsed when present and ignored when absent.
OPTIONAL_COLUMNS = ("materialized_path", "split", "annotation", "format", "group")


def posix(path: str) -> str:
    """Normalize a path string to posix so cross-platform identities match."""
    return str(path).replace("\\", "/")


def _resolve(raw: str, csv_dir: Path) -> Path:
    """Resolve a relative path against the CSV dir; reject ``..`` escapes; absolutes pass through.

    Paths are stored relative to the CSV so the manifest is portable. A ``..`` component would let
    a row reach outside the CSV's own directory tree, defeating portability and opening a traversal
    footgun, so it is rejected.
    """
    p = Path(raw)
    if ".." in p.parts:
        raise ValueError(f"universe path must not contain '..': {raw!r}")
    return p if p.is_absolute() else (csv_dir / p).resolve()


def parse_universe(
    csv_path: Path,
    *,
    require_materialized_path: bool,
    accept_split: bool,
    unique_materialized_path: bool,
    allow_index_ranges: bool,
) -> list[dict[str, Any]]:
    """Parse a ``universe.csv`` into canonical row dicts.

    Each returned row carries ``frame_id`` (a stable global counter, unique across a ranged
    fan-out), ``source`` (posix identity), ``index``, ``materialized_path`` (resolved absolute),
    ``annotation`` (resolved absolute or ``""``), ``split`` (or ``""``), ``group`` (or ``None``),
    and ``format`` (or ``""``).

    The per-module flags capture the two readers' contracts: ``npz_multi`` requires a distinct
    ``materialized_path`` per row and rejects a ``split`` column; ``cu3s_multi`` defaults
    ``materialized_path`` to ``source``, allows a ``split`` column, allows several frames to share
    one recording, and expands a ranged ``index`` (``0-49``) into one row per measurement.
    """
    csv_path = Path(csv_path)
    csv_dir = csv_path.parent
    out: list[dict[str, Any]] = []
    seen_identity: set[tuple[str, int]] = set()
    seen_path: set[str] = set()
    frame_counter = 0
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in fields]
        if missing:
            raise ValueError(
                f"{csv_path}: missing required column(s) {missing}. "
                f"Required: {list(REQUIRED_COLUMNS)} (optional: {list(OPTIONAL_COLUMNS)}). "
                "Extra columns are allowed and ignored."
            )
        if "split" in fields and not accept_split:
            raise ValueError(
                f"{csv_path}: a 'split' column is not accepted here; this module is "
                "selector-driven (give it a splits.json). Only cu3s_multi honors an inline split."
            )
        for row in reader:
            source = posix(str(row["source"]).strip())
            mat_raw = (row.get("materialized_path") or "").strip()
            if not mat_raw:
                if require_materialized_path:
                    raise ValueError(
                        f"{csv_path}: source={source!r} needs a 'materialized_path'; this module "
                        "cannot read the source identity directly (it is not the physical file)."
                    )
                mat_raw = str(row["source"]).strip()
            resolved_path = str(_resolve(mat_raw, csv_dir))
            annotation = (row.get("annotation") or "").strip()
            annotation_resolved = str(_resolve(annotation, csv_dir)) if annotation else ""
            split = (row.get("split") or "").strip() if accept_split else ""
            group_raw = (row.get("group") or "").strip()
            fmt = (row.get("format") or "").strip()
            cell = str(row["index"]).strip()
            if allow_index_ranges and "-" in cell:
                indices = [int(x) for x in expand_range_selectors([cell])]
            else:
                indices = [int(cell)]
            for idx in indices:
                identity = (source, idx)
                if identity in seen_identity:
                    raise ValueError(
                        f"{csv_path}: duplicate identity (source, index)={identity}; "
                        "each (source, index) must be unique."
                    )
                seen_identity.add(identity)
                if unique_materialized_path:
                    if resolved_path in seen_path:
                        raise ValueError(
                            f"{csv_path}: duplicate materialized_path {resolved_path!r}; "
                            "each row must point at a distinct file."
                        )
                    seen_path.add(resolved_path)
                out.append(
                    {
                        "frame_id": frame_counter,
                        "source": source,
                        "index": idx,
                        "materialized_path": resolved_path,
                        "annotation": annotation_resolved,
                        "split": split,
                        "group": posix(group_raw) if group_raw else None,
                        "format": fmt,
                    }
                )
                frame_counter += 1
    if not out:
        raise ValueError(f"{csv_path}: no rows.")
    return out


def validate_universe_csv_param(params: dict[str, Any], module_name: str) -> None:
    """Shared ``validate_params`` check: ``universe_csv`` given, ends ``.csv``, and exists."""
    universe_csv = params.get("universe_csv")
    if not universe_csv:
        raise ValueError(f"{module_name} requires 'universe_csv' in params.")
    if not str(universe_csv).endswith(".csv"):
        raise ValueError(f"universe_csv must end with .csv: {universe_csv!r}")
    if not Path(universe_csv).is_file():
        raise ValueError(f"universe_csv does not exist: {universe_csv}")
