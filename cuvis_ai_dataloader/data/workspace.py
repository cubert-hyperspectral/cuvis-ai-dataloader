"""Workspace + splits sidecar parsing for the ``cu3s_workspace`` DataModule.

A *workspace* is a folder whose root holds ``workspace.json`` (explicit, file-level
membership: absolute ``.cu3s`` paths plus bookkeeping such as ``scan_roots`` and
``excluded``) and ``splits.json`` (per-file frame-id lists produced by the split
resolver). Measurements may live anywhere on disk; membership is the list, never a
directory scan at load time.

Everything here is pure stdlib (json / pathlib / hashlib / dataclasses) so it can be
imported by ``validate_params`` without touching the heavy cu3s/COCO extras.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

WORKSPACE_FILENAME = "workspace.json"
SPLITS_FILENAME = "splits.json"
_SPLIT_KEYS = ("train", "val", "test", "predict")


def canonical(path: str | Path) -> str:
    """One identity per file: absolute, symlink-resolved, OS-normalized."""
    return os.path.normcase(str(Path(path).resolve()))


def sha256_of_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def coco_sibling(cu3s_path: str | Path) -> Path | None:
    """The measurement's annotation sidecar: ``<stem>.json`` next to the ``.cu3s``."""
    sibling = Path(cu3s_path).with_suffix(".json")
    return sibling if sibling.is_file() else None


@dataclass
class Measurement:
    """One member file. ``frames`` is cached display metadata (counting requires
    opening the multi-GB session file and is immutable for a recording); whether a
    measurement is annotated is *never* stored — derive it live via
    :func:`coco_sibling`."""

    path: str
    added_at: str | None = None
    frames: int | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Measurement":
        if not isinstance(d, dict) or not d.get("path"):
            raise ValueError(f"workspace measurement entries need a 'path': got {d!r}")
        return cls(
            path=str(d["path"]),
            added_at=d.get("added_at"),
            frames=int(d["frames"]) if d.get("frames") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"path": self.path}
        if self.added_at is not None:
            out["added_at"] = self.added_at
        if self.frames is not None:
            out["frames"] = self.frames
        return out


@dataclass
class Workspace:
    """Parsed ``workspace.json``. ``root`` is the folder that holds the file."""

    root: Path
    name: str = ""
    task_type: str = "anomaly"
    default_processing_mode: str = "Reflectance"
    default_seed: int = 42
    frame_grouping: dict[str, Any] | None = None
    measurements: list[Measurement] = field(default_factory=list)
    scan_roots: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    version: int = 1

    @classmethod
    def load(cls, workspace_path: str | Path) -> "Workspace":
        """Load from a workspace folder (or a direct path to ``workspace.json``).

        The ``workspace.json`` file is the qualifier: a folder without one is not a
        workspace, and a corrupt file is a hard error (never silently re-created).
        """
        p = Path(workspace_path)
        file = p if p.is_file() else p / WORKSPACE_FILENAME
        if not file.is_file():
            raise ValueError(
                f"not a workspace: {p} (no {WORKSPACE_FILENAME}; create one via the "
                f"UI or initialize the folder explicitly)"
            )
        try:
            raw = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"corrupt {file}: {exc}") from exc
        version = int(raw.get("version", 0))
        if version != 1:
            raise ValueError(f"{file}: unsupported workspace version {version!r} (expected 1)")
        measurements = [Measurement.from_dict(m) for m in raw.get("measurements", [])]
        return cls(
            root=file.parent,
            name=str(raw.get("name", "")),
            task_type=str(raw.get("task_type", "anomaly")),
            default_processing_mode=str(raw.get("default_processing_mode", "Reflectance")),
            default_seed=int(raw.get("default_seed", 42)),
            frame_grouping=raw.get("frame_grouping"),
            measurements=measurements,
            scan_roots=[str(s) for s in raw.get("scan_roots", [])],
            excluded=[str(s) for s in raw.get("excluded", [])],
            version=version,
        )

    # -- membership --------------------------------------------------------------
    def member_measurements(self) -> list[Measurement]:
        """Members deduped by canonical path (first entry wins, order preserved)."""
        seen: set[str] = set()
        out: list[Measurement] = []
        for m in self.measurements:
            key = canonical(m.path)
            if key in seen:
                continue
            seen.add(key)
            out.append(m)
        return out

    def member_paths(self) -> list[str]:
        return [m.path for m in self.member_measurements()]

    def duplicate_paths(self) -> list[str]:
        """Raw entries dropped by dedupe (for a one-line warning, listed once each)."""
        seen: set[str] = set()
        dupes: list[str] = []
        for m in self.measurements:
            key = canonical(m.path)
            if key in seen and m.path not in dupes:
                dupes.append(m.path)
            seen.add(key)
        return dupes

    def missing_paths(self) -> list[str]:
        """Member paths that do not exist on disk — callers must NAME these in errors
        (and the UI shows them greyed with Locate/Remove), never silently skip."""
        return [m.path for m in self.member_measurements() if not Path(m.path).is_file()]

    def frames_by_path(self) -> dict[str, int]:
        """Cached frame counts keyed by canonical path (only entries that carry one)."""
        return {
            canonical(m.path): m.frames for m in self.member_measurements() if m.frames is not None
        }

    # -- scan roots (the "new measurements found" banner) -------------------------
    def new_files_under_scan_roots(self, list_files: Any | None = None) -> list[str]:
        """``(cu3s under scan_roots) - members - excluded``, sorted.

        ``list_files(root) -> Iterable[str|Path]`` is injectable for tests; the
        default recursively globs ``*.cu3s``. Files declined at import time live in
        ``excluded`` and are never re-offered; banner "Dismiss" intentionally does
        NOT add to ``excluded`` (defer, not decline).
        """

        def _default_list(root: str) -> Iterable[Path]:
            base = Path(root)
            return base.rglob("*.cu3s") if base.is_dir() else ()

        lister = list_files or _default_list
        members = {canonical(p) for p in self.member_paths()}
        excluded = {canonical(p) for p in self.excluded}
        found: dict[str, str] = {}
        for root in self.scan_roots:
            for f in lister(root):
                key = canonical(f)
                if key not in members and key not in excluded and key not in found:
                    found[key] = str(f)
        return sorted(found.values())


@dataclass
class SplitsFile:
    """Parsed ``splits.json``: per-file selector lists + annotation staleness hashes.

    ``files`` maps the member path string (verbatim from ``workspace.json``) to
    ``{train|val|test|predict: list[int|str]}``; selectors follow ``DataSplitConfig``
    semantics (ints or range strings, expanded downstream by the base datamodule
    machinery). ``predict`` is a first-class list — the resolver defaults it to the
    test ids rather than overloading ``test``.
    """

    files: dict[str, dict[str, list[int | str]]] = field(default_factory=dict)
    coco_hash_per_file: dict[str, str | None] = field(default_factory=dict)
    version: int = 1

    @classmethod
    def load(cls, workspace_root: str | Path) -> "SplitsFile":
        p = Path(workspace_root)
        file = p if p.is_file() else p / SPLITS_FILENAME
        if not file.is_file():
            raise ValueError(f"no {SPLITS_FILENAME} in {p}; run the split resolver first")
        try:
            raw = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"corrupt {file}: {exc}") from exc
        version = int(raw.get("version", 0))
        if version != 1:
            raise ValueError(f"{file}: unsupported splits version {version!r} (expected 1)")
        files: dict[str, dict[str, list[int | str]]] = {}
        for path, lists in (raw.get("files") or {}).items():
            if not isinstance(lists, dict):
                raise ValueError(f"{file}: entry for {path!r} must be a mapping of split lists")
            entry = {k: list(lists.get(k, [])) for k in _SPLIT_KEYS}
            unknown = set(lists) - set(_SPLIT_KEYS)
            if unknown:
                raise ValueError(f"{file}: {path!r} has unknown split keys {sorted(unknown)}")
            files[path] = entry
        return cls(
            files=files,
            coco_hash_per_file=dict(raw.get("coco_hash_per_file") or {}),
            version=version,
        )

    def save(self, workspace_root: str | Path) -> Path:
        """Atomic write (temp + replace) — the UI and server both touch this file."""
        target = Path(workspace_root) / SPLITS_FILENAME
        payload = {
            "version": self.version,
            "coco_hash_per_file": self.coco_hash_per_file,
            "files": self.files,
        }
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, target)
        return target

    # -- validation ----------------------------------------------------------------
    def validate_against(self, workspace: Workspace) -> None:
        """Structural checks against the membership; raises with NAMED offenders.

        Frame-count range checks use the workspace's cached ``frames`` where present
        (cheap); the authoritative bound is re-checked when the dataset actually
        reads, since the cache could in principle be stale.
        """
        members = {canonical(p) for p in workspace.member_paths()}
        frames = workspace.frames_by_path()
        problems: list[str] = []
        for path, lists in self.files.items():
            key = canonical(path)
            if key not in members:
                problems.append(f"{path}: in splits.json but not a workspace member")
                continue
            n = frames.get(key)
            if n is None:
                continue
            for split_name in _SPLIT_KEYS:
                bad = [
                    s for s in lists.get(split_name, []) if isinstance(s, int) and not 0 <= s < n
                ]
                if bad:
                    problems.append(f"{path}: {split_name} ids {bad} out of range for {n} frames")
        if problems:
            raise ValueError("invalid splits.json:\n  " + "\n  ".join(problems))


__all__ = [
    "Measurement",
    "SplitsFile",
    "Workspace",
    "WORKSPACE_FILENAME",
    "SPLITS_FILENAME",
    "canonical",
    "coco_sibling",
    "sha256_of_file",
]
