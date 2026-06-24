"""Anomaly-aware split resolver: workspace membership -> ``splits.json``.

Strategies (both AD-aware — the train split receives ONLY normal frames, since a
reconstruction-based anomaly model must never see defects in training; anomalous
frames are distributed across val/test):

- ``random``     — per file: normal groups shuffle into train/val/test by ratio;
                   anomalous groups shuffle into val/test by the val:test share.
- ``stratified`` — pooled across the selected files: groups bucket by the union of
                   their anomaly category ids ("none" = normal); the normal bucket
                   fills train/val/test, every anomaly bucket fills val/test, so
                   each defect signature lands proportionally in both eval splits.

A frame is *anomalous* iff its COCO sibling carries an effective non-normal
annotation (``CocoLabeler.is_anomalous``); files without a sibling are all-normal
(the typical anomaly "good parts" recordings). ``frame_grouping`` (e.g. lentils:
4 consecutive frames = one scene under different lighting) keeps whole groups in
one split so near-duplicate frames never leak across train/eval.

The resolver is deterministic for a given seed and writes ``splits.json``
atomically with per-file annotation hashes for staleness detection. ``predict``
is emitted as a first-class list (defaulting to the test ids) rather than
overloading ``test``.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .workspace import (
    Measurement,
    SplitsFile,
    Workspace,
    canonical,
    coco_sibling,
    sha256_of_file,
)

_STRATEGIES = ("random", "stratified")


@dataclass
class _Group:
    """The atomic unit of split assignment: one or more consecutive frames."""

    file: str
    frames: list[int]
    anomaly_cats: frozenset[int]

    @property
    def is_anomalous(self) -> bool:
        return bool(self.anomaly_cats)

    @property
    def strat_key(self) -> str:
        return "_".join(str(c) for c in sorted(self.anomaly_cats)) or "none"


def _default_labeler_factory(annotation_json_path: str) -> Any:
    from .labelers.coco_labeler import CocoLabeler

    return CocoLabeler(annotation_json_path=annotation_json_path)


def _default_frame_counter(cu3s_path: str) -> int:
    from ._extras import require_cuvis

    cuvis = require_cuvis()
    return len(cuvis.SessionFile(cu3s_path))


def _ratio_slices(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    """(n_train, n_val) for a bucket of ``n``; the remainder is test."""
    n_train = min(n, round(n * train_ratio))
    n_val = min(n - n_train, round(n * val_ratio))
    return n_train, n_val


def _val_share_slice(n: int, val_ratio: float, train_ratio: float) -> int:
    """How many of ``n`` anomalous groups go to val (rest to test)."""
    eval_ratio = 1.0 - train_ratio
    if eval_ratio <= 0:
        return 0
    return min(n, round(n * (val_ratio / eval_ratio)))


def _group_frames(
    n_frames: int, anomalous_cats_by_frame: dict[int, frozenset[int]], group_size: int, file: str
) -> list[_Group]:
    groups: list[_Group] = []
    for start in range(0, n_frames, group_size):
        frames = list(range(start, min(start + group_size, n_frames)))
        cats: set[int] = set()
        for f in frames:
            cats |= anomalous_cats_by_frame.get(f, frozenset())
        groups.append(_Group(file=file, frames=frames, anomaly_cats=frozenset(cats)))
    return groups


def resolve_splits(
    workspace: Workspace | str | Path,
    *,
    strategy: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int | None = None,
    selected_files: list[str] | None = None,
    frame_counts: dict[str, int] | None = None,
    labeler_factory: Callable[[str], Any] | None = None,
    write: bool = True,
) -> SplitsFile:
    """Compute (and by default write) ``splits.json`` for a workspace.

    ``selected_files`` restricts resolution to a subset of members (paths as they
    appear in ``workspace.json``); ``frame_counts`` (canonical path -> n) and
    ``labeler_factory`` are injectable for tests — by default counts come from the
    workspace's cached ``frames`` metadata and only fall back to opening the
    session file when uncached.
    """
    ws = workspace if isinstance(workspace, Workspace) else Workspace.load(workspace)
    if strategy not in _STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {_STRATEGIES}")
    if not 0 < train_ratio < 1 or val_ratio < 0 or train_ratio + val_ratio > 1:
        raise ValueError(
            f"bad ratios: train={train_ratio}, val={val_ratio} "
            f"(need 0<train<1, val>=0, train+val<=1; test is the remainder)"
        )

    members = ws.member_measurements()
    if not members:
        raise ValueError(f"workspace {ws.root} has no measurements; add files before splitting")
    missing = ws.missing_paths()
    if missing:
        raise ValueError(
            "cannot resolve splits, missing measurement file(s):\n  " + "\n  ".join(missing)
        )
    members = _select(members, selected_files)

    seed = ws.default_seed if seed is None else seed
    group_size = _group_size(ws.frame_grouping)
    counts = dict(frame_counts or {})
    make_labeler = labeler_factory or _default_labeler_factory

    # -- classify every frame of every selected file -------------------------------
    groups_by_file: dict[str, list[_Group]] = {}
    coco_hashes: dict[str, str | None] = {}
    for m in members:
        key = canonical(m.path)
        n = counts.get(key, m.frames)
        if n is None:
            n = _default_frame_counter(m.path)
        sibling = coco_sibling(m.path)
        cats_by_frame: dict[int, frozenset[int]] = {}
        if sibling is not None:
            labeler = make_labeler(str(sibling))
            cats_by_frame = {
                f: labeler.anomaly_category_ids(f) for f in range(int(n)) if labeler.is_anomalous(f)
            }
            coco_hashes[m.path] = sha256_of_file(sibling)
        else:
            coco_hashes[m.path] = None
        groups_by_file[m.path] = _group_frames(int(n), cats_by_frame, group_size, m.path)

    total_anomalous = sum(1 for groups in groups_by_file.values() for g in groups if g.is_anomalous)
    if total_anomalous == 0:
        warnings.warn(
            "workspace has no anomalous frames: val/test will contain only normal "
            "frames, so anomaly metrics (AUROC/F1) are undefined for this split.",
            UserWarning,
            stacklevel=2,
        )

    rng = random.Random(seed)
    assignment: dict[str, dict[str, list[int]]] = {
        m.path: {"train": [], "val": [], "test": []} for m in members
    }

    if strategy == "random":
        for m in members:  # member order: deterministic
            groups = groups_by_file[m.path]
            _assign_bucket(
                [g for g in groups if not g.is_anomalous],
                assignment,
                rng,
                train_ratio,
                val_ratio,
                with_train=True,
            )
            _assign_bucket(
                [g for g in groups if g.is_anomalous],
                assignment,
                rng,
                train_ratio,
                val_ratio,
                with_train=False,
            )
    else:  # stratified: pool groups across files, bucket by anomaly signature
        buckets: dict[str, list[_Group]] = {}
        for m in members:
            for g in groups_by_file[m.path]:
                buckets.setdefault(g.strat_key, []).append(g)
        for key in sorted(buckets):
            _assign_bucket(
                buckets[key],
                assignment,
                rng,
                train_ratio,
                val_ratio,
                with_train=(key == "none"),
            )

    files: dict[str, dict[str, list[int | str]]] = {}
    for m in members:
        lists = assignment[m.path]
        files[m.path] = {
            "train": sorted(lists["train"]),
            "val": sorted(lists["val"]),
            "test": sorted(lists["test"]),
            "predict": sorted(lists["test"]),  # first-class predict, defaults to test
        }

    result = SplitsFile(files=files, coco_hash_per_file=coco_hashes)
    result.validate_against(ws)
    if write:
        result.save(ws.root)
    return result


def _assign_bucket(
    groups: list[_Group],
    assignment: dict[str, dict[str, list[int]]],
    rng: random.Random,
    train_ratio: float,
    val_ratio: float,
    *,
    with_train: bool,
) -> None:
    """Shuffle one bucket and deal its groups into splits (whole groups only)."""
    pool = list(groups)
    rng.shuffle(pool)
    if with_train:
        n_train, n_val = _ratio_slices(len(pool), train_ratio, val_ratio)
        slices = (
            ("train", pool[:n_train]),
            ("val", pool[n_train : n_train + n_val]),
            ("test", pool[n_train + n_val :]),
        )
    else:
        n_val = _val_share_slice(len(pool), val_ratio, train_ratio)
        slices = (("val", pool[:n_val]), ("test", pool[n_val:]))
    for split_name, chunk in slices:
        for g in chunk:
            assignment[g.file][split_name].extend(g.frames)


def _select(members: list[Measurement], selected_files: list[str] | None) -> list[Measurement]:
    if selected_files is None:
        return members
    by_key = {canonical(m.path): m for m in members}
    out: list[Measurement] = []
    unknown: list[str] = []
    for f in selected_files:
        m = by_key.get(canonical(f))
        (out.append(m) if m is not None else unknown.append(f))
    if unknown:
        raise ValueError("selected file(s) are not workspace members:\n  " + "\n  ".join(unknown))
    if not out:
        raise ValueError("selected_files is empty; select at least one measurement")
    return out


def _group_size(frame_grouping: dict[str, Any] | None) -> int:
    if not frame_grouping:
        return 1
    size = int(frame_grouping.get("group_size", 1))
    if size < 1:
        raise ValueError(f"frame_grouping.group_size must be >= 1, got {size}")
    return size


__all__ = ["resolve_splits"]
