"""Build committable ``DataSplitConfig`` selector sets from a sample universe.

Used by the ``resolve-splits`` CLI. Turns a module's ``enumerate()`` universe (or a
cu3s_multi CSV) into a ``DataSplitConfig`` of per-source selectors that can be written to a
``splits.json``. Splitting is deterministic given a seed; ``group_by`` keeps a multi-sample
file whole; ``ad_aware`` keeps only normals (no category) in train.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from typing import TYPE_CHECKING

from cuvis_ai_schemas.training.data import DataSplitConfig, Selector, SelectorKind

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cuvis_ai_schemas.training.data import SampleRef


def selectors_from_refs(refs: list[SampleRef]) -> list[Selector]:
    """Group refs by source into selectors: FILE_INDICES (measurements) or FILES (whole-file)."""
    by_source: OrderedDict[str, list[SampleRef]] = OrderedDict()
    for ref in refs:
        by_source.setdefault(ref.source, []).append(ref)
    selectors: list[Selector] = []
    for source, group in by_source.items():
        indices = [r.index for r in group if r.index is not None]
        if len(indices) == len(group):
            selectors.append(
                Selector(kind=SelectorKind.FILE_INDICES, source=source, ids=sorted(set(indices)))
            )
        else:
            selectors.append(Selector(kind=SelectorKind.FILES, paths=[source]))
    return selectors


def _groups(refs: list[SampleRef], group_by: str | None) -> list[list[SampleRef]]:
    if group_by in ("source", "group"):
        groups: OrderedDict[str, list[SampleRef]] = OrderedDict()
        for ref in refs:
            key = ref.group if (group_by == "group" and ref.group) else ref.source
            groups.setdefault(key, []).append(ref)
        return list(groups.values())
    return [[ref] for ref in refs]


def resolve_random(
    refs: list[SampleRef],
    *,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 0,
    group_by: str | None = None,
    ad_aware: bool = False,
) -> DataSplitConfig:
    """Random split (seeded, reproducible). ``group_by`` keeps a file's samples together."""
    groups = _groups(refs, group_by)
    random.Random(seed).shuffle(groups)
    n = len(groups)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    test_groups = groups[:n_test]
    val_groups = groups[n_test : n_test + n_val]
    train_groups = groups[n_test + n_val :]

    def flat(grouped: list[list[SampleRef]]) -> list[SampleRef]:
        return [ref for group in grouped for ref in group]

    train, val, test = flat(train_groups), flat(val_groups), flat(test_groups)
    if ad_aware:
        # Anomaly detection: train on normals only (no annotated category).
        train = [ref for ref in train if not ref.category_ids]
    return _to_config(train, val, test)


def resolve_stratified(
    refs: list[SampleRef],
    *,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 0,
    group_by: str | None = None,
    ad_aware: bool = False,
) -> DataSplitConfig:
    """Split stratified by anomalous-vs-normal so each stage keeps the class balance."""
    train: list[SampleRef] = []
    val: list[SampleRef] = []
    test: list[SampleRef] = []
    normal = [r for r in refs if not r.category_ids]
    anomalous = [r for r in refs if r.category_ids]
    for subset in (normal, anomalous):
        if not subset:
            continue
        groups = _groups(subset, group_by)
        random.Random(seed).shuffle(groups)
        n = len(groups)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))

        def flat(grouped: list[list[SampleRef]]) -> list[SampleRef]:
            return [ref for group in grouped for ref in group]

        test += flat(groups[:n_test])
        val += flat(groups[n_test : n_test + n_val])
        train += flat(groups[n_test + n_val :])
    if ad_aware:
        train = [ref for ref in train if not ref.category_ids]
    return _to_config(train, val, test)


def import_csv_splits(module) -> DataSplitConfig:
    """Build a DataSplitConfig from a cu3s_multi module's CSV ``split`` column.

    Groups each stage's rows by source into FILE_INDICES selectors (sorted, deduped),
    outcome-equivalent to the CSV (same rows per stage), in canonical order.
    """
    stage_pairs: dict[str, list[tuple[str, int]]] = {"train": [], "val": [], "test": []}
    for rec in module._rows:  # noqa: SLF001 - same package, intentional
        split = rec["split"]
        if split in stage_pairs:
            stage_pairs[split].append((rec["cu3s_path"], int(rec["read_index"])))

    def selectors(pairs: list[tuple[str, int]]) -> list[Selector]:
        by_source: OrderedDict[str, list[int]] = OrderedDict()
        for source, idx in pairs:
            by_source.setdefault(source, []).append(idx)
        return [
            Selector(kind=SelectorKind.FILE_INDICES, source=source, ids=sorted(set(ids)))
            for source, ids in by_source.items()
        ]

    return DataSplitConfig(
        train=selectors(stage_pairs["train"]),
        val=selectors(stage_pairs["val"]),
        test=selectors(stage_pairs["test"]),
    )


def _to_config(
    train: list[SampleRef], val: list[SampleRef], test: list[SampleRef]
) -> DataSplitConfig:
    return DataSplitConfig(
        train=selectors_from_refs(train),
        val=selectors_from_refs(val),
        test=selectors_from_refs(test),
    )


__all__ = [
    "import_csv_splits",
    "resolve_random",
    "resolve_stratified",
    "selectors_from_refs",
]
