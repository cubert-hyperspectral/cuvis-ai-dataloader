"""Tests for the split resolvers + the resolve-splits CLI (no SDK needed)."""

from __future__ import annotations

import sys

import pytest

from cuvis_ai_core.data.splits_io import load_splits
from cuvis_ai_schemas.training.data import SampleRef, SelectorKind

from cuvis_ai_dataloader.data.resolvers import (
    resolve_random,
    resolve_stratified,
    selectors_from_refs,
)


def _refs(n, source="a.cu3s", anomalous=()):
    return [
        SampleRef(
            source=source,
            index=i,
            label_id=i,
            category_ids=([1] if i in anomalous else []),
        )
        for i in range(n)
    ]


def _ids_in(selectors, source):
    out: list[int] = []
    for sel in selectors:
        if sel.kind == SelectorKind.FILE_INDICES and sel.source == source:
            out.extend(sel.ids)
    return out


def test_selectors_from_refs_file_indices():
    sels = selectors_from_refs(_refs(3))
    assert len(sels) == 1
    assert sels[0].kind == SelectorKind.FILE_INDICES
    assert sels[0].ids == [0, 1, 2]


def test_selectors_from_refs_whole_file_for_indexless():
    refs = [
        SampleRef(source="a.tif", index=None, stem="a"),
        SampleRef(source="b.tif", index=None, stem="b"),
    ]
    sels = selectors_from_refs(refs)
    assert {s.kind for s in sels} == {SelectorKind.FILES}
    assert len(sels) == 2


def test_resolve_random_deterministic():
    refs = _refs(10)
    assert resolve_random(refs, seed=7).to_dict() == resolve_random(refs, seed=7).to_dict()


def test_resolve_random_ratios_partition_all():
    refs = _refs(10, source="a.cu3s")
    cfg = resolve_random(refs, val_ratio=0.2, test_ratio=0.2, seed=0)
    train = _ids_in(cfg.train, "a.cu3s")
    val = _ids_in(cfg.val, "a.cu3s")
    test = _ids_in(cfg.test, "a.cu3s")
    assert sorted(train + val + test) == list(range(10))  # partitions the universe
    assert len(test) == 2 and len(val) == 2 and len(train) == 6


def test_resolve_random_ad_aware_excludes_anomalous_from_train():
    refs = _refs(10, anomalous={0, 1, 2, 3, 4})
    cfg = resolve_random(refs, val_ratio=0.0, test_ratio=0.2, seed=1, ad_aware=True)
    train = _ids_in(cfg.train, "a.cu3s")
    assert all(i >= 5 for i in train)  # no anomalous (0..4) in train


def test_resolve_random_group_by_source_keeps_files_whole():
    refs = _refs(3, source="a.cu3s") + _refs(2, source="b.cu3s")
    cfg = resolve_random(refs, val_ratio=0.0, test_ratio=0.5, seed=0, group_by="source")
    train_sources = {s.source for s in cfg.train}
    test_sources = {s.source for s in cfg.test}
    assert train_sources.isdisjoint(test_sources)  # a source is wholly in one stage


def test_resolve_stratified_keeps_both_classes():
    refs = _refs(10, anomalous={0, 1, 2, 3, 4})
    cfg = resolve_stratified(refs, val_ratio=0.0, test_ratio=0.4, seed=0)
    train = _ids_in(cfg.train, "a.cu3s")
    # stratified: train keeps a mix of normal (5..9) and anomalous (0..4)
    assert any(i < 5 for i in train) and any(i >= 5 for i in train)


def test_resolve_stratified_group_by_keeps_mixed_file_whole():
    # Each file holds one normal + one anomalous sample. group_by="source" must keep a file
    # wholly in one stage even though stratification partitions by class.
    refs = [
        SampleRef(source="a.cu3s", index=0, label_id=0),
        SampleRef(source="a.cu3s", index=1, label_id=1, category_ids=[1]),
        SampleRef(source="b.cu3s", index=0, label_id=0),
        SampleRef(source="b.cu3s", index=1, label_id=1, category_ids=[1]),
    ]
    cfg = resolve_stratified(refs, val_ratio=0.0, test_ratio=0.5, seed=0, group_by="source")
    train_sources = {s.source for s in cfg.train}
    test_sources = {s.source for s in cfg.test}
    assert train_sources and test_sources
    assert train_sources.isdisjoint(test_sources)  # no file split across stages


def test_resolve_rejects_ambiguous_read_index():
    # One read index carrying two label_ids can't be addressed by a FILE_INDICES selector.
    refs = [
        SampleRef(source="x.cu3s", index=0, label_id=3),
        SampleRef(source="x.cu3s", index=0, label_id=7),
    ]
    with pytest.raises(ValueError, match="index-addressable"):
        resolve_random(refs)


def test_resolve_splits_cli_from_csv(tmp_path, monkeypatch):
    (tmp_path / "a.cu3s").write_bytes(b"")
    (tmp_path / "b.cu3s").write_bytes(b"")
    csv_path = tmp_path / "universe.csv"
    csv_path.write_text("split,source,index\ntrain,a.cu3s,0\ntest,b.cu3s,0\n")
    out = tmp_path / "splits.json"
    monkeypatch.setattr(
        sys, "argv", ["resolve-splits", "--from-csv", str(csv_path), "--out", str(out)]
    )
    from cuvis_ai_dataloader.scripts.resolve_splits import resolve_splits_cli

    resolve_splits_cli()
    assert out.exists()
    loaded = load_splits(out)
    assert len(loaded.train) == 1 and len(loaded.test) == 1
    assert loaded.train[0].kind == SelectorKind.FILE_INDICES
