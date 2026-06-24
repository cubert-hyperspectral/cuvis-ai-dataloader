"""Tests for the AD-aware split resolver (no SDK: cached frame counts + stub labeler)."""

from __future__ import annotations

import json

import pytest

from cuvis_ai_dataloader.data.split_resolver import resolve_splits
from cuvis_ai_dataloader.data.workspace import SPLITS_FILENAME, SplitsFile, Workspace


class StubLabeler:
    """Maps frame -> anomaly category ids from a per-sibling table."""

    table: dict[str, dict[int, frozenset[int]]] = {}

    def __init__(self, annotation_json_path: str) -> None:
        self._cats = self.table.get(annotation_json_path, {})

    def anomaly_category_ids(self, image_id: int) -> frozenset[int]:
        return self._cats.get(image_id, frozenset())

    def is_anomalous(self, image_id: int) -> bool:
        return bool(self._cats.get(image_id))


def _make_ws(tmp_path, files, *, grouping=None, seed=42):
    """files: list of (name, frames, anomalies-or-None). None = no COCO sibling."""
    ws_root = tmp_path / "ws"
    ws_root.mkdir(exist_ok=True)
    data = tmp_path / "data"
    measurements = []
    StubLabeler.table = dict(StubLabeler.table)
    for name, frames, anomalies in files:
        cu3s = data / name / "Auto_000.cu3s"
        cu3s.parent.mkdir(parents=True, exist_ok=True)
        cu3s.write_bytes(b"\x00")
        if anomalies is not None:
            sibling = cu3s.with_suffix(".json")
            sibling.write_text(json.dumps({"name": name}), encoding="utf-8")
            StubLabeler.table[str(sibling)] = {f: frozenset(cats) for f, cats in anomalies.items()}
        measurements.append({"path": str(cu3s), "frames": frames})
    (ws_root / "workspace.json").write_text(
        json.dumps(
            {
                "version": 1,
                "name": "rs-test",
                "task_type": "anomaly",
                "default_seed": seed,
                "frame_grouping": grouping,
                "measurements": measurements,
            }
        ),
        encoding="utf-8",
    )
    return Workspace.load(ws_root)


def _partition_ok(lists, n):
    ids = lists["train"] + lists["val"] + lists["test"]
    assert sorted(ids) == list(range(n)), "train/val/test must partition the frames exactly"


def test_random_is_per_file_and_train_has_no_anomalies(tmp_path):
    ws = _make_ws(
        tmp_path,
        [
            ("a", 20, {3: {1}, 7: {1}, 11: {2}}),
            ("b", 10, {}),  # sibling exists, no anomalies
            ("c", 8, None),  # no sibling at all -> all normal
        ],
    )
    result = resolve_splits(ws, strategy="random", labeler_factory=StubLabeler)
    assert set(result.files) == set(ws.member_paths())
    for path, n, anomalies in [
        (ws.member_paths()[0], 20, {3, 7, 11}),
        (ws.member_paths()[1], 10, set()),
        (ws.member_paths()[2], 8, set()),
    ]:
        lists = result.files[path]
        _partition_ok(lists, n)
        assert not anomalies & set(lists["train"]), "train must contain only normal frames"
        assert anomalies <= set(lists["val"]) | set(lists["test"])
        assert lists["predict"] == lists["test"]


def test_coco_hashes_recorded_per_file(tmp_path):
    ws = _make_ws(tmp_path, [("a", 4, {0: {1}}), ("c", 4, None)])
    result = resolve_splits(ws, strategy="random", labeler_factory=StubLabeler, write=False)
    hashes = list(result.coco_hash_per_file.values())
    assert isinstance(hashes[0], str) and len(hashes[0]) == 64
    assert hashes[1] is None


def test_stratified_pools_and_keeps_train_clean(tmp_path):
    anomalies_a = {f: {1} for f in range(0, 12, 3)}  # cat 1 on file a
    anomalies_b = {f: {2} for f in range(1, 12, 3)}  # cat 2 on file b
    ws = _make_ws(tmp_path, [("a", 12, anomalies_a), ("b", 12, anomalies_b)])
    result = resolve_splits(ws, strategy="stratified", labeler_factory=StubLabeler, write=False)
    a, b = ws.member_paths()
    _partition_ok(result.files[a], 12)
    _partition_ok(result.files[b], 12)
    for path, anomalies in ((a, set(anomalies_a)), (b, set(anomalies_b))):
        assert not anomalies & set(result.files[path]["train"])
        eval_ids = set(result.files[path]["val"]) | set(result.files[path]["test"])
        assert anomalies <= eval_ids
    # each anomaly signature lands in BOTH eval splits (4 groups per category)
    val_all = set(result.files[a]["val"]) | set(result.files[b]["val"])
    test_all = set(result.files[a]["test"]) | set(result.files[b]["test"])
    assert val_all & (set(anomalies_a) | set(anomalies_b))
    assert test_all & (set(anomalies_a) | set(anomalies_b))


def test_frame_grouping_keeps_groups_whole(tmp_path):
    ws = _make_ws(
        tmp_path,
        [("a", 12, {5: {1}})],
        grouping={"mode": "fixed_size", "group_size": 4},
    )
    result = resolve_splits(ws, strategy="stratified", labeler_factory=StubLabeler, write=False)
    lists = result.files[ws.member_paths()[0]]
    _partition_ok(lists, 12)
    for start in range(0, 12, 4):
        group = set(range(start, start + 4))
        placements = [s for s in ("train", "val", "test") if group & set(lists[s])]
        assert len(placements) == 1, f"group {group} split across {placements}"
    # the group containing anomalous frame 5 (frames 4-7) must avoid train entirely
    assert not {4, 5, 6, 7} & set(lists["train"])


def test_seeded_reproducibility(tmp_path):
    files = [("a", 30, {i: {1} for i in range(0, 30, 5)}), ("b", 30, {})]
    ws = _make_ws(tmp_path, files)
    r1 = resolve_splits(ws, strategy="random", seed=7, labeler_factory=StubLabeler, write=False)
    r2 = resolve_splits(ws, strategy="random", seed=7, labeler_factory=StubLabeler, write=False)
    r3 = resolve_splits(ws, strategy="random", seed=8, labeler_factory=StubLabeler, write=False)
    assert r1.files == r2.files
    assert r1.files != r3.files


def test_selected_files_subset_and_unknown(tmp_path):
    ws = _make_ws(tmp_path, [("a", 6, {}), ("b", 6, {})])
    a, b = ws.member_paths()
    with pytest.warns(UserWarning):
        result = resolve_splits(
            ws,
            strategy="random",
            selected_files=[a],
            labeler_factory=StubLabeler,
            write=False,
        )
    assert list(result.files) == [a]
    with pytest.raises(ValueError, match="not workspace members"):
        resolve_splits(
            ws,
            strategy="random",
            selected_files=["/nope.cu3s"],
            labeler_factory=StubLabeler,
            write=False,
        )


def test_empty_workspace_and_missing_file_errors(tmp_path):
    ws_empty = _make_ws(tmp_path, [])
    with pytest.raises(ValueError, match="no measurements"):
        resolve_splits(ws_empty, strategy="random", labeler_factory=StubLabeler)

    ws = _make_ws(tmp_path, [("a", 4, {})])
    gone = tmp_path / "data" / "a" / "Auto_000.cu3s"
    gone.unlink()
    with pytest.raises(ValueError, match=str(gone.name)):
        resolve_splits(ws, strategy="random", labeler_factory=StubLabeler)


def test_no_anomalies_warns_but_succeeds(tmp_path):
    ws = _make_ws(tmp_path, [("c", 6, None)])
    with pytest.warns(UserWarning, match="no anomalous frames"):
        result = resolve_splits(ws, strategy="random", labeler_factory=StubLabeler, write=False)
    _partition_ok(result.files[ws.member_paths()[0]], 6)


def test_bad_ratios_and_strategy_rejected(tmp_path):
    ws = _make_ws(tmp_path, [("a", 4, {0: {1}})])
    with pytest.raises(ValueError, match="strategy"):
        resolve_splits(ws, strategy="bogus", labeler_factory=StubLabeler)
    with pytest.raises(ValueError, match="ratios"):
        resolve_splits(
            ws,
            strategy="random",
            train_ratio=0.9,
            val_ratio=0.5,
            labeler_factory=StubLabeler,
        )


def test_write_persists_loadable_splits(tmp_path):
    ws = _make_ws(tmp_path, [("a", 6, {1: {1}})])
    resolve_splits(ws, strategy="random", labeler_factory=StubLabeler, write=True)
    assert (ws.root / SPLITS_FILENAME).is_file()
    loaded = SplitsFile.load(ws.root)
    loaded.validate_against(ws)
    assert set(loaded.files) == set(ws.member_paths())
