"""Tests for workspace.json / splits.json parsing (pure stdlib, no SDK needed)."""

from __future__ import annotations

import json

import pytest

from cuvis_ai_dataloader.data.workspace import (
    SPLITS_FILENAME,
    WORKSPACE_FILENAME,
    Measurement,
    SplitsFile,
    Workspace,
    canonical,
    coco_sibling,
)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00cu3s")
    return path


def _write_workspace(root, **overrides):
    payload = {
        "version": 1,
        "name": "test-ws",
        "task_type": "anomaly",
        "default_processing_mode": "Reflectance",
        "default_seed": 42,
        "frame_grouping": None,
        "measurements": [],
        "scan_roots": [],
        "excluded": [],
    }
    payload.update(overrides)
    root.mkdir(parents=True, exist_ok=True)
    (root / WORKSPACE_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    return root


def test_load_happy_path(tmp_path):
    cu3s = _touch(tmp_path / "data" / "sessionA" / "Auto_000.cu3s")
    root = _write_workspace(
        tmp_path / "ws",
        measurements=[{"path": str(cu3s), "added_at": "2026-06-12T10:00:00", "frames": 36}],
        scan_roots=[str(tmp_path / "data")],
    )
    ws = Workspace.load(root)
    assert ws.name == "test-ws"
    assert ws.default_seed == 42
    assert ws.member_paths() == [str(cu3s)]
    assert ws.measurements[0].frames == 36
    assert ws.frames_by_path() == {canonical(cu3s): 36}


def test_not_a_workspace_names_the_qualifier(tmp_path):
    with pytest.raises(ValueError, match=WORKSPACE_FILENAME):
        Workspace.load(tmp_path)


def test_corrupt_workspace_is_a_hard_error(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / WORKSPACE_FILENAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        Workspace.load(root)


def test_unsupported_version_rejected(tmp_path):
    root = _write_workspace(tmp_path / "ws", version=2)
    with pytest.raises(ValueError, match="version"):
        Workspace.load(root)


def test_measurement_without_path_rejected():
    with pytest.raises(ValueError, match="path"):
        Measurement.from_dict({"frames": 3})


def test_dedupe_by_canonical_path(tmp_path):
    cu3s = _touch(tmp_path / "data" / "s1" / "Auto_000.cu3s")
    spelled_differently = tmp_path / "data" / "s1" / ".." / "s1" / "Auto_000.cu3s"
    root = _write_workspace(
        tmp_path / "ws",
        measurements=[
            {"path": str(cu3s), "frames": 5},
            {"path": str(cu3s)},
            {"path": str(spelled_differently)},
        ],
    )
    ws = Workspace.load(root)
    assert len(ws.member_paths()) == 1
    assert ws.member_measurements()[0].frames == 5  # first entry wins
    assert ws.duplicate_paths() == [str(cu3s), str(spelled_differently)]


def test_missing_paths_are_named(tmp_path):
    present = _touch(tmp_path / "data" / "ok" / "Auto_000.cu3s")
    gone = tmp_path / "data" / "gone" / "Auto_000.cu3s"
    root = _write_workspace(
        tmp_path / "ws",
        measurements=[{"path": str(present)}, {"path": str(gone)}],
    )
    assert Workspace.load(root).missing_paths() == [str(gone)]


def test_banner_set_subtracts_members_and_excluded(tmp_path):
    data = tmp_path / "data"
    member = _touch(data / "a" / "Auto_000.cu3s")
    declined = _touch(data / "b" / "Auto_000.cu3s")
    fresh = _touch(data / "c" / "Auto_000.cu3s")
    root = _write_workspace(
        tmp_path / "ws",
        measurements=[{"path": str(member)}],
        scan_roots=[str(data)],
        excluded=[str(declined)],
    )
    assert Workspace.load(root).new_files_under_scan_roots() == [str(fresh)]


def test_banner_set_with_injected_lister(tmp_path):
    root = _write_workspace(tmp_path / "ws", scan_roots=["fake-root"])
    ws = Workspace.load(root)
    fake = [tmp_path / "x.cu3s", tmp_path / "y.cu3s"]
    for f in fake:
        _touch(f)
    out = ws.new_files_under_scan_roots(list_files=lambda _root: fake)
    assert out == sorted(str(f) for f in fake)


def test_coco_sibling(tmp_path):
    cu3s = _touch(tmp_path / "Auto_000.cu3s")
    assert coco_sibling(cu3s) is None
    sib = tmp_path / "Auto_000.json"
    sib.write_text("{}", encoding="utf-8")
    assert coco_sibling(cu3s) == sib


def test_splits_roundtrip_and_atomic_save(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    splits = SplitsFile(
        files={"/abs/a.cu3s": {"train": [0, 1], "val": [2], "test": [3], "predict": [3]}},
        coco_hash_per_file={"/abs/a.cu3s": "deadbeef"},
    )
    target = splits.save(root)
    assert target.name == SPLITS_FILENAME
    assert not list(root.glob("*.tmp"))
    loaded = SplitsFile.load(root)
    assert loaded.files == splits.files
    assert loaded.coco_hash_per_file == splits.coco_hash_per_file


def test_splits_missing_mentions_resolver(tmp_path):
    with pytest.raises(ValueError, match="resolver"):
        SplitsFile.load(tmp_path)


def test_splits_unknown_key_rejected(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / SPLITS_FILENAME).write_text(
        json.dumps({"version": 1, "files": {"/a.cu3s": {"train": [], "bogus": [1]}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="bogus"):
        SplitsFile.load(root)


def test_validate_against_names_offenders(tmp_path):
    cu3s = _touch(tmp_path / "data" / "s" / "Auto_000.cu3s")
    root = _write_workspace(tmp_path / "ws", measurements=[{"path": str(cu3s), "frames": 4}])
    ws = Workspace.load(root)

    ok = SplitsFile(files={str(cu3s): {"train": [0, 1], "val": [2], "test": [3], "predict": [3]}})
    ok.validate_against(ws)  # no raise

    non_member = SplitsFile(
        files={"/elsewhere/x.cu3s": {k: [] for k in ("train", "val", "test", "predict")}}
    )
    with pytest.raises(ValueError, match="not a workspace member"):
        non_member.validate_against(ws)

    out_of_range = SplitsFile(
        files={str(cu3s): {"train": [0, 9], "val": [], "test": [], "predict": []}}
    )
    with pytest.raises(ValueError, match="out of range"):
        out_of_range.validate_against(ws)


def test_validate_skips_range_strings_and_unknown_counts(tmp_path):
    counted = _touch(tmp_path / "d" / "a.cu3s")
    uncounted = _touch(tmp_path / "d" / "b.cu3s")
    root = _write_workspace(
        tmp_path / "ws",
        measurements=[{"path": str(counted), "frames": 10}, {"path": str(uncounted)}],
    )
    ws = Workspace.load(root)
    splits = SplitsFile(
        files={
            str(counted): {"train": ["0-5"], "val": [6], "test": [7], "predict": [7]},
            str(uncounted): {"train": [999], "val": [], "test": [], "predict": []},
        }
    )
    splits.validate_against(ws)  # range strings + unknown counts are not range-checked here
