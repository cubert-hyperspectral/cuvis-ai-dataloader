"""Tests for WorkspaceCu3sDataModule (mock SDK; workspace + splits fixtures on tmp_path)."""

from __future__ import annotations

import json

import pytest

from cuvis_ai_dataloader.data.datamodule_cu3s_workspace import WorkspaceCu3sDataModule


def _make_workspace(tmp_path, *, with_splits=True, frames=(4, 3), predict=None):
    """Two fake member files; member 0 gets a COCO sibling. Returns ws root."""
    root = tmp_path / "ws"
    root.mkdir()
    data = tmp_path / "data"
    members, splits_files = [], {}
    for i, n in enumerate(frames):
        cu3s = data / f"s{i}" / "Auto_000.cu3s"
        cu3s.parent.mkdir(parents=True, exist_ok=True)
        cu3s.write_bytes(b"\x00")
        if i == 0:
            cu3s.with_suffix(".json").write_text("{}", encoding="utf-8")
        members.append({"path": str(cu3s), "frames": n})
        ids = list(range(n))
        splits_files[str(cu3s)] = {
            "train": ids[: max(n - 2, 0)],
            "val": ids[max(n - 2, 0) : n - 1],
            "test": ids[n - 1 :],
            "predict": predict if predict is not None else ids[n - 1 :],
        }
    (root / "workspace.json").write_text(
        json.dumps(
            {
                "version": 1,
                "name": "dm-test",
                "default_processing_mode": "Raw",
                "measurements": members,
            }
        ),
        encoding="utf-8",
    )
    if with_splits:
        (root / "splits.json").write_text(
            json.dumps({"version": 1, "coco_hash_per_file": {}, "files": splits_files}),
            encoding="utf-8",
        )
    return root, members


def test_stage_datasets_match_splits(tmp_path, mock_cuvis_sdk):
    root, members = _make_workspace(tmp_path, frames=(4, 3))
    dm = WorkspaceCu3sDataModule(workspace_path=str(root))
    dm.setup()
    # train: (4-2) + (3-2) = 3, val: 1+1, test: 1+1
    assert len(dm.train_dataloader().dataset) == 3
    assert len(dm.val_dataloader().dataset) == 2
    assert len(dm.test_dataloader().dataset) == 2
    item = dm.train_dataloader().dataset[0]
    assert "cube" in item and "frame_id" in item and "mask" in item  # member 0 has a sibling
    # member 1 has no sibling -> no mask key for its rows
    last = dm.train_dataloader().dataset[2]
    assert last["annotation_json"] == "" and "mask" not in last


def test_processing_mode_defaults_from_workspace(tmp_path, mock_cuvis_sdk):
    root, _ = _make_workspace(tmp_path)
    dm = WorkspaceCu3sDataModule(workspace_path=str(root))
    assert dm._processing_mode == "Raw"
    dm2 = WorkspaceCu3sDataModule(workspace_path=str(root), processing_mode="Reflectance")
    assert dm2._processing_mode == "Reflectance"


def test_params_dict_construction_like_create_data_module(tmp_path, mock_cuvis_sdk):
    root, _ = _make_workspace(tmp_path)
    dm = WorkspaceCu3sDataModule(params={"workspace_path": str(root)})
    dm.setup(stage="test")
    assert len(dm.test_dataloader().dataset) == 2


def test_predict_uses_first_class_predict_ids(tmp_path, mock_cuvis_sdk):
    root, members = _make_workspace(tmp_path, frames=(4, 3), predict=[0, 1])
    dm = WorkspaceCu3sDataModule(workspace_path=str(root))
    dm.setup(stage="predict")
    assert len(dm.predict_dataloader().dataset) == 4  # [0,1] per member


def test_empty_predict_means_all_frames(tmp_path, mock_cuvis_sdk):
    root, members = _make_workspace(tmp_path, frames=(4, 3), predict=[])
    dm = WorkspaceCu3sDataModule(workspace_path=str(root))
    dm.setup(stage="predict")
    assert len(dm.predict_dataloader().dataset) == 7  # all frames of both members


def test_range_string_selectors_in_splits_json(tmp_path, mock_cuvis_sdk):
    root, members = _make_workspace(tmp_path, frames=(6, 3))
    splits = json.loads((root / "splits.json").read_text())
    splits["files"][members[0]["path"]]["train"] = ["0-3"]
    (root / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    dm = WorkspaceCu3sDataModule(workspace_path=str(root))
    dm.setup(stage="fit")
    train_rows = dm.train_dataloader().dataset
    member0_frames = [
        r["read_index"] for r in train_rows._rows if r["cu3s_path"] == members[0]["path"]
    ]
    assert member0_frames == [0, 1, 2, 3]


def test_validate_params_messages(tmp_path):
    with pytest.raises(ValueError, match="workspace_path"):
        WorkspaceCu3sDataModule.validate_params({})
    with pytest.raises(ValueError, match="workspace.json"):
        WorkspaceCu3sDataModule.validate_params({"workspace_path": str(tmp_path)})
    root, _ = _make_workspace(tmp_path, with_splits=False)
    with pytest.raises(ValueError, match="resolver|Create splits"):
        WorkspaceCu3sDataModule.validate_params({"workspace_path": str(root)})


def test_missing_measurement_is_named(tmp_path, mock_cuvis_sdk):
    root, members = _make_workspace(tmp_path)
    gone = members[1]["path"]
    import os

    os.remove(gone)
    with pytest.raises(ValueError, match="missing measurement"):
        WorkspaceCu3sDataModule(workspace_path=str(root))


def test_member_without_splits_entry_is_not_served(tmp_path, mock_cuvis_sdk):
    root, members = _make_workspace(tmp_path)
    splits = json.loads((root / "splits.json").read_text())
    del splits["files"][members[1]["path"]]
    (root / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    dm = WorkspaceCu3sDataModule(workspace_path=str(root))
    dm.setup()
    paths = {r["cu3s_path"] for r in dm.train_dataloader().dataset._rows}
    assert paths == {members[0]["path"]}


def test_dispatch_via_core_create_data_module(tmp_path, mock_cuvis_sdk):
    """The generic DataConfig route: data_module name + params, no special config type."""
    from types import SimpleNamespace

    from cuvis_ai_core.data.datamodule import create_data_module
    from cuvis_ai_schemas.training.data import DataConfig

    root, _ = _make_workspace(tmp_path)
    registry = SimpleNamespace(data_modules={"cu3s_workspace": WorkspaceCu3sDataModule})
    cfg = DataConfig(
        data_module="cu3s_workspace",
        splits=None,
        batch_size=2,
        params={"workspace_path": str(root)},
    )
    dm = create_data_module(registry, cfg)
    dm.setup(stage="test")
    assert dm.batch_size == 2
    assert len(dm.test_dataloader().dataset) == 2


def test_manifest_declares_cu3s_workspace():
    from pathlib import Path

    manifest = Path(__file__).resolve().parents[3] / "configs/plugins/cuvis_ai_dataloader.yaml"
    text = manifest.read_text(encoding="utf-8")
    assert "data_module_name: cu3s_workspace" in text
    assert "datamodule_cu3s_workspace.WorkspaceCu3sDataModule" in text
