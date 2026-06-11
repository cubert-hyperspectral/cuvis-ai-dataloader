"""Tests for MultiCu3sDataModule (multi-file cu3s + CSV-encoded splits)."""

from __future__ import annotations

import pytest

from cuvis_ai_core.data.datamodule import BaseHyperspectralDataModule
from cuvis_ai_dataloader.data.datamodule_cu3s_multi import MultiCu3sDataModule


def _write_dataset(tmp_path):
    (tmp_path / "frame_a.cu3s").write_bytes(b"")
    (tmp_path / "frame_b.cu3s").write_bytes(b"")
    (tmp_path / "frame_c.cu3s").write_bytes(b"")
    (tmp_path / "day1.json").write_text("{}")
    csv_path = tmp_path / "splits.csv"
    csv_path.write_text(
        "split,cu3s_path,annotation_json,image_id,extra_col\n"
        "test,frame_a.cu3s,day1.json,0,ignored\n"
        "test,frame_b.cu3s,day1.json,1,ignored\n"
        "train,frame_c.cu3s,day1.json,0,ignored\n"
    )
    return csv_path


def test_data_module_name_and_subclass():
    assert MultiCu3sDataModule.DATA_MODULE_NAME == "cu3s_multi"
    assert issubclass(MultiCu3sDataModule, BaseHyperspectralDataModule)


def test_validate_params_rejects_non_csv(tmp_path):
    bad = tmp_path / "x.txt"
    bad.write_text("")
    with pytest.raises(ValueError, match=r"\.csv"):
        MultiCu3sDataModule.validate_params({"splits_csv": str(bad)})


def test_csv_parsing_resolves_paths_and_frame_ids(mock_cuvis_sdk, tmp_path):
    csv_path = _write_dataset(tmp_path)
    dm = MultiCu3sDataModule(splits_csv=str(csv_path))
    assert len(dm._rows) == 3
    # frame_id is the CSV row index (globally unique even when image_id repeats).
    assert [r["frame_id"] for r in dm._rows] == [0, 1, 2]
    assert dm._rows[0]["cu3s_path"].endswith("frame_a.cu3s")
    assert dm._rows[0]["annotation_json"].endswith("day1.json")


def test_missing_required_column_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("split,cu3s_path,image_id\ntest,a.cu3s,0\n")
    with pytest.raises(ValueError, match="missing required column"):
        MultiCu3sDataModule(splits_csv=str(csv_path))


def test_build_stage_filters_by_split(mock_cuvis_sdk, tmp_path):
    csv_path = _write_dataset(tmp_path)
    dm = MultiCu3sDataModule(splits_csv=str(csv_path))
    dm.setup(stage="test")  # predict honors split; test maps to the "test" rows
    assert len(dm._test_ds) == 2
    item = dm._test_ds[0]
    assert set(item.keys()) >= {
        "cube",
        "wavelengths",
        "mesu_index",
        "frame_id",
        "annotation_json",
        "mask",
    }
    assert item["mesu_index"] == 0
    assert item["frame_id"] == 0


def test_labeler_reuse_across_rows(mock_cuvis_sdk, tmp_path):
    csv_path = _write_dataset(tmp_path)
    dm = MultiCu3sDataModule(splits_csv=str(csv_path))
    dm.setup(stage="test")
    ds = dm._test_ds
    _ = ds[0]
    _ = ds[1]
    # Both test rows share day1.json -> exactly one CocoLabeler cached.
    assert len(ds._labelers) == 1
