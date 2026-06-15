"""Tests for MultiCu3sDataModule (multi-file cu3s + CSV-encoded splits)."""

from __future__ import annotations

import pytest

from cuvis_ai_core.data.datamodule import BaseCuvisAIDataModule
from cuvis_ai_dataloader.data.datamodule_cu3s_multi import MultiCu3sDataModule
from cuvis_ai_dataloader.data.resolvers import import_csv_splits
from cuvis_ai_schemas.training.data import DataSplitConfig, Selector, SelectorKind


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
    assert issubclass(MultiCu3sDataModule, BaseCuvisAIDataModule)


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


def _write_ranged_dataset(tmp_path):
    (tmp_path / "clip.cu3s").write_bytes(b"")
    (tmp_path / "day1.json").write_text("{}")
    csv_path = tmp_path / "splits.csv"
    csv_path.write_text(
        "split,cu3s_path,annotation_json,image_id\n"
        "train,clip.cu3s,day1.json,0-2\n"  # per-file frame range -> 3 samples
        "test,clip.cu3s,,5\n"  # scalar -> legacy single read(0)
    )
    return csv_path


def test_ranged_image_id_fans_out_rows(mock_cuvis_sdk, tmp_path):
    csv_path = _write_ranged_dataset(tmp_path)
    dm = MultiCu3sDataModule(splits_csv=str(csv_path))
    train_rows = [r for r in dm._rows if r["split"] == "train"]
    # "0-2" -> measurements 0,1,2; read_index tracks the measurement, image_id too.
    assert [r["read_index"] for r in train_rows] == [0, 1, 2]
    assert [r["image_id"] for r in train_rows] == [0, 1, 2]
    test_rows = [r for r in dm._rows if r["split"] == "test"]
    assert len(test_rows) == 1
    assert test_rows[0]["read_index"] == 0  # scalar keeps legacy read(0)
    assert test_rows[0]["image_id"] == 5
    # frame_id is unique + contiguous across the fan-out.
    assert [r["frame_id"] for r in dm._rows] == [0, 1, 2, 3]


def test_ranged_image_id_reads_the_right_measurement(mock_cuvis_sdk, tmp_path):
    csv_path = _write_ranged_dataset(tmp_path)
    dm = MultiCu3sDataModule(splits_csv=str(csv_path))
    dm.setup(stage="fit")
    assert len(dm._train_ds) == 3
    session = mock_cuvis_sdk["session"]
    session.get_measurement.reset_mock()
    _ = dm._train_ds[2]  # third train sample -> measurement 2
    session.get_measurement.assert_any_call(2)


def test_import_csv_splits_groups_per_source(mock_cuvis_sdk, tmp_path):
    csv_path = _write_dataset(tmp_path)
    dm = MultiCu3sDataModule(splits_csv=str(csv_path))
    splits = import_csv_splits(dm)
    # test split: frame_a + frame_b (one FILE_INDICES selector each); train: frame_c.
    assert len(splits.train) == 1
    assert len(splits.test) == 2
    assert splits.train[0].kind == SelectorKind.FILE_INDICES
    assert splits.train[0].source.endswith("frame_c.cu3s")
    assert splits.train[0].ids == [0]


def test_selector_path_resolves_against_csv_universe(mock_cuvis_sdk, tmp_path):
    csv_path = _write_dataset(tmp_path)
    frame_c = str((tmp_path / "frame_c.cu3s").resolve())
    dm = MultiCu3sDataModule(
        splits_csv=str(csv_path),
        splits=DataSplitConfig(
            train=[Selector(kind=SelectorKind.FILE_INDICES, source=frame_c, ids=[0])]
        ),
    )
    dm.setup(stage="fit")
    assert len(dm._train_ds) == 1  # selector resolved against the CSV universe


def test_read_index_exceeding_measurements_raises_at_build(mock_cuvis_sdk, tmp_path):
    (tmp_path / "clip.cu3s").write_bytes(b"")
    csv_path = tmp_path / "splits.csv"
    csv_path.write_text(
        "split,cu3s_path,annotation_json,image_id\ntrain,clip.cu3s,,0-99\n"  # 99 >= mock 7
    )
    dm = MultiCu3sDataModule(splits_csv=str(csv_path))
    with pytest.raises(ValueError, match="read_index 99 >= 7"):
        dm.setup(stage="fit")


def test_no_coco_sibling_yields_zero_mask(mock_cuvis_sdk, tmp_path):
    # A row with an empty annotation_json has no COCO sibling. The frame must
    # still carry a 'mask' key (zeros) so val/test metric nodes that consume
    # 'targets' get a tensor of the right shape, matching CocoLabeler's
    # unannotated path. Regression for the no-COCO zero-mask gap (ALL-5766).
    import numpy as np

    csv_path = _write_ranged_dataset(tmp_path)
    dm = MultiCu3sDataModule(splits_csv=str(csv_path))
    dm.setup(stage='test')  # the 'test' split row is clip.cu3s with no annotation
    ds = dm._test_ds
    assert ds._rows[0]['annotation_json'] == ''  # the no-sibling row
    item = ds[0]
    assert 'mask' in item
    h, w = mock_cuvis_sdk['hw']
    assert item['mask'].shape == (h, w)
    assert item['mask'].dtype == np.int32
    assert int(item['mask'].sum()) == 0
    # No COCO file was opened for this frame.
    assert len(ds._labelers) == 0
