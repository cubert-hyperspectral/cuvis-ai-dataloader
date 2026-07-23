"""Tests for MultiCu3sDataModule (multi-file cu3s over the shared universe.csv vocabulary)."""

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
    csv_path = tmp_path / "universe.csv"
    csv_path.write_text(
        "split,source,annotation,index,extra_col\n"
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
        MultiCu3sDataModule.validate_params({"universe_csv": str(bad)})


def test_unknown_kwarg_raises():
    # Unknown / removed options fail loudly instead of being silently dropped.
    with pytest.raises(TypeError, match="bogus"):
        MultiCu3sDataModule(universe_csv="x.csv", bogus=1)


def test_csv_parsing_resolves_paths_and_frame_ids(mock_cuvis_sdk, tmp_path):
    csv_path = _write_dataset(tmp_path)
    dm = MultiCu3sDataModule(universe_csv=str(csv_path))
    assert len(dm._rows) == 3
    # frame_id is the CSV row index (globally unique even when index repeats across sources).
    assert [r["frame_id"] for r in dm._rows] == [0, 1, 2]
    # `source` is the posix identity as written (not resolved to an absolute path).
    assert dm._rows[0]["source"] == "frame_a.cu3s"
    # `materialized_path` defaults to `source`, resolved to the physical file.
    assert dm._rows[0]["materialized_path"].endswith("frame_a.cu3s")
    assert dm._rows[0]["annotation"].endswith("day1.json")


def test_missing_required_column_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("split,annotation,index\ntest,day1.json,0\n")  # no source column
    with pytest.raises(ValueError, match="missing required column"):
        MultiCu3sDataModule(universe_csv=str(csv_path))


def test_build_stage_filters_by_split(mock_cuvis_sdk, tmp_path):
    csv_path = _write_dataset(tmp_path)
    dm = MultiCu3sDataModule(universe_csv=str(csv_path))
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
    dm = MultiCu3sDataModule(universe_csv=str(csv_path))
    dm.setup(stage="test")
    ds = dm._test_ds
    _ = ds[0]
    _ = ds[1]
    # Both test rows share day1.json -> exactly one CocoLabeler cached.
    assert len(ds._labelers) == 1


def _write_ranged_dataset(tmp_path):
    (tmp_path / "clip.cu3s").write_bytes(b"")
    (tmp_path / "day1.json").write_text("{}")
    csv_path = tmp_path / "universe.csv"
    csv_path.write_text(
        "split,source,annotation,index\n"
        "train,clip.cu3s,day1.json,0-2\n"  # per-file frame range -> 3 samples
        "test,clip.cu3s,,5\n"  # scalar -> read measurement 5, image_id 5
    )
    return csv_path


def test_ranged_index_fans_out_rows(mock_cuvis_sdk, tmp_path):
    csv_path = _write_ranged_dataset(tmp_path)
    dm = MultiCu3sDataModule(universe_csv=str(csv_path))
    train_rows = [r for r in dm._rows if r["split"] == "train"]
    # "0-2" -> measurements 0,1,2; index is both the read position and the image_id.
    assert [r["index"] for r in train_rows] == [0, 1, 2]
    test_rows = [r for r in dm._rows if r["split"] == "test"]
    assert len(test_rows) == 1
    assert test_rows[0]["index"] == 5  # scalar reads (and labels) measurement 5
    # frame_id is unique + contiguous across the fan-out.
    assert [r["frame_id"] for r in dm._rows] == [0, 1, 2, 3]


def test_ranged_index_reads_the_right_measurement(mock_cuvis_sdk, tmp_path):
    csv_path = _write_ranged_dataset(tmp_path)
    dm = MultiCu3sDataModule(universe_csv=str(csv_path))
    dm.setup(stage="fit")
    assert len(dm._train_ds) == 3
    session = mock_cuvis_sdk["session"]
    session.get_measurement.reset_mock()
    _ = dm._train_ds[2]  # third train sample -> measurement 2
    session.get_measurement.assert_any_call(2)


def test_import_csv_splits_groups_per_source(mock_cuvis_sdk, tmp_path):
    csv_path = _write_dataset(tmp_path)
    dm = MultiCu3sDataModule(universe_csv=str(csv_path))
    splits = import_csv_splits(dm)
    # test split: frame_a + frame_b (one FILE_INDICES selector each); train: frame_c.
    assert len(splits.train) == 1
    assert len(splits.test) == 2
    assert splits.train[0].kind == SelectorKind.FILE_INDICES
    # selectors key on the posix `source` identity, not a resolved absolute path.
    assert splits.train[0].source == "frame_c.cu3s"
    assert splits.train[0].ids == [0]


def test_import_csv_splits_without_split_column_raises(mock_cuvis_sdk, tmp_path):
    (tmp_path / "frame_a.cu3s").write_bytes(b"")
    csv_path = tmp_path / "universe.csv"
    csv_path.write_text("source,index\nframe_a.cu3s,0\n")  # no split values to import
    dm = MultiCu3sDataModule(universe_csv=str(csv_path))
    with pytest.raises(ValueError, match="no 'split' column values"):
        import_csv_splits(dm)


def test_selector_path_resolves_against_posix_identity(mock_cuvis_sdk, tmp_path):
    csv_path = _write_dataset(tmp_path)
    # Selector keys on the CSV's posix `source` identity (the portable form), NOT an absolute path.
    dm = MultiCu3sDataModule(
        universe_csv=str(csv_path),
        splits=DataSplitConfig(
            train=[Selector(kind=SelectorKind.FILE_INDICES, source="frame_c.cu3s", ids=[0])]
        ),
    )
    dm.setup(stage="fit")
    assert len(dm._train_ds) == 1  # selector resolved against the CSV universe


def test_split_column_toggles_owns_splits(mock_cuvis_sdk, tmp_path):
    # A `split` column makes the module own its splits; without it, it does not. A core that
    # honors OWNS_SPLITS then refuses a split-less training stage (needs a splits.json); the
    # toggle is what this module controls, so that is what we assert here.
    (tmp_path / "frame_a.cu3s").write_bytes(b"")
    with_split = tmp_path / "with_split.csv"
    with_split.write_text("split,source,index\ntrain,frame_a.cu3s,0\n")
    assert MultiCu3sDataModule(universe_csv=str(with_split)).OWNS_SPLITS is True
    no_split = tmp_path / "no_split.csv"
    no_split.write_text("source,index\nframe_a.cu3s,0\n")
    assert MultiCu3sDataModule(universe_csv=str(no_split)).OWNS_SPLITS is False


def test_split_all_iterates_whole_universe(mock_cuvis_sdk, tmp_path):
    # `split="all"` in module-owned predict serves every row regardless of split assignment.
    (tmp_path / "frame_a.cu3s").write_bytes(b"")
    csv_path = tmp_path / "universe.csv"
    csv_path.write_text("source,index\nframe_a.cu3s,0\n")
    dm = MultiCu3sDataModule(universe_csv=str(csv_path), split="all")
    dm.setup(stage="predict")
    assert len(dm._predict_ds) == 1


def test_splits_json_overrides_inline_split_column(mock_cuvis_sdk, tmp_path):
    csv_path = _write_dataset(tmp_path)  # has a split column (frame_c -> train, a/b -> test)
    # An explicit splits.json wins: only frame_a (a "test" row in the CSV) is selected into train.
    dm = MultiCu3sDataModule(
        universe_csv=str(csv_path),
        splits=DataSplitConfig(
            train=[Selector(kind=SelectorKind.FILE_INDICES, source="frame_a.cu3s", ids=[0])]
        ),
    )
    dm.setup(stage="fit")
    assert len(dm._train_ds) == 1


def test_read_index_exceeding_measurements_raises_at_build(mock_cuvis_sdk, tmp_path):
    (tmp_path / "clip.cu3s").write_bytes(b"")
    csv_path = tmp_path / "universe.csv"
    csv_path.write_text("split,source,annotation,index\ntrain,clip.cu3s,,0-99\n")  # 99 >= mock 7
    dm = MultiCu3sDataModule(universe_csv=str(csv_path))
    with pytest.raises(ValueError, match="read index 99 >= 7"):
        dm.setup(stage="fit")
