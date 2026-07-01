"""Tests for MultiNpzDataModule (one-frame-per-file NPZ + CSV-encoded splits)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cuvis_ai_core.data.datamodule import BaseCuvisAIDataModule
from cuvis_ai_dataloader.data.datamodule_npz_multi import MultiNpzDataModule, _MultiNpzDataset


def _write_npz(path: Path, *, with_mask: bool) -> None:
    h, w, c = 8, 10, 5
    cube = np.zeros((h, w, c), dtype=np.float32)
    wavelengths = np.linspace(450, 850, c).astype(np.float32)
    if with_mask:
        mask = np.zeros((h, w), dtype=np.int32)
        mask[2:5, 3:7] = 2
        np.savez(path, cube=cube, wavelengths=wavelengths, mask=mask)
    else:
        np.savez(path, cube=cube, wavelengths=wavelengths)


def _write_dataset(tmp_path: Path) -> Path:
    for name in ("train.npz", "val.npz", "test.npz"):
        _write_npz(tmp_path / name, with_mask=True)
    csv_path = tmp_path / "splits.csv"
    # Extra columns (cu3s_path, annotation_json) are allowed and ignored.
    csv_path.write_text(
        "split,npz_path,cu3s_path,annotation_json,image_id\n"
        "train,train.npz,a.cu3s,,1\n"
        "val,val.npz,b.cu3s,,2\n"
        "test,test.npz,c.cu3s,,3\n"
    )
    return csv_path


def test_data_module_name_and_subclass():
    assert MultiNpzDataModule.DATA_MODULE_NAME == "npz_multi"
    assert issubclass(MultiNpzDataModule, BaseCuvisAIDataModule)


def test_validate_params_rejects_non_csv(tmp_path):
    bad = tmp_path / "x.txt"
    bad.write_text("")
    with pytest.raises(ValueError, match=r"\.csv"):
        MultiNpzDataModule.validate_params({"splits_csv": str(bad)})


def test_unknown_kwarg_raises():
    with pytest.raises(TypeError, match="bogus"):
        MultiNpzDataModule(splits_csv="x.csv", bogus=1)


def test_missing_required_column_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("split,image_id\ntest,0\n")  # no npz_path
    with pytest.raises(ValueError, match="missing required column"):
        MultiNpzDataModule(splits_csv=str(csv_path))


def test_csv_parsing_resolves_paths_and_frame_ids(tmp_path):
    csv_path = _write_dataset(tmp_path)
    dm = MultiNpzDataModule(splits_csv=str(csv_path))
    assert len(dm._rows) == 3
    assert [r["frame_id"] for r in dm._rows] == [0, 1, 2]
    assert dm._rows[0]["npz_path"].endswith("train.npz")
    assert dm._rows[0]["image_id"] == 1


def test_dataset_reads_cube_mask_and_wavelengths(tmp_path):
    npz = tmp_path / "frame.npz"
    _write_npz(npz, with_mask=True)
    ds = _MultiNpzDataset([{"npz_path": str(npz), "image_id": 7, "frame_id": 0}])
    item = ds[0]
    assert item["cube"].shape == (8, 10, 5)
    assert item["mask"].shape == (8, 10)
    assert item["wavelengths"].shape == (5,)
    assert item["wavelengths"].dtype == np.int32
    assert int(item["mesu_index"]) == 7
    assert ds.num_channels == 5


def test_dataset_builds_empty_mask_when_absent(tmp_path):
    npz = tmp_path / "frame_nomask.npz"
    _write_npz(npz, with_mask=False)
    ds = _MultiNpzDataset([{"npz_path": str(npz), "image_id": 11, "frame_id": 0}])
    item = ds[0]
    assert item["mask"].shape == (8, 10)
    assert item["mask"].dtype == np.int32
    assert np.all(item["mask"] == 0)


def test_build_stage_filters_by_split(tmp_path):
    csv_path = _write_dataset(tmp_path)
    dm = MultiNpzDataModule(splits_csv=str(csv_path))
    dm.setup(stage="test")
    assert len(dm._test_ds) == 1
    item = dm._test_ds[0]
    assert set(item.keys()) >= {"cube", "mask", "wavelengths", "mesu_index", "frame_id"}
    assert item["mesu_index"] == 3


def test_setup_fit_builds_train_and_val(tmp_path):
    csv_path = _write_dataset(tmp_path)
    dm = MultiNpzDataModule(splits_csv=str(csv_path), batch_size=1, num_workers=0)
    dm.setup(stage="fit")
    assert len(dm.train_ds) == 1
    assert len(dm.val_ds) == 1
    batch = next(iter(dm.train_dataloader()))
    assert set(batch.keys()) >= {"cube", "mask", "wavelengths", "mesu_index"}
    assert batch["cube"].shape == (1, 8, 10, 5)


def test_loader_before_setup_raises(tmp_path):
    csv_path = _write_dataset(tmp_path)
    dm = MultiNpzDataModule(splits_csv=str(csv_path))
    with pytest.raises(RuntimeError):
        dm.train_dataloader()
    with pytest.raises(RuntimeError):
        dm.val_dataloader()
    with pytest.raises(RuntimeError):
        dm.test_dataloader()


def test_loader_honors_pin_memory(tmp_path):
    csv_path = _write_dataset(tmp_path)
    dm = MultiNpzDataModule(splits_csv=str(csv_path), pin_memory=True, num_workers=0)
    dm.setup(stage="fit")
    assert dm.train_dataloader().pin_memory is True
