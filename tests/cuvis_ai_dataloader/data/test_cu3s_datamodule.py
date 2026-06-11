"""Tests for Cu3sDataModule (cu3s cubes + optional COCO masks)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai_core.data.datamodule import BaseHyperspectralDataModule
from cuvis_ai_dataloader.data import SingleCu3sDataModule, SingleCu3sDataset
from cuvis_ai_dataloader.data.datamodule_cu3s import Cu3sDataModule
from cuvis_ai_schemas.training.data import DataSplitConfig


def _make_cu3s(tmp_path, name="x.cu3s"):
    path = tmp_path / name
    path.write_bytes(b"")  # exists + .cu3s suffix is all the reader checks
    return str(path)


def test_data_module_name_and_subclass():
    assert Cu3sDataModule.DATA_MODULE_NAME == "cu3s"
    assert issubclass(Cu3sDataModule, BaseHyperspectralDataModule)
    assert SingleCu3sDataModule is Cu3sDataModule  # back-compat alias


def test_validate_params_requires_cu3s_path():
    with pytest.raises(ValueError, match="cu3s_file_path"):
        Cu3sDataModule.validate_params({})


def test_validate_params_rejects_bad_suffix(tmp_path):
    bad = tmp_path / "x.txt"
    bad.write_bytes(b"")
    with pytest.raises(ValueError, match=r"\.cu3s"):
        Cu3sDataModule.validate_params({"cu3s_file_path": str(bad)})


def test_predict_iterates_all_measurements(mock_cuvis_sdk, tmp_path):
    dm = Cu3sDataModule(cu3s_file_path=_make_cu3s(tmp_path), batch_size=1)
    dm.setup(stage="predict")
    loader = dm.predict_dataloader()
    batches = list(loader)
    assert len(batches) == 7  # mock session has 7 measurements
    batch = batches[0]
    assert set(batch.keys()) >= {"cube", "mesu_index", "wavelengths"}
    assert "stem" not in batch
    assert isinstance(batch["cube"], torch.Tensor)
    assert batch["cube"].shape[0] == 1  # batch dim
    assert batch["cube"].shape[-1] == mock_cuvis_sdk["channels"]


def test_sample_dict_types(mock_cuvis_sdk, tmp_path):
    dm = Cu3sDataModule(cu3s_file_path=_make_cu3s(tmp_path), batch_size=1)
    dm.setup(stage="predict")
    sample = dm._predict_ds[0]
    assert isinstance(sample["cube"], np.ndarray)
    assert isinstance(sample["mesu_index"], int)
    assert sample["wavelengths"].dtype == np.int32


def test_mask_attached_when_annotation_given(mock_cuvis_sdk, tmp_path):
    ann = tmp_path / "x.json"
    ann.write_text("{}")  # COCOData.from_path is mocked, content irrelevant
    dm = Cu3sDataModule(
        cu3s_file_path=_make_cu3s(tmp_path),
        annotation_json_path=str(ann),
        batch_size=1,
    )
    dm.setup(stage="predict")
    sample = dm._predict_ds[0]
    assert "mask" in sample
    h, w = mock_cuvis_sdk["hw"]
    assert sample["mask"].shape == (h, w)
    assert sample["mask"].dtype == np.int32


def test_setup_fit_from_splits(mock_cuvis_sdk, tmp_path):
    dm = Cu3sDataModule(
        cu3s_file_path=_make_cu3s(tmp_path),
        splits=DataSplitConfig(train_ids=[0, 2, 3], val_ids=[1, 5]),
        batch_size=2,
    )
    dm.setup(stage="fit")
    assert len(dm._train_ds) == 3
    assert len(dm._val_ds) == 2
    train_batch = next(iter(dm.train_dataloader()))
    assert train_batch["cube"].shape[0] == 2


def test_back_compat_flat_ids_fold_into_splits(mock_cuvis_sdk, tmp_path):
    dm = Cu3sDataModule(cu3s_file_path=_make_cu3s(tmp_path), train_ids=[0, 1], val_ids=[2])
    assert dm.splits is not None
    assert dm.splits.train_ids == [0, 1]
    dm.setup(stage="fit")
    assert len(dm._train_ds) == 2


def test_nested_cfg_data_construction(mock_cuvis_sdk, tmp_path):
    # `Cu3sDataModule(**cfg.data)` with the nested DataConfig shape (data_module,
    # splits-as-dict, params) must work for config-driven (hydra) call sites.
    cfg_data = {
        "data_module": "cu3s",
        "splits": {"predict_ids": []},
        "batch_size": 1,
        "num_workers": 0,
        "params": {"cu3s_file_path": _make_cu3s(tmp_path)},
    }
    dm = Cu3sDataModule(**cfg_data)
    dm.setup(stage="predict")
    assert len(dm._predict_ds) == 7


def test_single_cu3s_dataset_shim(mock_cuvis_sdk, tmp_path):
    ds = SingleCu3sDataset(_make_cu3s(tmp_path), measurement_indices=[0, 2, 4])
    assert len(ds) == 3
    item = ds[0]
    assert item["mesu_index"] == 0
    assert "cube" in item and "wavelengths" in item
