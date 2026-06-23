"""Tests for Cu3sDataModule (cu3s cubes + optional COCO masks)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai_core.data.datamodule import BaseCuvisAIDataModule
from cuvis_ai_dataloader.data.datamodule_cu3s import Cu3sDataModule
from cuvis_ai_schemas.training.data import DataSplitConfig, Selector, SelectorKind


def _make_cu3s(tmp_path, name="x.cu3s"):
    path = tmp_path / name
    path.write_bytes(b"")  # exists + .cu3s suffix is all the reader checks
    return str(path)


def _make_cu3s_folder(tmp_path, n=4):
    folder = tmp_path / "session_dir"
    folder.mkdir()
    for i in range(n):
        (folder / f"scan_{i:02d}.cu3s").write_bytes(b"")
    return folder


def _fi(source, ids):
    return [Selector(kind=SelectorKind.FILE_INDICES, source=source, ids=ids)]


def _dir(ids):
    return [Selector(kind=SelectorKind.DIR_INDICES, ids=ids)]


def test_data_module_name_and_subclass():
    assert Cu3sDataModule.DATA_MODULE_NAME == "cu3s"
    assert issubclass(Cu3sDataModule, BaseCuvisAIDataModule)


def test_validate_params_requires_cu3s_path():
    with pytest.raises(ValueError, match="cu3s_file_path"):
        Cu3sDataModule.validate_params({})


def test_validate_params_rejects_bad_suffix(tmp_path):
    bad = tmp_path / "x.txt"
    bad.write_bytes(b"")
    with pytest.raises(ValueError, match=r"\.cu3s"):
        Cu3sDataModule.validate_params({"cu3s_file_path": str(bad)})


def test_unknown_processing_mode_raises(mock_cuvis_sdk, tmp_path):
    import types

    import cuvis  # the fake module patched into sys.modules by the fixture

    from cuvis_ai_dataloader.data.readers.cu3s_reader import Cu3sCubeReader

    # The real ProcessingMode is an enum where an unknown name resolves to None; a Mock would
    # fabricate one, so swap in a namespace to exercise the unknown-mode guard.
    cuvis.ProcessingMode = types.SimpleNamespace(
        Raw="Raw", Reflectance="Reflectance", SpectralRadiance="SpectralRadiance"
    )
    with pytest.raises(ValueError, match="unknown processing_mode"):
        Cu3sCubeReader(_make_cu3s(tmp_path), processing_mode="Reflectence")


def test_predict_iterates_all_measurements(mock_cuvis_sdk, tmp_path):
    dm = Cu3sDataModule(cu3s_file_path=_make_cu3s(tmp_path), batch_size=1)
    dm.setup(stage="predict")
    loader = dm.predict_dataloader()
    batches = list(loader)
    assert len(batches) == 7  # mock session has 7 measurements
    batch = batches[0]
    assert set(batch.keys()) >= {"cube", "mesu_index", "wavelengths"}
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


def test_setup_fit_from_selectors(mock_cuvis_sdk, tmp_path):
    cu3s = _make_cu3s(tmp_path)
    dm = Cu3sDataModule(
        cu3s_file_path=cu3s,
        splits=DataSplitConfig(train=_fi(cu3s, [0, 2, 3]), val=_fi(cu3s, [1, 5])),
        batch_size=2,
    )
    dm.setup(stage="fit")
    assert len(dm._train_ds) == 3
    assert len(dm._val_ds) == 2
    train_batch = next(iter(dm.train_dataloader()))
    assert train_batch["cube"].shape[0] == 2


def test_nested_cfg_data_construction(mock_cuvis_sdk, tmp_path):
    # `Cu3sDataModule(**cfg.data)` with the nested DataConfig shape (data_module,
    # splits-as-dict, params) must work for config-driven (hydra) call sites.
    cfg_data = {
        "data_module": "cu3s",
        "splits": {"predict": []},
        "batch_size": 1,
        "num_workers": 0,
        "params": {"cu3s_file_path": _make_cu3s(tmp_path)},
    }
    dm = Cu3sDataModule(**cfg_data)
    dm.setup(stage="predict")
    assert len(dm._predict_ds) == 7


def test_unknown_kwarg_raises(tmp_path):
    # A removed or misspelled option must fail loudly, not be silently dropped.
    with pytest.raises(TypeError, match="train_ids"):
        Cu3sDataModule(cu3s_file_path=_make_cu3s(tmp_path), train_ids=[0, 1])


def test_data_module_passthrough_key_accepted(mock_cuvis_sdk, tmp_path):
    # The nested cfg.data shape carries `data_module`; it is accepted and ignored.
    dm = Cu3sDataModule(cu3s_file_path=_make_cu3s(tmp_path), data_module="cu3s")
    dm.setup(stage="predict")
    assert len(dm._predict_ds) == 7


def test_predict_dataset_with_measurement_indices(mock_cuvis_sdk, tmp_path):
    dm = Cu3sDataModule(cu3s_file_path=_make_cu3s(tmp_path), measurement_indices=[0, 2, 4])
    dm.setup(stage="predict")
    ds = dm.predict_ds
    assert len(ds) == 3
    item = ds[0]
    assert item["mesu_index"] == 0
    assert "cube" in item and "wavelengths" in item


def test_setup_fit_expands_range_selectors(mock_cuvis_sdk, tmp_path):
    cu3s = _make_cu3s(tmp_path)
    dm = Cu3sDataModule(
        cu3s_file_path=cu3s,
        splits=DataSplitConfig(train=_fi(cu3s, ["0-3"]), val=_fi(cu3s, [5, 6])),
        batch_size=1,
    )
    dm.setup(stage="fit")
    assert len(dm._train_ds) == 4  # "0-3" -> [0, 1, 2, 3]
    assert len(dm._val_ds) == 2


def test_measurement_indices_accepts_range(mock_cuvis_sdk, tmp_path):
    dm = Cu3sDataModule(
        cu3s_file_path=_make_cu3s(tmp_path),
        measurement_indices="0-4",  # inclusive range string
        batch_size=1,
    )
    dm.setup(stage="predict")
    assert len(dm._predict_ds) == 5  # measurements 0..4


def test_folder_source_predict_iterates_all_files(mock_cuvis_sdk, tmp_path):
    folder = _make_cu3s_folder(tmp_path, n=4)
    dm = Cu3sDataModule(data_dir=str(folder), batch_size=1)
    dm.setup(stage="predict")
    assert len(dm._predict_ds) == 4
    sample = dm._predict_ds[0]
    assert sample["stem"] == "scan_00"
    assert "cube" in sample and "wavelengths" in sample


def test_folder_source_splits_by_position_and_stem(mock_cuvis_sdk, tmp_path):
    folder = _make_cu3s_folder(tmp_path, n=5)
    dm = Cu3sDataModule(
        data_dir=str(folder),
        splits=DataSplitConfig(
            train=[
                Selector(kind=SelectorKind.DIR_INDICES, ids=[0]),
                Selector(kind=SelectorKind.STEMS, stems=["scan_02"]),
            ],
            val=_dir(["3-4"]),  # disjoint from train (scan_00 + scan_02)
        ),
        batch_size=1,
    )
    dm.setup(stage="fit")
    assert len(dm._train_ds) == 2  # position 0 + stem scan_02
    assert len(dm._val_ds) == 2  # "3-4" -> positions 3, 4


def test_folder_source_glob_filters_extensions(mock_cuvis_sdk, tmp_path):
    folder = tmp_path / "mixed"
    folder.mkdir()
    (folder / "a.cu3s").write_bytes(b"")
    (folder / "b.cu3s").write_bytes(b"")
    (folder / "note.txt").write_text("ignore me")
    dm = Cu3sDataModule(data_dir=str(folder), batch_size=1)
    dm.setup(stage="predict")
    assert len(dm._predict_ds) == 2


def test_folder_validate_params_accepts_dir_and_rejects_empty(tmp_path):
    folder = _make_cu3s_folder(tmp_path, n=2)
    Cu3sDataModule.validate_params({"data_dir": str(folder)})  # no raise
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="holds no"):
        Cu3sDataModule.validate_params({"data_dir": str(empty)})


def test_folder_unknown_selector_raises(mock_cuvis_sdk, tmp_path):
    folder = _make_cu3s_folder(tmp_path, n=2)
    dm = Cu3sDataModule(
        data_dir=str(folder),
        splits=DataSplitConfig(train=[Selector(kind=SelectorKind.STEMS, stems=["does_not_exist"])]),
    )
    with pytest.raises(ValueError, match="matched 0 samples"):
        dm.setup(stage="fit")
