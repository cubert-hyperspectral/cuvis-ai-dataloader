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


def test_dataset_exposes_wavelengths(mock_cuvis_sdk, tmp_path):
    # Consumers read the wavelength axis once off the dataset (no per-item iteration).
    dm = Cu3sDataModule(cu3s_file_path=_make_cu3s(tmp_path), batch_size=1)
    dm.setup(stage="predict")
    wl = dm.predict_ds.wavelengths_nm
    assert len(wl) > 0
    assert list(dm.predict_ds.wavelengths) == list(wl)  # back-compat alias


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


def test_splitless_training_stages_refused(mock_cuvis_sdk, tmp_path):
    # Without DataConfig.splits, fit/validate/test would silently feed the whole universe
    # (incl. anomalous frames) into statistical init; the module refuses instead.
    dm = Cu3sDataModule(cu3s_file_path=_make_cu3s(tmp_path))
    for stage in ("fit", "validate", "test"):
        with pytest.raises(ValueError, match="does not own split semantics"):
            dm.setup(stage=stage)


def test_splitless_setup_none_builds_predict_only(mock_cuvis_sdk, tmp_path):
    dm = Cu3sDataModule(cu3s_file_path=_make_cu3s(tmp_path))
    dm.setup()  # stage=None: whole-universe predict is the one valid split-less dataset
    assert len(dm._predict_ds) == 7
    assert dm._train_ds is None and dm._val_ds is None and dm._test_ds is None


def test_folder_frames_measurements_enumerates_per_measurement(mock_cuvis_sdk, tmp_path):
    folder = _make_cu3s_folder(tmp_path, n=2)
    dm = Cu3sDataModule(data_dir=str(folder), frames="measurements")
    refs = dm.enumerate()
    assert len(refs) == 2 * 7  # mock session has 7 measurements per file
    assert [r.index for r in refs[:7]] == list(range(7))
    assert all("\\" not in r.source for r in refs)  # canonical forward-slash sources


def test_folder_frames_file_default_unchanged(mock_cuvis_sdk, tmp_path):
    folder = _make_cu3s_folder(tmp_path, n=3)
    refs = Cu3sDataModule(data_dir=str(folder)).enumerate()
    assert [(r.index, r.label_id) for r in refs] == [(0, 0)] * 3  # legacy one-ref-per-file


def test_folder_recursive_walks_subfolders(mock_cuvis_sdk, tmp_path):
    root = tmp_path / "dataset"
    for rel in ("day2/a.cu3s", "day3/b.cu3s"):
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"")
    dm = Cu3sDataModule(data_dir=str(root), frames="measurements", recursive=True)
    assert len(dm.enumerate()) == 2 * 7
    with pytest.raises(FileNotFoundError):
        Cu3sDataModule(data_dir=str(root)).enumerate()  # non-recursive finds nothing


def test_frames_param_via_nested_params(mock_cuvis_sdk, tmp_path):
    folder = _make_cu3s_folder(tmp_path, n=2)
    dm = Cu3sDataModule(params={"data_dir": str(folder), "frames": "measurements"})
    assert dm.frames == "measurements"
    assert len(dm.enumerate()) == 2 * 7


def test_invalid_frames_rejected(tmp_path):
    folder = _make_cu3s_folder(tmp_path, n=1)
    with pytest.raises(ValueError, match="frames"):
        Cu3sDataModule(data_dir=str(folder), frames="frame")
    with pytest.raises(ValueError, match="frames"):
        Cu3sDataModule.validate_params({"data_dir": str(folder), "frames": "frame"})
    with pytest.raises(ValueError, match="recursive"):
        Cu3sDataModule.validate_params({"data_dir": str(folder), "recursive": "yes-please"})
