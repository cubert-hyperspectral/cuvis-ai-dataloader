"""Tests for Cu3sDataModule (cu3s cubes + optional COCO masks)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai_core.data.datamodule import BaseCuvisAIDataModule, create_data_module
from cuvis_ai_dataloader.data.datamodule_cu3s import Cu3sDataModule
from cuvis_ai_schemas.training.data import (
    DataConfig,
    DataSplitConfig,
    Selector,
    SelectorKind,
)


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


def test_count_measurements_opens_one_bare_session(mock_cuvis_sdk, tmp_path):
    """The enumeration probe asks the handle for its size and nothing else.

    A full Cu3sCubeReader would also build a ProcessingContext (SDK GPU processing pools)
    and read measurement 0, per file, which is what made enumerating a folder expensive.
    """
    import sys

    from cuvis_ai_dataloader.data.readers.cu3s_reader import count_measurements

    path = _make_cu3s(tmp_path)
    assert count_measurements(path) == 7
    assert sys.modules["cuvis"].SessionFile.call_count == 1
    assert sys.modules["cuvis"].ProcessingContext.call_count == 0


def test_count_measurements_names_the_recording_it_could_not_open(mock_cuvis_sdk, tmp_path):
    """The SDK raises without the path in it, which is useless across a whole split."""
    import sys

    from cuvis_ai_dataloader.data.readers.cu3s_reader import count_measurements

    path = _make_cu3s(tmp_path, name="broken.cu3s")
    sys.modules["cuvis"].SessionFile.side_effect = RuntimeError("SDKException")
    with pytest.raises(ValueError, match="cannot open cu3s .*broken.cu3s"):
        count_measurements(path)


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


def test_nested_unknown_key_raises(tmp_path):
    # A stray/typo'd key inside the nested `params` now fails loudly, matching the flat
    # path (test_unknown_kwarg_raises) instead of being silently dropped.
    with pytest.raises(TypeError, match="bogus"):
        Cu3sDataModule(params={"cu3s_file_path": _make_cu3s(tmp_path), "bogus": 1})


def test_flat_kwarg_wins_over_params(tmp_path):
    # Standardized precedence: an explicit flat kwarg wins over the same key in `params`.
    dm = Cu3sDataModule(
        processing_mode="Raw",
        params={"processing_mode": "Reflectance", "cu3s_file_path": _make_cu3s(tmp_path)},
    )
    assert dm.processing_mode == "Raw"


def test_create_data_module_builds_cu3s(mock_cuvis_sdk, tmp_path):
    # The registry/production path spreads `params` into flat kwargs; the decorator is inert
    # there, and construction + setup succeed end to end.
    class _Reg:
        data_modules = {"cu3s": Cu3sDataModule}

    cfg = DataConfig(data_module="cu3s", params={"cu3s_file_path": _make_cu3s(tmp_path)})
    dm = create_data_module(_Reg(), cfg)
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


# --------------------------------------------------------- batched fetch + reader threads
def test_getitems_matches_per_index_reads(mock_cuvis_sdk, tmp_path):
    # torch calls __getitems__ instead of __getitem__ per index, so the two must agree.
    dm = Cu3sDataModule(cu3s_file_path=_make_cu3s(tmp_path), measurement_indices=[0, 2, 4])
    dm.setup(stage="predict")
    ds = dm.predict_ds
    indices = [2, 0, 1]
    batched = ds.__getitems__(indices)
    assert [i["mesu_index"] for i in batched] == [ds[i]["mesu_index"] for i in indices]
    assert [i["read_index"] for i in batched] == [ds[i]["read_index"] for i in indices]


def test_getitems_spans_several_recordings_in_order(mock_cuvis_sdk, tmp_path):
    folder = _make_cu3s_folder(tmp_path, n=3)
    dm = Cu3sDataModule(data_dir=str(folder), frames="measurements", batch_size=2)
    dm.setup(stage="predict")
    ds = dm.predict_ds
    indices = [0, len(ds) - 1, 1]
    assert [i["stem"] for i in ds.__getitems__(indices)] == [ds[i]["stem"] for i in indices]


def test_dataloader_uses_the_batched_hook(mock_cuvis_sdk, tmp_path):
    dm = Cu3sDataModule(cu3s_file_path=_make_cu3s(tmp_path), batch_size=3)
    dm.setup(stage="predict")
    assert hasattr(dm.predict_ds, "__getitems__")
    batch = next(iter(dm.predict_dataloader()))
    assert batch["cube"].shape[0] == 3


def test_reader_threads_rejects_process_workers(tmp_path):
    # Each worker process would build its own sessions and its own ProcessingContext.
    with pytest.raises(ValueError, match="cannot be combined with num_workers"):
        Cu3sDataModule(cu3s_file_path=_make_cu3s(tmp_path), read_threads=4, num_workers=2)


def test_reader_threads_rejects_negative(tmp_path):
    with pytest.raises(ValueError, match="read_threads must be >= 0"):
        Cu3sDataModule(cu3s_file_path=_make_cu3s(tmp_path), read_threads=-1)


def test_reader_threads_arrives_through_the_params_shape(mock_cuvis_sdk, tmp_path):
    dm = Cu3sDataModule(
        **{
            "data_module": "cu3s",
            "splits": {"predict": []},
            "batch_size": 2,
            "num_workers": 0,
            "params": {"cu3s_file_path": _make_cu3s(tmp_path), "read_threads": 4},
        }
    )
    assert dm.read_threads == 4


def test_source_coherent_batches_keep_a_batch_within_one_recording(mock_cuvis_sdk, tmp_path):
    folder = _make_cu3s_folder(tmp_path, n=3)
    dm = Cu3sDataModule(
        data_dir=str(folder),
        frames="measurements",
        batch_size=7,
        source_coherent_batches=True,
    )
    dm.setup(stage="predict")
    stems = {tuple(sorted(set(b["stem"]))) for b in dm.predict_dataloader()}
    assert all(len(s) == 1 for s in stems)


def test_source_coherent_batches_hand_the_reader_cache_the_whole_thread_budget(
    mock_cuvis_sdk, tmp_path
):
    folder = _make_cu3s_folder(tmp_path, n=4)
    coherent = Cu3sDataModule(
        data_dir=str(folder), frames="measurements", read_threads=4, source_coherent_batches=True
    )
    coherent.setup(stage="predict")
    assert coherent.predict_ds._cache._per_file_threads == 4

    divided = Cu3sDataModule(data_dir=str(folder), frames="measurements", read_threads=4)
    divided.setup(stage="predict")
    assert divided.predict_ds._cache._per_file_threads == 1


def test_samples_per_frame_with_coherent_batches_keeps_recordings_together(
    mock_cuvis_sdk, tmp_path
):
    """The repeat wrapper hides the base dataset; the coherent sampler still has to see the
    source of every repeated index, or its batches would mix recordings at random."""
    from cuvis_ai_dataloader.data.readers.cu3s_pool import SourceCoherentBatchSampler

    folder = _make_cu3s_folder(tmp_path, n=2)
    first, second = (str(p) for p in sorted(folder.glob("*.cu3s")))
    dm = Cu3sDataModule(
        data_dir=str(folder),
        frames="measurements",
        splits=DataSplitConfig(train=_fi(first, [0, 1, 2]) + _fi(second, [0, 1, 2])),
        samples_per_frame=2,
        batch_size=4,
        num_workers=0,
        source_coherent_batches=True,
    )
    dm.setup(stage="fit")
    loader = dm.train_dataloader()
    assert len(dm.train_ds) == 6
    assert len(loader.dataset) == 12
    sampler = loader.batch_sampler
    assert isinstance(sampler, SourceCoherentBatchSampler)
    assert len(sampler) == 3
    sources = loader.dataset._base.sample_sources
    batches = list(sampler)
    assert sorted(i for batch in batches for i in batch) == list(range(12))
    mixed = [b for b in batches if len({sources[i % len(sources)] for i in b}) > 1]
    assert len(mixed) <= 1, "only the batch straddling the two recordings may mix them"


# ------------------------------------------------------------------------------ read-ahead
@pytest.fixture
def releases_gil(monkeypatch):
    """Force the capability probe positive; the fake SDK never releases the GIL."""
    monkeypatch.setattr(
        "cuvis_ai_dataloader.data.readers.cu3s_pool.cuvis_releases_gil", lambda _call: True
    )


def _warnings_during(fn):
    from loguru import logger

    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        result = fn()
    finally:
        logger.remove(sink)
    return result, messages


def test_read_ahead_arrives_through_the_params_shape(mock_cuvis_sdk, tmp_path):
    dm = Cu3sDataModule(
        **{
            "data_module": "cu3s",
            "splits": {"predict": []},
            "batch_size": 1,
            "num_workers": 0,
            "params": {"cu3s_file_path": _make_cu3s(tmp_path), "read_ahead": 2},
        }
    )
    assert dm.read_ahead == 2


def test_read_ahead_rejects_process_workers(tmp_path):
    with pytest.raises(ValueError, match="read_ahead=2 cannot be combined with num_workers=2"):
        Cu3sDataModule(cu3s_file_path=_make_cu3s(tmp_path), read_ahead=2, num_workers=2)


def test_read_ahead_loader_reads_each_frame_once_in_order(mock_cuvis_sdk, tmp_path, releases_gil):
    from cuvis_ai_dataloader.data.readers.read_ahead import LookaheadBatchSampler

    dm = Cu3sDataModule(
        cu3s_file_path=_make_cu3s(tmp_path), measurement_indices=[0, 1, 2, 3, 4], read_ahead=2
    )
    dm.setup(stage="predict")
    loader = dm.predict_dataloader()
    assert isinstance(loader.batch_sampler, LookaheadBatchSampler)
    assert [int(b["mesu_index"][0]) for b in loader] == [0, 1, 2, 3, 4]
    session = mock_cuvis_sdk["session"]
    reads = [c.args[0] for c in session.get_measurement.call_args_list][1:]  # after the open
    assert sorted(reads) == [0, 1, 2, 3, 4], "every frame read exactly once, none re-read"


def test_read_ahead_keeps_the_shuffled_order_of_a_plain_loader(
    mock_cuvis_sdk, tmp_path, releases_gil
):
    cu3s = _make_cu3s(tmp_path)

    def epoch_order(read_ahead):
        dm = Cu3sDataModule(
            cu3s_file_path=cu3s,
            splits=DataSplitConfig(train=_fi(cu3s, [0, 1, 2, 3, 4, 5]), val=_fi(cu3s, [6])),
            read_ahead=read_ahead,
        )
        dm.setup(stage="fit")
        loader = dm.train_dataloader()
        torch.manual_seed(7)
        return [int(b["mesu_index"][0]) for b in loader]

    plain, ahead = epoch_order(0), epoch_order(2)
    assert sorted(plain) == [0, 1, 2, 3, 4, 5]
    assert ahead == plain, "the look-ahead sampler must draw the same permutation as torch"


def test_samples_per_frame_keeps_the_train_loader_synchronous_and_says_so(
    mock_cuvis_sdk, tmp_path, releases_gil
):
    from cuvis_ai_dataloader.data.readers.read_ahead import LookaheadBatchSampler

    cu3s = _make_cu3s(tmp_path)
    dm = Cu3sDataModule(
        cu3s_file_path=cu3s,
        splits=DataSplitConfig(train=_fi(cu3s, [0, 1, 2]), val=_fi(cu3s, [6])),
        samples_per_frame=2,
        read_ahead=2,
    )
    dm.setup(stage="fit")
    loader, messages = _warnings_during(dm.train_dataloader)
    assert not isinstance(loader.batch_sampler, LookaheadBatchSampler)
    assert len(loader.dataset) == 6
    assert any("samples_per_frame" in m and "read_ahead" in m for m in messages), messages


def test_read_ahead_composes_with_source_coherent_batches(mock_cuvis_sdk, tmp_path, releases_gil):
    from cuvis_ai_dataloader.data.readers.read_ahead import AnnouncingBatchSampler

    folder = _make_cu3s_folder(tmp_path, n=2)
    dm = Cu3sDataModule(
        data_dir=str(folder),
        frames="measurements",
        batch_size=2,
        source_coherent_batches=True,
        read_ahead=2,
    )
    dm.setup(stage="predict")
    loader = dm.predict_dataloader()
    assert isinstance(loader.batch_sampler, AnnouncingBatchSampler)
    batches = [b["stem"] for b in loader]
    assert sorted(s for b in batches for s in b) == ["scan_00"] * 7 + ["scan_01"] * 7
    mixed = [b for b in batches if len(set(b)) > 1]
    assert len(mixed) <= 1, "only the batch straddling the two recordings may mix them"


def test_an_abandoned_predict_iterator_leaves_no_frame_in_flight(
    mock_cuvis_sdk, tmp_path, releases_gil
):
    """Lightning's sanity check takes two batches and drops the iterator; the frames read ahead
    of it must not stay alive until the next epoch."""
    import gc

    dm = Cu3sDataModule(
        cu3s_file_path=_make_cu3s(tmp_path), measurement_indices=[0, 1, 2, 3, 4, 5], read_ahead=2
    )
    dm.setup(stage="predict")
    it = iter(dm.predict_dataloader())
    next(it)
    next(it)
    del it
    gc.collect()
    plan = dm.predict_ds._plan
    assert not plan._pending and not plan.active


# ---------------------------------------------------- recordings without a labels file
def _polygon_annotation():
    """One polygon on category 1: rasterizes to a nonzero block, unlike the fixture's []."""
    from types import SimpleNamespace

    return SimpleNamespace(
        id=1, category_id=1, segmentation=[[8, 8, 40, 8, 40, 40, 8, 40]], mask=None
    )


def _labelled_and_label_free_folder(tmp_path):
    """``a.cu3s`` with a sibling ``a.json`` beside ``b.cu3s`` without one; canonical sources."""
    folder = tmp_path / "mixed"
    folder.mkdir()
    (folder / "a.cu3s").write_bytes(b"")
    (folder / "a.json").write_text("{}")  # COCOData.from_path is mocked, content irrelevant
    (folder / "b.cu3s").write_bytes(b"")
    a = (folder / "a.cu3s").resolve().as_posix()
    b = (folder / "b.cu3s").resolve().as_posix()
    return folder, a, b


def test_recording_without_sidecar_yields_an_all_zero_mask(mock_cuvis_sdk, tmp_path):
    """No labels file means label-free: every frame reads as normal and carries a zero mask."""
    dm = Cu3sDataModule(cu3s_file_path=_make_cu3s(tmp_path), batch_size=1)
    dm.setup(stage="predict")
    sample = dm._predict_ds[0]
    assert "mask" in sample
    assert sample["mask"].shape == mock_cuvis_sdk["hw"]
    assert sample["mask"].dtype == np.int32
    assert not sample["mask"].any()


@pytest.mark.parametrize("order", ["labelled_first", "label_free_first"])
def test_labelled_and_label_free_recordings_collate_in_one_batch(mock_cuvis_sdk, tmp_path, order):
    """One batch holds a frame with labels and a frame without: both rows carry a mask, the
    labelled one byte for byte what its labels rasterize to, the other all zeros."""
    from unittest.mock import Mock

    from cuvis_ai_dataloader.data.labelers.coco_labeler import create_mask

    ann = _polygon_annotation()
    mock_cuvis_sdk["coco"].annotations.where = Mock(return_value=[ann])
    folder, a, b = _labelled_and_label_free_folder(tmp_path)
    selectors = _fi(a, [0]) + _fi(b, [0])
    if order == "label_free_first":
        selectors = list(reversed(selectors))
    dm = Cu3sDataModule(
        data_dir=str(folder),
        frames="measurements",
        splits=DataSplitConfig(val=selectors),
        batch_size=2,
    )
    dm.setup(stage="validate")
    batch = next(iter(dm.val_dataloader()))
    h, w = mock_cuvis_sdk["hw"]
    assert batch["mask"].shape == (2, h, w)
    rows = dict(zip(batch["stem"], batch["mask"]))
    assert set(rows) == {"a", "b"}
    expected = create_mask(annotations=[ann], image_height=h, image_width=w)
    assert expected.any(), "the labelled mock must rasterize to a nonzero mask"
    assert torch.equal(rows["a"], torch.from_numpy(expected))
    assert not rows["b"].any()


def test_setup_warns_once_per_stage_for_recordings_without_a_labels_file(mock_cuvis_sdk, tmp_path):
    """One WARNING per val/test stage the call built, naming the recordings; never for train
    or predict, and no memory across calls."""
    folder, a, b = _labelled_and_label_free_folder(tmp_path)
    dm = Cu3sDataModule(
        data_dir=str(folder),
        frames="measurements",
        splits=DataSplitConfig(
            train=_fi(a, [0, 1, 2]) + _fi(b, [0, 1, 2]),
            val=_fi(a, [3, 4]) + _fi(b, [3, 4]),
            test=_fi(b, [5, 6]),
        ),
    )

    def label_free_warnings(stage):
        _, messages = _warnings_during(lambda: dm.setup(stage=stage))
        return [m for m in messages if "no labels file" in m]

    fit = label_free_warnings("fit")
    assert len(fit) == 1, fit
    assert "val: 1 of 2 recordings have no labels file" in fit[0]
    assert "b.cu3s" in fit[0] and "a.cu3s" not in fit[0] and "train" not in fit[0]
    assert "label-free" in fit[0] and "normal" in fit[0]

    test = label_free_warnings("test")
    assert len(test) == 1, test
    assert "test: 1 of 1 recordings have no labels file" in test[0]

    assert label_free_warnings("predict") == []
    assert len(label_free_warnings("fit")) == 1  # the stages this call built, again


def test_enumerate_tags_a_recording_without_a_labels_file_as_normal(mock_cuvis_sdk, tmp_path):
    """A label-free frame carries the normal tag, so a tag selector asking for normal frames
    finds it; the labelled recording reads anomalous here because the mock annotates it."""
    from unittest.mock import Mock

    mock_cuvis_sdk["coco"].annotations.where = Mock(return_value=[_polygon_annotation()])
    folder, a, b = _labelled_and_label_free_folder(tmp_path)
    refs = Cu3sDataModule(data_dir=str(folder), frames="measurements").enumerate(
        frozenset({"tags", "category_ids"})
    )
    label_free = [r for r in refs if r.source == b]
    assert len(label_free) == 7
    assert all(r.tags == ["normal"] and r.category_ids == [] for r in label_free)
    assert all(r.tags == ["anomalous"] for r in refs if r.source == a)

    dm = Cu3sDataModule(
        data_dir=str(folder),
        frames="measurements",
        splits=DataSplitConfig(val=[Selector(kind=SelectorKind.TAG, any_of=["normal"])]),
    )
    dm.setup(stage="validate")
    assert dm._val_ds.sample_sources == [b] * 7
