"""SDK device selection: ``sdk_cuda`` -> ``cuvis.init(SdkSettings(force_gpu_mode=...))``.

Runs entirely on the fake SDK from ``conftest``, whose ``init`` and ``SdkSettings`` are Mocks,
so these assert *what the package asks the SDK for* rather than what the SDK then does. The
device actually taking effect is a real-SDK claim and lives in the integration test.
"""

from __future__ import annotations

import pickle

import pytest
from loguru import logger

from cuvis_ai_dataloader.data import _extras
from cuvis_ai_dataloader.data._extras import configure_cuvis_sdk, require_cuvis
from cuvis_ai_dataloader.data.datamodule_cu3s import Cu3sDataModule
from cuvis_ai_dataloader.data.readers.cu3s_pool import Cu3sReaderCache, open_reader


@pytest.fixture
def cu3s(tmp_path):
    """An empty file with a .cu3s suffix, which is all the reader validates."""
    path = tmp_path / "session.cu3s"
    path.write_bytes(b"")
    return str(path)


def _gpu_mode_of(fake_cuvis):
    """The force_gpu_mode the package passed to SdkSettings."""
    return fake_cuvis.SdkSettings.call_args.kwargs["force_gpu_mode"]


def test_cuda_true_asks_the_sdk_for_the_gpu(mock_cuvis_sdk):
    """The flag reaches the SDK as force_gpu_mode=cuda."""
    import cuvis

    configure_cuvis_sdk(cuda=True)
    require_cuvis()
    assert cuvis.init.call_count == 1
    assert _gpu_mode_of(cuvis) == "cuda"


def test_cuda_false_asks_the_sdk_for_the_host(mock_cuvis_sdk):
    """Turning the flag off selects the SDK's host mode rather than skipping the init."""
    import cuvis

    configure_cuvis_sdk(cuda=False)
    require_cuvis()
    assert _gpu_mode_of(cuvis) == "host"


def test_the_sdk_is_initialized_once_however_often_it_is_required(mock_cuvis_sdk):
    """require_cuvis is on every read path, so a per-call init would be a per-cube init."""
    import cuvis

    configure_cuvis_sdk(cuda=True)
    for _ in range(5):
        require_cuvis()
    assert cuvis.init.call_count == 1


def test_no_configure_leaves_the_sdk_untouched(mock_cuvis_sdk):
    """A host application that initialized the SDK itself keeps whatever device it chose."""
    import cuvis

    require_cuvis()
    assert cuvis.init.call_count == 0


def test_a_conflicting_second_choice_warns_and_does_not_reinitialize(mock_cuvis_sdk):
    """The SDK fixes its device at the first init, so a later disagreement cannot win."""
    import cuvis

    configure_cuvis_sdk(cuda=True)
    require_cuvis()
    configure_cuvis_sdk(cuda=False)
    require_cuvis()
    assert cuvis.init.call_count == 1
    assert _gpu_mode_of(cuvis) == "cuda"
    assert _extras._applied_gpu_mode == "cuda"


def _warnings_during(fn):
    """Loguru WARNING messages emitted while ``fn`` runs."""
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        fn()
    finally:
        logger.remove(sink)
    return messages


def test_a_disagreement_before_the_first_init_warns_and_the_later_request_wins(mock_cuvis_sdk):
    """Nothing has reached the SDK yet, so the later choice can still take effect; letting it
    win silently would make the last-constructed DataModule decide for the whole process."""
    import cuvis

    def disagree():
        configure_cuvis_sdk(cuda=True)
        configure_cuvis_sdk(cuda=False)

    messages = _warnings_during(disagree)
    assert any("after an earlier request for 'cuda'" in m for m in messages), messages
    require_cuvis()
    assert cuvis.init.call_count == 1
    assert _gpu_mode_of(cuvis) == "host"


def test_repeating_the_same_choice_before_the_first_init_does_not_warn(mock_cuvis_sdk):
    def agree():
        configure_cuvis_sdk(cuda=True)
        configure_cuvis_sdk(cuda=True)

    assert not _warnings_during(agree)


def test_repeating_the_same_choice_is_not_a_conflict(mock_cuvis_sdk):
    """Two DataModules agreeing on the device must not warn at each other."""
    import cuvis

    configure_cuvis_sdk(cuda=True)
    require_cuvis()
    configure_cuvis_sdk(cuda=True)
    assert cuvis.init.call_count == 1
    assert _extras._applied_gpu_mode == "cuda"


def test_open_reader_selects_the_device_before_opening_a_session(mock_cuvis_sdk, cu3s):
    """The choice has to be made in the DataLoader worker too, and open_reader runs there."""
    import cuvis

    with open_reader(cu3s, sdk_cuda=False):
        pass
    assert _gpu_mode_of(cuvis) == "host"


def test_the_cache_carries_the_choice_into_a_worker(mock_cuvis_sdk, cu3s):
    """A worker is a fresh process: the flag only survives if it pickles with the cache."""
    import cuvis

    cache = Cu3sReaderCache(processing_mode="Raw", sdk_cuda=False)
    revived = pickle.loads(pickle.dumps(cache))
    assert cuvis.init.call_count == 0  # nothing opened yet, so nothing initialized yet
    revived.get(cu3s)
    try:
        assert _gpu_mode_of(cuvis) == "host"
    finally:
        revived.close()


def test_the_cache_defaults_to_the_gpu(mock_cuvis_sdk, cu3s):
    """The default restores what the SDK did on its own before 3.6.0."""
    import cuvis

    cache = Cu3sReaderCache(processing_mode="Raw")
    cache.get(cu3s)
    try:
        assert _gpu_mode_of(cuvis) == "cuda"
    finally:
        cache.close()


# ------------------------------------------------------------------ the DataModule parameter
def test_the_datamodule_defaults_to_the_gpu(mock_cuvis_sdk, cu3s):
    """Nothing configured means the GPU, not the SDK 3.6.0 host default."""
    import cuvis

    Cu3sDataModule(cu3s_file_path=cu3s)
    require_cuvis()
    assert _gpu_mode_of(cuvis) == "cuda"


def test_the_datamodule_selects_the_device_before_anything_opens_a_session(mock_cuvis_sdk, cu3s):
    """Construction is the last moment the choice can still win, so it happens there."""
    import cuvis

    Cu3sDataModule(cu3s_file_path=cu3s, sdk_cuda=False)
    require_cuvis()
    assert _gpu_mode_of(cuvis) == "host"


@pytest.mark.parametrize(
    ("given", "expected"),
    [("false", False), ("0", False), ("no", False), ("true", True), ("on", True)],
)
def test_a_data_arg_string_is_coerced(mock_cuvis_sdk, cu3s, given, expected):
    """--data-arg sdk_cuda=false arrives as a string and must not read as truthy."""
    assert Cu3sDataModule(cu3s_file_path=cu3s, sdk_cuda=given).sdk_cuda is expected


def test_an_unparseable_flag_fails_loudly(mock_cuvis_sdk, cu3s):
    """A typo must not silently select the wrong device."""
    with pytest.raises(ValueError, match="sdk_cuda"):
        Cu3sDataModule(cu3s_file_path=cu3s, sdk_cuda="maybe")


def test_the_flag_arrives_through_the_params_shape(mock_cuvis_sdk, cu3s):
    """The nested DataConfig shape is how a training yaml reaches the module."""
    dm = Cu3sDataModule(
        **{
            "data_module": "cu3s",
            "splits": {"predict": []},
            "batch_size": 1,
            "num_workers": 0,
            "params": {"cu3s_file_path": cu3s, "sdk_cuda": "false"},
        }
    )
    assert dm.sdk_cuda is False
