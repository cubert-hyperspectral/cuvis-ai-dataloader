"""Device-resident cubes (``cuda_cubes``): gating, plumbing, and the two guards.

Runs on the fake SDK from ``conftest``, which carries no ``cuvis.cuda``, so the default here
is the unavailable path -- the one every machine without the SDK takes. ``cuda`` is added
explicitly where the available path is under test. That a device tensor actually comes back
is a real-SDK claim and lives in ``test_cuda_cubes_integration.py``.
"""

from __future__ import annotations

import types
from unittest.mock import Mock

import pytest

from cuvis_ai_dataloader.data.datamodule_cu3s import Cu3sDataModule
from cuvis_ai_dataloader.data.readers import cu3s_cuda
from cuvis_ai_dataloader.data.readers.cu3s_reader import Cu3sCubeReader


@pytest.fixture
def cu3s(tmp_path):
    """An empty file with a .cu3s suffix, which is all the reader validates."""
    path = tmp_path / "session.cu3s"
    path.write_bytes(b"")
    return str(path)


@pytest.fixture
def cuda_capable(monkeypatch):
    """Give the fake SDK a cuvis.cuda that reports the same-process path as usable."""
    import cuvis

    cuda = types.ModuleType("cuvis.cuda")
    cuda.capabilities = Mock(return_value=types.SimpleNamespace(same_process=True))
    cuda.enable = Mock()
    monkeypatch.setattr(cuvis, "cuda", cuda, raising=False)
    monkeypatch.setattr(cu3s_cuda, "_repair_binding", lambda: True)
    return cuda


def test_an_sdk_without_cuda_support_falls_back_rather_than_raising(mock_cuvis_sdk, cu3s):
    """cuvis.cuda arrived in 3.6.0; an older binding must degrade, not crash."""
    reader = Cu3sCubeReader(cu3s, cuda_cubes=True)
    try:
        assert reader.cuda_cubes is False
        assert cu3s_cuda.is_enabled() is False
    finally:
        reader.close()


def test_a_device_the_sdk_rejects_falls_back(mock_cuvis_sdk, cu3s, cuda_capable):
    """capabilities() is the SDK's verdict on the hardware, and it is allowed to say no."""
    cuda_capable.capabilities.return_value = types.SimpleNamespace(same_process=False)
    reader = Cu3sCubeReader(cu3s, cuda_cubes=True)
    try:
        assert reader.cuda_cubes is False
        cuda_capable.enable.assert_not_called()
    finally:
        reader.close()


def test_a_binding_without_the_device_view_falls_back(
    mock_cuvis_sdk, cu3s, cuda_capable, monkeypatch
):
    """capabilities() probes the native symbols and cannot see the Python glue over them."""
    monkeypatch.setattr(cu3s_cuda, "_repair_binding", lambda: False)
    reader = Cu3sCubeReader(cu3s, cuda_cubes=True)
    try:
        assert reader.cuda_cubes is False
        cuda_capable.enable.assert_not_called()
    finally:
        reader.close()


def test_the_mode_is_switched_on_before_the_session_is_opened(mock_cuvis_sdk, cu3s, cuda_capable):
    """The SDK reads the flag while filling a measurement, so afterwards would be too late."""
    import cuvis

    order = []
    session = mock_cuvis_sdk["session"]
    cuda_capable.enable.side_effect = lambda: order.append("cuda.enable")
    cuvis.SessionFile = Mock(side_effect=lambda *a, **k: (order.append("SessionFile"), session)[1])

    Cu3sCubeReader(cu3s, cuda_cubes=True, processing_mode=None).close()

    assert order[:2] == ["cuda.enable", "SessionFile"]


def test_the_mode_is_switched_on_once_per_process(mock_cuvis_sdk, cu3s, cuda_capable):
    """It is process-global on the SDK's side; re-enabling per reader would be noise."""
    for _ in range(3):
        Cu3sCubeReader(cu3s, cuda_cubes=True).close()
    assert cuda_capable.enable.call_count == 1


def test_off_by_default(mock_cuvis_sdk, cu3s, cuda_capable):
    """It changes the batch's dtype and device, so it cannot be a silent default."""
    reader = Cu3sCubeReader(cu3s)
    try:
        assert reader.cuda_cubes is False
        cuda_capable.enable.assert_not_called()
    finally:
        reader.close()


# ------------------------------------------------------------------------------ the guards
def test_device_cubes_require_the_device(mock_cuvis_sdk, cu3s):
    """Processing on the host leaves no device buffer to hand out."""
    with pytest.raises(ValueError, match="requires sdk_cuda=True"):
        Cu3sDataModule(cu3s_file_path=cu3s, cuda_cubes=True, sdk_cuda=False)


def test_device_cubes_refuse_worker_processes(mock_cuvis_sdk, cu3s):
    """A CUDA tensor does not cross the worker queue, so this is refused, not demoted."""
    with pytest.raises(ValueError, match="num_workers=2"):
        Cu3sDataModule(cu3s_file_path=cu3s, cuda_cubes=True, num_workers=2)


@pytest.mark.parametrize(("given", "expected"), [("true", True), ("false", False)])
def test_a_data_arg_string_is_coerced(mock_cuvis_sdk, cu3s, cuda_capable, given, expected):
    """--data-arg cuda_cubes=true arrives as a string."""
    assert Cu3sDataModule(cu3s_file_path=cu3s, cuda_cubes=given).cuda_cubes is expected


def test_the_flag_reaches_the_reader_cache_through_the_datamodule(
    mock_cuvis_sdk, cu3s, cuda_capable
):
    """The whole point: a dataset built by the module opens its readers in device mode."""
    dm = Cu3sDataModule(cu3s_file_path=cu3s, cuda_cubes=True)
    dm.setup(stage="predict")
    dataset = dm.predict_dataloader().dataset
    assert dataset._cache._cuda_cubes is True
