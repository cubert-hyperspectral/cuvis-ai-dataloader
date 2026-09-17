"""The reader's per-measurement contract on the fake SDK from ``conftest``."""

from __future__ import annotations

import numpy as np
import pytest

from cuvis_ai_dataloader.data.readers.cu3s_reader import Cu3sCubeReader


@pytest.fixture
def cu3s(tmp_path):
    """An empty file with a .cu3s suffix, which is all the reader validates."""
    path = tmp_path / "session.cu3s"
    path.write_bytes(b"")
    return str(path)


def test_a_past_the_end_index_raises_index_error(mock_cuvis_sdk, cu3s):
    """The SDK hands back None for an index past the last measurement; applying the processing
    context to that None used to fail with an SDKException about the wrong thing."""
    session = mock_cuvis_sdk["session"]
    measurement = mock_cuvis_sdk["measurement"]
    session.get_measurement.side_effect = lambda i, *a, **k: measurement if i < 7 else None
    reader = Cu3sCubeReader(cu3s, processing_mode="Raw")
    try:
        assert reader.read(6)["mesu_index"] == 6
        with pytest.raises(IndexError, match=r"measurement 7 is out of range .*(7 measurements)"):
            reader.read(7)
    finally:
        reader.close()


def test_wavelengths_nm_is_the_open_time_capture_not_a_read(mock_cuvis_sdk, cu3s):
    """A cube read per call was the cost; the wavelengths captured at open answer the same."""
    session = mock_cuvis_sdk["session"]
    reader = Cu3sCubeReader(cu3s, processing_mode="Raw")
    try:
        reads_at_open = session.get_measurement.call_count
        wavelengths = reader.wavelengths_nm
        assert session.get_measurement.call_count == reads_at_open
        assert wavelengths is reader.wavelengths
    finally:
        reader.close()


def test_wavelengths_are_int32_nanometres(mock_cuvis_sdk, cu3s):
    """The first-measurement capture and wavelengths_nm agree on dtype."""
    reader = Cu3sCubeReader(cu3s, processing_mode="Raw")
    try:
        assert reader.wavelengths.dtype == np.int32
        assert reader.wavelengths_nm.dtype == np.int32
        assert reader.read(0)["wavelengths"].dtype == np.int32
    finally:
        reader.close()
