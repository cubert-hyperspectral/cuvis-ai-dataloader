"""Real-SDK integration parity for threaded cu3s reading.

Confirms against the **real** cuvis SDK and real ``.cu3s`` data that reading on a pool of
session handles produces cubes byte-identical to reading them one at a time, and that the
pool really shares one ``ProcessingContext`` rather than letting each handle build its own.
The mocked unit tests in ``test_cu3s_pool.py`` cannot cover either: the fake SDK returns one
session object for every open, and it processes nothing.

Gated and skipped by default (CI has neither the SDK nor sample data). To run, install the
machine-level cuvis SDK plus this package's ``[cu3s]`` extra and point the env var at a
session::

    export CUVIS_AI_IT_TARGET=/path/to/scene.cu3s   # needs >= 4 measurements
    # optional: CUVIS_AI_IT_MODE (default Raw, avoids needing references)
    pytest -m integration \
        tests/cuvis_ai_dataloader/data/test_threaded_reading_integration.py

A stock binding holds the GIL for every SDK call, so the pool is disabled there and the
parity tests still pass, just without any overlap. ``test_binding_releases_the_gil`` is the
one that tells you which binding you are on.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from cuvis_ai_dataloader.data._extras import cuvis_releases_gil
from cuvis_ai_dataloader.data.readers.cu3s_pool import Cu3sPrefetchReader
from cuvis_ai_dataloader.data.readers.cu3s_reader import Cu3sCubeReader

_TARGET = os.environ.get("CUVIS_AI_IT_TARGET")
_MODE = os.environ.get("CUVIS_AI_IT_MODE", "Raw")
_FRAMES = 4

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _TARGET,
        reason="set CUVIS_AI_IT_TARGET (needs the real cuvis SDK + .cu3s data)",
    ),
]


@pytest.fixture(scope="module")
def reference_cubes():
    """The first few cubes read one at a time, as the parity baseline."""
    pytest.importorskip("cuvis")
    reader = Cu3sCubeReader(_TARGET, processing_mode=_MODE)
    try:
        if reader.total_measurements < _FRAMES:
            pytest.skip(f"{_TARGET} has fewer than {_FRAMES} measurements")
        return [reader.read(i)["cube"].copy() for i in range(_FRAMES)]
    finally:
        reader.close()


def test_pooled_reads_are_byte_identical_to_single_threaded(reference_cubes):
    reader = Cu3sPrefetchReader(_TARGET, threads=4, processing_mode=_MODE)
    try:
        cubes = [item["cube"] for item in reader.read_many(range(_FRAMES))]
        assert len(cubes) == len(reference_cubes)
        for index, (got, expected) in enumerate(zip(cubes, reference_cubes)):
            assert np.array_equal(got, expected), f"cube {index} differs under 4 threads"
    finally:
        reader.close()


def test_pooled_reads_hold_up_when_the_same_frames_are_read_repeatedly(reference_cubes):
    # Re-reading the same window is what a shuffled epoch does, and it is where a shared
    # context would show up as a wrong cube rather than as an error.
    reader = Cu3sPrefetchReader(_TARGET, threads=4, processing_mode=_MODE)
    try:
        order = [i % _FRAMES for i in range(_FRAMES * 3)]
        for index, item in zip(order, reader.read_many(order)):
            assert np.array_equal(item["cube"], reference_cubes[index])
    finally:
        reader.close()


def test_the_pool_opens_one_handle_per_thread_and_one_context():
    cuvis = pytest.importorskip("cuvis")
    reader = Cu3sPrefetchReader(_TARGET, threads=3, processing_mode=_MODE)
    try:
        assert len(reader._sessions) == 3
        assert all(isinstance(session, cuvis.SessionFile) for session in reader._sessions)
        # The lazy Measurement.cube path would otherwise give each handle its own context.
        assert all(session._pc is reader.pc for session in reader._sessions)
    finally:
        reader.close()


def test_binding_releases_the_gil():
    """Report which binding is installed; a stock one silently disables the pool."""
    reader = Cu3sCubeReader(_TARGET, processing_mode=_MODE)
    try:
        releases = cuvis_releases_gil(lambda: reader.read(0))
    finally:
        reader.close()
    if not releases:
        pytest.skip("this cuvis binding holds the GIL; reader threads cannot help on it")
