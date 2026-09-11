"""Threaded cu3s reading: handle pooling, ordering, teardown, gating, and the reader cache.

Runs entirely on the fake SDK from ``conftest``, so it needs no binding and no data. Note the
fake returns the *same* session object from every ``SessionFile(...)`` call, so handle
multiplicity is only observable through the constructor's call count.
"""

from __future__ import annotations

import pytest

from cuvis_ai_dataloader.data.readers.cu3s_pool import (
    Cu3sPrefetchReader,
    Cu3sReaderCache,
    SourceCoherentBatchSampler,
    open_reader,
)
from cuvis_ai_dataloader.data.readers.cu3s_reader import Cu3sCubeReader


@pytest.fixture
def cu3s(tmp_path):
    """An empty file with a .cu3s suffix, which is all the reader validates."""
    path = tmp_path / "session.cu3s"
    path.write_bytes(b"")
    return str(path)


@pytest.fixture
def releases_gil(monkeypatch):
    """Force the capability probe positive; the fake SDK never releases the GIL."""
    monkeypatch.setattr(
        "cuvis_ai_dataloader.data.readers.cu3s_pool.cuvis_releases_gil", lambda _call: True
    )


# --------------------------------------------------------------------------- the pool
def test_one_handle_per_thread_and_a_single_shared_context(mock_cuvis_sdk, cu3s):
    import cuvis

    reader = Cu3sPrefetchReader(cu3s, threads=4)
    try:
        assert cuvis.SessionFile.call_count == 4
        # The whole point of the topology: the expensive context is built once, not per handle.
        assert cuvis.ProcessingContext.call_count == 1
        assert reader.threads == 4
    finally:
        reader.close()


def test_every_handle_points_at_the_shared_context(mock_cuvis_sdk, cu3s):
    reader = Cu3sPrefetchReader(cu3s, threads=3)
    try:
        assert reader._sessions and all(s._pc is reader.pc for s in reader._sessions)
    finally:
        reader.close()


def test_read_many_preserves_the_requested_order(mock_cuvis_sdk, cu3s):
    reader = Cu3sPrefetchReader(cu3s, threads=4)
    try:
        assert [item["mesu_index"] for item in reader.read_many([3, 1, 2, 0])] == [3, 1, 2, 0]
    finally:
        reader.close()


def test_read_many_matches_single_index_reads(mock_cuvis_sdk, cu3s):
    reader = Cu3sPrefetchReader(cu3s, threads=4)
    try:
        indices = [0, 2, 4, 1]
        assert [i["mesu_index"] for i in reader.read_many(indices)] == [
            reader.read(i)["mesu_index"] for i in indices
        ]
    finally:
        reader.close()


def test_iter_reads_does_not_submit_the_whole_range(mock_cuvis_sdk, cu3s, monkeypatch):
    reader = Cu3sPrefetchReader(cu3s, threads=1, queue_depth=2)
    try:
        seen: list[int] = []
        original = Cu3sPrefetchReader._read_with

        def _tracked(self, session, mesu_index):
            seen.append(mesu_index)
            return original(self, session, mesu_index)

        monkeypatch.setattr(Cu3sPrefetchReader, "_read_with", _tracked)
        reads = reader.iter_reads(range(6))
        assert next(reads)["mesu_index"] == 0
        # Bounded at queue_depth, so consuming one frame must not have read all six.
        assert 1 <= len(seen) <= 2
    finally:
        reader.close()


def test_a_failing_read_names_its_measurement_and_returns_the_handle(
    mock_cuvis_sdk, cu3s, monkeypatch
):
    reader = Cu3sPrefetchReader(cu3s, threads=2)
    try:
        original = Cu3sPrefetchReader._read_with

        def _explode_on_two(self, session, mesu_index):
            if mesu_index == 2:
                raise RuntimeError("sdk said no")
            return original(self, session, mesu_index)

        monkeypatch.setattr(Cu3sPrefetchReader, "_read_with", _explode_on_two)
        with pytest.raises(RuntimeError) as excinfo:
            reader.read_many([0, 1, 2])
        assert any("measurement 2" in note for note in excinfo.value.__notes__)

        monkeypatch.undo()
        # The lease was returned in the finally, so the pool is intact and still full.
        assert [i["mesu_index"] for i in reader.read_many([0, 1])] == [0, 1]
    finally:
        reader.close()


def test_close_drops_every_handle_and_is_idempotent(mock_cuvis_sdk, cu3s):
    reader = Cu3sPrefetchReader(cu3s, threads=3)
    reader.close()
    assert reader._sessions == []
    assert reader._pool is None
    assert reader._leases.empty()
    reader.close()  # must not raise


def test_disable_pool_keeps_the_reader_usable(mock_cuvis_sdk, cu3s):
    reader = Cu3sPrefetchReader(cu3s, threads=4)
    try:
        reader.disable_pool()
        assert reader._pool is None
        assert reader._sessions == [reader.session]
        assert [i["mesu_index"] for i in reader.read_many([1, 0])] == [1, 0]
    finally:
        reader.close()


# --------------------------------------------------------------------------- the gate
def test_threading_off_returns_a_plain_reader(mock_cuvis_sdk, cu3s):
    import cuvis

    reader = open_reader(cu3s, read_threads=0)
    try:
        assert type(reader) is Cu3sCubeReader
        assert cuvis.SessionFile.call_count == 1
    finally:
        reader.close()


def test_a_gil_holding_binding_falls_back_instead_of_raising(mock_cuvis_sdk, cu3s, monkeypatch):
    # The probe has to be forced: the fake SDK is pure Python, which does interleave with
    # other threads, so it cannot stand in for a C call that holds the GIL.
    monkeypatch.setattr(
        "cuvis_ai_dataloader.data.readers.cu3s_pool.cuvis_releases_gil", lambda _call: False
    )
    reader = open_reader(cu3s, read_threads=4)
    try:
        assert isinstance(reader, Cu3sPrefetchReader)
        assert reader._pool is None  # pooled reads disabled, reader still works
        assert reader.read(0)["mesu_index"] == 0
    finally:
        reader.close()


def test_the_probe_separates_a_gil_holding_call_from_a_releasing_one(monkeypatch):
    """The gate's actual discriminator, checked without the SDK.

    ``math.factorial`` of a large number is one C call that never drops the GIL, which is
    exactly the shape of an SDK call on a stock binding; ``sleep`` is the releasing shape.
    """
    import math
    import time

    from cuvis_ai_dataloader.data import _extras

    monkeypatch.setattr(_extras, "_RELEASES_GIL", None)
    assert _extras.cuvis_releases_gil(lambda: math.factorial(150_000)) is False

    monkeypatch.setattr(_extras, "_RELEASES_GIL", None)
    assert _extras.cuvis_releases_gil(lambda: time.sleep(0.005)) is True

    # Cached per process: the second call must not re-probe.
    assert _extras.cuvis_releases_gil(lambda: math.factorial(150_000)) is True


def test_a_releasing_binding_keeps_the_pool(mock_cuvis_sdk, cu3s, releases_gil):
    reader = open_reader(cu3s, read_threads=4)
    try:
        assert isinstance(reader, Cu3sPrefetchReader)
        assert reader._pool is not None
    finally:
        reader.close()


def test_too_many_threads_is_rejected(mock_cuvis_sdk, cu3s):
    with pytest.raises(ValueError, match="read_threads must be <= 16"):
        open_reader(cu3s, read_threads=17)


# --------------------------------------------------------------------------- the cache
def test_the_thread_budget_is_split_across_the_sources_held_open():
    one = Cu3sReaderCache(processing_mode=None, max_open_sessions=4, read_threads=6, sources=1)
    many = Cu3sReaderCache(processing_mode=None, max_open_sessions=4, read_threads=6, sources=4)
    # One recording spends the whole budget inside itself; four spend it across them.
    assert one._per_file_threads == 6
    assert many._per_file_threads == 1


def test_cache_rejects_nonsense_limits():
    with pytest.raises(ValueError, match="max_open_sessions"):
        Cu3sReaderCache(processing_mode=None, max_open_sessions=0)
    with pytest.raises(ValueError, match="read_threads"):
        Cu3sReaderCache(processing_mode=None, read_threads=-1)


def test_cache_read_many_keeps_caller_order_across_recordings(
    mock_cuvis_sdk, tmp_path, releases_gil
):
    paths = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.cu3s"
        path.write_bytes(b"")
        paths.append(str(path))
    cache = Cu3sReaderCache(processing_mode=None, max_open_sessions=2, read_threads=4, sources=2)
    try:
        positions = [(paths[0], 1), (paths[1], 3), (paths[0], 0), (paths[1], 2)]
        assert [i["mesu_index"] for i in cache.read_many(positions)] == [1, 3, 0, 2]
    finally:
        cache.close()


def test_cache_evicts_and_closes_the_oldest_reader(mock_cuvis_sdk, tmp_path):
    paths = []
    for name in ("a", "b", "c"):
        path = tmp_path / f"{name}.cu3s"
        path.write_bytes(b"")
        paths.append(str(path))
    cache = Cu3sReaderCache(processing_mode=None, max_open_sessions=2, sources=3)
    try:
        first = cache.get(paths[0])
        cache.get(paths[1])
        cache.get(paths[2])  # evicts paths[0]
        assert list(cache._readers) == paths[1:]
        assert first.session is None  # closed on eviction
    finally:
        cache.close()
        assert not cache._readers


# ------------------------------------------------------------------------- the sampler
def test_source_coherent_batches_stay_within_a_recording():
    sources = ["a", "b", "a", "b", "a", "b"]
    batches = list(SourceCoherentBatchSampler(sources, 3, shuffle=False))
    assert sorted(i for batch in batches for i in batch) == list(range(6))
    assert all(len({sources[i] for i in batch}) == 1 for batch in batches)


def test_sampler_covers_the_epoch_once_when_shuffled():
    sources = ["a"] * 5 + ["b"] * 4
    sampler = SourceCoherentBatchSampler(sources, 2, shuffle=True, seed=7)
    assert len(sampler) == 5
    assert sorted(i for batch in sampler for i in batch) == list(range(9))


def test_sampler_drop_last_discards_only_the_short_batch():
    sampler = SourceCoherentBatchSampler(["a"] * 5, 2, shuffle=False, drop_last=True)
    batches = list(sampler)
    assert len(sampler) == 2
    assert [len(b) for b in batches] == [2, 2]
