"""Read-ahead at batch 1: the plan, the look-ahead samplers, the pool's futures, the cache's pins.

Runs on the fake SDK from ``conftest``. The fake returns instantly, so every claim about
overlap or bounds blocks ``get_measurement`` on a ``threading.Event`` and counts calls, and the
fake hands out the same session object from every ``SessionFile(...)`` call, so handles are
only observable through the constructor's call count. That a device tensor is correctly
ordered across threads is a real-SDK claim and lives in the integration tests.
"""

from __future__ import annotations

import threading
import time
import types
from unittest.mock import Mock

import pytest
import torch
from loguru import logger
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler

from cuvis_ai_dataloader.data.readers import cu3s_pool
from cuvis_ai_dataloader.data.readers.cu3s_pool import (
    Cu3sPrefetchReader,
    Cu3sReaderCache,
    SourceCoherentBatchSampler,
)
from cuvis_ai_dataloader.data.readers.read_ahead import (
    AnnouncingBatchSampler,
    LookaheadBatchSampler,
    ReadAheadPlan,
)


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


@pytest.fixture
def cuda_capable(monkeypatch):
    """Give the fake SDK a cuvis.cuda that reports the same-process path as usable."""
    import cuvis

    cuda = types.ModuleType("cuvis.cuda")
    cuda.capabilities = Mock(return_value=types.SimpleNamespace(same_process=True))
    cuda.enable = Mock()
    monkeypatch.setattr(cuvis, "cuda", cuda, raising=False)
    return cuda


def _indices_read(session) -> list[int]:
    """Measurement indices the fake session was asked for, after the reader's open-time read.

    ``Cu3sCubeReader`` reads measurement 0 once when it opens; everything after it is a read
    the code under test asked for.
    """
    return [call.args[0] for call in session.get_measurement.call_args_list][1:]


def _gate_reads(session, gate: threading.Event, started: list[int], *, only=None) -> None:
    """Make the fake block every read (or only the indices in ``only``) until ``gate`` is set."""
    measurement = session.get_measurement.return_value

    def blocked(idx):
        if only is None or idx in only:
            started.append(idx)
            gate.wait(timeout=5)
        return measurement

    session.get_measurement.side_effect = blocked


def _wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _warnings_during(fn):
    """Loguru WARNING messages emitted while ``fn`` runs, plus its result."""
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        result = fn()
    finally:
        logger.remove(sink)
    return result, messages


def _plan(cu3s: str, depth: int, read_threads: int = 0):
    """A cache with one recording already open, and a plan of ``depth`` over it.

    Opening first matters: a test that then blocks ``get_measurement`` must not block the
    reader's own open-time read on the calling thread.
    """
    cache = Cu3sReaderCache(processing_mode=None, read_threads=read_threads, read_ahead=depth)
    cache.get(cu3s)
    return cache, ReadAheadPlan(cache, depth=depth)


# ----------------------------------------------------------------------------- the plan
def test_take_in_the_announced_order_returns_each_frame_once(mock_cuvis_sdk, cu3s, releases_gil):
    session = mock_cuvis_sdk["session"]
    cache, plan = _plan(cu3s, depth=2)
    try:
        keys = [(cu3s, 3), (cu3s, 1), (cu3s, 2), (cu3s, 0)]
        plan.announce(keys)
        assert [plan.take([key])[0]["mesu_index"] for key in keys] == [3, 1, 2, 0]
        assert sorted(_indices_read(session)) == [0, 1, 2, 3]
        assert plan.active
    finally:
        plan.close()
        cache.close()


def test_take_serves_a_whole_batch_from_the_announced_order(mock_cuvis_sdk, cu3s, releases_gil):
    cache, plan = _plan(cu3s, depth=4)
    try:
        keys = [(cu3s, i) for i in range(6)]
        plan.announce(keys)
        assert [item["mesu_index"] for item in plan.take(keys[:2])] == [0, 1]
        assert [item["mesu_index"] for item in plan.take(keys[2:4])] == [2, 3]
        assert [item["mesu_index"] for item in plan.take(keys[4:])] == [4, 5]
    finally:
        plan.close()
        cache.close()


def test_never_more_than_depth_frames_are_read_ahead(mock_cuvis_sdk, cu3s, releases_gil):
    session = mock_cuvis_sdk["session"]
    cache, plan = _plan(cu3s, depth=2)
    gate, started = threading.Event(), []
    _gate_reads(session, gate, started)
    try:
        keys = [(cu3s, i) for i in range(6)]
        plan.announce(keys)
        assert _wait_until(lambda: len(started) == 2)
        time.sleep(0.1)
        assert len(started) == 2, "announce must submit depth frames, not the whole epoch"
        gate.set()
        assert plan.take([keys[0]])[0]["mesu_index"] == 0
        assert _wait_until(lambda: len(started) == 3)
        time.sleep(0.1)
        assert len(started) == 3, "one consumed frame tops the queue up by exactly one"
    finally:
        gate.set()
        plan.close()
        cache.close()


def test_a_take_outside_the_announced_order_falls_back_and_stops_looking_ahead(
    mock_cuvis_sdk, cu3s, releases_gil
):
    cache, plan = _plan(cu3s, depth=2)
    try:
        plan.announce([(cu3s, 0), (cu3s, 1), (cu3s, 2)])
        items, messages = _warnings_during(lambda: plan.take([(cu3s, 2)]))
        assert items[0]["mesu_index"] == 2
        assert not plan.active
        assert len([m for m in messages if "read-ahead" in m]) == 1, messages
        assert not plan._pending, "the frames read ahead were drained, not left in flight"
        # Later takes stay synchronous and quiet.
        _, again = _warnings_during(lambda: plan.take([(cu3s, 0)]))
        assert not again
    finally:
        plan.close()
        cache.close()


def test_announcing_a_new_epoch_drains_the_previous_one(mock_cuvis_sdk, cu3s, releases_gil):
    session = mock_cuvis_sdk["session"]
    cache, plan = _plan(cu3s, depth=2)
    try:
        plan.announce([(cu3s, i) for i in range(4)])  # 0 and 1 go in flight
        plan.announce([(cu3s, 3), (cu3s, 2)])
        assert [plan.take([k])[0]["mesu_index"] for k in [(cu3s, 3), (cu3s, 2)]] == [3, 2]
        assert plan.active
        # The new epoch's frames were read exactly once; the abandoned ones at most once
        # (cancelled if they had not started, dropped if they had), and never twice.
        reads = _indices_read(session)
        assert reads.count(3) == 1 and reads.count(2) == 1
        assert all(reads.count(i) <= 1 for i in (0, 1)), reads
    finally:
        plan.close()
        cache.close()


def test_close_waits_for_in_flight_reads_and_submits_nothing_more(
    mock_cuvis_sdk, cu3s, releases_gil
):
    session = mock_cuvis_sdk["session"]
    cache, plan = _plan(cu3s, depth=2)
    gate, started = threading.Event(), []
    _gate_reads(session, gate, started)
    try:
        plan.announce([(cu3s, i) for i in range(6)])
        assert _wait_until(lambda: len(started) == 2)
        closer = threading.Thread(target=plan.close)
        closer.start()
        time.sleep(0.1)
        assert closer.is_alive(), "close must wait for the reads it started"
        gate.set()
        closer.join(timeout=5)
        assert not closer.is_alive()
        time.sleep(0.1)
        assert len(started) == 2, "nothing may be submitted after close"
        assert not plan.active
    finally:
        gate.set()
        cache.close()


def test_a_failed_read_surfaces_at_its_frame_with_its_note_and_drains_the_rest(
    mock_cuvis_sdk, cu3s, releases_gil, monkeypatch
):
    cache, plan = _plan(cu3s, depth=2)
    try:
        original = Cu3sPrefetchReader._read_with

        def _explode_on_two(self, session, mesu_index):
            if mesu_index == 2:
                raise RuntimeError("sdk said no")
            return original(self, session, mesu_index)

        monkeypatch.setattr(Cu3sPrefetchReader, "_read_with", _explode_on_two)
        keys = [(cu3s, i) for i in range(4)]
        plan.announce(keys)  # 0 and 1 in flight
        assert plan.take([keys[0]])[0]["mesu_index"] == 0  # tops up 2, which fails in its worker
        assert plan.take([keys[1]])[0]["mesu_index"] == 1  # tops up 3
        with pytest.raises(RuntimeError, match="sdk said no") as excinfo:
            plan.take([keys[2]])
        assert any("measurement 2" in note for note in excinfo.value.__notes__)
        assert not plan._pending and not plan.active, "frame 3 was drained, not leaked"

        monkeypatch.undo()
        assert plan.take([keys[3]])[0]["mesu_index"] == 3  # the synchronous path still serves
    finally:
        plan.close()
        cache.close()


def test_a_plan_over_a_pool_less_reader_reads_synchronously(mock_cuvis_sdk, cu3s, monkeypatch):
    """A binding that holds the GIL disables the pool; the plan must keep working without it."""
    monkeypatch.setattr(
        "cuvis_ai_dataloader.data.readers.cu3s_pool.cuvis_releases_gil", lambda _call: False
    )
    cache, plan = _plan(cu3s, depth=2)
    try:
        assert cache.get(cu3s)._pool is None
        keys = [(cu3s, 1), (cu3s, 0)]
        plan.announce(keys)
        assert [plan.take([k])[0]["mesu_index"] for k in keys] == [1, 0]
    finally:
        plan.close()
        cache.close()


# ------------------------------------------------------------------------- the samplers
def test_lookahead_sampler_yields_torch_batches_and_announces_their_order():
    data = list(range(10))
    torch.manual_seed(1234)
    expected = list(BatchSampler(RandomSampler(data), 3, drop_last=False))
    torch.manual_seed(1234)
    announced: list[list[int]] = []
    sampler = LookaheadBatchSampler(
        RandomSampler(data), 3, drop_last=False, on_epoch=announced.append
    )
    assert list(sampler) == expected
    assert announced == [[i for batch in expected for i in batch]]
    assert len(sampler) == len(expected) == 4
    assert isinstance(sampler, BatchSampler)  # Lightning re-instantiates torch batch samplers


def test_lookahead_sampler_announces_every_epoch_afresh():
    data = list(range(6))
    announced: list[list[int]] = []
    sampler = LookaheadBatchSampler(
        SequentialSampler(data), 2, drop_last=False, on_epoch=announced.append
    )
    list(sampler)
    list(sampler)
    assert announced == [list(range(6)), list(range(6))]


def test_announcing_wrapper_flattens_source_coherent_batches():
    sources = ["a", "b", "a", "b"]
    expected = list(SourceCoherentBatchSampler(sources, 2, shuffle=False))
    announced: list[list[int]] = []
    wrapped = AnnouncingBatchSampler(
        SourceCoherentBatchSampler(sources, 2, shuffle=False), on_epoch=announced.append
    )
    assert list(wrapped) == expected
    assert announced == [[i for batch in expected for i in batch]]
    assert len(wrapped) == len(expected)


# ------------------------------------------------------------------------ pool + cache
def test_submit_reads_on_a_leased_handle_and_returns_a_future(mock_cuvis_sdk, cu3s):
    reader = Cu3sPrefetchReader(cu3s, threads=2)
    try:
        before = reader._leases.qsize()
        future = reader.submit(3)
        assert future.result(timeout=5)["mesu_index"] == 3
        assert reader._leases.qsize() == before
    finally:
        reader.close()


def test_submit_on_a_disabled_pool_raises(mock_cuvis_sdk, cu3s):
    reader = Cu3sPrefetchReader(cu3s, threads=2)
    try:
        reader.disable_pool()
        with pytest.raises(RuntimeError, match="pool"):
            reader.submit(1)
    finally:
        reader.close()


def test_read_ahead_opens_a_pool_even_for_a_single_thread(mock_cuvis_sdk, cu3s, releases_gil):
    import cuvis

    cache = Cu3sReaderCache(processing_mode=None, read_threads=0, read_ahead=1)
    try:
        reader = cache.get(cu3s)
        assert isinstance(reader, Cu3sPrefetchReader)
        assert reader._pool is not None
        assert reader.threads == 1
        assert cuvis.SessionFile.call_count == 1, "depth 1 needs no extra handle"
    finally:
        cache.close()


def test_the_pool_is_as_wide_as_the_depth_or_the_thread_budget(
    mock_cuvis_sdk, tmp_path, releases_gil
):
    paths = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.cu3s"
        path.write_bytes(b"")
        paths.append(str(path))
    deep = Cu3sReaderCache(processing_mode=None, read_threads=0, read_ahead=3)
    wide = Cu3sReaderCache(processing_mode=None, read_threads=6, read_ahead=2)
    try:
        assert deep.get(paths[0]).threads == 3
        assert wide.get(paths[1]).threads == 6
    finally:
        deep.close()
        wide.close()


def test_a_reader_with_reads_in_flight_is_not_evicted(mock_cuvis_sdk, tmp_path, releases_gil):
    session = mock_cuvis_sdk["session"]
    paths = []
    for name in ("a", "b", "c"):
        path = tmp_path / f"{name}.cu3s"
        path.write_bytes(b"")
        paths.append(str(path))
    cache = Cu3sReaderCache(processing_mode=None, max_open_sessions=1, read_ahead=1, sources=3)
    gate, started = threading.Event(), []
    try:
        first = cache.get(paths[0])
        _gate_reads(session, gate, started, only={1})  # open-time reads of 0 pass through
        future = cache.submit(paths[0], 1)
        assert _wait_until(lambda: started == [1])

        _, messages = _warnings_during(lambda: cache.get(paths[1]))
        assert first.session is not None, "a reader with a read in flight was closed"
        assert list(cache._readers) == paths[:2]
        assert any("in flight" in m for m in messages), messages

        gate.set()
        assert future.result(timeout=5)["mesu_index"] == 1
        cache.get(paths[2])  # nothing pinned any more: the oldest goes
        assert first.session is None
        assert paths[0] not in cache._readers
    finally:
        gate.set()
        cache.close()


def test_cache_submit_without_a_pool_completes_synchronously(mock_cuvis_sdk, cu3s, monkeypatch):
    monkeypatch.setattr(
        "cuvis_ai_dataloader.data.readers.cu3s_pool.cuvis_releases_gil", lambda _call: False
    )
    cache = Cu3sReaderCache(processing_mode=None, read_ahead=2)
    try:
        future = cache.submit(cu3s, 2)
        assert future.done()
        assert future.result()["mesu_index"] == 2
    finally:
        cache.close()


# ------------------------------------------------------------------ device cubes, threads
def test_device_cubes_read_on_a_pool_thread_are_synchronized_before_hand_off(
    mock_cuvis_sdk, cu3s, cuda_capable, monkeypatch
):
    synced: list = []
    monkeypatch.setattr(cu3s_pool, "sync_device_cube", lambda cube: synced.append(cube))
    reader = Cu3sPrefetchReader(cu3s, threads=1, cuda_cubes=True)
    try:
        assert reader.cuda_cubes is True
        item = reader.submit(1).result(timeout=5)
        assert len(synced) == 1 and synced[0] is item["cube"]
        reader.read(1)  # the caller's own read crosses no thread boundary
        assert len(synced) == 1
    finally:
        reader.close()


def test_sync_device_cube_is_a_no_op_for_host_data():
    from cuvis_ai_dataloader.data.readers.cu3s_cuda import sync_device_cube

    sync_device_cube(torch.zeros(2, 2))  # must not raise without a device
    sync_device_cube(torch.zeros(2, 2).numpy())


# ------------------------------------------------------------- lifecycle edges (review)
class _ReadyCache:
    """A cache stand-in whose futures are complete on submission; ``missing`` fails to open."""

    def __init__(self) -> None:
        self.submitted: list[tuple[str, int]] = []

    def submit(self, source: str, index: int):
        self.submitted.append((source, index))
        if source == "missing":
            raise FileNotFoundError(source)
        from concurrent.futures import Future

        future: Future = Future()
        future.set_result({"mesu_index": index, "source": source})
        return future

    def read_many(self, keys):
        return [self.submit(*key).result() for key in keys]


def test_the_cache_trims_back_to_its_limit_once_the_pins_release(
    mock_cuvis_sdk, tmp_path, releases_gil
):
    session = mock_cuvis_sdk["session"]
    paths = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.cu3s"
        path.write_bytes(b"")
        paths.append(str(path))
    cache = Cu3sReaderCache(processing_mode=None, max_open_sessions=1, read_ahead=1, sources=2)
    gate, started = threading.Event(), []
    try:
        first = cache.get(paths[0])
        _gate_reads(session, gate, started, only={1})
        future = cache.submit(paths[0], 1)
        assert _wait_until(lambda: started == [1])
        cache.get(paths[1])  # exceeds the limit while a is pinned
        assert len(cache._readers) == 2
        gate.set()
        future.result(timeout=5)
        assert _wait_until(lambda: not cache._pinned)
        cache.get(paths[1])  # a hit; the limit is restored on the next access
        assert list(cache._readers) == [paths[1]]
        assert first.session is None
    finally:
        gate.set()
        cache.close()


def test_a_source_that_fails_to_open_surfaces_at_its_frame_not_at_announce():
    cache = _ReadyCache()
    plan = ReadAheadPlan(cache, depth=2)
    plan.announce([("ok", 0), ("missing", 0), ("ok", 1)])  # must not raise here
    assert plan.take([("ok", 0)])[0]["mesu_index"] == 0
    with pytest.raises(FileNotFoundError, match="missing"):
        plan.take([("missing", 0)])
    assert not plan._pending and not plan.active
    assert plan.take([("ok", 1)])[0]["mesu_index"] == 1  # synchronous fallback still serves


def test_dropping_the_loader_iterator_releases_the_frames_read_ahead():
    import gc

    from torch.utils.data import DataLoader

    cache = _ReadyCache()
    plan = ReadAheadPlan(cache, depth=2)

    class _Frames:
        def __len__(self):
            return 10

        def __getitems__(self, indices):
            return plan.take([("ok", i) for i in indices])

    frames = _Frames()
    sampler = LookaheadBatchSampler(
        SequentialSampler(frames),
        1,
        on_epoch=lambda order: plan.announce([("ok", i) for i in order]),
        on_epoch_end=plan.release,
    )
    loader = DataLoader(frames, batch_sampler=sampler, collate_fn=lambda items: items)
    it = iter(loader)
    next(it)
    next(it)
    assert plan.active and plan._pending
    del it
    gc.collect()
    assert not plan._pending, "frames read ahead of a dropped iterator must be released"
    assert not plan.active
    # A full epoch afterwards still works and reads every frame once.
    cache.submitted.clear()
    assert [item[0]["mesu_index"] for item in loader] == list(range(10))
    assert sorted(cache.submitted) == [("ok", i) for i in range(10)]
