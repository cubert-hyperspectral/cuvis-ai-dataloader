"""Threaded cu3s reading: a pooled reader, a bounded reader cache, and their factory.

Kept out of ``cu3s_reader.py`` so that module stays readable as the single-threaded
contract and no consumer imports ``threading`` merely to read a cube.

The topology here is the one measured free of wrong cubes: N ``SessionFile`` handles on one
file with a single ``ProcessingContext`` shared between them. A context carries its
originating session's calibration and references, so it is per file and never shared
across files.

::

    Cu3sPrefetchReader(path, threads=N)
    +------------------------------------------------------------------+
    |  SessionFile #0 (self.session) --+                               |
    |  SessionFile #1                  +-- all share ONE               |
    |  ...                             |   ProcessingContext           |
    |  SessionFile #N-1 ---------------+   (session._pc = self.pc)     |
    |             | put / get                                          |
    |       lease queue (SimpleQueue) <---- ThreadPoolExecutor(N)      |
    |             |                             ^  submit(_leased_read)|
    |  read(i) ---+-- _leased_read(i) --> _read_with(session, i)       |
    |                                                                  |
    |  iter_reads(indices): deque of <= N+2 futures, popleft().result()|
    |                       -> order preserved, <= N+2 cubes in RAM    |
    |  submit(i) -> Future: one read, for a queue that outlives a call |
    +------------------------------------------------------------------+
    Cu3sReaderCache: LRU(max_open_sessions) of readers, close-on-evict.
      budget per file = read_threads                     if coherent batches
                      = read_threads // min(max_open, sources)  otherwise (warns if < 2)
      pool width per file = max(budget, read_ahead); read_ahead > 0 pools even one thread
      cross-file overlap on an outer pool of min(max_open, read_threads);
      a failed group waits for its siblings before raising, so no eviction
      closes a reader another group is still reading from;
      submit(source, i) pins its reader until the future resolves, so a
      read-ahead's frames in flight never lose their session to an eviction.
"""

from __future__ import annotations

import functools
import itertools
import queue
import random
import threading
from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any

from loguru import logger
from torch.utils.data import Sampler

from .._extras import configure_cuvis_sdk, cuvis_releases_gil, require_cuvis
from .cu3s_cuda import sync_device_cube
from .cu3s_reader import Cu3sCubeReader

# Past 12 threads SpectralRadiance fails intermittently even with cuda_host_memory_maximum_gb
# raised above its 12.0 default, which this package cannot raise.
MAX_READ_THREADS = 16
# The pool's threads are named after this; a read knows it runs on one by its thread name.
_POOL_THREAD_PREFIX = "cu3s-read"


def _chunked(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    """Split into consecutive runs of at most ``size`` (no itertools.batched on 3.11)."""
    return (items[start : start + size] for start in range(0, len(items), size))


class Cu3sPrefetchReader(Cu3sCubeReader):
    """A ``Cu3sCubeReader`` serving indices from a pool of extra session handles.

    Substitutes for a plain reader wherever one is cached: ``read`` keeps its single-index
    semantics and ``read_many`` / ``iter_reads`` are the parallel entry points. Both preserve
    the requested order, because callers pair each cube with per-index metadata positionally.
    """

    def __init__(
        self,
        cu3s_file_path: str,
        *,
        threads: int = 6,
        queue_depth: int | None = None,
        **reader_kwargs: Any,
    ) -> None:
        super().__init__(cu3s_file_path, **reader_kwargs)
        self.threads = max(1, int(threads))
        self._depth = int(queue_depth or self.threads + 2)
        # Handles are opened only here, once the base has installed the processing mode and
        # any custom references. Every ProcessingContext mutator is a read-modify-write on
        # state every thread observes, and a mid-flight change yields wrong cubes with no
        # error, so configuration has to be finished before a thread exists.
        self._sessions = self._open_handles(require_cuvis())
        self._leases: queue.SimpleQueue = queue.SimpleQueue()
        for session in self._sessions:
            self._leases.put(session)
        self._pool: ThreadPoolExecutor | None = ThreadPoolExecutor(
            self.threads, thread_name_prefix=_POOL_THREAD_PREFIX
        )

    def _open_handles(self, cuvis) -> list:
        """This reader's own handle plus ``threads - 1`` more, all on the shared context."""
        sessions = [self.session] + [
            cuvis.SessionFile(self.cu3s_file_path) for _ in range(self.threads - 1)
        ]
        # Without this the lazy Measurement.cube path builds a private context per handle,
        # silently undoing the one-context topology and duplicating its GPU buffers.
        for session in sessions:
            session._pc = self.pc
        return sessions

    def _leased_read(self, mesu_index: int) -> dict:
        """Read one index on a borrowed handle, returned even if the read raises."""
        session = self._leases.get()
        try:
            item = self._read_with(session, mesu_index)
            if self.cuda_cubes and threading.current_thread().name.startswith(_POOL_THREAD_PREFIX):
                # A device cube read here is consumed on the caller's thread, and nothing
                # else orders the SDK's writes against that thread's stream.
                sync_device_cube(item["cube"])
            return item
        except BaseException as exc:
            # The SDK error channel is process-wide, so a threaded failure can surface with
            # another thread's message; record which measurement this call was really on.
            exc.add_note(f"while reading measurement {mesu_index} of {self.cu3s_file_path}")
            raise
        finally:
            self._leases.put(session)

    def read(self, mesu_index: int) -> dict:
        """Read one measurement on a leased handle.

        ``self.session`` is one of the pooled handles, so a plain read on it could share the
        handle with a read the pool has in flight; going through the lease queue rules that
        out whatever thread calls this.
        """
        return self._leased_read(mesu_index)

    def submit(self, mesu_index: int) -> Future:
        """Schedule one read on the pool; the future resolves to what ``read`` returns.

        The entry point a read-ahead builds on: it keeps its own bounded queue of these
        across the loader's calls, which ``iter_reads`` cannot, since its queue lives inside
        one call.
        """
        if self._pool is None:
            raise RuntimeError("the reader pool is disabled; read synchronously instead")
        return self._pool.submit(self._leased_read, mesu_index)

    def iter_reads(self, indices: Iterable[int]) -> Iterator[dict]:
        """Yield one read per index, in order, with at most ``queue_depth`` in flight.

        Bounded because a cube is 63 to 251 MB, so submitting a whole session at once would
        hold every frame of it in RAM.
        """
        if self._pool is None:
            yield from super().iter_reads(indices)
            return
        remaining = iter(indices)
        pending = deque(self.submit(index) for index in itertools.islice(remaining, self._depth))
        for index in remaining:
            yield pending.popleft().result()
            pending.append(self.submit(index))
        while pending:
            yield pending.popleft().result()

    def disable_pool(self) -> None:
        """Drop to single-handle reads, leaving the reader usable."""
        self._shutdown_pool()
        self._drain_leases()
        self._sessions = [self.session]
        self._leases.put(self.session)

    def _shutdown_pool(self) -> None:
        """Stop the pool and wait, so no handle is still leased when handles are dropped."""
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)

    def _drain_leases(self) -> None:
        """Empty the lease queue, which would otherwise keep handles alive."""
        while True:
            try:
                self._leases.get_nowait()
            except queue.Empty:
                return

    def close(self) -> None:
        """Stop the pool, release the extra handles, then close as a plain reader."""
        self._shutdown_pool()
        self._drain_leases()
        self._sessions = []
        super().close()


def open_reader(
    cu3s_file_path: str,
    *,
    read_threads: int = 0,
    sdk_cuda: bool = True,
    force_pool: bool = False,
    **reader_kwargs: Any,
) -> Cu3sCubeReader:
    """Open a pooled reader when threads are asked for and the binding supports them.

    Falls back with a warning rather than raising, because one config has to run both on a
    dev box with a GIL-releasing binding and in CI on a stock one, where extra threads are a
    measured loss rather than a gain.

    ``force_pool`` opens a pool even for a single thread: a read-ahead needs somewhere to run
    its one frame in flight, and one pool thread on the reader's own handle is enough for it.

    The SDK device is chosen here rather than by the caller, because this runs in the
    DataLoader worker too, and a worker is a fresh process that has initialized nothing.
    """
    configure_cuvis_sdk(cuda=sdk_cuda)
    if read_threads > MAX_READ_THREADS:
        raise ValueError(f"read_threads must be <= {MAX_READ_THREADS}, got {read_threads}")
    if read_threads < 2 and not force_pool:
        return Cu3sCubeReader(cu3s_file_path, **reader_kwargs)
    reader = Cu3sPrefetchReader(cu3s_file_path, threads=max(1, read_threads), **reader_kwargs)
    if not cuvis_releases_gil(lambda: reader.read(0)):
        logger.warning(
            "cuvis binding holds the GIL during SDK calls, so read_threads={} cannot help; "
            "reading {} single-threaded. This needs a binding built with the GIL release.",
            read_threads,
            cu3s_file_path,
        )
        reader.disable_pool()
    return reader


class Cu3sReaderCache:
    """Bounded LRU of open cu3s readers, close-on-evict, shared by both cu3s datasets.

    Every open session holds native SDK resources including GPU processing pools, and past a
    handful of concurrently open Reflectance sessions the SDK's CUDA allocator fails hard
    (an illegal memory access that kills the process). ``read_threads`` is therefore a budget
    for the cache as a whole rather than a per-file count: it is divided across the sessions
    the cache may hold open, so the total handle count stays flat however many sources an
    epoch touches. One source spends the whole budget inside that file; several spend it
    across them, which is the right split, since multi-file cost is dominated by the
    per-file context build rather than by reads.
    """

    def __init__(
        self,
        *,
        processing_mode: str | None,
        max_open_sessions: int = 4,
        read_threads: int = 0,
        sources: int = 1,
        coherent: bool = False,
        sdk_cuda: bool = True,
        cuda_cubes: bool = False,
        read_ahead: int = 0,
    ) -> None:
        if max_open_sessions < 1:
            raise ValueError(f"max_open_sessions must be >= 1, got {max_open_sessions}")
        if read_threads < 0:
            raise ValueError(f"read_threads must be >= 0, got {read_threads}")
        if read_ahead < 0:
            raise ValueError(f"read_ahead must be >= 0, got {read_ahead}")
        self._processing_mode = processing_mode
        self._sdk_cuda = bool(sdk_cuda)
        self._cuda_cubes = bool(cuda_cubes)
        # Frames a read-ahead keeps in flight per recording. The pool that carries them is as
        # wide as the depth, so a depth needs no read_threads of its own.
        self._read_ahead = int(read_ahead)
        # Readers with reads in flight (read-ahead futures) and the lock the bookkeeping runs
        # under: futures resolve on pool threads, get() runs on the loader's.
        self._pinned: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        self._overflow_warned = False
        self._max_open = min(int(max_open_sessions), max(1, int(sources)))
        # With source-coherent batches a batch reads one recording, so cross-file overlap
        # cannot carry the budget and the whole of it belongs inside each file (at a cost of
        # up to max_open_sessions x read_threads open handles). Otherwise it is divided across
        # the sessions the cache may hold open, so the handle count stays flat.
        split = 1 if coherent else self._max_open
        self._per_file_threads = int(read_threads) // split
        self._outer_size = min(self._max_open, int(read_threads)) if read_threads else 0
        if read_threads and not coherent and self._per_file_threads < 2 and self._max_open > 1:
            logger.warning(
                "read_threads={} divided across up to {} open sessions leaves {} thread(s) per "
                "recording, so reads inside a recording stay single-threaded and only batches "
                "spanning several recordings overlap. Use read_threads >= {} or set "
                "source_coherent_batches=True to spend the whole budget inside each recording.",
                read_threads,
                self._max_open,
                self._per_file_threads,
                2 * self._max_open,
            )
        self._readers: OrderedDict[str, Cu3sCubeReader] = OrderedDict()
        self._outer: ThreadPoolExecutor | None = None
        self._evictions = 0

    def __getstate__(self) -> dict:
        # Native handles, thread pools and locks do not pickle; a DataLoader worker reopens
        # lazily and starts with nothing in flight.
        return {
            **self.__dict__,
            "_readers": OrderedDict(),
            "_outer": None,
            "_lock": None,
            "_pinned": defaultdict(int),
        }

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def get(self, source: str) -> Cu3sCubeReader:
        """The reader for ``source``, opening it and evicting the oldest idle one when full."""
        with self._lock:
            reader = self._readers.get(source)
            if reader is not None:
                self._readers.move_to_end(source)
                # A hit still restores the limit: a reader kept open past it while its reads
                # were in flight goes as soon as the cache is next asked for anything.
                self._evict_for(source, opening=False)
                return reader
            self._evict_for(source, opening=True)
        # Opened outside the lock: a context build takes seconds, and a pool thread finishing
        # a read must not wait on it to give its reader back.
        reader = open_reader(
            source,
            read_threads=max(self._per_file_threads, self._read_ahead),
            force_pool=self._read_ahead > 0,
            processing_mode=self._processing_mode,
            sdk_cuda=self._sdk_cuda,
            cuda_cubes=self._cuda_cubes,
        )
        with self._lock:
            self._readers[source] = reader
        return reader

    def _evict_for(self, incoming: str, *, opening: bool) -> None:
        """Close the oldest idle readers until ``incoming`` fits within the limit. Lock held.

        ``opening`` leaves room for a reader about to be added; otherwise ``incoming`` is
        already open and only the others are candidates.
        """
        limit = self._max_open - 1 if opening else self._max_open
        while len(self._readers) > limit:
            victim = next(
                (s for s in self._readers if s != incoming and not self._pinned.get(s)), None
            )
            if victim is None:
                if opening and not self._overflow_warned:
                    logger.warning(
                        "cu3s reader cache: every open session has a read in flight, so {} "
                        "opens as the {}th of {} allowed; the oldest closes once its reads land.",
                        Path(incoming).name,
                        len(self._readers) + 1,
                        self._max_open,
                    )
                    self._overflow_warned = True
                return
            evicted = self._readers.pop(victim)
            evicted.close()
            self._note_eviction(victim, incoming)

    def submit(self, source: str, index: int) -> Future:
        """Schedule one read and keep its reader open until the future resolves.

        A reader without a pool (a binding that holds the GIL, or ``read_ahead`` 0) reads on
        the calling thread and hands back a future that is already done, so a read-ahead can
        run on top of either without caring which it got. A recording that fails to open
        fails the same way: through the future, at the frame that asked for it, never out of
        the caller that merely scheduled it.
        """
        future: Future
        try:
            reader = self.get(source)
        except Exception as exc:
            future = Future()
            future.set_exception(exc)
            return future
        with self._lock:
            self._pinned[source] += 1
        try:
            future = reader.submit(index)  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError):
            future = Future()
            try:
                future.set_result(reader.read(index))
            except BaseException as exc:
                future.set_exception(exc)
        future.add_done_callback(functools.partial(self._unpin, source))
        return future

    def _unpin(self, source: str, _future: Future | None = None) -> None:
        """Release one pin on ``source`` (a future's done-callback, hence the extra argument)."""
        with self._lock:
            left = self._pinned[source] - 1
            if left > 0:
                self._pinned[source] = left
            else:
                self._pinned.pop(source, None)

    def _note_eviction(self, evicted: str, incoming: str) -> None:
        """Say once that the cache is cycling recordings.

        Every eviction rebuilds a ProcessingContext, about 13 s on the GPU, and nothing else
        reports it: an epoch over more recordings than the cache holds open just runs slowly.
        """
        if self._evictions == 0:
            logger.warning(
                "cu3s reader cache is full ({} open sessions): closing {} to open {}. Every such "
                "eviction rebuilds a ProcessingContext (about 13 s). Raise max_open_sessions, or "
                "set source_coherent_batches=True so a batch stays inside one recording.",
                self._max_open,
                Path(evicted).name,
                Path(incoming).name,
            )
        self._evictions += 1

    def read_many(self, positions: Sequence[tuple[str, int]]) -> list[dict]:
        """Read ``(source, index)`` pairs, keeping the caller's order.

        Grouping by source first means a batch touches each file once, so a batch spanning
        files overlaps their reads instead of serialising them. Groups are resolved in runs
        of at most ``max_open_sessions``, because resolving more would evict a reader another
        group in the same batch is still reading from.
        """
        groups: dict[str, list[int]] = defaultdict(list)
        for slot, (source, _) in enumerate(positions):
            groups[source].append(slot)

        results: list[dict] = [None] * len(positions)  # type: ignore[list-item]
        for chunk in _chunked(list(groups.items()), self._max_open):
            jobs = [(slots, self.get(source)) for source, slots in chunk]
            for (slots, _), items in zip(jobs, self._read_groups(jobs, positions)):
                for slot, item in zip(slots, items):
                    results[slot] = item
        return results

    def _read_groups(
        self,
        jobs: Sequence[tuple[list[int], Cu3sCubeReader]],
        positions: Sequence[tuple[str, int]],
    ) -> list[list[dict]]:
        """Read each file's group, overlapping files when the cache has a thread budget."""
        work = [(reader, [positions[slot][1] for slot in slots]) for slots, reader in jobs]
        if self._outer_size < 2 or len(work) < 2:
            return [reader.read_many(indices) for reader, indices in work]
        if self._outer is None:
            self._outer = ThreadPoolExecutor(self._outer_size, thread_name_prefix="cu3s-file")
        futures = [self._outer.submit(reader.read_many, indices) for reader, indices in work]
        # Every group finishes before any result is inspected: raising on the first failure
        # while a sibling still reads would let the caller's next get() evict and close the
        # reader that sibling holds.
        wait(futures)
        return [future.result() for future in futures]

    def close(self) -> None:
        """Release every open reader and stop the cross-file pool (safe to call twice)."""
        outer, self._outer = self._outer, None
        if outer is not None:
            outer.shutdown(wait=True)
        while self._readers:
            _, reader = self._readers.popitem(last=False)
            reader.close()


class SourceCoherentBatchSampler(Sampler):
    """Batches drawn from as few recordings as possible, so the reader cache stops thrashing.

    A shuffled multi-file epoch otherwise touches more recordings per batch than the cache can
    hold open, and every eviction costs a full ProcessingContext rebuild. Grouping by source
    leaves the epoch's contents and length untouched and changes only which samples share a
    batch, but that does change the gradient noise structure of training, which is why it is
    opt-in. It also replaces the loader's sampler, so Lightning cannot inject a
    ``DistributedSampler``: do not use it under DDP.
    """

    def __init__(
        self,
        sources: Sequence[str],
        batch_size: int,
        *,
        shuffle: bool,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self._groups: dict[str, list[int]] = defaultdict(list)
        for index, source in enumerate(sources):
            self._groups[source].append(index)
        self._total = len(sources)
        self._batch_size = max(1, int(batch_size))
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._seed = seed
        self._epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        """Yield index batches, consecutive within a source, reshuffled every epoch."""
        rng = random.Random(self._seed + self._epoch)
        self._epoch += 1
        groups = list(self._groups.values())
        if self._shuffle:
            groups = [rng.sample(group, len(group)) for group in groups]
            rng.shuffle(groups)
        order = list(itertools.chain.from_iterable(groups))
        batches = [
            order[start : start + self._batch_size]
            for start in range(0, len(order), self._batch_size)
        ]
        if self._drop_last and batches and len(batches[-1]) < self._batch_size:
            batches.pop()
        return iter(batches)

    def __len__(self) -> int:
        """Number of batches per epoch."""
        if self._drop_last:
            return self._total // self._batch_size
        return -(-self._total // self._batch_size)
