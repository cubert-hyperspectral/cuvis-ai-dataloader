"""Threaded cu3s reading: a pooled reader, a bounded reader cache, and their factory.

Kept out of ``cu3s_reader.py`` so that module stays readable as the single-threaded
contract and no consumer imports ``threading`` merely to read a cube.

The topology here is the one measured free of wrong cubes: N ``SessionFile`` handles on one
file with a single ``ProcessingContext`` shared between them. A context carries its
originating session's calibration and references, so it is per file and never shared
across files.
"""

from __future__ import annotations

import itertools
import queue
import random
from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from loguru import logger
from torch.utils.data import Sampler

from .._extras import cuvis_releases_gil, require_cuvis
from .cu3s_reader import Cu3sCubeReader

# Past 12 threads SpectralRadiance fails intermittently even with cuda_host_memory_maximum_gb
# raised above its 12.0 default, which this package cannot raise.
MAX_READ_THREADS = 16


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
            self.threads, thread_name_prefix="cu3s-read"
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
            return self._read_with(session, mesu_index)
        except BaseException as exc:
            # The SDK error channel is process-wide, so a threaded failure can surface with
            # another thread's message; record which measurement this call was really on.
            exc.add_note(f"while reading measurement {mesu_index} of {self.cu3s_file_path}")
            raise
        finally:
            self._leases.put(session)

    @property
    def wavelengths_nm(self) -> np.ndarray:
        """Per-channel wavelengths, on a leased handle so it cannot race a batch read."""
        return self._leased_read(0)["wavelengths"]

    def iter_reads(self, indices: Iterable[int]) -> Iterator[dict]:
        """Yield one read per index, in order, with at most ``queue_depth`` in flight.

        Bounded because a cube is 63 to 251 MB, so submitting a whole session at once would
        hold every frame of it in RAM.
        """
        if self._pool is None:
            yield from super().iter_reads(indices)
            return
        remaining = iter(indices)
        pending = deque(
            self._pool.submit(self._leased_read, index)
            for index in itertools.islice(remaining, self._depth)
        )
        for index in remaining:
            yield pending.popleft().result()
            pending.append(self._pool.submit(self._leased_read, index))
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
    cu3s_file_path: str, *, read_threads: int = 0, **reader_kwargs: Any
) -> Cu3sCubeReader:
    """Open a pooled reader when threads are asked for and the binding supports them.

    Falls back with a warning rather than raising, because one config has to run both on a
    dev box with a GIL-releasing binding and in CI on a stock one, where extra threads are a
    measured loss rather than a gain.
    """
    if read_threads > MAX_READ_THREADS:
        raise ValueError(f"read_threads must be <= {MAX_READ_THREADS}, got {read_threads}")
    if read_threads < 2:
        return Cu3sCubeReader(cu3s_file_path, **reader_kwargs)
    reader = Cu3sPrefetchReader(cu3s_file_path, threads=read_threads, **reader_kwargs)
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
    ) -> None:
        if max_open_sessions < 1:
            raise ValueError(f"max_open_sessions must be >= 1, got {max_open_sessions}")
        if read_threads < 0:
            raise ValueError(f"read_threads must be >= 0, got {read_threads}")
        self._processing_mode = processing_mode
        self._max_open = min(int(max_open_sessions), max(1, int(sources)))
        self._per_file_threads = int(read_threads) // self._max_open
        self._outer_size = min(self._max_open, int(read_threads)) if read_threads else 0
        self._readers: OrderedDict[str, Cu3sCubeReader] = OrderedDict()
        self._outer: ThreadPoolExecutor | None = None

    def __getstate__(self) -> dict:
        # Native handles and thread pools do not pickle; a DataLoader worker reopens lazily.
        return {**self.__dict__, "_readers": OrderedDict(), "_outer": None}

    def get(self, source: str) -> Cu3sCubeReader:
        """The reader for ``source``, opening it and evicting the oldest when full."""
        reader = self._readers.get(source)
        if reader is not None:
            self._readers.move_to_end(source)
            return reader
        while len(self._readers) >= self._max_open:
            _, evicted = self._readers.popitem(last=False)
            evicted.close()
        reader = open_reader(
            source,
            read_threads=self._per_file_threads,
            processing_mode=self._processing_mode,
        )
        self._readers[source] = reader
        return reader

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
