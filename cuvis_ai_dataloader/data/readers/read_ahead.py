"""Read-ahead over an announced index order, so a batch-1 loader overlaps reads with compute.

torch hands a map-style dataset one batch of indices at a time and nothing earlier, and
Lightning's fetcher never prefetches for a sized loader, so at ``batch_size: 1`` the SDK read
of frame i+1 waits for the model step on frame i. Two pieces close that gap without touching
the DataLoader contract:

- a batch sampler that materializes the epoch's order when torch asks for it and announces
  it to the dataset (``LookaheadBatchSampler`` around torch's own samplers,
  ``AnnouncingBatchSampler`` around a batch sampler such as ``SourceCoherentBatchSampler``);
- a ``ReadAheadPlan`` the dataset consults from ``__getitems__``: it keeps at most ``depth``
  frames in flight on the reader pool, in the announced order, and hands each one over when
  the loader asks for it.

::

    epoch start                            per batch
    sampler.__iter__()                     dataset.__getitems__([i])
      batches = list(order)  ---------->     plan.take([key_i])
      on_epoch(flattened order)                pop the head future, .result()  <- pool thread
                                               top up to depth                    reads i+1..i+depth
    plan.announce(keys):
      drain the previous epoch, submit depth

A consumer that leaves the announced order (a statistical pass that stops early, a caller of
``__getitem__``) is served synchronously and the plan stops looking ahead for that epoch, so
the read-ahead can never hand out the wrong frame. The plan is driven by the loader's thread
only; the pool threads never touch it. Replacing the sampler means Lightning cannot inject a
``DistributedSampler``: not for DDP.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future
from typing import Any

from loguru import logger
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, Sampler, SequentialSampler

from .cu3s_pool import Cu3sReaderCache, SourceCoherentBatchSampler

Key = tuple[str, int]
"""``(source path, measurement index)``: what the reader cache reads by."""


class ReadAheadPlan:
    """At most ``depth`` frames of the announced order in flight, handed over in order."""

    def __init__(self, cache: Cu3sReaderCache, depth: int) -> None:
        if int(depth) < 1:
            raise ValueError(f"read-ahead depth must be >= 1, got {depth}")
        self._cache = cache
        self._depth = int(depth)
        self._order: list[Key] = []
        self._next = 0  # position of the key the consumer is expected to ask for next
        self._cursor = 0  # position of the next key to submit
        self._pending: deque[tuple[Key, Future]] = deque()
        self._active = False
        self._closed = False

    @property
    def active(self) -> bool:
        """Whether an order is announced and the plan is still reading ahead of it."""
        return self._active

    def announce(self, keys: Sequence[Key]) -> None:
        """Start an epoch: drop what the previous one left in flight, submit the first frames."""
        self.release()
        if self._closed:
            return
        self._order = [(source, int(index)) for source, index in keys]
        self._next = 0
        self._cursor = 0
        self._active = bool(self._order)
        self._top_up()

    def take(self, keys: Sequence[Key]) -> list[dict]:
        """The items for ``keys``: from the frames read ahead when they are next, else read now."""
        wanted: list[Key] = [(source, int(index)) for source, index in keys]
        if self._active and self._order[self._next : self._next + len(wanted)] == wanted:
            return self._take_head(len(wanted))
        if self._active:
            logger.warning(
                "read-ahead: the loader asked for {} but the announced order continues with {}; "
                "serving synchronously and reading ahead no further this epoch",
                wanted,
                self._order[self._next : self._next + len(wanted)],
            )
            self.release()
        return self._cache.read_many(wanted)

    def release(self) -> None:
        """Forget the announced order and drop every frame in flight (waiting for the reads).

        Called when an epoch ends or its loader iterator is dropped (Lightning's sanity check
        takes two batches and walks away; early stopping ends an epoch early), so the cubes
        read ahead do not sit in memory until the next epoch is announced.
        """
        self._active = False
        while self._pending:
            _, future = self._pending.popleft()
            if future.cancel():
                continue
            try:
                item = future.result()
            except BaseException as exc:  # the consumer never asked for this frame
                logger.debug("read-ahead: a frame read ahead failed and was dropped: {!r}", exc)
                continue
            del item  # explicit, so a device cube's buffer goes back promptly
        self._order = []
        self._next = 0
        self._cursor = 0

    def close(self) -> None:
        """Stop reading ahead for good and wait for the frames already in flight."""
        self._closed = True
        self.release()

    def _take_head(self, count: int) -> list[dict]:
        items: list[dict] = []
        try:
            for _ in range(count):
                _, future = self._pending.popleft()
                self._next += 1
                # Refill before blocking on this frame, so the pool stays busy while we wait.
                self._top_up()
                items.append(future.result())
        except BaseException:
            # Dropped here, inside the handler, and not by the frame's teardown while the
            # exception propagates: a device cube's DLPack deleter runs through ctypes, which
            # cannot run while an error is pending on this thread.
            items.clear()
            self.release()
            raise
        return items

    def _top_up(self) -> None:
        while len(self._pending) < self._depth and self._cursor < len(self._order):
            key = self._order[self._cursor]
            self._cursor += 1
            try:
                future = self._cache.submit(*key)
            except Exception as exc:
                # A recording that fails to open must fail at its own frame, where the loader
                # attributes it, not inside the sampler that announced the epoch.
                future = Future()
                future.set_exception(exc)
            self._pending.append((key, future))


class LookaheadBatchSampler(BatchSampler):
    """torch's ``BatchSampler`` that announces the epoch's order before yielding it.

    ``BatchSampler.__iter__`` draws ``iter(self.sampler)`` exactly once, at the same moment a
    plain ``DataLoader(shuffle=True)`` would, so a ``RandomSampler`` inside it produces the
    permutation torch would have produced for the same seed. Staying a ``BatchSampler`` with
    public ``batch_size`` and ``drop_last`` keeps Lightning's loader re-instantiation working;
    ``on_epoch`` is optional so a re-instantiation without it still yields (and the dataset,
    hearing no order, simply reads synchronously).
    """

    def __init__(
        self,
        sampler: Sampler[int],
        batch_size: int,
        drop_last: bool = False,
        *,
        on_epoch: Callable[[list[int]], None] | None = None,
        on_epoch_end: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self._on_epoch = on_epoch
        self._on_epoch_end = on_epoch_end

    def __iter__(self) -> Iterator[list[int]]:
        # A generator, like torch's own: the permutation is drawn at the first next(), after
        # the DataLoader iterator has taken its base seed from the global RNG, so the order
        # equals a plain shuffle=True loader's for the same seed.
        batches = [list(batch) for batch in super().__iter__()]
        yield from _announced(batches, self._on_epoch, self._on_epoch_end)


class AnnouncingBatchSampler(Sampler[list[int]]):
    """Announce the order of a batch sampler that already owns its batches, then yield it."""

    def __init__(
        self,
        batch_sampler: Sampler[list[int]],
        *,
        on_epoch: Callable[[list[int]], None],
        on_epoch_end: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._inner = batch_sampler
        self._on_epoch = on_epoch
        self._on_epoch_end = on_epoch_end

    def __iter__(self) -> Iterator[list[int]]:
        batches = [list(batch) for batch in self._inner]
        yield from _announced(batches, self._on_epoch, self._on_epoch_end)

    def __len__(self) -> int:
        return len(self._inner)  # type: ignore[arg-type]


def _announced(
    batches: list[list[int]],
    on_epoch: Callable[[list[int]], None] | None,
    on_epoch_end: Callable[[], None] | None,
) -> Iterator[list[int]]:
    """Yield the batches between the two hooks.

    ``on_epoch_end`` runs when the epoch is exhausted and, through ``GeneratorExit``, when the
    loader's iterator is dropped before that: the DataLoader holds this generator, so the
    frames read ahead of an abandoned iterator are released the moment torch lets go of it.
    """
    if on_epoch is not None:
        on_epoch([index for batch in batches for index in batch])
    try:
        yield from batches
    finally:
        if on_epoch_end is not None:
            on_epoch_end()


def build_loader(
    dataset: Any,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    name: str,
    source_coherent_batches: bool,
    read_ahead: int,
    plain: Callable[[], DataLoader],
) -> DataLoader:
    """The loader both cu3s DataModules build: plain, source-coherent, read-ahead, or both.

    Owning the batch sampler is what the last two have in common (and why neither works
    under DDP). ``plain`` builds the base module's loader when nothing here applies.
    ``samples_per_frame`` wraps the dataset in core's repeat, which fetches one index at a
    time and has no ``__getitems__``, so the read-ahead cannot reach the base there and the
    loader says so instead of silently reading synchronously.
    """
    base = getattr(dataset, "_base", dataset)
    sources = getattr(base, "sample_sources", None)
    announce = getattr(base, "announce_order", None) if read_ahead else None
    release = getattr(base, "release_read_ahead", None) if announce is not None else None
    if announce is not None and base is not dataset:
        logger.warning(
            "read_ahead={} is off for the {} loader: samples_per_frame wraps the dataset in a "
            "repeat that fetches one index at a time, so the batched fetch never reaches the "
            "recordings.",
            read_ahead,
            name,
        )
        announce = None
    if source_coherent_batches and sources:
        # samples_per_frame wraps the dataset in a repeat whose index i reads base i % len,
        # so repeating the base's source list reproduces that mapping exactly.
        sources = list(sources) * max(1, len(dataset) // len(sources))
        batch_sampler: Sampler = SourceCoherentBatchSampler(sources, batch_size, shuffle=shuffle)
        if announce is not None:
            batch_sampler = AnnouncingBatchSampler(
                batch_sampler, on_epoch=announce, on_epoch_end=release
            )
        return DataLoader(dataset, num_workers=num_workers, batch_sampler=batch_sampler)
    if announce is None:
        return plain()
    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    return DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=LookaheadBatchSampler(
            sampler, batch_size, drop_last=False, on_epoch=announce, on_epoch_end=release
        ),
    )
