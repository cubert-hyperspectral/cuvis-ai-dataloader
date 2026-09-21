"""Lazy heavy-dependency imports, string param parsers, and the DataConfig-shape decorator.

Module-top imports across the plugin are limited to stdlib + numpy + torch +
pytorch_lightning + the base class. Heavy deps (cuvis, tifffile, pycocotools,
scikit-image) are imported only inside the methods that use them, via the
``require_*`` helpers here, so a manifest with only some extras installed still
imports cleanly. The first use of a module whose extra is missing raises a clear
``ImportError`` naming the install command.

``accepts_data_config`` lets a DataModule ``__init__`` accept the nested ``DataConfig``
shape (``DataModule(**cfg.data)``) without every subclass re-implementing the unpack.

``configure_cuvis_sdk`` picks the device the cuvis SDK processes on; ``require_cuvis``
applies it, because it is the one call every SDK entry in this package goes through.
"""

from __future__ import annotations

import functools
import itertools
import logging
import threading
import time
from collections.abc import Callable
from typing import Any, NamedTuple

from loguru import logger


def accepts_data_config(init: Callable) -> Callable:
    """Let a DataModule ``__init__`` also accept the nested ``DataConfig`` shape.

    The registry path (``create_data_module``) already spreads ``params`` into flat kwargs,
    but direct config-driven callers splat a whole ``DataConfig`` dict
    (``{data_module, splits, batch_size, num_workers, params}``) via ``DataModule(**cfg.data)``.
    This normalizes that shape onto the flat signature: it drops the redundant ``data_module``
    (the class identity fixes the module) and splices ``params`` entries in as flat kwargs. An
    explicit flat kwarg wins over the same key in ``params``. Unknown keys (flat or nested)
    still reach the wrapped ``__init__`` and raise ``TypeError`` -- no silent swallow.
    """

    @functools.wraps(init)
    def wrapper(self, **kwargs):
        """Normalize the nested ``DataConfig`` shape, then call the real ``__init__``."""
        kwargs.pop("data_module", None)
        params = kwargs.pop("params", None)
        if params:
            for key, value in params.items():
                kwargs.setdefault(key, value)
        return init(self, **kwargs)

    return wrapper


_requested_gpu_mode: str | None = None
_applied_gpu_mode: str | None = None


def configure_cuvis_sdk(*, cuda: bool) -> None:
    """Choose the device the SDK processes on. Applied on the first SDK use in this process.

    SDK 3.6.0 processes on the host unless a process calls ``cuvis.init``, which costs about
    260 ms per cube against 67 ms and leaves ``read_threads`` with almost nothing to overlap.
    The choice is only recorded here: ``cuvis.init`` has to happen before the first
    ``SessionFile``, so ``require_cuvis`` applies it rather than each caller remembering to.

    The SDK fixes its device at the **first** ``cuvis.init`` of a process and silently ignores
    every later one -- it returns success and keeps the original device -- so a second,
    conflicting choice cannot take effect and warns instead of pretending. For the same reason
    a host application that initialized the SDK itself keeps whatever it chose. Before the
    first SDK call the later request wins, and a disagreement warns too: the last-constructed
    DataModule would otherwise pick the device for every other one in the process in silence.
    """
    global _requested_gpu_mode
    mode = "cuda" if cuda else "host"
    if _applied_gpu_mode is not None and mode != _applied_gpu_mode:
        logger.warning(
            "cuvis SDK already initialized on '{}'; the request for '{}' cannot take effect, "
            "because the SDK fixes its device at the first init of a process.",
            _applied_gpu_mode,
            mode,
        )
        return
    if _requested_gpu_mode is not None and mode != _requested_gpu_mode:
        logger.warning(
            "cuvis SDK device requested as '{}' after an earlier request for '{}'; the SDK is "
            "initialized once per process, so '{}' is what every reader in this process gets.",
            mode,
            _requested_gpu_mode,
            mode,
        )
    _requested_gpu_mode = mode


def require_cuvis():
    """Return the ``cuvis`` SDK module, initialized once, or raise a clear install hint.

    Every SDK entry in this package goes through here, which is what makes it the one place
    that can guarantee ``cuvis.init`` runs before the first ``SessionFile`` -- including in a
    DataLoader worker, which is a fresh process that has initialized nothing.
    """
    global _applied_gpu_mode
    try:
        import cuvis
    except ImportError as e:  # pragma: no cover - exercised via the lazy-extras smoke
        raise ImportError(
            "The 'cuvis' SDK is required for the cu3s data modules. "
            "Install with: uv pip install 'cuvis-ai-dataloader[cu3s]'"
        ) from e

    if _requested_gpu_mode is not None and _applied_gpu_mode is None:
        # WARNING, not the SDK's own DEBUG default, which prints a line per processed cube.
        cuvis.init(
            cuvis.SdkSettings(force_gpu_mode=_requested_gpu_mode),
            global_loglevel=logging.WARNING,
        )
        _applied_gpu_mode = _requested_gpu_mode
        logger.debug("cuvis SDK initialized with force_gpu_mode={}", _applied_gpu_mode)
    return cuvis


_RELEASES_GIL: bool | None = None
_PROBE_SECONDS = 0.03


def _spin_rate(work: Callable[[], object]) -> float:
    """Counter iterations per second reached by a second thread while ``work`` repeats.

    ``work`` repeats until the probe window is full rather than running once, because a call
    shorter than the window would otherwise be measured over a span too brief to mean
    anything: the spinner's rate over a few microseconds is scheduling noise. A real cube read
    is longer than the window, so it runs exactly once.
    """
    counter = itertools.count()
    stop = threading.Event()

    def spin() -> None:
        """Increment until told to stop."""
        while not stop.is_set():
            next(counter)

    spinner = threading.Thread(target=spin, daemon=True)
    spinner.start()
    started = time.perf_counter()
    try:
        while time.perf_counter() - started < _PROBE_SECONDS:
            work()
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        spinner.join()
    return next(counter) / elapsed if elapsed > 0 else 0.0


def cuvis_releases_gil(sdk_call: Callable[[], object]) -> bool:
    """Whether SDK calls let other Python threads run. Probed once per process.

    Probed rather than read off a version, because the GIL release is an unversioned
    binding change that no requirement can express, and on a binding that holds the GIL
    extra reader threads cost throughput rather than gaining it. ``sdk_call`` must be a
    real SDK call long enough to observe, i.e. a cube read. The baseline uses ``sleep``,
    which does release the GIL, so the ratio is near 1 when the SDK does too and near 0
    when it does not: an effect large enough to survive a loaded machine.
    """
    global _RELEASES_GIL
    if _RELEASES_GIL is None:
        baseline = _spin_rate(lambda: time.sleep(0.005))
        _RELEASES_GIL = bool(baseline) and _spin_rate(sdk_call) / baseline > 0.2
    return _RELEASES_GIL


def require_tifffile():
    """Return the ``tifffile`` module, or raise a clear install hint."""
    try:
        import tifffile

        return tifffile
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "tifffile is required for --data-module tiff_paired. "
            "Install with: uv pip install 'cuvis-ai-dataloader[tiff]'"
        ) from e


def require_pycocotools():
    """Return ``pycocotools.coco.COCO``, or raise a clear install hint."""
    try:
        from pycocotools.coco import COCO

        return COCO
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pycocotools is required for COCO labels. "
            "Install with: uv pip install 'cuvis-ai-dataloader[coco]'"
        ) from e


def require_skimage_polygon2mask():
    """Return ``skimage.draw.polygon2mask``, or raise a clear install hint."""
    try:
        from skimage.draw import polygon2mask

        return polygon2mask
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "scikit-image is required for COCO polygon labels. "
            "Install with: uv pip install 'cuvis-ai-dataloader[coco]'"
        ) from e


_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}


def parse_bool(s, *, key: str) -> bool:
    """Coerce a ``--data-arg`` string (or bool) to bool."""
    if isinstance(s, bool):
        return s
    token = str(s).lower()
    if token in _TRUE:
        return True
    if token in _FALSE:
        return False
    raise ValueError(f"--data-arg {key}={s!r}: expected one of {sorted(_TRUE | _FALSE)}")


def parse_float_list(s, *, key: str) -> list[float]:
    """Comma-floats (or an existing list) -> list[float]."""
    if isinstance(s, (list, tuple)):
        return [float(x) for x in s]
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_int_list(s, *, key: str) -> list[int]:
    """Comma list of ints and inclusive ``start-stop[:step]`` ranges -> list[int].

    Accepts an existing list/tuple, or a string like ``"0,2,4"``, ``"0-100"``, or
    ``"0-10:2, 20"``. Range tokens expand via the shared core helper, so
    ``measurement_indices`` accepts the same range syntax as split id-lists.
    """
    from cuvis_ai_core.utils.general import expand_range_selectors

    tokens = (
        list(s)
        if isinstance(s, (list, tuple))
        else [t.strip() for t in str(s).split(",") if t.strip()]
    )
    return [int(x) for x in expand_range_selectors(tokens)]


def parse_str_list(s, *, key: str) -> list[str]:
    """Comma-strings (or an existing list) -> list[str]."""
    if isinstance(s, (list, tuple)):
        return [str(x).strip() for x in s]
    return [x.strip() for x in str(s).split(",") if x.strip()]


# Every frame read ahead is a whole cube (about 264 MB at 1000x1080x61 float32) held in host
# or device memory until the model asks for it; the 8 GB laptops set this ceiling.
MAX_READ_AHEAD = 8


class Cu3sReaderOptions(NamedTuple):
    """The reader-side parameters both cu3s DataModules validate the same way."""

    max_open_sessions: int
    read_threads: int
    source_coherent_batches: bool
    sdk_cuda: bool
    cuda_cubes: bool
    read_ahead: int = 0


def parse_cu3s_reader_options(
    *,
    max_open_sessions: Any,
    read_threads: Any,
    source_coherent_batches: Any,
    sdk_cuda: Any,
    cuda_cubes: Any,
    num_workers: Any,
    read_ahead: Any = 0,
    batch_size: Any = 1,
) -> Cu3sReaderOptions:
    """Validate the reader-side parameters of a cu3s DataModule and record the SDK device.

    Shared by ``Cu3sDataModule`` and ``MultiCu3sDataModule`` so the two cannot drift on a
    guard or a message. Every refusal names the parameters involved, and the SDK device is
    recorded only once the whole set is acceptable, so a rejected module leaves no trace.
    ``batch_size`` is only looked at, never changed: CuvisNEXT patches it into the trainrun at
    fill time, so the one rule that depends on it is a warning here rather than a check on
    the yaml.
    """
    open_sessions = int(max_open_sessions)
    if open_sessions < 1:
        raise ValueError(f"max_open_sessions must be >= 1, got {max_open_sessions}")
    threads = int(read_threads)
    if threads < 0:
        raise ValueError(f"read_threads must be >= 0, got {read_threads}")
    ahead = int(read_ahead)
    if ahead < 0:
        raise ValueError(f"read_ahead must be >= 0, got {read_ahead}")
    if ahead > MAX_READ_AHEAD:
        raise ValueError(
            f"read_ahead must be <= {MAX_READ_AHEAD}, got {read_ahead}; every frame read ahead "
            "is a whole cube held in memory until the model asks for it."
        )
    coherent = bool(source_coherent_batches)
    cuda = parse_bool(sdk_cuda, key="sdk_cuda")
    device_cubes = parse_bool(cuda_cubes, key="cuda_cubes")
    # A device-resident cube only exists when the SDK processed it on the device, and a
    # CUDA tensor cannot be handed across the worker queue, so neither is a silent demotion.
    if device_cubes and not cuda:
        raise ValueError(
            "cuda_cubes=True requires sdk_cuda=True; the SDK has no device buffer to "
            "hand out when it processes on the host."
        )
    if device_cubes and int(num_workers) > 0:
        raise ValueError(
            f"cuda_cubes=True cannot be combined with num_workers={num_workers}; CUDA "
            "tensors are not sent across DataLoader worker processes, so set num_workers=0."
        )
    # Process workers each build their own sessions and their own ProcessingContext, so
    # combining them multiplies both the handle count and the ~9 s context build. The
    # failure mode is an OOM or a killed CUDA process, not a slowdown, so refuse instead
    # of silently overriding either knob.
    if threads > 1 and int(num_workers) > 0:
        raise ValueError(
            f"read_threads={read_threads} cannot be combined with num_workers="
            f"{num_workers}; reader threads replace DataLoader worker processes, so set "
            "num_workers=0 to use them."
        )
    if ahead > 0 and int(num_workers) > 0:
        raise ValueError(
            f"read_ahead={read_ahead} cannot be combined with num_workers={num_workers}; the "
            "read-ahead runs on reader threads inside the training process, so set "
            "num_workers=0 to use it."
        )
    # torch hands a map-style dataset one batch of indices at a time, so at batch_size 1
    # extra handles never read concurrently: they cost memory and buy nothing. A warning,
    # not an error, because the batch size is decided at run time by the caller.
    if threads > 1 and int(batch_size) == 1 and ahead == 0:
        logger.warning(
            "read_threads={} with batch_size=1 and no read_ahead opens {} session handles that "
            "never read concurrently: torch asks for one frame at a time. Set read_ahead (frames "
            "to read ahead of the model step) or raise batch_size.",
            read_threads,
            threads,
        )
    # Recorded before anything can open a session, since the SDK fixes its device at the
    # first init of a process and ignores every later one.
    configure_cuvis_sdk(cuda=cuda)
    return Cu3sReaderOptions(open_sessions, threads, coherent, cuda, device_cubes, ahead)
