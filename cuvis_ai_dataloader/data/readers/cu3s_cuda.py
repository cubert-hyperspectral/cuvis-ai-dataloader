"""Device-resident cu3s cubes: the capability probe, the SDK mode switch, and the read.

Without this, a cube processed on the GPU is copied to host memory and the device copy is
freed, only for torch to copy it straight back for training. ``cuvis.cuda`` keeps it where it
already is and hands out a zero-copy device tensor through DLPack.

The switch is process-global on the SDK's side (``cuvis.cuda.enable`` clears
``Measurement._refresh_images`` for the whole process), so it is process-global here too, and
it has to be thrown before the first measurement is processed.

Kept out of ``cu3s_reader.py`` so that module stays readable as the host-memory contract, and
so nothing imports the CUDA probe merely to read a cube.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from loguru import logger

_enabled = False
_available: bool | None = None


def cuda_cubes_available(cuvis) -> bool:
    """Whether this build and device can hand out device-resident cubes.

    Two things can deny it: the library may lack the CUDA functions, which ``cuvis.binding``
    answers without calling anything, and the device or driver may not support them, which
    only the SDK can answer. ``capabilities()`` covers both.
    """
    global _available
    if _available is None:
        cuda = getattr(cuvis, "cuda", None)  # absent before cuvis 3.6.0
        _available = cuda is not None and bool(cuda.capabilities().same_process)
    return _available


def enable_cuda_cubes(cuvis) -> bool:
    """Switch the SDK to device-resident cubes for this process. Returns whether it took.

    Must run before the first measurement is processed: the flag it sets is read while a
    measurement is being filled, so a cube processed earlier has already been copied to the
    host and its device copy freed.
    """
    global _enabled
    if _enabled:
        return True
    if not cuda_cubes_available(cuvis):
        logger.warning(
            "device-resident cubes are unavailable here (needs cuvis >= 3.6.0.0 and a CUDA "
            "device the SDK accepts); reading cubes through host memory instead."
        )
        return False
    cuvis.cuda.enable()
    _enabled = True
    logger.debug("cuvis CUDA cube mode enabled; cubes stay in device memory")
    return True


def is_enabled() -> bool:
    """Whether this process reads cubes as device tensors."""
    return _enabled


def read_cube(mesu, *, device: bool) -> tuple[Any, np.ndarray]:
    """The measurement's cube and wavelengths, as the reader that owns it promised them.

    ``device`` is the reader's own ``cuda_cubes``. The SDK's mode is process-wide and one-way,
    so once any reader has switched it on, ``mesu.cube`` is ``None`` for every reader: the host
    fetch that would populate it is exactly what was skipped, and everything has to come off
    the ``CudaImageData``. A reader that promised host cubes then copies back, so its callers
    keep getting ``numpy.ndarray`` rather than a CUDA tensor they never asked for.
    """
    if not _enabled:
        return mesu.cube.array, np.array(mesu.cube.wavelength, dtype=np.int32).ravel()
    image = mesu.get_cube_cuda()
    wavelengths = np.array(image.wavelength, dtype=np.int32).ravel()
    tensor = image.to_torch()
    if device:
        return tensor, wavelengths
    return tensor.cpu().numpy(), wavelengths


def sync_device_cube(cube: Any) -> None:
    """Wait for the device before a cube read on one thread is consumed on another.

    The SDK's DLPack export ignores the stream torch passes it and the SDK exposes no stream
    or event of its own, so nothing orders the SDK's writes against the consumer's stream. A
    device-wide synchronize on the producing thread is the one primitive that covers a stream
    nobody can name; it is provisional until the SDK offers an event. A host cube needs none.
    """
    if isinstance(cube, torch.Tensor) and cube.is_cuda:
        torch.cuda.synchronize(cube.device)
