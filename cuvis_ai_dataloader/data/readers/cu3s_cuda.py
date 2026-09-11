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
            "device-resident cubes are unavailable here (needs cuvis >= 3.6.0.0rc2 and a CUDA "
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


def read_cube(mesu) -> tuple[Any, np.ndarray]:
    """The measurement's cube and wavelengths, from device memory when that mode is on.

    Returns a ``torch.Tensor`` on the GPU in device mode and a ``numpy.ndarray`` otherwise.
    In device mode ``mesu.cube`` is not merely slower, it is ``None``: the host fetch that
    would populate it is exactly what was skipped, so everything has to come off the
    ``CudaImageData``.
    """
    if not _enabled:
        return mesu.cube.array, np.array(mesu.cube.wavelength, dtype=np.int32).ravel()
    image = mesu.get_cube_cuda()
    return image.to_torch(), np.array(image.wavelength, dtype=np.int32).ravel()
