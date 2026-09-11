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


def _compatible_mem_free(cuvis_il, real_free):
    """Free a device buffer through whichever calling convention the binding accepts.

    The C API declares ``cuvis_cuda_mem_free(CUVIS_CUDA_MEM* i_mem)``, a pointer, but
    ``cuvis`` 3.6.0.0rc1 passes the handle by value from both ``CudaImageData.__del__`` and
    the DLPack capsule deleter, so every free raises ``TypeError`` and no buffer is ever
    returned to the SDK's pool. Trying the wrapper's own form first means this stops doing
    anything once the wrapper is fixed (cuvis.python#98).
    """

    def free(handle):
        """Release one CUVIS_CUDA_MEM handle."""
        try:
            return real_free(handle)
        except TypeError:
            box = cuvis_il.new_p_int()
            cuvis_il.p_int_assign(box, int(handle))
            return real_free(box)

    return free


def _repair_binding() -> bool:
    """Work around ``cuvis`` 3.6.0.0rc1's broken device-buffer path (cuvis.python#98).

    ``CudaImageData._view`` calls ``cuvis_il.cuvis_cuda_view_ptr``, which no ``cuvis-il``
    build exports and which was never written, so ``to_torch`` raises ``AttributeError`` on an
    otherwise working device buffer. SWIG already converts the ``void *`` field to an int,
    which is all the missing helper would have done. Both repairs here are in the wrapper's
    Python layer; the native SDK and the binding are correct. Returns whether the device
    buffer could be made reachable.
    """
    try:
        from cuvis_il import cuvis_il
    except ImportError:  # pragma: no cover - require_cuvis has already raised by here
        return False
    if not hasattr(cuvis_il, "cuvis_cuda_view_ptr"):
        if not hasattr(cuvis_il.cuvis_cuda_mem_view_t(), "device_ptr"):
            return False
        cuvis_il.cuvis_cuda_view_ptr = lambda view: int(view.device_ptr)
        logger.debug("patched the absent cuvis_il.cuvis_cuda_view_ptr helper")
    if not getattr(cuvis_il.cuvis_cuda_mem_free, "_cuvis_ai_compatible", False):
        patched = _compatible_mem_free(cuvis_il, cuvis_il.cuvis_cuda_mem_free)
        patched._cuvis_ai_compatible = True
        cuvis_il.cuvis_cuda_mem_free = patched
    return True


def cuda_cubes_available(cuvis) -> bool:
    """Whether this build, device and binding can hand out device-resident cubes.

    Three separate things can deny it and the SDK answers only two: the library may lack the
    CUDA functions, and the device may not support them. The third is the wrapper defect
    :func:`_repair_binding` covers, which ``capabilities()`` cannot see because it probes the
    native symbols rather than the Python layer over them.
    """
    global _available
    if _available is None:
        cuda = getattr(cuvis, "cuda", None)  # absent before cuvis 3.6.0
        _available = (
            cuda is not None and bool(cuda.capabilities().same_process) and _repair_binding()
        )
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
            "device-resident cubes are unavailable here (needs cuvis >= 3.6.0, a CUDA device "
            "the SDK accepts, and a binding exposing the device-buffer view); reading cubes "
            "through host memory instead."
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
