"""Internal cu3s cube reader (cuvis SDK). Not a plugin contract.

Opens a ``.cu3s`` session, applies the processing mode, and reads per-measurement
cube dicts. The heavy ``cuvis`` import happens lazily in ``__init__`` via
``require_cuvis`` so importing this module never pulls the SDK.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from loguru import logger

from .._extras import require_cuvis


class Cu3sCubeReader:
    """Reads cube frames from a ``.cu3s`` session via the cuvis SDK."""

    def __init__(self, cu3s_file_path: str, *, processing_mode: str | None = "Reflectance") -> None:
        cuvis = require_cuvis()
        self.cu3s_file_path = str(cu3s_file_path)
        if not os.path.exists(self.cu3s_file_path):
            raise ValueError(f"cu3s path does not exist: {self.cu3s_file_path}")
        if Path(self.cu3s_file_path).suffix != ".cu3s":
            raise ValueError(f"path must point to a .cu3s file: {self.cu3s_file_path}")

        self.session = cuvis.SessionFile(self.cu3s_file_path)
        self.pc = cuvis.ProcessingContext(self.session)

        self.fps: float | None = None
        try:
            self.fps = float(self.session.fps)
        except Exception:
            self.fps = None

        self._apply_processing_mode(cuvis, processing_mode)

        mesu0 = self.session.get_measurement(0)
        self.num_channels = mesu0.cube.channels
        self.wavelengths = np.array(mesu0.cube.wavelength).ravel()
        self.total_measurements = len(self.session)
        logger.debug(
            f"Opened cu3s {self.cu3s_file_path}: {self.total_measurements} measurements, "
            f"{self.num_channels} channels"
        )

    def _apply_processing_mode(self, cuvis, processing_mode) -> None:
        if processing_mode is None:
            return
        if isinstance(processing_mode, str):
            resolved = getattr(cuvis.ProcessingMode, processing_mode, None)
            if resolved is None:
                raise ValueError(
                    f"unknown processing_mode {processing_mode!r}; "
                    "expected a cuvis.ProcessingMode name (e.g. 'Raw', 'Reflectance', "
                    "'SpectralRadiance')."
                )
            processing_mode = resolved
        has_white = self.session.get_reference(0, cuvis.ReferenceType.White) is not None
        has_dark = self.session.get_reference(0, cuvis.ReferenceType.Dark) is not None
        if processing_mode == cuvis.ProcessingMode.Reflectance and not (has_white and has_dark):
            raise ValueError(
                "Reflectance processing mode requires both White and Dark references "
                "in the cu3s file."
            )
        spectral_radiance_mode = getattr(cuvis.ProcessingMode, "SpectralRadiance", None)
        if (
            spectral_radiance_mode is not None
            and processing_mode == spectral_radiance_mode
            and not has_dark
        ):
            raise ValueError(
                "SpectralRadiance processing mode requires a Dark reference in the cu3s file."
            )
        self.pc.processing_mode = processing_mode

    @property
    def wavelengths_nm(self) -> np.ndarray:
        mesu = self.session.get_measurement(0)
        return np.array(mesu.cube.wavelength, dtype=np.int32).ravel()

    def read(self, mesu_index: int) -> dict:
        """Return ``{"cube", "mesu_index", "wavelengths"}`` for one measurement."""
        mesu = self.session.get_measurement(mesu_index)
        if "cube" not in mesu.data:
            mesu = self.pc.apply(mesu)
        cube_array: np.ndarray = mesu.cube.array
        wavelengths = np.array(mesu.cube.wavelength, dtype=np.int32).ravel()
        return {
            "cube": cube_array,
            "mesu_index": int(mesu_index),
            "wavelengths": wavelengths,
        }

    def close(self) -> None:
        """Release the SDK processing context + session (best-effort).

        Drops the native handles so they don't accumulate when many sources are
        opened (e.g. multi-file validation). Safe to call more than once.
        """
        for attr in ("pc", "session"):
            obj = getattr(self, attr, None)
            if obj is None:
                continue
            closer = getattr(obj, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # pragma: no cover - SDK teardown is best-effort
                    pass
            setattr(self, attr, None)

    def __enter__(self) -> Cu3sCubeReader:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
