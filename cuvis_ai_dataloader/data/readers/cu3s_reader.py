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

    def __init__(
        self,
        cu3s_file_path: str,
        *,
        processing_mode: str | None = "Reflectance",
        white_ref: str | Path | None = None,
        dark_ref: str | Path | None = None,
    ) -> None:
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

        # Custom references are installed BEFORE the processing mode is applied, so
        # the Reflectance/SpectralRadiance validation can count them.
        self._custom_ref_handles: list = []  # keep SDK handles alive for the reader's lifetime
        self.custom_references = self._set_custom_references(
            cuvis, white_ref=white_ref, dark_ref=dark_ref
        )
        self._processing_applied = self._apply_processing_mode(cuvis, processing_mode)

        mesu0 = self.session.get_measurement(0)
        self.num_channels = mesu0.cube.channels
        self.wavelengths = np.array(mesu0.cube.wavelength).ravel()
        self.total_measurements = len(self.session)
        logger.debug(
            f"Opened cu3s {self.cu3s_file_path}: {self.total_measurements} measurements, "
            f"{self.num_channels} channels"
        )

    def _set_custom_references(
        self, cuvis, *, white_ref: str | Path | None, dark_ref: str | Path | None
    ) -> dict[str, str]:
        """Override the session's baked white/dark references from external cu3s recordings.

        Some sessions carry wrong baked references — e.g. a factory-fallback flat white/dark
        captured at a mismatched integration time — which renders but is radiometrically wrong
        (reflectance an order of magnitude off). Passing ``white_ref`` / ``dark_ref`` (paths to
        cu3s reference recordings) re-references processing: each reference's **measurement 0**
        is loaded via ``get_measurement(0)`` — deliberately NOT ``get_reference(...)``, which on
        an affected recording can itself return the bogus baked factory reference — and installed
        with ``ProcessingContext.set_reference`` before any ``apply``.

        References should be day-matched to the measurement (same site/session conditions and
        integration time); never cross days. Returns the applied overrides as
        ``{"white": path, "dark": path}`` (only the ones given).
        """
        applied: dict[str, str] = {}
        for kind, ref_path, ref_type in (
            ("white", white_ref, cuvis.ReferenceType.White),
            ("dark", dark_ref, cuvis.ReferenceType.Dark),
        ):
            if ref_path is None:
                continue
            ref_path = str(ref_path)
            if not os.path.exists(ref_path):
                raise ValueError(f"{kind} reference cu3s does not exist: {ref_path}")
            if Path(ref_path).suffix != ".cu3s":
                raise ValueError(f"{kind} reference must be a .cu3s file: {ref_path}")
            ref_session = cuvis.SessionFile(ref_path)
            try:
                ref_mesu = ref_session.get_measurement(0)
            except Exception as exc:
                raise ValueError(
                    f"failed to read measurement 0 of {kind} reference {ref_path}: {exc}"
                ) from exc
            if ref_mesu is None:
                raise ValueError(f"{kind} reference {ref_path} has no measurement 0")
            self.pc.set_reference(ref_mesu, ref_type)
            self._custom_ref_handles.extend((ref_session, ref_mesu))
            applied[kind] = ref_path
            logger.info(
                "cu3s {}: {} reference overridden from {}",
                Path(self.cu3s_file_path).name,
                kind,
                ref_path,
            )
        return applied

    def _apply_processing_mode(self, cuvis, processing_mode) -> bool:
        """Configure the processing context for ``processing_mode``.

        Returns ``True`` when a mode was set, so ``read`` knows to always apply it rather than
        trusting a possibly-raw cube already present in the measurement.
        """
        if processing_mode is None:
            return False
        if isinstance(processing_mode, str):
            resolved = getattr(cuvis.ProcessingMode, processing_mode, None)
            if resolved is None:
                raise ValueError(
                    f"unknown processing_mode {processing_mode!r}; "
                    "expected a cuvis.ProcessingMode name (e.g. 'Raw', 'Reflectance', "
                    "'SpectralRadiance')."
                )
            processing_mode = resolved
        # A custom reference satisfies the requirement without consulting the session's
        # baked references (short-circuit: get_reference is not even called for that slot).
        has_white = (
            "white" in self.custom_references
            or self.session.get_reference(0, cuvis.ReferenceType.White) is not None
        )
        has_dark = (
            "dark" in self.custom_references
            or self.session.get_reference(0, cuvis.ReferenceType.Dark) is not None
        )
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
        return True

    @property
    def wavelengths_nm(self) -> np.ndarray:
        """Per-channel wavelengths (nm, int32) from the first measurement."""
        mesu = self.session.get_measurement(0)
        return np.array(mesu.cube.wavelength, dtype=np.int32).ravel()

    def read(self, mesu_index: int) -> dict:
        """Return ``{"cube", "mesu_index", "wavelengths"}`` for one measurement."""
        mesu = self.session.get_measurement(mesu_index)
        # A requested processing mode is always applied: a cube already present in mesu.data may
        # be the recorded (raw) cube, so trusting it would silently bypass the requested mode.
        # With no mode set (processing_mode=None) the file's data is used as-is unless absent.
        if self._processing_applied or "cube" not in mesu.data:
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
        for obj in getattr(self, "_custom_ref_handles", ()):  # custom-reference sessions
            closer = getattr(obj, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # pragma: no cover - SDK teardown is best-effort
                    pass
        self._custom_ref_handles = []

    def __enter__(self) -> Cu3sCubeReader:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
