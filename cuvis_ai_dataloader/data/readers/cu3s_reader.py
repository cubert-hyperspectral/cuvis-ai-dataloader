"""Internal cu3s cube reader (cuvis SDK). Not a plugin contract.

Opens a ``.cu3s`` session, applies the processing mode, and reads per-measurement
cube dicts. The heavy ``cuvis`` import happens lazily in ``__init__`` via
``require_cuvis`` so importing this module never pulls the SDK.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import numpy as np
from loguru import logger

from .._extras import require_cuvis


def count_measurements(cu3s_file_path: str | Path) -> int:
    """How many measurements a ``.cu3s`` holds, without building a processing context.

    Enumeration needs the count and nothing else. A full :class:`Cu3sCubeReader` costs a
    ``ProcessingContext`` (which reserves SDK GPU processing pools) plus a read of
    measurement 0, per file; ``len(SessionFile)`` is ``get_size()`` on the bare handle, so
    this opens the recording, asks, and lets it go again.

    Failures name the recording. The SDK raises a bare ``SDKException`` with no path in it,
    which says nothing useful when a split names dozens of files and one of them is corrupt
    or held open by another program. The handle is dropped in ``finally``: several probe
    handles alive at once is exactly the situation the training pre-flight exists to avoid.
    """
    cuvis = require_cuvis()
    path = str(cu3s_file_path)
    session = None
    try:
        session = cuvis.SessionFile(path)
        total = int(len(session))
    except Exception as e:
        raise ValueError(f"cannot open cu3s {path}: {e}") from e
    finally:
        del session
    logger.debug(f"Probed cu3s {path}: {total} measurements")
    return total


def _parse_ref_spec(spec: str) -> tuple[str, int]:
    """Parse a reference spec ``"path"`` or ``"path:frame"`` into ``(path, frame)``.

    ``frame`` defaults to ``0`` (the reference session's first measurement); ``frame == -1``
    selects that session's own embedded/baked reference (matching ``cuvis_batch_exporter``'s
    ``:frame_no`` with ``-1`` = embedded). The trailing ``:`` is split off only when the tail is
    an integer, so plain paths (including paths that carry no frame suffix) are unaffected.
    """
    head, sep, tail = spec.rpartition(":")
    if sep and head:
        try:
            return head, int(tail)
        except ValueError:
            pass
    return spec, 0


def total_measurements_of(cu3s_file_path: str | Path) -> int:
    """Frame count of a ``.cu3s`` without building a ProcessingContext.

    ``Cu3sCubeReader`` builds one in ``__init__`` whatever the processing mode, and that build
    is the expensive part, so a probe that only wants the count must not go through it.
    """
    cuvis = require_cuvis()
    return int(len(cuvis.SessionFile(str(cu3s_file_path))))


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
        # Share our context so the SDK's lazy ``Measurement.cube`` path reuses it. Otherwise reading
        # mesu0.cube below leaves session._pc unset and the SDK builds a second, never-used
        # ProcessingContext, wasting a full init and holding duplicate GPU + host buffers for the
        # reader's lifetime (~130 MB VRAM per reader on a 410x410x164 session).
        self.session._pc = self.pc

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
        """Override the session's baked white/dark references with external cu3s recordings.

        Lets an application supply its own references at load time instead of the ones baked into
        the session — e.g. reusing a shared calibration across sessions, non-destructively
        re-processing with updated references without re-exporting, or reading sessions that carry
        no usable baked references.

        Each reference is given as ``path`` or ``path:frame``: ``path`` / ``path:0`` uses the
        reference session's **measurement 0** (the default); ``path:N`` uses measurement ``N`` (for a
        session that holds several references); and ``path:-1`` uses that session's own
        **embedded/baked** reference (matching ``cuvis_batch_exporter``'s ``:frame_no`` with ``-1`` =
        embedded). References are loaded via ``get_measurement`` — deliberately NOT
        ``get_reference``, which on some sessions can return an unintended baked reference — except
        the explicit ``-1`` case, which asks for the embedded reference on purpose. The chosen
        reference is installed with ``ProcessingContext.set_reference`` before any ``apply``.

        This *supplies* references; it does not *repair* wrong ones. If a session's baked references
        are actually incorrect, that is a data problem — correct them at the source with the exporter
        (``cuvis_batch_exporter --force_white/--force_dark``), which is byte-identical to this
        override. Supplied references should match the measurement's capture conditions (same
        site/session and integration time). Returns the applied overrides as
        ``{"white": spec, "dark": spec}`` (only the ones given, echoing the input spec).
        """
        applied: dict[str, str] = {}
        for kind, ref_spec, ref_type in (
            ("white", white_ref, cuvis.ReferenceType.White),
            ("dark", dark_ref, cuvis.ReferenceType.Dark),
        ):
            if ref_spec is None:
                continue
            ref_spec = str(ref_spec)
            ref_path, frame = _parse_ref_spec(ref_spec)
            if not os.path.exists(ref_path):
                raise ValueError(f"{kind} reference cu3s does not exist: {ref_path}")
            if Path(ref_path).suffix != ".cu3s":
                raise ValueError(f"{kind} reference must be a .cu3s file: {ref_path}")
            ref_session = cuvis.SessionFile(ref_path)
            self._custom_ref_handles.append(ref_session)
            if frame < 0:
                # Explicit opt-in to the reference session's own embedded/baked reference
                # (``path:-1``). Unlike the get_measurement path this consults get_reference,
                # because the user has asked for the embedded reference specifically.
                ref_pc = cuvis.ProcessingContext(ref_session)
                self._custom_ref_handles.append(ref_pc)
                try:
                    ref_mesu = ref_pc.get_reference(ref_type)
                except Exception as exc:
                    raise ValueError(
                        f"failed to read embedded {kind} reference from {ref_path}: {exc}"
                    ) from exc
                if ref_mesu is None:
                    raise ValueError(
                        f"{kind} reference {ref_path} has no embedded {kind} reference (frame -1)"
                    )
            else:
                try:
                    ref_mesu = ref_session.get_measurement(frame)
                except Exception as exc:
                    raise ValueError(
                        f"failed to read measurement {frame} of {kind} reference {ref_path}: {exc}"
                    ) from exc
                if ref_mesu is None:
                    raise ValueError(f"{kind} reference {ref_path} has no measurement {frame}")
            self.pc.set_reference(ref_mesu, ref_type)
            self._custom_ref_handles.append(ref_mesu)
            applied[kind] = ref_spec
            logger.info(
                "cu3s {}: {} reference overridden from {}",
                Path(self.cu3s_file_path).name,
                kind,
                ref_spec,
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
        return self._read_with(self.session, mesu_index)

    def _read_with(self, session, mesu_index: int) -> dict:
        """Read one measurement through ``session``, sharing this reader's context.

        Split out so a pooled subclass can read on a leased handle without copying the
        processing-mode rule below, which the two must not drift on.
        """
        mesu = session.get_measurement(mesu_index)
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

    def iter_reads(self, indices: Iterable[int]) -> Iterator[dict]:
        """Yield one read per index, in the order requested."""
        return (self.read(index) for index in indices)

    def read_many(self, indices: Sequence[int]) -> list[dict]:
        """Read several measurements, in the order requested.

        Defined here so callers can hand a whole batch over without caring which reader they
        hold; ``Cu3sPrefetchReader`` overrides ``iter_reads`` to overlap them.
        """
        return list(self.iter_reads(indices))

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
