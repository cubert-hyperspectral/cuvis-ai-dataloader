"""Internal TIFF cube reader (tifffile). Not a plugin contract.

Reads multi-band TIFFs (axes SYX / YXS / YX) and parses wavelengths from the
GDAL_METADATA ENVI-format tag (id 42112). Heavy import (tifffile) happens lazily
inside ``read`` via ``require_tifffile``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .._extras import require_tifffile

_WAVELENGTH_RE = re.compile(r'<Item name="wavelength"[^>]*>\{([^}]+)\}')


def _parse_wavelengths_from_gdal(gdal_metadata: str) -> np.ndarray:
    """Extract the wavelength vector from a GDAL_METADATA ENVI tag string."""
    match = _WAVELENGTH_RE.search(gdal_metadata or "")
    if not match:
        raise ValueError("GDAL_METADATA tag has no <Item name='wavelength'> entry")
    return np.array([float(v) for v in match.group(1).split(",") if v.strip()], dtype=np.float32)


class TiffCubeReader:
    """Reads a single multi-band TIFF into ``{"cube", "wavelengths"}``."""

    def __init__(self, *, wavelengths_override: list[float] | None = None) -> None:
        self._wavelengths_override = (
            np.asarray(wavelengths_override, dtype=np.float32) if wavelengths_override else None
        )

    def read(self, path: str | Path) -> dict:
        """Read ``path`` into ``{"cube": (H, W, C) float32, "wavelengths": float32[C]}``."""
        tifffile = require_tifffile()
        path = Path(path)
        with tifffile.TiffFile(str(path)) as tf:
            series = tf.series[0]
            raw = series.asarray()
            axes = series.axes
            gdal = ""
            try:
                gdal = tf.pages[0].tags["GDAL_METADATA"].value
            except (KeyError, AttributeError, IndexError):
                gdal = ""

        cube = self._to_hwc(raw, axes, path)

        if self._wavelengths_override is not None:
            wavelengths = self._wavelengths_override
        else:
            wavelengths = _parse_wavelengths_from_gdal(gdal)
        if wavelengths.size != cube.shape[-1]:
            raise ValueError(
                f"{path}: wavelength count {wavelengths.size} != band count {cube.shape[-1]}"
            )
        return {"cube": cube, "wavelengths": wavelengths}

    @staticmethod
    def _to_hwc(raw: np.ndarray, axes: str, path: Path) -> np.ndarray:
        """Normalize the array to (H, W, C) float32 from SYX / YXS / YX layouts."""
        if axes == "SYX" or (raw.ndim == 3 and axes not in ("YXS", "YX")):
            cube = np.transpose(raw, (1, 2, 0))  # (C, H, W) -> (H, W, C)
        elif axes == "YXS":
            cube = raw  # already (H, W, C)
        elif axes == "YX" or raw.ndim == 2:
            cube = raw[..., None]  # grayscale -> (H, W, 1)
        else:
            raise ValueError(
                f"Unsupported TIFF axis layout: {axes!r} (shape {raw.shape}). "
                f"Supported: 'SYX', 'YXS', 'YX'. File: {path}"
            )
        return np.ascontiguousarray(cube.astype(np.float32))
