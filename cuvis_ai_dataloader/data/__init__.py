"""Concrete DataModules and their internal readers/labelers.

Public DataModules (selected by ``DATA_MODULE_NAME`` in the manifest):

- ``Cu3sDataModule`` (``cu3s``, extras ``[cu3s, coco]``)
- ``TiffPairedDataModule`` (``tiff_paired``, extras ``[tiff]``)

``SingleCu3sDataModule`` / ``SingleCu3sDataset`` are back-compat aliases for the
former ``cuvis_ai_core.data.datasets`` symbols, so old call sites migrate with
only an import-path change. Module tops stay free of heavy deps (cuvis / tifffile
/ pycocotools / scikit-image load lazily on first use).
"""

from __future__ import annotations

from .datamodule_cu3s import (
    Cu3sDataModule,
    SingleCu3sDataModule,
    SingleCu3sDataset,
)
from .datamodule_tiff_paired import TiffPairedDataModule

__all__ = [
    "Cu3sDataModule",
    "SingleCu3sDataModule",
    "SingleCu3sDataset",
    "TiffPairedDataModule",
]
