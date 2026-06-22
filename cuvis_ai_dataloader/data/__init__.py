"""Concrete DataModules and their internal readers/labelers.

Public DataModules (selected by ``DATA_MODULE_NAME`` in the manifest):

- ``Cu3sDataModule`` (``cu3s``, extras ``[cu3s, coco]``)
- ``TiffPairedDataModule`` (``tiff_paired``, extras ``[tiff]``)

Module tops stay free of heavy deps (cuvis / tifffile / pycocotools / scikit-image
load lazily on first use).
"""

from __future__ import annotations

from .datamodule_cu3s import Cu3sDataModule
from .datamodule_cu3s_multi import MultiCu3sDataModule
from .datamodule_tiff_paired import TiffPairedDataModule

__all__ = [
    "Cu3sDataModule",
    "MultiCu3sDataModule",
    "TiffPairedDataModule",
]
