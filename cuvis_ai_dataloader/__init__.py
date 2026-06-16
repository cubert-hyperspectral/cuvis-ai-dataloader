"""cuvis-ai-dataloader: pluggable hyperspectral DataModules for Cuvis.AI.

Ships concrete ``BaseCuvisAIDataModule`` subclasses (cu3s + COCO, TIFF +
paired-PNG) with per-format heavy deps gated behind optional extras. The ``cuvis``
SDK lives here only, behind the ``[cu3s]`` extra; no other Cuvis.AI repo pins it.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - version is environment-dependent
    __version__ = version("cuvis-ai-dataloader")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
