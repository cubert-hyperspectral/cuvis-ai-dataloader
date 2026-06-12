"""Lazy heavy-dependency imports and string param parsers.

Module-top imports across the plugin are limited to stdlib + numpy + torch +
pytorch_lightning + the base class. Heavy deps (cuvis, tifffile, pycocotools,
scikit-image) are imported only inside the methods that use them, via the
``require_*`` helpers here, so a manifest with only some extras installed still
imports cleanly. The first use of a module whose extra is missing raises a clear
``ImportError`` naming the install command.
"""

from __future__ import annotations


def require_cuvis():
    """Return the ``cuvis`` SDK module, or raise a clear install hint."""
    try:
        import cuvis

        return cuvis
    except ImportError as e:  # pragma: no cover - exercised via the lazy-extras smoke
        raise ImportError(
            "The 'cuvis' SDK is required for the cu3s data modules. "
            "Install with: uv pip install 'cuvis-ai-dataloader[cu3s]'"
        ) from e


def require_tifffile():
    """Return the ``tifffile`` module, or raise a clear install hint."""
    try:
        import tifffile

        return tifffile
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "tifffile is required for --data-module tiff_paired. "
            "Install with: uv pip install 'cuvis-ai-dataloader[tiff]'"
        ) from e


def require_pycocotools():
    """Return ``pycocotools.coco.COCO``, or raise a clear install hint."""
    try:
        from pycocotools.coco import COCO

        return COCO
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pycocotools is required for COCO labels. "
            "Install with: uv pip install 'cuvis-ai-dataloader[coco]'"
        ) from e


def require_skimage_polygon2mask():
    """Return ``skimage.draw.polygon2mask``, or raise a clear install hint."""
    try:
        from skimage.draw import polygon2mask

        return polygon2mask
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "scikit-image is required for COCO polygon labels. "
            "Install with: uv pip install 'cuvis-ai-dataloader[coco]'"
        ) from e


_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}


def parse_bool(s, *, key: str) -> bool:
    """Coerce a ``--data-arg`` string (or bool) to bool."""
    if isinstance(s, bool):
        return s
    token = str(s).lower()
    if token in _TRUE:
        return True
    if token in _FALSE:
        return False
    raise ValueError(f"--data-arg {key}={s!r}: expected one of {sorted(_TRUE | _FALSE)}")


def parse_float_list(s, *, key: str) -> list[float]:
    """Comma-floats (or an existing list) -> list[float]."""
    if isinstance(s, (list, tuple)):
        return [float(x) for x in s]
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_int_list(s, *, key: str) -> list[int]:
    """Comma list of ints and inclusive ``start-stop[:step]`` ranges -> list[int].

    Accepts an existing list/tuple, or a string like ``"0,2,4"``, ``"0-100"``, or
    ``"0-10:2, 20"``. Range tokens expand via the shared core helper, so
    ``measurement_indices`` accepts the same range syntax as split id-lists.
    """
    from cuvis_ai_core.utils.general import expand_range_selectors

    tokens = (
        list(s)
        if isinstance(s, (list, tuple))
        else [t.strip() for t in str(s).split(",") if t.strip()]
    )
    return [int(x) for x in expand_range_selectors(tokens)]


def parse_str_list(s, *, key: str) -> list[str]:
    """Comma-strings (or an existing list) -> list[str]."""
    if isinstance(s, (list, tuple)):
        return [str(x).strip() for x in s]
    return [x.strip() for x in str(s).split(",") if x.strip()]
