"""Crop margins, shared by the npz converter and the cu3s_multi reader."""

from __future__ import annotations

from typing import Any

import numpy as np

# (top, bottom, left, right) pixel margins removed from the first two axes.
CropMargins = tuple[int, int, int, int]


def parse_crop(crop: Any) -> CropMargins | None:
    """Validate ``crop`` as four non-negative ``(top, bottom, left, right)`` margins.

    Accepts any sequence of four ints, or a ``"top,bottom,left,right"`` string (how a CLI
    ``--data-arg`` carries it). ``None`` stays ``None``: the frame is kept whole.
    """
    if crop is None:
        return None
    parts = crop.split(",") if isinstance(crop, str) else crop
    margins = tuple(int(v) for v in parts)
    if len(margins) != 4 or min(margins) < 0:
        raise ValueError(
            f"crop must be four non-negative pixel margins (top, bottom, left, right), got {crop!r}"
        )
    return margins  # type: ignore[return-value]


def apply_crop(arr: np.ndarray, crop: CropMargins | None) -> np.ndarray:
    """Remove ``(top, bottom, left, right)`` margins from the first two axes. ``None`` -> as-is.

    Applied identically to the cube and the masks so they stay pixel-aligned. Works on a torch
    tensor as well (a device-resident cube), since it only slices.
    """
    if crop is None:
        return arr
    top, bottom, left, right = (int(v) for v in crop)
    if min(top, bottom, left, right) < 0:
        raise ValueError(f"crop margins must be non-negative, got {crop}")
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if top + bottom >= h or left + right >= w:
        raise ValueError(f"crop margins {crop} too large for spatial shape {(h, w)}")
    return arr[top : h - bottom, left : w - right]
