"""Foreground-biased crop-window sampling for dataset-level cropping.

Given a per-pixel ``mask`` and an output ``size``, :func:`fg_crop_window` returns a ``(top, left)``
offset that — with probability ``fg_percent`` — centers the crop on a random pixel of a random
eligible foreground class (nnU-Net-style oversampling, so rare classes are hit as often as common
ones); otherwise it draws a uniform random offset. Frames with no eligible foreground fall back to
the uniform offset.

A foreground-centered window may extend past the frame border (the offset is deliberately *not*
clamped, so a foreground pixel near an edge stays centered); :func:`crop_with_pad` fills the
out-of-frame region with the selected ``pad_mode`` (``"constant"`` = 0, the default, or
``"reflect"``). The uniform offset is always fully in-bounds, so a background crop never pads.
``size`` may not exceed the frame's spatial dims, so the in-frame part of any window is non-empty.

The algorithm is ported from ``RandomForegroundBiasedCrop`` (cuvis-ai-augment PR #13). It lives here
so a DataModule can crop inside ``__getitem__`` — shipping small patches instead of whole frames —
without depending on the augment plugin (a higher layer) or its batched/torch-Generator transform
API. Pure numpy; reusable by any datamodule that exposes a ``[H, W]`` mask.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

PAD_MODES = ("constant", "reflect")


def _fg_center(
    mask: np.ndarray, fg_labels: Sequence[int] | None, rng: np.random.Generator
) -> tuple[int, int] | None:
    """Pick a random pixel of a random eligible foreground class in ``mask``.

    Eligible labels are ``> 0`` (default) or exactly ``fg_labels`` when given. A class is chosen
    uniformly first, then a pixel of that class uniformly. Returns ``(y, x)``, or ``None`` when the
    mask has no eligible foreground.
    """
    labels = np.unique(mask)
    if fg_labels is None:
        labels = labels[labels > 0]
    else:
        allowed = np.asarray(list(fg_labels), dtype=labels.dtype)
        labels = labels[np.isin(labels, allowed)]
    if labels.size == 0:
        return None
    cls = labels[int(rng.integers(labels.size))]
    ys, xs = np.nonzero(mask == cls)
    j = int(rng.integers(ys.size))
    return int(ys[j]), int(xs[j])


def fg_crop_window(
    mask: np.ndarray,
    size: tuple[int, int],
    *,
    fg_percent: float,
    fg_labels: Sequence[int] | None,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Return a ``(top, left)`` offset for a ``size=(h, w)`` crop over a ``[H, W]`` ``mask``.

    With probability ``fg_percent`` the window is centered on a foreground pixel (see
    :func:`_fg_center`); this offset is **not** clamped to the frame, so a foreground pixel near an
    edge stays centered and the window may extend past the border (fill it with
    :func:`crop_with_pad`). Otherwise a uniform random offset is drawn, always fully in-bounds so a
    background crop never needs padding. Raises ``ValueError`` if ``size`` exceeds the mask's
    spatial dimensions (the in-frame part of the window would otherwise risk being empty).
    """
    height, width = int(mask.shape[0]), int(mask.shape[1])
    out_h, out_w = int(size[0]), int(size[1])
    if out_h > height or out_w > width:
        raise ValueError(
            f"crop_size {(out_h, out_w)} exceeds frame spatial dims {(height, width)}."
        )
    if fg_percent > 0.0 and rng.random() < fg_percent:
        center = _fg_center(mask, fg_labels, rng)
        if center is not None:
            cy, cx = center
            return cy - out_h // 2, cx - out_w // 2
    max_top, max_left = height - out_h, width - out_w
    return int(rng.integers(max_top + 1)), int(rng.integers(max_left + 1))


def crop_with_pad(
    arr: np.ndarray, top: int, left: int, size: tuple[int, int], pad_mode: str
) -> np.ndarray:
    """Crop ``arr`` to ``size=(h, w)`` at ``(top, left)``, padding any out-of-frame part.

    ``top`` / ``left`` may be negative or place the window past ``arr``'s spatial extent (a
    foreground-centered window near a border); the in-frame overlap is sliced and the remainder is
    filled with ``pad_mode`` (``"constant"`` → 0, ``"reflect"`` → mirror). ``arr`` is ``[H, W, ...]``
    (only the first two axes are cropped). The result is always exactly ``(h, w, ...)`` and
    contiguous. Callers guarantee a non-empty in-frame overlap (see :func:`fg_crop_window`).
    """
    out_h, out_w = int(size[0]), int(size[1])
    height, width = int(arr.shape[0]), int(arr.shape[1])
    y0, y1 = max(top, 0), min(top + out_h, height)
    x0, x1 = max(left, 0), min(left + out_w, width)
    patch = arr[y0:y1, x0:x1]
    pad_top, pad_bottom = y0 - top, (top + out_h) - y1
    pad_left, pad_right = x0 - left, (left + out_w) - x1
    if not (pad_top or pad_bottom or pad_left or pad_right):
        return np.ascontiguousarray(patch)
    pad_width = [(pad_top, pad_bottom), (pad_left, pad_right)] + [(0, 0)] * (arr.ndim - 2)
    return np.pad(patch, pad_width, mode=pad_mode)
