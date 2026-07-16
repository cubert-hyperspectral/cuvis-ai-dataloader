"""Foreground-biased crop-window sampling for dataset-level cropping.

Given a per-pixel ``mask`` and an output ``size``, :func:`fg_crop_window` returns a ``(top, left)``
offset that — with probability ``fg_percent`` — centers the crop on a random pixel of a random
eligible foreground class (nnU-Net-style oversampling, so rare classes are hit as often as common
ones); otherwise it draws a uniform random offset. The window is clamped inward so it always stays
fully in-bounds. Frames with no eligible foreground fall back to the uniform offset.

The algorithm is ported from ``RandomForegroundBiasedCrop`` (cuvis-ai-augment PR #13). It lives here
so a DataModule can crop inside ``__getitem__`` — shipping small patches instead of whole frames —
without depending on the augment plugin (a higher layer) or its batched/torch-Generator transform
API. Pure numpy; reusable by any datamodule that exposes a ``[H, W]`` mask.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


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
    :func:`_fg_center`), else a uniform random offset is drawn. The offset is clamped so the crop
    stays fully in-bounds, so the resulting crop is always exactly ``size``. Raises ``ValueError``
    if ``size`` exceeds the mask's spatial dimensions.
    """
    height, width = int(mask.shape[0]), int(mask.shape[1])
    out_h, out_w = int(size[0]), int(size[1])
    if out_h > height or out_w > width:
        raise ValueError(
            f"crop_size {(out_h, out_w)} exceeds frame spatial dims {(height, width)}."
        )
    max_top, max_left = height - out_h, width - out_w
    if fg_percent > 0.0 and rng.random() < fg_percent:
        center = _fg_center(mask, fg_labels, rng)
        if center is not None:
            cy, cx = center
            top = min(max(cy - out_h // 2, 0), max_top)
            left = min(max(cx - out_w // 2, 0), max_left)
            return top, left
    return int(rng.integers(max_top + 1)), int(rng.integers(max_left + 1))
