"""Internal paired-PNG labeler. Not a plugin contract.

Pairs a TIFF cube with a stem-keyed PNG label (``<labels_dir>/<stem>.png``),
nearest-neighbor resized to the cube's (H, W) when they differ. Default output
is ``label_rgb`` ``(H, W, 3) uint8`` (what the HSIMetalScrap viz nodes consume);
``label_mode='label_map'`` + ``label_output_key='mask'`` gives the cu3s-shape
``(H, W) uint8`` variant. ``pillow`` is a base dep, so no lazy guard is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class PairedPngLabeler:
    """Reads ``<labels_dir>/<stem>.png`` and attaches it under ``label_output_key``."""

    def __init__(
        self,
        labels_dir: str | Path,
        *,
        label_output_key: str = "label_rgb",
        label_mode: str = "rgb",
    ) -> None:
        self.labels_dir = Path(labels_dir)
        self.label_output_key = label_output_key
        self.label_mode = label_mode

    def has_label(self, stem: str) -> bool:
        """True if a paired ``<stem>.png`` exists (i.e. the sample is annotated)."""
        return (self.labels_dir / f"{stem}.png").exists()

    def categories_for(self, stem: str) -> list[int]:
        """Category ids for ``stem`` from the PNG mask (empty -> unannotated / normal).

        ``label_map`` mode returns the distinct non-zero label values; ``rgb`` mode has no
        per-value labels, so any non-zero region marks the single category ``1`` while a
        present-but-empty (all-zero) mask is a normal sample (empty category list).
        """
        png_path = self.labels_dir / f"{stem}.png"
        if not png_path.exists():
            return []
        from PIL import Image

        arr = np.asarray(Image.open(png_path).convert("L"), dtype=np.uint8)
        if self.label_mode == "label_map":
            return sorted({int(v) for v in np.unique(arr) if int(v) != 0})
        return [1] if arr.any() else []

    def load_for(self, stem: str, cube_hw: tuple[int, int]) -> dict:
        from PIL import Image

        png_path = self.labels_dir / f"{stem}.png"
        if not png_path.exists():
            raise FileNotFoundError(
                f"No label PNG for stem {stem!r}: expected {png_path} in {self.labels_dir}"
            )
        target_h, target_w = int(cube_hw[0]), int(cube_hw[1])
        img = Image.open(png_path)
        if self.label_mode == "label_map":
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        if img.size != (target_w, target_h):
            img = img.resize((target_w, target_h), Image.NEAREST)
        arr = np.asarray(img, dtype=np.uint8)
        return {self.label_output_key: arr}
