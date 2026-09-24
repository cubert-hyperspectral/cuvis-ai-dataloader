"""Frames of a recording without a labels file: label-free, every one of them reads as normal.

A cu3s recording carries its COCO labels in a sibling ``<stem>.json`` (single-folder mode) or
in the ``annotation`` column of a ``universe.csv`` row (``cu3s_multi``). A recording without
one is *label-free*: the split designer places it in train, val and test like any other, and
every frame of it counts as normal. This module holds the two things both DataModules need
for that: the mask such a frame carries, and the warning that names such recordings when
they enter a val or test stage.

``label_free_mask`` is the same value ``CocoLabeler.load_for`` returns for an image id a
labels file does not annotate, so "no labels file" and "labels file without this image" hand
the metric nodes the same all-background target. The warning exists because an unlabelled
anomaly in val or test scores as normal: that is the documented contract, but worth seeing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
from loguru import logger

from cuvis_ai_core.data.datamodule import DataStage


def label_free_mask(cube: Any) -> np.ndarray:
    """All-zero int32 ``[H, W]`` mask for a frame of a recording without a labels file.

    ``cube`` is ``[H, W, C]``, a numpy array or, with ``cuda_cubes``, a torch tensor; only its
    first two dimensions are read, so the mask stays a host array either way.
    """
    return np.zeros((int(cube.shape[0]), int(cube.shape[1])), dtype=np.int32)


def warn_label_free_sources(stage: str, label_free: list[str], total: int) -> None:
    """Log one WARNING naming the recordings of ``stage`` that have no labels file.

    Silent when ``label_free`` is empty. ``label_free`` holds canonical source paths (stems
    repeat across folders); the message shows their file names. ``total`` is the number of
    distinct recordings the stage reads.
    """
    if not label_free:
        return
    names = ", ".join(Path(source).name for source in label_free)
    logger.warning(
        f"{stage}: {len(label_free)} of {total} recordings have no labels file ({names}); "
        "their frames are label-free and score as normal"
    )


class _KnowsItsSources(Protocol):
    """What ``warn_label_free_stages`` reads off a stage dataset."""

    @property
    def sources(self) -> list[str]:
        """The distinct recordings the dataset reads."""
        ...

    @property
    def label_free_sources(self) -> list[str]:
        """The recordings among them without a labels file."""
        ...


def warn_label_free_stages(
    stage: str | None,
    *,
    val: _KnowsItsSources | None,
    test: _KnowsItsSources | None,
) -> None:
    """Warn for the val/test datasets a ``setup(stage)`` call built, mirroring core's stage map.

    ``fit``, ``validate`` and ``None`` build val; ``test`` and ``None`` build test; train and
    predict never warn. There is no memory across calls: a repeated ``setup`` warns again for
    what it rebuilt, and a later ``setup("test")`` says nothing about val.
    """
    if stage in (DataStage.FIT, DataStage.VALIDATE, None) and val is not None:
        warn_label_free_sources("val", val.label_free_sources, len(val.sources))
    if stage in (DataStage.TEST, None) and test is not None:
        warn_label_free_sources("test", test.label_free_sources, len(test.sources))
