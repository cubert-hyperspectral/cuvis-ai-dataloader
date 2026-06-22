"""Shared test fixtures for the cuvis-ai-dataloader plugin."""

from __future__ import annotations

import sys
import types
from unittest.mock import Mock, patch

import numpy as np
import pytest


@pytest.fixture
def mock_cuvis_sdk():
    """Patch the ``cuvis`` SDK + COCO loading so cu3s tests run without real data.

    Replaces ``sys.modules['cuvis']`` with a fake module (so ``require_cuvis``
    returns it) and patches the plugin's ``COCOData.from_path``. Yields the mocks.
    """
    h = w = 64
    channels = 61
    rng = np.random.default_rng(42)
    cube = rng.random((h, w, channels)).astype(np.float32)
    wavelengths = np.linspace(430.0, 910.0, channels).astype(np.float32)

    mock_measurement = Mock()
    mock_measurement.cube = Mock()
    mock_measurement.cube.array = cube
    mock_measurement.cube.channels = channels
    mock_measurement.cube.wavelength = wavelengths
    mock_measurement.data = {"cube": True}

    mock_session = Mock()
    mock_session.get_measurement = Mock(return_value=mock_measurement)
    mock_session.__len__ = Mock(return_value=7)
    mock_session.fps = 30.0

    mock_pc = Mock()
    mock_pc.apply = Mock(return_value=mock_measurement)
    mock_pc.processing_mode = Mock()

    fake_cuvis = types.ModuleType("cuvis")
    fake_cuvis.SessionFile = Mock(return_value=mock_session)
    fake_cuvis.ProcessingContext = Mock(return_value=mock_pc)
    pm = Mock()
    pm.Raw, pm.Reflectance, pm.SpectralRadiance = "Raw", "Reflectance", "SpectralRadiance"
    fake_cuvis.ProcessingMode = pm
    rt = Mock()
    rt.White, rt.Dark = "White", "Dark"
    fake_cuvis.ReferenceType = rt

    white_ref, dark_ref = Mock(), Mock()

    def _get_reference(idx, ref_type):
        if ref_type == rt.White:
            return white_ref
        if ref_type == rt.Dark:
            return dark_ref
        return None

    mock_session.get_reference = Mock(side_effect=_get_reference)

    mock_coco = Mock()
    mock_coco.category_id_to_name = {0: "background", 1: "anomaly"}
    mock_coco.image_ids = [0, 1, 2, 3, 4, 5, 6]
    mock_coco.annotations = Mock()
    mock_coco.annotations.where = Mock(return_value=[])
    mock_coco.images = None  # force canvas-size fallback to the cube's (H, W)

    with (
        patch.dict(sys.modules, {"cuvis": fake_cuvis}),
        patch(
            "cuvis_ai_dataloader.data.labelers.coco_labeler.COCOData.from_path",
            return_value=mock_coco,
        ),
    ):
        yield {
            "session": mock_session,
            "processing_context": mock_pc,
            "measurement": mock_measurement,
            "coco": mock_coco,
            "channels": channels,
            "hw": (h, w),
        }


@pytest.fixture
def tiff_dataset_dir(tmp_path):
    """Write a tiny synthetic TIFF dataset (SYX) + paired PNG; return its dirs.

    Returns ``(images_dir, labels_dir, wavelengths)``. Skips if tifffile/pillow
    are unavailable.
    """
    tifffile = pytest.importorskip("tifffile")
    from PIL import Image

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    h, w, c = 8, 6, 4
    wavelengths = [400.0, 410.0, 420.0, 430.0]
    for stem in ("scrap_01", "scrap_02"):
        cube_chw = (np.random.default_rng(0).random((c, h, w)) * 255).astype(np.float32)
        tifffile.imwrite(
            images_dir / f"{stem}.tif",
            cube_chw,
            photometric="minisblack",
            planarconfig="separate",
        )
        png = (np.random.default_rng(1).integers(0, 255, (h, w, 3))).astype(np.uint8)
        Image.fromarray(png, mode="RGB").save(labels_dir / f"{stem}.png")
    return images_dir, labels_dir, wavelengths
