"""Tests for TiffPairedDataModule + the internal TIFF reader / PNG labeler."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai_core.data.datamodule import BaseCuvisAIDataModule
from cuvis_ai_dataloader.data.datamodule_tiff_paired import TiffPairedDataModule
from cuvis_ai_dataloader.data.readers.tiff_reader import (
    TiffCubeReader,
    _parse_wavelengths_from_gdal,
)
from cuvis_ai_schemas.training.data import DataSplitConfig, Selector, SelectorKind


def test_data_module_name_and_subclass():
    assert TiffPairedDataModule.DATA_MODULE_NAME == "tiff_paired"
    assert issubclass(TiffPairedDataModule, BaseCuvisAIDataModule)


def test_parse_wavelengths_from_gdal():
    xml = '<GDALMetadata><Item name="wavelength">{400,410.5,420}</Item></GDALMetadata>'
    wl = _parse_wavelengths_from_gdal(xml)
    assert wl.tolist() == [400.0, 410.5, 420.0]


def test_parse_wavelengths_missing_raises():
    with pytest.raises(ValueError, match="wavelength"):
        _parse_wavelengths_from_gdal("<GDALMetadata></GDALMetadata>")


def test_validate_params_requires_images_dir(tmp_path):
    with pytest.raises(ValueError, match="images_dir"):
        TiffPairedDataModule.validate_params({})


def test_build_dataset_emits_expected_keys(tiff_dataset_dir):
    images_dir, labels_dir, wavelengths = tiff_dataset_dir
    dm = TiffPairedDataModule(
        images_dir=str(images_dir),
        labels_dir=str(labels_dir),
        wavelengths=",".join(str(w) for w in wavelengths),
        batch_size=1,
    )
    dm.setup(stage="predict")
    loader = dm.predict_dataloader()
    batches = list(loader)
    assert len(batches) == 2
    sample = dm._predict_ds[0]
    assert set(sample.keys()) >= {"cube", "wavelengths", "stem", "mesu_index", "label_rgb"}
    assert sample["cube"].shape == (8, 6, 4)  # (H, W, C) from SYX
    assert sample["cube"].dtype == np.float32
    assert sample["label_rgb"].shape == (8, 6, 3)
    assert sample["label_rgb"].dtype == np.uint8
    assert isinstance(sample["mesu_index"], int)
    # collated batch tensorizes
    assert isinstance(batches[0]["cube"], torch.Tensor)
    assert batches[0]["cube"].shape == (1, 8, 6, 4)


def test_stem_selectors(tiff_dataset_dir):
    images_dir, labels_dir, wavelengths = tiff_dataset_dir
    dm = TiffPairedDataModule(
        images_dir=str(images_dir),
        wavelengths=",".join(str(w) for w in wavelengths),
        splits=DataSplitConfig(predict=[Selector(kind=SelectorKind.STEMS, stems=["scrap_02"])]),
    )
    dm.setup(stage="predict")
    assert len(dm._predict_ds) == 1
    assert dm._predict_ds[0]["stem"] == "scrap_02"


def test_missing_png_is_unannotated_not_an_error(tiff_dataset_dir, tmp_path):
    # A TIFF with no paired PNG is a valid unannotated sample (AD-aware: normals carry no
    # label), not a crash. enumerate() leaves its annotation None and __getitem__ skips it.
    images_dir, _labels_dir, wavelengths = tiff_dataset_dir
    empty_labels = tmp_path / "empty_labels"
    empty_labels.mkdir()
    (empty_labels / "placeholder.png").write_bytes(b"")  # satisfy validate_params (dir has a PNG)
    dm = TiffPairedDataModule(
        images_dir=str(images_dir),
        labels_dir=str(empty_labels),
        wavelengths=",".join(str(w) for w in wavelengths),
    )
    dm.setup(stage="predict")
    sample = dm._predict_ds[0]  # no matching PNG -> no label key, no error
    assert "label_rgb" not in sample
    assert "cube" in sample


def test_label_map_variant(tiff_dataset_dir):
    images_dir, labels_dir, wavelengths = tiff_dataset_dir
    dm = TiffPairedDataModule(
        images_dir=str(images_dir),
        labels_dir=str(labels_dir),
        wavelengths=",".join(str(w) for w in wavelengths),
        label_mode="label_map",
        label_output_key="mask",
    )
    dm.setup(stage="predict")
    sample = dm._predict_ds[0]
    assert "mask" in sample
    assert sample["mask"].shape == (8, 6)


def test_tiff_reader_yxs_and_yx(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    # YXS (H, W, C)
    yxs = (np.random.default_rng(0).random((8, 6, 3)) * 255).astype(np.float32)
    p1 = tmp_path / "yxs.tif"
    tifffile.imwrite(p1, yxs, photometric="rgb")
    reader = TiffCubeReader(wavelengths_override=[400.0, 410.0, 420.0])
    out = reader.read(p1)
    assert out["cube"].shape == (8, 6, 3)
    # Wavelengths are emitted as int32 nm (parity with cu3s + the channel selectors).
    assert out["wavelengths"].dtype == np.int32
    assert out["wavelengths"].tolist() == [400, 410, 420]
    # YX (grayscale) -> (H, W, 1)
    yx = (np.random.default_rng(0).random((8, 6)) * 255).astype(np.float32)
    p2 = tmp_path / "yx.tif"
    tifffile.imwrite(p2, yx, photometric="minisblack")
    out2 = TiffCubeReader(wavelengths_override=[500.0]).read(p2)
    assert out2["cube"].shape == (8, 6, 1)
