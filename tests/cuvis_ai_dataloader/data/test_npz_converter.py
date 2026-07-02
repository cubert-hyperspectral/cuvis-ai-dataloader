"""Tests for the cu3s -> per-frame NPZ converter.

The pure helpers (derive_masks / apply_crop / write_index_csv) are tested directly; the
orchestration (convert_cu3s_file / convert_cu3s) is tested with a fake cu3s reader + COCO
labeler patched onto the lazily-imported source modules, so no Cuvis SDK / real cu3s is needed.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from cuvis_ai_dataloader.data.npz_converter import (
    apply_crop,
    convert_cu3s,
    convert_cu3s_file,
    derive_masks,
    write_index_csv,
)


# --------------------------------------------------------------------------- pure helpers
def test_derive_masks_binary_and_class():
    cat = np.zeros((5, 6), dtype=np.int32)
    cat[1:3, 2:4] = 3
    mask, class_mask = derive_masks(cat)
    assert mask.dtype == np.int32 and class_mask.dtype == np.uint8
    assert mask.shape == (5, 6) and class_mask.shape == (5, 6)
    assert int(class_mask.max()) == 3  # category id preserved
    assert np.array_equal(mask > 0, class_mask > 0)  # binary agrees with category>0
    assert int(mask.sum()) == 4  # 2x2 region


def test_apply_crop_margins():
    arr = np.arange(6 * 8 * 3, dtype=np.float32).reshape(6, 8, 3)
    out = apply_crop(arr, (1, 1, 2, 2))  # top,bottom,left,right
    assert out.shape == (4, 4, 3)
    assert np.array_equal(out, arr[1:5, 2:6])
    assert apply_crop(arr, None) is arr  # None -> unchanged


def test_apply_crop_rejects_too_large_and_negative():
    arr = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="too large"):
        apply_crop(arr, (2, 2, 0, 0))  # 2+2 >= 4
    with pytest.raises(ValueError, match="non-negative"):
        apply_crop(arr, (-1, 0, 0, 0))


def test_write_index_csv_has_no_split_column(tmp_path):
    recs = [
        {"npz_path": "a/x_000000.npz", "source_cu3s": "a.cu3s", "image_id": 0, "frame_index": 0},
        {"npz_path": "a/x_000001.npz", "source_cu3s": "a.cu3s", "image_id": 1, "frame_index": 1},
    ]
    out = tmp_path / "index.csv"
    write_index_csv(recs, out)
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == ["npz_path", "source_cu3s", "image_id", "frame_index"]
    assert "split" not in rows[0]
    assert rows[1]["image_id"] == "1"


# --------------------------------------------------------------------------- fakes
class _FakeReader:
    """Stand-in for Cu3sCubeReader: 2 frames, no SDK."""

    def __init__(self, path, *, processing_mode="Reflectance"):
        self.path = path
        self.processing_mode = processing_mode
        self.total_measurements = 2

    def read(self, i):
        cube = np.full((6, 8, 3), float(i + 1), dtype=np.float32)  # [H, W, C]
        return {"cube": cube, "mesu_index": int(i), "wavelengths": np.array([450, 550, 650])}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeLabeler:
    """Stand-in for CocoLabeler: paints a category-3 box at full cube size."""

    def __init__(self, json_path):
        self.json_path = json_path

    def load_for(self, image_id, item):
        cube = item["cube"]
        h, w = cube.shape[0], cube.shape[1]
        cat = np.zeros((h, w), dtype=np.int32)
        cat[1:3, 2:5] = 3
        return {"mask": cat}


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr("cuvis_ai_dataloader.data.readers.cu3s_reader.Cu3sCubeReader", _FakeReader)
    monkeypatch.setattr("cuvis_ai_dataloader.data.labelers.coco_labeler.CocoLabeler", _FakeLabeler)


# --------------------------------------------------------------------------- orchestration
def test_convert_cu3s_file_with_annotations(patched, tmp_path):
    recs = convert_cu3s_file(
        tmp_path / "Auto_000.cu3s", tmp_path / "out", annotation_json=tmp_path / "a.json"
    )
    assert len(recs) == 2
    assert [r["image_id"] for r in recs] == [0, 1]
    assert all("split" not in r for r in recs)  # NO split assigned
    with np.load(recs[0]["npz_path"]) as z:
        assert z["cube"].shape == (6, 8, 3) and z["cube"].dtype == np.float32
        assert z["wavelengths"].shape == (3,)
        assert z["mask"].shape == (6, 8) and z["mask"].dtype == np.int32
        assert z["class_mask"].shape == (6, 8) and z["class_mask"].dtype == np.uint8
        assert int(z["class_mask"].max()) == 3  # category preserved (loads via npz_multi)
        assert str(z["source_cu3s"]).endswith("Auto_000.cu3s")


class _ImageIdLabeler:
    """Fake labeler that paints category == image_id, to prove load_for keys on image_id."""

    def __init__(self, json_path):
        self.json_path = json_path

    def load_for(self, image_id, item):
        cube = item["cube"]
        h, w = cube.shape[0], cube.shape[1]
        cat = np.zeros((h, w), dtype=np.int32)
        cat[1:3, 2:5] = int(image_id)
        return {"mask": cat}


def test_convert_cu3s_file_image_ids_decouple_read_from_label(monkeypatch, tmp_path):
    # Merged-cu3s case: read frame_indices, but label by a DIFFERENT image_id per frame.
    monkeypatch.setattr("cuvis_ai_dataloader.data.readers.cu3s_reader.Cu3sCubeReader", _FakeReader)
    monkeypatch.setattr("cuvis_ai_dataloader.data.labelers.coco_labeler.CocoLabeler", _ImageIdLabeler)
    recs = convert_cu3s_file(
        tmp_path / "merged.cu3s",
        tmp_path / "out",
        annotation_json=tmp_path / "a.json",
        frame_indices=[0, 1],
        image_ids=[10, 20],
    )
    assert [r["image_id"] for r in recs] == [10, 20]     # label id comes from image_ids
    assert [r["frame_index"] for r in recs] == [0, 1]    # read index comes from frame_indices
    with np.load(recs[0]["npz_path"]) as z:
        assert float(z["cube"][0, 0, 0]) == 1.0          # cube read from FRAME 0 (_FakeReader: i+1)
        assert int(z["class_mask"].max()) == 10          # mask looked up by IMAGE_ID 10
    with np.load(recs[1]["npz_path"]) as z:
        assert float(z["cube"][0, 0, 0]) == 2.0          # cube read from FRAME 1
        assert int(z["class_mask"].max()) == 20          # mask looked up by IMAGE_ID 20
    # filenames use the read (frame) index for uniqueness within the cu3s
    assert recs[0]["npz_path"].endswith("merged_000000.npz")


def test_convert_cu3s_file_image_ids_length_mismatch_raises(patched, tmp_path):
    with pytest.raises(ValueError, match="parallel to frame_indices"):
        convert_cu3s_file(
            tmp_path / "m.cu3s",
            tmp_path / "out",
            annotation_json=tmp_path / "a.json",
            frame_indices=[0, 1],
            image_ids=[10],
        )


def test_convert_cu3s_file_without_annotations_writes_no_mask(patched, tmp_path):
    recs = convert_cu3s_file(tmp_path / "Auto_001.cu3s", tmp_path / "out", annotation_json=None)
    with np.load(recs[0]["npz_path"]) as z:
        assert "cube" in z.files and "wavelengths" in z.files
        assert "mask" not in z.files and "class_mask" not in z.files


def test_convert_cu3s_file_applies_crop_to_cube_and_masks(patched, tmp_path):
    recs = convert_cu3s_file(
        tmp_path / "c.cu3s",
        tmp_path / "out",
        annotation_json=tmp_path / "a.json",
        crop=(1, 1, 2, 2),
    )
    with np.load(recs[0]["npz_path"]) as z:
        assert z["cube"].shape == (4, 4, 3)  # (6-2, 8-4)
        assert z["mask"].shape == (4, 4) and z["class_mask"].shape == (4, 4)


def test_cli_parses_and_dispatches(monkeypatch, tmp_path):
    import sys

    from cuvis_ai_dataloader.scripts.convert_cu3s_to_npz import cu3s_to_npz_cli

    captured: dict = {}

    def fake_convert_cu3s(paths, out_dir, **kw):
        captured["paths"] = [str(p) for p in paths]
        captured["out_dir"] = str(out_dir)
        captured["kw"] = kw
        return [{"npz_path": "x.npz", "source_cu3s": "a.cu3s", "image_id": 0}]

    monkeypatch.setattr("cuvis_ai_dataloader.data.npz_converter.convert_cu3s", fake_convert_cu3s)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cu3s-to-npz",
            "--cu3s",
            "a.cu3s",
            "b.cu3s",
            "--out-dir",
            str(tmp_path),
            "--crop",
            "300,300,300,300",
            "--annotations",
            "none",
            "--limit",
            "2",
            "--processing-mode",
            "none",
        ],
    )
    cu3s_to_npz_cli()
    assert captured["paths"] == ["a.cu3s", "b.cu3s"]
    assert captured["kw"]["crop"] == (300, 300, 300, 300)
    assert captured["kw"]["annotations"] is None  # "none" -> None
    assert captured["kw"]["processing_mode"] is None  # "none" -> None
    assert captured["kw"]["frame_limit"] == 2


def test_cli_crop_parser_rejects_bad():
    import argparse

    from cuvis_ai_dataloader.scripts.convert_cu3s_to_npz import _parse_crop

    assert _parse_crop("1,2,3,4") == (1, 2, 3, 4)
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_crop("1,2,3")  # not 4 values


def test_convert_cu3s_multi_writes_index(patched, tmp_path):
    (tmp_path / "s1.cu3s").touch()
    (tmp_path / "s2.cu3s").touch()
    idx = tmp_path / "index.csv"
    recs = convert_cu3s(
        [tmp_path / "s1.cu3s", tmp_path / "s2.cu3s"],
        tmp_path / "out",
        annotations=tmp_path / "shared.json",
        index_csv=idx,
    )
    assert len(recs) == 4  # 2 files x 2 frames
    with idx.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 and "split" not in rows[0]
    assert {Path(r["source_cu3s"]).name for r in rows} == {"s1.cu3s", "s2.cu3s"}
