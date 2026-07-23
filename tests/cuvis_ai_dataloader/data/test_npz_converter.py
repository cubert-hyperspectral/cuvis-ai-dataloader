"""Tests for the cu3s -> per-frame NPZ converter.

The pure helpers (derive_masks / apply_crop / write_universe_csv) are tested directly; the
orchestration (convert_cu3s_file / convert_cu3s) is tested with a fake cu3s reader + COCO
labeler patched onto the lazily-imported source modules, so no Cuvis SDK / real cu3s is needed.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from cuvis_ai_dataloader.data.datamodule_npz_multi import MultiNpzDataModule
from cuvis_ai_dataloader.data.npz_converter import (
    apply_crop,
    convert_cu3s,
    convert_cu3s_file,
    convert_split_manifest,
    convert_universe,
    derive_masks,
    write_universe_csv,
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


def test_write_universe_csv_has_no_split_column(tmp_path):
    recs = [
        {"path": "a/x_000000.npz", "source": "a.cu3s", "index": 0},
        {"path": "a/x_000001.npz", "source": "a.cu3s", "index": 1},
    ]
    out = tmp_path / "universe.csv"
    write_universe_csv(recs, out)
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == ["source", "index", "materialized_path"]
    assert "split" not in rows[0]
    assert rows[1]["index"] == "1"
    assert rows[0]["materialized_path"] == "a/x_000000.npz"


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
    assert [r["index"] for r in recs] == [0, 1]
    assert all("split" not in r for r in recs)  # NO split assigned
    with np.load(recs[0]["path"]) as z:
        assert z["cube"].shape == (6, 8, 3) and z["cube"].dtype == np.float32
        assert z["wavelengths"].shape == (3,)
        assert z["mask"].shape == (6, 8) and z["mask"].dtype == np.int32
        assert z["class_mask"].shape == (6, 8) and z["class_mask"].dtype == np.uint8
        assert int(z["class_mask"].max()) == 3  # category preserved (loads via npz_multi)
        assert str(z["source"]).endswith("Auto_000.cu3s")


def test_convert_cu3s_file_names_prefixed_by_parent_folder(patched, tmp_path):
    # Same session stem in two day folders must not collide in a shared out_dir.
    out = tmp_path / "out"
    r2 = convert_cu3s_file(tmp_path / "day2" / "Sess.cu3s", out, annotation_json=None)
    r3 = convert_cu3s_file(tmp_path / "day3" / "Sess.cu3s", out, annotation_json=None)
    n2, n3 = Path(r2[0]["path"]).name, Path(r3[0]["path"]).name
    assert n2.startswith("day2_Sess_") and n3.startswith("day3_Sess_")
    assert n2 != n3  # no clobber


def test_convert_cu3s_file_rejects_out_of_range_frame_indices(patched, tmp_path):
    # _FakeReader exposes 2 measurements; index 5 must raise here, not deep in the SDK read.
    with pytest.raises(ValueError, match="out of range"):
        convert_cu3s_file(tmp_path / "s.cu3s", tmp_path / "out", frame_indices=[0, 5])


def test_convert_cu3s_file_frame_limit_clamps_to_length(patched, tmp_path):
    recs = convert_cu3s_file(tmp_path / "s.cu3s", tmp_path / "out", frame_limit=100)
    assert len(recs) == 2  # clamped to the 2 available measurements


def test_derive_masks_rejects_category_over_255():
    cat = np.zeros((4, 4), dtype=np.int32)
    cat[0, 0] = 300
    with pytest.raises(ValueError, match="uint8"):
        derive_masks(cat)


def test_convert_cu3s_rejects_colliding_inputs(tmp_path):
    # Two inputs sharing (parent, stem) would overwrite each other in the flat out_dir.
    with pytest.raises(ValueError, match="collide"):
        convert_cu3s(
            [tmp_path / "a" / "day1" / "S.cu3s", tmp_path / "b" / "day1" / "S.cu3s"],
            tmp_path / "out",
            annotations=None,
        )


def test_convert_cu3s_file_without_annotations_writes_no_mask(patched, tmp_path):
    recs = convert_cu3s_file(tmp_path / "Auto_001.cu3s", tmp_path / "out", annotation_json=None)
    with np.load(recs[0]["path"]) as z:
        assert "cube" in z.files and "wavelengths" in z.files
        assert "mask" not in z.files and "class_mask" not in z.files


def test_convert_cu3s_file_applies_crop_to_cube_and_masks(patched, tmp_path):
    recs = convert_cu3s_file(
        tmp_path / "c.cu3s",
        tmp_path / "out",
        annotation_json=tmp_path / "a.json",
        crop=(1, 1, 2, 2),
    )
    with np.load(recs[0]["path"]) as z:
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
        return [{"path": "x.npz", "source": "a.cu3s", "index": 0}]

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


def test_convert_cu3s_multi_writes_universe(patched, tmp_path):
    (tmp_path / "s1.cu3s").touch()
    (tmp_path / "s2.cu3s").touch()
    uni = tmp_path / "universe.csv"
    recs = convert_cu3s(
        [tmp_path / "s1.cu3s", tmp_path / "s2.cu3s"],
        tmp_path / "out",
        annotations=tmp_path / "shared.json",
        universe_csv=uni,
    )
    assert len(recs) == 4  # 2 files x 2 frames
    with uni.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 and "split" not in rows[0]
    assert {Path(r["source"]).name for r in rows} == {"s1.cu3s", "s2.cu3s"}


# --------------------------------------------------------------------------- split manifest
def _write_manifest(path: Path, rows: list[dict[str, str]]) -> Path:
    fields = ["cu3s_path", "json_path", "local_image_id", "split"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_convert_split_manifest_end_to_end(patched, tmp_path):
    root = tmp_path / "raw"
    (root / "day2").mkdir(parents=True)
    (root / "day3").mkdir(parents=True)
    (root / "day2" / "A.cu3s").touch()
    (root / "day2" / "A.json").touch()
    (root / "day3" / "B.cu3s").touch()
    manifest = _write_manifest(
        tmp_path / "splits_manifest.csv",
        [
            {
                "cu3s_path": "day2/A.cu3s",
                "json_path": "day2/A.json",
                "local_image_id": "0",
                "split": "train",
            },
            {
                "cu3s_path": "day2/A.cu3s",
                "json_path": "day2/A.json",
                "local_image_id": "1",
                "split": "val",
            },
            {"cu3s_path": "day3/B.cu3s", "json_path": "", "local_image_id": "0", "split": "test"},
            # not in the default splits -> ignored
            {
                "cu3s_path": "day3/B.cu3s",
                "json_path": "",
                "local_image_id": "1",
                "split": "adaclip_train",
            },
        ],
    )

    out = tmp_path / "npz"
    result = convert_split_manifest(manifest, root, out)

    # Two artifacts: the universe + selector splits.json.
    assert result.universe_csv == out / "universe.csv"
    assert result.splits_json == out / "splits.json"

    with result.universe_csv.open() as f:
        idx = list(csv.DictReader(f))
    assert list(idx[0].keys()) == ["source", "index", "materialized_path"]
    assert {r["source"] for r in idx} == {"day2/A.cu3s", "day3/B.cu3s"}  # posix identity
    # materialized_path is stored relative to the universe.csv dir (cwd-independent).
    assert all(not Path(r["materialized_path"]).is_absolute() for r in idx)
    assert all((result.universe_csv.parent / r["materialized_path"]).is_file() for r in idx)

    # splits.json loads as a DataSplitConfig and resolves against the npz_multi universe.
    from cuvis_ai_core.data.splits_io import load_splits

    cfg = load_splits(result.splits_json)
    assert [s.source for s in cfg.train] == ["day2/A.cu3s"] and cfg.train[0].ids == [0]
    assert [s.source for s in cfg.val] == ["day2/A.cu3s"] and cfg.val[0].ids == [1]
    assert [s.source for s in cfg.test] == ["day3/B.cu3s"]
    # predict mirrors test (empty predict would resolve to the whole universe).
    assert [(s.source, s.ids) for s in cfg.predict] == [(s.source, s.ids) for s in cfg.test]

    dm = MultiNpzDataModule(splits=cfg, universe_csv=str(result.universe_csv))
    dm.setup(stage="fit")
    dm.setup(stage="test")
    assert len(dm.train_ds) == 1 and len(dm.val_ds) == 1 and len(dm.test_ds) == 1
    # Annotated session bakes masks; unannotated session stays mask-free.
    with np.load(dm.train_ds.rows[0]["materialized_path"]) as z:
        assert "mask" in z.files and int(z["class_mask"].max()) == 3
    with np.load(dm.test_ds.rows[0]["materialized_path"]) as z:
        assert "mask" not in z.files


def test_convert_split_manifest_splits_path_resolves_via_core(patched, tmp_path):
    """E2E: converter splits.json + a splits_path pointer resolve through core into subsets.

    Exercises the whole contract in one line of construction: the converter's emitted
    splits.json (Phase 3), core's splits_path load/merge in the base datamodule (Phase 1),
    and npz_multi's selector path (Phase 2).
    """
    from cuvis_ai_schemas.training.data import DataSplitConfig

    (tmp_path / "day2").mkdir()
    (tmp_path / "day2" / "A.cu3s").touch()
    manifest = _write_manifest(
        tmp_path / "m.csv",
        [
            {"cu3s_path": "day2/A.cu3s", "json_path": "", "local_image_id": "0", "split": "train"},
            {"cu3s_path": "day2/A.cu3s", "json_path": "", "local_image_id": "1", "split": "test"},
        ],
    )
    result = convert_split_manifest(manifest, tmp_path, tmp_path / "npz")

    # Pass only a splits_path pointer; core must load + resolve it (not train on nothing).
    dm = MultiNpzDataModule(
        splits=DataSplitConfig(splits_path=str(result.splits_json)),
        universe_csv=str(result.universe_csv),
    )
    dm.setup(stage="fit")
    dm.setup(stage="test")
    assert len(dm.train_ds) == 1
    assert len(dm.test_ds) == 1


def test_convert_split_manifest_limit_per_split(patched, tmp_path):
    (tmp_path / "day2").mkdir()
    (tmp_path / "day2" / "A.cu3s").touch()
    manifest = _write_manifest(
        tmp_path / "m.csv",
        [
            {"cu3s_path": "day2/A.cu3s", "json_path": "", "local_image_id": "0", "split": "train"},
            {"cu3s_path": "day2/A.cu3s", "json_path": "", "local_image_id": "1", "split": "train"},
        ],
    )

    result = convert_split_manifest(manifest, tmp_path, tmp_path / "npz", limit=1)

    with result.universe_csv.open() as f:
        assert len(list(csv.DictReader(f))) == 1


def test_convert_split_manifest_resume_skips_existing(patched, tmp_path, monkeypatch):
    (tmp_path / "day2").mkdir()
    (tmp_path / "day2" / "A.cu3s").touch()
    manifest = _write_manifest(
        tmp_path / "m.csv",
        [
            {"cu3s_path": "day2/A.cu3s", "json_path": "", "local_image_id": "0", "split": "train"},
            {"cu3s_path": "day2/A.cu3s", "json_path": "", "local_image_id": "1", "split": "test"},
        ],
    )
    out = tmp_path / "npz"
    convert_split_manifest(manifest, tmp_path, out)  # first pass writes both npz

    reads: list[int] = []
    orig_read = _FakeReader.read

    def _counting_read(self, i):
        reads.append(i)
        return orig_read(self, i)

    monkeypatch.setattr(_FakeReader, "read", _counting_read)
    result = convert_split_manifest(manifest, tmp_path, out)  # second pass: valid npz reused
    assert reads == []  # nothing reconverted
    assert result.universe_csv.is_file() and result.splits_json.is_file()


def test_convert_split_manifest_rejects_bad_manifest_and_missing_files(patched, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("split,foo\ntrain,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks column"):
        convert_split_manifest(bad, tmp_path, tmp_path / "npz")

    manifest = _write_manifest(
        tmp_path / "m.csv",
        [{"cu3s_path": "nope/C.cu3s", "json_path": "", "local_image_id": "0", "split": "train"}],
    )
    with pytest.raises(FileNotFoundError, match="missing cu3s"):
        convert_split_manifest(manifest, tmp_path, tmp_path / "npz")


# --------------------------------------------------------------------------- universe + splits
def _write_universe(path: Path, rows: list[dict[str, str]]) -> Path:
    fields = ["source", "index", "annotation"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _save_selectors(path: Path, per_stage: dict[str, list[tuple[str, list[int]]]]) -> Path:
    from cuvis_ai_core.data.splits_io import save_splits
    from cuvis_ai_schemas.training.data import DataSplitConfig, Selector, SelectorKind

    def sels(stage):
        return [
            Selector(kind=SelectorKind.FILE_INDICES, source=src, ids=ids)
            for src, ids in per_stage.get(stage, [])
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    save_splits(
        DataSplitConfig(
            train=sels("train"), val=sels("val"), test=sels("test"), predict=sels("predict")
        ),
        path,
    )
    return path


def test_convert_universe_end_to_end(patched, tmp_path):
    root = tmp_path / "raw"
    (root / "day2").mkdir(parents=True)
    (root / "day3").mkdir(parents=True)
    (root / "day2" / "A.cu3s").touch()
    (root / "day2" / "A.json").touch()
    (root / "day3" / "B.cu3s").touch()

    # The shape a dataset publishes: a unified universe.csv + a splits.json selector.
    universe_in = _write_universe(
        root / "universe.csv",
        [
            {"source": "day2/A.cu3s", "index": "0", "annotation": "day2/A.json"},
            {"source": "day2/A.cu3s", "index": "1", "annotation": "day2/A.json"},
            {"source": "day3/B.cu3s", "index": "0", "annotation": ""},
            # present in the universe but NOT selected -> must not be materialized
            {"source": "day3/B.cu3s", "index": "1", "annotation": ""},
        ],
    )
    splits_in = _save_selectors(
        root / "splits" / "sel.json",
        {
            "train": [("day2/A.cu3s", [0])],
            "val": [("day2/A.cu3s", [1])],
            "test": [("day3/B.cu3s", [0])],
            "predict": [("day3/B.cu3s", [0])],
        },
    )

    out = tmp_path / "npz"
    result = convert_universe(universe_in, root, out, splits_json=splits_in)

    assert result.universe_csv == out / "universe.csv"
    assert result.splits_json == out / "splits.json"

    with result.universe_csv.open() as f:
        idx = list(csv.DictReader(f))
    # Only the 3 SELECTED frames are converted; the unselected day3/B#1 is skipped.
    assert list(idx[0].keys()) == ["source", "index", "materialized_path"]
    assert len(idx) == 3
    assert {(r["source"], r["index"]) for r in idx} == {
        ("day2/A.cu3s", "0"),
        ("day2/A.cu3s", "1"),
        ("day3/B.cu3s", "0"),
    }
    assert all(not Path(r["materialized_path"]).is_absolute() for r in idx)
    assert all((result.universe_csv.parent / r["materialized_path"]).is_file() for r in idx)

    # The SAME splits.json resolves against the converted npz universe (unchanged identities).
    from cuvis_ai_core.data.splits_io import load_splits

    cfg = load_splits(result.splits_json)
    dm = MultiNpzDataModule(splits=cfg, universe_csv=str(result.universe_csv))
    dm.setup(stage="fit")
    dm.setup(stage="test")
    assert len(dm.train_ds) == 1 and len(dm.val_ds) == 1 and len(dm.test_ds) == 1
    # Annotated session bakes masks; the unannotated one stays mask-free.
    with np.load(dm.train_ds.rows[0]["materialized_path"]) as z:
        assert "mask" in z.files and int(z["class_mask"].max()) == 3
    with np.load(dm.test_ds.rows[0]["materialized_path"]) as z:
        assert "mask" not in z.files


def test_convert_universe_rejects_selector_absent_from_universe(patched, tmp_path):
    root = tmp_path / "raw"
    (root / "day2").mkdir(parents=True)
    (root / "day2" / "A.cu3s").touch()
    universe_in = _write_universe(
        root / "universe.csv",
        [{"source": "day2/A.cu3s", "index": "0", "annotation": ""}],
    )
    # selector references index 1, which the universe does not list
    splits_in = _save_selectors(root / "splits" / "sel.json", {"train": [("day2/A.cu3s", [0, 1])]})
    with pytest.raises(ValueError, match="absent from the universe"):
        convert_universe(universe_in, root, tmp_path / "npz", splits_json=splits_in)
