"""Tests for MultiNpzDataModule (one-frame-per-file NPZ + selector splits over a universe.csv)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cuvis_ai_core.data.datamodule import BaseCuvisAIDataModule
from cuvis_ai_dataloader.data.datamodule_npz_multi import MultiNpzDataModule, _MultiNpzDataset


def _write_npz(path: Path, *, with_mask: bool, with_class_mask: bool = False) -> None:
    h, w, c = 8, 10, 5
    cube = np.zeros((h, w, c), dtype=np.float32)
    wavelengths = np.linspace(450, 850, c).astype(np.float32)
    arrays: dict[str, np.ndarray] = {"cube": cube, "wavelengths": wavelengths}
    if with_mask:
        mask = np.zeros((h, w), dtype=np.int32)
        mask[2:5, 3:7] = 2
        arrays["mask"] = mask
    if with_class_mask:
        class_mask = np.zeros((h, w), dtype=np.uint8)
        class_mask[2:5, 3:7] = 3  # COCO category id 3
        arrays["class_mask"] = class_mask
    np.savez(path, **arrays)


def test_data_module_name_and_subclass():
    assert MultiNpzDataModule.DATA_MODULE_NAME == "npz_multi"
    assert issubclass(MultiNpzDataModule, BaseCuvisAIDataModule)


def test_validate_params_rejects_non_csv(tmp_path):
    bad = tmp_path / "x.txt"
    bad.write_text("")
    with pytest.raises(ValueError, match=r"\.csv"):
        MultiNpzDataModule.validate_params({"universe_csv": str(bad)})


def test_unknown_kwarg_raises(tmp_path):
    universe = _write_universe(tmp_path)
    with pytest.raises(TypeError, match="bogus"):
        MultiNpzDataModule(splits=_split_cfg(train=[0]), universe_csv=str(universe), bogus=1)


def test_missing_required_column_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("source,index\ns.cu3s,0\n")  # no path column
    with pytest.raises(ValueError, match="missing required column"):
        MultiNpzDataModule(splits=_split_cfg(train=[0]), universe_csv=str(csv_path))


def test_dataset_reads_cube_mask_and_wavelengths(tmp_path):
    npz = tmp_path / "frame.npz"
    _write_npz(npz, with_mask=True)
    ds = _MultiNpzDataset([{"path": str(npz), "index": 7, "frame_id": 0}])
    item = ds[0]
    assert item["cube"].shape == (8, 10, 5)
    assert item["mask"].shape == (8, 10)
    assert item["wavelengths"].shape == (5,)
    assert item["wavelengths"].dtype == np.int32
    assert int(item["mesu_index"]) == 7
    assert ds.num_channels == 5


def test_dataset_builds_empty_mask_when_absent(tmp_path):
    npz = tmp_path / "frame_nomask.npz"
    _write_npz(npz, with_mask=False)
    ds = _MultiNpzDataset([{"path": str(npz), "index": 11, "frame_id": 0}])
    item = ds[0]
    assert item["mask"].shape == (8, 10)
    assert item["mask"].dtype == np.int32
    assert np.all(item["mask"] == 0)


def test_dataset_reads_class_mask_when_present(tmp_path):
    npz = tmp_path / "frame_cm.npz"
    _write_npz(npz, with_mask=True, with_class_mask=True)
    ds = _MultiNpzDataset([{"path": str(npz), "index": 5, "frame_id": 0}])
    item = ds[0]
    assert item["class_mask"].shape == (8, 10)
    assert item["class_mask"].dtype == np.uint8
    assert int(item["class_mask"].max()) == 3  # COCO category id preserved
    # binary mask and class_mask agree on the anomalous region
    assert np.array_equal(item["class_mask"] > 0, item["mask"] > 0)


def test_dataset_emits_zero_class_mask_when_absent(tmp_path):
    npz = tmp_path / "frame_nocm.npz"
    _write_npz(npz, with_mask=True)  # no class_mask key
    ds = _MultiNpzDataset([{"path": str(npz), "index": 6, "frame_id": 0}])
    item = ds[0]
    assert item["class_mask"].shape == (8, 10)
    assert item["class_mask"].dtype == np.uint8
    assert np.all(item["class_mask"] == 0)


# --------------------------------------------------------------------------- selector path
def _write_universe(tmp_path: Path) -> Path:
    """6 frames from one source `s.cu3s`, index 0..5; universe.csv + npz files."""
    for i in range(6):
        _write_npz(tmp_path / f"f{i}.npz", with_mask=True)
    universe = tmp_path / "universe.csv"
    rows = "".join(f"s.cu3s,{i},f{i}.npz\n" for i in range(6))
    universe.write_text("source,index,path\n" + rows)
    return universe


def _split_cfg(**stages):
    from cuvis_ai_schemas.training.data import DataSplitConfig, Selector, SelectorKind

    def fi(ids):
        return [Selector(kind=SelectorKind.FILE_INDICES, source="s.cu3s", ids=ids)]

    return DataSplitConfig(**{k: fi(v) for k, v in stages.items()})


def test_selector_path_requires_universe():
    with pytest.raises(ValueError, match="requires 'universe_csv'"):
        MultiNpzDataModule(splits=_split_cfg(train=[0]))


def test_validate_params_accepts_universe(tmp_path):
    universe = _write_universe(tmp_path)
    MultiNpzDataModule.validate_params({"universe_csv": str(universe)})  # no raise


def test_validate_params_requires_universe():
    with pytest.raises(ValueError, match="requires 'universe_csv'"):
        MultiNpzDataModule.validate_params({})


def test_requires_splits_selector(tmp_path):
    universe = _write_universe(tmp_path)
    with pytest.raises(ValueError, match="requires a 'splits' selector"):
        MultiNpzDataModule(universe_csv=str(universe))  # no splits


def test_enumerate_identity_and_order(tmp_path):
    universe = _write_universe(tmp_path)
    dm = MultiNpzDataModule(splits=_split_cfg(train=[0]), universe_csv=str(universe))
    refs = dm.enumerate()
    assert [(r.source, r.index) for r in refs] == [("s.cu3s", i) for i in range(6)]
    assert refs[0].stem == "s"


def test_enumerate_rejects_tag_selectors(tmp_path):
    universe = _write_universe(tmp_path)
    dm = MultiNpzDataModule(splits=_split_cfg(train=[0]), universe_csv=str(universe))
    with pytest.raises(NotImplementedError, match="category name->id"):
        dm.enumerate(frozenset({"tags"}))


def test_selector_setup_fit_and_test_resolve_subsets(tmp_path):
    universe = _write_universe(tmp_path)
    dm = MultiNpzDataModule(
        splits=_split_cfg(train=[0, 1, 2], val=[3], test=[4, 5]), universe_csv=str(universe)
    )
    dm.setup(stage="fit")
    dm.setup(stage="test")
    assert len(dm.train_ds) == 3
    assert len(dm.val_ds) == 1
    assert len(dm.test_ds) == 2
    # build_dataset_from_refs mapped identity -> the right npz
    assert dm.test_ds.rows[0]["path"].endswith("f4.npz")


def test_selector_predict_empty_iterates_universe(tmp_path):
    universe = _write_universe(tmp_path)
    from cuvis_ai_schemas.training.data import DataSplitConfig

    dm = MultiNpzDataModule(splits=DataSplitConfig(predict=[]), universe_csv=str(universe))
    dm.setup(stage="predict")
    assert len(dm.predict_ds) == 6


def test_selector_leakage_fires_on_overlap(tmp_path):
    from cuvis_ai_core.data.selectors import SplitLeakageError

    universe = _write_universe(tmp_path)
    dm = MultiNpzDataModule(
        splits=_split_cfg(train=[0, 1], test=[1, 2]), universe_csv=str(universe)
    )
    with pytest.raises(SplitLeakageError):
        dm.setup(stage="fit")


def test_universe_duplicate_identity_raises(tmp_path):
    _write_npz(tmp_path / "f0.npz", with_mask=True)
    _write_npz(tmp_path / "f1.npz", with_mask=True)
    universe = tmp_path / "universe.csv"
    universe.write_text("source,index,path\ns.cu3s,0,f0.npz\ns.cu3s,0,f1.npz\n")
    with pytest.raises(ValueError, match="duplicate identity"):
        MultiNpzDataModule(splits=_split_cfg(train=[0]), universe_csv=str(universe))


def test_universe_duplicate_path_raises(tmp_path):
    _write_npz(tmp_path / "f0.npz", with_mask=True)
    universe = tmp_path / "universe.csv"
    # Distinct identities, same npz path -> rejected (each row must be a distinct file).
    universe.write_text("source,index,path\ns.cu3s,0,f0.npz\ns.cu3s,1,f0.npz\n")
    with pytest.raises(ValueError, match="duplicate path"):
        MultiNpzDataModule(splits=_split_cfg(train=[0]), universe_csv=str(universe))


def test_universe_rejects_parent_escape_path(tmp_path):
    universe = tmp_path / "universe.csv"
    universe.write_text("source,index,path\ns.cu3s,0,../f0.npz\n")
    with pytest.raises(ValueError, match="must not contain"):
        MultiNpzDataModule(splits=_split_cfg(train=[0]), universe_csv=str(universe))


def test_universe_group_column_carried(tmp_path):
    _write_npz(tmp_path / "f0.npz", with_mask=True)
    universe = tmp_path / "universe.csv"
    # The optional `group` column is parsed and carried onto SampleRef.group.
    universe.write_text("source,index,path,group\ns.cu3s,0,f0.npz,batch_a\n")
    dm = MultiNpzDataModule(splits=_split_cfg(train=[0]), universe_csv=str(universe))
    assert dm.enumerate()[0].group == "batch_a"


def test_universe_source_normalized_to_posix(tmp_path):
    _write_npz(tmp_path / "f0.npz", with_mask=True)
    universe = tmp_path / "universe.csv"
    universe.write_text("source,index,path\nday2\\s.cu3s,0,f0.npz\n")
    # A selector authored with a posix source must still resolve the backslash-authored row.
    dm = MultiNpzDataModule(splits=_split_cfg(train=[0]), universe_csv=str(universe))
    from cuvis_ai_schemas.training.data import DataSplitConfig, Selector, SelectorKind

    dm.splits = DataSplitConfig(
        train=[Selector(kind=SelectorKind.FILE_INDICES, source="day2/s.cu3s", ids=[0])]
    )
    dm._refs = None  # re-enumerate against the new selector
    dm.setup(stage="fit")
    assert len(dm.train_ds) == 1


def test_selector_loader_before_setup_raises(tmp_path):
    universe = _write_universe(tmp_path)
    dm = MultiNpzDataModule(splits=_split_cfg(train=[0]), universe_csv=str(universe))
    with pytest.raises(RuntimeError):
        dm.train_dataloader()


def test_selector_loader_honors_pin_memory_and_batch(tmp_path):
    universe = _write_universe(tmp_path)
    dm = MultiNpzDataModule(
        splits=_split_cfg(train=[0, 1]), universe_csv=str(universe), pin_memory=True, num_workers=0
    )
    dm.setup(stage="fit")
    loader = dm.train_dataloader()
    assert loader.pin_memory is True
    batch = next(iter(loader))
    assert set(batch.keys()) >= {"cube", "mask", "wavelengths", "mesu_index"}
    assert batch["cube"].shape[0] == 1


def test_samples_per_frame_repeats_train_loader_only(tmp_path):
    universe = _write_universe(tmp_path)
    dm = MultiNpzDataModule(
        splits=_split_cfg(train=[0, 1], val=[2]),
        universe_csv=str(universe),
        samples_per_frame=3,
        batch_size=1,
        num_workers=0,
    )
    dm.setup(stage="fit")
    # Multiplicity is applied by the base train_dataloader (train split only): the
    # loader's dataset is N x the train frames, while train_ds stays the frame count.
    assert len(dm.train_ds) == 2  # 2 train frames
    assert len(dm.train_dataloader().dataset) == 6  # 2 frames x 3
    assert len(dm.val_dataloader().dataset) == 1  # val is never repeated
    # Every repeat references the same frames; per-sample randomness (e.g. crops) is
    # drawn downstream, so the duplicates ARE independent training samples.
    rep = dm.train_dataloader().dataset
    ids = sorted(rep[i]["frame_id"] for i in range(len(rep)))
    assert ids == [0, 0, 0, 1, 1, 1]  # each train frame appears exactly 3x


def test_samples_per_frame_default_is_identity(tmp_path):
    universe = _write_universe(tmp_path)
    dm = MultiNpzDataModule(
        splits=_split_cfg(train=[0, 1]), universe_csv=str(universe), num_workers=0
    )
    dm.setup(stage="fit")
    # default (1): the base does not wrap, so the loader dataset == unique frames.
    assert len(dm.train_dataloader().dataset) == 2


def test_samples_per_frame_via_params_dict(tmp_path):
    universe = _write_universe(tmp_path)
    dm = MultiNpzDataModule(
        splits=_split_cfg(train=[0, 1]),
        params={"universe_csv": str(universe), "samples_per_frame": 2},
    )
    dm.setup(stage="fit")
    assert len(dm.train_dataloader().dataset) == 4  # 2 frames x 2


def test_samples_per_frame_validation(tmp_path):
    universe = _write_universe(tmp_path)
    with pytest.raises(ValueError, match="samples_per_frame"):
        MultiNpzDataModule(
            splits=_split_cfg(train=[0]), universe_csv=str(universe), samples_per_frame=0
        )


# --------------------------------------------------------------------------- crop-in-dataset
def _write_gradient_npz(path: Path, h: int = 8, w: int = 10, c: int = 5) -> None:
    """Write a frame whose cube encodes position (``cube[y, x, :] == y*100 + x``) + empty mask.

    A position-encoding cube lets a test tell crops apart by their top-left value, since the
    default fixture cube is all zeros (indistinguishable across offsets).
    """
    cube = np.tile(
        (np.arange(h)[:, None] * 100 + np.arange(w)[None, :]).astype(np.float32)[..., None],
        (1, 1, c),
    )
    np.savez(
        path,
        cube=cube,
        wavelengths=np.linspace(450, 850, c).astype(np.float32),
        mask=np.zeros((h, w), dtype=np.int32),
    )


def test_fg_crop_window_in_bounds_hits_fg_and_rejects_oversize():
    from cuvis_ai_dataloader.data._crop import fg_crop_window

    rng = np.random.default_rng(0)
    mask = np.zeros((8, 10), dtype=np.int32)
    mask[2:5, 3:7] = 1
    top, left = fg_crop_window(mask, (4, 6), fg_percent=1.0, fg_labels=None, rng=rng)
    assert 0 <= top <= 4 and 0 <= left <= 4  # clamped in-bounds
    assert mask[top : top + 4, left : left + 6].max() > 0  # foreground-centered window hits object
    with pytest.raises(ValueError, match="exceeds"):
        fg_crop_window(mask, (9, 6), fg_percent=1.0, fg_labels=None, rng=rng)


def test_crop_default_off_ships_full_frame(tmp_path):
    universe = _write_universe(tmp_path)
    dm = MultiNpzDataModule(
        splits=_split_cfg(train=[0, 1]), universe_csv=str(universe), batch_size=1, num_workers=0
    )
    dm.setup(stage="fit")
    batch = next(iter(dm.train_dataloader()))
    assert batch["cube"].shape == (1, 8, 10, 5)  # whole frame, unchanged


def test_crop_train_ships_patch(tmp_path):
    universe = _write_universe(tmp_path)
    dm = MultiNpzDataModule(
        splits=_split_cfg(train=[0, 1]),
        universe_csv=str(universe),
        crop_size=(4, 6),
        batch_size=1,
        num_workers=0,
    )
    dm.setup(stage="fit")
    batch = next(iter(dm.train_dataloader()))
    assert batch["cube"].shape == (1, 4, 6, 5)
    assert batch["mask"].shape == (1, 4, 6)
    assert batch["class_mask"].shape == (1, 4, 6)


def test_crop_not_applied_to_val_and_test(tmp_path):
    universe = _write_universe(tmp_path)
    dm = MultiNpzDataModule(
        splits=_split_cfg(train=[0], val=[1], test=[2]),
        universe_csv=str(universe),
        crop_size=(4, 6),
        batch_size=1,
        num_workers=0,
    )
    dm.setup(stage="fit")
    dm.setup(stage="test")
    assert next(iter(dm.train_dataloader()))["cube"].shape == (1, 4, 6, 5)  # cropped
    assert next(iter(dm.val_dataloader()))["cube"].shape == (1, 8, 10, 5)  # full frame
    assert next(iter(dm.test_dataloader()))["cube"].shape == (1, 8, 10, 5)  # full frame


def test_crop_with_samples_per_frame_yields_independent_patches(tmp_path):
    _write_gradient_npz(tmp_path / "f0.npz")
    universe = tmp_path / "universe.csv"
    universe.write_text("source,index,path\ns.cu3s,0,f0.npz\n")
    dm = MultiNpzDataModule(
        splits=_split_cfg(train=[0]),
        universe_csv=str(universe),
        crop_size=(4, 6),
        crop_fg_percent=0.0,  # uniform offsets so patches vary by position
        samples_per_frame=8,
        batch_size=1,
        num_workers=0,
    )
    dm.setup(stage="fit")
    assert len(dm.train_ds) == 1  # property stays the unwrapped frame count
    rep = dm.train_dataloader().dataset
    assert len(rep) == 8  # 1 frame x 8 samples
    # cube[0,0,0] == top*100 + left, so distinct top-left values prove independent crop offsets.
    offsets = {int(rep[i]["cube"][0, 0, 0]) for i in range(len(rep))}
    assert len(offsets) > 1


def test_crop_foreground_biased_hits_object(tmp_path):
    _write_npz(tmp_path / "f0.npz", with_mask=True)  # mask block [2:5, 3:7] = 2
    universe = tmp_path / "universe.csv"
    universe.write_text("source,index,path\ns.cu3s,0,f0.npz\n")
    dm = MultiNpzDataModule(
        splits=_split_cfg(train=[0]),
        universe_csv=str(universe),
        crop_size=(4, 6),
        crop_fg_percent=1.0,  # every crop centers on the object
        samples_per_frame=5,
        batch_size=1,
        num_workers=0,
    )
    dm.setup(stage="fit")
    rep = dm.train_dataloader().dataset
    assert all(int(rep[i]["mask"].max()) > 0 for i in range(len(rep)))  # crop always contains fg


def test_crop_via_params_dict(tmp_path):
    universe = _write_universe(tmp_path)
    dm = MultiNpzDataModule(
        splits=_split_cfg(train=[0, 1]),
        params={"universe_csv": str(universe), "crop_size": (4, 6)},
        num_workers=0,
    )
    dm.setup(stage="fit")
    assert next(iter(dm.train_dataloader()))["cube"].shape == (1, 4, 6, 5)


def test_crop_size_validation(tmp_path):
    universe = _write_universe(tmp_path)
    with pytest.raises(ValueError, match="crop_size"):
        MultiNpzDataModule(
            splits=_split_cfg(train=[0]), universe_csv=str(universe), crop_size=(0, 6)
        )


def test_crop_fg_percent_validation(tmp_path):
    universe = _write_universe(tmp_path)
    with pytest.raises(ValueError, match="crop_fg_percent"):
        MultiNpzDataModule(
            splits=_split_cfg(train=[0]), universe_csv=str(universe), crop_fg_percent=1.5
        )
