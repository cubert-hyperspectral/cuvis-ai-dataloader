"""The GUI-authored splits contract over a cu3s folder (frames="measurements").

``tests/cuvis_ai_dataloader/fixtures/gui_authored_splits.json`` is the **golden
cross-language fixture**: the exact ``splits.json`` an external split author (the
CuvisNEXT split designer) writes. The same file is committed in the cuvis-next test
suite, where the C++ serializer must reproduce it byte-for-byte (after substituting
the ``{DATA_DIR}`` token). Changing the fixture is a contract change for both repos.

The fixture keeps machine independence via the ``{DATA_DIR}`` token: tests substitute
it with the canonical form of a temp folder (``Path.resolve().as_posix()``, the same
form ``enumerate`` emits). The fixture's ``universe_hash`` is the sha256 over the
**tokenized** universe uids, so it doubles as the hash golden vector shared with the
C++ implementation (``sha256(uid + "\\n" for each uid, ordered)``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cuvis_ai_core.data.splits_io import load_splits, universe_hash
from cuvis_ai_dataloader.data.datamodule_cu3s import Cu3sDataModule
from cuvis_ai_schemas.training.data import DataSplitConfig

FIXTURE = Path(__file__).parents[1] / "fixtures" / "gui_authored_splits.json"

# Per-file measurement count comes from the mocked SDK session (see conftest).
MOCK_MEASUREMENTS = 7

# The frozen assignment the fixture encodes, per (relative source, index).
EXPECTED_TRAIN = {("day2/a.cu3s", i) for i in (0, 2, 3, 4)} | {
    ("day3/b.cu3s", i) for i in (1, 2, 3)
}
EXPECTED_VAL = {("day2/a.cu3s", 5), ("day3/b.cu3s", 0), ("day3/b.cu3s", 6)}
EXPECTED_TEST = {("day2/a.cu3s", 1), ("day2/a.cu3s", 6), ("day3/b.cu3s", 4)}


def _materialize_universe(tmp_path) -> str:
    """Create the fixture's folder layout; return the canonical ``{DATA_DIR}`` value."""
    for rel in ("day2/a.cu3s", "day3/b.cu3s"):
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"")
        f.with_suffix(".json").write_text("{}")  # sibling COCO (content mocked)
    return Path(tmp_path).resolve().as_posix()


def _write_substituted(tmp_path) -> tuple[str, str]:
    """Substitute ``{DATA_DIR}`` and write the splits.json; return (path, data_dir)."""
    data_dir = _materialize_universe(tmp_path)
    text = FIXTURE.read_text(encoding="utf-8").replace("{DATA_DIR}", data_dir)
    out = tmp_path / "splits.json"
    out.write_text(text, encoding="utf-8")
    return str(out), data_dir


def _module(splits_path: str, data_dir: str) -> Cu3sDataModule:
    return Cu3sDataModule(
        data_dir=data_dir,
        frames="measurements",
        recursive=True,
        splits=DataSplitConfig(splits_path=splits_path),
        batch_size=1,
    )


def _pairs(ds, data_dir: str) -> set[tuple[str, int]]:
    prefix = data_dir + "/"
    return {(r.source.removeprefix(prefix), r.index) for r in ds._refs}


def test_fixture_is_schema_valid():
    # The tokenized fixture itself must parse as a DataSplitConfig (source is opaque).
    cfg = DataSplitConfig.model_validate(json.loads(FIXTURE.read_text(encoding="utf-8")))
    assert cfg.leakage_check == "error"
    assert cfg.predict == []


def test_fixture_universe_hash_is_the_golden_vector():
    # sha256 over the ordered tokenized uids: the cross-language hash vector. The C++
    # split serializer must produce this exact value for this universe.
    uids = [f"{{DATA_DIR}}/day2/a.cu3s#{i}" for i in range(MOCK_MEASUREMENTS)] + [
        f"{{DATA_DIR}}/day3/b.cu3s#{i}" for i in range(MOCK_MEASUREMENTS)
    ]
    digest = hashlib.sha256()
    for uid in uids:
        digest.update(uid.encode("utf-8"))
        digest.update(b"\n")
    stored = json.loads(FIXTURE.read_text(encoding="utf-8"))["universe_hash"]
    assert digest.hexdigest() == stored


def test_measurements_enumeration_is_canonical(mock_cuvis_sdk, tmp_path):
    _, data_dir = _write_substituted(tmp_path)
    dm = Cu3sDataModule(data_dir=data_dir, frames="measurements", recursive=True)
    refs = dm.enumerate(frozenset({"tags"}))
    assert len(refs) == 2 * MOCK_MEASUREMENTS
    # Canonical absolute sources: forward slashes, resolve()-form, sorted (source, index).
    assert refs[0].source == f"{data_dir}/day2/a.cu3s"
    assert "\\" not in refs[0].source
    assert [r.index for r in refs[:MOCK_MEASUREMENTS]] == list(range(MOCK_MEASUREMENTS))
    assert refs[MOCK_MEASUREMENTS].source == f"{data_dir}/day3/b.cu3s"
    # uid = source#index (label_id == index adds nothing); sibling COCO attached.
    assert refs[0].uid == f"{data_dir}/day2/a.cu3s#0"
    assert refs[0].annotation and refs[0].annotation.endswith("a.json")
    assert refs[0].tags in (["normal"], ["anomalous"])
    # universe_hash over the enumerated refs is well-formed (value is machine-specific).
    assert len(universe_hash(refs)) == 64


def test_fixture_resolves_to_exact_splits(mock_cuvis_sdk, tmp_path):
    splits_path, data_dir = _write_substituted(tmp_path)
    assert load_splits(splits_path)  # schema-valid after substitution
    dm = _module(splits_path, data_dir)
    dm.setup(stage="fit")
    dm.setup(stage="test")
    assert _pairs(dm._train_ds, data_dir) == EXPECTED_TRAIN
    assert _pairs(dm._val_ds, data_dir) == EXPECTED_VAL
    assert _pairs(dm._test_ds, data_dir) == EXPECTED_TEST
    # b#5 is deliberately unassigned: assigned 13 of 14.
    assert len(EXPECTED_TRAIN | EXPECTED_VAL | EXPECTED_TEST) == 13


def test_empty_predict_serves_whole_universe(mock_cuvis_sdk, tmp_path):
    splits_path, data_dir = _write_substituted(tmp_path)
    dm = _module(splits_path, data_dir)
    dm.setup(stage="predict")
    assert len(dm._predict_ds) == 2 * MOCK_MEASUREMENTS


def test_train_dataloader_serves_exactly_the_frozen_train_split(mock_cuvis_sdk, tmp_path):
    # The module half of the stat-init invariant: the train loader (the only thing the
    # StatisticalTrainer iterates) yields exactly the frozen train assignment.
    splits_path, data_dir = _write_substituted(tmp_path)
    dm = _module(splits_path, data_dir)
    dm.setup(stage="fit")
    seen = set()
    for batch in dm.train_dataloader():
        seen.add((batch["stem"][0], int(batch["mesu_index"][0])))
    assert seen == {(Path(src).stem, idx) for src, idx in EXPECTED_TRAIN}


def test_overlap_raises_leakage(mock_cuvis_sdk, tmp_path):
    splits_path, data_dir = _write_substituted(tmp_path)
    doc = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    doc["val"][0]["ids"].append(0)  # a#0 is already in train
    Path(splits_path).write_text(json.dumps(doc), encoding="utf-8")
    dm = _module(splits_path, data_dir)
    with pytest.raises(Exception, match="leakage"):
        dm.setup(stage="fit")


def test_moved_file_fails_loud(mock_cuvis_sdk, tmp_path):
    # A renamed/moved member cu3s must fail resolution, not silently shrink a split.
    splits_path, data_dir = _write_substituted(tmp_path)
    root = Path(data_dir)
    (root / "day3" / "b.cu3s").rename(root / "day3" / "renamed.cu3s")
    dm = _module(splits_path, data_dir)
    with pytest.raises(ValueError, match="matched 0 samples"):
        dm.setup(stage="fit")
