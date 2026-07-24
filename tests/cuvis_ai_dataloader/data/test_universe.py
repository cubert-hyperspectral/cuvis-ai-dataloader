"""Tests for the shared universe.csv parser and the invariants it enforces for both modules.

Covers ``_universe.parse_universe`` directly (no SDK needed), the cross-module ``source`` identity
match that the posix normalization fixes, and the load-bearing invariant that an all-npz run never
imports the cuvis SDK.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from cuvis_ai_dataloader.data._universe import parse_universe

CU3S_FLAGS = dict(
    require_materialized_path=False,
    accept_split=True,
    unique_materialized_path=False,
    allow_index_ranges=True,
)
NPZ_FLAGS = dict(
    require_materialized_path=True,
    accept_split=False,
    unique_materialized_path=True,
    allow_index_ranges=False,
)


def _write(tmp_path, text: str):
    p = tmp_path / "universe.csv"
    p.write_text(text)
    return p


def test_missing_required_column(tmp_path):
    csv = _write(tmp_path, "source,materialized_path\ns.cu3s,f.npz\n")  # no index
    with pytest.raises(ValueError, match="missing required column"):
        parse_universe(csv, **NPZ_FLAGS)


def test_empty_universe_raises(tmp_path):
    csv = _write(tmp_path, "source,index\n")
    with pytest.raises(ValueError, match="no rows"):
        parse_universe(csv, **CU3S_FLAGS)


def test_cu3s_materialized_path_defaults_to_source(tmp_path):
    csv = _write(tmp_path, "source,index\nrec.cu3s,0\n")
    rows = parse_universe(csv, **CU3S_FLAGS)
    assert rows[0]["source"] == "rec.cu3s"
    assert rows[0]["materialized_path"].endswith("rec.cu3s")  # resolved from source


def test_npz_requires_materialized_path(tmp_path):
    csv = _write(tmp_path, "source,index\nrec.cu3s,0\n")
    with pytest.raises(ValueError, match="materialized_path"):
        parse_universe(csv, **NPZ_FLAGS)


def test_npz_rejects_split_column(tmp_path):
    csv = _write(tmp_path, "source,index,materialized_path,split\ns.cu3s,0,f.npz,train\n")
    with pytest.raises(ValueError, match="split"):
        parse_universe(csv, **NPZ_FLAGS)


def test_source_posix_normalized(tmp_path):
    csv = _write(tmp_path, "source,index\nday2\\rec.cu3s,0\n")
    rows = parse_universe(csv, **CU3S_FLAGS)
    assert rows[0]["source"] == "day2/rec.cu3s"


def test_duplicate_identity_rejected(tmp_path):
    csv = _write(tmp_path, "source,index\nrec.cu3s,0\nrec.cu3s,0\n")
    with pytest.raises(ValueError, match="duplicate identity"):
        parse_universe(csv, **CU3S_FLAGS)


def test_duplicate_materialized_path_rejected_for_npz(tmp_path):
    csv = _write(tmp_path, "source,index,materialized_path\ns.cu3s,0,f.npz\ns.cu3s,1,f.npz\n")
    with pytest.raises(ValueError, match="duplicate materialized_path"):
        parse_universe(csv, **NPZ_FLAGS)


def test_shared_materialized_path_allowed_for_cu3s(tmp_path):
    # One .cu3s recording legitimately backs many frames, so cu3s does not require unique paths.
    csv = _write(tmp_path, "source,index\nrec.cu3s,0\nrec.cu3s,1\n")
    rows = parse_universe(csv, **CU3S_FLAGS)
    assert len(rows) == 2
    assert rows[0]["materialized_path"] == rows[1]["materialized_path"]


def test_parent_escape_rejected(tmp_path):
    csv = _write(tmp_path, "source,index,materialized_path\ns.cu3s,0,../f.npz\n")
    with pytest.raises(ValueError, match="must not contain"):
        parse_universe(csv, **NPZ_FLAGS)


def test_range_fan_out_for_cu3s(tmp_path):
    csv = _write(tmp_path, "source,index\nrec.cu3s,0-2\n")
    rows = parse_universe(csv, **CU3S_FLAGS)
    assert [r["index"] for r in rows] == [0, 1, 2]
    assert [r["frame_id"] for r in rows] == [0, 1, 2]  # contiguous across the fan-out


def test_group_and_annotation_carried(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    csv = _write(tmp_path, "source,index,annotation,group\nrec.cu3s,0,a.json,batch_a\n")
    rows = parse_universe(csv, **CU3S_FLAGS)
    assert rows[0]["group"] == "batch_a"
    assert rows[0]["annotation"].endswith("a.json")


def test_cross_module_source_identity_matches(tmp_path):
    """A cu3s universe and its converted-npz universe agree on ``(source, index)``.

    This is the bug the shared parser fixes: cu3s previously keyed on a resolved (backslash)
    absolute path, npz on the posix identity, so one splits.json could not resolve against both on
    Windows. Both now enumerate the same posix ``source``.
    """
    from cuvis_ai_dataloader.data.datamodule_cu3s_multi import MultiCu3sDataModule
    from cuvis_ai_dataloader.data.datamodule_npz_multi import MultiNpzDataModule
    from cuvis_ai_schemas.training.data import DataSplitConfig, Selector, SelectorKind

    cu3s_csv = tmp_path / "cu3s.csv"
    cu3s_csv.write_text("source,index\nday2\\rec.cu3s,0\n")  # backslash source
    cu3s = MultiCu3sDataModule(universe_csv=str(cu3s_csv))

    _write_npz(tmp_path / "f0.npz")
    npz_csv = tmp_path / "npz.csv"
    npz_csv.write_text("source,index,materialized_path\nday2/rec.cu3s,0,f0.npz\n")  # posix source
    npz = MultiNpzDataModule(
        splits=DataSplitConfig(
            train=[Selector(kind=SelectorKind.FILE_INDICES, source="day2/rec.cu3s", ids=[0])]
        ),
        universe_csv=str(npz_csv),
    )

    cu3s_ref = cu3s.enumerate()[0]  # no attrs -> no SDK, no file access
    npz_ref = npz.enumerate()[0]
    assert cu3s_ref.source == npz_ref.source == "day2/rec.cu3s"
    assert (cu3s_ref.source, cu3s_ref.index) == (npz_ref.source, npz_ref.index)


def _write_npz(path):
    np.savez(
        path,
        cube=np.zeros((4, 5, 3), dtype=np.float32),
        wavelengths=np.array([450, 550, 650], dtype=np.float32),
        mask=np.zeros((4, 5), dtype=np.int32),
    )


def test_npz_run_does_not_import_cuvis_sdk(tmp_path):
    """CRITICAL: an all-npz universe must never import the cuvis SDK (structural, no dispatch)."""
    for i in range(2):
        _write_npz(tmp_path / f"f{i}.npz")
    universe = tmp_path / "universe.csv"
    universe.write_text("source,index,materialized_path\ns.cu3s,0,f0.npz\ns.cu3s,1,f1.npz\n")
    code = textwrap.dedent(
        f"""
        import sys
        from cuvis_ai_dataloader.data.datamodule_npz_multi import MultiNpzDataModule
        from cuvis_ai_schemas.training.data import DataSplitConfig, Selector, SelectorKind
        dm = MultiNpzDataModule(
            splits=DataSplitConfig(
                train=[Selector(kind=SelectorKind.FILE_INDICES, source="s.cu3s", ids=[0, 1])]
            ),
            universe_csv=r"{universe}",
        )
        dm.setup(stage="fit")
        _ = dm.train_ds[0]
        assert "cuvis" not in sys.modules, "cuvis SDK was imported on an all-npz run"
        print("ISOLATION_OK")
        """
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert "ISOLATION_OK" in res.stdout
