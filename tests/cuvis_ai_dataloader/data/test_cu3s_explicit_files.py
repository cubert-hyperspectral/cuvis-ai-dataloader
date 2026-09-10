"""The cu3s module opens only the recordings a run actually uses.

Two ways to narrow, one helper behind both (``Cu3sDataModule._folder_files``):

* an explicit ``files`` list, which is what the CuvisNEXT training wizard sends, since it
  already knows every recording its split assigns;
* otherwise, inference from the split itself when every selector names its sources.

Anything else (a positional or attribute-driven selector) keeps the folder walk, because
only the full universe can answer it.

The assertions here are about *which files were opened*, read off the patched SDK's
``SessionFile`` constructor, because that is the cost the ticket is about: enumeration used
to build a full reader (an SDK session plus a ``ProcessingContext`` plus a read of
measurement 0) for every recording under the folder, before any selector was applied.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from cuvis_ai_dataloader.data.datamodule_cu3s import (
    Cu3sDataModule,
    ExplicitSources,
    explicit_sources,
)
from cuvis_ai_schemas.training.data import DataSplitConfig, Selector, SelectorKind

# -- helpers -------------------------------------------------------------------


def _make_cu3s_folder(tmp_path, names=("a", "b", "c"), sub=""):
    folder = tmp_path / "session_dir" / sub if sub else tmp_path / "session_dir"
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / f"{name}.cu3s").write_bytes(b"")
    return folder


def _opened_sessions():
    """Every path the patched SDK was asked to open, in order."""
    return [call.args[0] for call in sys.modules["cuvis"].SessionFile.call_args_list]


def _processing_contexts():
    return sys.modules["cuvis"].ProcessingContext.call_count


def _fi(source, ids):
    return Selector(kind=SelectorKind.FILE_INDICES, source=str(source), ids=ids)


def _files(*paths):
    return Selector(kind=SelectorKind.FILES, paths=[str(p) for p in paths])


def _canonical(path):
    return Path(path).resolve().as_posix()


# -- explicit_sources: a pure read of the selector tree ------------------------


def test_explicit_sources_collects_both_naming_kinds_across_stages():
    splits = DataSplitConfig(
        train=[_fi("a.cu3s", [0, 1])],
        val=[_files("b.cu3s")],
        test=[_fi("c.cu3s", [2])],
        predict=[_files("d.cu3s", "e.cu3s")],
    )
    got = explicit_sources(splits)
    assert got == ExplicitSources(
        required=frozenset({"a.cu3s", "b.cu3s", "c.cu3s", "d.cu3s", "e.cu3s"}),
        optional=frozenset(),
    )
    assert got.named == got.required


def test_explicit_sources_recurses_set_ops_as_optional():
    """A set operation's operands need not exist, so they are collected as optional.

    core resolves each child of a union/except/intersect without its zero-match check, so
    ``except(files[a], files[gone])`` is a legitimate split. Requiring every mentioned
    source to exist would break it.
    """
    splits = DataSplitConfig(
        train=[
            Selector(
                kind=SelectorKind.EXCEPT,
                of=[_files("a.cu3s"), _files("gone.cu3s")],
            )
        ],
        val=[
            Selector(
                kind=SelectorKind.UNION,
                of=[_fi("b.cu3s", [0]), _fi("c.cu3s", [1])],
            )
        ],
    )
    got = explicit_sources(splits)
    assert got.required == frozenset()
    assert got.optional == frozenset({"a.cu3s", "gone.cu3s", "b.cu3s", "c.cu3s"})


def test_explicit_sources_prefers_required_when_a_source_appears_both_ways():
    splits = DataSplitConfig(
        train=[_fi("a.cu3s", [0])],
        val=[Selector(kind=SelectorKind.EXCEPT, of=[_files("a.cu3s"), _files("b.cu3s")])],
    )
    got = explicit_sources(splits)
    assert got.required == frozenset({"a.cu3s"})
    assert got.optional == frozenset({"b.cu3s"})  # a.cu3s is not listed twice


@pytest.mark.parametrize(
    "selector",
    [
        Selector(kind=SelectorKind.ALL),
        Selector(kind=SelectorKind.DIR_INDICES, ids=[0, 1]),
        Selector(kind=SelectorKind.STEMS, stems=["a"]),
        Selector(kind=SelectorKind.GLOB, pattern="a*"),
        Selector(kind=SelectorKind.TAG, any_of=["normal"]),
        Selector(kind=SelectorKind.CATEGORIES, any_of=["scrap"]),
    ],
    ids=["all", "dir_indices", "stems", "glob", "tag", "categories"],
)
def test_explicit_sources_gives_up_on_whole_universe_selectors(selector):
    assert explicit_sources(DataSplitConfig(train=[selector])) is None


def test_explicit_sources_gives_up_on_a_whole_universe_selector_nested_in_a_set_op():
    splits = DataSplitConfig(
        train=[
            Selector(
                kind=SelectorKind.EXCEPT,
                of=[_files("a.cu3s"), Selector(kind=SelectorKind.TAG, any_of=["scrap"])],
            )
        ]
    )
    assert explicit_sources(splits) is None


def test_explicit_sources_returns_none_for_no_splits_and_empty_splits():
    assert explicit_sources(None) is None
    assert explicit_sources(DataSplitConfig()) is None


# -- the files param -----------------------------------------------------------


def test_files_opens_only_what_it_names(mock_cuvis_sdk, tmp_path):
    folder = _make_cu3s_folder(tmp_path)  # a, b, c
    named = [str(folder / "a.cu3s"), str(folder / "b.cu3s")]
    dm = Cu3sDataModule(files=named, data_dir=str(folder), frames="measurements")

    refs = dm.enumerate()

    assert {_canonical(p) for p in _opened_sessions()} == {_canonical(p) for p in named}
    assert {r.source for r in refs} == {_canonical(p) for p in named}
    assert len(refs) == 2 * 7  # the mock session reports 7 measurements


def test_files_probe_builds_no_processing_context(mock_cuvis_sdk, tmp_path):
    """Enumeration needs a count, not a processing pipeline."""
    folder = _make_cu3s_folder(tmp_path, names=("a",))
    Cu3sDataModule(
        files=[str(folder / "a.cu3s")], data_dir=str(folder), frames="measurements"
    ).enumerate()
    assert _processing_contexts() == 0


def test_files_wins_over_the_folder_walk(mock_cuvis_sdk, tmp_path):
    folder = _make_cu3s_folder(tmp_path)  # a, b, c all present in the folder
    dm = Cu3sDataModule(files=[str(folder / "c.cu3s")], data_dir=str(folder), frames="measurements")
    assert [Path(p).name for p in dm._folder_files()] == ["c.cu3s"]


def test_empty_files_list_falls_back_to_the_folder(mock_cuvis_sdk, tmp_path):
    """A trainrun preset can ship ``files: []`` as a placeholder without changing behaviour."""
    folder = _make_cu3s_folder(tmp_path)
    dm = Cu3sDataModule(files=[], data_dir=str(folder), frames="measurements")
    assert dm.files is None
    assert [Path(p).name for p in dm._folder_files()] == ["a.cu3s", "b.cu3s", "c.cu3s"]


def test_files_accepts_a_comma_string(mock_cuvis_sdk, tmp_path):
    """``restore-trainrun --data-arg files=a.cu3s,b.cu3s`` arrives as one string."""
    folder = _make_cu3s_folder(tmp_path)
    spec = f"{folder / 'a.cu3s'}, {folder / 'b.cu3s'}"
    dm = Cu3sDataModule(files=spec, data_dir=str(folder), frames="measurements")
    assert [Path(p).name for p in dm._folder_files()] == ["a.cu3s", "b.cu3s"]


def test_files_may_sit_outside_data_dir(mock_cuvis_sdk, tmp_path):
    """No containment rule on this path: a split may legitimately span folders or drives.

    CuvisNEXT sends the deepest folder containing every assigned recording, which for a
    split spanning two drives is a drive root. The caller that hands over the list is the
    authority on it.
    """
    folder = _make_cu3s_folder(tmp_path, names=("a",))
    elsewhere = tmp_path / "other"
    elsewhere.mkdir()
    (elsewhere / "z.cu3s").write_bytes(b"")

    dm = Cu3sDataModule(
        files=[str(folder / "a.cu3s"), str(elsewhere / "z.cu3s")],
        data_dir=str(folder),
        frames="measurements",
    )
    # Ordered by full path (deterministic), so compare the set of names.
    assert {Path(p).name for p in dm._folder_files()} == {"a.cu3s", "z.cu3s"}


def test_two_spellings_of_one_recording_are_opened_once(mock_cuvis_sdk, tmp_path):
    folder = _make_cu3s_folder(tmp_path, names=("a",))
    plain = str(folder / "a.cu3s")
    dm = Cu3sDataModule(
        files=[plain, plain.replace("/", "\\") if "/" in plain else plain.upper()],
        data_dir=str(folder),
        frames="measurements",
    )
    assert len(dm._folder_files()) == 1
    dm.enumerate()
    assert len(set(_opened_sessions())) == 1


def test_the_file_list_is_computed_once(mock_cuvis_sdk, tmp_path):
    folder = _make_cu3s_folder(tmp_path, names=("a", "b"))
    dm = Cu3sDataModule(files=[str(folder / "a.cu3s")], data_dir=str(folder), frames="measurements")
    first = dm._folder_files()
    assert dm._folder_files() is first  # same list object, no second resolution


# -- REGRESSION: narrowing must not change what a ref is -----------------------


def test_narrowed_refs_are_identical_to_the_walked_ones(mock_cuvis_sdk, tmp_path):
    """The refs for a named recording must match what the folder walk produced.

    ``SampleRef.uid`` is what a frozen splits.json resolves against, so if narrowing
    changed a source's spelling, every existing split would select nothing.
    """
    folder = _make_cu3s_folder(tmp_path)
    named = str(folder / "b.cu3s")

    walked = Cu3sDataModule(data_dir=str(folder), frames="measurements").enumerate()
    narrowed = Cu3sDataModule(
        files=[named], data_dir=str(folder), frames="measurements"
    ).enumerate()

    def identity(refs):
        return [(r.source, r.index, r.uid) for r in refs if r.source == _canonical(named)]

    assert identity(narrowed) == identity(walked)
    assert len(narrowed) == 7


def test_frames_file_keeps_a_relative_source_resolvable(mock_cuvis_sdk, tmp_path, monkeypatch):
    """``frames="file"`` must keep emitting the spelling it was given, not a resolved path.

    ``resolve-splits`` writes its selectors from these sources. A relative ``data_dir``
    yields relative sources, and rewriting them to absolute would make every selector it
    ever wrote match nothing.
    """
    _make_cu3s_folder(tmp_path, names=("a",))
    monkeypatch.chdir(tmp_path)
    relative = str(Path("session_dir") / "a.cu3s")

    refs = Cu3sDataModule(files=[relative], data_dir="session_dir").enumerate()

    assert len(refs) == 1
    assert not os.path.isabs(refs[0].source)
    assert refs[0].source == relative  # the round trip a FILES selector depends on


# -- validate_params -----------------------------------------------------------


def test_validate_params_checks_the_list_and_skips_the_folder(tmp_path):
    """With a list in hand, the folder is not consulted at all.

    That is the point: ``data_dir`` may be a filesystem root, and the folder check walks
    (recursively, when asked) until it meets its first recording.
    """
    folder = _make_cu3s_folder(tmp_path, names=("a",))
    Cu3sDataModule.validate_params(
        {
            "files": [str(folder / "a.cu3s")],
            "data_dir": "Z:/does/not/exist",  # never looked at
            "recursive": True,
        }
    )


def test_validate_params_rejects_a_missing_or_misnamed_entry(tmp_path):
    folder = _make_cu3s_folder(tmp_path, names=("a",))
    with pytest.raises(ValueError, match="files entry does not exist"):
        Cu3sDataModule.validate_params({"files": [str(folder / "gone.cu3s")]})
    with pytest.raises(ValueError, match="must end with .cu3s"):
        Cu3sDataModule.validate_params({"files": [str(folder / "a.txt")]})


def test_validate_params_names_files_as_an_option_when_nothing_is_given():
    with pytest.raises(ValueError, match="'files'"):
        Cu3sDataModule.validate_params({})


# -- inference from the split (no files given) ---------------------------------


def test_split_sources_narrow_the_folder(mock_cuvis_sdk, tmp_path):
    """``restore-trainrun`` over a folder opens the split's recordings and no others."""
    folder = _make_cu3s_folder(tmp_path)  # a, b, c
    a, b = _canonical(folder / "a.cu3s"), _canonical(folder / "b.cu3s")
    dm = Cu3sDataModule(
        data_dir=str(folder),
        frames="measurements",
        splits=DataSplitConfig(train=[_fi(a, [0, 1])], val=[_fi(b, [2])]),
    )

    dm.setup(stage="fit")

    opened = {Path(p).name for p in _opened_sessions()}
    assert opened == {"a.cu3s", "b.cu3s"}  # c.cu3s is never touched
    assert len(dm._train_ds) == 2 and len(dm._val_ds) == 1


def test_a_whole_universe_selector_keeps_the_walk(mock_cuvis_sdk, tmp_path):
    folder = _make_cu3s_folder(tmp_path)
    dm = Cu3sDataModule(
        data_dir=str(folder),
        frames="measurements",
        splits=DataSplitConfig(train=[Selector(kind=SelectorKind.DIR_INDICES, ids=[0])]),
    )
    dm.setup(stage="fit")
    assert {Path(p).name for p in _opened_sessions()} == {"a.cu3s", "b.cu3s", "c.cu3s"}


def test_a_missing_required_source_is_named(mock_cuvis_sdk, tmp_path):
    folder = _make_cu3s_folder(tmp_path, names=("a",))
    gone = _canonical(folder / "gone.cu3s")
    dm = Cu3sDataModule(
        data_dir=str(folder),
        frames="measurements",
        splits=DataSplitConfig(train=[_fi(gone, [0])]),
    )
    with pytest.raises(ValueError, match="gone.cu3s"):
        dm.enumerate()


def test_a_required_source_outside_data_dir_is_named_with_the_folder(mock_cuvis_sdk, tmp_path):
    folder = _make_cu3s_folder(tmp_path, names=("a",))
    elsewhere = tmp_path / "other"
    elsewhere.mkdir()
    (elsewhere / "z.cu3s").write_bytes(b"")
    outside = _canonical(elsewhere / "z.cu3s")

    dm = Cu3sDataModule(
        data_dir=str(folder),
        frames="measurements",
        splits=DataSplitConfig(train=[_fi(outside, [0])]),
    )
    with pytest.raises(ValueError, match="outside data_dir"):
        dm.enumerate()


def test_a_set_op_operand_that_is_gone_is_tolerated(mock_cuvis_sdk, tmp_path):
    """``except(files[a], files[gone])`` must still resolve; only the present file opens."""
    folder = _make_cu3s_folder(tmp_path, names=("a",))
    present = _canonical(folder / "a.cu3s")
    gone = _canonical(folder / "gone.cu3s")
    dm = Cu3sDataModule(
        data_dir=str(folder),
        frames="measurements",
        splits=DataSplitConfig(
            train=[
                Selector(
                    kind=SelectorKind.EXCEPT,
                    of=[_files(present), _files(gone)],
                )
            ]
        ),
    )

    dm.setup(stage="fit")

    assert {Path(p).name for p in _opened_sessions()} == {"a.cu3s"}
    assert len(dm._train_ds) == 7


def test_a_frozen_splits_file_narrows_too(mock_cuvis_sdk, tmp_path):
    """The GUI hands over ``splits_path``; the narrowing must see through it."""
    from cuvis_ai_core.data.splits_io import save_splits

    folder = _make_cu3s_folder(tmp_path)
    a = _canonical(folder / "a.cu3s")
    splits_file = tmp_path / "splits.json"
    save_splits(DataSplitConfig(train=[_fi(a, [0, 1])]), splits_file)

    dm = Cu3sDataModule(
        data_dir=str(folder),
        frames="measurements",
        splits=DataSplitConfig(splits_path=str(splits_file)),
    )
    dm.setup(stage="fit")

    assert {Path(p).name for p in _opened_sessions()} == {"a.cu3s"}
    assert len(dm._train_ds) == 2


def test_predict_with_no_predict_selectors_iterates_the_narrowed_universe(mock_cuvis_sdk, tmp_path):
    """Documented consequence: with an explicit split, "the whole universe" is its files.

    core's empty-predict rule serves every sample the module can see. Once the module only
    looks at the recordings the split names, that is what predict iterates.
    """
    folder = _make_cu3s_folder(tmp_path)
    a = _canonical(folder / "a.cu3s")
    dm = Cu3sDataModule(
        data_dir=str(folder),
        frames="measurements",
        splits=DataSplitConfig(train=[_fi(a, [0])]),
    )

    dm.setup(stage="predict")

    assert len(dm._predict_ds) == 7  # a.cu3s only, not 3 x 7
    assert {Path(p).name for p in _opened_sessions()} == {"a.cu3s"}


def test_category_map_reads_a_named_recordings_sidecar(mock_cuvis_sdk, tmp_path):
    """Not the first file the walk happens to find."""
    folder = _make_cu3s_folder(tmp_path)
    (folder / "b.json").write_text("{}")  # only b carries an annotation
    b = _canonical(folder / "b.cu3s")

    dm = Cu3sDataModule(
        data_dir=str(folder),
        frames="measurements",
        splits=DataSplitConfig(train=[_fi(b, [0])]),
    )
    assert dm.category_name_to_id() == {"background": 0, "anomaly": 1}
