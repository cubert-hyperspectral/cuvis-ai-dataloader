"""Tests for the custom white/dark reference override (reader, converter, CLI).

Covers the re-referencing path end to end against the mocked SDK: the reader installs
custom references via ``get_measurement(0)`` (never ``get_reference``) and BEFORE the
processing mode; the Reflectance validation counts custom references; the converter and
the ``cu3s-to-npz`` CLI thread the paths through; and the no-refs path stays untouched.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

# Imported at module top (collection time) on purpose: mock_cuvis_sdk patches
# sys.modules with patch.dict, whose teardown evicts any module first imported
# inside the patched block — torch (and its lazily-imported submodules) cannot
# survive an evict-and-reimport, so preload the full import graph the fixture
# and the tests touch, like the sibling cu3s test modules do.
import torch  # noqa: F401  (import-order guard, see comment)

import cuvis_ai_dataloader.data.labelers.coco_labeler  # noqa: F401  (fixture patch target)
from cuvis_ai_dataloader.data.npz_converter import (  # noqa: F401  (import-order guard)
    convert_cu3s,
    convert_cu3s_file,
)
from cuvis_ai_dataloader.data.readers.cu3s_reader import Cu3sCubeReader  # noqa: F401
from cuvis_ai_dataloader.scripts.convert_cu3s_to_npz import cu3s_to_npz_cli  # noqa: F401


def _make_cu3s(tmp_path, name="x.cu3s"):
    path = tmp_path / name
    path.write_bytes(b"")  # exists + .cu3s suffix is all the reader checks
    return str(path)


def _ref_session(mock_cuvis_sdk):
    """A distinct reference session whose measurement 0 is a unique sentinel.

    Its ``get_reference`` is rigged to return an unintended baked reference, so a
    test can prove the reader never consults it when loading a custom reference.
    """
    ref_mesu = Mock(name="ref_measurement_0")
    session = Mock(name="ref_session")
    session.get_measurement = Mock(return_value=ref_mesu)
    session.get_reference = Mock(return_value=Mock(name="unintended_baked_reference"))
    return session, ref_mesu


def _route_sessions(mock_cuvis_sdk, main_path, routes):
    """Make the fake ``cuvis.SessionFile`` return per-path sessions.

    ``routes`` maps path -> session mock; the main cu3s keeps the fixture's session.
    """
    import cuvis  # the fake module patched into sys.modules by the fixture

    table = {str(main_path): mock_cuvis_sdk["session"], **{str(k): v for k, v in routes.items()}}

    def _open(path):
        return table[str(path)]

    cuvis.SessionFile = Mock(side_effect=_open)


# ------------------------------------------------------------------------------- reader
def test_reader_installs_custom_references(mock_cuvis_sdk, tmp_path):

    main = _make_cu3s(tmp_path)
    white_session, white_mesu = _ref_session(mock_cuvis_sdk)
    dark_session, dark_mesu = _ref_session(mock_cuvis_sdk)
    white_path = _make_cu3s(tmp_path, "white.cu3s")
    dark_path = _make_cu3s(tmp_path, "dark.cu3s")
    _route_sessions(mock_cuvis_sdk, main, {white_path: white_session, dark_path: dark_session})

    reader = Cu3sCubeReader(main, white_ref=white_path, dark_ref=dark_path)

    pc = mock_cuvis_sdk["processing_context"]
    pc.set_reference.assert_any_call(white_mesu, "White")
    pc.set_reference.assert_any_call(dark_mesu, "Dark")
    assert pc.set_reference.call_count == 2
    assert reader.custom_references == {"white": white_path, "dark": dark_path}


def test_reader_loads_refs_via_get_measurement_not_get_reference(mock_cuvis_sdk, tmp_path):
    # The known SDK trap: a reference cu3s' own get_reference can return an unintended baked
    # reference. The override must come from get_measurement(0).

    main = _make_cu3s(tmp_path)
    white_session, white_mesu = _ref_session(mock_cuvis_sdk)
    white_path = _make_cu3s(tmp_path, "white.cu3s")
    _route_sessions(mock_cuvis_sdk, main, {white_path: white_session})

    Cu3sCubeReader(main, white_ref=white_path)

    white_session.get_measurement.assert_called_once_with(0)
    white_session.get_reference.assert_not_called()
    installed = mock_cuvis_sdk["processing_context"].set_reference.call_args_list[0].args[0]
    assert installed is white_mesu


def test_reader_without_refs_never_calls_set_reference(mock_cuvis_sdk, tmp_path):
    # The default path must stay byte-for-byte the baked-reference behaviour.

    reader = Cu3sCubeReader(_make_cu3s(tmp_path))

    mock_cuvis_sdk["processing_context"].set_reference.assert_not_called()
    assert reader.custom_references == {}


@pytest.mark.parametrize("kind", ["white", "dark"])
def test_reader_one_sided_override(mock_cuvis_sdk, tmp_path, kind):

    main = _make_cu3s(tmp_path)
    session, mesu = _ref_session(mock_cuvis_sdk)
    ref_path = _make_cu3s(tmp_path, f"{kind}.cu3s")
    _route_sessions(mock_cuvis_sdk, main, {ref_path: session})

    reader = Cu3sCubeReader(main, **{f"{kind}_ref": ref_path})

    pc = mock_cuvis_sdk["processing_context"]
    pc.set_reference.assert_called_once_with(mesu, kind.capitalize())
    assert reader.custom_references == {kind: ref_path}


class _OrderRecordingPC:
    """A ProcessingContext stand-in that records call/assignment order."""

    def __init__(self, order: list[str]):
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "apply", Mock())

    def set_reference(self, *args, **kwargs):
        self.order.append("set_reference")

    def __setattr__(self, name, value):
        if name == "processing_mode":
            self.order.append("processing_mode")
        object.__setattr__(self, name, value)


def test_refs_installed_before_processing_mode(mock_cuvis_sdk, tmp_path):
    # set_reference must precede the processing-mode assignment (and thus any apply),
    # otherwise the SDK would build the Reflectance pipeline against the baked refs.
    import cuvis  # the fake module patched into sys.modules by the fixture

    order: list[str] = []
    cuvis.ProcessingContext = Mock(return_value=_OrderRecordingPC(order))

    main = _make_cu3s(tmp_path)
    white_session, _ = _ref_session(mock_cuvis_sdk)
    white_path = _make_cu3s(tmp_path, "white.cu3s")
    _route_sessions(mock_cuvis_sdk, main, {white_path: white_session})

    Cu3sCubeReader(main, white_ref=white_path)

    assert order == ["set_reference", "processing_mode"]


def test_reflectance_validation_satisfied_by_custom_refs(mock_cuvis_sdk, tmp_path):
    # A session with NO usable baked references must be readable in Reflectance when
    # both references are supplied externally.

    main_session = mock_cuvis_sdk["session"]
    main_session.get_reference = Mock(return_value=None)  # no baked refs at all

    main = _make_cu3s(tmp_path)
    with pytest.raises(ValueError, match="requires both White and Dark"):
        Cu3sCubeReader(main)  # baked-only path still fails
    main_session.get_reference.reset_mock()  # those two lookups belong to the failing path

    white_session, _ = _ref_session(mock_cuvis_sdk)
    dark_session, _ = _ref_session(mock_cuvis_sdk)
    white_path = _make_cu3s(tmp_path, "white.cu3s")
    dark_path = _make_cu3s(tmp_path, "dark.cu3s")
    _route_sessions(mock_cuvis_sdk, main, {white_path: white_session, dark_path: dark_session})

    reader = Cu3sCubeReader(main, white_ref=white_path, dark_ref=dark_path)
    assert reader.custom_references == {"white": white_path, "dark": dark_path}
    main_session.get_reference.assert_not_called()  # short-circuited by the overrides


def test_reader_rejects_missing_or_wrong_suffix_ref(mock_cuvis_sdk, tmp_path):

    main = _make_cu3s(tmp_path)
    with pytest.raises(ValueError, match="white reference cu3s does not exist"):
        Cu3sCubeReader(main, white_ref=str(tmp_path / "nope.cu3s"))

    bad = tmp_path / "white.txt"
    bad.write_bytes(b"")
    with pytest.raises(ValueError, match=r"dark reference must be a \.cu3s"):
        Cu3sCubeReader(main, dark_ref=str(bad))


def test_reader_rejects_empty_ref_session(mock_cuvis_sdk, tmp_path):

    main = _make_cu3s(tmp_path)
    empty_session = Mock(name="empty_ref_session")
    empty_session.get_measurement = Mock(return_value=None)
    white_path = _make_cu3s(tmp_path, "white.cu3s")
    _route_sessions(mock_cuvis_sdk, main, {white_path: empty_session})

    with pytest.raises(ValueError, match="has no measurement 0"):
        Cu3sCubeReader(main, white_ref=white_path)


def test_close_releases_custom_ref_sessions(mock_cuvis_sdk, tmp_path):

    main = _make_cu3s(tmp_path)
    white_session, _ = _ref_session(mock_cuvis_sdk)
    white_path = _make_cu3s(tmp_path, "white.cu3s")
    _route_sessions(mock_cuvis_sdk, main, {white_path: white_session})

    reader = Cu3sCubeReader(main, white_ref=white_path)
    reader.close()
    white_session.close.assert_called()
    assert reader._custom_ref_handles == []
    reader.close()  # idempotent


# ---------------------------------------------------------------------------- converter
def test_convert_cu3s_file_threads_refs_to_reader(mock_cuvis_sdk, tmp_path):

    main = _make_cu3s(tmp_path)
    white_session, white_mesu = _ref_session(mock_cuvis_sdk)
    dark_session, dark_mesu = _ref_session(mock_cuvis_sdk)
    white_path = _make_cu3s(tmp_path, "white.cu3s")
    dark_path = _make_cu3s(tmp_path, "dark.cu3s")
    _route_sessions(mock_cuvis_sdk, main, {white_path: white_session, dark_path: dark_session})

    records = convert_cu3s_file(
        main,
        tmp_path / "out",
        annotation_json=None,
        white_ref=white_path,
        dark_ref=dark_path,
        frame_limit=2,
    )

    pc = mock_cuvis_sdk["processing_context"]
    pc.set_reference.assert_any_call(white_mesu, "White")
    pc.set_reference.assert_any_call(dark_mesu, "Dark")
    assert len(records) == 2
    with np.load(records[0]["path"]) as z:
        assert z["cube"].shape == (*mock_cuvis_sdk["hw"], mock_cuvis_sdk["channels"])


def test_convert_cu3s_batch_applies_refs_per_file(mock_cuvis_sdk, tmp_path):

    a = _make_cu3s(tmp_path, "a.cu3s")
    b = _make_cu3s(tmp_path, "b.cu3s")
    white_session, _ = _ref_session(mock_cuvis_sdk)
    white_path = _make_cu3s(tmp_path, "white.cu3s")

    import cuvis

    table = {
        a: mock_cuvis_sdk["session"],
        b: mock_cuvis_sdk["session"],
        white_path: white_session,
    }
    cuvis.SessionFile = Mock(side_effect=lambda p: table[str(p)])

    convert_cu3s([a, b], tmp_path / "out", annotations=None, white_ref=white_path, frame_limit=1)
    # One White override installed per converted file.
    assert mock_cuvis_sdk["processing_context"].set_reference.call_count == 2


def test_convert_cu3s_file_without_refs_unchanged(mock_cuvis_sdk, tmp_path):

    convert_cu3s_file(_make_cu3s(tmp_path), tmp_path / "out", frame_limit=1)
    mock_cuvis_sdk["processing_context"].set_reference.assert_not_called()


# ---------------------------------------------------------------------------------- CLI
def _run_cli(argv):

    with (
        patch("cuvis_ai_dataloader.data.npz_converter.convert_cu3s", return_value=[]) as conv,
        patch("sys.argv", ["cu3s-to-npz", *argv]),
    ):
        cu3s_to_npz_cli()
    return conv


def test_cli_passes_ref_flags_through(tmp_path):
    main = _make_cu3s(tmp_path)
    conv = _run_cli(
        [
            "--cu3s",
            main,
            "--out-dir",
            str(tmp_path / "out"),
            "--white-ref",
            "w.cu3s",
            "--dark-ref",
            "d.cu3s",
        ]
    )
    kwargs = conv.call_args.kwargs
    assert kwargs["white_ref"] == "w.cu3s"
    assert kwargs["dark_ref"] == "d.cu3s"


def test_cli_defaults_refs_to_none(tmp_path):
    main = _make_cu3s(tmp_path)
    conv = _run_cli(["--cu3s", main, "--out-dir", str(tmp_path / "out")])
    kwargs = conv.call_args.kwargs
    assert kwargs["white_ref"] is None
    assert kwargs["dark_ref"] is None


# ------------------------------------------------------------------------- frame selection
def test_reader_ref_frame_index(mock_cuvis_sdk, tmp_path):
    # `path:N` selects measurement N of the reference session (a session holding several refs).
    main = _make_cu3s(tmp_path)
    white_session, white_mesu = _ref_session(mock_cuvis_sdk)
    white_path = _make_cu3s(tmp_path, "white.cu3s")
    _route_sessions(mock_cuvis_sdk, main, {white_path: white_session})

    reader = Cu3sCubeReader(main, white_ref=f"{white_path}:5")

    white_session.get_measurement.assert_called_once_with(5)
    white_session.get_reference.assert_not_called()
    mock_cuvis_sdk["processing_context"].set_reference.assert_any_call(white_mesu, "White")
    assert reader.custom_references == {"white": f"{white_path}:5"}


def test_reader_ref_embedded_minus1(mock_cuvis_sdk, tmp_path):
    # `path:-1` uses the reference session's own embedded reference (get_reference on a PC),
    # NOT get_measurement -- the one case where consulting get_reference is intentional.
    main = _make_cu3s(tmp_path)
    white_session, _ = _ref_session(mock_cuvis_sdk)
    white_path = _make_cu3s(tmp_path, "white.cu3s")
    _route_sessions(mock_cuvis_sdk, main, {white_path: white_session})
    pc = mock_cuvis_sdk["processing_context"]
    embedded = Mock(name="embedded_white_reference")
    pc.get_reference = Mock(return_value=embedded)

    reader = Cu3sCubeReader(main, white_ref=f"{white_path}:-1")

    pc.get_reference.assert_any_call("White")
    white_session.get_measurement.assert_not_called()
    pc.set_reference.assert_any_call(embedded, "White")
    assert reader.custom_references == {"white": f"{white_path}:-1"}


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("/a/b.cu3s", ("/a/b.cu3s", 0)),
        ("/a/b.cu3s:0", ("/a/b.cu3s", 0)),
        ("/a/b.cu3s:5", ("/a/b.cu3s", 5)),
        ("/a/b.cu3s:-1", ("/a/b.cu3s", -1)),
        ("/a/b.cu3s:notanint", ("/a/b.cu3s:notanint", 0)),
    ],
)
def test_parse_ref_spec(spec, expected):
    from cuvis_ai_dataloader.data.readers.cu3s_reader import _parse_ref_spec

    assert _parse_ref_spec(spec) == expected
