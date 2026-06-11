"""Tests for the lazy-import helpers and string param parsers."""

from __future__ import annotations

import pytest

from cuvis_ai_dataloader.data import _extras


def test_parse_bool():
    assert _extras.parse_bool("true", key="k") is True
    assert _extras.parse_bool("0", key="k") is False
    assert _extras.parse_bool(True, key="k") is True
    with pytest.raises(ValueError, match="expected one of"):
        _extras.parse_bool("maybe", key="k")


def test_parse_int_list():
    assert _extras.parse_int_list("0,2,5", key="k") == [0, 2, 5]
    assert _extras.parse_int_list([1, 2], key="k") == [1, 2]
    assert _extras.parse_int_list("", key="k") == []


def test_parse_float_list():
    assert _extras.parse_float_list("400,410.5", key="k") == [400.0, 410.5]


def test_parse_str_list():
    assert _extras.parse_str_list("a, b ,c", key="k") == ["a", "b", "c"]


def test_require_helpers_return_modules():
    # In the dev env all extras are installed, so these resolve.
    assert _extras.require_tifffile() is not None
    assert _extras.require_skimage_polygon2mask() is not None
    assert _extras.require_pycocotools() is not None
