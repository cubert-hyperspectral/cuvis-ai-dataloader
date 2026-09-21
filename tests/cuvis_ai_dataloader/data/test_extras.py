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


# --------------------------------------------------------------- the reader options parser
def _options(**overrides):
    base = dict(
        max_open_sessions=4,
        read_threads=0,
        source_coherent_batches=False,
        sdk_cuda=False,
        cuda_cubes=False,
        num_workers=0,
    )
    base.update(overrides)
    return _extras.parse_cu3s_reader_options(**base)


def _warnings_during(fn):
    from loguru import logger

    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        result = fn()
    finally:
        logger.remove(sink)
    return result, messages


def test_read_ahead_is_off_unless_asked_for():
    assert _options().read_ahead == 0


def test_read_ahead_is_parsed_from_a_data_arg_string_and_capped():
    assert _options(read_ahead="2").read_ahead == 2
    with pytest.raises(ValueError, match="read_ahead must be >= 0"):
        _options(read_ahead=-1)
    with pytest.raises(ValueError, match="read_ahead must be <= 8"):
        _options(read_ahead=9)


def test_read_ahead_refuses_worker_processes():
    with pytest.raises(ValueError, match="read_ahead=2 cannot be combined with num_workers=2"):
        _options(read_ahead=2, num_workers=2)


def test_idle_reader_threads_at_batch_one_warn_and_name_read_ahead():
    # CuvisNEXT patches data.batch_size at fill time, so this is a runtime warning, not a
    # yaml-only rule: threads without a batch or a read-ahead cost handles and buy nothing.
    _, messages = _warnings_during(lambda: _options(read_threads=4, batch_size=1))
    assert any("read_threads=4" in m and "read_ahead" in m for m in messages), messages
    for quiet in (
        dict(read_threads=4, batch_size=1, read_ahead=2),
        dict(read_threads=4, batch_size=4),
        dict(read_threads=0, batch_size=1),
    ):
        _, messages = _warnings_during(lambda quiet=quiet: _options(**quiet))
        assert not any("read_ahead" in m for m in messages), (quiet, messages)
