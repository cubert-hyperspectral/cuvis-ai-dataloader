"""Real-SDK integration parity for reference frame selection (``path:N`` / ``path:-1``).

These tests exercise :class:`Cu3sCubeReader`'s white/dark reference override against the
**real** cuvis SDK and real ``.cu3s`` data, complementing the mocked unit tests in
``test_custom_references.py``. They confirm that a frame spec resolves to the intended
reference measurement and produces output byte-identical to selecting that same reference
by hand with ``ProcessingContext.set_reference`` -- the SDK primitives that
``cuvis_batch_exporter``'s ``--force_white``/``--force_dark`` wrap.

They are gated and skipped by default (CI has neither the SDK nor sample data). To run,
install the machine-level cuvis SDK plus this package's ``[cu3s]`` extra and point the env
vars at your own sessions::

    export CUVIS_AI_IT_TARGET=/path/to/scene.cu3s   # session to process (carries a cube)
    export CUVIS_AI_IT_REF=/path/to/multi.cu3s      # reference source, >= 2 measurements
    export CUVIS_AI_IT_DARK=/path/to/dark.cu3s       # a distinct session used as the Dark ref
    # optional: CUVIS_AI_IT_TARGET_IDX (default 0), CUVIS_AI_IT_REF_FRAME (default len//2)
    pytest -m integration \
        tests/cuvis_ai_dataloader/data/test_reference_selection_integration.py

The ``:-1`` tests additionally need ``CUVIS_AI_IT_TARGET`` to carry a baked reference; they
skip themselves when it does not.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from cuvis_ai_dataloader.data.readers.cu3s_reader import Cu3sCubeReader

_REQUIRED = ("CUVIS_AI_IT_TARGET", "CUVIS_AI_IT_REF", "CUVIS_AI_IT_DARK")
_missing = [name for name in _REQUIRED if not os.environ.get(name)]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        bool(_missing),
        reason="set " + ", ".join(_REQUIRED) + " (needs the real cuvis SDK + .cu3s data)",
    ),
]


@pytest.fixture(scope="module")
def env():
    """Open the configured sessions once and resolve the frame indices to use."""
    cuvis = pytest.importorskip("cuvis")
    target = os.environ["CUVIS_AI_IT_TARGET"]
    ref = os.environ["CUVIS_AI_IT_REF"]
    dark = os.environ["CUVIS_AI_IT_DARK"]

    ref_len = len(cuvis.SessionFile(ref))
    if ref_len < 2:
        pytest.skip("CUVIS_AI_IT_REF must hold >= 2 measurements for the :N tests")
    target_len = len(cuvis.SessionFile(target))

    n = int(os.environ.get("CUVIS_AI_IT_REF_FRAME", ref_len // 2))
    n = max(1, min(n, ref_len - 1))
    idx = int(os.environ.get("CUVIS_AI_IT_TARGET_IDX", 0))
    idx = max(0, min(idx, target_len - 1))

    return SimpleNamespace(
        cuvis=cuvis,
        rt=cuvis.ReferenceType,
        target=target,
        ref=ref,
        dark=dark,
        n=n,
        idx=idx,
    )


def _loader_cube(env, white_ref, dark_ref):
    """Reflectance cube for the target frame, via the reader's reference override."""
    reader = Cu3sCubeReader(
        env.target, processing_mode="Reflectance", white_ref=white_ref, dark_ref=dark_ref
    )
    try:
        return np.ascontiguousarray(reader.read(env.idx)["cube"])
    finally:
        reader.close()


def _manual_cube(env, white_mesu, dark_mesu):
    """Reflectance cube for the target frame, via hand-set references (ground truth)."""
    session = env.cuvis.SessionFile(env.target)
    pc = env.cuvis.ProcessingContext(session)
    pc.set_reference(white_mesu, env.rt.White)
    pc.set_reference(dark_mesu, env.rt.Dark)
    pc.processing_mode = env.cuvis.ProcessingMode.Reflectance
    applied = pc.apply(session.get_measurement(env.idx))
    return np.ascontiguousarray(applied.cube.array)


def _equal(a, b):
    """Exact array equality (NaN-aware for float cubes)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if np.issubdtype(a.dtype, np.floating):
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


def _embedded_white(env):
    """The target session's baked White reference, or ``None`` if it has none."""
    session = env.cuvis.SessionFile(env.target)
    try:
        return env.cuvis.ProcessingContext(session).get_reference(env.rt.White)
    except Exception:
        return None


def test_bare_path_equals_frame_zero(env):
    """A bare path and an explicit ``:0`` select the same (first) measurement."""
    bare = _loader_cube(env, env.ref, env.dark)
    explicit = _loader_cube(env, f"{env.ref}:0", env.dark)
    assert _equal(bare, explicit)


def test_frame_index_matches_manual_get_measurement(env):
    """``ref:N`` reproduces a hand-set ``get_measurement(N)`` reference byte-for-byte."""
    got = _loader_cube(env, f"{env.ref}:{env.n}", f"{env.dark}:0")
    white = env.cuvis.SessionFile(env.ref).get_measurement(env.n)
    dark = env.cuvis.SessionFile(env.dark).get_measurement(0)
    assert _equal(got, _manual_cube(env, white, dark))


def test_frame_index_changes_reference(env):
    """Frame ``0`` and frame ``N`` of a multi-measurement source yield different output."""
    frame0 = _loader_cube(env, f"{env.ref}:0", f"{env.dark}:0")
    frame_n = _loader_cube(env, f"{env.ref}:{env.n}", f"{env.dark}:0")
    assert not _equal(frame0, frame_n)


def test_embedded_reference_matches_manual(env):
    """``target:-1`` reproduces the baked reference from ``get_reference`` byte-for-byte."""
    white = _embedded_white(env)
    if white is None:
        pytest.skip("CUVIS_AI_IT_TARGET has no embedded White reference (needed for :-1)")
    got = _loader_cube(env, f"{env.target}:-1", f"{env.dark}:0")
    dark = env.cuvis.SessionFile(env.dark).get_measurement(0)
    assert _equal(got, _manual_cube(env, white, dark))


def test_embedded_reference_distinct_from_frame_zero(env):
    """The ``:-1`` (embedded) path differs from ``:0`` (first measurement)."""
    if _embedded_white(env) is None:
        pytest.skip("CUVIS_AI_IT_TARGET has no embedded White reference (needed for :-1)")
    embedded = _loader_cube(env, f"{env.target}:-1", f"{env.dark}:0")
    frame0 = _loader_cube(env, f"{env.target}:0", f"{env.dark}:0")
    assert not _equal(embedded, frame0)


def test_out_of_range_frame_raises(env):
    """An out-of-range frame index fails loudly with a ValueError."""
    bad = len(env.cuvis.SessionFile(env.dark)) + 10
    with pytest.raises(ValueError):
        _loader_cube(env, f"{env.dark}:{bad}", f"{env.dark}:0")
