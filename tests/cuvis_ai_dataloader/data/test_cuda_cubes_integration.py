"""Real-SDK checks for ``cuda_cubes``: a device tensor, the same numbers, bounded VRAM.

``cuvis.cuda.enable`` is process-global and cannot be undone for a measurement already
processed, so the device and host reads run in separate interpreters.

Gated and skipped by default (CI has neither the SDK nor sample data)::

    export CUVIS_AI_IT_TARGET=/path/to/scene.cu3s   # needs >= 12 measurements
    # optional: CUVIS_AI_IT_MODE (default Raw, avoids needing references)
    pytest -m integration \
        tests/cuvis_ai_dataloader/data/test_cuda_cubes_integration.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

_TARGET = os.environ.get("CUVIS_AI_IT_TARGET")
_MODE = os.environ.get("CUVIS_AI_IT_MODE", "Raw")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _TARGET,
        reason="set CUVIS_AI_IT_TARGET (needs the real cuvis SDK + .cu3s data)",
    ),
]

_CHILD = textwrap.dedent(
    """
    import json, subprocess, sys
    import numpy as np
    import torch
    from cuvis_ai_dataloader.data.readers.cu3s_pool import open_reader

    target, mode, on, out = sys.argv[1], sys.argv[2], sys.argv[3] == "on", sys.argv[4]

    def vram():
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        return int(done.stdout.strip().splitlines()[0])

    with open_reader(target, processing_mode=mode, cuda_cubes=on) as reader:
        enabled = bool(reader.cuda_cubes)
        cube = reader.read(1)["cube"]
        is_cuda = bool(getattr(cube, "is_cuda", False))
        np.save(out, cube.cpu().numpy() if is_cuda else np.asarray(cube))
        del cube
        # 30 reads never held at once: a buffer that is not returned to the SDK's pool
        # shows up here as steady growth.
        settled = vram()
        for index in range(2, 32):
            held = reader.read(index)["cube"]
            del held
        growth = vram() - settled
    print("RESULT" + json.dumps(
        {"enabled": enabled, "is_cuda": is_cuda, "vram_growth_mib": growth}))
    """
)


def _run(tmp_path, mode: str) -> dict:
    """Read in a fresh process with cuda_cubes on or off; return its report and cube."""
    out = tmp_path / f"{mode}.npy"
    done = subprocess.run(
        [sys.executable, "-c", _CHILD, _TARGET, _MODE, mode, str(out)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert done.returncode == 0, f"cuda_cubes={mode} child failed:\n{done.stderr[-2000:]}"
    line = next(ln for ln in done.stdout.splitlines() if ln.startswith("RESULT"))
    return {**json.loads(line[len("RESULT") :]), "cube": np.load(out)}


@pytest.fixture(scope="module")
def modes(tmp_path_factory):
    """One read per mode, so both tests below pay the ~9 s LUT build once between them."""
    tmp_path = tmp_path_factory.mktemp("cuda_cubes")
    return {mode: _run(tmp_path, mode) for mode in ("off", "on")}


def test_the_cube_arrives_on_the_device(modes):
    """The point of the flag: no host round trip, so the cube is already a CUDA tensor."""
    assert modes["on"]["enabled"] is True, "device mode was requested but fell back"
    assert modes["on"]["is_cuda"] is True
    assert modes["off"]["is_cuda"] is False


def test_the_device_cube_is_the_same_cube(modes):
    """Keeping the cube on the GPU must not change it; it is the same buffer, not a re-read."""
    on, off = modes["on"]["cube"], modes["off"]["cube"]
    assert (on.dtype, on.shape) == (off.dtype, off.shape)
    assert np.array_equal(on, off)


def test_device_buffers_are_returned_to_the_sdk(modes):
    """Each cube is ~66 MB of VRAM; 30 unreturned would be ~2 GB and an OOM soon after.

    The SDK pools the buffers rather than releasing them to the driver, so the check is that
    a steady stream of reads plateaus, not that VRAM returns to where it started.
    """
    growth = modes["on"]["vram_growth_mib"]
    assert growth < 256, f"VRAM grew {growth} MiB over 30 reads; buffers are not being freed"
