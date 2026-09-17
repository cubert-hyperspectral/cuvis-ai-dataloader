"""Real-SDK checks for ``cuda_cubes``: a device tensor, the same numbers, bounded VRAM.

``cuvis.cuda.enable`` is process-global and cannot be undone for a measurement already
processed, so the device and host reads run in separate interpreters.

Gated and skipped by default (CI has neither the SDK nor sample data)::

    export CUVIS_AI_IT_TARGET=/path/to/scene.cu3s   # needs >= 4 measurements
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
import torch

from cuvis_ai_dataloader.data.readers.cu3s_pool import open_reader

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
        # shows up here as steady growth. The reads cycle over the frames the file has.
        settled = vram()
        for step in range(30):
            held = reader.read(2 + step % (reader.total_measurements - 2))["cube"]
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


# ---------------------------------------------------------- in-process: threads + retention
def _host_cubes(indices):
    """Cubes through the host path in this process, copied out of the reader."""
    with open_reader(_TARGET, processing_mode=_MODE) as reader:
        return [np.array(reader.read(i)["cube"], copy=True) for i in indices]


def test_threaded_device_cubes_match_the_host_path():
    """The pool and the device path together: every leased handle hands out the same cube the
    host path produces, on the device, in the requested order."""
    reference = _host_cubes(range(4))
    with open_reader(_TARGET, read_threads=4, processing_mode=_MODE, cuda_cubes=True) as reader:
        assert reader.cuda_cubes, "device mode was requested but fell back"
        items = reader.read_many(range(4))
        for expected, item in zip(reference, items):
            assert item["cube"].is_cuda
            assert np.array_equal(item["cube"].cpu().numpy(), expected)


def test_a_retained_device_tensor_survives_later_reads_and_the_reader_closing():
    """DLPack ties the buffer to the tensor: a cube kept from an earlier batch must still hold
    its values after the SDK has produced thirty more and the reader is gone."""
    with open_reader(_TARGET, processing_mode=_MODE, cuda_cubes=True) as reader:
        assert reader.cuda_cubes, "device mode was requested but fell back"
        held = reader.read(1)["cube"]
        snapshot = held.clone()
        for step in range(30):
            reader.read(2 + step % (reader.total_measurements - 2))
    torch.cuda.synchronize()
    assert torch.equal(held, snapshot)
