"""Real-SDK parity and effect for ``sdk_cuda``, one device per child process.

The SDK fixes its device at the first ``cuvis.init`` of a process and ignores every later one,
so the two devices cannot be compared in-process however the test is written. Each case runs in
a fresh interpreter that goes through the package's own seam (``configure_cuvis_sdk`` ->
``require_cuvis`` -> ``Cu3sCubeReader``), which is the path a DataLoader worker takes.

Gated and skipped by default (CI has neither the SDK nor sample data)::

    export CUVIS_AI_IT_TARGET=/path/to/scene.cu3s   # needs >= 4 measurements
    # optional: CUVIS_AI_IT_MODE (default Raw, avoids needing references)
    pytest -m integration \
        tests/cuvis_ai_dataloader/data/test_sdk_device_integration.py
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
_FRAME = 3

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _TARGET,
        reason="set CUVIS_AI_IT_TARGET (needs the real cuvis SDK + .cu3s data)",
    ),
]

_CHILD = textwrap.dedent(
    """
    import json, statistics, sys, time
    import numpy as np
    from cuvis_ai_dataloader.data._extras import configure_cuvis_sdk
    from cuvis_ai_dataloader.data.readers.cu3s_reader import Cu3sCubeReader

    target, mode, device, frame, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    configure_cuvis_sdk(cuda=device == "cuda")
    reader = Cu3sCubeReader(target, processing_mode=mode)
    try:
        reader.read(0)  # warm the LUT; its build is not what is being timed
        spans = []
        for index in range(1, 4):
            started = time.perf_counter()
            reader.read(index)
            spans.append(time.perf_counter() - started)
        np.save(out, np.ascontiguousarray(reader.read(frame)["cube"]))
        print("RESULT" + json.dumps({"seconds": statistics.median(spans)}))
    finally:
        reader.close()
    """
)


def _run(tmp_path, device: str) -> dict:
    """Read in a fresh process pinned to ``device``; return its timing and cube."""
    out = tmp_path / f"{device}.npy"
    done = subprocess.run(
        [sys.executable, "-c", _CHILD, _TARGET, _MODE, device, str(_FRAME), str(out)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert done.returncode == 0, f"{device} child failed:\n{done.stderr[-2000:]}"
    line = next(ln for ln in done.stdout.splitlines() if ln.startswith("RESULT"))
    return {**json.loads(line[len("RESULT") :]), "cube": np.load(out)}


@pytest.fixture(scope="module")
def devices(tmp_path_factory):
    """One read per device, so both tests below pay the ~9 s LUT build once between them."""
    tmp_path = tmp_path_factory.mktemp("sdk_device")
    return {device: _run(tmp_path, device) for device in ("host", "cuda")}


def test_the_device_barely_changes_the_cube(devices):
    """Switching the flag is a performance decision, and must stay close to a free one.

    Not bit-identity: the two implementations round the cubalize interpolation differently.
    On the reference session (940 frames, `Raw`, uint8) they disagree by one LSB on 0.0001%
    of elements and by no more than one LSB anywhere, which is noise against sensor noise.
    A wider gap would mean a model trained on one device sees different inputs on the other,
    so it is worth failing on.
    """
    host, cuda = devices["host"]["cube"], devices["cuda"]["cube"]
    assert (host.dtype, host.shape) == (cuda.dtype, cuda.shape)
    difference = np.abs(host.astype(np.int64) - cuda.astype(np.int64))
    assert difference.max() <= 1, f"max abs difference {difference.max()}"
    assert (difference > 0).mean() < 1e-4, f"{(difference > 0).mean():.2%} of elements differ"


@pytest.mark.gpu
def test_the_gpu_is_actually_faster(devices):
    """The flag has to reach the SDK, not just be stored; the only proof is the clock.

    A wide margin is asserted rather than a tight one: the measured gap on the reference
    machine is about 4.8x (287 ms host against 60 ms cuda), and anything above 1.5x is
    already outside what scheduling noise produces between two fresh processes.
    """
    host, cuda = devices["host"]["seconds"], devices["cuda"]["seconds"]
    assert cuda < host / 1.5, f"host {host * 1000:.0f} ms vs cuda {cuda * 1000:.0f} ms"
