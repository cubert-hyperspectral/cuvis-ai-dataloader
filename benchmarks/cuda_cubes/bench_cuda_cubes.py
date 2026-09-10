"""Evidence run for ``cuda_cubes``: what the host round trip costs a training step.

Both arms are measured to the same finish line, a cube sitting on the GPU ready to use. The
host arm therefore pays the SDK's device-to-host copy and then torch's host-to-device copy
back; the device arm pays neither.

``cuvis.cuda.enable`` is process-global and cannot be undone for an already-processed
measurement, so this re-execs itself once per mode and merges the results.

    python bench_cuda_cubes.py <cu3s> <out_dir> [mode] [iters] [repeats]
"""

import json
import os
import statistics
import subprocess
import sys
import time
import types

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PACKAGE = os.path.join(REPO, "cuvis_ai_dataloader")
for name, path in (
    ("cuvis_ai_dataloader", PACKAGE),
    ("cuvis_ai_dataloader.data", os.path.join(PACKAGE, "data")),
    ("cuvis_ai_dataloader.data.readers", os.path.join(PACKAGE, "data", "readers")),
):
    module = types.ModuleType(name)
    module.__path__ = [path]
    sys.modules[name] = module

from cuvis_ai_dataloader.data.readers.cu3s_pool import open_reader  # noqa: E402

CU3S = sys.argv[1]
OUT = sys.argv[2]
MODE = sys.argv[3] if len(sys.argv) > 3 else "Raw"
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 24
REPEATS = int(sys.argv[5]) if len(sys.argv) > 5 else 3
CUBES = sys.argv[6] if len(sys.argv) > 6 else None

WINDOW = 10
THREADS = (1, 2, 4, 8)
ORDER = [(i * 3) % WINDOW for i in range(ITERS)]


def to_gpu(item):
    """A cube on the GPU, which is where a training step needs it either way."""
    cube = item["cube"]
    return cube if getattr(cube, "is_cuda", False) else torch.as_tensor(np.asarray(cube)).cuda()


def measure(cuda_cubes):
    """Median cubes/s per thread count, all the way to a GPU-resident cube."""
    rows = []
    for threads in THREADS:
        reader = open_reader(
            CU3S, read_threads=threads, processing_mode=MODE, cuda_cubes=cuda_cubes
        )
        try:
            if cuda_cubes and not reader.cuda_cubes:
                raise SystemExit("device-resident cubes unavailable here; nothing to compare")
            for item in reader.read_many(ORDER[:threads]):  # warm up
                del item
            rates = []
            for _ in range(REPEATS):
                started = time.perf_counter()
                for item in reader.read_many(ORDER):
                    tensor = to_gpu(item)
                    del tensor, item
                torch.cuda.synchronize()
                rates.append(ITERS / (time.perf_counter() - started))
            rows.append(
                {
                    "cuda_cubes": cuda_cubes,
                    "threads": threads,
                    "fps": round(statistics.median(rates), 2),
                    "fps_runs": [round(r, 2) for r in rates],
                }
            )
        finally:
            reader.close()
        print(rows[-1], flush=True)
    return rows


if CUBES is not None:
    print("RESULT" + json.dumps(measure(CUBES == "on")))
    raise SystemExit(0)

os.makedirs(OUT, exist_ok=True)
rows = []
for cubes in ("off", "on"):
    done = subprocess.run(
        [
            sys.executable,
            os.path.abspath(__file__),
            CU3S,
            OUT,
            MODE,
            str(ITERS),
            str(REPEATS),
            cubes,
        ],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if done.returncode != 0:
        raise SystemExit(f"cuda_cubes={cubes} child failed:\n{done.stderr[-3000:]}")
    line = next(ln for ln in done.stdout.splitlines() if ln.startswith("RESULT"))
    rows.extend(json.loads(line[len("RESULT") :]))

import cuvis  # noqa: E402  (imported late: the parent never initializes the SDK)

result = {
    "session": CU3S,
    "mode": MODE,
    "iters": ITERS,
    "repeats": REPEATS,
    "window": WINDOW,
    "sdk_version": cuvis.version(),
    "cuvis_wrapper_version": cuvis.General.wrapper_version(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "rows": rows,
}
with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as fh:
    json.dump(result, fh, indent=2)
print("wrote", os.path.join(OUT, "results.json"))
