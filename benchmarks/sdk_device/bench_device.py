"""Evidence run for ``sdk_cuda``: what the SDK's processing device costs, with and without threads.

The SDK fixes its device at the first ``cuvis.init`` of a process and ignores every later one,
so one device per process is not a convenience here, it is the only way to measure both. This
script re-execs itself once per device and merges the two results.

Goes through the package's own seam (``configure_cuvis_sdk`` -> ``require_cuvis``), so what is
measured is the parameter a user sets, not a hand-rolled ``cuvis.init``.

    python bench_device.py <cu3s> <out_dir> [mode] [iters] [repeats]
"""

import json
import os
import statistics
import subprocess
import sys
import time
import types

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

from cuvis_ai_dataloader.data._extras import configure_cuvis_sdk  # noqa: E402
from cuvis_ai_dataloader.data.readers.cu3s_pool import open_reader  # noqa: E402

CU3S = sys.argv[1]
OUT = sys.argv[2]
MODE = sys.argv[3] if len(sys.argv) > 3 else "Raw"
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 24
REPEATS = int(sys.argv[5]) if len(sys.argv) > 5 else 3
DEVICE = sys.argv[6] if len(sys.argv) > 6 else None

WINDOW = 10
THREADS = (1, 2, 4, 8)
ORDER = [(i * 3) % WINDOW for i in range(ITERS)]


def measure(device):
    """Median fps at each thread count, on a process pinned to ``device``."""
    cuda = device == "cuda"
    configure_cuvis_sdk(cuda=cuda)
    rows = []
    for threads in THREADS:
        started = time.perf_counter()
        reader = open_reader(CU3S, read_threads=threads, processing_mode=MODE, sdk_cuda=cuda)
        setup_s = time.perf_counter() - started
        try:
            reader.read_many(ORDER[:threads])  # warm up
            rates = []
            for _ in range(REPEATS):
                started = time.perf_counter()
                reader.read_many(ORDER)
                rates.append(ITERS / (time.perf_counter() - started))
            rows.append(
                {
                    "device": device,
                    "threads": threads,
                    "fps": round(statistics.median(rates), 2),
                    "setup_s": round(setup_s, 2),
                    "fps_runs": [round(r, 2) for r in rates],
                }
            )
        finally:
            reader.close()
        print(rows[-1], flush=True)
    return rows


if DEVICE is not None:
    print("RESULT" + json.dumps(measure(DEVICE)))
    raise SystemExit(0)

os.makedirs(OUT, exist_ok=True)
rows = []
for device in ("host", "cuda"):
    done = subprocess.run(
        [sys.executable, os.path.abspath(__file__), CU3S, OUT, MODE, str(ITERS), str(REPEATS), device],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if done.returncode != 0:
        raise SystemExit(f"{device} child failed:\n{done.stderr[-3000:]}")
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
    "rows": rows,
}
with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as fh:
    json.dump(result, fh, indent=2)
print("wrote", os.path.join(OUT, "results.json"))
