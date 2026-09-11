"""Evidence run for the threaded cu3s reader: thread scaling + the batch-size ceiling.

Imports only the reader modules, bypassing cuvis_ai_dataloader.data.__init__ (which pulls the
DataModules and therefore pytorch_lightning, absent from a bare SDK venv).

    python bench_evidence.py <cu3s> <out_dir> [mode] [iters] [gpu_mode] [repeats]
"""

import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import types

REPO = r"C:\dev\cuvis_ai\cuvis-ai-dataloader\cuvis_ai_dataloader"
for name, path in (
    ("cuvis_ai_dataloader", REPO),
    ("cuvis_ai_dataloader.data", os.path.join(REPO, "data")),
    ("cuvis_ai_dataloader.data.readers", os.path.join(REPO, "data", "readers")),
):
    module = types.ModuleType(name)
    module.__path__ = [path]
    sys.modules[name] = module

import cuvis  # noqa: E402
import psutil  # noqa: E402

from cuvis_ai_dataloader.data._extras import cuvis_releases_gil  # noqa: E402
from cuvis_ai_dataloader.data.readers.cu3s_pool import Cu3sPrefetchReader  # noqa: E402
from cuvis_ai_dataloader.data.readers.cu3s_reader import (  # noqa: E402
    Cu3sCubeReader,
    count_measurements,
)

CU3S = sys.argv[1]
OUT = sys.argv[2]
MODE = sys.argv[3] if len(sys.argv) > 3 else "Raw"
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 24
GPU_MODE = sys.argv[5] if len(sys.argv) > 5 else "cuda"
REPEATS = int(sys.argv[6]) if len(sys.argv) > 6 else 3
WINDOW = 10
THREADS = (2, 4, 6, 8, 12, 16)
BATCHES = (1, 2, 4, 6, 8)

os.makedirs(OUT, exist_ok=True)

# SDK 3.6.0 processes on the host unless a process calls init() with force_gpu_mode=cuda, and
# host mode is ~4.5x slower per cube, which would drown the effect being measured here. The
# library itself does not do this yet.
cuvis.init(cuvis.SdkSettings(force_gpu_mode=GPU_MODE), global_loglevel=logging.WARNING)


def vram_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()[0]
        return int(out)
    except Exception:
        return -1


def rss_gb():
    return round(psutil.Process().memory_info().rss / 2**30, 2)


def digest(arr):
    return hashlib.blake2b(arr.tobytes(), digest_size=8).hexdigest()


def timed_fps(read_batch, chunks):
    """Frames per second for one pass over ``chunks``, plus the items it produced."""
    started = time.perf_counter()
    items = [item for chunk in chunks for item in read_batch(chunk)]
    return len(items) / (time.perf_counter() - started), items


def repeated_fps(read_batch, chunks):
    """Median fps over ``REPEATS`` passes, and the items from the last one.

    A single pass per cell is not separable from run-to-run spread on this machine: the
    curve's shape survives it but individual cells do not, so every cell is a median.
    """
    runs = [timed_fps(read_batch, chunks) for _ in range(REPEATS)]
    rates = sorted(rate for rate, _ in runs)
    return rates[len(rates) // 2], rates, runs[-1][1]


VRAM0 = vram_mib()
order = [(i * 3) % WINDOW for i in range(ITERS)]
result = {
    "session": CU3S,
    "frames_in_session": count_measurements(CU3S),
    "mode": MODE,
    "gpu_mode": GPU_MODE,
    "sdk_version": cuvis.version(),
    "cuvis_wrapper_version": cuvis.General.wrapper_version(),
    "iters": ITERS,
    "repeats": REPEATS,
    "window": WINDOW,
    "vram_baseline_mib": VRAM0,
    "scaling": [],
    "batch_size": [],
}

# ---------------------------------------------------------------- baseline, single handle
t = time.perf_counter()
plain = Cu3sCubeReader(CU3S, processing_mode=MODE)
setup_1 = round(time.perf_counter() - t, 2)
reference = {i: digest(plain.read(i)["cube"]) for i in range(WINDOW)}
result["releases_gil"] = bool(cuvis_releases_gil(lambda: plain.read(0)))

base_fps, base_runs, _ = repeated_fps(plain.read_many, [order])
result["scaling"].append({
    "threads": 1, "fps": round(base_fps, 2), "scaling": 1.0, "setup_s": setup_1,
    "fps_runs": [round(r, 2) for r in base_runs],
    "vram_mib": vram_mib() - VRAM0, "rss_gb": rss_gb(), "wrong_cubes": 0,
})
plain.close()
print(f"releases_gil={result['releases_gil']}  threads=1 {base_fps:.2f} fps setup {setup_1}s")

# ---------------------------------------------------------------------- thread scaling
for threads in THREADS:
    t = time.perf_counter()
    reader = Cu3sPrefetchReader(CU3S, threads=threads, processing_mode=MODE)
    setup_s = round(time.perf_counter() - t, 2)
    try:
        reader.read_many(order[:threads])  # warm up the pool
        fps, runs, items = repeated_fps(reader.read_many, [order])
        wrong = sum(1 for i, item in zip(order, items) if digest(item["cube"]) != reference[i])
        row = {
            "threads": threads, "fps": round(fps, 2),
            "scaling": round(fps / base_fps, 2), "setup_s": setup_s,
            "fps_runs": [round(r, 2) for r in runs],
            "vram_mib": vram_mib() - VRAM0, "rss_gb": rss_gb(), "wrong_cubes": wrong,
        }
    except Exception as exc:
        row = {"threads": threads, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        reader.close()
    result["scaling"].append(row)
    print(row)

# --------------------------------------------- the batch-size ceiling, at a fixed budget
reader = Cu3sPrefetchReader(CU3S, threads=8, processing_mode=MODE)
try:
    reader.read_many(order[:8])
    for batch in BATCHES:
        chunks = [order[i : i + batch] for i in range(0, len(order), batch)]
        fps, runs, _ = repeated_fps(reader.read_many, chunks)
        row = {"batch_size": batch, "read_threads": 8, "fps": round(fps, 2),
               "scaling": round(fps / base_fps, 2), "fps_runs": [round(r, 2) for r in runs]}
        result["batch_size"].append(row)
        print(row)
finally:
    reader.close()

with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as fh:
    json.dump(result, fh, indent=2)
print("wrote", os.path.join(OUT, "results.json"))
