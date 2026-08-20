"""Evidence run for the threaded cu3s reader: thread scaling + the batch-size ceiling.

Imports only the reader modules, bypassing cuvis_ai_dataloader.data.__init__ (which pulls the
DataModules and therefore pytorch_lightning, absent from the SDK venv).

    PYTHONPATH="<pyil>;<cuvis.python>" python bench_evidence.py <cu3s> <out_dir> [mode] [iters]
"""

import hashlib
import json
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

import psutil  # noqa: E402

from cuvis_ai_dataloader.data._extras import cuvis_releases_gil  # noqa: E402
from cuvis_ai_dataloader.data.readers.cu3s_pool import Cu3sPrefetchReader  # noqa: E402
from cuvis_ai_dataloader.data.readers.cu3s_reader import (  # noqa: E402
    Cu3sCubeReader,
    total_measurements_of,
)

CU3S = sys.argv[1]
OUT = sys.argv[2]
MODE = sys.argv[3] if len(sys.argv) > 3 else "Raw"
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 24
WINDOW = 10
THREADS = (2, 4, 6, 8, 12, 16)
BATCHES = (1, 2, 4, 6, 8)

os.makedirs(OUT, exist_ok=True)


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


VRAM0 = vram_mib()
order = [(i * 3) % WINDOW for i in range(ITERS)]
result = {
    "session": CU3S,
    "frames_in_session": total_measurements_of(CU3S),
    "mode": MODE,
    "iters": ITERS,
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

t = time.perf_counter()
for i in order:
    plain.read(i)
span = time.perf_counter() - t
base_fps = ITERS / span
result["scaling"].append({
    "threads": 1, "fps": round(base_fps, 2), "scaling": 1.0, "setup_s": setup_1,
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
        t = time.perf_counter()
        items = reader.read_many(order)
        span = time.perf_counter() - t
        wrong = sum(1 for i, item in zip(order, items) if digest(item["cube"]) != reference[i])
        row = {
            "threads": threads, "fps": round(ITERS / span, 2),
            "scaling": round((ITERS / span) / base_fps, 2), "setup_s": setup_s,
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
        t = time.perf_counter()
        done = sum(len(reader.read_many(chunk)) for chunk in chunks)
        span = time.perf_counter() - t
        row = {"batch_size": batch, "read_threads": 8, "fps": round(done / span, 2),
               "scaling": round((done / span) / base_fps, 2)}
        result["batch_size"].append(row)
        print(row)
finally:
    reader.close()

with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as fh:
    json.dump(result, fh, indent=2)
print("wrote", os.path.join(OUT, "results.json"))
