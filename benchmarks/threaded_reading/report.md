# Threaded cu3s reading: `read_threads` scaling and the `batch_size` ceiling

Session: `D:\Measurements\hbf\Auto_013+01.cu3s`, 940 frames, cube 1000x1080x61, `Raw` mode.
Machine: 20 cores, RTX 4070 (8 GB), Windows 11, native CUBERT SDK 3.5.3.
Binding: locally built `cuvis.pyil` carrying the GIL release (`cuvis.swig` `0ccf8ca`), **not** a published wheel.
The probe in `data/_extras.py` confirmed the binding releases the GIL before any number below was taken.

Topology under test: N `cuvis.SessionFile` handles on one file, all sharing a single `cuvis.ProcessingContext`.
Each cell reads 24 frames over a rotating 10-frame window through `Cu3sPrefetchReader.read_many`, after a warm-up of `read_threads` frames.
Correctness is checked per frame with blake2b against a single-threaded reference, and the hashing happens outside the timed region.

Raw numbers: [`results.json`](results.json).

## Thread scaling

| read_threads | fps | scaling | setup s | VRAM MiB | RSS GB | wrong cubes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 12.96 | 1.00x | 8.16 | 1939 | 1.15 | 0 |
| 2 | 27.90 | 2.15x | 7.73 | 2024 | 2.93 | 0 |
| 4 | 38.22 | 2.95x | 7.95 | 2568 | 3.57 | 0 |
| 6 | 39.29 | 3.03x | 7.89 | 2690 | 4.13 | 0 |
| 8 | **43.05** | **3.32x** | 7.86 | 2855 | 4.70 | 0 |
| 12 | 38.73 | 2.99x | 7.80 | 2950 | 5.63 | 0 |
| 16 | 40.73 | 3.14x | 8.17 | 3143 | 6.69 | 0 |

![chart](chart.png)

## The `batch_size` ceiling

`read_threads` fixed at 8, varying how many indices are handed over per call, which is exactly what a torch `DataLoader`'s `batch_size` controls.

| batch_size | fps | scaling |
| --- | --- | --- |
| 1 | 15.08 | 1.16x |
| 2 | 22.55 | 1.74x |
| 4 | 29.87 | 2.31x |
| 6 | 35.41 | 2.73x |
| 8 | 39.55 | 3.05x |

![batch size](batch_size.png)

## Findings

Robust:

- **Threading works, and the win is large.** 12.96 to 43.05 fps, 3.32x, on the same hardware and the same session.
  Nothing in the Python layer changed except that reads now overlap.
- **Zero wrong cubes at every thread count**, 1 through 16, verified per frame.
  This is the claim that mattered most: a shared `ProcessingContext` could plausibly have mixed state between concurrent callers, and it does not.
- **The context is built once, not per handle.** Setup stays inside 7.73-8.17 s at every thread count.
  Per-handle construction would have cost roughly `threads x 8 s`, which is the dashed line in the bottom-left panel.
- **VRAM grows with handles, not with contexts.** 1939 MiB at one handle to 3143 MiB at sixteen.
  Sixteen private contexts would not have fit on an 8 GB card at all, which is what makes the shared-context topology the enabling choice rather than an optimisation.
- **Host memory is the cost that does scale**, at roughly 0.35 GB per handle: 1.15 GB at one, 6.69 GB at sixteen.
  This is the number to budget with.
- **Concurrency is bounded by `batch_size`, not by `read_threads`.** With eight handles open, feeding one index at a time yields 15.08 fps against a 13.0 fps baseline, i.e. essentially nothing.
  torch hands a map-style dataset a whole batch of indices and nothing earlier, so there is no other moment at which the threads can be used.

Do not over-read:

- **The ceiling here is about 43 fps, and 8 threads reaches it.** 12 and 16 threads do not improve on 8 (38.73 and 40.73 fps), so they only cost memory.
  Treat 4 to 8 as the useful range on this machine, and re-measure elsewhere rather than assuming.
- **This is a single run per cell, not a median of repeats.** The 12-thread dip below the 8-thread figure is within the run-to-run spread seen across three separate sessions of this benchmark, where the single-handle baseline alone moved between 12.8 and 14.6 fps.
  The shape of the curve is reliable; individual cells to two significant figures are not.
- **`Raw` mode only.** `Reflectance` and `SpectralRadiance` hold materially more VRAM per handle and will hit the card sooner, so the thread ceiling there is lower.
- **A 10-frame window means the OS file cache is warm.** This isolates the SDK read and processing path rather than measuring cold disk throughput, which is the right target here but is not what a full shuffled epoch over 940 frames would see.

## Reproducing

```powershell
$env:PYTHONPATH = "C:\dev\cuvis_sdk\cuvis.pyil;C:\dev\cuvis_sdk\cuvis.python-await"
& C:\dev\cuvis_sdk\.venv-pyil312\Scripts\python.exe bench_evidence.py `
    "D:\Measurements\hbf\Auto_013+01.cu3s" benchmarks\threaded_reading Raw 24
& C:\dev\cuvis_sdk\.venv-pyil312\Scripts\python.exe make_charts.py benchmarks\threaded_reading
```

Both scripts import only `data/readers/` and `data/_extras.py`, bypassing `data/__init__.py`, because that pulls the DataModules and therefore `pytorch_lightning`, which the SDK venv does not carry.
On a stock binding the probe rejects the pool and every cell collapses onto the single-handle row, which is the intended behaviour and also the reason CI can never measure this.
