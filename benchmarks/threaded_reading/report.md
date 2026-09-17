# Threaded cu3s reading: `read_threads` scaling and the `batch_size` ceiling

Session: `D:\Measurements\hbf\Auto_013+01.cu3s`, 940 frames, cube 1000x1080x61, `Raw` mode.
Machine: 20 cores, RTX 4070 (8 GB), Windows 11, native CUBERT SDK 3.6.0 (build `987fdcf`).
Binding: the published `cuvis` / `cuvis-il` 3.6.0.0 wheels from PyPI.
The probe in `data/_extras.py` confirmed the binding releases the GIL before any number below was taken.

Topology under test: N `cuvis.SessionFile` handles on one file, all sharing a single `cuvis.ProcessingContext`.
Each cell reads 24 frames over a rotating 10-frame window through `Cu3sPrefetchReader.read_many`, after a warm-up of `read_threads` frames.
Every cell is the median of five timed passes on one open reader, so a cell is separable from run-to-run spread.
Three was not enough: at three repeats the four-thread cell landed on 32.7 cubes/s against 48.8 at six, which the wider sample showed was sampling noise rather than a dip.
The GPU has to be otherwise idle. An earlier attempt at this table was taken while another process was holding and then releasing GPU memory; it read 8.3 cubes/s single-threaded and produced an impossible 3.3x at two threads, with negative VRAM deltas as the tell.
Correctness is checked per frame with blake2b against a single-threaded reference, and the hashing happens outside the timed region.

The process forces `force_gpu_mode=cuda` through `cuvis.init` before opening anything.
That is a new requirement: on SDK 3.6.0 a process that never calls `cuvis.init` processes on the host at about 3.9 fps single-threaded, against 14.9 fps here, which would drown the effect being measured.
The library does not yet do this itself; the host-versus-cuda comparison belongs with that work.

Raw numbers: [`results.json`](results.json), including the individual passes behind each median as `fps_runs`.

## Thread scaling

| read_threads | fps | scaling | setup s | VRAM MiB | RSS GB | wrong cubes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 12.85 | 1.00x | 13.46 | 1330 | 2.73 | 0 |
| 2 | 25.61 | 1.99x | 12.87 | 1424 | 4.52 | 0 |
| 4 | 40.72 | 3.17x | 12.86 | 1615 | 5.16 | 0 |
| 6 | 50.70 | 3.95x | 12.72 | 1777 | 5.72 | 0 |
| 8 | **52.64** | **4.10x** | 13.14 | 1959 | 6.28 | 0 |
| 12 | 47.72 | 3.71x | 12.87 | 2197 | 7.41 | 0 |
| 16 | 50.06 | 3.90x | 13.24 | 2510 | 8.47 | 0 |

![chart](chart.png)

## The `batch_size` ceiling

`read_threads` fixed at 8, varying how many indices are handed over per call, which is exactly what a torch `DataLoader`'s `batch_size` controls.

| batch_size | fps | scaling |
| --- | --- | --- |
| 1 | 16.54 | 1.29x |
| 2 | 24.94 | 1.94x |
| 4 | 34.97 | 2.72x |
| 6 | 42.63 | 3.32x |
| 8 | 47.08 | 3.67x |

![batch size](batch_size.png)

## Findings

Robust:

- **A published wheel now carries the GIL release.** Every earlier run of this benchmark needed a locally built `cuvis.pyil`, which is why the feature shipped with a probe and a fallback and why no user could reach the fast path from `pip install`.
  `cuvis-il` 3.6.0.0 from PyPI releases the GIL, and the probe agrees.
- **Threading works, and the win is large.** 12.85 to 52.64 fps at eight threads, 4.10x, on the same hardware and the same session.
  Nothing in the Python layer changed except that reads now overlap.
- **Zero wrong cubes at every thread count**, 1 through 16, verified per frame.
  This is the claim that mattered most: a shared `ProcessingContext` could plausibly have mixed state between concurrent callers, and it does not.
  SDK 3.6.0 adds per-object exclusive locking on writing functions, which could have serialised every `apply` on the one shared context; the scaling curve shows it does not.
- **The context is built once, not per handle.** Setup stays inside 12.72-13.46 s at every thread count.
  Per-handle construction would have cost roughly `threads x 13 s`, which is the dashed line in the bottom-left panel.
- **VRAM grows with handles, not with contexts.** 1330 MiB at one handle to 2510 MiB at sixteen.
  Sixteen private contexts would not have fit on an 8 GB card at all, which is what makes the shared-context topology the enabling choice rather than an optimisation.
- **Host memory is the cost that does scale**, at roughly 0.38 GB per handle: 2.73 GB at one, 8.47 GB at sixteen.
  It is the one axis that did not improve between SDK builds, so it stays the budgeting number.
  This is the number to budget with.
- **Concurrency is bounded by `batch_size`, not by `read_threads`.** With eight handles open, feeding one index at a time yields 16.54 fps against a 12.85 fps baseline, i.e. essentially nothing.
  torch hands a map-style dataset a whole batch of indices and nothing earlier, so there is no other moment at which the threads can be used.

Do not over-read:

- **Eight threads is both the knee and the peak here.** Six reaches 96% of it for 91% of its memory, and twelve and sixteen are slower, not faster.
  Treat 6 to 8 as the useful range on this machine, and re-measure elsewhere rather than assuming.
- **Where the curve turns over moved between SDK builds.** On build `dfcc3e3` sixteen threads was the fastest cell; on `987fdcf` it is eight, and twelve and sixteen fall back to 3.71x and 3.90x.
  The useful range is stable, the exact peak is not, which is the argument for re-measuring rather than copying a thread count out of this table.
- **The first pass of each cell is the slowest**, e.g. 36.8 fps against a 53.3 fps best at eight threads.
  The warm-up reads only `read_threads` frames of a ten-frame window, so the first timed pass still pays part of the cold cost.
  Medians absorb that; means would not.
- **Setup got slower between 3.6.0 builds, and stayed there.** The `ProcessingContext` LUT build was 9.1-9.7 s on build `22f4255` and is 12.7-13.5 s on `987fdcf`, on the same session and machine.
  It is paid once per process and is excluded from the throughput above, but it is what a `num_workers > 0` epoch pays per worker, so it is worth knowing before trading reader threads for worker processes.
- **`Raw` mode only.** `Reflectance` and `SpectralRadiance` hold materially more VRAM per handle and will hit the card sooner, so the thread ceiling there is lower.
- **A 10-frame window means the OS file cache is warm.** This isolates the SDK read and processing path rather than measuring cold disk throughput, which is the right target here but is not what a full shuffled epoch over 940 frames would see.

## Reproducing

Runs in the project venv now that the SDK binding is a published wheel; earlier revisions of this
report needed a separate one built against a local `cuvis.pyil`.

```powershell
uv sync --all-extras --extra bench
uv run python benchmarks\threaded_reading\bench_evidence.py `
    "D:\Measurements\hbf\Auto_013+01.cu3s" benchmarks\threaded_reading Raw 24 cuda 5
uv run python benchmarks\threaded_reading\make_charts.py benchmarks\threaded_reading
```

Both scripts import only `data/readers/` and `data/_extras.py`, bypassing `data/__init__.py`, because that pulls the DataModules and therefore `pytorch_lightning`, which a bare SDK venv does not carry.
On a binding that holds the GIL the probe rejects the pool and every cell collapses onto the single-handle row, which is the intended behaviour and also the reason CI can never measure this.
