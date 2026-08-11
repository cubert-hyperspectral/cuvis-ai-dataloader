# CPU-only worker scaling: cu3s (host) vs npz

410x410x164, effective 120 samples/epoch, steady epoch (persistent_workers=True), no GPU
(`force_gpu_mode=host`). Machine: 20 logical / 14 physical cores; SDK `processing_thread_count=8`.

| workers | cu3s fps | npz fps | cu3s CPU% mean/peak | npz CPU% mean/peak | cu3s RAM GB | npz RAM GB |
|---|---|---|---|---|---|---|
| 1 | 3.67 | 2.28 | 32.5 / 71.2 | 19.5 / 75.5 | 3.00 | 0.63 |
| 2 | 6.03 | 4.61 | 55.1 / 84.4 | 15.4 / 51.6 | 5.72 | 1.05 |
| 4 | 7.48 | 8.72 | 74.9 / 95.4 | 25.7 / 43.2 | 11.17 | 1.58 |
| 6 | 8.55 | 12.72 | 83.5 / 100 | 38.3 / 60.8 | 16.56 | 2.26 |
| 8 | 8.67 | 15.16 | 91.7 / 100 | 48.7 / 77.1 | 21.92 | 2.87 |

## Conclusion

- **cu3s CPU saturates early (~6 workers).** Throughput plateaus at ~8.6 fps (1->2 +64%, 2->4 +24%, 4->6 +14%, 6->8 +1.4%), CPU peak hits 100% at 6 workers, mean 84-92%. Parallel efficiency at 8w is only ~30% (2.4x). Beyond 6 workers you burn RAM (22 GB at 8w) for nothing.
- **npz scales near-linearly and does not saturate within 8 workers.** 2.28 -> 15.16 fps (6.65x, ~83% efficiency), CPU mean only 49% and peak 77% at 8 workers - it would keep climbing toward the ~14-20 core limit. RAM stays tiny (2.9 GB at 8w).
- **Crossover at ~4 workers.** Below it cu3s is faster (one worker already runs ~8 SDK processing threads vs npz's single decompression thread); at and above it npz wins, reaching 1.75x cu3s at 8 workers.
- **Why:** each cu3s CPU worker spawns ~8 SDK threads, so a few workers already oversubscribe 20 cores (6x8=48 threads) and plateau; npz uses 1 thread/worker, filling cores efficiently.

## Full-load point (the question asked)

- **cu3s CPU: full load at ~6 workers** (peak 100%, throughput flat, mean ~84%).
- **npz: not saturated even at 8 workers** (mean 49%); its knee is beyond this sweep.

## Practical takeaway

If restricted to CPU, **npz with ~8 workers is best** (15 fps, ~3 GB RAM). cu3s-CPU caps at ~8.6 fps and costs 22 GB. For reference, a single GPU cu3s process hits 6.83 fps at ~1 GB VRAM and near-zero CPU - competitive with cu3s-CPU's ceiling at a fraction of the host resources.
