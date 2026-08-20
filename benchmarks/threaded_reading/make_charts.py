"""Render the threaded-reading evidence charts from results.json."""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = sys.argv[1]
with open(os.path.join(OUT, "results.json"), encoding="utf-8") as fh:
    R = json.load(fh)

scaling = [r for r in R["scaling"] if "error" not in r]
threads = [r["threads"] for r in scaling]
fps = [r["fps"] for r in scaling]
vram = [r["vram_mib"] for r in scaling]
rss = [r["rss_gb"] for r in scaling]
setup = [r["setup_s"] for r in scaling]
base_fps = fps[0]
base_setup = setup[0]

BLUE, GREY, RED = "#1f77b4", "#888888", "#d62728"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

# --------------------------------------------------------------------------- chart.png
fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
fig.suptitle(
    f"Threaded cu3s reading: {R['frames_in_session']}-frame session, {R['mode']} mode, "
    f"{R['iters']} reads over a {R['window']}-frame window\n"
    f"one ProcessingContext shared by N SessionFile handles"
    f"   (GIL-releasing binding: {R['releases_gil']})",
    fontsize=10,
)

a = ax[0][0]
a.plot(threads, fps, "o-", color=BLUE, lw=2, label="measured")
a.axhline(base_fps, color=GREY, ls="--", lw=1.2, label=f"1 handle = {base_fps:.1f} fps")
best = max(scaling, key=lambda r: r["fps"])
a.annotate(
    f"{best['fps']:.1f} fps ({best['scaling']:.1f}x)\nat {best['threads']} handles",
    xy=(best["threads"], best["fps"]), xytext=(3.1, max(fps) * 1.02),
    arrowprops=dict(arrowstyle="->", color="black", lw=0.9),
    bbox=dict(boxstyle="round", fc="#e3f2fd", ec="#1565c0"),
)
a.set_title("Throughput scales with handles, then saturates")
a.set_xlabel("read_threads (SessionFile handles)")
a.set_ylabel("frames / s")
a.set_xticks(threads)
a.set_ylim(0, max(fps) * 1.3)
a.legend(loc="lower right")

a = ax[0][1]
a.plot(threads, vram, "o-", color=BLUE, lw=2, label="measured (one shared context)")
a.plot(threads, [vram[0] / 2 * t for t in threads], "--", color=RED, lw=1.4,
       label="if each handle built its own context (est.)")
a.axhline(8192, color=GREY, ls=":", lw=1.4, label="8 GB card")
a.set_title("VRAM: the shared context is what makes this fit")
a.set_xlabel("read_threads (SessionFile handles)")
a.set_ylabel("VRAM above baseline (MiB)")
a.set_xticks(threads)
a.set_yscale("log")
a.legend(loc="upper left")

a = ax[1][0]
a.plot(threads, setup, "o-", color=BLUE, lw=2, label="measured (built once)")
a.plot(threads, [base_setup * t for t in threads], "--", color=RED, lw=1.4,
       label="if built per handle (est.)")
a.set_title("Setup cost stays flat: the context is built once")
a.set_xlabel("read_threads (SessionFile handles)")
a.set_ylabel("reader open time (s)")
a.set_xticks(threads)
a.set_yscale("log")
a.legend(loc="upper left")

a = ax[1][1]
a.plot(threads, rss, "o-", color=BLUE, lw=2, label="measured")
a.set_title("Host memory: budget roughly 0.35 GB per handle")
a.set_xlabel("read_threads (SessionFile handles)")
a.set_ylabel("RSS (GB)")
a.set_xticks(threads)
a.set_ylim(0, max(rss) * 1.3)
wrong = sum(r["wrong_cubes"] for r in scaling)
a.text(0.5, 0.08, f"cubes differing from the single-threaded reference: {wrong}",
       transform=a.transAxes, ha="center",
       bbox=dict(boxstyle="round", fc="#e8f5e9", ec="#4caf50"))
a.legend(loc="upper left")

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUT, "chart.png"), dpi=130)
print("wrote chart.png")

# ---------------------------------------------------------------------- batch_size.png
batch = R["batch_size"]
sizes = [r["batch_size"] for r in batch]
bfps = [r["fps"] for r in batch]

fig, a = plt.subplots(figsize=(7.5, 4.6))
a.plot(sizes, bfps, "o-", color=BLUE, lw=2, label=f"read_threads={batch[0]['read_threads']}")
a.axhline(base_fps, color=GREY, ls="--", lw=1.3, label=f"single handle = {base_fps:.1f} fps")
a.annotate(
    "batch_size=1 gains nothing:\ntorch hands over one index,\nso there is nothing to overlap",
    xy=(sizes[0], bfps[0]), xytext=(1.35, base_fps * 2.0),
    arrowprops=dict(arrowstyle="->", color="black", lw=0.9),
    bbox=dict(boxstyle="round", fc="#fff3e0", ec="#ef6c00"),
)
a.set_title("Concurrency is bounded by batch_size, not by read_threads")
a.set_xlabel("indices handed over at once (the DataLoader's batch_size)")
a.set_ylabel("frames / s")
a.set_xticks(sizes)
a.set_ylim(0, max(bfps) * 1.3)
a.grid(alpha=0.3)
a.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "batch_size.png"), dpi=130)
print("wrote batch_size.png")
