"""Render the device-resident cube evidence chart from results.json."""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = sys.argv[1]
with open(os.path.join(OUT, "results.json"), encoding="utf-8") as fh:
    R = json.load(fh)

BLUE, GREY = "#1f77b4", "#888888"
device = [row for row in R["rows"] if row["cuda_cubes"]]
host = [row for row in R["rows"] if not row["cuda_cubes"]]
threads = [row["threads"] for row in device]

fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

for rows, colour, marker, label in (
    (device, BLUE, "o", "device (cuda_cubes: true)"),
    (host, GREY, "s", "host round trip (cuda_cubes: false)"),
):
    left.plot(
        [r["threads"] for r in rows],
        [r["fps"] for r in rows],
        marker=marker,
        color=colour,
        label=label,
    )
left.set_xlabel("read_threads")
left.set_ylabel("cubes/s onto the GPU")
left.set_title("Cubes delivered ready for a training step")
left.set_xscale("log", base=2)
left.set_xticks(threads)
left.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
left.grid(alpha=0.3)
left.legend()

ratios = [d["fps"] / h["fps"] for d, h in zip(device, host)]
right.bar([str(t) for t in threads], ratios, color=BLUE)
for x, ratio in enumerate(ratios):
    right.text(x, ratio, f"{ratio:.2f}x", ha="center", va="bottom")
right.axhline(1.0, color=GREY, linewidth=1, linestyle="--")
right.set_xlabel("read_threads")
right.set_ylabel("device / host")
right.set_title("The copy hurts more the more threads read")
right.set_ylim(0, max(ratios) * 1.18)
right.grid(alpha=0.3, axis="y")

sdk = R["sdk_version"].replace("CUBERT SDK v. ", "SDK ").split(" build")[0]
fig.suptitle(f"Device-resident cu3s cubes - {R['mode']} mode, {sdk}, {R['gpu']}")
fig.tight_layout()
path = os.path.join(OUT, "chart.png")
fig.savefig(path, dpi=140)
print("wrote", path)
