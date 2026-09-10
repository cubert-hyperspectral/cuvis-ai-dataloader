"""Render the SDK-device evidence chart from results.json."""

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
by_device = {
    device: [row for row in R["rows"] if row["device"] == device] for device in ("cuda", "host")
}

fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

for device, colour, marker in (("cuda", BLUE, "o"), ("host", GREY, "s")):
    rows = by_device[device]
    left.plot(
        [r["threads"] for r in rows],
        [r["fps"] for r in rows],
        marker=marker,
        color=colour,
        label="GPU (sdk_cuda: true)" if device == "cuda" else "host (sdk_cuda: false)",
    )
left.set_xlabel("read_threads")
left.set_ylabel("cubes/s")
left.set_title("Threading is a GPU-only lever")
left.set_xscale("log", base=2)
left.set_xticks([r["threads"] for r in by_device["cuda"]])
left.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
left.grid(alpha=0.3)
left.legend()

threads = [r["threads"] for r in by_device["cuda"]]
ratios = [c["fps"] / h["fps"] for c, h in zip(by_device["cuda"], by_device["host"])]
right.bar([str(t) for t in threads], ratios, color=BLUE)
for x, ratio in enumerate(ratios):
    right.text(x, ratio, f"{ratio:.1f}x", ha="center", va="bottom")
right.set_xlabel("read_threads")
right.set_ylabel("cuda / host")
right.set_title("Cost of processing on the host")
right.set_ylim(0, max(ratios) * 1.18)
right.grid(alpha=0.3, axis="y")

sdk = R["sdk_version"].replace("CUBERT SDK v. ", "SDK ").split(" build")[0]
fig.suptitle(f"cu3s reading by SDK processing device - {R['mode']} mode, {sdk}")
fig.tight_layout()
path = os.path.join(OUT, "chart.png")
fig.savefig(path, dpi=140)
print("wrote", path)
