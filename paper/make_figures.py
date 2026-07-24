#!/usr/bin/env python3
"""Generate all paper figures from benchmarks/results/ JSONs.
Okabe-Ito colorblind-safe palette; grayscale-legible; one axis per figure.
Run: benchmarks/python/.venv/bin/python paper/make_figures.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../benchmarks/results")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito (CVD-safe)
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "pink": "#CC79A7", "sky": "#56B4E9", "verm": "#D55E00", "gray": "#777777"}

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

CHIPS = [
    ("Apple_M1_8GB_8CPU_7GPU_16ANE", "M1 8GB\n(fanless)", C["gray"]),
    ("Apple_M2_8GB_8CPU_8GPU_16ANE", "M2 8GB", C["sky"]),
    ("Apple_M3_Pro_18GB_11CPU_14GPU_18ANE", "M3 Pro 18GB", C["green"]),
    ("Apple_M4_24GB_10CPU_10GPU_16ANE", "M4 24GB", C["blue"]),
    ("Apple_M4_Pro_24GB_12CPU_16GPU_20ANE", "M4 Pro 24GB", C["orange"]),
]
SEQS = ["1024", "2048", "4096", "8192"]


def load(slug, name):
    p = os.path.join(RES, slug, name)
    return json.load(open(p)) if os.path.exists(p) else None


# --- Figure 1: five-chip Llama split speedups ------------------------------

fig, ax = plt.subplots(figsize=(5.6, 2.6))
n = len(CHIPS)
w = 0.15
xs = np.arange(len(SEQS))
for i, (slug, label, color) in enumerate(CHIPS):
    d = load(slug, "prefill_scale_benchmark.json")
    vals = []
    for L in SEQS:
        arms = d["results"]["llama"].get(L, {})
        m, s = arms.get("mlx_fp16"), arms.get("fusion_split")
        vals.append(m["median"] / s["median"] if m and s else np.nan)
    ax.bar(xs + (i - n / 2 + 0.5) * w, vals, w * 0.92, label=label.replace("\n", " "),
           color=color, edgecolor="white", linewidth=0.5)
ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(xs, [f"{s}" for s in SEQS])
ax.set_xlabel("Prefill sequence length (tokens)")
ax.set_ylabel("Speedup vs. MLX-FP16")
ax.set_ylim(0.9, 1.45)
ax.legend(ncol=3, fontsize=7, frameon=False, loc="upper left")
fig.savefig(os.path.join(OUT, "fig_fivechip.pdf"))
plt.close(fig)

# --- Figure 2: the mechanism probe -----------------------------------------

fig, ax = plt.subplots(figsize=(3.2, 2.3))
cases = ["input\nmaterialized", "input in\nsame graph", "eager\nboundary"]
vals = [1.376, 0.658, 1.342]
cols = [C["blue"], C["verm"], C["blue"]]
bars = ax.bar(cases, vals, 0.6, color=cols, edgecolor="white")
ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}×",
            ha="center", fontsize=8)
ax.set_ylabel("Split speedup vs. GPU-only")
ax.set_ylim(0, 1.6)
fig.savefig(os.path.join(OUT, "fig_mechanism.pdf"))
plt.close(fig)

# --- Figure 3: TTFT on real checkpoints (M4, M4 Pro) ------------------------

fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.3), sharey=True)
for ax, (slug, label, color) in zip(
        axes, [CHIPS[3], CHIPS[4]]):
    t = load(slug, "mlxlm_ttft_benchmark.json")
    Ls = sorted(t["results"], key=int)
    sp = [t["results"][L]["ttft_speedup"] for L in Ls]
    prompts = [t["results"][L]["prompt_tokens"] for L in Ls]
    bars = ax.bar([f"{p//1000}k" for p in prompts], sp, 0.55, color=color,
                  edgecolor="white")
    for b, v in zip(bars, sp):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}×",
                ha="center", fontsize=8)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(label.replace("\n", " "), fontsize=9)
    ax.set_xlabel("Prompt length")
axes[0].set_ylabel("TTFT speedup vs. stock MLX-LM")
axes[0].set_ylim(0.9, 1.35)
fig.savefig(os.path.join(OUT, "fig_ttft.pdf"))
plt.close(fig)

# --- Figure 4: single block vs full depth (M4) ------------------------------

fig, ax = plt.subplots(figsize=(3.4, 2.3))
d1 = load(CHIPS[3][0], "prefill_scale_benchmark.json")
d32 = load(CHIPS[3][0], "full_depth_prefill_benchmark.json")
seqs = ["2048", "4096", "8192"]
b1, b32 = [], []
for L in seqs:
    a = d1["results"]["llama"][L]
    b1.append(a["mlx_fp16"]["median"] / a["fusion_split"]["median"])
    a = d32["results"][L]
    b32.append(a["mlx_fp16"]["median"] / a["fusion_split"]["median"])
xs = np.arange(len(seqs))
ax.bar(xs - 0.18, b1, 0.34, label="1 block", color=C["sky"], edgecolor="white")
ax.bar(xs + 0.18, b32, 0.34, label="32 blocks (15.6 GB)", color=C["blue"],
       edgecolor="white")
ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(xs, seqs)
ax.set_xlabel("Prefill sequence length (tokens)")
ax.set_ylabel("Speedup vs. MLX-FP16")
ax.set_ylim(0.9, 1.35)
ax.legend(fontsize=8, frameon=False)
fig.savefig(os.path.join(OUT, "fig_fulldepth.pdf"))
plt.close(fig)

print("figures written to", OUT)
