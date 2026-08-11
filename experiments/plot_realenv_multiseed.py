#!/usr/bin/env python3
"""
Redraw the XRL-Bench real-environment comparison figure WITH 95% CI error bars,
from the multi-seed results produced by realenv_multiseed.py.

Reads  ../results/realenv_xrlbench_multiseed.json
Writes ../figures/realenv_all_compare.png   (same layout as the single-seed
figure, now with error bars = 95% CI over seeds).

Deps: matplotlib, numpy (std lib otherwise).
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "figure.titlesize": 16,
})

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "realenv_xrlbench_multiseed.json")
FIGDIR = os.path.join(HERE, "..", "figures")
METRICS = ["PGI", "PGU", "RIS"]
MLABEL = ["PGI↑", "PGU↓", "RIS↓"]
METHODS = [("KernelSHAP", "KernelSHAP"), ("Gradient", "Gradient"),
           ("Ours_decomposition", "Ours")]
COLORS = ["#8888cc", "#88bb88", "#cc8866"]
ENV_LABEL = {"CartPole-v1": "CartPole", "LunarLander-v3": "LunarLander",
             "FlappyBird-v0": "FlappyBird"}


def main():
    data = json.load(open(RES, encoding="utf-8"))
    results = data["results"]
    envs = [e for e in ["CartPole-v1", "LunarLander-v3", "FlappyBird-v0"]
            if e in results and "KernelSHAP" in results[e]]
    if not envs:
        print("no usable envs in", RES); sys.exit(1)
    os.makedirs(FIGDIR, exist_ok=True)
    fig, axes = plt.subplots(1, len(envs), figsize=(5.2 * len(envs), 4.8), sharey=False)
    if len(envs) == 1:
        axes = [axes]
    for ax, ekey in zip(axes, envs):
        d = results[ekey]; x = np.arange(len(METRICS)); w = 0.25
        for i, (mkey, mlbl) in enumerate(METHODS):
            means = [float(d.get(mkey, {}).get(m, {}).get("mean", 0.0)) for m in METRICS]
            cis = [float(d.get(mkey, {}).get(m, {}).get("ci95", 0.0)) for m in METRICS]
            ax.bar(x + (i - 1) * w, means, w, yerr=cis, capsize=4,
                   label=mlbl, color=COLORS[i])
        ax.set_xticks(x); ax.set_xticklabels(MLABEL)
        ret = d.get("agent_mean_return", {})
        rstr = f"{ret.get('mean','?')}±{ret.get('ci95','?')}" if isinstance(ret, dict) else ret
        ax.set_title(f"{ENV_LABEL.get(ekey, ekey)} (return {rstr})", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Score")
    n_seeds = results[envs[0]].get("n_seeds", "?")
    axes[-1].legend(fontsize=9)
    fig.suptitle(f"XRL-Bench Real Environments: Explainer Comparison "
                 f"(mean ± 95% CI over {n_seeds} seeds)", fontsize=12)
    out = os.path.join(FIGDIR, "realenv_all_compare.png")
    plt.tight_layout(); plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
