#!/usr/bin/env python3
"""Emit the LaTeX data rows for the XRL-Bench table (tab:xrl) from the
multi-seed results, using per-cell means (95% CI is shown in the figure to keep
the table within page width). Run after realenv_multiseed.py."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "realenv_xrlbench_multiseed.json")
ENVS = [("CartPole-v1", "CartPole"), ("LunarLander-v3", "LunarLander"),
        ("FlappyBird-v0", "FlappyBird")]
METHODS = [("KernelSHAP", "KernelSHAP"), ("Gradient", "Gradient"),
           ("Ours_decomposition", "NashLens")]
METRICS = ["AIM", "AUM", "PGI", "PGU", "RIS"]


def cell(d, expl, met):
    v = d.get(expl, {}).get(met, {})
    return f"${v.get('mean', 0.0):.3f}$" if isinstance(v, dict) else "---"


def main():
    data = json.load(open(RES, encoding="utf-8"))
    res = data["results"]
    n = None
    rows = []
    for ekey, elabel in ENVS:
        d = res.get(ekey, {})
        if "KernelSHAP" not in d:
            continue
        n = d.get("n_seeds", n)
        ret = d.get("agent_mean_return", {})
        rlabel = f"{ret.get('mean','?'):.0f}" if isinstance(ret, dict) else str(ret)
        rows.append(f"\\multirow{{3}}{{*}}{{{elabel} ($\\bar r{{=}}{rlabel}$)}}")
        for i, (mkey, mlabel) in enumerate(METHODS):
            pre = " & " if i == 0 else " & "
            cells = " & ".join(cell(d, mkey, m) for m in METRICS)
            rows.append(f"{pre}{mlabel} & {cells} \\\\")
        rows.append("\\midrule")
    if rows and rows[-1] == "\\midrule":
        rows[-1] = "\\bottomrule"
    print(f"% n_seeds = {n}")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
