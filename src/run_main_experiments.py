#!/usr/bin/env python3
"""
Reproduce the main NashLens paper experiments (OpenSpiel poker + XRL-Bench).

This is a thin wrapper around the experiment module `poker_experiment.run()`,
which computes everything reported in the paper's main tables/figures:
exact exploitability on Kuhn/Leduc/liars_dice (CFR/NFSP/Deep-CFR baselines,
multi-seed CIs), the orthogonality (lambda) sweep, the CFR-teacher-quality
study, factor-intervention case studies, aggregate factor statistics, the
semantic-validity battery, and the XRL-Bench AIM/AUM/PGI/PGU/RIS metrics on
trained DQN agents (CartPole/LunarLander/FlappyBird).

Requirements (see ../requirements.txt): numpy, torch, open_spiel (pyspiel).
These are heavy; the lightweight, stdlib-only opponent-shift stability
experiment lives in ../experiments/opponent_shift_stability.py and needs none
of them.

Usage:
    cd src && python run_main_experiments.py
Writes results/experiment_results.json (the canonical results the paper cites).
"""
import json
import os
import sys

# make `agent_core` (the config shim) importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import poker_experiment  # noqa: E402


def main():
    exp_data, ablation, train_logs = poker_experiment.run()
    out = dict(exp_data)
    out["ablation"] = ablation
    out["train_logs"] = train_logs
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.abspath(os.path.join(here, "..", "results", "experiment_results.json"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"wrote {out_path}")
    gm = out.get("game_metrics", {})
    print("Kuhn (ours):", gm.get("kuhn_mean_ci", {}).get("exploit_ours"))
    print("Leduc (ours):", gm.get("leduc_mean_ci", {}).get("exploit_ours"))


if __name__ == "__main__":
    main()
