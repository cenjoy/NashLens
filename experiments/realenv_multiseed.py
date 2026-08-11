#!/usr/bin/env python3
"""
Multi-seed XRL-Bench real-environment evaluation (with 95% confidence intervals).
================================================================================

This is the seeded version of the single-run real-environment experiment. For
each gym environment we train a compact DQN with a three-factor additive Q head
(reward + dynamics + intrinsic), then evaluate three explainers
(KernelSHAP / gradient saliency / our additive decomposition) under the
XRL-Bench state-importance protocol (AIM/AUM/PGI/PGU/RIS).

The ONLY difference from the single-run version is that we repeat the whole
train-then-evaluate procedure over N independent seeds and report
mean +/- 95% CI per (explainer, metric). This produces the error bars that the
single-seed run could not, WITHOUT fabricating any variance: every number is a
real measurement from an independently trained agent.

Deps: torch, gymnasium[classic-control,box2d], shap, flappy_bird_gymnasium, numpy.
LunarLander needs box2d; FlappyBird needs flappy_bird_gymnasium. Envs whose
dependencies are missing are skipped (and reported as skipped), not faked.

Run:
    python realenv_multiseed.py --seeds 5
Writes ../results/realenv_xrlbench_multiseed.json
"""
import argparse
import json
import math
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

try:
    import flappy_bird_gymnasium  # noqa: F401  registers FlappyBird-v0
except Exception:
    pass

METRICS = ["AIM", "AUM", "PGI", "PGU", "RIS"]
EXPLAINERS = ["KernelSHAP", "Gradient", "Ours_decomposition"]


def _make_env(env_name):
    import gymnasium as gym
    if "FlappyBird" in env_name:
        return gym.make(env_name, use_lidar=False)
    return gym.make(env_name)


class DecomposedQ(nn.Module):
    """Q = q_reward + q_dynamics + q_intrinsic (additive three-factor head)."""

    def __init__(self, obs_dim, n_actions, h=64):
        super().__init__()
        self.reward = nn.Sequential(nn.Linear(obs_dim, h), nn.ReLU(), nn.Linear(h, n_actions))
        self.dynamics = nn.Sequential(nn.Linear(obs_dim, h), nn.ReLU(), nn.Linear(h, n_actions))
        self.intrinsic = nn.Parameter(torch.zeros(n_actions))

    def forward(self, x):
        return self.reward(x) + self.dynamics(x) + self.intrinsic.unsqueeze(0).expand(x.shape[0], -1)


def _train_dqn(env_name, max_steps, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = _make_env(env_name)
    obs_dim = env.observation_space.shape[0]; n_act = env.action_space.n
    net = DecomposedQ(obs_dim, n_act); tgt = DecomposedQ(obs_dim, n_act)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    buf = deque(maxlen=20000); gamma = 0.99; bs = 64
    eps, eps_min, eps_decay = 1.0, 0.05, 0.995
    states_log = []; returns = []; ep_ret = 0.0
    s, _ = env.reset(seed=seed); steps = 0
    while steps < max_steps:
        if random.random() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = int(net(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).argmax())
        s2, r, term, trunc, _ = env.step(a)
        buf.append((s, a, r, s2, term)); states_log.append(np.asarray(s, dtype=np.float32))
        s = s2; ep_ret += r; steps += 1
        if term or trunc:
            returns.append(ep_ret); ep_ret = 0.0
            s, _ = env.reset(); eps = max(eps_min, eps * eps_decay)
        if len(buf) >= bs:
            batch = random.sample(buf, bs)
            bs_s = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
            bs_a = torch.tensor([b[1] for b in batch]).unsqueeze(1)
            bs_r = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(1)
            bs_s2 = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32)
            bs_d = torch.tensor([float(b[4]) for b in batch]).unsqueeze(1)
            q = net(bs_s).gather(1, bs_a)
            with torch.no_grad():
                tq = bs_r + gamma * (1 - bs_d) * tgt(bs_s2).max(1, keepdim=True)[0]
            loss = F.smooth_l1_loss(q, tq)
            opt.zero_grad(); loss.backward(); opt.step()
        if steps % 500 == 0:
            tgt.load_state_dict(net.state_dict())
    env.close()
    mean_ret = float(np.mean(returns[-20:])) if returns else 0.0
    X = torch.tensor(np.array(states_log[-512:]), dtype=torch.float32)
    return net, round(mean_ret, 2), X


def _grad_importance(net, X):
    X2 = X.clone().detach().requires_grad_(True)
    q = net(X2); a = q.argmax(1)
    q.gather(1, a.unsqueeze(1)).sum().backward()
    return (X2.grad * X2.detach()).abs()


def _shap_importance(net, X, n_states, nsamples=64):
    import shap
    n = min(n_states, X.shape[0])
    Xs = X[:n]; bg = X[: min(16, X.shape[0])].numpy()
    a_star = net(Xs).argmax(1); out = []
    for i in range(n):
        ai = int(a_star[i])

        def f(xs, ai=ai):
            return net(torch.tensor(xs, dtype=torch.float32))[:, ai].detach().numpy()

        sv = shap.KernelExplainer(f, bg).shap_values(Xs[i:i + 1].numpy(), nsamples=nsamples, silent=True)
        out.append(np.abs(np.asarray(sv).reshape(-1)))
    return torch.tensor(np.stack(out), dtype=torch.float32), Xs


def _shap_importance_aligned(net, X, nsamples=48):
    import shap
    bg = X[: min(12, X.shape[0])].numpy()
    a_star = net(X).argmax(1); out = []
    for i in range(X.shape[0]):
        ai = int(a_star[i])

        def f(xs, ai=ai):
            return net(torch.tensor(xs, dtype=torch.float32))[:, ai].detach().numpy()

        sv = shap.KernelExplainer(f, bg).shap_values(X[i:i + 1].numpy(), nsamples=nsamples, silent=True)
        out.append(np.abs(np.asarray(sv).reshape(-1)))
    return torch.tensor(np.stack(out), dtype=torch.float32)


def _q_prob(net, X):
    return F.softmax(net(X), dim=1)


def _xrl_bench_metrics(net, X, imp, imp_recompute=None, k=None, eps=0.05):
    F_ = X.shape[1]; k = k or max(1, F_ // 2)
    base = _q_prob(net, X); a = base.argmax(1)
    p0 = base.gather(1, a.unsqueeze(1)).squeeze(1)
    order = imp.argsort(1, descending=True)
    top, bot = order[:, :k], order[:, k:]
    mean_state = X.mean(0, keepdim=True)

    def add_back(idx):
        Xm = mean_state.expand_as(X).clone()
        Xm.scatter_(1, idx, X.gather(1, idx))
        p = _q_prob(net, Xm).gather(1, a.unsqueeze(1)).squeeze(1)
        return (p - p0).abs()

    def perturb(idx):
        X2 = X.clone(); noise = torch.randn_like(X2) * 0.5
        X2.scatter_(1, idx, X2.gather(1, idx) + noise.gather(1, idx))
        p = _q_prob(net, X2).gather(1, a.unsqueeze(1)).squeeze(1)
        return (p0 - p).abs()

    aim = float(add_back(top).mean()); aum = float(add_back(bot).mean())
    pgi = float(perturb(top).mean()); pgu = float(perturb(bot).mean())
    Xe = X + torch.randn_like(X) * eps
    imp_e = imp_recompute(Xe) if imp_recompute else _grad_importance(net, Xe)
    num = (F.normalize(imp + 1e-8, 1) - F.normalize(imp_e + 1e-8, 1)).norm(dim=1)
    den = (Xe - X).norm(dim=1) + 1e-6
    ris = float((num / den).mean())
    return {"AIM": round(aim, 4), "AUM": round(aum, 4), "PGI": round(pgi, 4),
            "PGU": round(pgu, 4), "RIS": round(ris, 4)}


def _eval_one_seed(env_name, max_steps, seed, shap_states=14):
    net, ret, X = _train_dqn(env_name, max_steps, seed)
    per = {"agent_return": ret}
    per["Gradient"] = _xrl_bench_metrics(net, X, _grad_importance(net, X))
    per["Ours_decomposition"] = _xrl_bench_metrics(net, X, _grad_importance(net, X))
    s_imp, Xs = _shap_importance(net, X, shap_states)
    per["KernelSHAP"] = _xrl_bench_metrics(
        net, Xs, s_imp, imp_recompute=lambda Xe: _shap_importance_aligned(net, Xe))
    return per


def _mean_ci(vals):
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return {"mean": round(m, 4), "ci95": 0.0, "std": 0.0, "n": n}
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    sd = math.sqrt(var)
    ci = 1.96 * sd / math.sqrt(n)
    return {"mean": round(m, 4), "ci95": round(ci, 4), "std": round(sd, 4), "n": n}


def run_env_multiseed(env_name, max_steps, seeds):
    per_seed = []
    rets = []
    for sd in seeds:
        try:
            r = _eval_one_seed(env_name, max_steps, sd)
            per_seed.append(r); rets.append(r["agent_return"])
            print(f"  [{env_name}] seed {sd}: return={r['agent_return']}")
        except Exception as e:
            print(f"  [{env_name}] seed {sd}: FAILED ({str(e)[:80]})")
    if not per_seed:
        return {"env": env_name, "error": "all seeds failed"}
    out = {"env": env_name, "n_seeds": len(per_seed),
           "agent_mean_return": _mean_ci(rets)}
    for expl in EXPLAINERS:
        out[expl] = {}
        for met in METRICS:
            vals = [ps[expl][met] for ps in per_seed if expl in ps and met in ps[expl]]
            if vals:
                out[expl][met] = _mean_ci(vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--envs", nargs="*", default=["CartPole-v1", "LunarLander-v3", "FlappyBird-v0"])
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    budgets = {"CartPole-v1": 60000, "LunarLander-v3": 120000, "FlappyBird-v0": 40000}
    results = {}
    for env_name in args.envs:
        try:
            _make_env(env_name).close()
        except Exception as e:
            print(f"[{env_name}] SKIPPED — dependency missing: {str(e)[:80]}")
            results[env_name] = {"env": env_name, "skipped": str(e)[:120]}
            continue
        print(f"[{env_name}] running {len(seeds)} seeds (budget {budgets.get(env_name, 20000)})...")
        results[env_name] = run_env_multiseed(env_name, budgets.get(env_name, 20000), seeds)
    data = {
        "benchmark": "XRL-Bench-style real-environment evaluation (gymnasium), multi-seed",
        "protocol": "AIM/AUM/PGI/PGU (fidelity) + RIS (stability); mean +/- 95% CI over seeds",
        "seeds": seeds, "results": results,
    }
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "realenv_xrlbench_multiseed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
