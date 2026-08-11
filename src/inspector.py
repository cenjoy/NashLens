"""NashLens inspector: 逐决策审计的命令行工具。

对应论文 Fig.1 的三个区域：
  (1) Scenario     选择博弈与信息集
  (2) Policy       动作概率 + 因素 logit 分解 + do-intervention（置零某因素后重算 softmax）+ 对比 CFR 教师
  (3) Evidence     精确可利用度、因素移除 delta、语义诊断（validated / uncertain）

用法：
    python src/inspector.py --game kuhn_poker                    # 列出全部信息集并审计前几个
    python src/inspector.py --game kuhn_poker --infoset 1b       # 审计单个信息集
    python src/inspector.py --game leduc_poker --cfr-iters 300

语义标签规则（与论文 §7 一致）：某个 head 只有在通过其代理相关性检验时才标 validated，
否则标 uncertain，此时界面抑制因果措辞（"reward-designated head dominates" 而非
"payoff structure caused this action"）。
"""
import argparse
import json

import numpy as np
import torch

from poker_experiment import (
    DecomposedPolicyNet,
    _distill,
    _exploitability,
    _masked_logprobs,
    _prepare_game,
    _semantic_correlation,
)

# 代理检验阈值：相关性符号与幅度都需符合预期方向才算 validated
PROXY_MIN_R = 0.30

FACTORS = ("reward", "opponent", "intrinsic")


def _fit(prep, seed=0, lam=0.1, steps=400):
    """蒸馏一个加性学生策略（与论文主实验同配置）。

    注意：必须在构造网络前设种子——权重初始化本身影响因素间如何分配 logit 质量，
    只给 _distill 设种子不足以保证可复现。
    """
    torch.manual_seed(seed)
    net = DecomposedPolicyNet(prep["in_dim"], prep["n_actions"])
    _distill(net, prep["X"], prep["mask"], prep["cfr_T"],
             steps=steps, seed=seed, lam=lam)
    return net


def _probs(net, x, legal, n_actions, drop=None):
    """给定信息集张量，返回动作概率；drop 指定要置零的因素（do-intervention）。"""
    mask = torch.zeros(1, n_actions)
    for a in legal:
        mask[0, a] = 1.0
    with torch.no_grad():
        z_r, z_o, z_i = net.factors(x)
        parts = {"reward": z_r, "opponent": z_o, "intrinsic": z_i}
        if drop is not None:
            parts[drop] = torch.zeros_like(parts[drop])
        logits = parts["reward"] + parts["opponent"] + parts["intrinsic"]
        p = torch.exp(_masked_logprobs(logits, mask)).squeeze(0).numpy()
    return p, {k: v.detach().squeeze(0).numpy() for k, v in parts.items()}


def _semantic_labels(net, prep):
    """按代理检验给每个 head 标 validated / uncertain（论文 §7 的规则）。"""
    corr = _semantic_correlation(net, prep["states"], prep["game"])
    out = {}
    # reward head 期望与手牌强度正相关
    r = corr.get("reward_vs_handstrength")
    out["reward"] = {
        "proxy": "hand strength",
        "r": r,
        "status": "validated" if (r is not None and r >= PROXY_MIN_R) else "uncertain",
    }
    # opponent head 期望与下注历史长度正相关
    r = corr.get("opponent_vs_historylen")
    out["opponent"] = {
        "proxy": "betting-history length",
        "r": r,
        "status": "validated" if (r is not None and r >= PROXY_MIN_R) else "uncertain",
    }
    # intrinsic 是常量（设计使然），逐信息集相关性无定义
    out["intrinsic"] = {
        "proxy": "state-independent by design",
        "r": None,
        "status": "n/a (constant)",
    }
    return out


def _cfr_probs(prep, key, n_actions):
    """CFR 教师在该信息集上的目标分布（用于 teacher 对比）。"""
    idx = prep["keys"].index(key)
    return prep["cfr_T"][idx].numpy()


def _fmt_p(p, legal):
    return "[" + ", ".join(f"{p[a]:.3f}" for a in legal) + "]"


def audit(net, prep, key, labels, exploit):
    """打印单个信息集的完整审计报告（Fig.1 的 2、3 区）。"""
    info = prep["states"][key]
    legal = info["legal"]
    n_actions = prep["n_actions"]
    x = torch.tensor(info["tensor"]).unsqueeze(0)

    p, parts = _probs(net, x, legal, n_actions)
    teacher = _cfr_probs(prep, key, n_actions)

    print(f"\n{'='*72}")
    print(f"  (1) SCENARIO   game={prep['game'].get_type().short_name}   "
          f"infoset=[{key}]   player={info['player']}   legal={legal}")
    print(f"{'='*72}")

    print("  (2) POLICY INSPECTOR")
    print(f"      student action probs   {_fmt_p(p, legal)}")
    print(f"      CFR teacher target     {_fmt_p(teacher, legal)}")
    print(f"      L1 gap vs teacher      {np.abs(p - teacher).sum():.4f}")
    print(f"\n      per-action logit decomposition (sums to the action score):")
    print(f"        {'action':>8s} {'reward':>10s} {'opponent':>10s} {'intrinsic':>10s} {'total':>10s}")
    for a in legal:
        zr, zo, zi = parts["reward"][a], parts["opponent"][a], parts["intrinsic"][a]
        print(f"        {a:8d} {zr:10.3f} {zo:10.3f} {zi:10.3f} {zr+zo+zi:10.3f}")

    print(f"\n      do-intervention (zero a factor, recompute softmax):")
    deltas = {}
    for f in FACTORS:
        p2, _ = _probs(net, x, legal, n_actions, drop=f)
        d = float(np.abs(p2 - p).sum())
        deltas[f] = d
        print(f"        remove {f:10s} -> {_fmt_p(p2, legal)}   L1 shift {d:.4f}")

    print(f"\n  (3) EVIDENCE")
    print(f"      exact best-response exploitability (NashConv)   {exploit:.5f}")
    print(f"      CFR teacher exploitability                      {prep['cfr_exploit']:.5f}")
    print(f"      semantic diagnostics (per-head proxy check):")
    for f in FACTORS:
        L = labels[f]
        r = "  n/a" if L["r"] is None else f"{L['r']:+.3f}"
        flag = {"validated": "[VALIDATED]", "uncertain": "[UNCERTAIN]"}.get(
            L["status"], "[N/A]")
        print(f"        {f:10s} vs {L['proxy']:28s} r={r}  {flag}")

    dom = max(deltas, key=deltas.get)
    if labels[dom]["status"] == "validated":
        verdict = (f"the {dom} factor dominates this decision "
                   f"(proxy check passed, causal reading licensed)")
        print(f"\n      audit verdict: the {dom} factor dominates this decision")
        print(f"        (proxy check passed, causal reading licensed)")
    else:
        verdict = (f"the {dom}-designated head dominates this decision; proxy check FAILED, "
                   f"so report which head carries the logit mass, NOT that {dom} structure "
                   f"caused the action")
        print(f"\n      audit verdict: the {dom}-designated head dominates this decision")
        print(f"        (proxy check FAILED -- report which head carries the logit mass,")
        print(f"         NOT that {dom} structure caused the action)")

    # 结构化审计记录（供 --export 落盘，构成可归档的审计轨迹）
    return {
        "game": prep["game"].get_type().short_name,
        "infoset": key,
        "player": int(info["player"]),
        "legal_actions": [int(a) for a in legal],
        "action_probs": {int(a): round(float(p[a]), 6) for a in legal},
        "cfr_teacher_probs": {int(a): round(float(teacher[a]), 6) for a in legal},
        "l1_gap_vs_teacher": round(float(np.abs(p - teacher).sum()), 6),
        "logit_decomposition": {
            int(a): {f: round(float(parts[f][a]), 6) for f in FACTORS} for a in legal
        },
        "factor_removal_l1": {f: round(deltas[f], 6) for f in FACTORS},
        "exploitability": {
            "student_nashconv": round(float(exploit), 6),
            "cfr_teacher_nashconv": round(float(prep["cfr_exploit"]), 6),
        },
        "semantic_diagnostics": {
            f: {"proxy": labels[f]["proxy"], "pearson_r": labels[f]["r"],
                "status": labels[f]["status"]} for f in FACTORS
        },
        "dominant_head": dom,
        "causal_reading_licensed": labels[dom]["status"] == "validated",
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser(description="NashLens per-decision inspector")
    ap.add_argument("--game", default="kuhn_poker",
                    help="kuhn_poker | leduc_poker | liars_dice")
    ap.add_argument("--cfr-iters", type=int, default=300)
    ap.add_argument("--lam", type=float, default=0.1, help="orthogonality penalty")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--infoset", default=None,
                    help="information-state string to audit (default: first --n)")
    ap.add_argument("--n", type=int, default=3, help="how many infosets to audit")
    ap.add_argument("--list", action="store_true", help="list all infosets and exit")
    ap.add_argument("--export", default=None, metavar="PATH",
                    help="write the audit trail for the inspected states to a JSON file")
    args = ap.parse_args()

    print(f"loading {args.game} and solving CFR ({args.cfr_iters} iters) ...")
    prep = _prepare_game(args.game, args.cfr_iters)
    print(f"  {prep['n_infostates']} information states "
          f"(distinct information_state_string over both players' decision nodes)")

    if args.list:
        for k in prep["keys"]:
            print("  ", k)
        return

    print(f"distilling additive student (lambda={args.lam}, {args.steps} steps, "
          f"seed={args.seed}) ...")
    net = _fit(prep, seed=args.seed, lam=args.lam, steps=args.steps)
    exploit = _exploitability(prep["game"], net, prep["states"])
    labels = _semantic_labels(net, prep)

    if args.infoset is not None:
        if args.infoset not in prep["states"]:
            raise SystemExit(f"infoset [{args.infoset}] not found; "
                             f"use --list to see all {prep['n_infostates']}")
        targets = [args.infoset]
    else:
        targets = prep["keys"][: args.n]

    records = [audit(net, prep, key, labels, exploit) for key in targets]

    if args.export:
        payload = {
            "config": {
                "game": args.game, "cfr_iters": args.cfr_iters, "lambda": args.lam,
                "distill_steps": args.steps, "seed": args.seed,
                "n_information_states": prep["n_infostates"],
            },
            "note": ("Factor names are intended-and-tested, not guaranteed. Removal effects are "
                     "exact properties of the policy logits; a head whose proxy check reports "
                     "'uncertain' must not be given a causal reading."),
            "audits": records,
        }
        with open(args.export, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\n  audit trail exported: {args.export}  ({len(records)} state(s))")


if __name__ == "__main__":
    main()
