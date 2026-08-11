"""测量归因头的部署成本：参数量与每决策延迟。

论文 §5.1 与摘要引用的成本数字由本脚本产生，可直接复跑核对：

    python experiments/measure_inference_cost.py

报告两项：
  1. 参数量——加性学生（两个因素头 + 学习到的 intrinsic 偏置）与同规模
     single-head 学生的对比，差值即"归因能力"的净增量。
  2. 每决策延迟——CPU 上前向一次并做因素分解 + softmax；另报含三次
     do-intervention（逐因素置零后重算策略）的完整审计延迟。
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from poker_experiment import (
    DecomposedPolicyNet,
    PlainPolicyNet,
    _masked_logprobs,
    _prepare_game,
)

REPEATS = 2000
WARMUP = 100
FACTORS = ("reward", "opponent", "intrinsic")


def _count(net):
    return sum(p.numel() for p in net.parameters())


def _decision_latency(net, x, mask, with_interventions=False):
    """单次决策的墙钟时间（毫秒）。with_interventions 时含三次因素置零重算。"""
    with torch.no_grad():
        for _ in range(WARMUP):
            net(x)
        t0 = time.perf_counter()
        for _ in range(REPEATS):
            z_r, z_o, z_i = net.factors(x)
            torch.exp(_masked_logprobs(z_r + z_o + z_i, mask))
            if with_interventions:
                parts = {"reward": z_r, "opponent": z_o, "intrinsic": z_i}
                for f in FACTORS:
                    kept = [v for k, v in parts.items() if k != f]
                    torch.exp(_masked_logprobs(kept[0] + kept[1], mask))
        t1 = time.perf_counter()
    return (t1 - t0) / REPEATS * 1000.0


def main():
    out = {"repeats": REPEATS, "device": "cpu", "games": {}}
    for game in ("kuhn_poker", "leduc_poker"):
        # CFR 迭代数不影响成本测量，取小值以缩短准备时间
        prep = _prepare_game(game, 30)
        ind, na = prep["in_dim"], prep["n_actions"]

        torch.manual_seed(0)
        additive = DecomposedPolicyNet(ind, na)
        torch.manual_seed(0)
        single = PlainPolicyNet(ind, na)

        info = next(iter(prep["states"].values()))
        x = torch.tensor(info["tensor"]).unsqueeze(0)
        mask = torch.zeros(1, na)
        for a in info["legal"]:
            mask[0, a] = 1.0

        rec = {
            "in_dim": ind,
            "n_actions": na,
            "params_additive": _count(additive),
            "params_single_head": _count(single),
            "ms_per_decision": round(_decision_latency(additive, x, mask), 4),
            "ms_per_decision_with_3_interventions": round(
                _decision_latency(additive, x, mask, with_interventions=True), 4
            ),
        }
        rec["params_additive_M"] = round(rec["params_additive"] / 1e6, 5)
        out["games"][game] = rec

        print(f"-- {game}  (in_dim={ind}, n_actions={na})")
        print(f"   additive student   {rec['params_additive']:6d} params "
              f"= {rec['params_additive_M']:.4f}M")
        print(f"   single-head student{rec['params_single_head']:6d} params")
        print(f"   per decision                     {rec['ms_per_decision']:.4f} ms")
        print(f"   per decision + 3 do-interventions "
              f"{rec['ms_per_decision_with_3_interventions']:.4f} ms")

    path = os.path.join(os.path.dirname(__file__), "..", "results", "inference_cost.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
