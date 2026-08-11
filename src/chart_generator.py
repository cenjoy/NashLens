"""
顶会标准图表生成：SOTA 对比图、消融实验图、收敛曲线图。
"""
import os

import matplotlib
matplotlib.use("Agg")  # 无显示环境安全
import matplotlib.pyplot as plt

from agent_core import config

plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
# 放大图表字体（评审：图例/坐标轴字体偏小）
plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "figure.titlesize": 16, "lines.linewidth": 2.0, "lines.markersize": 7,
})


def generate_all_figures(exp_data: dict, ablation_data: dict, train_logs: dict = None):
    """批量生成顶会标准图表，返回保存路径列表。"""
    fig_paths = []
    fig_dir = os.path.join(config.WORKSPACE_ROOT, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1) SOTA 对比柱状图
    plt.figure(figsize=(7, 5))
    labels = ["SOTA Baseline", "Ours"]
    acc = [exp_data["sota_baseline_acc"], exp_data["our_method_acc"]]
    bars = plt.bar(labels, acc, color=["#6699cc", "#cc6666"], alpha=0.85, width=0.6)
    lo = max(0, min(acc) - 5)
    plt.ylim(lo, min(100, max(acc) + 5))
    plt.title("SOTA Performance Comparison", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.grid(axis="y", alpha=0.3)
    for b in bars:
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                 f"{b.get_height():.2f}%", ha="center")
    p = os.path.join(fig_dir, "sota_compare.png")
    plt.tight_layout(); plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    fig_paths.append(p)

    # 2) 消融实验图
    if ablation_data:
        plt.figure(figsize=(7, 5))
        ks = list(ablation_data.keys())
        vs = list(ablation_data.values())
        bars2 = plt.bar(ks, vs, color="#77aa77", alpha=0.85)
        plt.title("Ablation Study Results", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=11)
        plt.xticks(rotation=12, fontsize=8)
        plt.grid(axis="y", alpha=0.3)
        for b in bars2:
            plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                     f"{b.get_height():.2f}%", ha="center")
        p2 = os.path.join(fig_dir, "ablation_result.png")
        plt.tight_layout(); plt.savefig(p2, dpi=300, bbox_inches="tight"); plt.close()
        fig_paths.append(p2)

    # 3) 收敛曲线图
    if train_logs:
        plt.figure(figsize=(7, 5))
        for name, log in train_logs.items():
            if not log:
                continue
            xs = [e["epoch"] for e in log]
            ys = [e["acc"] for e in log]
            plt.plot(xs, ys, marker="o", label=name)
        plt.title("Convergence Curve", fontsize=12)
        plt.xlabel("Epoch"); plt.ylabel("Test Accuracy (%)")
        plt.legend(); plt.grid(alpha=0.3)
        p3 = os.path.join(fig_dir, "convergence.png")
        plt.tight_layout(); plt.savefig(p3, dpi=300, bbox_inches="tight"); plt.close()
        fig_paths.append(p3)

    return fig_paths


def generate_poker_figures(exp_data: dict):
    """
    生成扑克实验的丰富对比图表（多博弈、多方法、含 95% CI 误差棒、解释器对齐对比）。
    返回图表路径列表。文件名含关键词以便 LaTeX 占位映射。
    """
    import numpy as np
    fig_dir = os.path.join(config.WORKSPACE_ROOT, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    gm = exp_data.get("game_metrics", {})
    k = gm.get("kuhn_mean_ci", {})
    l = gm.get("leduc_mean_ci", {})
    paths = []

    def mc(d, key):
        v = d.get(key, {})
        return float(v.get("mean", 0.0)), float(v.get("ci95", 0.0))

    # 1) 可利用度对比（Kuhn/Leduc × Ours/Baseline），对数轴 + 误差棒
    try:
        ko, ko_c = mc(k, "exploit_ours"); kb, kb_c = mc(k, "exploit_baseline")
        lo, lo_c = mc(l, "exploit_ours"); lb, lb_c = mc(l, "exploit_baseline")
        games = ["Kuhn", "Leduc"]
        ours = [ko, lo]; base = [kb, lb]; oc = [ko_c, lo_c]; bc = [kb_c, lb_c]
        x = np.arange(len(games)); w = 0.35
        plt.figure(figsize=(7, 5))
        plt.bar(x - w / 2, ours, w, yerr=oc, capsize=4, label="Ours (Nash-native)", color="#cc6666")
        plt.bar(x + w / 2, base, w, yerr=bc, capsize=4, label="Post-hoc/non-Nash baseline", color="#6699cc")
        plt.yscale("log")
        plt.xticks(x, games); plt.ylabel("Exploitability / NashConv (log, lower=better)")
        plt.title("Exact Exploitability across Imperfect-Information Poker")
        plt.legend(); plt.grid(axis="y", alpha=0.3, which="both")
        p = os.path.join(fig_dir, "exploitability_compare.png")
        plt.tight_layout(); plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close(); paths.append(p)
    except Exception:
        pass

    # 2) 解释器对齐对比（KernelSHAP / Gradient / Ours）on PGI/PGU/RIS
    try:
        ec = gm.get("explainer_comparison_kuhn", {})
        methods = [("KernelSHAP", ec.get("KernelSHAP", {})),
                   ("Gradient", ec.get("GradientSaliency", {})),
                   ("Ours", ec.get("Ours_intrinsic_feature", {}))]
        metrics = ["PGI", "PGU", "RIS"]
        x = np.arange(len(metrics)); w = 0.25
        colors = ["#8888cc", "#88bb88", "#cc8866"]
        plt.figure(figsize=(7.5, 5))
        for i, (name, d) in enumerate(methods):
            vals = [float(d.get(m) or 0.0) for m in metrics]
            plt.bar(x + (i - 1) * w, vals, w, label=name, color=colors[i])
        plt.xticks(x, ["PGI (↑)", "PGU (↓)", "RIS (↓)"]); plt.ylabel("Score")
        plt.title("Explainer Comparison (XRL-Bench protocol)")
        plt.legend(); plt.grid(axis="y", alpha=0.3)
        p = os.path.join(fig_dir, "explainer_compare.png")
        plt.tight_layout(); plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close(); paths.append(p)
    except Exception:
        pass

    # 3) 消融（Nash 安全分）
    try:
        abla = exp_data.get("ablation") or {}
        if abla:
            ks = list(abla.keys()); vs = [float(v) for v in abla.values()]
            plt.figure(figsize=(7, 5))
            bars = plt.bar(ks, vs, color="#77aa77", alpha=0.85)
            plt.ylabel("Nash robustness score = 100/(1+exploitability)")
            plt.title("Ablation (Kuhn): Nash objective vs. decomposition")
            plt.grid(axis="y", alpha=0.3)
            for b in bars:
                plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{b.get_height():.1f}", ha="center")
            p = os.path.join(fig_dir, "ablation_result.png")
            plt.tight_layout(); plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close(); paths.append(p)
    except Exception:
        pass

    # 4) 保真度对比（Ours / Baseline / w-o Decomp）含 CI
    try:
        fo, fo_c = mc(k, "faithfulness_ours")
        fb, fb_c = mc(k, "faithfulness_baseline")
        fd, fd_c = mc(k, "faithfulness_remove_decomposition")
        labels = ["Ours", "Baseline\n(post-hoc)", "w/o Decomp"]
        vals = [fo, fb, fd]; cis = [fo_c, fb_c, fd_c]
        plt.figure(figsize=(7, 5))
        bars = plt.bar(labels, vals, yerr=cis, capsize=4, color=["#cc6666", "#6699cc", "#aaaaaa"])
        plt.ylabel("Factor-level faithfulness (higher=better)")
        plt.title("Explanation Faithfulness (mean ± 95% CI)")
        plt.ylim(0, 1.05); plt.grid(axis="y", alpha=0.3)
        p = os.path.join(fig_dir, "faithfulness_compare.png")
        plt.tight_layout(); plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close(); paths.append(p)
    except Exception:
        pass

    # 5) 蒸馏收敛曲线
    try:
        logs = exp_data.get("train_logs") or {}
        if logs:
            plt.figure(figsize=(7, 5))
            for name, log in logs.items():
                if not log:
                    continue
                xs = [e["epoch"] for e in log]; ys = [e["acc"] for e in log]
                plt.plot(xs, ys, marker="o", label=name)
            plt.xlabel("Training step"); plt.ylabel("Distillation fit (%)")
            plt.title("Convergence (distillation to target policy)")
            plt.legend(); plt.grid(alpha=0.3)
            p = os.path.join(fig_dir, "convergence.png")
            plt.tight_layout(); plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close(); paths.append(p)
    except Exception:
        pass

    # 6) 真实环境（XRL-Bench gym, CartPole）解释器对比
    try:
        cp = gm.get("xrlbench_realenv", {}).get("CartPole-v1", {})
        if cp and "KernelSHAP" in cp:
            metrics = ["PGI", "PGU", "RIS"]
            methods = [("KernelSHAP", cp.get("KernelSHAP", {})),
                       ("Gradient", cp.get("Gradient", {})),
                       ("Ours", cp.get("Ours_decomposition", {}))]
            x = np.arange(len(metrics)); w = 0.25
            colors = ["#8888cc", "#88bb88", "#cc8866"]
            plt.figure(figsize=(7.5, 5))
            for i, (name, d) in enumerate(methods):
                vals = [float(d.get(m) or 0.0) for m in metrics]
                plt.bar(x + (i - 1) * w, vals, w, label=name, color=colors[i])
            plt.xticks(x, ["PGI (↑)", "PGU (↓)", "RIS (↓)"]); plt.ylabel("Score")
            plt.title("Real-Env XRL-Bench (CartPole): Explainer Comparison")
            plt.legend(); plt.grid(axis="y", alpha=0.3)
            p = os.path.join(fig_dir, "realenv_explainer.png")
            plt.tight_layout(); plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close(); paths.append(p)
    except Exception:
        pass

    # 7) 真实环境三环境合并对比（CartPole / LunarLander / FlappyBird × 三解释器，3 子图）
    try:
        re_all = gm.get("xrlbench_realenv", {})
        envs = [("CartPole-v1", "CartPole"), ("LunarLander-v3", "LunarLander"),
                ("FlappyBird-v0", "FlappyBird")]
        envs = [(k, lbl) for k, lbl in envs if re_all.get(k, {}).get("KernelSHAP")]
        if len(envs) >= 2:
            metrics = ["PGI", "PGU", "RIS"]
            methods = ["KernelSHAP", "Gradient", "Ours_decomposition"]
            mlabel = {"KernelSHAP": "KernelSHAP", "Gradient": "Gradient", "Ours_decomposition": "Ours"}
            colors = ["#8888cc", "#88bb88", "#cc8866"]
            fig, axes = plt.subplots(1, len(envs), figsize=(5.2 * len(envs), 4.6), sharey=False)
            if len(envs) == 1:
                axes = [axes]
            for ax, (ekey, elbl) in zip(axes, envs):
                d = re_all[ekey]; x = np.arange(len(metrics)); w = 0.25
                for i, mth in enumerate(methods):
                    vals = [float((d.get(mth) or {}).get(m) or 0.0) for m in metrics]
                    ax.bar(x + (i - 1) * w, vals, w, label=mlabel[mth], color=colors[i])
                ax.set_xticks(x); ax.set_xticklabels(["PGI↑", "PGU↓", "RIS↓"])
                ret = d.get("agent_mean_return", "?")
                ax.set_title(f"{elbl} (return {ret})", fontsize=10)
                ax.grid(axis="y", alpha=0.3)
            axes[0].set_ylabel("Score")
            axes[-1].legend(fontsize=8)
            fig.suptitle("XRL-Bench Real Environments: Explainer Comparison (KernelSHAP / Gradient / Ours)",
                         fontsize=12)
            p = os.path.join(fig_dir, "realenv_all_compare.png")
            plt.tight_layout(); plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close(); paths.append(p)
    except Exception:
        pass

    return paths


def generate_overview_figure(exp_data: dict = None):
    """生成论文总览/学术架构图（方法流程框图），返回路径。"""
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch
    fig_dir = os.path.join(config.WORKSPACE_ROOT, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                             fc=fc, ec="#333333", lw=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.2, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=14, lw=1.4, color="#555555"))

    ax.text(6, 6.65, "Game-Native Interpretable RL via Nash-Equilibrium Constraints",
            ha="center", fontsize=13, fontweight="bold")

    # 输入环境
    box(0.3, 4.2, 2.4, 1.6,
        "Environments\n\nKuhn / Leduc Poker\n(OpenSpiel)\nCartPole / LunarLander /\nFlappyBird (XRL-Bench)", "#dbe9f6")
    # 状态/信息集
    box(3.1, 4.5, 1.9, 1.0, "State /\nInformation set\n(features)", "#eef3da")
    # 三因素解耦策略（核心）
    box(5.4, 3.9, 3.0, 2.2,
        "Three-Factor Additive\nDecomposed Policy\n\n z_reward  +  z_opponent\n +  z_intrinsic  =  logits\n(intrinsic, faithful-by-design)", "#f6e0c8")
    # Nash 约束
    box(5.6, 1.6, 2.6, 1.4,
        "Nash-Equilibrium\nObjective\n(CFR distillation /\nmaximin)", "#f3d4d4")
    # 输出
    box(9.0, 4.5, 2.6, 1.4, "Low exploitability\n(near-Nash play)\n+ interpretable\nfactor attributions", "#d8efd8")
    # 评测
    box(9.0, 2.2, 2.6, 1.7,
        "Evaluation\n\n• Exploitability (NashConv)\n• Faithfulness (do-interv.)\n• XRL-Bench AIM/AUM/\n  PGI/PGU/RIS\n• vs. SHAP / gradient", "#e7e0f3")
    # 多种子
    box(2.9, 1.7, 2.2, 1.2, "Multi-seed\nmean ± 95% CI\n+ paired t-test", "#efe7d0")

    arrow(2.7, 5.0, 3.1, 5.0)
    arrow(5.0, 5.0, 5.4, 5.0)
    arrow(6.9, 3.9, 6.9, 3.0)      # policy <- nash
    arrow(8.4, 5.0, 9.0, 5.2)      # policy -> outputs
    arrow(10.3, 4.5, 10.3, 3.9)    # outputs -> eval
    arrow(5.1, 2.3, 8.0, 2.5)      # multiseed -> eval (context)

    p = os.path.join(fig_dir, "overview_architecture.png")
    plt.tight_layout(); plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    return p
