"""重画论文 Fig. 1 的 inspector 面板示意图。

替换 6/14 生成的旧版（生成脚本已丢失），修掉两处不实表述：
  - 旧版底部写 "walkthrough, screencast, ..."：两者都不存在
  - 旧版标题写 "interactive"：实际是命令行工具，非交互式 GUI
并把 Leduc 的精度对齐论文正文（3 位小数）。

所有数字取自 results/experiment_results.json：
  Kuhn  exploit_ours = 0.00425 +/- 0.00136  -> 0.0043 +/- 0.0014
  Leduc exploit_ours = 0.17538 +/- 0.01385  -> 0.175  +/- 0.014
  infoset [1b] action_probs = [0.65, 0.35]

用法：
    python src/render_paper_figure1.py figures/nashlens_ui.pdf
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ---- 调色：与论文其余图表同族的柔和色，深色描边保证印刷可辨 ----
INK = "#1F2933"
MUTED = "#6B7A8C"
HAIR = "#D3DCE6"
BAND = "#EDF1F5"

BLUE_F, BLUE_E = "#EAF2FD", "#5B9BE8"
GREEN_F, GREEN_E = "#E6F5EE", "#3FA97B"
AMBER_F, AMBER_E = "#FDF0DC", "#D89A3A"
LILAC_F, LILAC_E = "#F1EBFB", "#9B7BD4"
GREY_F, GREY_E = "#F5F7F9", "#8D9AA8"

BAR_R, BAR_O, BAR_I = "#1D8A6B", "#7B4FD4", "#E0A226"

W, H = 12.6, 7.3


def box(ax, x, y, w, h, text, fc, ec, size=13, weight="bold", radius=0.09):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=1.6, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, color=INK, weight=weight, zorder=3, linespacing=1.45)


def arrow(ax, p0, p1, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=17, linewidth=1.9,
        color="#41505F", connectionstyle=f"arc3,rad={rad}",
        shrinkA=2, shrinkB=3, zorder=4))


def render(out):
    fig = plt.figure(figsize=(W, H), dpi=200)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.set_facecolor("white")

    # 外框
    ax.add_patch(FancyBboxPatch(
        (0.18, 0.18), W - 0.36, H - 0.36,
        boxstyle="round,pad=0,rounding_size=0.16",
        facecolor="white", edgecolor=HAIR, linewidth=1.6, zorder=1))

    # 标题带（去掉 "interactive"：这是命令行工具）
    ax.add_patch(Rectangle((0.42, H - 1.12), W - 1.9, 0.62,
                           facecolor=BAND, edgecolor="none", zorder=2))
    ax.text(0.72, H - 0.81, "NashLens: CFR-distilled policy inspector",
            ha="left", va="center", fontsize=19, weight="bold", color=INK, zorder=3)

    # 分栏标题
    for cx, label in ((1.95, "1. Scenario"), (5.95, "2. Policy inspector"),
                      (10.15, "3. Evidence")):
        ax.text(cx, H - 1.44, label, ha="center", va="center",
                fontsize=15.5, weight="bold", color=INK, zorder=3)

    # ---------------- 1. Scenario ----------------
    box(ax, 0.72, 4.86, 2.46, 0.72, "Game: Kuhn / Leduc\n/ liars_dice", BLUE_F, BLUE_E, 12.5, "normal")
    box(ax, 0.72, 3.96, 2.46, 0.72, "Infoset: card\n+ betting history", BLUE_F, BLUE_E, 12.5, "normal")
    box(ax, 0.72, 3.20, 2.46, 0.60, "Action set and legal mask", BLUE_F, BLUE_E, 12.5, "normal")

    ax.text(0.85, 2.72, "Controls", ha="left", va="center",
            fontsize=15.5, weight="bold", color=INK, zorder=3)
    box(ax, 0.72, 1.74, 1.14, 0.62, "zero\nreward", GREEN_F, GREEN_E, 11.5)
    box(ax, 2.04, 1.74, 1.14, 0.62, "zero\nopponent", LILAC_F, LILAC_E, 11.5)
    box(ax, 0.72, 0.92, 1.14, 0.62, "teacher\ncompare", AMBER_F, AMBER_E, 11.5)
    # 真实 flag 是 --export，写成 export JSON
    box(ax, 2.04, 0.92, 1.14, 0.62, "export\nJSON", GREY_F, GREY_E, 11.5)

    # ---------------- 2. Policy inspector ----------------
    box(ax, 4.20, 4.66, 3.52, 0.92,
        "CFR teacher policy\nnear-equilibrium target", "white", BLUE_E, 13)
    box(ax, 4.20, 3.50, 3.52, 0.92,
        "Additive student logits\nreward + opponent + intrinsic", GREEN_F, GREEN_E, 13)
    arrow(ax, (5.96, 4.66), (5.96, 4.46))
    arrow(ax, (5.96, 3.50), (5.96, 3.06))

    # 动作表
    tx, ty, tw, th = 4.20, 1.16, 3.52, 1.86
    ax.add_patch(Rectangle((tx, ty), tw, th, facecolor="white",
                           edgecolor=HAIR, linewidth=1.4, zorder=2))
    ax.plot([tx, tx + tw], [ty + th - 0.44] * 2, color=HAIR, linewidth=1.2, zorder=3)
    for cx, head in ((tx + 0.60, "action"), (tx + 1.62, "prob."), (tx + 2.78, "logit factors")):
        ax.text(cx, ty + th - 0.22, head, ha="center", va="center",
                fontsize=11.5, weight="bold", color=INK, zorder=4)

    # infoset [1b] 的真实策略：[0.65, 0.35]
    for row, (act, prob, mark) in enumerate((("pass", "0.650", False), ("bet", "0.350", True))):
        ry = ty + th - 0.95 - row * 0.62
        ax.text(tx + 0.60, ry, act, ha="center", va="center", fontsize=12, color=INK, zorder=4)
        ax.text(tx + 1.62, ry, prob, ha="center", va="center", fontsize=12, color=INK, zorder=4)
        # 因素条：reward / opponent / intrinsic
        bx, bw, bh = tx + 2.10, 1.34, 0.30
        widths = (0.44, 0.60, 0.30) if not mark else (0.30, 0.68, 0.36)
        x0 = bx
        for wid, col in zip(widths, (BAR_R, BAR_O, BAR_I)):
            ax.add_patch(Rectangle((x0, ry - bh / 2), wid * bw / sum(widths) * 1.0 * (bw / bw),
                                   bh, facecolor=col, edgecolor="none", zorder=4))
            x0 += wid * bw / sum(widths)
        if mark:
            ax.text(bx + 0.15, ry, "R", ha="center", va="center",
                    fontsize=10.5, color="white", weight="bold", zorder=5)

    ax.text(tx + tw / 2, ty - 0.20, "Softmax is recomputed after interventions.",
            ha="center", va="center", fontsize=11, color=MUTED, style="italic", zorder=4)

    # ---------------- 3. Evidence ----------------
    box(ax, 8.62, 4.72, 3.28, 0.80, "Exploitability\nexact best response", AMBER_F, AMBER_E, 13)
    box(ax, 8.62, 3.82, 3.28, 0.80, "Factor intervention\nDelta when removed", LILAC_F, LILAC_E, 13)
    box(ax, 8.62, 2.70, 3.28, 0.80, "Semantic checks\nvalidated / uncertain", GREEN_F, GREEN_E, 13)

    # teacher -> exploitability
    arrow(ax, (7.72, 5.12), (8.62, 5.12))
    # 动作表 -> 竖干 -> factor intervention / semantic checks
    ax.plot([7.72, 8.34], [2.10, 2.10], color="#41505F", linewidth=1.9, zorder=4)
    ax.plot([8.34, 8.34], [2.10, 4.22], color="#41505F", linewidth=1.9, zorder=4)
    arrow(ax, (8.34, 4.22), (8.62, 4.22))
    arrow(ax, (8.34, 3.10), (8.62, 3.10))

    # live summary：数字取自 results/experiment_results.json，精度与论文正文一致
    sx, sy, sw, sh = 8.62, 0.92, 3.28, 1.50
    ax.add_patch(Rectangle((sx, sy), sw, sh, facecolor="white",
                           edgecolor=HAIR, linewidth=1.4, zorder=2))
    ax.text(sx + 0.20, sy + sh - 0.28, "live summary", ha="left", va="center",
            fontsize=12.5, weight="bold", color=INK, zorder=4)
    for i, line in enumerate((
            "Kuhn:  0.0043 +/- 0.0014",
            "Leduc: 0.175 +/- 0.014",
            "liars_dice: stress-test")):
        ax.text(sx + 0.20, sy + sh - 0.66 - i * 0.32, line, ha="left", va="center",
                fontsize=11.5, color=INK, zorder=4, family="DejaVu Sans")

    # ---------------- 页脚：只列真实存在的产物 ----------------
    ax.plot([0.62, W - 0.62], [0.70, 0.70], color=HAIR, linewidth=1.2, zorder=3)
    ax.text(W / 2, 0.45,
            "Outputs:  per-decision audit trail  ·  JSON export  ·  reproducible figure/table package",
            ha="center", va="center", fontsize=11.5, color=MUTED, zorder=4)

    fig.savefig(out, format="pdf", facecolor="white", bbox_inches="tight", pad_inches=0.04)
    print(f"wrote {out}")


if __name__ == "__main__":
    render(sys.argv[1] if len(sys.argv) > 1 else "nashlens_ui.pdf")
