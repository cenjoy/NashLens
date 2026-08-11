"""把 inspector 的真实终端输出渲染为矢量 PDF，用作论文 Fig.1。

用法：
    python src/render_inspector_figure.py /tmp/inspector_out.txt figures/nashlens_ui.pdf
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# 终端配色（深色背景 + 分区高亮），与论文里对 inspector 三个区域的描述对应
BG = "#1c2128"
FG = "#d7dde5"
ACCENT = "#4fc3f7"     # 区域标题
OK = "#66bb6a"         # VALIDATED
WARN = "#ffa726"       # UNCERTAIN
DIM = "#7d8792"        # 次要信息
NUM = "#f0f3f6"        # 数值


def _color(line):
    s = line.strip()
    if s.startswith("=" * 8):
        return DIM
    if s.startswith(("(1)", "(2)", "(3)")):
        return ACCENT
    if "[VALIDATED]" in s:
        return OK
    if "[UNCERTAIN]" in s or "FAILED" in s:
        return WARN
    if s.startswith(("loading", "distilling", "  ")) and "r=" not in s:
        return FG
    return FG


def render(src, out):
    lines = open(src).read().rstrip("\n").split("\n")
    # 去掉 loading/distilling 进度行——它们不属于界面本身
    lines = [l for l in lines if not l.startswith(("loading", "distilling"))]
    while lines and not lines[0].strip():
        lines.pop(0)
    n = len(lines)
    width = max(len(l) for l in lines)

    # 等宽字体：字号放大以便缩到 0.86\textwidth 后仍清晰
    char_w, line_h, fs = 0.098, 0.196, 9.6
    fig_w = width * char_w + 0.5
    fig_h = n * line_h + 0.32

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    y = fig_h - 0.26
    for line in lines:
        c = _color(line)
        weight = "bold" if line.strip().startswith(("(1)", "(2)", "(3)")) else "normal"
        ax.text(0.22, y, line, family="DejaVu Sans Mono", fontsize=fs,
                color=c, va="top", ha="left", weight=weight)
        y -= line_h

    fig.savefig(out, format="pdf", facecolor=BG, bbox_inches="tight", pad_inches=0.06)
    print(f"wrote {out}  ({fig_w:.2f} x {fig_h:.2f} in, {n} lines x {width} cols)")


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2])
