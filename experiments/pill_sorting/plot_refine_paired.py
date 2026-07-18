"""从 eval_refine 输出解析 40 组配对结果，画配对对比总览图。"""

import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
IMG_DIR = HERE.parent.parent / "docs" / "assets" / "images"


def parse(path):
    """读 eval_refine 输出的配对明细 CSV。"""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines()[1:]:
        f = line.split(",")
        rows.append((int(f[0]), float(f[5]), float(f[6]),
                     f[7] == "1", f[9] == "1"))
    return rows


def main():
    rows = parse(HERE / "debug" / "eval_refine_paired.csv")
    n = len(rows)
    script_ok = np.array([r[3] for r in rows])
    rl_ok = np.array([r[4] for r in rows])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10.5, 5.6), dpi=130,
        gridspec_kw={"height_ratios": [1.15, 1]})

    # 上：40 组配对格子图
    colors = {True: "#66bb6a", False: "#ef5350"}
    for i, (idx, sx, sy, s_ok, r_ok) in enumerate(rows):
        ax1.add_patch(plt.Rectangle((i, 1.05), 0.92, 0.9,
                                    color=colors[s_ok], ec="white", lw=1.5))
        ax1.add_patch(plt.Rectangle((i, 0.0), 0.92, 0.9,
                                    color=colors[r_ok], ec="white", lw=1.5))
        if not s_ok and r_ok:
            ax1.plot(i + 0.46, 0.45, marker="*", color="white", ms=9, mew=0)
    ax1.set_xlim(-0.3, n)
    ax1.set_ylim(-0.15, 2.1)
    ax1.set_yticks([1.5, 0.45])
    ax1.set_yticklabels(["零动作脚本", "RL 相位级修正"], fontsize=11)
    ax1.set_xticks(np.arange(0, n, 5) + 0.46)
    ax1.set_xticklabels(np.arange(0, n, 5))
    ax1.set_xlabel("配对组编号（每组 = 相同场景 + 相同物理参数 θ）", fontsize=10)
    ax1.set_title(f"配对对比总览（{n} 组）：脚本败而 RL 成 "
                  f"{int((~script_ok & rl_ok).sum())} 组（白星），"
                  f"反向仅 {int((script_ok & ~rl_ok).sum())} 组", fontsize=12)
    ax1.legend(handles=[Patch(color="#66bb6a", label="入盒 B 成功"),
                        Patch(color="#ef5350", label="失败")],
               loc="upper right", fontsize=9, ncols=2, framealpha=0.9)
    for s in ("top", "right", "left", "bottom"):
        ax1.spines[s].set_visible(False)
    ax1.tick_params(length=0)

    # 下：按感知偏移大小分桶的成功率（失败主因的直观呈现）
    sense_mag = np.array([np.hypot(r[1], r[2]) for r in rows])
    bins = [(0, 10), (10, 18), (18, 30)]
    labels = ["<10 mm", "10~18 mm", ">18 mm"]
    x = np.arange(len(bins))
    s_rate, r_rate, cnt = [], [], []
    for lo, hi in bins:
        m = (sense_mag >= lo) & (sense_mag < hi)
        cnt.append(int(m.sum()))
        s_rate.append(script_ok[m].mean() * 100 if m.any() else 0)
        r_rate.append(rl_ok[m].mean() * 100 if m.any() else 0)
    w = 0.36
    b1 = ax2.bar(x - w / 2, s_rate, w, color="#90a4ae", label="零动作脚本")
    b2 = ax2.bar(x + w / 2, r_rate, w, color="#3f51b5", label="RL 相位级修正")
    for bars in (b1, b2):
        ax2.bar_label(bars, fmt="%.0f%%", fontsize=9, padding=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{l}\n(n={c})" for l, c in zip(labels, cnt)], fontsize=10)
    ax2.set_ylabel("入盒 B 成功率 (%)")
    ax2.set_ylim(0, 112)
    ax2.set_xlabel("感知偏移幅度（水平合成，失败主因）", fontsize=10)
    ax2.legend(fontsize=9, loc="lower left")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("RL 精修配对评测：撕剪-投放子任务 @ 物理随机化", fontsize=13, y=1.0)
    fig.tight_layout()
    out = IMG_DIR / "rl_refine_paired.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"配对总览图: {out}（{n} 组）")


if __name__ == "__main__":
    main()
