"""从 VecMonitor csv 画 PPO 残差精修训练曲线（回报 + 成功率代理）。"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
IMG_DIR = HERE.parent.parent / "docs" / "assets" / "images"


def main():
    rows = []
    for line in (HERE / "checkpoints" / "ppo_refine_monitor.csv").read_text().splitlines():
        if line.startswith("#") or line.startswith("r,"):
            continue
        r, l, t = line.split(",")[:3]
        rows.append((float(r), int(l), float(t)))
    rew = np.array([r for r, _, _ in rows])
    steps = np.cumsum([l for _, l, _ in rows])

    win = 100
    smooth = np.convolve(rew, np.ones(win) / win, mode="valid")
    # 成功率代理：episode 回报 > 10（含 +10 成功奖）视为子任务成功
    succ = np.convolve((rew > 10).astype(float), np.ones(win) / win, mode="valid")

    fig, ax1 = plt.subplots(figsize=(9.5, 4.0), dpi=130)
    ax1.plot(steps[win - 1:], smooth, color="#3f51b5", lw=1.8, label="episode 回报（滑窗 100）")
    ax1.set_xlabel("环境步数")
    ax1.set_ylabel("episode 回报", color="#3f51b5")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(steps[win - 1:], succ * 100, color="#e57373", lw=1.8,
             label="子任务成功率（回报>10 代理，含探索噪声）")
    ax2.set_ylabel("成功率 (%)", color="#e57373")
    ax2.set_ylim(0, 100)
    ax1.set_title("PPO 相位级修正策略训练曲线（撕剪-投放子任务，物理随机化）")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower right", fontsize=9)
    fig.tight_layout()
    out = IMG_DIR / "rl_refine_curve.png"
    fig.savefig(out)
    print(f"训练曲线: {out}（{len(rows)} episodes, {steps[-1]} steps）")


if __name__ == "__main__":
    main()
