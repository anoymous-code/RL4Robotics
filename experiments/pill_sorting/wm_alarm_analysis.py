"""世界模型作为安全监控器：失败预警的离线评估。

对每条 ACT rollout，逐决策点（每 50 步）用 WM 的 fail 头算失败概率，
评估报警规则"连续 2 个点 > 阈值"的：
    - 召回率：失败 episode 中被预警的比例
    - 误报率：成功 episode 中被误警的比例
    - 提前量：预警时刻距 episode 结束还有多少秒（留给人工介入的时间）

医疗场景语义：分药宁可停机呼人，不可分错——预警器把"20% 的失败"
变成"报警 + 人工确认"，剩余未报警部分的实际错误率才是产品指标。

运行:
    ../../.venv/Scripts/python.exe wm_alarm_analysis.py --thresh 0.7
"""

import argparse
import io
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

from train_act import CAMS, CHUNK, IMG_MEAN, IMG_STD
from train_wm import ROLL_DIR, WorldModel

HERE = Path(__file__).resolve().parent
IMG_DIR = HERE.parents[1] / "docs" / "assets" / "images"


class FailProbe:
    def __init__(self, ckpt_path, device="cuda"):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model = WorldModel().to(device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.stats = {k: np.array(v, dtype=np.float32)
                      for k, v in ckpt["stats"].items()}
        self.device = device

    @torch.no_grad()
    def fail_prob(self, f, t, row):
        imgs = []
        for cam in CAMS:
            arr = np.asarray(Image.open(io.BytesIO(
                f[f"observations/images/{cam}"][t].tobytes())),
                dtype=np.float32) / 255.0
            imgs.append((arr - IMG_MEAN) / IMG_STD)
        imgs = np.stack(imgs).transpose(0, 3, 1, 2)
        qpos = ((f["observations/qpos"][t] - self.stats["qpos_mean"])
                / self.stats["qpos_std"])
        acts = ((f["action"][t:t + CHUNK] - self.stats["act_mean"])
                / self.stats["act_std"]).reshape(-1)
        imgs_t = torch.from_numpy(imgs.astype(np.float32)).unsqueeze(0).to(self.device)
        qpos_t = torch.from_numpy(qpos.astype(np.float32)).unsqueeze(0).to(self.device)
        acts_t = torch.from_numpy(acts.astype(np.float32)).unsqueeze(0).to(self.device)
        tgt_t = torch.tensor([row], device=self.device)
        with torch.autocast(self.device, dtype=torch.bfloat16):
            _, p_fail, _ = self.model(imgs_t, qpos_t, acts_t, tgt_t)
        return float(torch.sigmoid(p_fail).float().cpu())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresh", type=float, default=0.7)
    parser.add_argument("--consec", type=int, default=2)
    parser.add_argument("--holdout", type=int, default=0,
                        help="只用编号 >= holdout 的条目（0=全部，训练集内评估）")
    args = parser.parse_args()

    probe = FailProbe(HERE / "ckpt" / "wm_latest.pt")
    files = sorted(ROLL_DIR.glob("rollout_*.hdf5"))
    files = [p for p in files
             if int(p.stem.split("_")[1]) >= args.holdout]

    traces, labels, names = [], [], []
    for p in files:
        with h5py.File(p, "r") as f:
            n = f["observations/qpos"].shape[0]
            row = int(json.loads(f.attrs["cfg"])["target_seg"][1])
            ts_pts = list(range(0, n - CHUNK, CHUNK))
            tr = [probe.fail_prob(f, t, row) for t in ts_pts]
            traces.append((np.array(ts_pts) / 50.0, np.array(tr)))
            labels.append(not bool(f.attrs["success"]))
            names.append(p.stem)

    # 报警规则评估
    n_fail = sum(labels)
    n_ok = len(labels) - n_fail
    recall, false_alarm, leads = 0, 0, []
    for (tsec, tr), is_fail in zip(traces, labels):
        over = tr > args.thresh
        alarm_at = None
        for i in range(len(over) - args.consec + 1):
            if all(over[i:i + args.consec]):
                alarm_at = tsec[i + args.consec - 1]
                break
        if is_fail and alarm_at is not None:
            recall += 1
            leads.append(tsec[-1] - alarm_at)
        if not is_fail and alarm_at is not None:
            false_alarm += 1
    print(f"样本: 失败 {n_fail} / 成功 {n_ok}")
    print(f"报警规则: 连续 {args.consec} 点 fail_prob > {args.thresh}")
    print(f"召回率: {recall}/{n_fail} = {recall/max(n_fail,1)*100:.0f}%")
    print(f"误报率: {false_alarm}/{n_ok} = {false_alarm/max(n_ok,1)*100:.0f}%")
    if leads:
        print(f"预警提前量: 中位 {np.median(leads):.1f}s, 最小 {np.min(leads):.1f}s")

    # 可视化：失败/成功 episode 的 fail 概率轨迹
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=130, sharey=True)
    for (tsec, tr), is_fail in zip(traces, labels):
        ax = axes[1] if is_fail else axes[0]
        ax.plot(tsec, tr, lw=1.0, alpha=0.55,
                color="#e57373" if is_fail else "#66bb6a")
    for ax, title in zip(axes, (f"成功 episode（n={n_ok}）",
                                f"失败 episode（n={n_fail}）")):
        ax.axhline(args.thresh, color="gray", ls="--", lw=1.2,
                   label=f"报警阈值 {args.thresh}")
        ax.set_xlabel("episode 时间 (s)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="upper left")
    axes[0].set_ylabel("WM 预测失败概率")
    fig.suptitle("世界模型作为安全监控器：逐决策点失败概率轨迹", fontsize=13)
    fig.tight_layout()
    out = IMG_DIR / "wm_alarm_traces.png"
    fig.savefig(out)
    print(f"轨迹图: {out}")


if __name__ == "__main__":
    main()
