"""动作条件世界模型：给定当前观测 + 候选动作块，预测执行后果。

角色：ACT 的"把关人"。ACT 提议 K 个候选动作块，WM 预测每块的
    - Δ进度（chunk 结束时任务进度 - 当前进度，0~4 事件计数）
    - 风险（该 episode 最终失败的概率）
    - 未来 qpos（自监督辅助头，稳定表征学习）
推理时选"预测进度最高、风险最低"的块执行。

训练数据：collect_rollouts.py 采集的 ACT 自身 rollout（含失败——
演示数据全是成功，学不出风险；失败样本才是 WM 的价值来源）。
特权信息（真值进度）只作训练标签，推理时 WM 只吃学生观测。

结构：共享 ACT 的视觉词汇——ResNet18 backbone（冻结前几层）+
qpos + 动作块 MLP 编码 → Transformer 编码 → 三个预测头。

运行:
    ../../.venv/Scripts/python.exe train_wm.py --steps 8000
"""

import argparse
import io
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from train_act import CAMS, CHUNK, D_MODEL, IMG_MEAN, IMG_STD

HERE = Path(__file__).resolve().parent
ROLL_DIR = HERE / "rollouts"
CKPT_DIR = HERE / "ckpt"
IMAGE_DIR = HERE.parents[1] / "docs" / "assets" / "images"


class RolloutDataset(Dataset):
    """样本 = (o_t, a_{t:t+CHUNK}) → (Δprogress, fail, qpos_{t+CHUNK})。"""

    def __init__(self, files, stride=5):
        self.files = files
        self.index = []
        qpos_all, act_all = [], []
        self.meta = []
        for fi, path in enumerate(files):
            with h5py.File(path, "r") as f:
                n = f["observations/qpos"].shape[0]
                qpos_all.append(f["observations/qpos"][:])
                act_all.append(f["action"][:])
                self.meta.append({
                    "success": bool(f.attrs["success"]),
                    "target_row": int(json.loads(f.attrs["cfg"])["target_seg"][1]),
                })
            self.index += [(fi, t) for t in range(0, n - CHUNK, stride)]
        qpos_all = np.concatenate(qpos_all)
        act_all = np.concatenate(act_all)
        self.stats = {
            "qpos_mean": qpos_all.mean(0), "qpos_std": qpos_all.std(0) + 1e-4,
            "act_mean": act_all.mean(0), "act_std": act_all.std(0) + 1e-4,
        }
        self._h5 = {}

    def __len__(self):
        return len(self.index)

    def _file(self, fi):
        if fi not in self._h5:
            self._h5[fi] = h5py.File(self.files[fi], "r")
        return self._h5[fi]

    def __getitem__(self, i):
        fi, t = self.index[i]
        f = self._file(fi)
        imgs = []
        for cam in CAMS:
            arr = np.asarray(Image.open(io.BytesIO(
                f[f"observations/images/{cam}"][t].tobytes())),
                dtype=np.float32) / 255.0
            imgs.append((arr - IMG_MEAN) / IMG_STD)
        imgs = np.stack(imgs).transpose(0, 3, 1, 2)
        qpos = (f["observations/qpos"][t] - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        acts = (f["action"][t:t + CHUNK] - self.stats["act_mean"]) / self.stats["act_std"]
        prog = f["privileged/progress"]
        d_prog = prog[t + CHUNK - 1] - prog[t]
        qpos_next = (f["observations/qpos"][t + CHUNK - 1]
                     - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        fail = 0.0 if self.meta[fi]["success"] else 1.0
        return (imgs.astype(np.float32), qpos.astype(np.float32),
                acts.astype(np.float32).reshape(-1),
                np.float32(d_prog), np.float32(fail),
                qpos_next.astype(np.float32),
                self.meta[fi]["target_row"])


class WorldModel(nn.Module):
    def __init__(self, n_cams=3, d=D_MODEL):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        bb = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(bb.children())[:-2])
        self.proj = nn.Conv2d(512, d, 1)
        self.qpos_in = nn.Linear(14, d)
        self.act_in = nn.Sequential(nn.Linear(CHUNK * 14, 512), nn.GELU(),
                                    nn.Linear(512, d))
        self.target_emb = nn.Embedding(2, d)
        n_img = n_cams * 8 * 10
        self.pos_emb = nn.Parameter(torch.randn(1, n_img + 3, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, nhead=8, dim_feedforward=1024,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.head_prog = nn.Linear(d, 1)
        self.head_fail = nn.Linear(d, 1)
        self.head_qpos = nn.Linear(d, 14)

    def forward(self, imgs, qpos, acts, target_row):
        B, N, C, H, W = imgs.shape
        feat = self.proj(self.backbone(imgs.reshape(B * N, C, H, W)))
        feat = feat.flatten(2).transpose(1, 2).reshape(B, -1, D_MODEL)
        toks = torch.cat([feat, self.qpos_in(qpos).unsqueeze(1),
                          self.act_in(acts).unsqueeze(1),
                          self.target_emb(target_row).unsqueeze(1)], dim=1)
        x = self.encoder(toks + self.pos_emb)
        h = x[:, -2]                       # 动作 token 的输出作为决策摘要
        return (self.head_prog(h).squeeze(-1), self.head_fail(h).squeeze(-1),
                self.head_qpos(h))


def train(steps=8000, batch=24, lr=1e-4, holdout=0):
    files = sorted(ROLL_DIR.glob("rollout_*.hdf5"))
    if holdout:
        files = [p for p in files if int(p.stem.split("_")[1]) < holdout]
        print(f"留出集: 编号 >= {holdout} 不参与训练")
    assert files, "先运行 collect_rollouts.py"
    ds = RolloutDataset(files)
    n_fail = sum(1 for m in ds.meta if not m["success"])
    print(f"数据: {len(files)} 条 rollout（失败 {n_fail}）, {len(ds)} 样本")
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0,
                        pin_memory=True, drop_last=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = WorldModel().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    scaler = torch.amp.GradScaler(dev)
    bce = nn.BCEWithLogitsLoss()

    log = {"step": [], "loss": [], "prog": [], "fail": []}
    step, running, t0 = 0, np.zeros(3), time.time()
    model.train()
    while step < steps:
        for imgs, qpos, acts, d_prog, fail, qpos_next, tgt in loader:
            imgs, qpos = imgs.to(dev), qpos.to(dev)
            acts, tgt = acts.to(dev), tgt.to(dev)
            d_prog, fail = d_prog.to(dev), fail.to(dev)
            qpos_next = qpos_next.to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16):
                p_prog, p_fail, p_qpos = model(imgs, qpos, acts, tgt)
                l_prog = torch.abs(p_prog - d_prog).mean()
                l_fail = bce(p_fail, fail)
                l_qpos = torch.abs(p_qpos - qpos_next).mean()
                loss = l_prog + 0.5 * l_fail + 0.2 * l_qpos
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += [loss.item(), l_prog.item(), l_fail.item()]
            step += 1
            if step % 200 == 0:
                avg = running / 200
                running = np.zeros(3)
                log["step"].append(step)
                log["loss"].append(avg[0])
                log["prog"].append(avg[1])
                log["fail"].append(avg[2])
                print(f"step {step:5d}/{steps}  loss {avg[0]:.4f} "
                      f"(prog {avg[1]:.4f} fail {avg[2]:.4f})  "
                      f"{step/(time.time()-t0):.1f} it/s", flush=True)
            if step % 2000 == 0 or step == steps:
                torch.save({"model": model.state_dict(),
                            "stats": {k: v.tolist() for k, v in ds.stats.items()},
                            "step": step}, CKPT_DIR / "wm_latest.pt")
            if step >= steps:
                break

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=130)
    ax.plot(log["step"], log["loss"], lw=1.5, label="总损失")
    ax.plot(log["step"], log["prog"], lw=1.2, label="Δ进度 L1")
    ax.plot(log["step"], log["fail"], lw=1.2, label="失败 BCE")
    ax.set_xlabel("训练步数")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title(f"世界模型训练曲线（{len(files)} 条 rollout, {len(ds)} 样本）")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "wm_train_loss.png")
    print(f"完成: ckpt/wm_latest.pt, 曲线 {IMAGE_DIR / 'wm_train_loss.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--holdout", type=int, default=0,
                        help="编号 >= N 的 rollout 留出不训练（监控器评估用）")
    args = parser.parse_args()
    train(args.steps, args.batch, holdout=args.holdout)
