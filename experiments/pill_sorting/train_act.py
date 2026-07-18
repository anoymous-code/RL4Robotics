"""ACT-lite 模仿学习训练（精简版 Action Chunking Transformer）。

结构（对齐 ACT 思想，去掉 CVAE 以简化）：
    3 路相机 → ResNet18（ImageNet 预训练）→ 1x1 conv 到 d 维 token（80/相机）
    + qpos token + chunk 个可学习查询 token → TransformerEncoder
    → 查询 token 输出 → 线性头 → 未来 chunk 步的 14 维关节目标（L1 损失）

数据：demos/episode_*_ok.hdf5（collect_demos.py 产出，50 Hz）

运行:
    ../../.venv/Scripts/python.exe train_act.py --steps 20000 --batch 16
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

HERE = Path(__file__).resolve().parent
DEMO_DIR = HERE / "demos"
CKPT_DIR = HERE / "ckpt"
CKPT_DIR.mkdir(exist_ok=True)
IMAGE_DIR = HERE.parents[1] / "docs" / "assets" / "images"

CAMS = ("head_cam", "wrist_cam_left", "wrist_cam_right")
CHUNK = 50
D_MODEL = 256
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------- 数据 ----------------
class DemoDataset(Dataset):
    """(episode, t) 平铺索引；懒打开 HDF5（DataLoader worker 各自持有句柄）。"""

    def __init__(self, files):
        self.files = files
        self.index = []
        qpos_all, act_all = [], []
        for fi, path in enumerate(files):
            with h5py.File(path, "r") as f:
                n = f["observations/qpos"].shape[0]
                qpos_all.append(f["observations/qpos"][:])
                act_all.append(f["action"][:])
            self.index += [(fi, t) for t in range(n)]
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
        n = f["observations/qpos"].shape[0]
        imgs = []
        for cam in CAMS:
            arr = np.asarray(Image.open(io.BytesIO(
                f[f"observations/images/{cam}"][t].tobytes())), dtype=np.float32) / 255.0
            imgs.append((arr - IMG_MEAN) / IMG_STD)
        imgs = np.stack(imgs).transpose(0, 3, 1, 2)          # (3cam, C, H, W)
        qpos = (f["observations/qpos"][t] - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        end = min(t + CHUNK, n)
        act = f["action"][t:end]
        pad = CHUNK - len(act)
        mask = np.ones(CHUNK, dtype=np.float32)
        if pad > 0:
            act = np.concatenate([act, np.repeat(act[-1:], pad, 0)])
            mask[len(act) - pad:] = 0.0
        act = (act - self.stats["act_mean"]) / self.stats["act_std"]
        return (imgs.astype(np.float32), qpos.astype(np.float32),
                act.astype(np.float32), mask)


# ---------------- 模型 ----------------
class ACTLite(nn.Module):
    def __init__(self, n_cams=3, chunk=CHUNK, d=D_MODEL, act_dim=14):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        bb = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(bb.children())[:-2])   # → (512, H/32, W/32)
        self.proj = nn.Conv2d(512, d, 1)
        self.qpos_in = nn.Linear(14, d)
        n_img_tokens = n_cams * 8 * 10                             # 240x320 → 8x10
        self.pos_emb = nn.Parameter(torch.randn(1, n_img_tokens + 1 + chunk, d) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, chunk, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, nhead=8, dim_feedforward=1024,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=4)
        self.head = nn.Linear(d, act_dim)
        self.chunk = chunk

    def forward(self, imgs, qpos):
        B, N, C, H, W = imgs.shape
        feat = self.proj(self.backbone(imgs.reshape(B * N, C, H, W)))
        feat = feat.flatten(2).transpose(1, 2).reshape(B, -1, D_MODEL)   # (B, N*80, d)
        q_tok = self.qpos_in(qpos).unsqueeze(1)
        x = torch.cat([feat, q_tok, self.queries.expand(B, -1, -1)], dim=1)
        x = self.encoder(x + self.pos_emb)
        return self.head(x[:, -self.chunk:])


# ---------------- 训练 ----------------
def train(steps, batch, lr=1e-4, out_tag="act", max_files=None):
    files = sorted(DEMO_DIR.glob("episode_*_ok.hdf5"))
    if max_files:
        files = files[:max_files]
    assert files, "没有演示数据，先运行 collect_demos.py"
    print(f"扫描 {len(files)} 个数据文件...", flush=True)
    ds = DemoDataset(files)
    print(f"数据: {len(files)} 条演示, {len(ds)} 个训练样本")
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=4,
                        pin_memory=True, persistent_workers=True, drop_last=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ACTLite().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    scaler = torch.amp.GradScaler(dev)

    log = {"step": [], "loss": []}
    step, t0, running = 0, time.time(), 0.0
    model.train()
    while step < steps:
        for imgs, qpos, act, mask in loader:
            imgs, qpos = imgs.to(dev, non_blocking=True), qpos.to(dev, non_blocking=True)
            act, mask = act.to(dev, non_blocking=True), mask.to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=torch.bfloat16):
                pred = model(imgs, qpos)
                loss = (torch.abs(pred - act).mean(-1) * mask).sum() / mask.sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            step += 1
            if step % 200 == 0:
                avg = running / 200
                running = 0.0
                log["step"].append(step)
                log["loss"].append(avg)
                print(f"step {step:6d}/{steps}  L1 {avg:.4f}  "
                      f"{step / (time.time() - t0):.1f} it/s", flush=True)
            if step % 2000 == 0 or step == steps:
                torch.save({"model": model.state_dict(),
                            "stats": {k: v.tolist() for k, v in ds.stats.items()},
                            "step": step}, CKPT_DIR / f"{out_tag}_latest.pt")
            if step >= steps:
                break

    # 训练曲线
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=130)
    ax.plot(log["step"], log["loss"], lw=1.5)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("动作块 L1 损失（归一化）")
    ax.set_title(f"ACT-lite 训练曲线（{len(files)} 条演示，chunk={CHUNK}）")
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "act_train_loss.png")
    with open(CKPT_DIR / f"{out_tag}_log.json", "w") as f:
        json.dump(log, f)
    print(f"完成: ckpt/{out_tag}_latest.pt, 曲线 {IMAGE_DIR / 'act_train_loss.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-files", type=int, default=None,
                        help="只用前 N 个数据文件（避开采集进程正在写的文件）")
    args = parser.parse_args()
    train(args.steps, args.batch, args.lr, max_files=args.max_files)
