"""操作原语 BC 训练（原语分解架构：单臂 / 单腕相机 / 高清 / 短时程）。

与全流程 train_act.py 的区别：
    - 数据只取演示中该原语相位窗口的帧（/phase 字段切分），
      且只要求**该原语出口谓词成功**（attrs prim_ok）——全流程失败的
      episode 里成功的原语片段仍是有效演示；
    - 观测 = 该原语执行臂的腕相机（480x640 高清，特征网格 15x20
      保留空间细节——毫米级定位是端到端 ACT 的瓶颈）+ 14 维 qpos；
    - 动作 = 执行臂 7 维（另一臂由衔接控制段负责，不学习）；
    - 时程 = 原语窗口（数百 tick），chunk 相对占比大、复合误差短。

数据：demos/prim_*.hdf5（collect_demos.py --prefix prim --wrist-only --hires
      --strict --entry-jitter 0.004 产出）

运行:
    ../../.venv/Scripts/python.exe train_prim.py --prim p2_tear_seg --steps 16000
"""

import argparse
import io
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from primitives import PRIM_SPECS
from run_full_demo import PHASE_IDS
from train_act import D_MODEL, IMG_MEAN, IMG_STD, ACTLite

HERE = Path(__file__).resolve().parent
DEMO_DIR = HERE / "demos"
CKPT_DIR = HERE / "ckpt"
CKPT_DIR.mkdir(exist_ok=True)
IMAGE_DIR = HERE.parents[1] / "docs" / "assets" / "images"

CHUNK_PRIM = 50          # 1 s 动作块（与全流程一致；exec 步数评测时可调）
FEAT_HW = (15, 20)       # 480x640 经 ResNet18 /32 的天然特征网格（不池化）


# ---------------- 数据 ----------------
class PrimDataset(Dataset):
    """原语片段数据集：(episode, t) 索引只覆盖该原语相位窗口。

    动作 chunk 在窗口内截取，窗口末尾用最后动作填充（mask 置 0）——
    原语结束后的动作属于衔接段，不能学。
    """

    def __init__(self, files, prim, chunk=CHUNK_PRIM):
        spec = PRIM_SPECS[prim]
        self.cam = spec.cam
        self.sl = slice(spec.act_lo, spec.act_hi)
        self.chunk = chunk
        self.files, self.index, self.target_row = [], [], []
        qpos_all, act_all = [], []
        pid = PHASE_IDS[prim]
        n_skip = 0
        for path in files:
            with h5py.File(path, "r") as f:
                if not json.loads(f.attrs.get("prim_ok", "{}")).get(prim, False):
                    n_skip += 1
                    continue
                ph = f["phase"][:]
                tt = np.flatnonzero(ph == pid)
                if len(tt) == 0:
                    n_skip += 1
                    continue
                fi = len(self.files)
                self.files.append(path)
                self.target_row.append(int(json.loads(f.attrs["cfg"])["target_seg"][1]))
                # 原语窗口可能非连续（理论上单窗口；按连续段处理更稳）
                splits = np.split(tt, np.flatnonzero(np.diff(tt) > 1) + 1)
                for seg in splits:
                    a, b = int(seg[0]), int(seg[-1]) + 1
                    self.index += [(fi, t, b) for t in range(a, b)]
                    qpos_all.append(f["observations/qpos"][a:b])
                    act_all.append(f["action"][a:b, self.sl])
        assert self.index, f"没有 {prim} 的有效片段（跳过 {n_skip} 个文件）"
        qpos_all = np.concatenate(qpos_all)
        act_all = np.concatenate(act_all)
        self.stats = {
            "qpos_mean": qpos_all.mean(0), "qpos_std": qpos_all.std(0) + 1e-4,
            "act_mean": act_all.mean(0), "act_std": act_all.std(0) + 1e-4,
        }
        self.n_skip = n_skip
        self._h5 = {}

    def __len__(self):
        return len(self.index)

    def _file(self, fi):
        if fi not in self._h5:
            self._h5[fi] = h5py.File(self.files[fi], "r")
        return self._h5[fi]

    def __getitem__(self, i):
        fi, t, seg_end = self.index[i]
        f = self._file(fi)
        arr = np.asarray(Image.open(io.BytesIO(
            f[f"observations/images/{self.cam}"][t].tobytes())),
            dtype=np.float32) / 255.0
        img = ((arr - IMG_MEAN) / IMG_STD).transpose(2, 0, 1)[None]   # (1, C, H, W)
        qpos = (f["observations/qpos"][t] - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        end = min(t + self.chunk, seg_end)
        act = f["action"][t:end, self.sl]
        pad = self.chunk - len(act)
        mask = np.ones(self.chunk, dtype=np.float32)
        if pad > 0:
            act = np.concatenate([act, np.repeat(act[-1:], pad, 0)])
            mask[self.chunk - pad:] = 0.0
        act = (act - self.stats["act_mean"]) / self.stats["act_std"]
        return (img.astype(np.float32), qpos.astype(np.float32),
                act.astype(np.float32), mask, self.target_row[fi])


# ---------------- 训练 ----------------
def train(prim, steps, batch, lr=1e-4, chunk=CHUNK_PRIM, prefix="prim",
          out_tag=None, max_files=None):
    out_tag = out_tag or prim
    files = []
    for p in prefix.split(","):        # 逗号分隔多前缀（全流程演示 + 原语补采）
        files += sorted(DEMO_DIR.glob(f"{p}_*.hdf5"))
    if max_files:
        files = files[:max_files]
    assert files, "没有原语演示数据，先运行 collect_demos.py --prefix prim"
    print(f"扫描 {len(files)} 个数据文件...", flush=True)
    ds = PrimDataset(files, prim, chunk=chunk)
    print(f"数据: {len(ds.files)} 条有效片段（跳过 {ds.n_skip}）, {len(ds)} 个训练样本")
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0,
                        pin_memory=True, drop_last=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ACTLite(n_cams=1, chunk=chunk, act_dim=7, feat_hw=FEAT_HW).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    scaler = torch.amp.GradScaler(dev)

    log = {"step": [], "loss": []}
    step, t0, running = 0, time.time(), 0.0
    model.train()
    while step < steps:
        for imgs, qpos, act, mask, tgt in loader:
            imgs, qpos = imgs.to(dev, non_blocking=True), qpos.to(dev, non_blocking=True)
            act, mask = act.to(dev, non_blocking=True), mask.to(dev, non_blocking=True)
            tgt = tgt.to(dev, non_blocking=True)
            # 轻量传感器 dropout：原语内相位混淆风险低（时程短），
            # 保留少量 dropout 防止完全忽略 qpos
            drop = (torch.rand(imgs.shape[0], device=dev) < 0.1).view(-1, 1, 1, 1, 1)
            imgs = imgs * (~drop)
            with torch.autocast(dev, dtype=torch.bfloat16):
                pred = model(imgs, qpos, tgt)
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
                            "prim": prim, "chunk": chunk, "feat_hw": FEAT_HW,
                            "step": step}, CKPT_DIR / f"{out_tag}_latest.pt")
            if step >= steps:
                break

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=130)
    ax.plot(log["step"], log["loss"], lw=1.5)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("动作块 L1 损失（归一化）")
    ax.set_title(f"原语 {prim} BC 训练曲线（{len(ds.files)} 条片段，chunk={chunk}）")
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / f"{out_tag}_train_loss.png")
    with open(CKPT_DIR / f"{out_tag}_log.json", "w") as f:
        json.dump(log, f)
    print(f"完成: ckpt/{out_tag}_latest.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prim", type=str, required=True,
                        choices=list(PRIM_SPECS))
    parser.add_argument("--steps", type=int, default=16000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--chunk", type=int, default=CHUNK_PRIM)
    parser.add_argument("--prefix", type=str, default="prim")
    parser.add_argument("--out-tag", type=str, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()
    train(args.prim, args.steps, args.batch, lr=args.lr, chunk=args.chunk,
          prefix=args.prefix, out_tag=args.out_tag, max_files=args.max_files)
