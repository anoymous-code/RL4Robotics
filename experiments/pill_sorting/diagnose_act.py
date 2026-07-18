"""ACT 策略失败诊断：分离"训练没学会"与"训练-推理不一致"。

检查三层：
  A. 训练分布内：用演示数据自身的 obs 预测，与记录的动作块对比（应当很小）；
  B. 环境分布：同一 cfg 下 env.reset 的 obs 预测，与该演示第一个动作块对比；
  C. 图像一致性：训练 JPEG 帧 vs env 渲染帧的像素差。
"""

import io
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

import tear_scene as ts
from eval_act import ActPolicy
from pill_env import CAMS, PillTearEnv
from train_act import CHUNK

HERE = Path(__file__).resolve().parent

policy = ActPolicy(HERE / "ckpt" / "act_latest.pt")

# ---- A. 训练分布内预测误差 ----
path = sorted((HERE / "demos").glob("episode_*_ok.hdf5"))[0]
with h5py.File(path, "r") as f:
    cfg_d = json.loads(f.attrs["cfg"])
    n = f["observations/qpos"].shape[0]
    print(f"{path.name}: {n} ticks")
    for t in (0, 400, 800, 1200):
        obs = {"qpos": f["observations/qpos"][t].astype(np.float64)}
        for cam in CAMS:
            obs[cam] = np.asarray(Image.open(io.BytesIO(
                f[f"observations/images/{cam}"][t].tobytes())))
        pred = policy.predict_chunk(obs)
        act = f["action"][t:t + CHUNK]
        err = np.abs(pred[:len(act)] - act)
        print(f"  [A] t={t:5d}: 动作块 L1 {err.mean():.4f} rad, "
              f"首步 {np.abs(pred[0] - act[0]).mean():.4f}, 最大 {err.max():.3f}")
    first_action = f["action"][0]
    first_qpos = f["observations/qpos"][0]
    jpeg0 = {cam: np.asarray(Image.open(io.BytesIO(
        f[f"observations/images/{cam}"][0].tobytes()))) for cam in CAMS}

# ---- B. 环境分布预测误差（同 cfg） ----
cfg = ts.SceneCfg(box_a_xy=tuple(cfg_d["box_a_xy"]), box_b_xy=tuple(cfg_d["box_b_xy"]),
                  base_work=tuple(cfg_d["base_work"]), target_seg=tuple(cfg_d["target_seg"]))
env = PillTearEnv(seed=0)
obs, _ = env.reset(options={"cfg": cfg})
pred = policy.predict_chunk(obs)
print(f"\n[B] env reset obs 预测 vs 演示首块: L1 {np.abs(pred - first_action).mean():.4f} rad")
print(f"    env qpos vs 演示首帧 qpos: L1 {np.abs(obs['qpos'] - first_qpos).mean():.5f}")

# ---- C. 图像一致性 ----
print("\n[C] 图像差异（env 渲染 vs 演示 JPEG 首帧）:")
for cam in CAMS:
    d = np.abs(obs[cam].astype(np.float32) - jpeg0[cam].astype(np.float32))
    print(f"    {cam}: 平均 {d.mean():.1f} / 最大 {d.max():.0f} (0-255)")
    Image.fromarray(np.hstack([obs[cam], jpeg0[cam]])).save(
        HERE / "debug" / f"diag_{cam}.png")
env.close()
print("对比图: debug/diag_*.png（左 env / 右训练数据）")
