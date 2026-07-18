"""对照实验：在 PillTearEnv 中开环重放演示动作序列。

- 重放成功 → 环境与演示对齐，失败在策略本身；
- 重放失败 → env reset 状态与采集时不一致（问题在环境）。

另做 qpos 依赖检查：把 qpos 置零/加噪，看策略输出变化量——
判断模型是否只学了 "输出 ≈ qpos" 的恒等捷径而忽略图像。
"""

import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

import tear_scene as ts
from eval_act import ActPolicy
from pill_env import CAMS, PillTearEnv

HERE = Path(__file__).resolve().parent

# ---- A. 开环重放 ----
path = sorted((HERE / "demos").glob("episode_*_ok.hdf5"))[0]
with h5py.File(path, "r") as f:
    actions = f["action"][:]
    cfg_d = json.loads(f.attrs["cfg"])
cfg = ts.SceneCfg(box_a_xy=tuple(cfg_d["box_a_xy"]), box_b_xy=tuple(cfg_d["box_b_xy"]),
                  base_work=tuple(cfg_d["base_work"]), target_seg=tuple(cfg_d["target_seg"]))
env = PillTearEnv(seed=0)
obs, info = env.reset(options={"cfg": cfg})
last = {}
for t in range(len(actions)):
    obs, r, term, trunc, last = env.step(actions[t])
    if t % 200 == 0 or term:
        sp = env.data.body("strip").xpos
        print(f"    t={t:5d} latched={env._latched} 已断weld={12-len(env._aws)} "
              f"strip=({sp[0]:+.3f},{sp[1]:+.3f},{sp[2]:+.3f}) "
              f"lgrip_cmd={env.data.ctrl[env._lgrip_act]:.4f}")
    if term:
        break
print(f"[A] 开环重放 {path.name}: 撕剪入盒B={last.get('seg_in_box_b')}, "
      f"回槽={last.get('board_returned')}（{t+1} 步）")

# ---- B. qpos 依赖检查 ----
policy = ActPolicy(HERE / "ckpt" / "act_latest.pt")
obs2, _ = env.reset(options={"cfg": cfg})
base = policy.predict_chunk(obs2, cfg.target_seg[1])
obs_z = dict(obs2)
obs_z["qpos"] = obs2["qpos"] + np.random.default_rng(0).normal(0, 0.3, 14)
pert = policy.predict_chunk(obs_z, cfg.target_seg[1])
print(f"[B] qpos 加噪 σ=0.3 rad 后输出变化: L1 {np.abs(base - pert).mean():.4f} rad "
      f"（若 ≈0.3 说明输出跟随 qpos 恒等；若 ≈0 说明忽略 qpos）")

blank = dict(obs2)
for cam in CAMS:
    blank[cam] = np.zeros_like(obs2[cam])
b2 = policy.predict_chunk(blank, cfg.target_seg[1])
print(f"[B] 图像全黑后输出变化: L1 {np.abs(base - b2).mean():.4f} rad "
      f"（若 ≈0 说明模型完全没用图像）")
env.close()
