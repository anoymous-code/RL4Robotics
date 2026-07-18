"""闭环相位诊断：策略 rollout 时轨迹推进到演示的哪个"相位"。

同一 cfg 下滚动策略，每 50 步用 qpos 最近邻找到对应的演示帧号（相位）。
相位持续推进 → 只是慢/略偏；相位停滞 → 卡死在该阶段（打印卡点上下文）。
"""

import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

import tear_scene as ts
from eval_act import EXEC_STEPS, ActPolicy
from pill_env import CAMS, PillTearEnv
from train_act import CHUNK

HERE = Path(__file__).resolve().parent

path = sorted((HERE / "demos").glob("episode_*_ok.hdf5"))[0]
with h5py.File(path, "r") as f:
    demo_qpos = f["observations/qpos"][:]
    demo_act = f["action"][:]
    cfg_d = json.loads(f.attrs["cfg"])
cfg = ts.SceneCfg(box_a_xy=tuple(cfg_d["box_a_xy"]), box_b_xy=tuple(cfg_d["box_b_xy"]),
                  base_work=tuple(cfg_d["base_work"]), target_seg=tuple(cfg_d["target_seg"]))

policy = ActPolicy(HERE / "ckpt" / "act_latest.pt")
env = PillTearEnv(seed=0)
obs, _ = env.reset(options={"cfg": cfg})
row = cfg.target_seg[1]

print(f"演示长度 {len(demo_qpos)}，执行段 {EXEC_STEPS} 步/块")
steps = 0
while steps < 2200:
    chunk = policy.predict_chunk(obs, row)
    # 对照：当前相位的演示动作块
    phase = int(np.argmin(np.linalg.norm(demo_qpos - obs["qpos"], axis=1)))
    demo_chunk = demo_act[phase:phase + CHUNK]
    L = min(len(demo_chunk), CHUNK)
    diff = np.abs(chunk[:L] - demo_chunk).mean()
    # chunk 内部的运动量（是否在输出"静止"）
    motion = np.abs(np.diff(chunk, axis=0)).sum(0).max()
    qerr = np.linalg.norm(demo_qpos[phase] - obs["qpos"])
    print(f"step {steps:5d} | 相位 {phase:5d} ({phase/len(demo_qpos)*100:4.0f}%) "
          f"qpos偏差 {qerr:.3f} | 预测vs演示块 L1 {diff:.4f} | 块运动量 {motion:.3f}")
    for k in range(EXEC_STEPS):
        obs, r, term, trunc, info = env.step(chunk[k])
        steps += 1
        if term:
            break
    if term:
        break
print("结束:", info)
env.close()
