"""采集 ACT 策略自己的 rollout 数据（世界模型训练集）。

与演示数据的本质区别：**包含失败**。世界模型要学的是"在这个观测下执行
这个动作块会发生什么"，失败样本（滑脱/弹出/超时）正是价值所在——
演示数据全是成功，学不出风险。

每条 episode 记录（HDF5）：
    /observations/qpos        (T, 14)   本体感知
    /observations/images/{cam}(T,) vlen JPEG（三路机载相机）
    /action                   (T, 14)   实际执行的控制
    /privileged/progress      (T,)      进度计数 0~4（latched/torn/seg 入盒/板回槽）
    /privileged/seg_pos       (T, 3)    目标格真值位置（仅训练标签用）
    attrs: success, phys, cfg

采集分布：标称物理 + 物理随机化各半（扰动档失败率更高，风险样本更多）。

运行:
    ../../.venv/Scripts/python.exe collect_rollouts.py --n 100
"""

import argparse
import io
import json
import time
from pathlib import Path

import h5py
import mujoco
import numpy as np
from PIL import Image

import tear_scene as ts
from eval_act import EXEC_STEPS, ActPolicy
from pill_env import CAMS, IMG_H, IMG_W, PillTearEnv
from train_act import CHUNK

HERE = Path(__file__).resolve().parent
ROLL_DIR = HERE / "rollouts"
ROLL_DIR.mkdir(exist_ok=True)
EPISODE_SECS = 55


class RolloutRecorder:
    def __init__(self, env):
        self.renderer = mujoco.Renderer(env.model, height=IMG_H, width=IMG_W)
        self.env = env
        self.qpos, self.actions, self.progress, self.seg_pos = [], [], [], []
        self.jpegs = {cam: [] for cam in CAMS}
        self._ever = np.zeros(4)   # latched/torn/seg 入盒/全流程，单调事件累积

    def tick(self, obs, action):
        env = self.env
        self.qpos.append(obs["qpos"].astype(np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        for cam in CAMS:
            buf = io.BytesIO()
            Image.fromarray(obs[cam]).save(buf, format="jpeg", quality=60)
            self.jpegs[cam].append(np.frombuffer(buf.getvalue(), dtype=np.uint8))
        seg_ok, board_ok = env._success()
        now = np.array([env._latched, not env._aws, seg_ok,
                        seg_ok and board_ok], dtype=float)
        self._ever = np.maximum(self._ever, now)   # 进度只进不退（latch 释放是任务需要）
        self.progress.append(float(self._ever.sum()))
        self.seg_pos.append(
            env.data.body(ts.seg_name(*env.cfg.target_seg)).xpos.astype(np.float32).copy())

    def save(self, path, cfg, phys, success):
        with h5py.File(path, "w") as f:
            f.attrs["success"] = success
            f.attrs["phys"] = json.dumps(
                {k: (list(v) if isinstance(v, np.ndarray) else v)
                 for k, v in (phys or {}).items()})
            f.attrs["cfg"] = json.dumps({
                "box_a_xy": list(cfg.box_a_xy), "box_b_xy": list(cfg.box_b_xy),
                "base_work": list(cfg.base_work),
                "target_seg": list(cfg.target_seg)})
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.stack(self.qpos))
            f.create_dataset("action", data=np.stack(self.actions))
            priv = f.create_group("privileged")
            priv.create_dataset("progress", data=np.array(self.progress, np.float32))
            priv.create_dataset("seg_pos", data=np.stack(self.seg_pos))
            imgs = obs.create_group("images")
            dt = h5py.vlen_dtype(np.uint8)
            for cam in CAMS:
                d = imgs.create_dataset(cam, (len(self.jpegs[cam]),), dtype=dt)
                for i, j in enumerate(self.jpegs[cam]):
                    d[i] = j

    def close(self):
        self.renderer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default="act_latest.pt")
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    from tear_refine_env import sample_phys

    policy = ActPolicy(HERE / "ckpt" / args.ckpt)
    env = PillTearEnv(seed=args.seed)
    phys_rng = np.random.default_rng(args.seed + 555)
    n_succ = 0
    t0 = time.time()
    for ep in range(args.start, args.start + args.n):
        phys = None
        if ep % 2 == 1:                     # 奇数条走物理随机化档
            phys = sample_phys(phys_rng)
            phys.pop("sense", None)
        obs, info = env.reset(seed=50000 + ep,
                              options={"phys": phys} if phys else None)
        target_row = int(info["cfg"].target_seg[1])
        recorder = RolloutRecorder(env)
        max_steps = int(EPISODE_SECS * 50)
        steps, done, last = 0, False, {}
        while steps < max_steps and not done:
            chunk = policy.predict_chunk(obs, target_row)
            for k in range(min(EXEC_STEPS, CHUNK)):
                recorder.tick(obs, chunk[k])
                obs, r, term, trunc, last = env.step(chunk[k])
                steps += 1
                if term or steps >= max_steps:
                    done = term
                    break
        success = bool(last.get("board_returned"))
        n_succ += success
        path = ROLL_DIR / f"rollout_{ep:03d}_{'ok' if success else 'fail'}.hdf5"
        recorder.save(path, env.cfg, phys, success)
        recorder.close()
        print(f"[{ep:03d}] {'成功' if success else '失败'} "
              f"({'扰动' if phys else '标称'}) {steps} steps "
              f"进度终值 {recorder.progress[-1]:.0f} -> {path.name}", flush=True)
    print(f"\n完成: 成功 {n_succ}/{args.n}，{(time.time()-t0)/60:.0f} 分钟")


if __name__ == "__main__":
    main()
