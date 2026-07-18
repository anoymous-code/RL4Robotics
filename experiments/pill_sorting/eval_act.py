"""ACT-lite 策略评测：随机场景 rollout + 成功率统计 + 三机位视频。

推理方式：每次前向输出 chunk=50 步动作块，开环执行前 K 步后重推理
（K=25，即 0.5 s 重规划一次）。

运行:
    ../../.venv/Scripts/python.exe eval_act.py --n 20            # 评成功率
    ../../.venv/Scripts/python.exe eval_act.py --n 3 --video     # 附 rollout 视频
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

import tear_scene as ts
from pill_env import CAMS, PillTearEnv
from train_act import CHUNK, IMG_MEAN, IMG_STD, ACTLite

HERE = Path(__file__).resolve().parent
VIDEO_DIR = HERE.parents[1] / "docs" / "assets" / "videos"
EXEC_STEPS = 25          # 每个动作块开环执行的步数
EPISODE_SECS = 55


class ActPolicy:
    def __init__(self, ckpt_path, device="cuda"):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model = ACTLite().to(device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.stats = {k: np.array(v, dtype=np.float32) for k, v in ckpt["stats"].items()}
        self.device = device
        print(f"载入 {ckpt_path}（训练步数 {ckpt.get('step')}）")

    @torch.no_grad()
    def predict_chunk(self, obs):
        imgs = np.stack([
            ((obs[cam].astype(np.float32) / 255.0) - IMG_MEAN) / IMG_STD
            for cam in CAMS]).transpose(0, 3, 1, 2)
        qpos = (obs["qpos"].astype(np.float32) - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        imgs_t = torch.from_numpy(imgs).unsqueeze(0).to(self.device)
        qpos_t = torch.from_numpy(qpos).unsqueeze(0).to(self.device)
        with torch.autocast(self.device, dtype=torch.bfloat16):
            chunk = self.model(imgs_t, qpos_t)[0].float().cpu().numpy()
        return chunk * self.stats["act_std"] + self.stats["act_mean"]


def rollout(env, policy, video_path=None, seed=None):
    obs, info = env.reset(seed=seed)
    writer = None
    if video_path:
        writer = imageio.get_writer(video_path, fps=25, macro_block_size=1)
    steps, done = 0, False
    max_steps = int(EPISODE_SECS * 50)
    last_info = {}
    while steps < max_steps and not done:
        chunk = policy.predict_chunk(obs)
        for k in range(min(EXEC_STEPS, CHUNK)):
            obs, reward, terminated, truncated, last_info = env.step(chunk[k])
            steps += 1
            if writer and steps % 2 == 0:
                bottom = np.hstack([obs["wrist_cam_left"], obs["wrist_cam_right"]])[:, ::2]
                writer.append_data(np.vstack([obs["head_cam"], bottom]))
            if terminated or steps >= max_steps:
                done = terminated
                break
    if writer:
        writer.close()
    return last_info


def main(n, ckpt, video):
    env = PillTearEnv(seed=1234)
    policy = ActPolicy(HERE / "ckpt" / ckpt)
    n_seg, n_full = 0, 0
    for ep in range(n):
        vp = None
        if video and ep < 3:
            vp = VIDEO_DIR / f"act_rollout_{ep}.mp4"
        info = rollout(env, policy, video_path=vp, seed=10000 + ep)
        n_seg += bool(info.get("seg_in_box_b"))
        n_full += bool(info.get("board_returned"))
        print(f"[ep {ep:02d}] 撕剪入盒 B: {info.get('seg_in_box_b')}, "
              f"全流程: {info.get('board_returned')}" + (f" 视频 {vp}" if vp else ""))
    print(f"\n成功率: 撕剪入盒 B {n_seg}/{n}, 全流程 {n_full}/{n}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--ckpt", type=str, default="act_latest.pt")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    main(args.n, args.ckpt, args.video)
