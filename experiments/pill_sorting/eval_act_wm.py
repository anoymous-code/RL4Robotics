"""ACT + 世界模型评测：ACT 提议 K 个候选动作块，WM 把关选最优。

候选生成：ACT 原始块 + (K-1) 个扰动块（臂关节加平滑 OU 噪声）。
评分：预测Δ进度 - λ × 预测失败概率，选分最高的块开环执行。
零噪声候选恒在集合中——WM 若无把握（评分并列），行为退化为纯 ACT。

运行:
    ../../.venv/Scripts/python.exe eval_act_wm.py --n 20 --phys 1.0 --k 8
"""

import argparse
import io
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from eval_act import EXEC_STEPS, ActPolicy, jpeg_roundtrip
from pill_env import CAMS, PillTearEnv
from run_full_demo import QuadCam
from train_act import CHUNK, IMG_MEAN, IMG_STD
from train_wm import WorldModel

HERE = Path(__file__).resolve().parent
VIDEO_DIR = HERE.parents[1] / "docs" / "assets" / "videos"
EPISODE_SECS = 55
NOISE_SIGMA = 0.012     # 随机兜底候选扰动幅度 (rad)，OU 平滑
FAIL_W = 0.2            # 风险权重 λ（失败头是 episode 级 MC 标签，噪声大，降权）
MARGIN = 0.02           # 保守门控：候选须优于原块此幅度（Δ进度单位）才改选。
                        # 第一版无门控时 WM 54/55 步都改选（评分噪声主导），
                        # 等效持续注入执行噪声，标称档直接崩——把关人必须谦逊
# 结构化候选（诊断结论：随机噪声候选评分展布仅 ~0.003——WM 判断正确，
# 白噪声不改变成败；有意义的干预是 RL 精修学到的那类结构化修正）
GRIP_R, WRIST_ROT_R, WRIST_ANG_R = 13, 12, 11   # 右爪/右腕转/右腕俯仰列


class WmGate:
    def __init__(self, ckpt_path, device="cuda"):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model = WorldModel().to(device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.stats = {k: np.array(v, dtype=np.float32)
                      for k, v in ckpt["stats"].items()}
        self.device = device
        print(f"载入世界模型 {ckpt_path}（训练步数 {ckpt.get('step')}）")

    def make_candidates(self, chunk, k, rng):
        """原块 + 结构化变体（接触相位的语义修正）+ 随机兜底。"""
        cands = [chunk]
        for dg in (-0.0012, -0.0006, +0.0006):     # 右爪捏紧/放松
            c = chunk.copy()
            c[:, GRIP_R] = np.maximum(0.0, c[:, GRIP_R] + dg)
            cands.append(c)
        for col, dv in ((WRIST_ROT_R, +0.06), (WRIST_ROT_R, -0.06),
                        (WRIST_ANG_R, +0.05)):     # 扭幅/腕俯仰微调
            c = chunk.copy()
            c[:, col] += dv
            cands.append(c)
        arm_idx = list(range(0, 6)) + list(range(7, 13))
        while len(cands) < k:                      # 随机 OU 兜底
            noise = np.zeros((CHUNK, 14))
            ou = np.zeros(len(arm_idx))
            for t in range(CHUNK):
                ou = 0.9 * ou + rng.normal(0, NOISE_SIGMA, len(arm_idx))
                noise[t, arm_idx] = ou
            cands.append(chunk + noise)
        return np.stack(cands[:k])

    @torch.no_grad()
    def score(self, obs, cands, target_row):
        imgs = np.stack([
            ((jpeg_roundtrip(obs[cam]).astype(np.float32) / 255.0) - IMG_MEAN) / IMG_STD
            for cam in CAMS]).transpose(0, 3, 1, 2)
        qpos = ((obs["qpos"].astype(np.float32) - self.stats["qpos_mean"])
                / self.stats["qpos_std"])
        K = len(cands)
        acts = ((cands.astype(np.float32) - self.stats["act_mean"])
                / self.stats["act_std"]).reshape(K, -1)
        imgs_t = torch.from_numpy(imgs).unsqueeze(0).expand(K, -1, -1, -1, -1).to(self.device)
        qpos_t = torch.from_numpy(qpos).unsqueeze(0).expand(K, -1).to(self.device)
        acts_t = torch.from_numpy(acts).to(self.device)
        tgt_t = torch.full((K,), target_row, dtype=torch.long, device=self.device)
        with torch.autocast(self.device, dtype=torch.bfloat16):
            p_prog, p_fail, _ = self.model(imgs_t, qpos_t, acts_t, tgt_t)
        prog = p_prog.float().cpu().numpy()
        fail = torch.sigmoid(p_fail).float().cpu().numpy()
        return prog - FAIL_W * fail


def rollout(env, policy, gate, k, video_path=None, seed=None, phys=None, rng=None):
    obs, info = env.reset(seed=seed, options={"phys": phys} if phys else None)
    target_row = int(info["cfg"].target_seg[1])
    writer = quad = None
    if video_path:
        writer = imageio.get_writer(video_path, fps=25, quality=7, macro_block_size=1)
        quad = QuadCam(env.model)
    max_steps = int(EPISODE_SECS * 50)
    last, steps, done, n_override = {}, 0, False, 0
    while steps < max_steps and not done:
        chunk = policy.predict_chunk(obs, target_row)
        if gate is not None:
            cands = gate.make_candidates(chunk, k, rng)
            scores = gate.score(obs, cands, target_row)
            best = int(np.argmax(scores))
            if scores[best] - scores[0] < MARGIN:
                best = 0            # 优势不显著，保持纯 ACT 行为
            n_override += (best != 0)
            chunk = cands[best]
        for j in range(min(EXEC_STEPS, CHUNK)):
            obs, r, term, trunc, last = env.step(chunk[j])
            steps += 1
            if writer and steps % 2 == 0:
                writer.append_data(quad.composite(env.data))
            if term or steps >= max_steps:
                done = term
                break
    if writer:
        writer.close()
        quad.close()
    last["overrides"] = n_override
    return last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--phys", type=float, default=0.0)
    parser.add_argument("--ckpt", type=str, default="act_latest.pt")
    parser.add_argument("--wm", type=str, default="wm_latest.pt")
    parser.add_argument("--no-gate", action="store_true", help="纯 ACT 对照")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--tag", type=str, default="act_wm")
    args = parser.parse_args()

    env = PillTearEnv(seed=1234)
    policy = ActPolicy(HERE / "ckpt" / args.ckpt)
    gate = None if args.no_gate else WmGate(HERE / "ckpt" / args.wm)
    phys_rng = np.random.default_rng(777)     # 与 eval_act 相同序列，档间可配对
    cand_rng = np.random.default_rng(4242)
    n_seg = n_full = 0
    for ep in range(args.n):
        phys = None
        if args.phys > 0:
            from tear_refine_env import sample_phys

            phys = sample_phys(phys_rng, level=args.phys)
            phys.pop("sense", None)
        vp = None
        if args.video and ep < 1:
            vp = VIDEO_DIR / f"{args.tag}_rollout_{ep}.mp4"
        info = rollout(env, policy, gate, args.k, video_path=vp,
                       seed=10000 + ep, phys=phys, rng=cand_rng)
        n_seg += bool(info.get("seg_in_box_b"))
        n_full += bool(info.get("board_returned"))
        print(f"[ep {ep:02d}] 撕剪 {info.get('seg_in_box_b')}, "
              f"全流程 {info.get('board_returned')}, "
              f"WM 改选 {info.get('overrides')} 次"
              + (f" 视频 {vp}" if vp else ""), flush=True)
    print(f"\n成功率: 撕剪入盒 B {n_seg}/{args.n}, 全流程 {n_full}/{args.n}")
    env.close()


if __name__ == "__main__":
    main()
