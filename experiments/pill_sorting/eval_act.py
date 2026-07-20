"""ACT-lite 策略评测：随机场景 rollout + 成功率统计 + 三机位视频。

推理方式：每次前向输出 chunk=50 步动作块，开环执行前 K 步后重推理
（K=25，即 0.5 s 重规划一次）。

运行:
    ../../.venv/Scripts/python.exe eval_act.py --n 20            # 评成功率
    ../../.venv/Scripts/python.exe eval_act.py --n 3 --video     # 附 rollout 视频
"""

import argparse
import io
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image

import tear_scene as ts
from pill_env import CAMS, PillTearEnv
from run_full_demo import QuadCam
from train_act import CHUNK, IMG_MEAN, IMG_STD, ACTLite

HERE = Path(__file__).resolve().parent
VIDEO_DIR = HERE.parents[1] / "docs" / "assets" / "videos"
EXEC_STEPS = 50          # 整块开环执行：演示首段有静止 dwell，短执行段会让
                         # "图像推断相位"死锁在静止区（详见 rollout 文档字符串）
EPISODE_SECS = 55
ENSEMBLE_M = 0.1         # 时间集成指数权重系数（ACT 论文）


def jpeg_roundtrip(img, quality=60):
    """训练数据是 JPEG(60) 压缩帧，推理观测也过一遍编解码对齐分布。"""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="jpeg", quality=quality)
    return np.asarray(Image.open(buf))


class ActPolicy:
    def __init__(self, ckpt_path, device="cuda", jpeg_q=60):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model = ACTLite().to(device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.stats = {k: np.array(v, dtype=np.float32) for k, v in ckpt["stats"].items()}
        self.device = device
        self.jpeg_q = jpeg_q     # 与训练数据的 JPEG 质量对齐（hires 数据为 50）
        print(f"载入 {ckpt_path}（训练步数 {ckpt.get('step')}）")

    @torch.no_grad()
    def predict_chunk(self, obs, target_row=0):
        imgs = np.stack([
            ((jpeg_roundtrip(obs[cam], self.jpeg_q).astype(np.float32) / 255.0)
             - IMG_MEAN) / IMG_STD
            for cam in CAMS]).transpose(0, 3, 1, 2)
        qpos = (obs["qpos"].astype(np.float32) - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        imgs_t = torch.from_numpy(imgs).unsqueeze(0).to(self.device)
        qpos_t = torch.from_numpy(qpos).unsqueeze(0).to(self.device)
        tgt_t = torch.tensor([target_row], device=self.device)
        with torch.autocast(self.device, dtype=torch.bfloat16):
            chunk = self.model(imgs_t, qpos_t, tgt_t)[0].float().cpu().numpy()
        return chunk * self.stats["act_std"] + self.stats["act_mean"]


def rollout(env, policy, video_path=None, seed=None, phys=None,
            exec_steps=EXEC_STEPS):
    """动作块开环执行 rollout：一次推理执行前 exec_steps 步再重推理。

    注意不用"每步重推理 + 时间集成"：本策略实测忽略 qpos（图像与 qpos 在
    演示中完全冗余，模型走了图像捷径），每步重推理时图像几乎不变 →
    预测停在轨迹同一相位 → 机器人原地冻结。开环执行块内自洽的轨迹段
    可以实质推进，新图像随之明显变化，相位得以校准。"""
    obs, info = env.reset(seed=seed, options={"phys": phys} if phys else None)
    target_row = int(info["cfg"].target_seg[1])
    writer, quad = None, None
    if video_path:
        writer = imageio.get_writer(video_path, fps=25, quality=7, macro_block_size=1)
        quad = QuadCam(env.model)   # 四视角高清合成（全景/主视角/双手眼）
    max_steps = int(EPISODE_SECS * 50)
    last_info, steps, done = {}, 0, False
    while steps < max_steps and not done:
        chunk = policy.predict_chunk(obs, target_row)
        for k in range(min(exec_steps, CHUNK)):
            obs, reward, terminated, truncated, last_info = env.step(chunk[k])
            steps += 1
            if writer and steps % 2 == 0:
                writer.append_data(quad.composite(env.data))
            if terminated or steps >= max_steps:
                done = terminated
                break
    if writer:
        writer.close()
        quad.close()
    return last_info


def main(n, ckpt, video, phys_level=0.0, tag="act", strict=False,
         exec_steps=EXEC_STEPS, hires=False):
    env = PillTearEnv(seed=1234, strict_grip=strict,
                      img_hw=(480, 640) if hires else (240, 320))
    policy = ActPolicy(HERE / "ckpt" / ckpt, jpeg_q=50 if hires else 60)
    phys_rng = np.random.default_rng(777)   # 物理参数序列与 seed 解耦，档间可配对
    n_seg = n_full = n_clean = 0
    for ep in range(n):
        phys = None
        if phys_level > 0:
            from tear_refine_env import sample_phys

            phys = sample_phys(phys_rng, level=phys_level)
            phys.pop("sense", None)   # 视觉策略无标定概念，感知偏移不适用
        vp = None
        if video and ep < 1:   # 四视角高清视频体积大，只录代表性一条
            vp = VIDEO_DIR / f"{tag}_rollout_{ep}.mp4"
        info = rollout(env, policy, video_path=vp, seed=10000 + ep, phys=phys,
                       exec_steps=exec_steps)
        seg_ok = bool(info.get("seg_in_box_b"))
        clean = seg_ok and bool(info.get("clean_tear"))
        n_seg += seg_ok
        n_full += bool(info.get("board_returned"))
        n_clean += clean
        extra = (f" fric {phys['fric']:.2f} mass {phys['mass']:.2f} "
                 f"thr {phys['thresh']:.2f}") if phys else ""
        quality = ("规范撕剪" if clean else "敲断" if seg_ok else "-")
        print(f"[ep {ep:02d}] 撕剪入盒 B: {seg_ok} ({quality}), "
              f"全流程: {info.get('board_returned')}{extra}"
              + (f" 视频 {vp}" if vp else ""), flush=True)
    print(f"\n成功率: 撕剪入盒 B {n_seg}/{n}（其中规范撕剪 {n_clean}, "
          f"敲断 {n_seg - n_clean}）, 全流程 {n_full}/{n}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--ckpt", type=str, default="act_latest.pt")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--phys", type=float, default=0.0,
                        help="物理随机化档位（摩擦/质量/断裂阈值，0=标称）")
    parser.add_argument("--tag", type=str, default="act", help="视频文件名前缀")
    parser.add_argument("--strict", action="store_true",
                        help="严格物理评测：断裂需双指夹持（配严格数据训练的策略）")
    parser.add_argument("--exec-steps", type=int, default=EXEC_STEPS,
                        help="每次推理开环执行步数（小=更频繁重规划）")
    parser.add_argument("--hires", action="store_true",
                        help="高分辨率观测 480x640（配高分辨率数据训练的模型）")
    args = parser.parse_args()
    main(args.n, args.ckpt, args.video, phys_level=args.phys, tag=args.tag,
         strict=args.strict, exec_steps=args.exec_steps, hires=args.hires)
