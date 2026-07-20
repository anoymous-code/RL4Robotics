"""操作原语策略评测：入口状态池起步 + 出口谓词计分。

评测口径（原语分解架构）：衔接控制段已把臂送到原语入口（池条目
含 4mm 到位抖动），原语策略从入口接管单臂，出口谓词（真值）判成败。
对照组：
    --scripted     脚本原语基线（特权感知，池可行性口径下应 ≈100%）
    端到端 ACT     全流程 15% 规范撕剪（docs/journal/2026-07-20-precision.md）

运行:
    ../../.venv/Scripts/python.exe eval_prim.py --prim p2_tear_seg --n 40
    ../../.venv/Scripts/python.exe eval_prim.py --prim p2_tear_seg --n 40 --scripted
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from eval_act import jpeg_roundtrip
from primitives import PRIM_SPECS, load_pool, restore_session
from train_act import IMG_MEAN, IMG_STD, ACTLite

HERE = Path(__file__).resolve().parent
VIDEO_DIR = HERE.parents[1] / "docs" / "assets" / "videos"
EXEC_STEPS = 50          # 整块开环执行。实测 P2 上 50/25/12 步 = 62%/40%/12%：
                         # 短执行段的"相位死锁"（图像变化微小→预测停在原地）
                         # 在原语尺度下依然存在，整块开环仍是最优


class PrimPolicy:
    """单臂原语 ACT 策略（train_prim.py 检查点）。"""

    def __init__(self, ckpt_path, device="cuda", jpeg_q=50):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.prim = ckpt["prim"]
        self.spec = PRIM_SPECS[self.prim]
        self.chunk = ckpt["chunk"]
        self.model = ACTLite(n_cams=1, chunk=self.chunk, act_dim=7,
                             feat_hw=tuple(ckpt["feat_hw"])).to(device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.stats = {k: np.array(v, dtype=np.float32) for k, v in ckpt["stats"].items()}
        self.device = device
        self.jpeg_q = jpeg_q
        print(f"载入 {ckpt_path}（{self.prim}, 训练步数 {ckpt.get('step')}）")

    @torch.no_grad()
    def predict_chunk(self, obs, target_row=0):
        img = ((jpeg_roundtrip(obs[self.spec.cam], self.jpeg_q).astype(np.float32)
                / 255.0) - IMG_MEAN) / IMG_STD
        imgs_t = torch.from_numpy(img.transpose(2, 0, 1)[None, None]).to(self.device)
        qpos = (obs["qpos"].astype(np.float32)
                - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        qpos_t = torch.from_numpy(qpos[None]).to(self.device)
        tgt_t = torch.tensor([target_row], device=self.device)
        with torch.autocast(self.device, dtype=torch.bfloat16):
            chunk = self.model(imgs_t, qpos_t, tgt_t)[0].float().cpu().numpy()
        return chunk * self.stats["act_std"] + self.stats["act_mean"]


class PrimVideo:
    """双视角（场景全景 + 原语腕相机）录像，每 2 tick 一帧。

    tick 签名兼容 FullDemo.recorder 钩子（脚本基线录像直接挂 recorder）。
    """

    def __init__(self, sess, path):
        self.sess = sess
        self.r = mujoco.Renderer(sess.demo.model, height=480, width=640)
        self.writer = imageio.get_writer(path, fps=25, quality=7, macro_block_size=1)
        self.k = 0

    def tick(self, *_):
        self.k += 1
        if self.k % 2:
            return
        cells = []
        for cam in ("room", self.sess.spec.cam):
            self.r.update_scene(self.sess.demo.data, camera=cam)
            cells.append(self.r.render().copy())
        self.writer.append_data(np.hstack(cells))

    def close(self):
        self.writer.close()
        self.r.close()


def rollout_policy(sess, policy, exec_steps=EXEC_STEPS, video=None):
    """原语策略 rollout：动作块开环执行，出口谓词即停 / 超时截断。"""
    target_row = int(sess.demo.cfg.target_seg[1])
    while not sess.timeout():
        chunk = policy.predict_chunk(sess.obs(), target_row)
        for k in range(min(exec_steps, len(chunk))):
            sess.step(chunk[k])
            if video:
                video.tick()
            if sess.success():
                return True
            if sess.timeout():
                break
    return sess.success()


def main(prim, n, ckpt=None, tag="eval", scripted=False, exec_steps=EXEC_STEPS,
         video=0):
    pool = load_pool(prim, tag=tag, only_ok=True)
    n = min(n, len(pool))
    print(f"入口池 {tag}: {len(pool)} 条可行条目，评测 {n} 条")
    policy = None
    if not scripted:
        policy = PrimPolicy(HERE / "ckpt" / (ckpt or f"{prim}_latest.pt"))
    wins = 0
    for ep in range(n):
        sess = restore_session(pool[ep])
        vid = None
        if ep < video:
            mode = "scripted" if scripted else "policy"
            vp = VIDEO_DIR / f"prim_{prim}_{mode}_{ep}.mp4"
            vid = PrimVideo(sess, vp)
        if scripted:
            if vid:
                sess.demo.recorder = vid
            ok = sess.run_scripted()
        else:
            ok = rollout_policy(sess, policy, exec_steps=exec_steps, video=vid)
        if vid:
            vid.close()
        wins += ok
        print(f"[ep {ep:02d}] {'成功' if ok else '失败'}  "
              f"(elapsed {sess.elapsed():.1f}s)", flush=True)
        sess.close()
    mode = "脚本基线" if scripted else "学习策略"
    print(f"\n{prim} {mode}: {wins}/{n} = {wins/n*100:.0f}%")
    return wins / n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prim", type=str, required=True, choices=list(PRIM_SPECS))
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--tag", type=str, default="eval", help="入口池后缀")
    parser.add_argument("--scripted", action="store_true", help="脚本原语基线")
    parser.add_argument("--exec-steps", type=int, default=EXEC_STEPS)
    parser.add_argument("--video", type=int, default=0, help="录前 N 条 rollout 视频")
    args = parser.parse_args()
    main(args.prim, args.n, ckpt=args.ckpt, tag=args.tag, scripted=args.scripted,
         exec_steps=args.exec_steps, video=args.video)
