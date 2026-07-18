"""录制配对对比视频：同一（场景+物理参数）下，零动作脚本 vs RL 相位修正。

布局（1920x1440）：左半 = 脚本，右半 = RL；每半竖排两路
（机器人主视角 head_cam 上、右腕手眼 wrist_cam_right 下，各 960x720）。
两侧 rollout 时长不同：短的一侧定格最后一帧；结束后各侧盖结果印章，
尾部延时 2 秒方便看清结果。

运行:
    ../../.venv/Scripts/python.exe record_compare.py --rows 1 2
"""

import argparse
import io
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from inspect_refine_case import load_conds
from tear_refine_env import TearRefineEnv

HERE = Path(__file__).resolve().parent
VIDEO_DIR = HERE.parent.parent / "docs" / "assets" / "videos"
CELL_H, CELL_W = 720, 960
FONT = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 40)
FONT_BIG = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 72)


def label(img, text, color=(255, 255, 255)):
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im, "RGBA")
    draw.rectangle([12, 12, 44 + 40 * len(text), 74], fill=(0, 0, 0, 150))
    draw.text((24, 18), text, font=FONT, fill=(*color, 240))
    return np.asarray(im)


def stamp(img, text, ok):
    """在半侧画面中央盖结果印章。"""
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im, "RGBA")
    w, h = im.size
    tw = draw.textlength(text, font=FONT_BIG)
    x, y = (w - tw) / 2, h / 2 - 60
    color = (76, 175, 80) if ok else (229, 57, 53)
    draw.rectangle([x - 30, y - 20, x + tw + 30, y + 110],
                   fill=(0, 0, 0, 150), outline=(*color, 255), width=6)
    draw.text((x, y), text, font=FONT_BIG, fill=(*color, 255))
    return np.asarray(im)


def rollout_frames(env, cond, policy):
    """跑一条 rollout，返回 (jpeg 帧列表[单侧 960x1440], success)。"""
    renderer = mujoco.Renderer(env._peek_model(cond["pool_idx"]),
                               height=CELL_H, width=CELL_W)
    frames, tick = [], [0]

    def hook(data):
        tick[0] += 1
        if tick[0] % 2 == 0:
            cells = []
            for cam in ("head_cam", "wrist_cam_right"):
                renderer.update_scene(data, camera=cam)
                cells.append(renderer.render().copy())
            side = np.vstack(cells)
            buf = io.BytesIO()
            Image.fromarray(side).save(buf, format="jpeg", quality=90)
            frames.append(buf.getvalue())

    obs, _ = env.reset(options={"pool_idx": cond["pool_idx"],
                                "phys": dict(cond["phys"])})
    env.render_hook = hook
    done = False
    while not done:
        act = np.zeros(5, dtype=np.float32) if policy is None \
            else policy.predict(obs, deterministic=True)[0]
        obs, r, term, trunc, info = env.step(act)
        done = term or trunc
    env.render_hook = None
    renderer.close()
    return frames, bool(info["success"]), bool(info["torn"])


def compose(row, cond, frames_s, ok_s, torn_s, frames_r, ok_r, out_path):
    n = max(len(frames_s), len(frames_r))
    tail = 50   # 结果印章定格 2 s
    writer = imageio.get_writer(out_path, fps=25, quality=7, macro_block_size=1)
    ph = cond["phys"]
    sub = (f"同一场景与物理参数: 感知偏移 ({ph['sense'][0]*1000:+.0f}, "
           f"{ph['sense'][1]*1000:+.0f}) mm, 摩擦 x{ph['fric']:.2f}, "
           f"质量 x{ph['mass']:.2f}, 阈值 x{ph['thresh']:.2f}")
    txt_s = "× 未撕断" if not torn_s else ("√ 入盒 B" if ok_s else "× 未入盒")
    txt_r = "√ 入盒 B" if ok_r else "× 失败"
    for k in range(n + tail):
        left = np.asarray(Image.open(io.BytesIO(
            frames_s[min(k, len(frames_s) - 1)])))
        right = np.asarray(Image.open(io.BytesIO(
            frames_r[min(k, len(frames_r) - 1)])))
        if k >= len(frames_s):
            left = stamp(left, txt_s, ok_s)
        if k >= len(frames_r):
            right = stamp(right, txt_r, ok_r)
        left = label(left, "零动作脚本（不用 RL）")
        right = label(right, "RL 相位级修正")
        frame = np.hstack([left, right])
        frame[:, CELL_W - 2:CELL_W + 2] = 24
        im = Image.fromarray(frame)
        d = ImageDraw.Draw(im, "RGBA")
        d.rectangle([0, frame.shape[0] - 56, frame.shape[1], frame.shape[0]],
                    fill=(0, 0, 0, 160))
        d.text((20, frame.shape[0] - 50), sub, font=FONT.font_variant(size=30),
               fill=(255, 255, 255, 220))
        writer.append_data(np.asarray(im))
    writer.close()
    print(f"[组 {row}] 脚本 {'成' if ok_s else '败'} / RL {'成' if ok_r else '败'}"
          f" -> {out_path.name}（{(n + tail) / 25:.0f} s）")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--model", type=str,
                        default=str(HERE / "checkpoints" / "ppo_refine_final"))
    args = parser.parse_args()

    from stable_baselines3 import PPO

    model = PPO.load(args.model, device="cpu")
    conds = load_conds(HERE / "debug" / "eval_refine_paired.csv")
    env = TearRefineEnv(seed=0)
    for row in args.rows:
        cond = conds[row]
        frames_s, ok_s, torn_s = rollout_frames(env, cond, policy=None)
        frames_r, ok_r, _ = rollout_frames(env, cond, policy=model)
        compose(row, cond, frames_s, ok_s, torn_s, frames_r, ok_r,
                VIDEO_DIR / f"rl_vs_script_{row}.mp4")


if __name__ == "__main__":
    main()
