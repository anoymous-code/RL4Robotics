"""把一条 HDF5 演示数据合成三机位视频（数据样例可视化）。

布局与主演示一致：head_cam 放大在上，两路腕相机并排在下。

运行:
    ../../.venv/Scripts/python.exe demo_to_video.py demos/episode_000_ok.hdf5 \
        --out ../../docs/assets/videos/demo_sample.mp4
"""

import argparse
import io
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CAMS = ("head_cam", "wrist_cam_left", "wrist_cam_right")
LABELS = {"head_cam": "头部相机（策略观测）", "wrist_cam_left": "左腕相机",
          "wrist_cam_right": "右腕相机"}


def label(img, text, font):
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im, "RGBA")
    draw.rectangle([4, 4, 12 + 13 * len(text), 26], fill=(0, 0, 0, 130))
    draw.text((8, 6), text, font=font, fill=(255, 255, 255, 230))
    return np.asarray(im)


def main(h5path, out, fps=25, stride=2):
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 14)
    except Exception:
        font = None
    with h5py.File(h5path, "r") as f:
        n = f["observations/qpos"].shape[0]
        print(f"{h5path}: {n} ticks, cfg={json.loads(f.attrs['cfg'])}")
        writer = imageio.get_writer(out, fps=fps, macro_block_size=1)
        for t in range(0, n, stride):   # 50Hz 数据隔帧取 → 25fps 实时速度
            frames = {}
            for cam in CAMS:
                img = np.asarray(Image.open(io.BytesIO(
                    f[f"observations/images/{cam}"][t].tobytes())))
                frames[cam] = label(img, LABELS[cam], font) if font else img
            top = np.asarray(Image.fromarray(frames["head_cam"]).resize((640, 480)))
            bottom = np.hstack([frames["wrist_cam_left"], frames["wrist_cam_right"]])
            frame = np.vstack([top, bottom])
            frame[478:482, :] = 30
            frame[480:, 318:322] = 30
            writer.append_data(frame)
        writer.close()
    print(f"视频: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("h5", type=str)
    parser.add_argument("--out", type=str, default="debug/demo_sample.mp4")
    args = parser.parse_args()
    main(args.h5, args.out)
