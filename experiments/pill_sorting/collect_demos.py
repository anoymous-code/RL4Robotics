"""脚本专家自动采集演示数据（域随机化场景，ACT 风格 HDF5）。

每条 episode：随机采样场景（盒位/停车位姿/目标格）→ 脚本专家执行
"取板 → 撕目标格入盒 B → 剩板放回盒 A" → 成功则保存演示。

数据格式（demos/episode_*.hdf5，ALOHA-ACT 兼容布局）：
    /observations/qpos          (T, 14)  float32   关节角+爪开度
    /action                     (T, 14)  float32   位置伺服目标（ctrl）
    /observations/images/{cam}  (T,)     vlen u8   JPEG 字节流（head/wrist_left/wrist_right）
    attrs: cfg 各字段、success、sim_hz

运行:
    ../../.venv/Scripts/python.exe collect_demos.py --n 10 --seed 0
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
from ik_utils import ARM_JOINTS
from pill_env import CAMS, IMG_H, IMG_W, actuator_ids14, qpos_ids14
from run_full_demo import CTRL_HZ, FullDemo

HERE = Path(__file__).resolve().parent
DEMO_DIR = HERE / "demos"
DEMO_DIR.mkdir(exist_ok=True)


class DemoRecorder:
    """挂在 FullDemo.step_ctrl 上的录制器：每个控制周期存 (观测, 动作)。"""

    def __init__(self, model, jpeg_quality=60):
        self.renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)
        self.act_ids = actuator_ids14(model)
        self.qpos_ids = qpos_ids14(model)
        self.jpeg_quality = jpeg_quality
        self.qpos, self.actions = [], []
        self.jpegs = {cam: [] for cam in CAMS}

    def tick(self, model, data):
        self.qpos.append(data.qpos[self.qpos_ids].astype(np.float32))
        self.actions.append(data.ctrl[self.act_ids].astype(np.float32))
        for cam in CAMS:
            self.renderer.update_scene(data, camera=cam)
            buf = io.BytesIO()
            Image.fromarray(self.renderer.render()).save(
                buf, format="jpeg", quality=self.jpeg_quality)
            self.jpegs[cam].append(np.frombuffer(buf.getvalue(), dtype=np.uint8))

    def save(self, path, cfg, success):
        with h5py.File(path, "w") as f:
            f.attrs["success"] = success
            f.attrs["sim_hz"] = CTRL_HZ
            f.attrs["cfg"] = json.dumps({
                "box_a_xy": list(cfg.box_a_xy), "box_b_xy": list(cfg.box_b_xy),
                "base_work": list(cfg.base_work), "target_seg": list(cfg.target_seg)})
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.stack(self.qpos))
            f.create_dataset("action", data=np.stack(self.actions))
            imgs = obs.create_group("images")
            dt = h5py.vlen_dtype(np.uint8)
            for cam in CAMS:
                d = imgs.create_dataset(cam, (len(self.jpegs[cam]),), dtype=dt)
                for i, j in enumerate(self.jpegs[cam]):
                    d[i] = j

    def close(self):
        self.renderer.close()


def collect(n_episodes, seed=0, rand_level=1.0, verbose=False):
    rng = np.random.default_rng(seed)
    results = []
    for ep in range(n_episodes):
        cfg = ts.sample_cfg(rng, rand_level)
        t0 = time.time()
        demo = FullDemo(cfg=cfg, skip_drive=True, targets=[tuple(cfg.target_seg)],
                        make_video=False, verbose=verbose)
        recorder = DemoRecorder(demo.model)
        demo.recorder = recorder
        try:
            n_ok, returned = demo.run()
            success = (n_ok == 1) and returned
        except Exception as exc:
            print(f"[ep {ep:03d}] 专家执行异常: {exc}")
            success, n_ok, returned = False, 0, False
        wall = time.time() - t0
        tag = "ok" if success else "fail"
        path = DEMO_DIR / f"episode_{ep:03d}_{tag}.hdf5"
        recorder.save(path, cfg, success)
        recorder.close()
        results.append(success)
        print(f"[ep {ep:03d}] {'成功' if success else '失败'}"
              f"（撕剪 {n_ok}/1, 回槽 {returned}）"
              f" 目标格 {ts.seg_name(*cfg.target_seg)}"
              f" 盒A ({cfg.box_a_xy[0]:+.3f},{cfg.box_a_xy[1]:+.3f})"
              f" 盒B ({cfg.box_b_xy[0]:+.3f},{cfg.box_b_xy[1]:+.3f})"
              f" 停车 ({cfg.base_work[0]:+.3f},{cfg.base_work[1]:+.3f},"
              f"{np.degrees(cfg.base_work[2]):+.1f}°)"
              f" | {len(recorder.qpos)} ticks, {wall:.0f}s -> {path.name}")
    rate = float(np.mean(results)) if results else 0.0
    print(f"\n专家成功率: {sum(results)}/{len(results)} = {rate*100:.0f}%")
    return rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="脚本专家演示数据采集")
    parser.add_argument("--n", type=int, default=10, help="episode 数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rand", type=float, default=1.0, help="域随机化幅度 0~1")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    collect(args.n, seed=args.seed, rand_level=args.rand, verbose=args.verbose)
