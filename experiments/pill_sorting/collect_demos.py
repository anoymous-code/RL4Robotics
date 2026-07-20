"""脚本专家自动采集演示数据（域随机化场景，ACT 风格 HDF5）。

每条 episode：随机采样场景（盒位/停车位姿/目标格）→ 脚本专家执行
"取板 → 撕目标格入盒 B → 剩板放回盒 A" → 成功则保存演示。

数据格式（demos/episode_*.hdf5，ALOHA-ACT 兼容布局）：
    /observations/qpos          (T, 14)  float32   关节角+爪开度
    /action                     (T, 14)  float32   位置伺服目标（ctrl）
    /observations/images/{cam}  (T,)     vlen u8   JPEG 字节流（head/wrist_left/wrist_right）
    /phase                      (T,)     uint8     相位标签（PHASE_NAMES 索引，
                                                   0=transit 衔接段，1..4=操作原语）
    attrs: cfg 各字段、success、sim_hz、phase_names、
           prim_ok（各原语出口谓词判定，原语级训练筛选用）

运行:
    ../../.venv/Scripts/python.exe collect_demos.py --n 10 --seed 0
    原语数据集（高清双腕相机 + 入口抖动）:
    ../../.venv/Scripts/python.exe collect_demos.py --n 150 --strict --hires ^
        --prefix prim --wrist-only --entry-jitter 0.004
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
from run_full_demo import CTRL_HZ, PHASE_IDS, PHASE_NAMES, FullDemo

HERE = Path(__file__).resolve().parent
DEMO_DIR = HERE / "demos"
DEMO_DIR.mkdir(exist_ok=True)


class DemoRecorder:
    """挂在 FullDemo.step_ctrl 上的录制器：每个控制周期存 (观测, 动作, 相位)。

    phase_fn: 无参回调，返回当前相位名（FullDemo.phase_name）；
              None 时全程记 transit（教师子任务演示等无相位概念的场景）。
    cams:     录制的相机子集（原语数据集只需双腕相机，省渲染与磁盘）。
    """

    def __init__(self, model, jpeg_quality=60, img_hw=(IMG_H, IMG_W), phase_fn=None,
                 cams=CAMS):
        self.renderer = mujoco.Renderer(model, height=img_hw[0], width=img_hw[1])
        self.act_ids = actuator_ids14(model)
        self.qpos_ids = qpos_ids14(model)
        self.jpeg_quality = jpeg_quality
        self.phase_fn = phase_fn
        self.cams = tuple(cams)
        self.qpos, self.actions, self.phases = [], [], []
        self.jpegs = {cam: [] for cam in self.cams}

    def tick(self, model, data):
        self.qpos.append(data.qpos[self.qpos_ids].astype(np.float32))
        self.actions.append(data.ctrl[self.act_ids].astype(np.float32))
        self.phases.append(PHASE_IDS[self.phase_fn()] if self.phase_fn else 0)
        for cam in self.cams:
            self.renderer.update_scene(data, camera=cam)
            buf = io.BytesIO()
            Image.fromarray(self.renderer.render()).save(
                buf, format="jpeg", quality=self.jpeg_quality)
            self.jpegs[cam].append(np.frombuffer(buf.getvalue(), dtype=np.uint8))

    def save(self, path, cfg, success, prim_ok=None):
        with h5py.File(path, "w") as f:
            f.attrs["success"] = success
            f.attrs["sim_hz"] = CTRL_HZ
            f.attrs["phase_names"] = json.dumps(list(PHASE_NAMES))
            if prim_ok is not None:
                f.attrs["prim_ok"] = json.dumps(prim_ok)
            f.attrs["cfg"] = json.dumps({
                "box_a_xy": list(cfg.box_a_xy), "box_b_xy": list(cfg.box_b_xy),
                "base_work": list(cfg.base_work), "target_seg": list(cfg.target_seg)})
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.stack(self.qpos))
            f.create_dataset("action", data=np.stack(self.actions))
            f.create_dataset("phase", data=np.asarray(self.phases, dtype=np.uint8))
            imgs = obs.create_group("images")
            dt = h5py.vlen_dtype(np.uint8)
            for cam in self.cams:
                d = imgs.create_dataset(cam, (len(self.jpegs[cam]),), dtype=dt)
                for i, j in enumerate(self.jpegs[cam]):
                    d[i] = j

    def close(self):
        self.renderer.close()


def collect(n_episodes, seed=0, rand_level=1.0, verbose=False, start=0,
            action_noise=0.004, prefix="episode", strict_tear=False,
            img_hw=(IMG_H, IMG_W), jpeg_quality=60, entry_jitter=0.0, cams=CAMS):
    from primitives import PRIM_NAMES, prim_success

    rng = np.random.default_rng(seed)
    results = []
    for k in range(n_episodes):
        ep = start + k
        cfg = ts.sample_cfg(rng, rand_level)
        t0 = time.time()
        demo = FullDemo(cfg=cfg, skip_drive=True, targets=[tuple(cfg.target_seg)],
                        make_video=False, verbose=verbose, action_noise=action_noise,
                        strict_tear=strict_tear, entry_jitter=entry_jitter)
        recorder = DemoRecorder(demo.model, jpeg_quality=jpeg_quality, img_hw=img_hw,
                                phase_fn=lambda d=demo: d.phase_name, cams=cams)
        demo.recorder = recorder
        # 原语出口谓词打标（原语级训练筛选：全流程失败的 episode 里
        # 成功的原语片段仍是有效演示）
        prim_ok = {}

        def on_exit(d, name, ctx):
            prim_ok[name] = bool(prim_success(
                d, name, seg=ts.seg_name(*d.cfg.target_seg)))

        for p in PRIM_NAMES:
            demo.primitive_hooks[f"{p}/exit"] = on_exit
        try:
            n_ok, returned = demo.run()
            success = (n_ok == 1) and returned
        except Exception as exc:
            print(f"[ep {ep:03d}] 专家执行异常: {exc}")
            success, n_ok, returned = False, 0, False
        wall = time.time() - t0
        tag = "ok" if success else "fail"
        path = DEMO_DIR / f"{prefix}_{ep:03d}_{tag}.hdf5"
        recorder.save(path, cfg, success, prim_ok=prim_ok)
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
    parser.add_argument("--start", type=int, default=0, help="起始编号（续采时避免覆盖）")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--prefix", type=str, default="episode", help="输出文件名前缀")
    parser.add_argument("--strict", action="store_true",
                        help="规范撕剪物理：断裂需双指夹持 + 连续超阈值")
    parser.add_argument("--hires", action="store_true",
                        help="高分辨率观测 480x640（默认 240x320），JPEG 质量降为 50 控制体积")
    parser.add_argument("--entry-jitter", type=float, default=0.0,
                        help="衔接段到位目标随机偏差半径 (m)，模拟经典控制交接原语的定位误差")
    parser.add_argument("--wrist-only", action="store_true",
                        help="只录双腕相机（原语数据集用不到 head_cam，省渲染与磁盘）")
    args = parser.parse_args()
    collect(args.n, seed=args.seed, rand_level=args.rand, verbose=args.verbose,
            start=args.start, prefix=args.prefix, strict_tear=args.strict,
            img_hw=(480, 640) if args.hires else (IMG_H, IMG_W),
            jpeg_quality=50 if args.hires else 60, entry_jitter=args.entry_jitter,
            cams=("wrist_cam_left", "wrist_cam_right") if args.wrist_only else CAMS)
