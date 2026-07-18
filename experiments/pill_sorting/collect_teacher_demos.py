"""RL 特权教师生成蒸馏演示（DAgger 式：教师示范，学生格式记录）。

教师 = 脚本名义编排 + PPO 相位级修正（tear_refine_env）。采集时：
    - 物理随机化用教师的完整训练分布（摩擦/质量/断裂阈值 + 感知偏移）。
      感知偏移只存在于教师内部的名义编排里；学生记录到的是"编排 + 修正"
      合成后的最终控制流——偏移被修正抵消，从学生（像素观测）视角就是
      一条正常的好演示。实测教师在无偏移档反而只有 60%（修正过度，
      配对评测已发现该弱点），在完整分布上是 92%，因此按完整分布采；
    - 记录与 collect_demos.py 完全相同的 HDF5 布局（qpos14 + ctrl14 +
      三路 JPEG 图像 @50Hz），学生训练管线无缝混用；失败条直接丢弃。

Episode 为撕剪-投放子任务（从"板已在工作位"的重置池快照开始）——
对动作分块的 ACT 来说只是"短一些的演示"。

运行:
    ../../.venv/Scripts/python.exe collect_teacher_demos.py --n 150
    ../../.venv/Scripts/python.exe collect_teacher_demos.py --dry 20   # 教师成功率烟测
"""

import argparse
import time
from pathlib import Path

import numpy as np

import tear_scene as ts
from collect_demos import DEMO_DIR, DemoRecorder
from tear_refine_env import TearRefineEnv, sample_phys

HERE = Path(__file__).resolve().parent


def sample_phys_capped(rng, sense_cap=0.005):
    """完整物理分布 + 感知偏移截幅。

    第一轮蒸馏用完整 ±25mm 偏移采集，结果负迁移：教师的接近轨迹先奔向
    带偏移的假目标、抓取相位才修正回来，这种"绕路"对学生（像素观测，
    看不见偏移）是无法解释的噪声。截到 ±5mm 后绕路可忽略，教师又不至于
    掉进它校准不良的纯零偏移区。"""
    phys = sample_phys(rng)
    phys["sense"] = np.clip(phys["sense"], -sense_cap, sense_cap)
    return phys


def run_episode(env, model, pool_idx, phys, recorder=None):
    obs, info = env.reset(options={"pool_idx": pool_idx, "phys": phys})
    if recorder is not None:
        env.tick_hook = recorder.tick
    done = False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(act)
        done = term or trunc
    env.tick_hook = None
    return bool(info["success"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=0, help="采集成功演示条数")
    parser.add_argument("--dry", type=int, default=0, help="只测教师成功率不存盘")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str,
                        default=str(HERE / "checkpoints" / "ppo_refine_final"))
    args = parser.parse_args()

    from stable_baselines3 import PPO

    model = PPO.load(args.model, device="cpu")
    env = TearRefineEnv(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    if args.dry:
        wins = 0
        for ep in range(args.dry):
            idx = int(rng.integers(len(env.pool)))
            ok = run_episode(env, model, idx, sample_phys_capped(rng))
            wins += ok
            print(f"[dry {ep:02d}] pool {idx:02d} -> {'成' if ok else '败'}")
        print(f"\n教师成功率（物理随机化 + 偏移截幅）: {wins}/{args.dry}")
        return

    saved, tries = 0, 0
    t0 = time.time()
    while saved < args.n:
        tries += 1
        idx = int(rng.integers(len(env.pool)))
        phys = sample_phys_capped(rng)
        recorder = DemoRecorder(env._peek_model(idx))
        ok = run_episode(env, model, idx, phys, recorder=recorder)
        if ok:
            path = DEMO_DIR / f"teacher_{saved:03d}_ok.hdf5"
            recorder.save(path, env.cfg, True)
            saved += 1
            print(f"[{saved:03d}/{args.n}] pool {idx:02d} "
                  f"fric {phys['fric']:.2f} mass {phys['mass']:.2f} "
                  f"thr {phys['thresh']:.2f} | {len(recorder.qpos)} ticks "
                  f"-> {path.name}", flush=True)
        else:
            print(f"[skip] pool {idx:02d} 教师失败，丢弃", flush=True)
        recorder.close()
    print(f"\n完成: {saved} 条 / {tries} 次尝试，"
          f"{(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
