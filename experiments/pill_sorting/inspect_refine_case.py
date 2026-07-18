"""RL 精修成功案例解剖：重放指定配对组，打印每个相位的修正量。

对照运行零动作脚本（记录失败方式）与 RL 策略（记录相位级动作解读），
输出可直接贴进日志的案例记录。

运行:
    ../../.venv/Scripts/python.exe inspect_refine_case.py --rows 1 2
"""

import argparse
from pathlib import Path

import numpy as np

from tear_refine_env import (ACT_GRIP, ACT_POS, ACT_TWIST, IK_PHASES,
                             TearRefineEnv)

HERE = Path(__file__).resolve().parent


def load_conds(csv_path):
    conds = []
    for line in Path(csv_path).read_text(encoding="utf-8").splitlines()[1:]:
        f = line.split(",")
        conds.append({
            "pool_idx": int(f[1]),
            "phys": {"fric": float(f[2]), "mass": float(f[3]),
                     "thresh": float(f[4]),
                     "sense": np.array([float(f[5]), float(f[6]), 0.0]) / 1000},
        })
    return conds


def describe(env, action):
    p = env.phase
    a = np.clip(action, -1, 1)
    parts = []
    if p in IK_PHASES:
        c = a[:3] * ACT_POS * 1000
        parts.append(f"目标修正 ({c[0]:+.0f}, {c[1]:+.0f}, {c[2]:+.0f}) mm")
    if p == "close":
        parts.append(f"过盈修正 {a[3] * ACT_GRIP * 1000:+.2f} mm")
    if p == "twist":
        parts.append(f"扭幅 ×{1.0 + a[4] * ACT_TWIST:.2f}")
    return "; ".join(parts) if parts else "-"


def run_case(env, cond, policy=None, verbose_phases=True):
    obs, info = env.reset(options={"pool_idx": cond["pool_idx"],
                                   "phys": dict(cond["phys"])})
    log, done = [], False
    while not done:
        act = np.zeros(5, dtype=np.float32) if policy is None \
            else policy.predict(obs, deterministic=True)[0]
        phase = env.phase
        desc = describe(env, act)
        obs, r, term, trunc, info = env.step(act)
        if verbose_phases and desc != "-":
            log.append(f"    {phase:6s}: {desc}")
        done = term or trunc
    seg_p = env.data.body(env.seg_body).xpos.copy()
    return info, seg_p, log


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
        ph = cond["phys"]
        print(f"\n===== 案例（配对组 {row}, pool {cond['pool_idx']}） =====")
        print(f"条件: 摩擦 x{ph['fric']:.2f}, 质量 x{ph['mass']:.2f}, "
              f"阈值 x{ph['thresh']:.2f}, "
              f"感知偏移 ({ph['sense'][0]*1000:+.1f}, {ph['sense'][1]*1000:+.1f}) mm")
        info, seg_p, _ = run_case(env, cond, policy=None, verbose_phases=False)
        print(f"[脚本] 撕断 {info['torn']}, 入盒 {info['success']}, "
              f"格终位 {np.round(seg_p, 3)}")
        info, seg_p, log = run_case(env, cond, policy=model)
        print(f"[RL]   撕断 {info['torn']}, 入盒 {info['success']}, "
              f"格终位 {np.round(seg_p, 3)}")
        print("  RL 相位修正:")
        print("\n".join(log))


if __name__ == "__main__":
    main()
