"""PPO 训练撕剪-投放残差精修策略（特权信息教师）。

- 环境：TearRefineEnv（残差动作叠加脚本名义编排 + 物理随机化）
- 并行：SubprocVecEnv（环境不加载 torch/CUDA，规避 Windows 虚存问题）
- 策略：MLP 256x256（状态观测，CPU 训练足够快）

运行:
    ../../.venv/Scripts/python.exe train_ppo_refine.py --steps 1500000
"""

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "checkpoints"


def make_env(rank, n_envs, phys_level):
    def _init():
        import pickle

        from tear_refine_env import POOL_PATH, TearRefineEnv

        with open(POOL_PATH, "rb") as f:
            n_pool = len(pickle.load(f))
        # 重置池按 worker 分片：每个子进程只编译/缓存自己那份模型，
        # 否则 8 worker × 64 模型副本会耗尽内存（MuJoCo engine error）
        return TearRefineEnv(seed=1000 + rank, phys_level=phys_level,
                             pool_indices=range(rank, n_pool, n_envs))
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=60_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--phys", type=float, default=1.0)
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    env = VecMonitor(SubprocVecEnv(
        [make_env(i, args.envs, args.phys) for i in range(args.envs)]),
        filename=str(CKPT_DIR / "ppo_refine_monitor.csv"))
    CKPT_DIR.mkdir(exist_ok=True)

    if args.resume:
        model = PPO.load(args.resume, env=env, device="cpu")
        print(f"续训自 {args.resume}")
    else:
        # semi-MDP：一步 = 一个编排相位，episode 只有 ~13 步
        model = PPO(
            "MlpPolicy", env, device="cpu",
            n_steps=64, batch_size=256, n_epochs=10,
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=3e-3,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1, seed=0,
            tensorboard_log=None)

    ckpt_cb = CheckpointCallback(save_freq=max(10_000 // args.envs, 1),
                                 save_path=str(CKPT_DIR),
                                 name_prefix="ppo_refine")
    model.learn(total_timesteps=args.steps, callback=ckpt_cb,
                reset_num_timesteps=not args.resume)
    model.save(CKPT_DIR / "ppo_refine_final")
    print(f"训练完成: {CKPT_DIR / 'ppo_refine_final.zip'}")

    # 快速成功率评估（训练分布内）
    from tear_refine_env import TearRefineEnv

    eval_env = TearRefineEnv(seed=9000, phys_level=args.phys)
    wins = 0
    n = 20
    for _ in range(n):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = eval_env.step(act)
            done = term or trunc
        wins += info["success"]
    print(f"训练后快评: {wins}/{n}")


if __name__ == "__main__":
    main()
