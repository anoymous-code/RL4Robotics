"""RL 残差精修策略 vs 零残差脚本：配对对比评测。

同一批 (场景, 物理参数 θ) 上分别跑两种控制器，配对消除随机化方差：
    - zero  : 残差恒零 = 纯脚本名义编排（基线）
    - policy: PPO 残差精修

可选录制 QuadCam 四视角视频（默认录前 2 条配对中策略成功而脚本失败的场景）。

运行:
    ../../.venv/Scripts/python.exe eval_refine.py --model checkpoints/ppo_refine_final --n 40 --video
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

import tear_scene as ts
from tear_refine_env import TearRefineEnv, sample_phys

HERE = Path(__file__).resolve().parent
VIDEO_DIR = HERE.parent.parent / "docs" / "assets" / "videos"


def rollout(env, policy, pool_idx, phys, video_path=None):
    obs, info = env.reset(options={"pool_idx": pool_idx, "phys": dict(phys)})
    writer = quad = None
    if video_path:
        from run_full_demo import QuadCam

        writer = imageio.get_writer(video_path, fps=25, quality=7,
                                    macro_block_size=1)
        quad = QuadCam(env.model)
        tick = [0]

        def hook(data):
            tick[0] += 1
            if tick[0] % 2 == 0:
                writer.append_data(quad.composite(data))

        env.render_hook = hook   # env.step 内部按控制周期回调（相位级步进）
    done = False
    while not done:
        act = policy(obs)
        obs, r, term, trunc, info = env.step(act)
        done = term or trunc
    if writer:
        env.render_hook = None
        writer.close()
        quad.close()
    return bool(info["success"]), bool(info["torn"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default=str(HERE / "checkpoints" / "ppo_refine_final"))
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()

    from stable_baselines3 import PPO

    model = PPO.load(args.model, device="cpu")
    env = TearRefineEnv(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    def pi_zero(obs):
        return np.zeros(5, dtype=np.float32)

    def pi_rl(obs):
        act, _ = model.predict(obs, deterministic=True)
        return act

    conds = [(int(rng.integers(len(env.pool))), sample_phys(rng))
             for _ in range(args.n)]
    rows, n_videos = [], 0
    for i, (idx, phys) in enumerate(conds):
        z_ok, z_torn = rollout(env, pi_zero, idx, phys)
        vp = None
        if args.video and not z_ok and n_videos < 2:
            vp = VIDEO_DIR / f"rl_refine_{n_videos}.mp4"
        r_ok, r_torn = rollout(env, pi_rl, idx, phys, video_path=vp)
        if vp is not None and r_ok:
            n_videos += 1
            print(f"    -> 视频（脚本败/RL 成 场景）: {vp.name}")
        elif vp is not None:
            vp.unlink(missing_ok=True)   # RL 也失败，不留视频
        rows.append((z_ok, z_torn, r_ok, r_torn))
        print(f"[{i:02d}] pool {idx:02d} fric {phys['fric']:.2f} "
              f"mass {phys['mass']:.2f} thr {phys['thresh']:.2f} "
              f"sense ({phys['sense'][0]*1000:+.0f},{phys['sense'][1]*1000:+.0f})mm"
              f" | 脚本 {'√' if z_ok else '×'}  RL {'√' if r_ok else '×'}")

    arr = np.array(rows)
    n = len(arr)
    print(f"\n===== 配对对比（{n} 组相同场景+物理参数） =====")
    print(f"零残差脚本 : 撕断 {arr[:,1].sum()}/{n}, 入盒 B {arr[:,0].sum()}/{n}"
          f" = {arr[:,0].mean()*100:.0f}%")
    print(f"RL 残差精修: 撕断 {arr[:,3].sum()}/{n}, 入盒 B {arr[:,2].sum()}/{n}"
          f" = {arr[:,2].mean()*100:.0f}%")
    both = ((arr[:, 0] == 0) & (arr[:, 2] == 1)).sum()
    lost = ((arr[:, 0] == 1) & (arr[:, 2] == 0)).sum()
    print(f"脚本败而 RL 成: {both} | 脚本成而 RL 败: {lost}")


if __name__ == "__main__":
    main()
