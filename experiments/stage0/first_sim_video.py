"""阶段 0 烟雾测试：跑通 MuJoCo 仿真并录制第一段视频。

运行方式（在项目根目录）:
    .venv\\Scripts\\python.exe experiments\\stage0\\first_sim_video.py

产出:
    docs/assets/videos/stage0_random_walker.mp4  (随机动作)
    并在控制台打印环境的观测/动作空间信息，帮助建立直觉。
"""

from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEO_DIR = PROJECT_ROOT / "docs" / "assets" / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

ENV_ID = "Walker2d-v5"
EPISODES = 3
FPS = 50


def main() -> None:
    env = gym.make(ENV_ID, render_mode="rgb_array")

    print(f"环境: {ENV_ID}")
    print(f"观测空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")

    frames = []
    for ep in range(EPISODES):
        obs, info = env.reset(seed=ep)
        total_reward, steps = 0.0, 0
        while True:
            action = env.action_space.sample()  # 随机动作：这就是"学习前"的机器人
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            frames.append(env.render())
            if terminated or truncated:
                break
        print(f"第 {ep + 1} 回合: {steps} 步, 累积奖励 {total_reward:.1f}")

    env.close()

    out = VIDEO_DIR / "stage0_random_walker.mp4"
    imageio.mimsave(out, frames, fps=FPS, macro_block_size=1)
    print(f"视频已保存: {out} （共 {len(frames)} 帧）")


if __name__ == "__main__":
    main()
