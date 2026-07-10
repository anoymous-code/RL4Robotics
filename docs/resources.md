# 资源库

精选的学习资源，按主题分类。**原则：宁缺毋滥**，每个条目都注明"为什么值得看"。

## 课程与书籍

| 资源 | 类型 | 为什么值得看 |
|---|---|---|
| Sutton & Barto《Reinforcement Learning: An Introduction》 | 书 | RL 圣经，前 6 章打牢理论地基 |
| Hugging Face Deep RL Course | 免费课程 | 边学边跑代码，有排行榜，适合入门 |
| OpenAI Spinning Up | 教程+代码 | 策略梯度算法讲解最清晰的资料 |
| 《动手学深度学习》(d2l.ai) | 书+代码 | PyTorch 基础，中文友好 |
| Modern Robotics (Lynch & Park) | 书+视频 | 机器人运动学/动力学，配 Coursera 课程 |

## 关键论文（按学习顺序）

### 模仿学习
| 论文 | 一句话 |
|---|---|
| ALOHA / ACT (2023) | 低成本双臂遥操作 + Action Chunking，我们阶段 2 的主线 |
| Mobile ALOHA (2024) | ALOHA 装上轮子，与我们的目标形态最接近 |
| Diffusion Policy (2023) | 扩散模型做动作生成，处理多模态演示 |

### 世界模型
| 论文 | 一句话 |
|---|---|
| DreamerV3 (2023) | 在想象中学习，一套超参通吃百余任务 |
| TD-MPC2 (2024) | 模型预测控制路线的代表 |

### VLA
| 论文 | 一句话 |
|---|---|
| RT-2 (2023) | VLA 概念的开山之作 |
| OpenVLA (2024) | 7B 开源 VLA，可自己微调 |
| π0 / π0.5 (2024/2025) | flow matching 动作生成，业界公认最强之一，已开源 |
| RDT-1B (2024) | 清华出品，专为双臂设计的扩散 VLA |
| GR00T N1 (2025) | NVIDIA 人形基座模型，配套 Isaac 生态 |

### 灵巧操作
| 论文 | 一句话 |
|---|---|
| OpenAI Rubik's Cube (2019) | 灵巧手 + 域随机化的里程碑 |
| DexCap (2024) | 便携动捕采集灵巧手数据 |
| 各类 blister pack 相关文献 | 待调研：制药自动化领域的既有方案 |

## 开源框架与工具

| 工具 | 用途 | 备注 |
|---|---|---|
| [LeRobot](https://github.com/huggingface/lerobot) | 机器人学习全流程（数据/训练/评估） | 我们的主力框架，社区活跃 |
| [MuJoCo](https://mujoco.org) | 物理仿真 | 轻量、准确，阶段 0-1 主力 |
| [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) | GPU 大规模并行仿真 | 需要 RTX 显卡，阶段 3+ 用 |
| [ManiSkill 3](https://maniskill.ai) | 操作任务基准 + GPU 仿真 | 自带大量操作任务，对比基准好用 |
| Gymnasium | RL 环境标准接口 | 事实标准 |
| Stable-Baselines3 / CleanRL | RL 算法库 | SB3 省心，CleanRL 适合读懂算法 |
| Trackio / W&B | 实验追踪 | 训练曲线可视化，截图入日志 |

## 社区与资讯

- Hugging Face LeRobot Discord — 遥操作/数据采集问题的最佳求助地
- 具身智能相关公众号/知乎专栏 — 中文资讯（甄别质量）
- CoRL / RSS / ICRA 会议论文列表 — 每年扫一遍 keynote 与 best paper

!!! tip "资源使用心法"
    不要收藏夹吃灰！每份资源看完后，在[术语表](concepts/glossary.md)或对应概念笔记里留下痕迹，才算"消化"。
