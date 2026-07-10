# 术语表

学习过程中遇到的所有术语都会沉淀在这里，按主题分组，持续更新。

## 强化学习

| 术语 | 英文 | 一句话解释 |
|---|---|---|
| 马尔可夫决策过程 | MDP | RL 的数学框架：状态、动作、转移、奖励、折扣因子五元组 |
| 策略 | Policy (π) | 从状态到动作的映射，机器人的"行为准则" |
| 值函数 | Value Function | 评估某状态（或状态-动作对）未来能拿多少累积奖励 |
| 策略梯度 | Policy Gradient | 直接对策略参数求梯度来提升期望回报的方法族 |
| PPO | Proximal Policy Optimization | 最常用的策略梯度算法，稳定、好调参 |
| SAC | Soft Actor-Critic | 带最大熵探索的离线策略算法，样本效率高 |
| 奖励塑形 | Reward Shaping | 人为设计中间奖励引导学习，设计不当会被"钻空子" |
| 域随机化 | Domain Randomization | 训练时随机化仿真参数（摩擦、光照等），提升真机泛化 |
| Sim-to-Real | Sim-to-Real Transfer | 把仿真中学到的策略迁移到真实机器人 |

## 模仿学习与数据

| 术语 | 英文 | 一句话解释 |
|---|---|---|
| 模仿学习 | Imitation Learning (IL) | 从人类演示数据中学习策略 |
| 行为克隆 | Behavior Cloning (BC) | 最简单的 IL：把演示当监督学习数据拟合 |
| ACT | Action Chunking Transformer | ALOHA 系列使用的 IL 算法，一次预测一段动作序列 |
| 扩散策略 | Diffusion Policy | 用扩散模型生成动作，擅长多模态动作分布 |
| 遥操作 | Teleoperation | 人远程操控机器人来采集演示数据 |
| 主从臂 | Leader-Follower Arms | 人拖动小的主臂，大的从臂同步复现（ALOHA/GELLO 方案） |
| LeRobot | LeRobot | Hugging Face 的开源机器人学习全流程框架 |

## 世界模型

| 术语 | 英文 | 一句话解释 |
|---|---|---|
| 世界模型 | World Model | 学习环境动力学 p(s'\|s,a)，可在"想象"中预演 |
| DreamerV3 | DreamerV3 | 在潜空间想象中训练策略的代表算法 |
| TD-MPC2 | TD-MPC2 | 学习模型 + 在线模型预测控制的代表算法 |
| 潜空间 | Latent Space | 把高维观测（图像）压缩成的低维表示 |
| 模型预测控制 | MPC | 每一步都用模型向前推演多个候选动作序列，选最优的执行 |

## VLA 与大模型

| 术语 | 英文 | 一句话解释 |
|---|---|---|
| VLA | Vision-Language-Action Model | 输入图像+语言、输出动作的机器人大模型 |
| VLM | Vision-Language Model | 视觉-语言模型，VLA 的"上半身" |
| 动作分词化 | Action Tokenization | 把连续动作离散成 token，让 Transformer 能自回归输出 |
| 流匹配 | Flow Matching | π0 使用的连续动作生成方法，类似扩散但更快 |
| 具身智能 | Embodied AI | 拥有身体、能与物理世界交互的 AI 的统称 |

## 机器人学

| 术语 | 英文 | 一句话解释 |
|---|---|---|
| 自由度 | DoF (Degrees of Freedom) | 机构可独立运动的维度数，如 7-DoF 机械臂 |
| 正/逆运动学 | FK / IK | 关节角→末端位姿 / 末端位姿→关节角 |
| 末端执行器 | End Effector | 机械臂末端的工具：夹爪、灵巧手等 |
| 力控 / 阻抗控制 | Force / Impedance Control | 控制接触力而非位置，柔顺交互的基础 |
| 全身控制 | Whole-Body Control | 底盘+双臂+躯干统一协调的控制方法 |
| URDF | Unified Robot Description Format | 描述机器人结构的 XML 格式 |
| 移动操作 | Mobile Manipulation | 移动底盘 + 机械臂操作的复合能力 |

!!! tip "使用建议"
    读论文碰到不认识的词，先来这里查；查不到就添加进来——**解释给别人听是最好的学习**。
