# 目标任务总览

我们选定两个高价值、高难度的养老照护任务作为"北极星"，所有学习都围绕它们展开。

<div class="grid cards" markdown>

- :material-pill:{ .lg .middle } **任务 A · 自动分药** <span class="badge badge-hard">难度 ★★★★☆</span>

    ---

    从铝塑板（泡罩包装）中按压取出药片，按"人-天-时段"分装到药盒。核心挑战是**双臂协同 + 毫米级力控**。

    [:octicons-arrow-right-24: 任务拆解](pill-sorting.md)

- :material-heart-pulse:{ .lg .middle } **任务 B · 测量血压** <span class="badge badge-hard">难度 ★★★★★</span>

    ---

    为老人佩戴电子血压计袖带并完成测量。核心挑战是**安全的人机物理交互**——这是两个任务中风险最高的。

    [:octicons-arrow-right-24: 任务拆解](blood-pressure.md)

</div>

## 为什么先做分药、后做量血压？

| 维度 | 分药 | 量血压 |
|---|---|---|
| 操作对象 | 无生命物体（药板、药盒） | **老人的身体** |
| 失败后果 | 药片损坏、掉落（可重试） | 夹伤、拉伤、心理惊吓（不可接受） |
| 灵巧性要求 | 极高（毫米级、双手协同） | 高（袖带缠绕、松紧适度） |
| 感知难度 | 高（药片识别、药板姿态） | 极高（人体姿态、皮肤接触状态） |
| 可仿真程度 | 较高（刚体+简单变形） | 低（人体软组织难以高保真仿真） |

**结论**：分药可以在仿真中充分打磨，是理想的第一战场；量血压涉及人体，仿真难度和安全门槛都更高，放在真机阶段（阶段 6）攻关，且从"辅助老人自己戴袖带"的低风险形态切入。

## 任务状态看板

| 子任务 | 所属 | 状态 | 关联日志 |
|---|---|---|---|
| 任务定义与拆解 | A + B | <span class="badge badge-done">完成</span> | [2026-07-10](../journal/2026-07-10.md) |
| 分药场景仿真建模 v0 | A | <span class="badge badge-done">完成</span> | [2026-07-10 · 实验 2](../journal/2026-07-10.md#exp2-pill-demo) |
| 铝塑板按压（脚本化 v0，压杆工具，3/3） | A | <span class="badge badge-done">完成</span> | [2026-07-10 · 实验 2](../journal/2026-07-10.md#exp2-pill-demo) |
| 徒手双臂协同按压（v1，指尖直压，3/3） | A | <span class="badge badge-done">完成</span> | [2026-07-10 · 实验 3](../journal/2026-07-10.md#exp3-pill-v1) |
| 撕剪分装 v2（8 格板撕单格入托盘，2/2） | A | <span class="badge badge-done">完成</span> | [2026-07-17 (下)](../journal/2026-07-17-tear.md) |
| 全闭环 v3（盒 A 取板→撕剪→入盒 B→放回盒 A） | A | <span class="badge badge-done">完成</span> | [2026-07-17 (晚)](../journal/2026-07-17-full.md) |
| 移动操作机器人 v5（轮式双臂 + 固定桌，导航 + 全闭环分药） | A | <span class="badge badge-done">完成</span> | [2026-07-18](../journal/2026-07-18-mobile.md) |
| 目标格随机化 + 鲁棒性评测 | A | <span class="badge badge-todo">未开始</span> | - |
| 分药 Gymnasium 环境 + 域随机化 | A | <span class="badge badge-todo">未开始</span> | - |
| 铝塑板按压策略（学习版） | A | <span class="badge badge-todo">未开始</span> | - |
| 药片识别数据集 | A | <span class="badge badge-todo">未开始</span> | - |
| 袖带缠绕可行性调研 | B | <span class="badge badge-todo">未开始</span> | - |
