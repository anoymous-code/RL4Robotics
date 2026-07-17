# 学习日志

这里是我们的"实验记录本"：每次学习/实验后记录当天的收获、数据、图表和视频。**好记录的标准：三个月后回看，能立刻复现当时的结论。**

## 时间线

<div class="timeline" markdown>
<div class="tl-item" markdown>
<span class="tl-date">2026-07-17 (晚)</span>

**[分药 v3：盒 A 取板 → 撕剪 → 入盒 B → 放回盒 A 全闭环](2026-07-17-full.md)** — 去掉"药板固连左手"的简化：药板竖插在装有多块铝塑板的盒 A 插板架中，左手真实抓取手柄提出、空中转体到工作位，撕剪两格投入盒 B（**2/2**），剩板插回盒 A 原槽位（**成功**），全流程 53 秒。
</div>
<div class="tl-item" markdown>
<span class="tl-date">2026-07-17 (下)</span>

**[分药 v2：双臂撕剪单格泡罩入托盘](2026-07-17-tear.md)** — 任务升级为贴近真实分药流程的"撕剪"：8 格铝塑板用 12 条可断裂焊接约束当易撕线，右手真实摩擦夹持单格外缘、扭腕撕下、倒手投入托盘，**2/2 成功**（附易撕线载荷曲线）。
</div>
<div class="tl-item" markdown>
<span class="tl-date">2026-07-17</span>

**[三机位同步可视化](2026-07-17.md)** — 分药演示升级为主视角 + 左右腕相机三路同步录制（附 `--live` 实时弹窗模式）；腕相机正是未来模仿学习的标准观测输入，这条渲染管线是数据采集管线的雏形。
</div>
<div class="tl-item" markdown>
<span class="tl-date">2026-07-10</span>

**[启程：项目定义、知识库搭建与分药仿真首秀](2026-07-10.md)** — 确定目标任务与七阶段路线图；搭建本知识库网站并部署上线；跑通 MuJoCo 环境（实验 1：随机策略基线视频）；**ALOHA 双臂分药仿真 v0/v1 均达成 3/3 入杯**（实验 2/3：视频 + 破膜力曲线）。
</div>
</div>

## 日志模板

新建日志时复制以下模板（保存为 `docs/journal/YYYY-MM-DD.md` 并添加到 `mkdocs.yml` 导航）：

```markdown
# YYYY-MM-DD · 标题

## 今日目标
-

## 学到了什么
-

## 实验记录
<!-- 训练曲线图、仿真截图放 docs/assets/images/，视频用 <video> 标签嵌入 -->
| 实验 | 配置 | 结果 | 结论 |
|---|---|---|---|

## 踩坑与解决
-

## 明日计划
-
```

!!! tip "多媒体嵌入方法"
    - **图片**：放入 `docs/assets/images/`，用 `![说明](../assets/images/xxx.png)` 引用（Markdown 语法会自动修正路径）
    - **视频**：放入 `docs/assets/videos/`，用 `<video controls src="../../assets/videos/xxx.mp4"></video>` 嵌入（注意：HTML 标签不会自动修正路径，日志页面渲染后深一层目录，所以要用 `../../`）
    - **训练曲线**：matplotlib 导出 PNG/SVG，或后续接入 Trackio/W&B 截图
