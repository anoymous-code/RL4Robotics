# 学习日志

这里是我们的"实验记录本"：每次学习/实验后记录当天的收获、数据、图表和视频。**好记录的标准：三个月后回看，能立刻复现当时的结论。**

## 时间线

<div class="timeline" markdown>
<div class="tl-item" markdown>
<span class="tl-date">2026-07-10</span>

**[启程：项目定义与知识库搭建](2026-07-10.md)** — 确定目标任务（分药、量血压）、技术路线（RL × 世界模型 × VLA）与七阶段路线图；搭建本知识库网站。
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
