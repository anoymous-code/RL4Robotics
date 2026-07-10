# RL4Robotics · 移动操作学习实验室

强化学习 × 世界模型 × VLA —— 从零搭建养老场景双臂移动操作机器人（自动分药 / 辅助测量血压）的学习记录与知识库。

## 快速开始

```powershell
# 安装依赖（如遇代理错误，先执行 $env:NO_PROXY='*'）
python -m pip install -r requirements.txt

# 启动本地网站（默认 http://127.0.0.1:8000）
python -m mkdocs serve
```

## 目录结构

```
├── mkdocs.yml          # 网站配置与导航
├── docs/
│   ├── index.md        # 首页：项目愿景
│   ├── roadmap.md      # 七阶段学习路线图
│   ├── concepts/       # 概念笔记（RL / 世界模型 / VLA / 术语表）
│   ├── tasks/          # 目标任务拆解（分药 / 量血压）
│   ├── hardware/       # 硬件选型调研
│   ├── journal/        # 按日期的学习日志（实验数据、图表、视频）
│   ├── resources.md    # 论文 / 课程 / 工具精选
│   └── assets/         # 图片、视频、CSS、JS
└── requirements.txt
```

## 日常使用

1. 学习/实验后，在 `docs/journal/` 新建 `YYYY-MM-DD.md`（模板见日志索引页）
2. 图片放 `docs/assets/images/`，视频放 `docs/assets/videos/`
3. 在 `mkdocs.yml` 的 `nav:` 中添加新页面
4. `mkdocs serve` 实时预览；`mkdocs build` 生成静态站点（`site/` 目录，可部署到任意静态托管）
