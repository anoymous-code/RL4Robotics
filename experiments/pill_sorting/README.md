# 分药仿真实验（任务 A）

ALOHA 双臂 + 3 格铝塑药板 + 药杯的 MuJoCo 场景。v0 为脚本化演示：左臂持板、右臂按压、
接触力阈值触发铝膜"破裂"、药片坠入药杯。

## 准备

```powershell
# 下载 ALOHA 模型（third_party/ 不入库）
powershell -File ..\..\scripts\download_assets.ps1
```

## 文件

| 文件 | 说明 |
|---|---|
| `pill_scene.xml` | 顶层场景：药片、药杯、相机（include 下面两个文件） |
| `scene_nokey.xml` / `aloha_nokey.xml` | 去掉 keyframe 的 ALOHA 模型副本，内嵌药板与压杆 |
| `ik_utils.py` | 离线阻尼最小二乘 IK（带随机重启与 z 轴姿态目标） |
| `run_demo.py` | 主演示：编排 + 破膜模拟 + 录像 + 力曲线 |
| `inspect_scene.py` | 场景静态检查（渲染各机位静帧到 debug/） |
| `debug_ik2.py` | IK 可达性诊断 |

## 运行

```powershell
cd experiments\pill_sorting
..\..\.venv\Scripts\python.exe run_demo.py
# 产出: docs/assets/videos/pill_demo_v0.mp4 与 docs/assets/images/pill_demo_v0_force.png
```

## 已知简化（v1 待办）

- 药板"已抓稳"（固连在左夹爪），未做抓取
- 破膜阈值 1 N 为仿真权宜值（真实 5~30 N，待标定）；左臂伺服柔性限制了力的建立
- 按压用固连压杆代替夹爪指尖
- 无域随机化；相机/光照固定
