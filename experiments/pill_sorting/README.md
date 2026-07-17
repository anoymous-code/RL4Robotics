# 分药仿真实验（任务 A）

ALOHA 双臂 MuJoCo 分药场景，含两代任务形态：

- **v2 撕剪分装（主线）**：8 格铝塑板（2×4），格间以可断裂焊接约束模拟易撕线。左手持板，
  右手真实摩擦夹持目标格外缘，扭腕撕下单格（药片保持密封），倒手投入托盘。
- **v1 按压取药（保留）**：左手持 3 格药板悬于杯口，右手指尖直压泡罩，接触力阈值触发
  铝膜"破裂"、药片坠入药杯。

## 准备

```powershell
# 下载 ALOHA 模型（third_party/ 不入库）
powershell -File ..\..\scripts\download_assets.ps1
```

## 文件

| 文件 | 说明 |
|---|---|
| `run_tear_demo.py` | **v2 主演示**：撕剪编排 + 易撕线断裂 + 录像 + 载荷曲线 |
| `tear_scene.py` | v2 场景构建：程序化生成 8 格板 + 12 条焊接易撕线 + 托盘（直接运行可探测/渲染静帧） |
| `aloha_tear.xml` / `scene_tear.xml` | v2 机器人模型（左手持板条）与场景包装，由 `gen_tear_model.py` 从 v1 文件生成 |
| `run_demo.py` | v1 主演示：按压编排 + 破膜模拟 + 录像 + 力曲线（`MultiCam`/`LiveWindow` 亦被 v2 复用） |
| `pill_scene.xml` | v1 顶层场景：药片、药杯、相机 |
| `scene_nokey.xml` / `aloha_nokey.xml` | 去掉 keyframe 的 ALOHA 模型副本，内嵌 3 格药板 |
| `ik_utils.py` | 离线阻尼最小二乘 IK（随机重启 + 多轴姿态目标 `axes=[(局部轴, 世界方向), ...]`） |
| `inspect_scene.py` | v1 场景静态检查（渲染各机位静帧到 debug/） |

## 运行

```powershell
cd experiments\pill_sorting

# v2 撕剪分装（推荐）
..\..\.venv\Scripts\python.exe run_tear_demo.py          # 录制三机位视频
..\..\.venv\Scripts\python.exe run_tear_demo.py --live   # 同时弹窗实时观看（需图形界面）
# 产出: docs/assets/videos/pill_tear_v2_multicam.mp4 与 docs/assets/images/pill_tear_v2_load.png

# v1 按压取药
..\..\.venv\Scripts\python.exe run_demo.py [--live]
# 产出: docs/assets/videos/pill_demo_v1_multicam.mp4 与 docs/assets/images/pill_demo_v1_force.png
```

视频布局：主视角（1280x720）在上，左右腕相机（各 640x360）并排在下，三路画面同一仿真时刻同步渲染。

## v2 建模要点

- **易撕线 = weld 等式约束**：列间 6 + 行间 4 + 持板条 2 共 12 条，按真实邻接拓扑连接；
  监控 `efc_force` 中对应行的合力，超过阈值置 `eq_active=0` 即断裂（不可逆）。
- 焊接需硬化（`solref="0.0015 1" solimp="0.99 0.999 0.0001" torquescale="20"`），否则整板下垂。
- 夹持是真实摩擦夹持：板厚 2.4 mm、指尖咬合 7 mm、右夹爪增益 ×3、`ctrlrange` 下限放开到 0
  （真机软件限位 0.002 导致指间隙最小 4 mm，捏不住毫米级薄板）；抓取目标需补偿指腹中心比
  gripper 站点高 2.2 mm 的偏心；药泡穹顶与药片为纯视觉几何（无碰撞），可放心深捏。
- 投放用"倒手"：IK 转指尖朝下再松爪 + 3 Hz 抖腕。

## 已知简化（待办）

- 药板"已抓稳"（持板条固连在左夹爪），未做从桌面拿板的抓取
- 易撕线断裂阈值为仿真权宜值（真实撕裂力待标定）
- 目标格固定为第 4 列两格（自由端）；中间格需先撕掉相邻格，顺序规划待做
- 无域随机化；相机/光照固定
