# 分药仿真实验（任务 A）

ALOHA 双臂 MuJoCo 分药场景，任务形态经三代演进：

- **v3 全闭环（主线）**：药板竖插在装有多块铝塑板的**盒 A**（3 槽插板架）中。左手真实抓取
  手柄提出 → 空中转体到水平工作位 → 右手摩擦夹持目标格外缘、扭腕沿易撕线撕下单格
  （药片保持密封）→ 投入**盒 B**（药片盒）→ 剩板插回盒 A 原槽位。
- **v2 撕剪（并入 v3）**：8 格铝塑板（2×4），格间以 12 条可断裂焊接约束模拟易撕线；
  v2 中"药板固连左手"的简化已在 v3 中去除。
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
| `run_full_demo.py` | **v3 主演示**：取板 → 撕剪 ×2 → 入盒 B → 放回盒 A 全流程编排 + 录像 + 载荷曲线 |
| `tear_scene.py` | v3 场景构建：盒 A 插板架 + 自由体药板（8 格 + 12 条焊接易撕线）+ 盒 B（直接运行可探测/渲染静帧） |
| `aloha_tear.xml` / `scene_tear.xml` | v3 机器人模型（空手）与场景包装，由 `gen_tear_model.py` 从 v1 文件生成 |
| `run_demo.py` | v1 主演示：按压编排 + 破膜模拟 + 录像 + 力曲线（`MultiCam`/`LiveWindow` 亦被 v3 复用） |
| `pill_scene.xml` | v1 顶层场景：药片、药杯、相机 |
| `scene_nokey.xml` / `aloha_nokey.xml` | 去掉 keyframe 的 ALOHA 模型副本，内嵌 3 格药板 |
| `ik_utils.py` | 离线阻尼最小二乘 IK（随机重启 + 多轴姿态目标 `axes=[(局部轴, 世界方向), ...]`） |
| `inspect_scene.py` | v1 场景静态检查（渲染各机位静帧到 debug/） |

## 运行

```powershell
cd experiments\pill_sorting

# v3 全闭环（推荐）
..\..\.venv\Scripts\python.exe run_full_demo.py             # 录制三机位视频
..\..\.venv\Scripts\python.exe run_full_demo.py --live      # 同时弹窗实时观看（需图形界面）
..\..\.venv\Scripts\python.exe run_full_demo.py --no-latch  # 左手纯摩擦对照（撕剪反力矩下会滑）
# 产出: docs/assets/videos/pill_full_v3_multicam.mp4 与 docs/assets/images/pill_full_v3_load.png

# v1 按压取药
..\..\.venv\Scripts\python.exe run_demo.py [--live]
# 产出: docs/assets/videos/pill_demo_v1_multicam.mp4 与 docs/assets/images/pill_demo_v1_force.png
```

视频布局：主视角（1280x720）在上，左右腕相机（各 640x360）并排在下，三路画面同一仿真时刻同步渲染。

## v3 建模要点

- **易撕线 = weld 等式约束**：列间 6 + 行间 4 + 持板条 2 共 12 条，按真实邻接拓扑连接；
  监控 `efc_force` 中对应行的合力，超过阈值置 `eq_active=0` 即断裂（不可逆）。
  焊接需硬化（`solref="0.0015 1" solimp="0.99 0.999 0.0001" torquescale="20"`），否则整板下垂。
- **左手抓取 = 真实闭爪 + 锁定焊接（sticky gripper）**：预定义 `active="false"` 的 weld，
  闭爪后把当前手-板相对位姿写入 `eq_data` relpose 再激活，放板前解除。纯摩擦夹持对抗撕剪
  反力矩（~0.5 N·m）会滑移，锁定等效于真机更强夹持力/自锁指形；右手全程纯摩擦。
- **上游误差要在上游修**：板在槽中沉降有 ~2.5° 倾斜，若左手按标准姿态持板，右爪对倾斜
  薄板只剩单点接触。工作位姿态按"板要水平"用抓取时记录的相对旋转反解，倾角 0.2°。
- **断裂阈值与持物刚度耦合**：悬持板比固连板柔，同阈值下断裂弹射更猛；阈值 6.0 → 4.5，
  断后停 0.5 s 再运送。
- 右手薄板夹持：板厚 2.4 mm、指尖咬合 8.5 mm、夹爪增益 ×6、`ctrlrange` 下限放开到 0；
  抓取目标需补偿指腹中心相对 gripper 站点的偏心（站点系 [-13.8, +2.2, 0] mm）。
- 投放用"倒手"：指尖朝下 7.5 cm 低空松爪 + 0.28 rad / 3.5 Hz 抖腕。
- 放回插槽：下插深度按剩余板长动态计算（撕掉一列后板短 24 mm），板底入槽后松爪自行落座。

## 已知简化（待办）

- 左手夹持用锁定焊接抽象（`--no-latch` 可对照），真实指形自锁/更大夹持力待真机验证
- 易撕线断裂阈值为仿真权宜值（真实撕裂力待标定）
- 盒 A 中装饰板为静态几何；目标板固定在中间槽
- 目标格固定为第 4 列两格（自由端）；中间格需先撕掉相邻格，顺序规划待做
- 无域随机化；相机/光照固定
