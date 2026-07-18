# 分药仿真实验（任务 A）

ALOHA 双臂 MuJoCo 分药场景，任务形态经多代演进：

- **v5 移动操作机器人（主线）**：机器人 = 小车体 + 两条 vx300s 臂**并排朝前**
  （Mobile ALOHA 形态；底盘平面三关节 x/y/yaw + 位置伺服近似差速车，轮-地接触不建模）。
  盒 A、盒 B、药板放在**固定桌子**上；机器人驶离充电桩停到桌前，双臂越过桌沿操作。
- **v3/v4 全闭环**：药板竖插在装有多块铝塑板的**盒 A**（3 槽插板架）中。左手真实
  抓取手柄提出 → 空中转体到水平工作位 → 右手摩擦夹持目标格外缘、扭腕沿易撕线撕下单格
  （药片保持密封）→ 投入**盒 B**（药片盒）→ 剩板插回盒 A 原槽位。
  （v4 曾把盒子装在车上的"移动分药车"形态，已被 v5 纠正为固定桌形态。）
- **v2 撕剪（并入 v3）**：8 格铝塑板（2×4），格间以 12 条可断裂焊接约束模拟易撕线。
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
| `run_full_demo.py` | **v5 主演示 / 脚本专家**：导航到桌前 → 取板 → 撕剪 → 入盒 B → 放回盒 A（支持 `SceneCfg` 域随机化、跳导航、录制钩子） |
| `pill_env.py` | Gymnasium 环境：reset 域随机化，动作 14 维（双臂+爪），观测 = qpos + 机载三相机图像 |
| `collect_demos.py` | 脚本专家自动采集演示数据（ACT 风格 HDF5，含成功率统计），产物在 `demos/`（不入库） |
| `train_act.py` | ACT-lite 模仿学习训练（ResNet18 + Transformer 动作分块，L1 损失，AMP），产出 `ckpt/` 与训练曲线 |
| `eval_act.py` | 策略评测：随机场景 rollout 成功率 + 四视角视频 |
| `tear_refine_env.py` | RL 精修环境：撕剪-投放子任务（semi-MDP 相位级修正动作 + 物理随机化 + 特权观测），`--gen-pool` 生成重置池 / `--smoke` 一致性检查 / `--baseline` 脚本基线 |
| `train_ppo_refine.py` | PPO 训练相位级修正策略（SB3，8 并行环境，CPU） |
| `eval_refine.py` | 配对评测：同一批（场景×物理参数）对比零动作脚本 vs RL 策略，可录四视角视频 |
| `plot_refine_curve.py` | 从 VecMonitor csv 画 PPO 训练曲线 |
| `demo_to_video.py` | 把一条 HDF5 演示合成三机位视频（数据样例可视化） |
| `wait_and_train.ps1` | 接力脚本：等采集进程结束自动启动训练（规避页面文件不足） |
| `tear_scene.py` | v5 场景构建：机器人车体几何 + 固定桌上的盒 A/盒 B + 自由体药板（8 格 + 12 条焊接易撕线）+ 底盘工具函数（直接运行可探测/渲染静帧） |
| `aloha_tear.xml` / `scene_tear.xml` | v5 机器人模型（双臂并排朝前挂 mobile_base）与房间场景（固定桌子），由 `gen_tear_model.py` 生成 |
| `run_demo.py` | v1 主演示：按压编排 + 破膜模拟 + 录像 + 力曲线（`MultiCam`/`LiveWindow` 亦被 v3 复用） |
| `pill_scene.xml` | v1 顶层场景：药片、药杯、相机 |
| `scene_nokey.xml` / `aloha_nokey.xml` | 去掉 keyframe 的 ALOHA 模型副本，内嵌 3 格药板 |
| `ik_utils.py` | 离线阻尼最小二乘 IK（随机重启 + 多轴姿态目标 `axes=[(局部轴, 世界方向), ...]`） |
| `inspect_scene.py` | v1 场景静态检查（渲染各机位静帧到 debug/） |

## 运行

```powershell
cd experiments\pill_sorting

# v5 移动机器人全流程（推荐）
..\..\.venv\Scripts\python.exe run_full_demo.py             # 录制三机位视频
..\..\.venv\Scripts\python.exe run_full_demo.py --live      # 同时弹窗实时观看（需图形界面）
..\..\.venv\Scripts\python.exe run_full_demo.py --no-latch  # 左手纯摩擦对照（撕剪反力矩下会滑）
# 产出: docs/assets/videos/pill_full_v5_mobile_multicam.mp4 与 docs/assets/images/pill_full_v5_load.png

# 学习管线：环境烟测 / 演示数据采集（域随机化 + 成功率统计）
..\..\.venv\Scripts\python.exe pill_env.py
..\..\.venv\Scripts\python.exe collect_demos.py --n 50 --seed 0   # 产物在 demos/

# v1 按压取药
..\..\.venv\Scripts\python.exe run_demo.py [--live]
# 产出: docs/assets/videos/pill_demo_v1_multicam.mp4 与 docs/assets/images/pill_demo_v1_force.png
```

视频布局：主视角（1280x720）在上，左右腕相机（各 640x360）并排在下，三路画面同一仿真时刻同步渲染。

## v3~v5 建模要点

- **机器人 = 小车体 + 双臂并排朝前（v5）**：两条 vx300s 臂间距 0.47 m 装在车顶平台前部，
  同朝车头（车体局部 +y）；臂基座平台与桌面同高，停车后双臂越过桌沿操作。并排布局下
  左/右臂仍分居操作区 -x/+x 两侧，v3 的姿态轴常量全部保留，只重排了盒子与工作位坐标。
- **底盘 = 平面三关节 + 位置伺服**：世界系 slide x/y + 车体 hinge yaw，脚本按差速车
  编排（原地转向 → 沿车头直行 → 回正停到桌前）；轮-地接触不建模，轮子为纯视觉几何。
  车体大质量（~70 kg）+ 关节阻尼使操作反力不会推动底盘。
- **停车包络**：车头缘距桌沿 6 cm，臂基座距盒 A ~0.39 m（vx300s 有效可达 ~0.6 m）——
  停车容差的设计依据。
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

- 底盘为平面关节 + 位置伺服近似（无轮胎滑移/里程计误差）；导航为直线预设路径，无避障
- 停车位姿假定精确到位（真机需感知重标定操作目标，操作目标目前以世界系表达）
- 左手夹持用锁定焊接抽象（`--no-latch` 可对照），真实指形自锁/更大夹持力待真机验证
- 易撕线断裂阈值为仿真权宜值（真实撕裂力待标定）
- 盒 A 中装饰板为静态几何；目标板固定在中间槽
- 目标格固定为第 4 列两格（自由端）；中间格需先撕掉相邻格，顺序规划待做
- 无域随机化；相机/光照固定
