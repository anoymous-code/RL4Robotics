"""TIAGo++（开源轮式双臂人形，PAL Robotics / Apache-2.0）集成可行性探测。

把 Menagerie 的 tiago_dual 放进"固定桌 + 盒 A/盒 B"场景：
- 报告自由度/执行器清单（底盘轮速、躯干升降、双臂位置伺服、平行爪）；
- 抬升躯干、双臂摆前伸位，检查手爪到桌面盒子的距离（可达性）；
- 渲染静帧到 debug/tiago_probe_*.png。

运行: ../../.venv/Scripts/python.exe probe_tiago.py
"""

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
TIAGO = (HERE.parents[1] / "third_party" / "mujoco_menagerie" / "pal_tiago_dual"
         / "tiago_dual_position.xml").as_posix()

# 本场景地面 z=0（TIAGo 模型原点在地面），桌面顶 z=0.75
TABLE_TOP = 0.75
TABLE_CENTER = np.array([0.95, 0.0])
BOX_A = np.array([0.80, 0.22, TABLE_TOP])   # 左手侧
BOX_B = np.array([0.82, -0.24, TABLE_TOP])  # 右手侧

XML = f"""
<mujoco model="tiago_pill_probe">
  <include file="tiago_dual_position.xml"/>
  <statistic center="0.6 0 0.8" extent="1.8"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/>
  </visual>
  <asset>
    <texture type="2d" name="groundplane2" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
      rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane2" texture="groundplane2" texuniform="true" texrepeat="5 5"/>
  </asset>
  <worldbody>
    <light pos="0.5 0 2.5"/>
    <geom name="floor2" size="3 3 0.05" type="plane" material="groundplane2"/>
    <geom name="table_top" type="box" size="0.45 0.60 0.02"
      pos="{TABLE_CENTER[0]} {TABLE_CENTER[1]} {TABLE_TOP - 0.02}" rgba="0.72 0.55 0.38 1"/>
    {"".join(f'<geom type="cylinder" size="0.02 {TABLE_TOP/2 - 0.02}" pos="{TABLE_CENTER[0] + sx*0.40} {TABLE_CENTER[1] + sy*0.55} {TABLE_TOP/2 - 0.02}" rgba="0.6 0.6 0.6 1"/>' for sx in (-1,1) for sy in (-1,1))}
    <geom name="boxa" type="box" size="0.028 0.038 0.05" pos="{BOX_A[0]} {BOX_A[1]} {BOX_A[2]+0.05}"
      rgba="0.55 0.42 0.30 1"/>
    <geom name="boxb" type="box" size="0.09 0.075 0.03" pos="{BOX_B[0]} {BOX_B[1]} {BOX_B[2]+0.03}"
      rgba="0.30 0.55 0.75 0.6"/>
    <camera name="probe_main" pos="2.35 -1.5 1.55" xyaxes="0.71 0.71 0 -0.30 0.30 0.9"/>
    <camera name="probe_side" pos="0.9 -1.9 1.3" xyaxes="1 0 0 0 0.28 0.96"/>
  </worldbody>
</mujoco>
"""

# 写到模型目录再加载：mesh 相对路径（./assets/）以主模型文件所在目录为基准
scene_path = Path(TIAGO).parent / "probe_pill_scene.xml"
scene_path.write_text(XML, encoding="utf-8")
model = mujoco.MjModel.from_xml_path(scene_path.as_posix())
data = mujoco.MjData(model)
print(f"模型编译成功: nq={model.nq}, nv={model.nv}, nu={model.nu}")
print("执行器清单:")
for i in range(model.nu):
    print(f"  [{i:2d}] {model.actuator(i).name}")

# 站到桌前，抬躯干，双臂前伸下探
free_adr = model.joint("reference").qposadr[0]
data.qpos[free_adr:free_adr + 7] = [0.0, 0.0, 0.0, 1, 0, 0, 0]

pose = {
    "torso_lift_joint": 0.30,
    "head_2_joint": -0.6,
    # 7DOF 臂前伸姿态（左右镜像）
    "arm_left_1_joint": 1.1, "arm_left_2_joint": 0.4, "arm_left_3_joint": 1.4,
    "arm_left_4_joint": 1.5, "arm_left_5_joint": -1.6, "arm_left_6_joint": 0.6,
    "arm_left_7_joint": 0.0,
    "arm_right_1_joint": 1.1, "arm_right_2_joint": 0.4, "arm_right_3_joint": -1.4,
    "arm_right_4_joint": 1.5, "arm_right_5_joint": 1.6, "arm_right_6_joint": 0.6,
    "arm_right_7_joint": 0.0,
    "gripper_left_left_finger_joint": 0.04, "gripper_left_right_finger_joint": 0.04,
    "gripper_right_left_finger_joint": 0.04, "gripper_right_right_finger_joint": 0.04,
}
for jname, q in pose.items():
    data.qpos[model.joint(jname).qposadr[0]] = q
mujoco.mj_forward(model, data)

for side, target in (("left", BOX_A), ("right", BOX_B)):
    tip = 0.5 * (data.body(f"gripper_{side}_left_finger_link").xpos
                 + data.body(f"gripper_{side}_right_finger_link").xpos)
    d = np.linalg.norm(tip - (target + np.array([0, 0, 0.10])))
    print(f"{side} 爪指间中点 {np.round(tip, 3)}，距盒上方 10cm 处 {d*100:.1f} cm")

renderer = mujoco.Renderer(model, height=720, width=1280)
(HERE / "debug").mkdir(exist_ok=True)
for cam in ("probe_main", "probe_side"):
    renderer.update_scene(data, camera=cam)
    imageio.imwrite(HERE / "debug" / f"tiago_probe_{cam}.png", renderer.render())
renderer.close()
print("已渲染 debug/tiago_probe_*.png")
