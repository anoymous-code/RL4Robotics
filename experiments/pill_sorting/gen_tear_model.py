"""一次性生成移动操作机器人模型文件（v5：轮式底盘 + 双臂并排朝前）：

- aloha_tear.xml：两条 vx300s 臂并排装在小车体前部（Mobile ALOHA 形态，车头=车体局部 +y），
  底盘用平面三关节（x / y / yaw）+ 位置伺服近似差速底盘运动学；
- scene_tear.xml：房间场景——固定桌子（承载盒 A / 盒 B / 药板）+ 药柜、矮柜、充电桩。

运行：../../.venv/Scripts/python.exe gen_tear_model.py
"""

from pathlib import Path

import tear_scene as ts

HERE = Path(__file__).resolve().parent

# ---- aloha_tear.xml：双臂 + 车体 ----
src = (HERE / "aloha_nokey.xml").read_text(encoding="utf-8")

# 1) 移除 v1 遗留的固连药板
start = src.index('<body name="pill_pack"')
end = src.index('<site name="left/gripper"', start)
src = src[:start] + src[end:]

# 2) 双臂从"桌面两侧面对面"改为"车体上并排朝前"（车头 = 局部 +y）
Q_FWD = "0.7071068 0 0 0.7071068"   # 绕 z +90°，臂伸展方向局部 +x → 车头 +y
src = src.replace(
    '<body name="left/base_link" childclass="vx300s" pos="-0.469 -0.019 0.02">',
    f'<body name="left/base_link" childclass="vx300s" pos="{-ts.ARM_SPACING/2} 0 0.02" quat="{Q_FWD}">')
src = src.replace(
    '<body name="right/base_link" childclass="vx300s" pos="0.469 -0.019 0.02" quat="0 0 0 1">',
    f'<body name="right/base_link" childclass="vx300s" pos="{ts.ARM_SPACING/2} 0 0.02" quat="{Q_FWD}">')

# 3) 双臂包进 mobile_base（车体几何来自 tear_scene.robot_xml，与布局常量同源）
base_open = f"""<body name="mobile_base" pos="0 0 0">
      <joint name="base_x" type="slide" axis="1 0 0" range="-3 3" damping="10"/>
      <joint name="base_y" type="slide" axis="0 1 0" range="-3 3" damping="10"/>
      <joint name="base_yaw" type="hinge" axis="0 0 1" range="-3.2 3.2" damping="5"/>
{ts.robot_xml()}
      """
anchor = '<body name="left/base_link"'
src = src.replace(anchor, base_open + anchor, 1)
src = src.replace("</worldbody>", "    </body>\n  </worldbody>", 1)

# 4) 底盘位置伺服执行器（近似差速底盘：脚本只命令 车头方向平移 + 原地转向）
actuators = """
  <actuator>
    <position name="base_x" joint="base_x" ctrlrange="-3 3" kp="5000" kv="800"/>
    <position name="base_y" joint="base_y" ctrlrange="-3 3" kp="5000" kv="800"/>
    <position name="base_yaw" joint="base_yaw" ctrlrange="-3.2 3.2" kp="3000" kv="300"/>
  </actuator>
"""
src = src.replace("</mujoco>", actuators + "</mujoco>", 1)
(HERE / "aloha_tear.xml").write_text(src, encoding="utf-8")
print("aloha_tear.xml written,", len(src), "chars")

# ---- scene_tear.xml：房间 + 固定桌子 ----
src = (HERE / "scene_nokey.xml").read_text(encoding="utf-8")
src = src.replace("aloha_scene_nokey", "aloha_scene_tear")
src = src.replace('<include file="aloha_nokey.xml"/>', '<include file="aloha_tear.xml"/>')

wb_start = src.index("<worldbody>") + len("<worldbody>")
wb_end = src.index("</worldbody>")
ty = ts.TABLE_CENTER_Y
room = f"""
    <light pos="0 0.1 2.5"/>
    <light pos="-1.2 -0.6 2.2" diffuse="0.35 0.35 0.35"/>
    <geom name="floor" size="3 3 0.05" type="plane" material="groundplane" pos="0 0 -.75"/>
    <site name="worldref" pos="0 0 -0.75"/>
    <!-- 固定桌子：桌面顶 z=0，盒 A/盒 B/药板都在其上 -->
    <geom mesh="tabletop" material="table" class="visual" pos="0 {ty} -0.75" quat="1 0 0 1"/>
    <geom mesh="tablelegs" material="table" class="visual" pos="0 {ty} -0.75" quat="1 0 0 1"/>
    <geom name="table" pos="0 {ty} -0.1009" size="0.61 0.37 0.1" type="box" class="collision"/>
    <!-- 房间道具：药柜、矮柜、充电桩 -->
    <geom name="med_cabinet" type="box" size="0.25 0.18 0.45" pos="1.20 0.85 -0.30" rgba="0.55 0.45 0.35 1"/>
    <geom type="box" size="0.22 0.02 0.12" pos="1.20 0.68 -0.28" rgba="0.65 0.56 0.45 1"/>
    <geom name="side_table" type="box" size="0.30 0.20 0.25" pos="-1.35 0.85 -0.50" rgba="0.42 0.50 0.55 1"/>
    <geom name="dock" type="box" size="0.04 0.28 0.18" pos="-1.72 -1.50 -0.57" rgba="0.20 0.55 0.38 1"/>
  """
src = src[:wb_start] + room + src[wb_end:]
(HERE / "scene_tear.xml").write_text(src, encoding="utf-8")
print("scene_tear.xml written")
