"""分药 v5 场景：轮式移动操作机器人（双臂并排朝前）+ 固定桌子。

- 机器人 = 小车体 + 两条 vx300s 臂并排朝前（Mobile ALOHA 形态，车头=车体局部 +y）；
  底盘用平面三关节（世界系 x / y + 车体 yaw）+ 位置伺服近似差速底盘运动学，
  轮-地接触不建模（轮子为纯视觉几何）；
- 盒 A（插板架）、盒 B（药片盒）、药板都放在**固定的桌子**上（桌面顶 z=0）；
- 药板 = 手柄条（strip，自由体）+ 8 格（2x4，自由体），以 12 条可断裂焊接约束
 （易撕线）相连，竖插在盒 A 中间槽位；
- 机器人从充电桩驶到桌前停稳，双臂执行取板 → 撕剪 → 入盒 B → 放回盒 A。

直接运行本文件可做几何/坐标探测并渲染静帧：
    ../../.venv/Scripts/python.exe tear_scene.py
"""

from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent

# ---------- 板块布局（strip 局部坐标，单位 m） ----------
N_COLS, N_ROWS = 4, 2
SEG_HX, SEG_HY, SEG_HZ = 0.0115, 0.0115, 0.0012   # 单格半尺寸（板厚 2.4mm，贴近真实铝塑板）
PITCH_X, PITCH_Y = 0.024, 0.025
COL_X0 = 0.0225            # 第 0 列中心相对 strip 中心的 x 偏移
ROW_Y = (-0.0125, 0.0125)  # 行中心 y（f=前排 -y，b=后排 +y）
ROW_TAG = ("f", "b")

TAB_LOCAL = np.array([-0.0245, 0.0, -0.0015])  # 手柄 tab 中心（strip 局部）
TAB_HALF = (0.0075, 0.012, 0.0045)             # tab 半尺寸（厚 9mm，便于夹持）
BOARD_XMAX = COL_X0 + 3 * PITCH_X + SEG_HX     # 板自由端（局部 +x 边缘）= 0.1055

PILL_COLORS = ["0.95 0.45 0.35", "0.98 0.80 0.30", "0.40 0.75 0.55", "0.45 0.60 0.90"]

# ---------- 机器人（轮式底盘 + 并排双臂） ----------
# 车体局部系：原点 = 两臂基座中点的地面投影，车头 = 局部 +y，臂基座平台顶 z=0；
# 底盘三关节 base_x/base_y（世界系平移）+ base_yaw，位置伺服近似差速运动学
ARM_SPACING = 0.47                            # 两臂基座间距（并排，左臂 -x / 右臂 +x）
BASE_START = np.array([-1.35, -1.50, 0.0])    # 充电桩旁出发位 (x, y, yaw)
BASE_WORK = np.array([0.0, -0.22, 0.0])       # 桌前停车位：车头(+y)朝向桌子

# ---------- 固定桌子（世界系，桌面顶 z=0） ----------
TABLE_CENTER_Y = 0.40                          # 桌面中心 y（桌深 0.76，近缘 y=0.02）

# ---------- 盒 A（插板架）与盒 B（药片盒），固定桌面世界坐标 ----------
BOX_A_CENTER = np.array([-0.20, 0.16, 0.0])
SLOT_PITCH = 0.014          # 槽间距（x 向）；槽宽 = pitch - 隔板厚
SLOT_WALL_T = 0.002         # 隔板半厚
BOX_A_HY = 0.032            # 内腔 y 半宽
BOX_A_WALL_H = 0.05
BOX_A_FLOOR_TOP = 0.006     # 内底面高度
# 目标板插中间槽；竖立时 strip 局部 +x 朝下（自由端在下，tab 在上）
BOARD_UP_QUAT = np.array([0.7071068, 0.0, 0.7071068, 0.0])  # Ry(90°)
BOARD_HOME = np.array([BOX_A_CENTER[0], BOX_A_CENTER[1],
                       BOX_A_FLOOR_TOP + BOARD_XMAX + 0.002])

BOX_B_CENTER = np.array([0.16, 0.14, 0.0])
BOX_B_HX, BOX_B_HY, BOX_B_WALL_H = 0.09, 0.075, 0.03


def seg_name(col, row):
    return f"seg_{col}{ROW_TAG[row]}"


def seg_offset(col, row):
    """段中心在 strip 局部坐标中的偏移。"""
    return np.array([COL_X0 + col * PITCH_X, ROW_Y[row], 0.0])


def all_segments():
    return [(c, r) for c in range(N_COLS) for r in range(N_ROWS)]


def weld_names_of(col, row):
    """返回连接该段的所有焊接（易撕线）名称。"""
    names = []
    if col == 0:
        names.append(f"w_strip_{ROW_TAG[row]}")
    else:
        names.append(f"w_col{col-1}{col}_{ROW_TAG[row]}")
    if col < N_COLS - 1:
        names.append(f"w_col{col}{col+1}_{ROW_TAG[row]}")
    names.append(f"w_row_c{col}")
    return names


def _quat_to_mat(quat):
    m = np.empty(9)
    mujoco.mju_quat2Mat(m, quat)
    return m.reshape(3, 3)


def robot_xml():
    """机器人车体几何（mobile_base 车体局部坐标，车头 = 局部 +y）。

    由 gen_tear_model.py 嵌入 aloha_tear.xml 的 mobile_base body 中；
    双臂基座另行挂在 mobile_base 下（并排 ±ARM_SPACING/2，朝 +y）。
    """
    parts = [
        # 臂基座平台（顶面 z=0）
        '<geom name="robot_deck" type="box" size="0.30 0.20 0.015" pos="0 0 -0.015" '
        'mass="8" rgba="0.30 0.32 0.35 1"/>',
        # 车身立柱（纯视觉，底盘高度由无 z 自由度的平面关节保证）
        '<geom name="robot_body" type="box" size="0.26 0.17 0.26" pos="0 -0.02 -0.29" '
        'contype="0" conaffinity="0" mass="40" rgba="0.25 0.27 0.30 1"/>',
        '<geom type="box" size="0.265 0.175 0.025" pos="0 -0.02 -0.08" '
        'contype="0" conaffinity="0" mass="1" rgba="0.20 0.55 0.55 1"/>',
        # 底盘（碰撞体，防撞桌腿/道具）
        '<geom name="robot_chassis" type="box" size="0.28 0.20 0.075" pos="0 -0.02 -0.62" '
        'mass="20" rgba="0.15 0.16 0.18 1"/>',
    ]
    for sx in (-1, 1):  # 四轮（纯视觉）
        for sy in (-1, 1):
            parts.append(f'<geom type="cylinder" size="0.075 0.030" '
                         f'pos="{sx * 0.22} {sy * 0.14 - 0.02} -0.675" quat="0.7071 0.7071 0 0" '
                         f'contype="0" conaffinity="0" mass="1.5" rgba="0.12 0.12 0.13 1"/>')
    return "\n".join("      " + p for p in parts)


def boxes_xml():
    """固定桌面上的盒 A（插板架 + 装饰板）与盒 B（药片盒），世界坐标。"""
    parts = []
    # 盒 A：3 槽插板架
    ax, ay = BOX_A_CENTER[0], BOX_A_CENTER[1]
    inner_hx = 1.5 * SLOT_PITCH + SLOT_WALL_T
    parts.append(f'<geom name="boxa_bottom" type="box" size="{inner_hx + 0.006} {BOX_A_HY + 0.006} 0.003" '
                 f'pos="{ax} {ay} 0.003" rgba="0.55 0.42 0.30 1"/>')
    for sx in (-1, 1):
        parts.append(f'<geom type="box" size="0.003 {BOX_A_HY + 0.006} {BOX_A_WALL_H}" '
                     f'pos="{ax + sx * (inner_hx + 0.003):.4f} {ay} {BOX_A_WALL_H}" rgba="0.55 0.42 0.30 0.7"/>')
    for sy in (-1, 1):
        parts.append(f'<geom type="box" size="{inner_hx + 0.006} 0.003 {BOX_A_WALL_H}" '
                     f'pos="{ax} {ay + sy * (BOX_A_HY + 0.003):.4f} {BOX_A_WALL_H}" rgba="0.55 0.42 0.30 0.7"/>')
    for k in (-1, 1):
        parts.append(f'<geom type="box" size="{SLOT_WALL_T} {BOX_A_HY} 0.045" '
                     f'pos="{ax + k * SLOT_PITCH / 2:.4f} {ay} {0.006 + 0.045}" rgba="0.60 0.48 0.35 0.9"/>')
    # 两块装饰板（静态），比目标板矮，避免左手指下降抓取时磕碰
    for k, h in ((-1, 0.042), (1, 0.038)):
        px = ax + k * SLOT_PITCH
        parts.append(f'<geom type="box" size="0.0012 0.026 {h}" '
                     f'pos="{px:.4f} {ay} {BOX_A_FLOOR_TOP + h + 0.001:.4f}" rgba="0.78 0.79 0.84 1"/>')
        parts.append(f'<geom type="box" size="0.002 0.011 0.006" '
                     f'pos="{px:.4f} {ay} {BOX_A_FLOOR_TOP + 2 * h - 0.004:.4f}" '
                     f'contype="0" conaffinity="0" rgba="0.3 0.35 0.45 1"/>')

    # 盒 B：药片盒
    bx, by = BOX_B_CENTER[0], BOX_B_CENTER[1]
    parts.append(f'<geom name="boxb_bottom" type="box" size="{BOX_B_HX} {BOX_B_HY} 0.0015" '
                 f'pos="{bx} {by} 0.0015" rgba="0.30 0.55 0.75 1"/>')
    for sx, sy, hx, hy in ((1, 0, 0.003, BOX_B_HY), (-1, 0, 0.003, BOX_B_HY),
                           (0, 1, BOX_B_HX, 0.003), (0, -1, BOX_B_HX, 0.003)):
        px = bx + sx * (BOX_B_HX + 0.003)
        py = by + sy * (BOX_B_HY + 0.003)
        parts.append(f'<geom type="box" size="{hx} {hy} {BOX_B_WALL_H}" '
                     f'pos="{px:.4f} {py:.4f} {BOX_B_WALL_H}" rgba="0.30 0.55 0.75 0.5"/>')
    return "\n".join("    " + p for p in parts)


def build_xml():
    """生成完整场景 XML 字符串。

    药板初始竖插在固定桌上盒 A 的中间槽（焊接 relpose 取 qpos0，
    整板任意全局位姿不影响格间刚性布局）。
    """
    R0 = _quat_to_mat(BOARD_UP_QUAT)
    quat_str = " ".join(f"{q:.7f}" for q in BOARD_UP_QUAT)

    # 药板：strip 自由体，初始在固定桌上的盒 A 槽中
    strip_p = BOARD_HOME
    bodies = [f"""
    <body name="strip" pos="{strip_p[0]:.6f} {strip_p[1]:.6f} {strip_p[2]:.6f}" quat="{quat_str}">
      <freejoint name="strip_joint"/>
      <geom name="strip_tab" type="box" size="{TAB_HALF[0]} {TAB_HALF[1]} {TAB_HALF[2]}"
        pos="{TAB_LOCAL[0]} {TAB_LOCAL[1]} {TAB_LOCAL[2]}" friction="2 0.01 0.001"
        solref="0.004 1" rgba="0.25 0.3 0.4 1" mass="0.004"/>
      <geom name="strip_bar" type="box" size="0.010 0.027 0.0012" pos="0 0 0"
        friction="1.5 0.01 0.001" rgba="0.80 0.81 0.85 1" mass="0.003"/>
      <site name="strip_center" pos="0 0 0" group="5"/>
      <site name="strip_tab_top" pos="{TAB_LOCAL[0]} 0 0" group="5"/>
    </body>"""]

    for c, r in all_segments():
        p = strip_p + R0 @ seg_offset(c, r)
        color = PILL_COLORS[c % len(PILL_COLORS)]
        bodies.append(f"""
    <body name="{seg_name(c, r)}" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" quat="{quat_str}">
      <freejoint name="{seg_name(c, r)}_joint"/>
      <geom name="{seg_name(c, r)}_plate" type="box" size="{SEG_HX} {SEG_HY} {SEG_HZ}"
        friction="2 0.01 0.001" solref="0.004 1" rgba="0.82 0.83 0.87 1" mass="0.002"/>
      <geom name="{seg_name(c, r)}_dome" type="ellipsoid" size="0.0075 0.0075 0.0042"
        pos="0 0 {SEG_HZ + 0.002}" contype="0" conaffinity="0" rgba="0.88 0.92 0.97 0.45" mass="0.0004"/>
      <geom name="{seg_name(c, r)}_pill" type="cylinder" size="0.005 0.0016"
        pos="0 0 {SEG_HZ + 0.001}" contype="0" conaffinity="0" rgba="{color} 1" mass="0.0006"/>
    </body>""")

    # 易撕线（可断裂焊接）：solref/solimp 硬化 + torquescale 放大，否则板绕焊点下垂
    W = ('solref="0.0015 1" solimp="0.99 0.999 0.0001" torquescale="20"')
    welds = []
    for r in range(N_ROWS):
        welds.append(f'    <weld name="w_strip_{ROW_TAG[r]}" body1="strip" '
                     f'body2="{seg_name(0, r)}" anchor="-{SEG_HX} 0 0" {W}/>')
        for c in range(N_COLS - 1):
            welds.append(f'    <weld name="w_col{c}{c+1}_{ROW_TAG[r]}" body1="{seg_name(c, r)}" '
                         f'body2="{seg_name(c+1, r)}" anchor="-{SEG_HX} 0 0" {W}/>')
    for c in range(N_COLS):
        welds.append(f'    <weld name="w_row_c{c}" body1="{seg_name(c, 0)}" '
                     f'body2="{seg_name(c, 1)}" anchor="0 -{SEG_HY} 0" {W}/>')
    # 备用抓取锁定（左爪 ↔ strip），默认关闭；启用前需运行时写入实际相对位姿。
    # 必须硬化，否则板绕抓点下垂倾斜
    welds.append(f'    <weld name="grasp_latch" body1="left/gripper_link" body2="strip" '
                 f'active="false" relpose="0 0 0.19 1 0 0 0" {W}/>')

    return f"""
<mujoco model="aloha_pill_full">
  <include file="{(HERE / 'scene_tear.xml').as_posix()}"/>

  <option timestep="0.001" integrator="implicitfast"/>

  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>

  <worldbody>
    <camera name="follow_pack" mode="targetbody" target="strip" pos="0.55 -0.55 0.45"/>
    <camera name="side" pos="-0.75 -0.35 0.40" xyaxes="0.5165 -0.8563 0.0000 0.2453 0.1480 0.9581"/>
    <camera name="room" mode="targetbody" target="mobile_base" pos="1.55 -1.60 0.90"/>
{boxes_xml()}
{''.join(bodies)}
  </worldbody>

  <equality>
{chr(10).join(welds)}
  </equality>
</mujoco>"""


def load_model():
    return mujoco.MjModel.from_xml_string(build_xml())


def place_segments(model, data):
    """把所有未撕下的板块按 strip 当前位姿刚性摆放（保持焊接约束满足）。"""
    mujoco.mj_forward(model, data)
    strip = data.body("strip")
    R = strip.xmat.reshape(3, 3)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, strip.xmat)
    for c, r in all_segments():
        p = strip.xpos + R @ seg_offset(c, r)
        adr = model.joint(f"{seg_name(c, r)}_joint").qposadr[0]
        data.qpos[adr: adr + 3] = p
        data.qpos[adr + 3: adr + 7] = quat
        vadr = model.joint(f"{seg_name(c, r)}_joint").dofadr[0]
        data.qvel[vadr: vadr + 6] = 0.0
    mujoco.mj_forward(model, data)


def weld_load(model, data, weld_name):
    """某条易撕线上的约束合力（efc 行绝对值之和，含力与力矩行）。"""
    eq_id = model.equality(weld_name).id
    total = 0.0
    for i in range(data.nefc):
        if (data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
                and data.efc_id[i] == eq_id):
            total += abs(data.efc_force[i])
    return total


def engage_latch(model, data):
    """把左爪-药板锁定焊接的 relpose 写为当前实际相对位姿并激活。"""
    eq = model.equality("grasp_latch")
    g = data.body("left/gripper_link")
    s = data.body("strip")
    Rg = g.xmat.reshape(3, 3)
    relpos = Rg.T @ (s.xpos - g.xpos)
    qg_inv, relquat = np.empty(4), np.empty(4)
    gq = np.empty(4)
    mujoco.mju_mat2Quat(gq, g.xmat)
    sq = np.empty(4)
    mujoco.mju_mat2Quat(sq, s.xmat)
    mujoco.mju_negQuat(qg_inv, gq)
    mujoco.mju_mulQuat(relquat, qg_inv, sq)
    model.eq_data[eq.id, 0:3] = 0.0
    model.eq_data[eq.id, 3:6] = relpos
    model.eq_data[eq.id, 6:10] = relquat
    data.eq_active[eq.id] = 1


def release_latch(model, data):
    data.eq_active[model.equality("grasp_latch").id] = 0


def set_base(model, data, xyyaw):
    """设置底盘三关节的 qpos 与 ctrl。"""
    for jname, v in zip(("base_x", "base_y", "base_yaw"), xyyaw):
        data.qpos[model.joint(jname).qposadr[0]] = v
        data.ctrl[model.actuator(jname).id] = v


if __name__ == "__main__":
    import imageio.v2 as imageio

    from ik_utils import ARM_JOINTS

    NEUTRAL_ARM = np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0])
    model = mujoco.MjModel.from_xml_string(build_xml())
    data = mujoco.MjData(model)
    print(f"模型编译成功: nq={model.nq}, nbody={model.nbody}, neq={model.neq}")

    set_base(model, data, BASE_START)
    for side in ("left", "right"):
        for jname, q in zip(ARM_JOINTS, NEUTRAL_ARM):
            data.qpos[model.joint(f"{side}/{jname}").qposadr[0]] = q
            data.ctrl[model.actuator(f"{side}/{jname}").id] = q
    mujoco.mj_forward(model, data)

    # 沉降：板在固定桌上盒 A 槽中落座后应保持竖立
    for _ in range(1500):
        mujoco.mj_step(model, data)
    s = data.body("strip")
    Rz = s.xmat.reshape(3, 3)
    print(f"沉降后 strip 位置: {np.round(s.xpos, 4)}（初始 {np.round(BOARD_HOME, 4)}）")
    print(f"沉降后板局部 x 轴（应≈[0,0,-1]）: {np.round(Rz[:, 0], 3)}")
    print(f"底盘位置: {np.round([data.qpos[model.joint('base_x').qposadr[0]], data.qpos[model.joint('base_y').qposadr[0]]], 4)}")
    print(f"w_strip_f 静载: {weld_load(model, data, 'w_strip_f'):.3f}")
    for side in ("left", "right"):
        b = data.body(f"{side}/base_link")
        print(f"{side} 臂基座世界位置: {np.round(b.xpos, 3)}")

    # 再看停到桌前工作位的可达性
    set_base(model, data, BASE_WORK)
    mujoco.mj_forward(model, data)
    place_segments(model, data)
    renderer = mujoco.Renderer(model, height=720, width=1280)
    (HERE / "debug").mkdir(exist_ok=True)
    for cam in ("room", "follow_pack", "side"):
        renderer.update_scene(data, camera=cam)
        imageio.imwrite(HERE / "debug" / f"full_probe_{cam}.png", renderer.render())
    renderer.close()
    print("已渲染 debug/full_probe_*.png（底盘置于桌前工作位）")
