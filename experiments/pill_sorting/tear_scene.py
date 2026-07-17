"""分药 v3 场景：盒 A（插板架，内有多块铝塑板）→ 撕剪 → 盒 B（药片盒）。

- 药板 = 手柄条（strip，自由体）+ 8 格（2x4，自由体），以 12 条可断裂焊接约束（易撕线）相连；
- 初始竖插在盒 A 中间槽位，手柄朝上，需左手真实抓取；
- 盒 A 另两槽插有装饰板（静态几何），示意"一堆铝板"；
- 盒 B 接收撕下的单格。

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

# ---------- 盒 A（插板架）与盒 B（药片盒） ----------
BOX_A_CENTER = np.array([-0.22, -0.10, 0.0])
SLOT_PITCH = 0.014          # 槽间距（x 向）；槽宽 = pitch - 隔板厚
SLOT_WALL_T = 0.002         # 隔板半厚
BOX_A_HY = 0.032            # 内腔 y 半宽
BOX_A_WALL_H = 0.05
BOX_A_FLOOR_TOP = 0.006     # 内底面高度
# 目标板插中间槽；竖立时 strip 局部 +x 朝下（自由端在下，tab 在上）
BOARD_UP_QUAT = np.array([0.7071068, 0.0, 0.7071068, 0.0])  # Ry(90°)
BOARD_HOME = np.array([BOX_A_CENTER[0], BOX_A_CENTER[1],
                       BOX_A_FLOOR_TOP + BOARD_XMAX + 0.002])

BOX_B_CENTER = np.array([0.13, -0.09, 0.0])
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


def build_xml():
    """生成完整场景 XML 字符串。板初始竖插在盒 A 中间槽（焊接 relpose 取 qpos0）。"""
    R0 = _quat_to_mat(BOARD_UP_QUAT)
    quat_str = " ".join(f"{q:.7f}" for q in BOARD_UP_QUAT)

    # 药板：strip 自由体
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

    # 盒 A：3 槽插板架
    ax, ay = BOX_A_CENTER[0], BOX_A_CENTER[1]
    inner_hx = 1.5 * SLOT_PITCH + SLOT_WALL_T
    boxa = [f'    <geom name="boxa_bottom" type="box" size="{inner_hx + 0.006} {BOX_A_HY + 0.006} 0.003" '
            f'pos="{ax} {ay} 0.003" rgba="0.55 0.42 0.30 1"/>']
    for sx in (-1, 1):  # x 向外壁
        boxa.append(f'    <geom type="box" size="0.003 {BOX_A_HY + 0.006} {BOX_A_WALL_H}" '
                    f'pos="{ax + sx * (inner_hx + 0.003):.4f} {ay} {BOX_A_WALL_H}" rgba="0.55 0.42 0.30 0.7"/>')
    for sy in (-1, 1):  # y 向外壁
        boxa.append(f'    <geom type="box" size="{inner_hx + 0.006} 0.003 {BOX_A_WALL_H}" '
                    f'pos="{ax} {ay + sy * (BOX_A_HY + 0.003):.4f} {BOX_A_WALL_H}" rgba="0.55 0.42 0.30 0.7"/>')
    for k in (-1, 1):   # 两块内隔板 → 3 槽
        boxa.append(f'    <geom type="box" size="{SLOT_WALL_T} {BOX_A_HY} 0.045" '
                    f'pos="{ax + k * SLOT_PITCH / 2:.4f} {ay} {0.006 + 0.045}" rgba="0.60 0.48 0.35 0.9"/>')
    # 两块装饰板（静态），插在边槽；比目标板矮，避免左手指下降抓取时磕碰
    for k, h in ((-1, 0.042), (1, 0.038)):
        px = ax + k * SLOT_PITCH
        boxa.append(f'    <geom type="box" size="0.0012 0.026 {h}" '
                    f'pos="{px:.4f} {ay} {BOX_A_FLOOR_TOP + h + 0.001:.4f}" '
                    f'rgba="0.78 0.79 0.84 1"/>')
        boxa.append(f'    <geom type="box" size="0.002 0.011 0.006" '
                    f'pos="{px:.4f} {ay} {BOX_A_FLOOR_TOP + 2 * h - 0.004:.4f}" '
                    f'contype="0" conaffinity="0" rgba="0.3 0.35 0.45 1"/>')

    # 盒 B：药片盒
    bx, by = BOX_B_CENTER[0], BOX_B_CENTER[1]
    boxb = [f'    <geom name="boxb_bottom" type="box" size="{BOX_B_HX} {BOX_B_HY} 0.0015" '
            f'pos="{bx} {by} 0.0015" rgba="0.30 0.55 0.75 1"/>']
    for sx, sy, hx, hy in ((1, 0, 0.003, BOX_B_HY), (-1, 0, 0.003, BOX_B_HY),
                           (0, 1, BOX_B_HX, 0.003), (0, -1, BOX_B_HX, 0.003)):
        px = bx + sx * (BOX_B_HX + 0.003)
        py = by + sy * (BOX_B_HY + 0.003)
        boxb.append(f'    <geom type="box" size="{hx} {hy} {BOX_B_WALL_H}" '
                    f'pos="{px:.4f} {py:.4f} {BOX_B_WALL_H}" rgba="0.30 0.55 0.75 0.5"/>')

    return f"""
<mujoco model="aloha_pill_full">
  <include file="{(HERE / 'scene_tear.xml').as_posix()}"/>

  <option timestep="0.001" integrator="implicitfast"/>

  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>

  <worldbody>
    <camera name="follow_pack" mode="targetbody" target="strip" pos="0.42 -0.40 0.38"/>
    <camera name="side" pos="-0.60 -0.28 0.34" xyaxes="0.5165 -0.8563 0.0000 0.2453 0.1480 0.9581"/>
{''.join(bodies)}
{chr(10).join(boxa)}
{chr(10).join(boxb)}
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


if __name__ == "__main__":
    import imageio.v2 as imageio

    from ik_utils import ARM_JOINTS

    NEUTRAL_ARM = np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0])
    model = mujoco.MjModel.from_xml_string(build_xml())
    data = mujoco.MjData(model)
    print(f"模型编译成功: nq={model.nq}, nbody={model.nbody}, neq={model.neq}")

    for side in ("left", "right"):
        for jname, q in zip(ARM_JOINTS, NEUTRAL_ARM):
            data.qpos[model.joint(f"{side}/{jname}").qposadr[0]] = q
            data.ctrl[model.actuator(f"{side}/{jname}").id] = q
    mujoco.mj_forward(model, data)

    # 沉降：板在槽中落座后应保持竖立
    for _ in range(1500):
        mujoco.mj_step(model, data)
    s = data.body("strip")
    Rz = s.xmat.reshape(3, 3)
    print(f"沉降后 strip 位置: {np.round(s.xpos, 4)}（初始 {np.round(BOARD_HOME, 4)}）")
    print(f"沉降后板局部 x 轴（应≈[0,0,-1]）: {np.round(Rz[:, 0], 3)}")
    print(f"tab 顶部世界位置: {np.round(data.site('strip_tab_top').xpos, 4)}")
    print(f"w_strip_f 静载: {weld_load(model, data, 'w_strip_f'):.3f}")

    renderer = mujoco.Renderer(model, height=720, width=1280)
    (HERE / "debug").mkdir(exist_ok=True)
    for cam in ("follow_pack", "side"):
        renderer.update_scene(data, camera=cam)
        imageio.imwrite(HERE / "debug" / f"full_probe_{cam}.png", renderer.render())
    renderer.close()
    print("已渲染 debug/full_probe_*.png")
